# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Node Classification PyG Example.

This example shows how to train a PyTorch Geometric (PyG) model leveraging DGF
capabilities like dataset IO, samplers, and normalization, running on GPU.

Usage example:

blaze run -c opt --config=cuda \
//third_party/py/dgf/examples:node_classification_pyg
"""

import dataclasses
import itertools
import typing

from absl import app
from absl import flags
import dgf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.data
import torch_geometric.nn
import tqdm

_OUTPUT_DIR = flags.DEFINE_string("output_dir", "/tmp/dgf_pyg_model", "")
_SPLIT_FEATURE_NAME = "#split"


@dataclasses.dataclass
class HyperParameters:
  batch_size: int = 256
  learning_rate: float = 0.001
  target_nodeset: str = "paper"
  target_column: str = "labels"
  train_steps: int = 1000


class PyGHeteroGNN(torch.nn.Module):
  """Simple PyG Heterogeneous GNN for showcase."""

  def __init__(
      self,
      in_channels_dict: typing.Mapping[str, int],
      hidden_channels: int,
      out_channels: int,
      num_layers: int,
      edge_types: typing.Sequence[typing.Tuple[str, str, str]],
      target_nodeset: str = "paper",
  ):
    super().__init__()
    self.target_nodeset = target_nodeset
    # Project initial features
    self.lins = torch.nn.ModuleDict({
        node_type: torch.nn.Linear(in_channels, hidden_channels)
        for node_type, in_channels in in_channels_dict.items()
    })

    self.convs = torch.nn.ModuleList()
    for _ in range(num_layers):
      # GraphSAGE on heterogeneous graphs
      conv = torch_geometric.nn.HeteroConv(
          {
              edge_type: torch_geometric.nn.SAGEConv(
                  hidden_channels, hidden_channels
              )
              for edge_type in edge_types
          },
          aggr="sum",
      )
      self.convs.append(conv)

    self.lin = torch.nn.Linear(hidden_channels, out_channels)

  def forward(
      self,
      x_dict: typing.Mapping[str, torch.Tensor],
      edge_index_dict: typing.Mapping[
          typing.Tuple[str, str, str], torch.Tensor
      ],
      seed_node_idxs: torch.Tensor,
  ) -> torch.Tensor:
    # Initial projection
    x_dict = {
        node_type: torch.nn.functional.relu(self.lins[node_type](x))
        for node_type, x in x_dict.items()
    }

    # Message passing layers
    for conv in self.convs:
      out_dict = conv(x_dict, edge_index_dict)
      new_x_dict = {}
      for node_type, x in x_dict.items():
        if node_type in out_dict:
          new_x_dict[node_type] = torch.nn.functional.relu(out_dict[node_type])
        else:
          new_x_dict[node_type] = x
      x_dict = new_x_dict

    # Extract final embeddings for the target nodes
    target_embeds = x_dict[self.target_nodeset][seed_node_idxs]
    return self.lin(target_embeds)


def main(argv: typing.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  hparams = HyperParameters()

  dgf.filesystem.makedirs(_OUTPUT_DIR.value)

  # Check device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if device.type == "cuda":
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
  else:
    print(f"Using device: {device} (No GPU detected)")

  # Use DGF to load and prepare the data
  print("Loading graph with DGF...")
  graph, schema = dgf.io.fetch_ogb_graph("mag")

  print("Graph schema:")
  dgf.analyse.print_schema(schema)

  # Use #split feature to determine train/valid sets
  splits = graph.node_sets[hparams.target_nodeset].features[_SPLIT_FEATURE_NAME]
  train_seed_node_idxs = np.where(splits == b"train")[0]
  valid_seed_node_idxs = np.where(splits == b"valid")[0]

  # Remove #split from the schema early since it is only needed for dataset
  # splitting.
  del schema.node_sets[hparams.target_nodeset].features[_SPLIT_FEATURE_NAME]

  print(f"Num train nodes: {len(train_seed_node_idxs)}")
  print(f"Num valid nodes: {len(valid_seed_node_idxs)}")

  # Create a DGF graph sampler
  plan = dgf.sampling.SimpleSamplingConfig(
      seed_nodeset=hparams.target_nodeset,
      num_hops=2,
      hop_width=10,
      reverse=True,
  )

  sampler = dgf.sampling.create_sampler(
      graph=graph,
      plan=plan,
      schema=schema,
      batch_size=hparams.batch_size,
  )

  def batch_generator(seed_node_idxs, batch_size):
    graph_merger = dgf.transform.GraphMerger(
        schema=schema,
        padding=None,
        sentinel_offset=False,
    )
    for indices in dgf.transform.batch_indices_generator(
        seed_node_idxs,
        batch_size=batch_size,
        drop_remainder=True,
        shuffle=False,
    ):
      samples = sampler.sample(indices.tolist())

      try:
        merged_samples, merge_offsets = graph_merger(samples)
      except dgf.exception.InsufficientPaddingError:
        continue

      yield merged_samples, merge_offsets

  # Let DGF compute feature statistics for auto-normalization
  print("Computing feature statistics for normalization...")
  # For stat gathering, only analyze a few batches from train set
  stat_graphs = (
      batch[0] for batch in batch_generator(train_seed_node_idxs, 32)
  )
  feature_stats = dgf.analyse.feature_statistics_from_graphs(
      graphs=itertools.islice(stat_graphs, 100),
      schema=schema,
  )

  # DGF Feature normalizer
  normalizer = dgf.transform.auto_normalize(schema=schema, stats=feature_stats)

  print("Normalized Graph schema:")
  dgf.analyse.print_schema(normalizer.output_schema())

  # Filter schema for the model (drop labels from PyG mapping)
  normalized_schema = normalizer.output_schema()
  num_label_classes = (
      normalized_schema.node_sets[hparams.target_nodeset]
      .features[hparams.target_column]
      .num_categorical_values
  )
  assert num_label_classes is not None
  del normalized_schema.node_sets[hparams.target_nodeset].features[
      hparams.target_column
  ]

  # PyG data prep
  def process_batch_pyg(raw_batch, merge_offsets):
    # 1. Use DGF to normalize the features
    normalized_graph = normalizer.normalize_numpy(raw_batch)

    # 2. Extract labels and map offsets
    seed_node_idxs = merge_offsets[hparams.target_nodeset]
    labels = normalized_graph.node_sets[hparams.target_nodeset].features[
        hparams.target_column
    ][seed_node_idxs]

    # 3. Convert normalized graph to PyG HeteroData structure
    pyg_data = dgf.convert.graph_to_pyg_data(
        normalized_graph, normalized_schema
    )

    # 4. Transfer to targeted device (GPU if available)
    pyg_data = pyg_data.to(device)

    # PyG expects all nodes to have at least dummy features if they are part of
    # message passing.
    for node_type in pyg_data.node_types:
      if not hasattr(pyg_data[node_type], "x") or pyg_data[node_type].x is None:
        pyg_data[node_type].x = torch.ones(
            pyg_data[node_type].num_nodes, 1, device=device
        )

    labels = torch.from_numpy(labels).long().to(device)
    seed_node_idxs = torch.from_numpy(seed_node_idxs).long().to(device)

    return pyg_data, seed_node_idxs, labels

  def generate_batches(indices):
    for raw_batch, merge_offsets in batch_generator(
        indices, hparams.batch_size
    ):
      yield process_batch_pyg(raw_batch, merge_offsets)

  # Model instantiation require dimensions from schema
  # To map the exact in_channels, let's process one batch to get feature dims
  print("Instantiating PyTorch Geometric model...")
  sample_pyg_data, _, _ = next(iter(generate_batches(train_seed_node_idxs)))
  in_channels_dict = {
      node_type: data_x.shape[-1]
      for node_type, data_x in sample_pyg_data.x_dict.items()
  }

  model = PyGHeteroGNN(
      in_channels_dict=in_channels_dict,
      hidden_channels=128,
      out_channels=num_label_classes,
      num_layers=2,
      edge_types=sample_pyg_data.edge_types,
      target_nodeset=hparams.target_nodeset,
  ).to(device)

  optimizer = torch.optim.AdamW(
      model.parameters(), lr=hparams.learning_rate, weight_decay=1e-5
  )
  loss_fn = torch.nn.CrossEntropyLoss()

  print("Starting PyG training loop with DGF sampler...")

  train_steps = hparams.train_steps
  valid_every = 200

  train_losses = []
  train_accs = []
  valid_steps = []
  valid_losses = []
  valid_accs = []

  model.train()
  train_iter = iter(generate_batches(train_seed_node_idxs))

  pbar = tqdm.tqdm(range(1, train_steps + 1))
  for step in pbar:
    try:
      pyg_data, seed_node_idxs, labels = next(train_iter)
    except StopIteration:
      train_iter = iter(generate_batches(train_seed_node_idxs))
      pyg_data, seed_node_idxs, labels = next(train_iter)

    optimizer.zero_grad()
    out = model(pyg_data.x_dict, pyg_data.edge_index_dict, seed_node_idxs)
    loss = loss_fn(out, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    acc = (out.argmax(dim=-1) == labels).float().mean().item()
    train_losses.append(loss.item())
    train_accs.append(acc)

    pbar.set_description(f"Loss: {loss.item():.4f} Acc: {acc:.4f}")

    if step % valid_every == 0:
      model.eval()
      val_loss, val_acc = 0.0, 0.0
      val_batches = 0
      with torch.no_grad():
        for val_data, val_seeds, val_labels in generate_batches(
            valid_seed_node_idxs
        ):

          val_out = model(val_data.x_dict, val_data.edge_index_dict, val_seeds)
          val_loss += loss_fn(val_out, val_labels).item()
          val_acc += (
              (val_out.argmax(dim=-1) == val_labels).float().mean().item()
          )
          val_batches += 1

      if val_batches > 0:
        val_loss /= val_batches
        val_acc /= val_batches
        valid_steps.append(step)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        print(
            f"Step {step} - Valid Loss: {val_loss:.4f} Valid Acc: {val_acc:.4f}"
        )

      model.train()

  # Plot logs
  plt.figure(figsize=(16, 6))
  plt.subplot(1, 2, 1)
  plt.plot(range(1, train_steps + 1), train_losses, label="train", alpha=0.3)
  if valid_steps:
    plt.plot(valid_steps, valid_losses, label="valid", marker="o")
  plt.xlabel("Step")
  plt.ylabel("Loss")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(range(1, train_steps + 1), train_accs, label="train", alpha=0.3)
  if valid_steps:
    plt.plot(valid_steps, valid_accs, label="valid", marker="o")
  plt.xlabel("Step")
  plt.ylabel("Accuracy")
  plt.legend()

  plt.tight_layout()
  plot_path = f"{_OUTPUT_DIR.value}/training.png"
  plt.savefig(plot_path)

  print(f"Results are available in : {_OUTPUT_DIR.value}")
  print(f"Plot saved at: {plot_path}")


if __name__ == "__main__":
  app.run(main)
