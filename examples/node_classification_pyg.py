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

This is an advanced example: it assumes you are already familiar with PyTorch
and PyTorch Geometric (PyG). If you are new to Graph Flow, start with the
simpler `node_classification_model.py` example instead.

This example shows how to train a PyTorch Geometric (PyG) model on the OGB-MAG
dataset, using Graph Flow (DGF) for everything around the model:

1. DGF loads the graph (`dgf.io.fetch_ogb_graph`).
2. DGF samples a 2-hop neighborhood around each seed paper
   (`dgf.sampling.create_sampler`) and merges the samples into a batch
   (`dgf.transform.GraphMerger`).
3. DGF computes feature statistics and normalizes the features automatically
   (`dgf.transform.auto_normalize`).
4. DGF converts the batch into a PyG `HeteroData` (`dgf.convert.graph_to_pyg_data`).
5. PyG trains a heterogeneous GraphSAGE model.

The model only consumes the paper embeddings and years (no learned node id
embeddings). It reaches ~31% test accuracy in ~20 minutes on a single GPU, in
line with the OGB GraphSAGE baseline (31.5%).

Usage example:

# Run on GPU.
blaze run -c opt --config=cuda \
//third_party/py/dgf/examples:node_classification_pyg

# Quick smoke test (also works on CPU, but training is much slower).
blaze run -c opt //third_party/py/dgf/examples:node_classification_pyg -- \
  --train_steps=50 --valid_every=50
"""

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import itertools

from absl import app
from absl import flags
import dgf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.nn
import tqdm

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "/tmp/dgf_pyg_model", "Directory for the training plot."
)
_TRAIN_STEPS = flags.DEFINE_integer(
    "train_steps", 20000, "Number of training steps."
)
_VALID_EVERY = flags.DEFINE_integer(
    "valid_every", 2500, "Evaluate on the validation set every N steps."
)

_SPLIT_FEATURE_NAME = "#split"

# A PyG edge type is a (source nodeset, edgeset name, target nodeset) triplet.
EdgeType = tuple[str, str, str]


@dataclasses.dataclass
class HyperParameters:
  """Hyper-parameters of the example."""

  # Task
  target_nodeset: str = "paper"
  target_column: str = "labels"
  # Sampling
  num_hops: int = 2
  hop_width: int = 10
  # Model
  hidden_channels: int = 128
  num_layers: int = 2
  # Training
  batch_size: int = 256
  learning_rate: float = 0.001
  weight_decay: float = 1e-5
  seed: int = 42


@dataclasses.dataclass
class Batch:
  """A batch of training / evaluation examples, ready for the PyG model."""

  data: torch_geometric.data.HeteroData
  # Index of the seed nodes in the target nodeset of `data`.
  seed_node_idxs: torch.Tensor
  labels: torch.Tensor


def pyg_edge_types(schema: dgf.data.GraphSchema) -> list[EdgeType]:
  """Lists the edge types, including the reverse ones added by ToUndirected."""
  edge_types = []
  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    src, dst = edgeset_schema.source, edgeset_schema.target
    edge_types.append((src, edgeset_name, dst))
    edge_types.append((dst, f"rev_{edgeset_name}", src))
  return edge_types


def pyg_input_dims(schema: dgf.data.GraphSchema) -> dict[str, int]:
  """Size of the concatenated "x" features of each nodeset."""
  input_dims = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    input_dim = 0
    for feature_name, feature_schema in nodeset_schema.features.items():
      if not feature_schema.is_static_shape():
        raise ValueError(
            f"Feature {nodeset_name}.{feature_name} has a variable shape"
            f" {feature_schema.shape}, which cannot be fed to a PyG model."
        )
      input_dim += feature_schema.static_size()
    input_dims[nodeset_name] = input_dim if input_dim > 0 else 1
  return input_dims


def add_dummy_features(
    data: torch_geometric.data.HeteroData,
    schema: dgf.data.GraphSchema,
) -> None:
  """Adds a constant "x" feature to featureless nodesets (required by PyG)."""
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    if not nodeset_schema.features:
      data[nodeset_name].x = torch.ones(data[nodeset_name].num_nodes, 1)


class PyGHeteroGNN(torch.nn.Module):
  """Simple heterogeneous GraphSAGE model."""

  def __init__(
      self,
      input_dims: Mapping[str, int],
      edge_types: Sequence[EdgeType],
      hidden_channels: int,
      out_channels: int,
      num_layers: int,
      target_nodeset: str,
  ):
    super().__init__()
    self.target_nodeset = target_nodeset

    self.input_projections = torch.nn.ModuleDict({
        nodeset_name: torch.nn.Linear(input_dim, hidden_channels)
        for nodeset_name, input_dim in input_dims.items()
    })

    self.convs = torch.nn.ModuleList([
        torch_geometric.nn.HeteroConv(
            {
                edge_type: torch_geometric.nn.SAGEConv(
                    hidden_channels, hidden_channels
                )
                for edge_type in edge_types
            },
            aggr="sum",
        )
        for _ in range(num_layers)
    ])

    self.classifier = torch.nn.Linear(hidden_channels, out_channels)

  def forward(
      self,
      x_dict: Mapping[str, torch.Tensor],
      edge_index_dict: Mapping[EdgeType, torch.Tensor],
      seed_node_idxs: torch.Tensor,
  ) -> torch.Tensor:
    x_dict = {
        nodeset_name: torch.relu(self.input_projections[nodeset_name](x))
        for nodeset_name, x in x_dict.items()
    }

    for conv in self.convs:
      x_dict = {
          nodeset_name: torch.relu(x)
          for nodeset_name, x in conv(x_dict, edge_index_dict).items()
      }

    return self.classifier(x_dict[self.target_nodeset][seed_node_idxs])


def plot_training_logs(
    train_losses: Sequence[float],
    train_accs: Sequence[float],
    valid_steps: Sequence[int],
    valid_losses: Sequence[float],
    valid_accs: Sequence[float],
    path: str,
) -> None:
  """Plots the training and validation curves."""

  def smooth(values: Sequence[float], window: int = 50) -> np.ndarray:
    window = max(1, min(window, len(values)))
    return np.convolve(values, np.ones(window) / window, mode="valid")

  def plot(
      name: str,
      train_values: Sequence[float],
      valid_values: Sequence[float],
  ) -> None:
    smoothed = smooth(train_values)
    plt.plot(
        np.arange(len(smoothed)) + len(train_values) - len(smoothed) + 1,
        smoothed,
        label="train (smoothed)",
    )
    plt.plot(valid_steps, valid_values, label="valid", marker="o")
    plt.xlabel("Step")
    plt.ylabel(name)
    plt.legend()

  plt.figure(figsize=(16, 6))
  plt.subplot(1, 2, 1)
  plot("Loss", train_losses, valid_losses)
  plt.subplot(1, 2, 2)
  plot("Accuracy", train_accs, valid_accs)
  plt.tight_layout()
  plt.savefig(path)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  hparams = HyperParameters()
  np.random.seed(hparams.seed)
  torch.manual_seed(hparams.seed)
  dgf.filesystem.makedirs(_OUTPUT_DIR.value)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if device.type == "cuda":
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
  else:
    print(f"Using device: {device} (No GPU detected, training will be slow)")

  # 1. Load the graph with DGF
  # ==========================
  print("Loading graph with DGF...")
  graph, schema = dgf.io.fetch_ogb_graph("mag")
  dgf.analyse.print_schema(schema)

  # "#split" is used to split the dataset, not as a model input.
  splits = graph.node_sets[hparams.target_nodeset].features[_SPLIT_FEATURE_NAME]
  train_seed_node_idxs = np.where(splits == b"train")[0]
  valid_seed_node_idxs = np.where(splits == b"valid")[0]
  test_seed_node_idxs = np.where(splits == b"test")[0]
  del schema.node_sets[hparams.target_nodeset].features[_SPLIT_FEATURE_NAME]
  print(
      f"Num seed nodes: train={len(train_seed_node_idxs)}"
      f" valid={len(valid_seed_node_idxs)} test={len(test_seed_node_idxs)}"
  )

  # 2. Sample neighborhoods with DGF
  # ================================
  sampler = dgf.sampling.create_sampler(
      graph=graph,
      plan=dgf.sampling.SimpleSamplingConfig(
          seed_nodeset=hparams.target_nodeset,
          num_hops=hparams.num_hops,
          hop_width=hparams.hop_width,
          reverse=True,
      ),
      schema=schema,
      batch_size=hparams.batch_size,
  )
  # No padding needed: PyTorch supports dynamic shapes.
  graph_merger = dgf.transform.GraphMerger(
      schema=schema, padding=None, sentinel_offset=False
  )

  def sample_batches(
      seed_node_idxs: np.ndarray, batch_size: int, training: bool
  ) -> Iterator[tuple[dgf.data.InMemoryGraph, dict[str, np.ndarray]]]:
    """Yields merged graph samples and the position of their seed nodes."""
    for batch_seed_node_idxs in dgf.transform.batch_indices_generator(
        seed_node_idxs,
        batch_size=batch_size,
        drop_remainder=training,
        shuffle=training,
    ):
      samples = sampler.sample(batch_seed_node_idxs.tolist())
      yield graph_merger(samples)

  # 3. Normalize the features with DGF
  # ==================================
  print("Computing feature statistics for normalization...")
  feature_stats = dgf.analyse.feature_statistics_from_graphs(
      graphs=(
          merged_graph
          for merged_graph, _ in itertools.islice(
              sample_batches(train_seed_node_idxs, 32, training=True), 100
          )
      ),
      schema=schema,
  )
  normalizer = dgf.transform.auto_normalize(schema=schema, stats=feature_stats)

  # The model sees the normalized features, except for the label.
  model_schema = normalizer.output_schema()
  num_label_classes = (
      model_schema.node_sets[hparams.target_nodeset]
      .features[hparams.target_column]
      .num_categorical_values
  )
  assert num_label_classes is not None
  del model_schema.node_sets[hparams.target_nodeset].features[
      hparams.target_column
  ]
  print("Model input schema:")
  dgf.analyse.print_schema(model_schema)

  # 4. Convert the batches to PyG with DGF
  # ======================================
  # Adds reverse edges so messages flow in both directions.
  to_undirected = torch_geometric.transforms.ToUndirected(merge=False)

  def to_pyg_batch(
      merged_graph: dgf.data.InMemoryGraph,
      merge_offsets: Mapping[str, np.ndarray],
  ) -> Batch:
    normalized_graph = normalizer.normalize_numpy(merged_graph)
    seed_node_idxs = merge_offsets[hparams.target_nodeset]
    labels = normalized_graph.node_sets[hparams.target_nodeset].features[
        hparams.target_column
    ][seed_node_idxs]

    data = dgf.convert.graph_to_pyg_data(normalized_graph, model_schema)
    add_dummy_features(data, model_schema)
    data = to_undirected(data)
    return Batch(
        data=data.to(device),
        seed_node_idxs=torch.from_numpy(seed_node_idxs).long().to(device),
        labels=torch.from_numpy(labels).long().to(device),
    )

  def pyg_batches(
      seed_node_idxs: np.ndarray, training: bool
  ) -> Iterator[Batch]:
    for merged_graph, merge_offsets in sample_batches(
        seed_node_idxs, hparams.batch_size, training=training
    ):
      yield to_pyg_batch(merged_graph, merge_offsets)

  def infinite_train_batches() -> Iterator[Batch]:
    while True:
      yield from pyg_batches(train_seed_node_idxs, training=True)

  # 5. Train a PyG model
  # ====================
  model = PyGHeteroGNN(
      input_dims=pyg_input_dims(model_schema),
      edge_types=pyg_edge_types(model_schema),
      hidden_channels=hparams.hidden_channels,
      out_channels=num_label_classes,
      num_layers=hparams.num_layers,
      target_nodeset=hparams.target_nodeset,
  ).to(device)
  print(model)

  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=hparams.learning_rate,
      weight_decay=hparams.weight_decay,
  )
  loss_fn = torch.nn.CrossEntropyLoss()

  def forward(batch: Batch) -> torch.Tensor:
    return model(
        batch.data.x_dict, batch.data.edge_index_dict, batch.seed_node_idxs
    )

  @torch.no_grad()
  def evaluate(seed_node_idxs: np.ndarray) -> tuple[float, float]:
    """Returns the loss and accuracy over all the given seed nodes."""
    model.eval()
    sum_loss, num_correct, num_examples = 0.0, 0, 0
    for batch in pyg_batches(seed_node_idxs, training=False):
      logits = forward(batch)
      sum_loss += loss_fn(logits, batch.labels).item() * len(batch.labels)
      num_correct += (logits.argmax(dim=-1) == batch.labels).sum().item()
      num_examples += len(batch.labels)
    model.train()
    return sum_loss / num_examples, num_correct / num_examples

  print("Training...")
  train_losses, train_accs = [], []
  valid_steps, valid_losses, valid_accs = [], [], []

  model.train()
  train_batches = infinite_train_batches()
  pbar = tqdm.tqdm(range(1, _TRAIN_STEPS.value + 1))
  for step in pbar:
    batch = next(train_batches)
    optimizer.zero_grad()
    logits = forward(batch)
    loss = loss_fn(logits, batch.labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    acc = (logits.argmax(dim=-1) == batch.labels).float().mean().item()
    train_losses.append(loss.item())
    train_accs.append(acc)
    pbar.set_description(f"loss={loss.item():.4f} acc={acc:.4f}")

    if step % _VALID_EVERY.value == 0 or step == _TRAIN_STEPS.value:
      valid_loss, valid_acc = evaluate(valid_seed_node_idxs)
      valid_steps.append(step)
      valid_losses.append(valid_loss)
      valid_accs.append(valid_acc)
      pbar.write(
          f"Step {step}: valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
      )

  test_loss, test_acc = evaluate(test_seed_node_idxs)
  print(f"Test: loss={test_loss:.4f} accuracy={test_acc:.4f}")

  plot_path = f"{_OUTPUT_DIR.value}/training.png"
  plot_training_logs(
      train_losses, train_accs, valid_steps, valid_losses, valid_accs, plot_path
  )
  print(f"Training plot saved at: {plot_path}")


if __name__ == "__main__":
  app.run(main)
