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

r"""Node Classification Advanced Example.

This example shows how to train a node classification model with the low level
API.

This example contains more-of-less the same code as the Advanced API Notebook,
but without detailed explanations:
http://go/graph-flow/getting_started_advanced_api

Usage example:

# Run locally (you need a GPU)
blaze run -c opt \
//third_party/py/dgf/examples:node_classification_model_advanced
"""

from itertools import islice
import typing
from absl import app
from absl import flags
import dgf
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping
import matplotlib.pyplot as plt
import numpy as np
import optax

_OUTPUT_DIR = flags.DEFINE_string("output_dir", "/tmp/dgf_model", "")


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  dgf.filesystem.makedirs(_OUTPUT_DIR.value)

  # Get the graph data
  graph, schema = dgf.io.fetch_ogb_graph("arxiv")

  # Print the graph schema
  dgf.analyse.print_schema(schema)

  # If your graph doesn't have semantic information (e.g., features are
  # numerical, categorical, embedding), training will be poor and you'll get
  # warning messages. You can set the feature semantics manually. For example:
  #   schema.node_sets["paper"].features["labels"].semantic = dgf.data.FeatureSemantic.CATEGORICAL
  # Or use the automatic inference tool. For example:
  #   schema = dgf.analyse.infer_schema_semantic(schema)

  # [Optional] Validate the graph data
  dgf.validate.validate_graph(graph, schema)

  # Split the dataset between training and validation
  num_nodes = graph.node_sets["nodes"].num_nodes
  assert num_nodes is not None
  all_seed_node_idxs = np.arange(num_nodes)
  np.random.seed(42)
  np.random.shuffle(all_seed_node_idxs)
  num_valid = int(num_nodes * 0.2)  # Keep 20% of the nodes for validation
  valid_seed_node_idxs = all_seed_node_idxs[:num_valid]
  train_seed_node_idxs = all_seed_node_idxs[num_valid:]
  print("Num train nodes:", len(train_seed_node_idxs))
  print("Num valid nodes:", len(valid_seed_node_idxs))

  # Create a Graph Sampler
  plan = dgf.sampling.SimpleSamplingConfig(
      seed_nodeset="nodes",
      # Maximum distances to consider.
      num_hops=2,
      # How many neighbors we consider at each hop.
      hop_width=10,
      # Follow the edges on both directions.
      reverse=True,
  )

  # Create a graph sampler / index the graph.
  sampler = dgf.sampling.create_sampler(
      graph=graph,
      plan=plan,
      schema=schema,
      batch_size=32,
  )

  def batch_generator(
      seed_node_idxs,
      padding=None,
      also_return_merge_offsets=False,
      batch_size=32,
  ):
    for seed_node_idxs in dgf.transform.batch_indices_generator(
        seed_node_idxs,
        batch_size=batch_size,
        drop_remainder=True,
        shuffle=False,
    ):
      # Sample the graphs
      samples = sampler.sample(seed_node_idxs.tolist())

      try:
        # Merge the graph samples into a single graph.
        merged_samples, merge_offsets = dgf.transform.merge_graphs(
            graphs=samples,
            schema=schema,
            padding=padding,
            sentinel_offset=False,
        )
      except dgf.exception.InsufficientPaddingError:
        # Skip if the number of nodes is too large for the padding.
        continue

      if also_return_merge_offsets:
        yield merged_samples, merge_offsets
      else:
        yield merged_samples

  # Compute the optimal padding for JAX
  padding = dgf.analyse.padding_from_graph_generator(
      schema=schema, graphs=islice(batch_generator(train_seed_node_idxs), 200)
  )

  # Compute feature statistics
  feature_stats = dgf.analyse.feature_statistics_from_graphs(
      graphs=islice(batch_generator(train_seed_node_idxs), 200),
      schema=schema,
  )

  # Feature normalizer
  normalizer = dgf.transform.auto_normalize(schema=schema, stats=feature_stats)

  # Define and train model
  class Model(nn.Module):
    schema: dgf.data.GraphSchema
    num_label_classes: int

    @nn.compact
    def __call__(
        self,
        batch: tuple[dgf.data.JaxInMemoryGraph, jnp.ndarray],
        training: bool,
    ) -> jnp.ndarray:
      print("...Tracing model")
      graph, seed_node_idxs = batch

      # Embed features into a single embedding per nodeset.
      embedder_config = dgf.jax.layers.EmbedGraphConfig()
      embedder = embedder_config.make(schema=self.schema)
      embedded_output_schema = embedder_config.output_schema(self.schema)
      graph = embedder(graph, training=training)

      # A MLP layer on each nodeset independently. This also ensure that all the
      # nodeset embeddings have the same size.
      for _, nodeset_value in graph.node_sets.items():
        mlp = dgf.jax.layers.ResidualMLPV2Config(
            dims=128, norm="layer_norm", residual=False
        ).make()
        nodeset_value.features["embedding"] = mlp(
            nodeset_value.features["embedding"]
        )

      # Message passing between nodes
      for _ in range(2):
        message_passer = dgf.jax.layers.HeterogeneousGraphConvolutionConfig(
            dims=128
        ).make(embedded_output_schema)
        graph = message_passer(graph, training=training)

      # Extract embedding of seed nodes
      node_embedding = graph.node_sets["nodes"].features["embedding"][
          seed_node_idxs
      ]

      logits = nn.Dense(self.num_label_classes)(node_embedding)
      return logits

  # The model received the normalized schema, without the label column.
  model_schema = normalizer.output_schema()
  num_label_classes = (
      model_schema.node_sets["nodes"].features["labels"].num_categorical_values
  )
  assert num_label_classes is not None
  del model_schema.node_sets["nodes"].features["labels"]

  # Instantiate the model
  model = Model(schema=model_schema, num_label_classes=num_label_classes)

  # The model loss
  def loss_fn(
      params: jaxtyping.PyTree,
      batch: jaxtyping.PyTree,
      labels: jnp.ndarray,
      rng_key: jnp.ndarray | None,
      training: bool,
  ) -> jnp.ndarray | jaxtyping.PyTree:
    if rng_key is not None:
      rngs = {"dropout": rng_key}
    else:
      rngs = None
    logits = typing.cast(
        jnp.ndarray, model.apply(params, batch, training=training, rngs=rngs)
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    accuracy = jnp.argmax(logits, axis=-1) == labels
    return jnp.mean(loss), {"accuracy": accuracy.mean()}

  # The model training step
  @jax.jit
  def train_step(params, opt_state, batch, rng_key):
    graph, seed_node_idxs = batch
    labels = graph.node_sets["nodes"].features["labels"][seed_node_idxs]
    (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, batch, labels, rng_key, True
    )
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, {"loss": loss, **aux_data}

  # The model validation step
  @jax.jit
  def valid_step(params, opt_state, batch):
    graph, seed_node_idxs = batch
    labels = graph.node_sets["nodes"].features["labels"][seed_node_idxs]
    loss, aux = loss_fn(params, batch, labels, None, False)
    return {"loss": loss, **aux}

  # Process a batch of data before sending it to the model.
  def process_batch(
      graph: dgf.data.InMemoryGraph,
      merge_offsets: dict[str, np.ndarray],
  ):
    normalized_graph = normalizer.normalize_numpy(graph)
    jax_normalized_graph = dgf.convert.graph_to_jax_graph(normalized_graph)
    seed_node_idxs = jnp.asarray(merge_offsets["nodes"])
    return jax_normalized_graph, seed_node_idxs

  # Generate the training data
  def infinite_train_dataset_iterator():
    while True:
      for raw_batch, merge_offsets in batch_generator(
          train_seed_node_idxs,
          batch_size=32,
          padding=padding,
          also_return_merge_offsets=True,
      ):
        yield process_batch(raw_batch, merge_offsets)

  # Generate the validation data
  def finite_valid_dataset_iterator():
    for raw_batch, merge_offsets in batch_generator(
        valid_seed_node_idxs,
        batch_size=32,
        padding=padding,
        also_return_merge_offsets=True,
    ):
      yield process_batch(raw_batch, merge_offsets)

  # A basic optimizer
  opt = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=0.0001),
  )

  # Train the model
  training_output = dgf.jax.train(
      model=model,
      opt=opt,
      train_step=train_step,
      valid_step=valid_step,
      dataset_iterator=infinite_train_dataset_iterator(),
      valid_dataset_iterator_fn=finite_valid_dataset_iterator,
      num_train_steps=10_000,
      valid_every_n_steps=1000,
      train_log_every_n_steps=100,
      rng_key=jax.random.PRNGKey(42),
      print_logs=True,
  )

  # Plot the training logs.
  plt.figure(figsize=(16, 6))

  def extract_steps(logs):
    return [log.step for log in logs]

  def extract_metric(metric, logs):
    return [log.metrics[metric] for log in logs]

  metrics = ["loss", "accuracy"]
  for metrix_idx, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), metrix_idx + 1)
    plt.plot(
        extract_steps(training_output.train_logs),
        extract_metric(metric, training_output.train_logs),
        label="train",
    )
    plt.plot(
        extract_steps(training_output.valid_logs),
        extract_metric(metric, training_output.valid_logs),
        label="valid",
    )
    plt.xlabel("step")
    plt.ylabel(metric)

  plt.tight_layout()
  plt.savefig(_OUTPUT_DIR.value + "/training.png")

  print("Results are available in :", _OUTPUT_DIR.value)


if __name__ == "__main__":
  app.run(main)
