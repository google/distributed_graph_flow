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

"""The core JAX/Flax model for unsupervised node embedding with GraphMAE."""

import dataclasses
import functools

import dataclasses_json
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import graphmae as graphmae_lib
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from dgf.src.util import log
import flax.linen as nn
import jax
import jax.numpy as jnp

# A batch of data for the node embedding model: a normalized padded graph and
# the indices of the seed nodes in the target nodeset.
Batch = graphmae_lib.Batch


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True, kw_only=True)
class CoreModelConfig(common.ArchitectureProvider):
  """Configuration of the core JAX/Flax node embedding model module."""

  embed_graph: preprocess.EmbedGraphConfig = layer_registry.field()
  pre_mlp: common.GenericLayer = layer_registry.field()
  graphmae: graphmae_lib.GraphMAEConfig = layer_registry.field()

  def make(self, schema: schema_lib.GraphSchema) -> "CoreModel":
    # Fails early if `pre_mlp` does not project to the GNN dimension.
    self.projected_schema(schema)
    return CoreModel(
        config=self,
        schema=schema,
    )

  def projected_schema(
      self, schema: schema_lib.GraphSchema
  ) -> schema_lib.GraphSchema:
    """Returns the schema of the graph after `embed_graph` and `pre_mlp`.

    This is the schema of the graph given to `GraphMAE`. Since `pre_mlp` is a
    generic layer that does not expose its output dimension, the output shape
    of `pre_mlp` is inferred with `jax.eval_shape` (no computation is done).

    Args:
      schema: The schema of the input graph.

    Returns:
      The schema with a single "embedding" feature per nodeset, of dimension
      `graphmae.encoder_conv.dims`.
    """
    embedded_schema = self.embed_graph.output_schema(schema)
    pre_mlp = self.pre_mlp.make()
    gnn_dims = self.graphmae.get_gnn_dims()
    node_sets = {}
    for nodeset_name, nodeset_schema in embedded_schema.node_sets.items():
      feature_schema = nodeset_schema.features[graphmae_lib.EMBEDDING_FEATURE]
      if feature_schema.shape is None:
        raise ValueError(
            f"The embedding of nodeset {nodeset_name!r} has an unknown shape."
        )
      projected, _ = jax.eval_shape(
          functools.partial(pre_mlp.init_with_output, training=False),
          jax.random.key(0),
          jax.ShapeDtypeStruct((1, *feature_schema.shape), jnp.float32),
      )
      projected_shape = tuple(projected.shape[1:])
      if projected_shape != (gnn_dims,):
        raise ValueError(
            f"`pre_mlp` must project the embedding of nodeset {nodeset_name!r}"
            f" to the GNN dimension `graphmae.encoder_conv.dims` ({gnn_dims})."
            f" Got output shape {projected_shape} for input shape"
            f" {tuple(feature_schema.shape)}."
        )
      node_sets[nodeset_name] = schema_lib.NodeSchema(
          features={
              graphmae_lib.EMBEDDING_FEATURE: dataclasses.replace(
                  feature_schema, shape=projected_shape
              )
          }
      )
    return schema_lib.GraphSchema(
        node_sets=node_sets, edge_sets=embedded_schema.edge_sets
    )

  def architecture(self) -> str:
    return "\n".join([
        self.embed_graph.architecture(),
        self.pre_mlp.architecture(),
        self.graphmae.architecture(),
    ])

  def compute_loss(
      self, output: graphmae_lib.GraphMAEOutput, batch: Batch
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes the total GraphMAE loss and diagnostic metrics for a batch."""
    del batch  # Unused.
    losses = graphmae_lib.compute_losses(output, self.graphmae)
    metrics = graphmae_lib.compute_metrics(output)
    return losses.total(), {
        "recon_loss": losses.reconstruction,
        "var_loss": losses.variance,
        "cov_loss": losses.covariance,
        "recon_cosine": metrics.reconstruction_cosine,
        "recon_relative_error": metrics.reconstruction_relative_error,
        "embedding_rank_ratio": metrics.embedding_rank_ratio,
        "target_rank_ratio": metrics.target_rank_ratio,
    }


class CoreModel(nn.Module):
  """Core JAX/Flax node embedding module wrapping `EmbedGraph` and `GraphMAE`."""

  config: CoreModelConfig
  schema: schema_lib.GraphSchema

  def setup(self) -> None:
    self.embed_graph = self.config.embed_graph.make(
        schema=self.schema, name="embed_graph"
    )
    projected_schema = self.config.projected_schema(self.schema)
    self.pre_mlps = {
        nodeset_name: self.config.pre_mlp.make(name=f"pre_mlp_{nodeset_name}")
        for nodeset_name in projected_schema.node_sets
    }
    self.graphmae = self.config.graphmae.make(
        schema=projected_schema, name="graphmae"
    )

  def _embed_features(
      self,
      graph: jax_in_memory_graph_lib.JaxInMemoryGraph,
      training: bool,
  ) -> jax_in_memory_graph_lib.JaxInMemoryGraph:
    """Embeds raw node features and projects each nodeset to `dims`."""
    embedded = self.embed_graph(graph, training=training)
    new_node_sets = {}
    for nodeset_name in self.schema.node_sets:
      nodeset_value = embedded.node_sets[nodeset_name]
      proj = self.pre_mlps[nodeset_name](
          nodeset_value.features[graphmae_lib.EMBEDDING_FEATURE],
          training=training,
      )
      new_node_sets[nodeset_name] = (
          jax_in_memory_graph_lib.JaxInMemoryNodeSet(
              num_nodes=nodeset_value.num_nodes,
              features={graphmae_lib.EMBEDDING_FEATURE: proj},
          )
      )
    return jax_in_memory_graph_lib.JaxInMemoryGraph(
        node_sets=new_node_sets,
        edge_sets=embedded.edge_sets,
    )

  def __call__(
      self, batch: Batch, training: bool
  ) -> graphmae_lib.GraphMAEOutput:
    log.info("...Tracing model")
    graph, seed_node_idxs = batch
    embedded_graph = self._embed_features(graph, training=training)
    return self.graphmae((embedded_graph, seed_node_idxs), training=training)

  def embed(
      self, batch: Batch, training: bool = False
  ) -> jax.Array:
    """Returns encoder node embeddings of shape `[S, D]` for the seed nodes."""
    graph, seed_node_idxs = batch
    embedded_graph = self._embed_features(graph, training=training)
    all_embeddings = self.graphmae.embed(embedded_graph, training=training)
    return all_embeddings[seed_node_idxs]
