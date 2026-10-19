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
    return CoreModel(
        config=self,
        schema=schema,
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
    return losses.total(), {
        "recon_loss": losses.reconstruction,
        "var_loss": losses.variance,
        "cov_loss": losses.covariance,
    }


class CoreModel(nn.Module):
  """Core JAX/Flax node embedding module wrapping `EmbedGraph` and `GraphMAE`."""

  config: CoreModelConfig
  schema: schema_lib.GraphSchema

  def setup(self) -> None:
    self.embed_graph = self.config.embed_graph.make(
        schema=self.schema, name="embed_graph"
    )
    embedded_schema = self.config.embed_graph.output_schema(self.schema)
    dims = self.config.graphmae.get_gnn_dims()
    for nodeset_schema in embedded_schema.node_sets.values():
      nodeset_schema.features["embedding"].shape = (dims,)
    self.pre_mlps = {
        nodeset_name: self.config.pre_mlp.make(name=f"pre_mlp_{nodeset_name}")
        for nodeset_name in embedded_schema.node_sets
    }
    self.graphmae = self.config.graphmae.make(
        schema=embedded_schema, name="graphmae"
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
          nodeset_value.features["embedding"], training=training
      )
      new_node_sets[nodeset_name] = (
          jax_in_memory_graph_lib.JaxInMemoryNodeSet(
              num_nodes=nodeset_value.num_nodes,
              features={"embedding": proj},
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
