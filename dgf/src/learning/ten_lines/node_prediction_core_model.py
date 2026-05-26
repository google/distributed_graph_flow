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

"""The core model a.k.a., the jax model at the center of a model."""

import dataclasses
import textwrap
from typing import Tuple
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import classification as classification_lib
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers import regression as regression_lib
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from dgf.src.util import log
import flax.linen as nn
import jax

# A batch of data for the node prediction model.
# It is composed of a graph and the indices of the seed nodes in the target
# nodeset.
Batch = Tuple[jax_in_memory_graph_lib.JaxInMemoryGraph, jax.Array]


@dataclasses.dataclass(frozen=True, kw_only=True)
class CoreModelConfig(common.ArchitectureProvider):
  """Configuration of the core JAX/FLAX model module."""

  embbed_graph: preprocess.EmbedGraphConfig = layer_registry.field()
  pre_mlp: common.GenericLayer = layer_registry.field()
  graph_conv: hetero_gnn.HeterogeneousGraphConvolutionConfig = (
      layer_registry.field()
  )
  post_mlp: common.GenericLayer = layer_registry.field()
  head: (
      classification_lib.ClassificationHeadConfig
      | regression_lib.RegressionHeadConfig
  ) = layer_registry.field()
  target_nodeset: str
  num_layers: int
  dropout: float

  def make(self, schema: schema_lib.GraphSchema) -> "CoreModel":
    return CoreModel(
        config=self,
        schema=schema,
    )

  def architecture(self) -> str:
    parts = []
    parts.append(self.embbed_graph.architecture())
    parts.append(self.pre_mlp.architecture())
    parts.append(f"Graph Convolution Block x{self.num_layers}:")
    parts.append(textwrap.indent(self.graph_conv.architecture(), prefix="    "))
    parts.append(self.post_mlp.architecture())
    parts.append(self.head.architecture())
    return "\n".join(parts)


class CoreModel(nn.Module):
  """Core JAX/FLAX model module."""

  config: CoreModelConfig
  schema: schema_lib.GraphSchema

  @nn.compact
  def __call__(self, batch: Batch, training: bool):
    log.info("...Tracing model")
    graph, seed_node_idxs = batch

    # Embed the features.
    graph_embedder = self.config.embbed_graph.make(schema=self.schema)
    embedded_schema = self.config.embbed_graph.output_schema(self.schema)
    graph = graph_embedder(graph, training=training)

    # A MLP layer on each nodeset independently. Also make sure all the node
    # embeddings have the same size.
    for nodeset_name, nodeset_value in graph.node_sets.items():
      nodeset_value.features["embedding"] = self.config.pre_mlp.make(
          name=f"pre_mlp_{nodeset_name}"
      )(nodeset_value.features["embedding"])

    # Message passing between nodes
    conv_config = self.config.graph_conv

    class GNNBlock(nn.Module):

      @nn.compact
      def __call__(self, graph, _):
        message_passer = conv_config.make(embedded_schema)
        graph = message_passer(graph, training=training)
        return graph, None

    scanned_gnn = nn.scan(
        GNNBlock,
        variable_axes={"params": 0},
        split_rngs={"params": True, "dropout": True},
        length=self.config.num_layers,
    )
    graph, _ = scanned_gnn()(graph, None)

    # Extract embedding of seed nodes
    node_embedding = graph.node_sets[self.config.target_nodeset].features[
        "embedding"
    ][seed_node_idxs]

    node_embedding = self.config.post_mlp.make(name="post_mlp")(node_embedding)

    # Classification or regression head (no learning parameters).
    logits = self.config.head.make()(node_embedding, training=training)
    return logits
