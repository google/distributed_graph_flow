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

"""The core JAX/FLAX model for link prediction."""

import dataclasses
import textwrap
from typing import Optional, Tuple
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from dgf.src.util import log
import flax.linen as nn
import jax
from jax import tree_util as jax_tree_util
import jax.numpy as jnp


@jax_tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class Batch:
  """Batch of training data in JAX format.

  This is the JAX equivalent of `GNNLinkDatasetPreparatorSample`.

  It contains graph samples centered around positive and negative edges, used
  for
  link prediction training.
  `positive_source_graph`, `positive_target_graph`, and `negative_target_graph`
  are graphs sampled around source nodes, target nodes of positive edges, and
  target nodes of negative edges, respectively. Each graph is a mini-batch of
  merged graph samples.

  `positive_source_offset`, `positive_target_offset`, and
  `negative_target_offset` contain the indices of seed nodes in their respective
  graphs. If `b` is batch size and `n` is the number of negative samples per
  positive edge, then `positive_source_offset` and `positive_target_offset`
  each contain `b` indices (one for each positive edge in the batch).
  `negative_target_offset` contains `b * n` indices, where elements `[i*n :
  (i+1)*n]` are offsets for negative target nodes corresponding to the i-th
  positive edge.
  """

  positive_source_graph: jax_in_memory_graph_lib.JaxInMemoryGraph
  positive_target_graph: jax_in_memory_graph_lib.JaxInMemoryGraph
  negative_target_graph: jax_in_memory_graph_lib.JaxInMemoryGraph

  positive_source_offset: jax.Array
  positive_target_offset: jax.Array
  negative_target_offset: jax.Array


@jax_tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class InferenceBatch:
  """Batch of inference data in JAX format."""

  source_graph: jax_in_memory_graph_lib.JaxInMemoryGraph
  target_graph: jax_in_memory_graph_lib.JaxInMemoryGraph

  source_offset: jax.Array
  target_offset: jax.Array


@dataclasses.dataclass(frozen=True, kw_only=True)
class EncoderConfig(common.ArchitectureProvider):
  """Configuration of a FLAX module to encode a graph into an embedding."""

  embbed_graph: preprocess.EmbedGraphConfig = layer_registry.field()
  pre_mlp: common.GenericLayer = layer_registry.field()
  graph_conv: hetero_gnn.HeterogeneousGraphConvolutionConfig = (
      layer_registry.field()
  )
  post_mlp: common.GenericLayer = layer_registry.field()
  num_layers: int
  dropout: float

  def make(
      self,
      schema: schema_lib.GraphSchema,
      target_nodeset: str,
      name: Optional[str] = None,
  ) -> "Encoder":
    return Encoder(
        config=self, schema=schema, target_nodeset=target_nodeset, name=name
    )

  def architecture(self) -> str:
    parts = []
    parts.append(self.embbed_graph.architecture())
    parts.append(self.pre_mlp.architecture())
    parts.append(f"Graph Convolution Block x{self.num_layers}:")
    parts.append(textwrap.indent(self.graph_conv.architecture(), prefix="    "))
    parts.append(self.post_mlp.architecture())
    return "\n".join(parts)


class Encoder(nn.Module):
  """Encode a graph into an embedding."""

  config: EncoderConfig
  target_nodeset: str
  schema: schema_lib.GraphSchema

  @nn.compact
  def __call__(self, batch, training: bool):
    graph, seed_idxs = batch

    # Embed the features.
    graph_embedder = self.config.embbed_graph.make(schema=self.schema)
    embedded_schema = self.config.embbed_graph.output_schema(self.schema)
    graph = graph_embedder(graph, training=training)

    # A MLP layer on each nodeset independently. Also make sure all the node
    # embeddings have the same size.
    for nodeset_name, nodeset_value in graph.node_sets.items():
      nodeset_value.features["embedding"] = self.config.pre_mlp.make(
          name=f"pre_conv_{nodeset_name}"
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
    embedding = graph.node_sets[self.target_nodeset].features["embedding"][
        seed_idxs
    ]

    return embedding


@dataclasses.dataclass(frozen=True, kw_only=True)
class CoreModelConfig(common.ArchitectureProvider):
  """Configuration of the core JAX/FLAX model module."""

  source_nodeset: str
  target_nodeset: str
  encoder_config: EncoderConfig

  def make(
      self,
      source_schema: schema_lib.GraphSchema,
      target_schema: schema_lib.GraphSchema,
  ) -> "CoreModel":
    return CoreModel(
        config=self,
        source_schema=source_schema,
        target_schema=target_schema,
    )

  def architecture(self) -> str:
    return self.encoder_config.architecture()


class CoreModel(nn.Module):
  """Core JAX/FLAX model module for edge prediction.

  This module return the logits between the positive-sources <->
  positive-targets, and positive-sources <-> negative targets.
  """

  config: CoreModelConfig
  source_schema: schema_lib.GraphSchema
  target_schema: schema_lib.GraphSchema

  def setup(self):
    # Initialize the encoders
    self.src_encoder = self.config.encoder_config.make(
        self.source_schema,
        target_nodeset=self.config.source_nodeset,
        name="source_encoder",
    )
    self.trg_encoder = self.config.encoder_config.make(
        self.target_schema,
        target_nodeset=self.config.target_nodeset,
        name="target_encoder",
    )

  def _embeddings_to_logits(
      self, src_embedding: jax.Array, trg_embedding: jax.Array
  ):
    return jnp.sum(src_embedding * trg_embedding, axis=-1)

  def __call__(
      self, batch: Batch, training: bool
  ) -> Tuple[jax.Array, jax.Array]:
    """Computes positive and negative logits for a training batch."""

    log.info("...Tracing model")

    # Encode the graphs
    pos_src_embedding = self.src_encoder(
        (batch.positive_source_graph, batch.positive_source_offset),
        training=training,
    )
    pos_trg_embedding = self.trg_encoder(
        (batch.positive_target_graph, batch.positive_target_offset),
        training=training,
    )
    neg_trg_embedding = self.trg_encoder(
        (batch.negative_target_graph, batch.negative_target_offset),
        training=training,
    )

    # Match the negatives to the positives
    num_negatives_per_positive = (
        neg_trg_embedding.shape[0] // pos_trg_embedding.shape[0]
    )
    repeated_pos_src_embedding = jnp.repeat(
        pos_src_embedding, num_negatives_per_positive, axis=0
    )

    positive_logits = self._embeddings_to_logits(
        pos_src_embedding, pos_trg_embedding
    )
    negative_logits = self._embeddings_to_logits(
        repeated_pos_src_embedding, neg_trg_embedding
    )
    return positive_logits, negative_logits

  def call_inference(self, batch: InferenceBatch) -> jax.Array:
    """Computes logics for an inference batch."""

    log.info("...Tracing inference model")

    # Encode the graphs
    src_embedding = self.src_encoder(
        (batch.source_graph, batch.source_offset), training=False
    )
    trg_embedding = self.trg_encoder(
        (batch.target_graph, batch.target_offset), training=False
    )
    return self._embeddings_to_logits(src_embedding, trg_embedding)

  def call_src_encoder(
      self, graph: jax_in_memory_graph_lib.JaxInMemoryGraph, offset: jax.Array
  ) -> jax.Array:
    """Computes embeddings for source nodes."""
    return self.src_encoder((graph, offset), training=False)

  def call_trg_encoder(
      self, graph: jax_in_memory_graph_lib.JaxInMemoryGraph, offset: jax.Array
  ) -> jax.Array:
    """Computes embeddings for target nodes."""
    return self.trg_encoder((graph, offset), training=False)
