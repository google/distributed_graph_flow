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

"""Graph masking layers for self-supervised learning.

Masking corrupts the input graph so that a model can be trained to recover the
original information, e.g., as in masked graph auto-encoders (GraphMAE,
https://arxiv.org/abs/2205.10803).

All the operations keep the array shapes static.

IMPORTANT: The last node in each nodeset is required to be a padding node. If it
is not, the layers in this file are not safe to use.
"""

import dataclasses

import dataclasses_json
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True, kw_only=True)
class DropEdgesConfig(common.ArchitectureProvider):
  """Configuration of `DropEdge`.

  During training, each edge is dropped independently with probability
  `drop_rate`. Like `nn.Dropout`, the layer does nothing when `training=False`.

  Dropped edges are redirected to the last node of their source and target
  nodesets, so that the array shapes stay fixed. The last node of each nodeset
  must be a padding (sentinel) node with no real edges, in which case this is
  equivalent to removing the edges. Graphs padded by `GraphMerger` satisfy this
  condition: their padding edges are connected in the same way.

  Usage example:

    ```python
    layer = DropEdgesConfig(drop_rate=0.2).make(schema)
    graph = layer.apply({}, graph, training=True, rngs={"dropout": rng})
    ```

  Attributes:
    drop_rate: Fraction of edges to drop (0 = keep all, 1 = drop all).
  """

  drop_rate: float

  def make(
      self, schema: schema_lib.GraphSchema, name: str | None = None
  ) -> "DropEdges":
    return DropEdges(config=self, schema=schema, name=name)

  def architecture(self) -> str:
    return f"DropEdges(drop_rate={self.drop_rate})"


class DropEdges(nn.Module):
  """Drops random edges. See `DropEdgesConfig`."""

  config: DropEdgesConfig
  schema: schema_lib.GraphSchema

  @nn.compact
  def __call__(
      self,
      graph: jax_in_memory_graph_lib.JaxInMemoryGraph,
      training: bool,
  ) -> jax_in_memory_graph_lib.JaxInMemoryGraph:
    """Drops random edges.

    Args:
      graph: The input graph. The last node of each nodeset is a padding node.
      training: Whether the model is in training mode. If False (evaluation or
        inference), the graph is returned unchanged.

    Returns:
      The graph with the dropped edges redirected to the padding nodes.
    """

    # No-op if not training or drop rate is zero.
    if not training or self.config.drop_rate <= 0.0:
      return graph

    edge_sets = dict(graph.edge_sets)
    for name, edge_schema in self.schema.edge_sets.items():
      edge_set = graph.edge_sets[name]
      num_source_nodes = graph.node_sets[edge_schema.source].num_nodes
      num_target_nodes = graph.node_sets[edge_schema.target].num_nodes
      assert num_source_nodes is not None and num_target_nodes is not None
      # Dropped edges connect the last (padding) nodes of the source and target
      # nodesets, like the padding edges added by `GraphMerger`.
      padding_edge = jnp.array([[num_source_nodes - 1], [num_target_nodes - 1]])
      keep_mask = jax.random.bernoulli(
          self.make_rng("dropout"),
          p=1.0 - self.config.drop_rate,
          shape=(edge_set.num_edges(),),
      )
      edge_sets[name] = dataclasses.replace(
          edge_set,
          adjacency=jnp.where(keep_mask, edge_set.adjacency, padding_edge),
      )

    return dataclasses.replace(graph, edge_sets=edge_sets)


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True, kw_only=True)
class MaskNodeFeaturesConfig(common.ArchitectureProvider):
  """Configuration of a layer masking the features of random nodes.

  A fraction `mask_rate` of the nodes of `nodeset` is selected at random. The
  feature of the selected nodes is replaced by a learned mask token or, for a
  fraction `replace_rate` of them, by the feature of another random node.

  The layer uses the "masking" random stream. Masking defines the
  reconstruction objective, so it is applied both during training and
  evaluation, and the "masking" stream must be provided in both cases.

  Usage example:

    ```python
    layer = MaskNodeFeaturesConfig(nodeset="paper", mask_rate=0.3).make()
    variables = layer.init({"params": params_rng, "masking": rng}, graph)
    masked_graph, is_masked = layer.apply(
        variables, graph, rngs={"masking": rng}
    )
    ```

  Attributes:
    nodeset: The nodeset to mask.
    feature: The masked feature. Must be a float tensor of shape [num_nodes,
      dim].
    mask_rate: Fraction of the nodes of `nodeset` to mask.
    replace_rate: Fraction of the masked nodes whose feature is replaced by the
      feature of a random node instead of the mask token.
  """

  nodeset: str
  feature: str = "embedding"
  mask_rate: float
  replace_rate: float = 0.0

  def make(self, name: str | None = None) -> "MaskNodeFeatures":
    return MaskNodeFeatures(config=self, name=name)

  def architecture(self) -> str:
    return (
        f"MaskNodeFeatures(nodeset={self.nodeset!r},"
        f" mask_rate={self.mask_rate}, replace_rate={self.replace_rate})"
    )


class MaskNodeFeatures(nn.Module):
  """Masks the features of random nodes. See `MaskNodeFeaturesConfig`."""

  config: MaskNodeFeaturesConfig

  @nn.compact
  def __call__(
      self,
      graph: jax_in_memory_graph_lib.JaxInMemoryGraph,
  ) -> tuple[jax_in_memory_graph_lib.JaxInMemoryGraph, jax.Array]:
    """Masks the features of random nodes.

    Args:
      graph: The input graph.

    Returns:
      A tuple `(masked_graph, is_masked)` where `is_masked` is a boolean array
      of shape [num_nodes]. Padding nodes can be masked; callers should combine
      `is_masked` with their own mask of real nodes if needed.
    """
    nodeset_name = self.config.nodeset
    nodeset = graph.node_sets[nodeset_name]
    x = nodeset.features[self.config.feature]
    num_nodes = nodeset.num_nodes
    assert num_nodes is not None

    mask_token = self.param(
        "mask_token", nn.initializers.zeros, (1, x.shape[-1])
    )

    rng = self.make_rng("masking")
    perm_rng, replace_rng, choice_rng = jax.random.split(rng, 3)

    perm = jax.random.permutation(perm_rng, num_nodes)
    num_mask_nodes = int(self.config.mask_rate * num_nodes)
    is_masked = (
        jnp.zeros(num_nodes, dtype=jnp.bool_)
        .at[perm[:num_mask_nodes]]
        .set(True)
    )

    # A fraction `replace_rate` of the masked nodes get the feature of a random
    # donor node, the others get the mask token.
    replace_draw = jax.random.uniform(replace_rng, shape=(num_nodes,))
    is_replaced = is_masked & (replace_draw < self.config.replace_rate)
    donor_indices = jax.random.randint(
        choice_rng, shape=(num_nodes,), minval=0, maxval=num_nodes
    )
    out_x = jnp.where(is_masked[:, None], mask_token, x)
    out_x = jnp.where(is_replaced[:, None], x[donor_indices], out_x)

    masked_nodeset = dataclasses.replace(
        nodeset, features={**nodeset.features, self.config.feature: out_x}
    )
    masked_graph = dataclasses.replace(
        graph, node_sets={**graph.node_sets, nodeset_name: masked_nodeset}
    )
    return masked_graph, is_masked
