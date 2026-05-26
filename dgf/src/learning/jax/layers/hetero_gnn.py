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

"""GNN layers for heterogeneous graphs."""

import collections
import dataclasses
import textwrap
from typing import Dict, List, Optional, Tuple
import dataclasses_json
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from flax import linen as nn
import jax
import jax.numpy as jnp

# A plan is a list of (edge name, is_reversed) indicating which edge
# is used to propagate the message.
Plan = List[Tuple[str, bool]]

# A sorted plan groups plan items by destination nodesets. More precisely, a
# sorted plan maps for each target nodesets, the list of
# (edgeset name, source nodeset, is_reversed).
SortedPlan = Dict[str, List[Tuple[str, str, bool]]]


def sort_plan(plan: Plan, schema: schema_lib.GraphSchema) -> SortedPlan:
  """Sorts a message passing plan by destination nodeset.

  Args:
    plan: A list of (edge name, is_reversed) indicating which edge is used to
      propagate the message.
    schema: The graph schema.

  Returns:
    A dictionary where keys are destination nodeset names and values are lists
    of (edgeset name, source nodeset name, is_reversed).
  """
  sorted_plan = collections.defaultdict(list)
  for edgeset_name, reverse in plan:
    orig_src = schema.edge_sets[edgeset_name].source
    orig_dst = schema.edge_sets[edgeset_name].target
    if reverse:
      # Message flows from orig_dst to orig_src
      message_dst = orig_src
      message_src = orig_dst
    else:
      # Message flows from orig_src to orig_dst
      message_dst = orig_dst
      message_src = orig_src
    sorted_plan[message_dst].append((edgeset_name, message_src, reverse))
  return sorted_plan


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class HeterogeneousGraphConvolutionConfig(common.ArchitectureProvider):
  """Configuration for HeterogeneousGraphConvolution.

  Attributes:
    plan: Message passing plan. A list of (edge_set_name, is_reversed) tuples.
      If None, messages are passed along all edges in both directions.
    embedding_feature: Name of the node feature to use for embeddings.
    dims: Dimension of the embeddings and hidden layers. Used to build the
      default values of `message`, `update`, and `post`.
    dropout_rate: Dropout rate. Used to build the default values of `update` and
      `post`.
    activation: Activation function to use. Used to build the default values of
      `update` and `post`.
    message_pooling: Pooling method for aggregating messages ('sum' or 'mean').
    message: Optional module to apply to edge features to generate messages.
      Defaults to a single-layer MLP.
    update: Optional module to apply to node embeddings after message passing,
      combining the old embedding and aggregated messages. Defaults to a
      single-layer MLP with layer norm and activation.
    post: Optional module applied after the update step, typically a
      transformer-like MLP. Defaults to a two-layer ResidualMLP.
  """

  plan: Optional[List[Tuple[str, bool]]] = None
  embedding_feature: str = "embedding"
  dims: int = 128
  dropout_rate: float = 0.1
  message_pooling: str = "sum"

  message: Optional[common.BuildableModule] = layer_registry.field(default=None)
  update: Optional[common.BuildableModule] = layer_registry.field(default=None)
  post: Optional[common.BuildableModule] = layer_registry.field(default=None)

  def __post_init__(self):

    if self.message is None:
      self.message = standard.GenericBlockConfig("LAL", dims=self.dims)
    if self.update is None:
      self.update = standard.GenericBlockConfig(
          "LADL", dims=self.dims, dropout_rate=self.dropout_rate
      )

    if self.post is None:
      self.post = standard.modern_residual_mlp(
          dims=self.dims, dropout_rate=self.dropout_rate
      )

  def make(
      self, schema: schema_lib.GraphSchema, name: Optional[str] = None
  ) -> "HeterogeneousGraphConvolution":
    return HeterogeneousGraphConvolution(config=self, schema=schema, name=name)

  def architecture(self) -> str:
    parts = []
    parts.append("X = ...")
    parts.append("MPNN:")
    parts.append("  Message:")
    parts.append(textwrap.indent(self.message.architecture(), prefix="    "))
    parts.append("  Update:")
    parts.append(textwrap.indent(self.update.architecture(), prefix="    "))
    parts.append("Residual(X)")
    parts.append("# Post MPNN")
    parts.append(self.post.architecture())
    return "\n".join(parts)


class HeterogeneousGraphConvolution(nn.Module):
  """A single layer of heterogeneous Graph Neural Network message passing.

  This layer performs message passing on a heterogeneous graph. It consists of
  two main blocks:
  1.  A GNN step with a residual connection.
  2.  A transformer-like residual Multi-Layer Perceptron (MLP).

  All node sets are assumed to have a feature specified by
  `config.embedding_feature` (defaulting to "embedding"), and these features
  must all have the same dimension. The `EmbedGraph` layer can be used to
  preprocess the graph to meet this requirement.

  Usage Example:

  ```python
  # Assume a graph and schema with node sets 'n1' and 'n2', and edge sets 'e1'
  # (n1->n1) and 'e2' (n1->n2).
  graph: JaxInMemoryGraph = ...
  schema: GraphSchema = ...

  # Embed all features into a single embedding vector per node.
  graph_embedder = EmbedGraphConfig().make(schema)
  embedded_graph = graph_embedder(graph)
  embedded_schema = graph_embedder.output_schema(schema)

  # Configure and apply message passing layers.
  message_passer_config = HeterogeneousGraphConvolutionConfig()
  for _ in range(num_layers):
    # Re-create the layer in each iteration to ensure separate weights.
    message_passer = message_passer_config.make(embedded_schema)
    embedded_graph = message_passer(embedded_graph, training=True)
  ```

  Message Passing Plan:

  By default, messages are passed along all edges in both forward and backward
  directions. This behavior can be customized using the `plan` argument in the
  `HeterogeneousGraphConvolutionConfig`. The plan is a list of tuples, where
  each
  tuple `(edge_set_name, is_reversed)` specifies an edge set and the direction
  of message flow.

  ```python
  # Example of a custom message passing plan:
  message_plan = [
      ("e1", False),  # Messages flow forward along 'e1' (n1 -> n1)
      ("e1", True),   # Messages flow backward along 'e1' (n1 <- n1)
      ("e2", False),  # Messages flow forward along 'e2' (n1 -> n2)
      # No ("e2", True), so no messages flow backward along 'e2' (n2 <- n1)
  ]
  config = HeterogeneousGraphConvolutionConfig(plan=message_plan)
  ```

  Attributes:
    config: The configuration object for the message passing layer.
    schema: The graph schema. Only the source and target fields of the edge sets
      are essential for this layer.
    sorted_plan: The message passing plan, sorted and grouped by the destination
      node set. Each entry maps a destination node set name to a list of
      `(edge_set_name, source_node_set_name, is_reversed)` tuples.
  """

  config: HeterogeneousGraphConvolutionConfig
  schema: schema_lib.GraphSchema

  sorted_plan: SortedPlan = dataclasses.field(init=False)

  def __post_init__(self):
    if self.config.plan is None:
      self.config.plan = []
      for edge_name in self.schema.edge_sets:
        self.config.plan.append((edge_name, False))
        self.config.plan.append((edge_name, True))
    self.sorted_plan = sort_plan(self.config.plan, self.schema)
    super().__post_init__()

  @nn.compact
  def __call__(
      self,
      graph: jax_in_memory_graph.JaxInMemoryGraph,
      training: bool,
  ) -> jax_in_memory_graph.JaxInMemoryGraph:
    """Computes the message passing.

    Args:
      graph: The graph structure.
      training: Is the model in training or serving/evaluation?

    Returns:
      The graph after message passing.
    """

    config = self.config
    assert config.message is not None
    assert config.update is not None
    assert config.post is not None

    # Initialize with all original node sets to avoid dropping any.
    new_node_sets = dict(graph.node_sets)

    # TODO: Add position encoding.

    # Save the nodeset values for the residual.
    res_node_features = {
        nodeset_name: nodeset.features[config.embedding_feature]
        for nodeset_name, nodeset in graph.node_sets.items()
    }

    # Message passing for each nodeset.
    for dst_nodeset_name in sorted(self.sorted_plan.keys()):
      # Compute the new state of the "dst_nodeset" node.

      # Grab data.
      edges_and_src_nodes = self.sorted_plan[dst_nodeset_name]
      num_dst_nodes = graph.node_sets[dst_nodeset_name].num_nodes
      assert num_dst_nodes is not None
      dst_values = res_node_features[dst_nodeset_name]

      # Compute and aggregate messages from connected nodes.
      neighbor_aggregates = []
      for edgeset_name, message_src_nodeset, reverse in edges_and_src_nodes:

        # Message passing
        # ===============

        # Gather the edges
        source_idxs = graph.edge_sets[edgeset_name].adjacency[0]
        target_idxs = graph.edge_sets[edgeset_name].adjacency[1]
        if reverse:
          source_idxs, target_idxs = target_idxs, source_idxs

        # Gather the edge values
        src_values = res_node_features[message_src_nodeset]
        src_edge_values = src_values[source_idxs]
        dst_edge_values = dst_values[target_idxs]
        edge_values = jnp.concatenate(
            [src_edge_values, dst_edge_values], axis=-1
        )

        # Weights + activation on the edges.
        messages = config.message.make()(edge_values, training=training)

        # Group by target nodeset
        # TODO(gbm): Create utilities so this does not feel this manual.
        neighbor_aggregate = jax.ops.segment_sum(
            messages,
            target_idxs,
            num_dst_nodes,
        )
        if self.config.message_pooling == "mean":
          degrees = jax.ops.segment_sum(
              jnp.ones((target_idxs.shape[0], 1)),
              target_idxs,
              num_dst_nodes,
          )
          degrees = jnp.maximum(degrees, 1.0)
          neighbor_aggregate = neighbor_aggregate / degrees
        elif self.config.message_pooling == "sum":
          pass
        else:
          raise ValueError("Unsupported message_pooling:")
        neighbor_aggregates.append(neighbor_aggregate)

      # Join messages + first residual
      stacked_aggregates = jnp.stack(neighbor_aggregates)
      if self.config.message_pooling == "mean":
        combined_aggregates = jnp.mean(stacked_aggregates, axis=0)
      elif self.config.message_pooling == "sum":
        combined_aggregates = jnp.sum(stacked_aggregates, axis=0)
      else:
        raise ValueError("Unsupported message_pooling")

      combined = jnp.concatenate([dst_values, combined_aggregates], axis=1)
      combined = config.update.make()(combined, training=training)

      # Residual
      node_values = combined + res_node_features[dst_nodeset_name]

      # Feed-forward
      node_values = config.post.make()(node_values, training=training)

      new_node_sets[dst_nodeset_name] = jax_in_memory_graph.JaxInMemoryNodeSet(
          num_nodes=num_dst_nodes,
          features={self.config.embedding_feature: node_values},
      )

    return jax_in_memory_graph.JaxInMemoryGraph(
        node_sets=new_node_sets,
        edge_sets=graph.edge_sets,
    )
