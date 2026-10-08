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
from collections.abc import Callable
import dataclasses
import functools
import textwrap
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
Plan = list[tuple[str, bool]]

# A sorted plan groups plan items by destination nodesets. More precisely, a
# sorted plan maps for each target nodesets, the list of
# (edgeset name, source nodeset, is_reversed).
SortedPlan = dict[str, list[tuple[str, str, bool]]]

# Minimum number of edges for sorting by target to pay off on GPU.
_SORT_MIN_EDGES = 50000
# Node indices below this bound fit in uint16 (narrow sort keys / values).
_UINT16_MAX_NODES = 2**16


def should_sort_edges(
    num_src_nodes: int, num_dst_nodes: int, num_edges: int
) -> bool:
  """Returns True when sorting edges by target improves GPU cache locality.

  Graph samples are not guaranteed to have edges sorted by target (merging and
  padding interleave edges from different samples), so the layer sorts them
  with `sort_edges_by_dst` rather than requiring a new sample representation.
  Sorted targets make the scatter-add of the message reduction write to
  contiguous memory and let XLA use `indices_are_sorted=True`. The sort is a
  device-side radix sort whose cost is only amortized on large GPU relations;
  on CPU and on small graphs it costs more than it saves.
  """
  return (
      jax.default_backend() == "gpu"
      and num_edges >= _SORT_MIN_EDGES
      and max(num_src_nodes, num_dst_nodes) >= _UINT16_MAX_NODES
  )


def sort_edges_by_dst(
    source_idxs: jax.Array,
    target_idxs: jax.Array,
    num_src_nodes: int,
    num_dst_nodes: int,
) -> tuple[jax.Array, jax.Array]:
  """Sorts (source_idxs, target_idxs) by target_idxs using narrow keys/values."""
  target_keys = (
      target_idxs.astype(jnp.uint16)
      if num_dst_nodes < _UINT16_MAX_NODES
      else target_idxs
  )
  source_values = (
      source_idxs.astype(jnp.uint16)
      if num_src_nodes < _UINT16_MAX_NODES
      else source_idxs
  )
  sorted_targets, sorted_sources = jax.lax.sort_key_val(
      target_keys, source_values
  )
  return sorted_sources.astype(jnp.int32), sorted_targets.astype(jnp.int32)


@functools.lru_cache(maxsize=None)
def _make_fused_activation_reduce(
    activation_fn: Callable[[jax.Array], jax.Array],
    indices_are_sorted: bool,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
  """Creates a fused "activate then reduce by target" op with a custom VJP.

  The returned function `f(src_projection, dst_projection, source_idxs,
  target_idxs)` is equivalent to:

    jax.ops.segment_sum(
        activation_fn(src_projection[source_idxs] +
        dst_projection[target_idxs]),
        target_idxs,
        num_segments=dst_projection.shape[0],
    )

  but its backward pass recomputes the per-edge pre-activation instead of
  storing it, which avoids materializing a `[num_edges, dims]` residual.
  """

  @jax.custom_vjp
  def fused_activation_reduce(
      src_projection: jax.Array,
      dst_projection: jax.Array,
      source_idxs: jax.Array,
      target_idxs: jax.Array,
  ) -> jax.Array:
    return jax.ops.segment_sum(
        activation_fn(
            src_projection[source_idxs] + dst_projection[target_idxs]
        ),
        target_idxs,
        dst_projection.shape[0],
        indices_are_sorted=indices_are_sorted,
    )

  def _forward(
      src_projection: jax.Array,
      dst_projection: jax.Array,
      source_idxs: jax.Array,
      target_idxs: jax.Array,
  ):
    out = fused_activation_reduce(
        src_projection, dst_projection, source_idxs, target_idxs
    )
    return out, (src_projection, dst_projection, source_idxs, target_idxs)

  def _backward(residuals, grad_output: jax.Array):
    src_projection, dst_projection, source_idxs, target_idxs = residuals
    pre_activation = src_projection[source_idxs] + dst_projection[target_idxs]
    _, vjp_fn = jax.vjp(activation_fn, pre_activation)
    grad_pre_activation = vjp_fn(grad_output[target_idxs])[0]
    grad_src_projection = jax.ops.segment_sum(
        grad_pre_activation,
        source_idxs,
        src_projection.shape[0],
        indices_are_sorted=False,
    )
    grad_dst_projection = jax.ops.segment_sum(
        grad_pre_activation,
        target_idxs,
        dst_projection.shape[0],
        indices_are_sorted=indices_are_sorted,
    )
    return grad_src_projection, grad_dst_projection, None, None

  fused_activation_reduce.defvjp(_forward, _backward)
  return fused_activation_reduce


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
    force_basic_implementation: If True, always compute messages with the
      generic per-edge implementation (no node-level projection, no edge
      sorting). Mostly useful in tests to validate the optimized paths.
  """

  plan: list[tuple[str, bool]] | None = None
  embedding_feature: str = "embedding"
  dims: int = 128
  dropout_rate: float = 0.1
  message_pooling: str = "sum"
  force_basic_implementation: bool = False

  message: common.GenericLayer | None = layer_registry.field(default=None)
  update: common.GenericLayer | None = layer_registry.field(default=None)
  post: common.GenericLayer | None = layer_registry.field(default=None)

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
      self, schema: schema_lib.GraphSchema, name: str | None = None
  ) -> "HeterogeneousGraphConvolution":
    return HeterogeneousGraphConvolution(config=self, schema=schema, name=name)

  def architecture(self) -> str:
    assert self.message is not None
    assert self.update is not None
    assert self.post is not None
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

    linear_activation_linear = None
    if not config.force_basic_implementation and isinstance(
        config.message, standard.GenericBlockConfig
    ):
      linear_activation_linear = config.message.as_linear_activation_linear()

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

        src_values = res_node_features[message_src_nodeset]
        num_src_nodes = src_values.shape[0]
        num_edges = source_idxs.shape[0]
        sort_edges = (
            not config.force_basic_implementation
            and should_sort_edges(num_src_nodes, num_dst_nodes, num_edges)
        )
        if sort_edges:
          source_idxs, target_idxs = sort_edges_by_dst(
              source_idxs, target_idxs, num_src_nodes, num_dst_nodes
          )
        relation_name = f"{edgeset_name}_{'rev' if reverse else 'fwd'}"

        if linear_activation_linear is not None:
          # The message function starts with a LAL (linear, activation, linear).
          # In this case, run the compute before the scatter.

          activation_fn = linear_activation_linear.activation
          dense_src = nn.Dense(
              linear_activation_linear.hidden_dims,
              use_bias=False,
              name=f"msg_{relation_name}_src",
          )
          dense_dst = nn.Dense(
              linear_activation_linear.hidden_dims,
              use_bias=True,
              name=f"msg_{relation_name}_dst",
          )
          dense_out = nn.Dense(
              linear_activation_linear.output_dims,
              use_bias=False,
              name=f"msg_{relation_name}_out",
          )

          # Project on nodes (N_src + 2 N_dst Dense rows) when cheaper than
          # projecting on edges (3 E Dense rows).
          if num_src_nodes + 2 * num_dst_nodes <= 3 * num_edges:
            # Node-level projections, then fused gather + activation +
            # segment_sum on the edges, then output projection on the nodes.
            src_projection = dense_src(src_values)  # [N_src, hidden_dims]
            dst_projection = dense_dst(dst_values)  # [N_dst, hidden_dims]
            aggregated_activations = _make_fused_activation_reduce(
                activation_fn, sort_edges
            )(
                src_projection, dst_projection, source_idxs, target_idxs
            )  # [N_dst, hidden_dims]
            neighbor_aggregate = dense_out(aggregated_activations)
          else:
            # Edge-level activation; each Dense is applied on the smallest of
            # (nodes, edges).
            src_edge_projection = (
                dense_src(src_values)[source_idxs]
                if num_src_nodes < num_edges
                else dense_src(src_values[source_idxs])
            )  # [E, hidden_dims]
            dst_edge_projection = (
                dense_dst(dst_values)[target_idxs]
                if num_dst_nodes < num_edges
                else dense_dst(dst_values[target_idxs])
            )  # [E, hidden_dims]
            edge_activations = activation_fn(
                src_edge_projection + dst_edge_projection
            )
            if num_dst_nodes <= num_edges:
              # Group by target nodeset, then output projection on the nodes.
              aggregated_activations = jax.ops.segment_sum(
                  edge_activations,
                  target_idxs,
                  num_dst_nodes,
                  indices_are_sorted=sort_edges,
              )
              neighbor_aggregate = dense_out(aggregated_activations)
            else:
              # Output projection on the edges, then group by target nodeset.
              messages = dense_out(edge_activations)
              neighbor_aggregate = jax.ops.segment_sum(
                  messages,
                  target_idxs,
                  num_dst_nodes,
                  indices_are_sorted=sort_edges,
              )
        else:
          # Basic / expensive fallback.

          # Gather the edge values.
          src_edge_values = src_values[source_idxs]
          dst_edge_values = dst_values[target_idxs]
          edge_values = jnp.concatenate(
              [src_edge_values, dst_edge_values], axis=-1
          )  # [E, 2 * dims]

          # Weights + activation on the edges.
          message_fn = config.message.make(name=f"msg_{relation_name}")
          messages = message_fn(edge_values, training=training)

          # Group by target nodeset.
          neighbor_aggregate = jax.ops.segment_sum(
              messages,
              target_idxs,
              num_dst_nodes,
              indices_are_sorted=sort_edges,
          )

        if self.config.message_pooling == "mean":
          degrees = jax.ops.segment_sum(
              jnp.ones((num_edges, 1), dtype=neighbor_aggregate.dtype),
              target_idxs,
              num_dst_nodes,
              indices_are_sorted=sort_edges,
          )
          degrees = jnp.maximum(degrees, 1.0)
          neighbor_aggregate = neighbor_aggregate / degrees
        elif self.config.message_pooling == "sum":
          pass
        else:
          raise ValueError("Unsupported message_pooling:")
        neighbor_aggregates.append(neighbor_aggregate)

      # Join messages + first residual
      combined_aggregates = neighbor_aggregates[0]
      for aggregate in neighbor_aggregates[1:]:
        combined_aggregates = combined_aggregates + aggregate
      if self.config.message_pooling == "mean":
        combined_aggregates = combined_aggregates * (
            1.0 / len(neighbor_aggregates)
        )
      elif self.config.message_pooling != "sum":
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
