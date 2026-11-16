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

"""Pure JAX operations for message passing on graphs.

This module contains the graph operations used by the message passing layers
(e.g. `hetero_gnn.HeterogeneousGraphConvolution`). The functions are pure JAX
functions: They do not create parameters and do not depend on layer configs.
"""

from collections.abc import Callable
import dataclasses
import functools
import math
from dgf.src.learning.jax import common
import jax
import jax.numpy as jnp
import jaxtyping as jt

# Minimum number of edges for sorting by target to pay off on GPU.
SORT_MIN_EDGES = 50000
# Node indices below this bound fit in uint16 (narrow sort keys / values).
UINT16_MAX_NODES = 2**16
# Added to the softmax denominator to avoid divisions by zero.
SOFTMAX_EPSILON = 1e-9


@dataclasses.dataclass(frozen=True)
class Relation:
  """One message passing direction: Messages flow from `src` to `dst` nodes.

  A relation is an edge set traversed in a given direction. For example, the
  edge set "cites" (paper -> paper) defines the relations "cites_fwd" and
  "cites_rev".

  Attributes:
    name: Name of the relation, e.g. "cites_fwd". Used as prefix of the
      parameter names.
    src_nodeset: Name of the node set sending the messages.
    dst_nodeset: Name of the node set receiving the messages.
    src_values: Embeddings of the source nodes.
    dst_values: Embeddings of the target nodes.
    source_idxs: Source node of each edge.
    target_idxs: Target node of each edge.
    num_src_nodes: Number of source nodes.
    num_dst_nodes: Number of target nodes.
    num_edges: Number of edges.
    indices_are_sorted: Whether the edges are sorted by target node.
  """

  name: str
  src_nodeset: str
  dst_nodeset: str
  src_values: jt.Float[jt.Array, "num_src_nodes dims"]
  dst_values: jt.Float[jt.Array, "num_dst_nodes dims"]
  source_idxs: jt.Int[jt.Array, " num_edges"]
  target_idxs: jt.Int[jt.Array, " num_edges"]
  num_src_nodes: int
  num_dst_nodes: int
  num_edges: int
  indices_are_sorted: bool


def should_sort_edges(
    num_src_nodes: int, num_dst_nodes: int, num_edges: int
) -> bool:
  """Returns True when sorting edges by target improves GPU cache locality.

  Graph samples are not guaranteed to have edges sorted by target (merging and
  padding interleave edges from different samples), so the layers sort them
  with `sort_edges_by_dst` rather than requiring a new sample representation.
  Sorted targets make the scatter-add of the message reduction write to
  contiguous memory and let XLA use `indices_are_sorted=True`. The sort is a
  device-side radix sort whose cost is only amortized on large GPU relations;
  on CPU and on small graphs it costs more than it saves.

  Usage example:

  ```python
  should_sort_edges(num_src_nodes=100_000, num_dst_nodes=80_000,
                    num_edges=500_000)
  # True on GPU, False on CPU.
  ```

  Args:
    num_src_nodes: Number of source nodes.
    num_dst_nodes: Number of target nodes.
    num_edges: Number of edges.

  Returns:
    Whether the edges should be sorted by target node.
  """
  return (
      jax.default_backend() == "gpu"
      and num_edges >= SORT_MIN_EDGES
      and max(num_src_nodes, num_dst_nodes) >= UINT16_MAX_NODES
  )


def sort_edges_by_dst(
    source_idxs: jt.Int[jt.Array, " num_edges"],
    target_idxs: jt.Int[jt.Array, " num_edges"],
    num_src_nodes: int,
    num_dst_nodes: int,
) -> tuple[jt.Int[jt.Array, " num_edges"], jt.Int[jt.Array, " num_edges"]]:
  """Sorts the edges by target node, using narrow keys / values when possible.

  Usage example:

  ```python
  source_idxs, target_idxs = sort_edges_by_dst(
      jnp.array([0, 1, 2]), jnp.array([2, 0, 1]), num_src_nodes=3,
      num_dst_nodes=3)
  # source_idxs = [1, 2, 0], target_idxs = [0, 1, 2]
  ```

  Args:
    source_idxs: Source node of each edge.
    target_idxs: Target node of each edge.
    num_src_nodes: Number of source nodes.
    num_dst_nodes: Number of target nodes.

  Returns:
    The (source_idxs, target_idxs) of the edges sorted by target node.
  """
  target_keys = (
      target_idxs.astype(jnp.uint16)
      if num_dst_nodes < UINT16_MAX_NODES
      else target_idxs
  )
  source_values = (
      source_idxs.astype(jnp.uint16)
      if num_src_nodes < UINT16_MAX_NODES
      else source_idxs
  )
  sorted_targets, sorted_sources = jax.lax.sort_key_val(
      target_keys, source_values
  )
  return sorted_sources.astype(jnp.int32), sorted_targets.astype(jnp.int32)


def node_degrees(
    node_idxs: jt.Int[jt.Array, " num_edges"],
    num_nodes: int,
    dtype: jnp.dtype,
    indices_are_sorted: bool = False,
) -> jt.Num[jt.Array, "num_nodes 1"]:
  """Gets the number of edges incident to each node.

  Usage example:

  ```python
  adjacency = jnp.array([[0, 0, 2],   # sources
                         [0, 1, 2]])  # targets
  node_degrees(adjacency[0], num_nodes=3, dtype=jnp.int32)
  # [[2], [0], [1]]: node 0 has two outgoing edges, node 1 none, node 2 one.
  ```

  Args:
    node_idxs: Node index of each edge, shape [num_edges]. `node_idxs[i] == j`
      means edge `i` is incident to node `j`. Pass the target side of the
      adjacency for in-degrees, the source side for out-degrees.
    num_nodes: Number of nodes.
    dtype: Dtype of the result.
    indices_are_sorted: Whether `node_idxs` is sorted.

  Returns:
    The node degrees, shape [num_nodes, 1].
  """
  return jax.ops.segment_sum(
      jnp.ones((node_idxs.shape[0], 1), dtype=dtype),
      node_idxs,
      num_nodes,
      indices_are_sorted=indices_are_sorted,
  )


def set_statistics_pool(
    source_values: jt.Float[jt.Array, "num_src_nodes dims"],
    source_idxs: jt.Int[jt.Array, " num_edges"],
    target_idxs: jt.Int[jt.Array, " num_edges"],
    num_dst_nodes: int,
    indices_are_sorted: bool = False,
    epsilon: float = 1e-6,
) -> jt.Float[jt.Array, "num_dst_nodes out_dims"]:
  """Summarizes the source nodes of each target node as [mean, std, log-count].

  For each target node, the values of the source nodes connected to it are
  reduced to their mean, their (population) standard deviation and the log of
  one plus their number. Unlike a mean or a sum, this tells apart a target node
  with few sources from one with many, and a homogeneous set of sources from a
  spread one. Target nodes without incoming edges get a zero mean,
  `sqrt(epsilon)` as standard deviation and a zero log-count.

  A source node connected to the same target node by several edges is counted
  once per edge.

  No padding mask is needed on graphs padded by `GraphMerger`: padding edges
  only connect padding nodes, so they do not change the output of the real
  target nodes. The output of the padding target nodes is not meaningful.

  The statistics are computed in `common.DEFAULT_POINTWISE_NORM_PRECISION` and
  returned in the dtype of `source_values`.

  Usage example:

  ```python
  # 3 authors with 2 dims, 2 papers. Paper 0 has authors 0 and 1, paper 1 has
  # author 2.
  author_values = jnp.array([[1.0, 2.0], [3.0, 2.0], [5.0, 6.0]])
  paper_values = set_statistics_pool(
      author_values,
      source_idxs=jnp.array([0, 1, 2]),
      target_idxs=jnp.array([0, 0, 1]),
      num_dst_nodes=2,
  )
  # With sqrt(epsilon) = 0.001 as the std of identical values:
  # paper_values[0] ~= [2, 2, 1, 0.001, log(3)]: 2 authors, std [1, 0].
  # paper_values[1] ~= [5, 6, 0.001, 0.001, log(2)]: 1 author, std [0, 0].
  ```

  Args:
    source_values: Value of each source node.
    source_idxs: Source node of each edge.
    target_idxs: Target node of each edge.
    num_dst_nodes: Number of target nodes.
    indices_are_sorted: Whether `target_idxs` is sorted. The edges of a graph
      padded by `GraphMerger` are generally not.
    epsilon: Added to the variance before the square root, so that the gradient
      is finite when all the sources of a node have the same value.

  Returns:
    The concatenated [mean, std, log(1 + count)] of each target node, of shape
    [num_dst_nodes, 2 * dims + 1].
  """
  values = source_values[source_idxs].astype(
      common.DEFAULT_POINTWISE_NORM_PRECISION
  )

  def sum_by_target(
      edge_values: jt.Float[jt.Array, "num_edges dims"],
  ) -> jt.Float[jt.Array, "num_dst_nodes dims"]:
    return jax.ops.segment_sum(
        edge_values,
        target_idxs,
        num_dst_nodes,
        indices_are_sorted=indices_are_sorted,
    )

  counts = node_degrees(
      target_idxs, num_dst_nodes, values.dtype, indices_are_sorted
  )
  # Nodes without sources have zero sums: Any non-zero divisor gives zero.
  safe_counts = jnp.maximum(counts, 1.0)
  mean = sum_by_target(values) / safe_counts
  variance = sum_by_target(jnp.square(values - mean[target_idxs])) / safe_counts
  std = jnp.sqrt(variance + epsilon)
  return jnp.concatenate([mean, std, jnp.log1p(counts)], axis=-1).astype(
      source_values.dtype
  )


@functools.lru_cache(maxsize=None)
def make_fused_activation_reduce(
    activation_fn: Callable[[jax.Array], jax.Array],
    indices_are_sorted: bool,
) -> Callable[
    [
        jt.Float[jt.Array, "num_src_nodes dims"],
        jt.Float[jt.Array, "num_dst_nodes dims"],
        jt.Int[jt.Array, " num_edges"],
        jt.Int[jt.Array, " num_edges"],
    ],
    jt.Float[jt.Array, "num_dst_nodes dims"],
]:
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

  Usage example:

  ```python
  fused_fn = make_fused_activation_reduce(jax.nn.relu, indices_are_sorted=False)
  aggregate = fused_fn(src_projection, dst_projection, source_idxs,
                       target_idxs)
  ```

  Args:
    activation_fn: Element-wise activation function.
    indices_are_sorted: Whether the target indices are sorted.

  Returns:
    The fused function.
  """

  @jax.custom_vjp
  def fused_activation_reduce(
      src_projection: jt.Float[jt.Array, "num_src_nodes dims"],
      dst_projection: jt.Float[jt.Array, "num_dst_nodes dims"],
      source_idxs: jt.Int[jt.Array, " num_edges"],
      target_idxs: jt.Int[jt.Array, " num_edges"],
  ) -> jt.Float[jt.Array, "num_dst_nodes dims"]:
    return jax.ops.segment_sum(
        activation_fn(
            src_projection[source_idxs] + dst_projection[target_idxs]
        ),
        target_idxs,
        dst_projection.shape[0],
        indices_are_sorted=indices_are_sorted,
    )

  def forward(
      src_projection: jt.Float[jt.Array, "num_src_nodes dims"],
      dst_projection: jt.Float[jt.Array, "num_dst_nodes dims"],
      source_idxs: jt.Int[jt.Array, " num_edges"],
      target_idxs: jt.Int[jt.Array, " num_edges"],
  ) -> tuple[
      jt.Float[jt.Array, "num_dst_nodes dims"],
      tuple[jax.Array, jax.Array, jax.Array, jax.Array],
  ]:
    out = fused_activation_reduce(
        src_projection, dst_projection, source_idxs, target_idxs
    )
    return out, (src_projection, dst_projection, source_idxs, target_idxs)

  def backward(
      residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
      grad_output: jt.Float[jt.Array, "num_dst_nodes dims"],
  ) -> tuple[jax.Array, jax.Array, None, None]:
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

  fused_activation_reduce.defvjp(forward, backward)
  return fused_activation_reduce


def segment_softmax_aggregate(
    logits: jt.Float[jt.Array, "num_edges num_heads"],
    messages: jt.Float[jt.Array, "num_edges num_heads head_dims"],
    target_idxs: jt.Int[jt.Array, " num_edges"],
    num_dst_nodes: int,
    indices_are_sorted: bool,
    weight_dropout: (
        Callable[
            [jt.Float[jt.Array, "num_edges num_heads"]],
            jt.Float[jt.Array, "num_edges num_heads"],
        ]
        | None
    ) = None,
) -> jt.Float[jt.Array, "num_dst_nodes num_heads head_dims"]:
  """Sums the messages received by each node, weighted by softmax(logits).

  For each target node `i` and head `h`:

    output[i, h] = sum_{e: target(e) = i} softmax_e(logits[:, h]) messages[e, h]

  where the softmax is computed over the edges incoming to `i`. Nodes without
  incoming edges get zero. The softmax is computed in
  `common.DEFAULT_SOFTMAX_PRECISION`.

  Usage example:

  ```python
  # 3 edges, 2 target nodes, 1 head of 4 dims.
  aggregate = segment_softmax_aggregate(
      logits=jnp.zeros((3, 1)),
      messages=jnp.ones((3, 1, 4)),
      target_idxs=jnp.array([0, 0, 1]),
      num_dst_nodes=2,
      indices_are_sorted=True,
  )
  # aggregate[0] == aggregate[1] == 1: the weights of each node sum to 1.
  ```

  Args:
    logits: Attention logit of each edge and head.
    messages: Message of each edge and head.
    target_idxs: Target node of each edge.
    num_dst_nodes: Number of target nodes.
    indices_are_sorted: Whether `target_idxs` is sorted.
    weight_dropout: Optional function applied to the normalized attention
      weights (e.g. dropout during training). If None, the normalization is
      applied once per node instead of once per edge.

  Returns:
    The aggregated messages of each target node and head.
  """
  logits = logits.astype(common.DEFAULT_SOFTMAX_PRECISION)
  # Max logit of each target node, for numerical stability.
  local_max = jax.ops.segment_max(
      jax.lax.stop_gradient(logits),
      target_idxs,
      num_dst_nodes,
      indices_are_sorted=indices_are_sorted,
  )  # [N_dst, H]
  exp_logits = jnp.exp(logits - local_max[target_idxs])  # [E, H]
  denominator = jax.ops.segment_sum(
      exp_logits,
      target_idxs,
      num_dst_nodes,
      indices_are_sorted=indices_are_sorted,
  )  # [N_dst, H]

  if weight_dropout is None:
    # Normalize once per node rather than once per edge.
    numerator = jax.ops.segment_sum(
        exp_logits[..., None] * messages,
        target_idxs,
        num_dst_nodes,
        indices_are_sorted=indices_are_sorted,
    )  # [N_dst, H, head_dims]
    aggregate = numerator / (denominator[..., None] + SOFTMAX_EPSILON)
  else:
    weights = weight_dropout(
        exp_logits / (denominator[target_idxs] + SOFTMAX_EPSILON)
    )  # [E, H]
    aggregate = jax.ops.segment_sum(
        weights[..., None] * messages,
        target_idxs,
        num_dst_nodes,
        indices_are_sorted=indices_are_sorted,
    )  # [N_dst, H, head_dims]
  return aggregate.astype(messages.dtype)


def attention_readout(
    queries: jt.Float[jt.Array, "num_dst_nodes num_heads head_dims"],
    keys: jt.Float[jt.Array, "num_src_nodes num_heads head_dims"],
    values: jt.Float[jt.Array, "num_src_nodes num_heads head_dims"],
    source_idxs: jt.Int[jt.Array, " num_edges"],
    target_idxs: jt.Int[jt.Array, " num_edges"],
    indices_are_sorted: bool = False,
) -> jt.Float[jt.Array, "num_dst_nodes dims"]:
  """Reads the source nodes of each target node with dot-product attention.

  For each target node `i` and head `h`:

    output[i, h] = sum_{e: target(e) = i} weight[e, h] values[source(e), h]

  where `weight[:, h]` is the softmax, over the edges incoming to `i`, of the
  scaled dot-product `queries[i, h] . keys[source(e), h] / sqrt(head_dims)`.
  Target nodes without incoming edges get zero.

  This is typically used to summarize a large node set into a few context
  nodes, each context node being connected to the nodes it summarizes. The
  queries, keys and values are computed by the caller, e.g. with dense layers.

  No padding mask is needed on graphs padded by `GraphMerger`: padding edges
  only connect padding nodes, so they do not change the output of the real
  target nodes. The output of the padding target nodes is not meaningful.

  The softmax is computed in `common.DEFAULT_SOFTMAX_PRECISION` and the output
  is returned in the dtype of `values`.

  Usage example:

  ```python
  # Summarize 3 "paper" nodes into 1 "venue" node, with 2 heads of 4 dims.
  venue_values = attention_readout(
      queries=jnp.ones((1, 2, 4)),
      keys=jnp.ones((3, 2, 4)),
      values=jnp.ones((3, 2, 4)),
      source_idxs=jnp.array([0, 1, 2]),
      target_idxs=jnp.array([0, 0, 0]),
  )
  # venue_values.shape == (1, 8) and venue_values == 1: the weights sum to 1.
  ```

  Args:
    queries: Query of each target node and head.
    keys: Key of each source node and head.
    values: Value of each source node and head.
    source_idxs: Source node of each edge.
    target_idxs: Target node of each edge.
    indices_are_sorted: Whether `target_idxs` is sorted. The edges of a graph
      padded by `GraphMerger` are generally not.

  Returns:
    The concatenated heads of each target node, of shape
    [num_dst_nodes, num_heads * head_dims].
  """
  num_dst_nodes, num_heads, head_dims = queries.shape
  logits = jnp.sum(
      queries[target_idxs] * keys[source_idxs], axis=-1
  ) / math.sqrt(head_dims)
  aggregate = segment_softmax_aggregate(
      logits=logits,
      messages=values[source_idxs],
      target_idxs=target_idxs,
      num_dst_nodes=num_dst_nodes,
      indices_are_sorted=indices_are_sorted,
  )
  return aggregate.reshape(num_dst_nodes, num_heads * head_dims)


def project_on_edges(
    dense: Callable[
        [jt.Float[jt.Array, "num_rows in_dims"]],
        jt.Float[jt.Array, "num_rows out_dims"],
    ],
    node_values: jt.Float[jt.Array, "num_nodes in_dims"],
    node_idxs: jt.Int[jt.Array, " num_edges"],
) -> jt.Float[jt.Array, "num_edges out_dims"]:
  """Computes `dense(node_values)[node_idxs]` on the smallest of nodes / edges.

  `dense(node_values)[node_idxs]` and `dense(node_values[node_idxs])` are equal
  for any row-wise function `dense` (e.g. a Dense layer). The first applies
  `dense` on the nodes, the second on the edges.

  Usage example:

  ```python
  # Project the source node embeddings on each edge.
  src_projection = project_on_edges(nn.Dense(64), src_values, source_idxs)
  ```

  Args:
    dense: Row-wise function, e.g. a Dense layer.
    node_values: Value of each node.
    node_idxs: Node of each edge.

  Returns:
    The projected value of each edge.
  """
  if node_values.shape[0] < node_idxs.shape[0]:
    return dense(node_values)[node_idxs]
  return dense(node_values[node_idxs])
