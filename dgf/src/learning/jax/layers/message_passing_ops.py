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
