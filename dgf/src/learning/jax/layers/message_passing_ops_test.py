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

"""Tests for the message passing operations."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.jax.layers import message_passing_ops
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def random_edges(
    num_edges: int, num_src_nodes: int, num_dst_nodes: int, seed: int = 0
) -> tuple[jax.Array, jax.Array]:
  """Returns random (source_idxs, target_idxs)."""
  k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
  return (
      jax.random.randint(k1, (num_edges,), 0, num_src_nodes, dtype=jnp.int32),
      jax.random.randint(k2, (num_edges,), 0, num_dst_nodes, dtype=jnp.int32),
  )


class MessagePassingOpsTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_node_degrees(self, indices_are_sorted: bool):
    # Node 0 has 2 edges, node 1 none, node 2 one and node 3 none.
    node_idxs = jnp.array([0, 0, 2], dtype=jnp.int32)
    degrees = message_passing_ops.node_degrees(
        node_idxs,
        num_nodes=4,
        dtype=jnp.float32,
        indices_are_sorted=indices_are_sorted,
    )
    self.assertEqual(degrees.dtype, jnp.float32)
    np.testing.assert_array_equal(degrees, [[2.0], [0.0], [1.0], [0.0]])

  def test_sort_edges_by_dst(self):
    source_idxs, target_idxs = random_edges(50, 7, 5)
    sorted_sources, sorted_targets = message_passing_ops.sort_edges_by_dst(
        source_idxs, target_idxs, num_src_nodes=7, num_dst_nodes=5
    )
    self.assertEqual(sorted_sources.dtype, jnp.int32)
    self.assertEqual(sorted_targets.dtype, jnp.int32)
    np.testing.assert_array_equal(sorted_targets, np.sort(target_idxs))
    # Same multiset of edges.
    self.assertCountEqual(
        zip(source_idxs.tolist(), target_idxs.tolist()),
        zip(sorted_sources.tolist(), sorted_targets.tolist()),
    )

  def test_should_sort_edges_small_graph(self):
    self.assertFalse(
        message_passing_ops.should_sort_edges(
            num_src_nodes=10, num_dst_nodes=10, num_edges=20
        )
    )

  @parameterized.parameters(False, True)
  def test_fused_activation_reduce_grad_matches_reference(
      self, indices_are_sorted: bool
  ):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    src_projection = jax.random.normal(k1, (12, 16))
    dst_projection = jax.random.normal(k2, (8, 16))
    src_idx, dst_idx = random_edges(40, 12, 8)
    if indices_are_sorted:
      src_idx, dst_idx = message_passing_ops.sort_edges_by_dst(
          src_idx, dst_idx, 12, 8
      )

    fused_fn = message_passing_ops.make_fused_activation_reduce(
        jax.nn.silu, indices_are_sorted
    )

    def ref_loss(a, b):
      out = jax.ops.segment_sum(
          jax.nn.silu(a[src_idx] + b[dst_idx]),
          dst_idx,
          b.shape[0],
          indices_are_sorted=indices_are_sorted,
      )
      return jnp.sum(out**2)

    def fused_loss(a, b):
      out = fused_fn(a, b, src_idx, dst_idx)
      return jnp.sum(out**2)

    value_ref, (grad_src_ref, grad_dst_ref) = jax.value_and_grad(
        ref_loss, argnums=(0, 1)
    )(src_projection, dst_projection)
    value_opt, (grad_src_opt, grad_dst_opt) = jax.value_and_grad(
        fused_loss, argnums=(0, 1)
    )(src_projection, dst_projection)
    np.testing.assert_allclose(value_opt, value_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_src_opt, grad_src_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_dst_opt, grad_dst_ref, rtol=1e-5, atol=1e-5)

  @parameterized.parameters(False, True)
  def test_segment_softmax_aggregate_matches_reference(
      self, indices_are_sorted: bool
  ):
    num_edges, num_dst_nodes, num_heads, head_dims = 30, 6, 2, 3
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    logits = 5.0 * jax.random.normal(k1, (num_edges, num_heads))
    messages = jax.random.normal(k2, (num_edges, num_heads, head_dims))
    _, target_idxs = random_edges(num_edges, 1, num_dst_nodes - 1)
    if indices_are_sorted:
      order = jnp.argsort(target_idxs)
      logits, messages, target_idxs = (
          logits[order],
          messages[order],
          target_idxs[order],
      )

    aggregate = message_passing_ops.segment_softmax_aggregate(
        logits, messages, target_idxs, num_dst_nodes, indices_are_sorted
    )

    # Dense reference: explicit softmax over the incoming edges of each node.
    expected = np.zeros((num_dst_nodes, num_heads, head_dims))
    for node in range(num_dst_nodes):
      edges = np.flatnonzero(np.asarray(target_idxs) == node)
      if edges.size == 0:
        continue  # Node without incoming edges: zero.
      node_logits = np.asarray(logits)[edges]  # [k, H]
      weights = np.exp(node_logits - node_logits.max(axis=0))
      weights /= weights.sum(axis=0)
      expected[node] = np.einsum(
          "kh,khd->hd", weights, np.asarray(messages)[edges]
      )
    np.testing.assert_allclose(aggregate, expected, rtol=1e-5, atol=1e-6)
    # The last node has no incoming edges.
    np.testing.assert_array_equal(aggregate[-1], 0.0)

  def test_segment_softmax_aggregate_weights_sum_to_one(self):
    # With constant messages, the aggregate is the sum of the weights.
    logits = jax.random.normal(jax.random.PRNGKey(2), (20, 3))
    _, target_idxs = random_edges(20, 1, 4)
    aggregate = message_passing_ops.segment_softmax_aggregate(
        logits, jnp.ones((20, 3, 1)), target_idxs, 4, False
    )
    has_edges = np.isin(np.arange(4), np.asarray(target_idxs))
    np.testing.assert_allclose(aggregate[has_edges], 1.0, rtol=1e-6, atol=1e-6)

  def test_segment_softmax_aggregate_weight_dropout(self):
    logits = jax.random.normal(jax.random.PRNGKey(3), (20, 2))
    messages = jax.random.normal(jax.random.PRNGKey(4), (20, 2, 4))
    _, target_idxs = random_edges(20, 1, 5)
    without_dropout = message_passing_ops.segment_softmax_aggregate(
        logits, messages, target_idxs, 5, False
    )
    # An identity "dropout" uses the per-edge normalization path.
    with_identity = message_passing_ops.segment_softmax_aggregate(
        logits, messages, target_idxs, 5, False, weight_dropout=lambda w: w
    )
    np.testing.assert_allclose(
        with_identity, without_dropout, rtol=1e-5, atol=1e-6
    )
    # Dropping all the weights gives zero.
    with_zero = message_passing_ops.segment_softmax_aggregate(
        logits, messages, target_idxs, 5, False, weight_dropout=jnp.zeros_like
    )
    np.testing.assert_array_equal(with_zero, 0.0)

  def test_segment_softmax_aggregate_bfloat16(self):
    # Large logits would overflow without the max subtraction.
    logits = (100.0 * jax.random.normal(jax.random.PRNGKey(5), (20, 1))).astype(
        jnp.bfloat16
    )
    messages = jnp.ones((20, 1, 2), dtype=jnp.bfloat16)
    _, target_idxs = random_edges(20, 1, 4)
    aggregate = message_passing_ops.segment_softmax_aggregate(
        logits, messages, target_idxs, 4, False
    )
    self.assertEqual(aggregate.dtype, jnp.bfloat16)
    self.assertTrue(np.all(np.isfinite(np.asarray(aggregate, np.float32))))
    has_edges = np.isin(np.arange(4), np.asarray(target_idxs))
    np.testing.assert_allclose(
        np.asarray(aggregate, np.float32)[has_edges], 1.0, rtol=1e-2
    )

  @parameterized.parameters(3, 30)  # More / fewer nodes than edges.
  def test_project_on_edges(self, num_edges: int):
    node_values = jax.random.normal(jax.random.PRNGKey(6), (10, 4))
    node_idxs, _ = random_edges(num_edges, 10, 1)
    dense = nn.Dense(8)
    variables = dense.init(jax.random.PRNGKey(7), node_values)

    def apply_dense(x: jax.Array) -> jax.Array:
      return dense.apply(variables, x)

    np.testing.assert_allclose(
        message_passing_ops.project_on_edges(
            apply_dense, node_values, node_idxs
        ),
        apply_dense(node_values)[node_idxs],
        rtol=1e-6,
        atol=1e-6,
    )


if __name__ == "__main__":
  absltest.main()
