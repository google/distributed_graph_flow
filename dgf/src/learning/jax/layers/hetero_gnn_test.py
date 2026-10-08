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

"""Tests for common layers."""

import logging
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import standard
import jax
import jax.numpy as jnp
import numpy as np


class HeteroGNNTest(parameterized.TestCase):

  def test_sort_plan(self):
    schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1"),
            "e2": schema_lib.EdgeSchema(source="n1", target="n2"),
        },
    )
    plan = [
        ("e1", False),  # n1 -> n1
        ("e1", True),  # n1 <- n1
        ("e2", False),  # n1 -> n2
    ]
    sorted_plan = hetero_gnn.sort_plan(plan, schema)
    expected_sorted_plan = {
        "n1": [("e1", "n1", False), ("e1", "n1", True)],
        "n2": [("e2", "n1", False)],
    }
    self.assertEqual(sorted_plan, expected_sorted_plan)

  def test_message_passing(self):
    schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1"),
            "e2": schema_lib.EdgeSchema(source="n1", target="n2"),
        },
    )
    gnn = hetero_gnn.HeterogeneousGraphConvolutionConfig(
        plan=[
            ("e1", False),  # n1 -> n1
            ("e1", True),  # n1 <- n1
            ("e2", False),  # n1 -> n2
        ],
    ).make(schema)
    input_graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.array([[1.0], [2.0]])},
                num_nodes=2,
            ),
            "n2": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.array([[3.0], [4.0]])},
                num_nodes=2,
            ),
        },
        edge_sets={
            "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0], [1]]),
            ),
            "e2": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0], [0]]),
            ),
        },
    )
    variables = gnn.init(jax.random.PRNGKey(42), input_graph, schema)
    logging.info("variables:\n%s", variables)
    output = gnn.apply(
        variables,
        input_graph,
        training=True,
        rngs={"dropout": jax.random.PRNGKey(42)},
    )

    self.assertEqual(
        output.node_sets["n1"].features["embedding"].shape,
        (2, 128),
    )
    self.assertEqual(
        output.node_sets["n2"].features["embedding"].shape,
        (2, 128),
    )

  @parameterized.parameters(False, True)
  def test_fused_activation_reduce_grad_matches_reference(
      self, indices_are_sorted: bool
  ):
    rng = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    src_projection = jax.random.normal(k1, (12, 16))
    dst_projection = jax.random.normal(k2, (8, 16))
    src_idx = jax.random.randint(k3, (40,), 0, 12, dtype=jnp.int32)
    dst_idx = jax.random.randint(k4, (40,), 0, 8, dtype=jnp.int32)
    if indices_are_sorted:
      src_idx, dst_idx = hetero_gnn.sort_edges_by_dst(src_idx, dst_idx, 12, 8)

    fused_fn = hetero_gnn._make_fused_activation_reduce(
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

  @parameterized.parameters("sum", "mean")
  def test_cross_regime_and_edge_permutation_equivalence(
      self, message_pooling: str
  ):
    schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n2"),
        },
    )
    gnn = hetero_gnn.HeterogeneousGraphConvolutionConfig(
        dims=16,
        message_pooling=message_pooling,
    ).make(schema)

    # Initialize on a sparse graph (Regime 2: N_src + 2*N_dst > 3*E)
    sparse_graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.ones((6, 16))},
                num_nodes=6,
            ),
            "n2": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.ones((5, 16))},
                num_nodes=5,
            ),
        },
        edge_sets={
            "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
            ),
        },
    )
    variables = gnn.init(jax.random.PRNGKey(7), sparse_graph, training=False)

    # Apply the exact same variables to a dense graph (Regime 1: N_src + 2*N_dst <= 3*E)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(9), 4)
    n1_feat = jax.random.normal(k1, (6, 16))
    n2_feat = jax.random.normal(k2, (5, 16))
    src_idx = jax.random.randint(k3, (30,), 0, 6, dtype=jnp.int32)
    dst_idx = jax.random.randint(k4, (30,), 0, 5, dtype=jnp.int32)
    perm = jax.random.permutation(jax.random.PRNGKey(11), 30)

    dense_graph_a = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": n1_feat}, num_nodes=6
            ),
            "n2": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": n2_feat}, num_nodes=5
            ),
        },
        edge_sets={
            "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.stack([src_idx, dst_idx], axis=0)
            ),
        },
    )
    dense_graph_b = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets=dense_graph_a.node_sets,
        edge_sets={
            "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.stack([src_idx[perm], dst_idx[perm]], axis=0)
            ),
        },
    )

    out_a = gnn.apply(variables, dense_graph_a, training=False)
    out_b = gnn.apply(variables, dense_graph_b, training=False)
    np.testing.assert_allclose(
        out_a.node_sets["n1"].features["embedding"],
        out_b.node_sets["n1"].features["embedding"],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        out_a.node_sets["n2"].features["embedding"],
        out_b.node_sets["n2"].features["embedding"],
        rtol=1e-5,
        atol=1e-5,
    )

  @parameterized.product(
      message_pooling=["sum", "mean"],
      num_edges=[2, 30],  # Regime 2 (few edges) and regime 1 (many edges).
  )
  def test_optimized_matches_basic_implementation(
      self, message_pooling: str, num_edges: int
  ):
    schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={"e1": schema_lib.EdgeSchema(source="n1", target="n2")},
    )
    kwargs = dict(dims=16, message_pooling=message_pooling)
    fast = hetero_gnn.HeterogeneousGraphConvolutionConfig(**kwargs).make(schema)
    basic = hetero_gnn.HeterogeneousGraphConvolutionConfig(
        force_basic_implementation=True, **kwargs
    ).make(schema)

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(3), 4)
    graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jax.random.normal(k1, (6, 16))},
                num_nodes=6,
            ),
            "n2": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jax.random.normal(k2, (5, 16))},
                num_nodes=5,
            ),
        },
        edge_sets={
            "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.stack([
                    jax.random.randint(k3, (num_edges,), 0, 6),
                    jax.random.randint(k4, (num_edges,), 0, 5),
                ])
            ),
        },
    )
    fast_vars = fast.init(jax.random.PRNGKey(0), graph, training=False)

    # Map the (src, dst, out) dense layers of the optimized implementation onto
    # the single "LAL" block applied to concat([src, dst]) by the basic one.
    params = dict(fast_vars["params"])
    for relation_name in ["e1_fwd", "e1_rev"]:
      src = params.pop(f"msg_{relation_name}_src")
      dst = params.pop(f"msg_{relation_name}_dst")
      out = params.pop(f"msg_{relation_name}_out")
      params[f"msg_{relation_name}"] = {
          "dense_0": {
              "kernel": jnp.concatenate([src["kernel"], dst["kernel"]], axis=0),
              "bias": dst["bias"],
          },
          "dense_2": {
              "kernel": out["kernel"],
              "bias": jnp.zeros(out["kernel"].shape[1]),
          },
      }
    basic_vars = {**fast_vars, "params": params}

    def loss(model, variables, graph):
      out = model.apply(variables, graph, training=False)
      return (
          sum(
              jnp.sum(ns.features["embedding"] ** 2)
              for ns in out.node_sets.values()
          ),
          out,
      )

    (loss_fast, out_fast), grad_fast = jax.value_and_grad(
        loss, argnums=2, has_aux=True, allow_int=True
    )(fast, fast_vars, graph)
    (loss_basic, out_basic), grad_basic = jax.value_and_grad(
        loss, argnums=2, has_aux=True, allow_int=True
    )(basic, basic_vars, graph)

    np.testing.assert_allclose(loss_fast, loss_basic, rtol=1e-4)
    for ns in ["n1", "n2"]:
      np.testing.assert_allclose(
          out_fast.node_sets[ns].features["embedding"],
          out_basic.node_sets[ns].features["embedding"],
          rtol=1e-4,
          atol=1e-4,
      )
      np.testing.assert_allclose(
          grad_fast.node_sets[ns].features["embedding"],
          grad_basic.node_sets[ns].features["embedding"],
          rtol=1e-4,
          atol=1e-4,
      )

  def test_non_lal_message_fallback(self):
    schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={"e1": schema_lib.EdgeSchema(source="n1", target="n1")},
    )
    gnn = hetero_gnn.HeterogeneousGraphConvolutionConfig(
        dims=16,
        message=standard.GenericBlockConfig("LA", dims=16),
    ).make(schema)
    graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.ones((3, 16))}, num_nodes=3
            )
        },
        edge_sets={
            "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
            )
        },
    )
    variables = gnn.init(jax.random.PRNGKey(0), graph, training=False)
    out = gnn.apply(variables, graph, training=False)
    self.assertEqual(out.node_sets["n1"].features["embedding"].shape, (3, 16))

  def test_architecture(self):
    config = hetero_gnn.HeterogeneousGraphConvolutionConfig()
    infra_str = config.architecture()
    logging.info("architecture:\n%s", infra_str)
    self.assertIn("X = ...", infra_str)
    self.assertIn("MPNN:", infra_str)
    self.assertIn("  Message:", infra_str)
    self.assertIn("    Dense(128)", infra_str)
    self.assertIn("  Update:", infra_str)
    self.assertIn("Residual(X)", infra_str)
    self.assertIn("# Post MPNN", infra_str)
    self.assertIn("Norm(rms_norm)", infra_str)


if __name__ == "__main__":
  absltest.main()
