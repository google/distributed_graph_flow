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

"""Tests for heterogeneous graph attention network layer."""

import logging
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import hetero_graph_attention_network
from dgf.src.learning.jax.layers import standard
import jax
import jax.numpy as jnp
import numpy as np


class HeteroGraphAttentionNetworkTest(parameterized.TestCase):

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
    sorted_plan = hetero_graph_attention_network.sort_plan(plan, schema)
    expected_sorted_plan = {
        "n1": [("e1", "n1", False), ("e1", "n1", True)],
        "n2": [("e2", "n1", False)],
    }
    self.assertEqual(sorted_plan, expected_sorted_plan)

  def test_message_passing(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
            "n2": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1"),
            "e2": schema_lib.EdgeSchema(source="n1", target="n2"),
        },
    )
    gnn = (
        hetero_graph_attention_network.HeterogeneousGraphAttentionNetworkConfig(
            plan=[
                ("e1", False),  # n1 -> n1
                ("e1", True),  # n1 <- n1
                ("e2", False),  # n1 -> n2
            ],
            dims=128,
            num_heads=4,
        ).make(schema)
    )
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

  def test_cross_regime_and_edge_permutation_equivalence(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
            "n2": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n2"),
        },
    )
    gnn = (
        hetero_graph_attention_network.HeterogeneousGraphAttentionNetworkConfig(
            dims=16,
            num_heads=4,
        ).make(schema)
    )
    sparse_graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.ones((8, 16))}, num_nodes=8
            ),
            "n2": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.ones((5, 16))}, num_nodes=5
            ),
        },
        edge_sets={
            "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
            ),
        },
    )
    variables = gnn.init(jax.random.PRNGKey(0), sparse_graph, training=False)

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(3), 4)
    n1_feat = jax.random.normal(k1, (5, 16))
    n2_feat = jax.random.normal(k2, (8, 16))
    src_idx = jax.random.randint(k3, (30,), 0, 5, dtype=jnp.int32)
    dst_idx = jax.random.randint(k4, (30,), 0, 8, dtype=jnp.int32)
    perm = jax.random.permutation(jax.random.PRNGKey(5), 30)

    dense_graph_a = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": n1_feat}, num_nodes=5
            ),
            "n2": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": n2_feat}, num_nodes=8
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

    def loss_fn(p, g):
      out = gnn.apply(p, g, training=False)
      return jnp.sum(out.node_sets["n1"].features["embedding"] ** 2) + jnp.sum(
          out.node_sets["n2"].features["embedding"] ** 2
      )

    val_a, grad_a = jax.value_and_grad(loss_fn)(variables, dense_graph_a)
    val_b, grad_b = jax.value_and_grad(loss_fn)(variables, dense_graph_b)
    np.testing.assert_allclose(val_a, val_b, rtol=1e-5, atol=1e-5)
    for leaf_a, leaf_b in zip(
        jax.tree_util.tree_leaves(grad_a), jax.tree_util.tree_leaves(grad_b)
    ):
      np.testing.assert_allclose(leaf_a, leaf_b, rtol=1e-5, atol=1e-5)

  @parameterized.parameters(2, 30)  # Fewer / more edges than nodes.
  def test_optimized_matches_basic_implementation(self, num_edges: int):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
            "n2": schema_lib.NodeSchema(features={}),
        },
        edge_sets={"e1": schema_lib.EdgeSchema(source="n1", target="n2")},
    )
    config_cls = (
        hetero_graph_attention_network.HeterogeneousGraphAttentionNetworkConfig
    )
    fast = config_cls(dims=16, num_heads=4).make(schema)
    basic = config_cls(
        dims=16, num_heads=4, force_basic_implementation=True
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
    for rel in ["e1_fwd", "e1_rev"]:
      src = params.pop(f"msg_{rel}_src")
      dst = params.pop(f"msg_{rel}_dst")
      out = params.pop(f"msg_{rel}_out")
      params[f"message_{rel}"] = {
          "dense_0": {
              "kernel": jnp.concatenate([src["kernel"], dst["kernel"]], axis=0),
              "bias": dst["bias"],
          },
          "dense_2": out,
      }
    basic_vars = {**fast_vars, "params": params}

    def loss(model, variables, g):
      out = model.apply(variables, g, training=False)
      return sum(
          jnp.sum(ns.features["embedding"] ** 2)
          for ns in out.node_sets.values()
      ), out

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
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={"e1": schema_lib.EdgeSchema(source="n1", target="n1")},
    )
    gnn = (
        hetero_graph_attention_network.HeterogeneousGraphAttentionNetworkConfig(
            dims=16,
            num_heads=4,
            message=standard.GenericBlockConfig("LA", dims=16),
        ).make(schema)
    )
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
    config = (
        hetero_graph_attention_network.HeterogeneousGraphAttentionNetworkConfig()
    )
    infra_str = config.architecture()
    logging.info("architecture:\n%s", infra_str)
    self.assertIn("X = ...", infra_str)
    self.assertIn("HeterogeneousGraphAttentionNetwork (heads=4):", infra_str)
    self.assertIn("  Message/Value:", infra_str)
    self.assertIn("    Dense(128)", infra_str)
    self.assertIn("  Update:", infra_str)
    self.assertIn("Residual(X)", infra_str)
    self.assertIn("# Post Attention FFN", infra_str)
    self.assertIn("Norm(rms_norm)", infra_str)


if __name__ == "__main__":
  absltest.main()
