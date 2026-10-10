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

"""Tests for the heterogeneous graph convolution layer.

The tests of the message passing operations are in `message_passing_ops_test`,
the tests of the attention scores in `graph_attention_test`, and the
comparisons against dense reference implementations in
`hetero_gnn_reference_test`.
"""

from collections.abc import Callable
import logging
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import graph_attention
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import standard
import jax
import jax.numpy as jnp
import numpy as np

Config = hetero_gnn.HeterogeneousGraphConvolutionConfig
MessageAggregation = hetero_gnn.MessageAggregation
RelationAggregation = hetero_gnn.RelationAggregation

DIMS = 16
NUM_N1_NODES = 6
NUM_N2_NODES = 5

# "e1": n1 -> n2, "e2": n2 -> n2. With the default plan, "n2" receives
# messages from two relations ("e1_fwd" and "e2_fwd"/"e2_rev").
SCHEMA = schema_lib.GraphSchema(
    node_sets={},
    edge_sets={
        "e1": schema_lib.EdgeSchema(source="n1", target="n2"),
        "e2": schema_lib.EdgeSchema(source="n2", target="n2"),
    },
)

# A single relation "e1_fwd": n1 -> n2. "n1" is not a target.
SINGLE_RELATION_PLAN = [("e1", False)]

# Configurations covering every `MessageAggregation` (and attention score).
CONFIGS: dict[
    str, Callable[..., hetero_gnn.HeterogeneousGraphConvolutionConfig]
] = {
    "sum": lambda **kw: Config(dims=DIMS, **kw),
    "mean": lambda **kw: Config(
        dims=DIMS, message_aggregation=MessageAggregation.MEAN, **kw
    ),
    "symmetric": lambda **kw: Config(
        dims=DIMS,
        message_aggregation=MessageAggregation.SYMMETRIC,
        message_consumes_target=False,
        **kw,
    ),
    "dot_product_attention": lambda **kw: Config(
        dims=DIMS,
        message_aggregation=MessageAggregation.ATTENTION,
        attention=graph_attention.DotProductAttentionConfig(
            num_heads=4, dims=DIMS
        ),
        **kw,
    ),
    "gat": lambda **kw: Config(
        dims=DIMS,
        message_aggregation=MessageAggregation.ATTENTION,
        message_consumes_target=False,
        attention=graph_attention.GatAttentionConfig(num_heads=4),
        **kw,
    ),
    "gatv2": lambda **kw: Config(
        dims=DIMS,
        message_aggregation=MessageAggregation.ATTENTION,
        attention=graph_attention.Gatv2AttentionConfig(num_heads=4, dims=DIMS),
        **kw,
    ),
}

TEMPLATES: dict[
    str, Callable[..., hetero_gnn.HeterogeneousGraphConvolutionConfig]
] = {
    "graphsage": Config.graphsage,
    "gcn": Config.gcn,
    "gat": Config.gat,
    "gatv2": Config.gatv2,
    "dot_product_attention": Config.dot_product_attention,
}


def make_graph(
    num_edges: int, seed: int = 3
) -> jax_in_memory_graph.JaxInMemoryGraph:
  """Random graph following `SCHEMA` with `num_edges` edges per edge set."""
  keys = iter(jax.random.split(jax.random.PRNGKey(seed), 6))
  return jax_in_memory_graph.JaxInMemoryGraph(
      node_sets={
          "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
              features={
                  "embedding": jax.random.normal(
                      next(keys), (NUM_N1_NODES, DIMS)
                  )
              },
              num_nodes=NUM_N1_NODES,
          ),
          "n2": jax_in_memory_graph.JaxInMemoryNodeSet(
              features={
                  "embedding": jax.random.normal(
                      next(keys), (NUM_N2_NODES, DIMS)
                  )
              },
              num_nodes=NUM_N2_NODES,
          ),
      },
      edge_sets={
          "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
              adjacency=jnp.stack([
                  jax.random.randint(next(keys), (num_edges,), 0, NUM_N1_NODES),
                  jax.random.randint(next(keys), (num_edges,), 0, NUM_N2_NODES),
              ])
          ),
          "e2": jax_in_memory_graph.JaxInMemoryEdgeSet(
              adjacency=jax.random.randint(
                  next(keys), (2, num_edges), 0, NUM_N2_NODES
              )
          ),
      },
  )


def permute_edges(
    graph: jax_in_memory_graph.JaxInMemoryGraph, seed: int = 11
) -> jax_in_memory_graph.JaxInMemoryGraph:
  """Returns the graph with the edges of each edge set shuffled."""
  edge_sets = {}
  for name, edge_set in graph.edge_sets.items():
    perm = jax.random.permutation(
        jax.random.PRNGKey(seed), edge_set.adjacency.shape[1]
    )
    edge_sets[name] = jax_in_memory_graph.JaxInMemoryEdgeSet(
        adjacency=edge_set.adjacency[:, perm]
    )
  return jax_in_memory_graph.JaxInMemoryGraph(
      node_sets=graph.node_sets, edge_sets=edge_sets
  )


def loss_and_grads(
    model: hetero_gnn.HeterogeneousGraphConvolution,
    variables: dict[str, dict[str, jax.Array]],
    graph: jax_in_memory_graph.JaxInMemoryGraph,
) -> tuple[
    jax.Array,
    jax_in_memory_graph.JaxInMemoryGraph,
    jax_in_memory_graph.JaxInMemoryGraph,
]:
  """Returns the (loss, output graph, gradient wrt. the input graph)."""

  def loss_fn(g):
    out = model.apply(variables, g, training=False)
    loss = sum(
        jnp.sum(ns.features["embedding"] ** 2) for ns in out.node_sets.values()
    )
    return loss, out

  (loss, out), grads = jax.value_and_grad(
      loss_fn, has_aux=True, allow_int=True
  )(graph)
  return loss, out, grads


def fast_to_basic_params(
    params: dict[str, dict[str, jax.Array]],
    relation_names: list[str],
    use_out_bias: bool,
) -> dict[str, dict[str, jax.Array]]:
  """Maps the parameters of the "LAL" decomposition onto the basic block.

  The optimized implementation applies three dense layers (src, dst, out); the
  basic one applies a single "LAL" block on `concat([src, dst])`.

  Args:
    params: Parameters of the optimized implementation.
    relation_names: Names of the relations.
    use_out_bias: Whether the output dense layer of the optimized implementation
      has a bias (True with attention).

  Returns:
    The parameters of the basic implementation.
  """
  params = dict(params)
  for relation_name in relation_names:
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
            "bias": (
                out["bias"]
                if use_out_bias
                else jnp.zeros(out["kernel"].shape[1])
            ),
        },
    }
  return params


class HeteroGNNTest(parameterized.TestCase):

  def assert_graphs_close(
      self,
      a: jax_in_memory_graph.JaxInMemoryGraph,
      b: jax_in_memory_graph.JaxInMemoryGraph,
      tolerance: float = 1e-5,
  ):
    self.assertEqual(set(a.node_sets), set(b.node_sets))
    for name in a.node_sets:
      np.testing.assert_allclose(
          a.node_sets[name].features["embedding"],
          b.node_sets[name].features["embedding"],
          rtol=tolerance,
          atol=tolerance,
      )

  # Plan and configuration
  # ======================

  def test_sort_plan(self):
    plan = [("e1", False), ("e2", False), ("e2", True)]
    self.assertEqual(
        hetero_gnn.sort_plan(plan, SCHEMA),
        {"n2": [("e1", "n1", False), ("e2", "n2", False), ("e2", "n2", True)]},
    )
    self.assertEqual(
        hetero_gnn.sort_plan([("e1", True)], SCHEMA),
        {"n1": [("e1", "n2", True)]},
    )

  def test_config_validation(self):
    with self.assertRaisesRegex(ValueError, "message_consumes_target"):
      Config(
          message_aggregation=MessageAggregation.SYMMETRIC,
          message_consumes_target=True,
      )
    with self.assertRaisesRegex(ValueError, "`attention` must be set"):
      Config(message_aggregation=MessageAggregation.ATTENTION)
    with self.assertRaisesRegex(ValueError, "`attention` must be set"):
      Config(attention=graph_attention.GatAttentionConfig())
    with self.assertRaisesRegex(ValueError, "attention_dropout_rate"):
      Config(attention_dropout_rate=0.1)
    with self.assertRaisesRegex(ValueError, "relation_aggregation='concat'"):
      Config(
          relation_aggregation=RelationAggregation.CONCAT,
          combine=hetero_gnn.Combine.SUM,
      )

  def test_deprecated_message_pooling_alias(self):
    config = Config(message_pooling="mean")
    self.assertEqual(config.message_aggregation, MessageAggregation.MEAN)
    self.assertIsNone(config.message_pooling)
    # Round-trips through the JSON serialization of the registry.
    restored = Config.from_json(config.to_json())  # pyrefly: ignore[missing-attribute]
    self.assertEqual(restored.message_aggregation, MessageAggregation.MEAN)

  @parameterized.parameters(*TEMPLATES)
  def test_template_json_round_trip(self, template: str):
    config = TEMPLATES[template](dims=DIMS)
    config.relation_aggregation = RelationAggregation.MEAN
    restored = Config.from_json(config.to_json())  # pyrefly: ignore[missing-attribute]
    self.assertEqual(restored, config)

  def test_architecture(self):
    architecture = Config().architecture()
    logging.info("architecture:\n%s", architecture)
    self.assertIn("X = ...", architecture)
    self.assertIn("MPNN:", architecture)
    self.assertIn("  Message(concat(source, target)):", architecture)
    self.assertIn("    Dense(128)", architecture)
    self.assertIn("  Aggregation(sum)", architecture)
    self.assertIn("  RelationAggregation(sum)", architecture)
    self.assertIn("  Update(concat(X, messages)):", architecture)
    self.assertIn("Residual(X)", architecture)
    self.assertIn("# Post MPNN", architecture)
    self.assertIn("Norm(rms_norm)", architecture)

  @parameterized.named_parameters(
      (
          "graphsage",
          "graphsage",
          ["Message(source):", "Aggregation(mean)", "Activation(relu)"],
      ),
      ("gcn", "gcn", ["Aggregation(symmetric)", "Update(X + messages):"]),
      (
          "gat",
          "gat",
          [
              "Message(source):",
              "Aggregation(attention):",
              "GatAttention(heads=8, negative_slope=0.2)",
              "Activation(elu)",
          ],
      ),
      (
          "gatv2",
          "gatv2",
          ["Aggregation(attention):", "Gatv2Attention(heads=8, dims=16"],
      ),
      (
          "dot_product_attention",
          "dot_product_attention",
          [
              "Message(concat(source, target)):",
              "Aggregation(attention):",
              "DotProductAttention(heads=4, dims=16)",
              "Residual(X)",
          ],
      ),
  )
  def test_templates_architecture(self, template: str, expected: list[str]):
    architecture = TEMPLATES[template](dims=DIMS).architecture()
    logging.info("%s architecture:\n%s", template, architecture)
    for part in expected:
      self.assertIn(part, architecture)

  # Layer behavior
  # ==============

  @parameterized.product(
      config=list(CONFIGS),
      relation_aggregation=list(RelationAggregation),
  )
  def test_message_passing(
      self, config: str, relation_aggregation: RelationAggregation
  ):
    gnn = CONFIGS[config](relation_aggregation=relation_aggregation).make(
        SCHEMA
    )
    graph = make_graph(num_edges=10)
    variables = gnn.init(jax.random.PRNGKey(0), graph, training=False)
    output = gnn.apply(
        variables,
        graph,
        training=True,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    for name, num_nodes in [("n1", NUM_N1_NODES), ("n2", NUM_N2_NODES)]:
      embedding = output.node_sets[name].features["embedding"]
      self.assertEqual(embedding.shape, (num_nodes, DIMS))
      self.assertTrue(np.all(np.isfinite(embedding)))

  @parameterized.parameters(*CONFIGS)
  def test_non_target_nodesets_are_unchanged(self, config: str):
    gnn = CONFIGS[config](plan=SINGLE_RELATION_PLAN).make(SCHEMA)
    graph = make_graph(num_edges=10)
    variables = gnn.init(jax.random.PRNGKey(0), graph, training=False)
    output = gnn.apply(variables, graph, training=False)
    np.testing.assert_array_equal(
        output.node_sets["n1"].features["embedding"],
        graph.node_sets["n1"].features["embedding"],
    )

  @parameterized.product(
      config=list(CONFIGS),
      relation_aggregation=list(RelationAggregation),
  )
  def test_cross_regime_and_edge_permutation_equivalence(
      self, config: str, relation_aggregation: RelationAggregation
  ):
    gnn = CONFIGS[config](relation_aggregation=relation_aggregation).make(
        SCHEMA
    )
    # Initialize on a sparse graph (few edges, nodes-heavy code paths) and
    # apply on a dense graph (edges-heavy code paths).
    variables = gnn.init(
        jax.random.PRNGKey(7), make_graph(num_edges=2), training=False
    )
    graph = make_graph(num_edges=30)
    loss_a, out_a, grads_a = loss_and_grads(gnn, variables, graph)
    loss_b, out_b, grads_b = loss_and_grads(
        gnn, variables, permute_edges(graph)
    )
    np.testing.assert_allclose(loss_a, loss_b, rtol=1e-5)
    self.assert_graphs_close(out_a, out_b)
    self.assert_graphs_close(grads_a, grads_b)

  @parameterized.product(
      config=list(CONFIGS) + ["graphsage", "gcn"],
      num_edges=[2, 30],  # Fewer / more edges than nodes.
  )
  def test_optimized_matches_basic_implementation(
      self, config: str, num_edges: int
  ):
    make_config = CONFIGS[config] if config in CONFIGS else TEMPLATES[config]
    fast_config = make_config(plan=SINGLE_RELATION_PLAN)
    basic_config = make_config(plan=SINGLE_RELATION_PLAN)
    basic_config.force_basic_implementation = True
    fast = fast_config.make(SCHEMA)
    basic = basic_config.make(SCHEMA)
    graph = make_graph(num_edges=num_edges)

    fast_vars = fast.init(jax.random.PRNGKey(0), graph, training=False)
    basic_vars = fast_vars
    if fast_config.message_consumes_target:
      basic_vars = {
          **fast_vars,
          "params": fast_to_basic_params(
              fast_vars["params"],
              ["e1_fwd"],
              use_out_bias=fast_config.attention is not None,
          ),
      }

    loss_fast, out_fast, grads_fast = loss_and_grads(fast, fast_vars, graph)
    loss_basic, out_basic, grads_basic = loss_and_grads(
        basic, basic_vars, graph
    )
    np.testing.assert_allclose(loss_fast, loss_basic, rtol=1e-4)
    self.assert_graphs_close(out_fast, out_basic, tolerance=1e-4)
    self.assert_graphs_close(grads_fast, grads_basic, tolerance=1e-4)

  @parameterized.parameters("sum", "dot_product_attention", "gatv2")
  def test_non_lal_message_fallback(self, config: str):
    gnn = CONFIGS[config](
        message=standard.GenericBlockConfig("LA", dims=DIMS)
    ).make(SCHEMA)
    graph = make_graph(num_edges=10)
    variables = gnn.init(jax.random.PRNGKey(0), graph, training=False)
    self.assertIn("msg_e1_fwd", variables["params"])
    output = gnn.apply(variables, graph, training=False)
    self.assertEqual(
        output.node_sets["n2"].features["embedding"].shape,
        (NUM_N2_NODES, DIMS),
    )

  def test_attention_dropout(self):
    graph = make_graph(num_edges=30)
    gnn = Config.gat(dims=DIMS, attention_dropout_rate=0.5).make(SCHEMA)
    no_dropout_gnn = Config.gat(dims=DIMS).make(SCHEMA)
    variables = gnn.init(jax.random.PRNGKey(0), graph, training=False)

    def apply(model, training, seed=0):
      return model.apply(
          variables,
          graph,
          training=training,
          rngs={"dropout": jax.random.PRNGKey(seed)},
      )

    # Without training, the attention dropout is disabled.
    self.assert_graphs_close(apply(gnn, False), apply(no_dropout_gnn, False))
    # The gat template has no other dropout: Without attention dropout, the
    # training mode does not change the output.
    self.assert_graphs_close(
        apply(no_dropout_gnn, True), apply(no_dropout_gnn, False)
    )
    # With training, the attention dropout changes the output.
    with self.assertRaises(AssertionError):
      self.assert_graphs_close(apply(gnn, True), apply(gnn, False))

  def test_message_dims_must_be_divisible_by_num_heads(self):
    gnn = Config(
        dims=DIMS,
        message_aggregation=MessageAggregation.ATTENTION,
        attention=graph_attention.GatAttentionConfig(num_heads=3),
    ).make(SCHEMA)
    with self.assertRaisesRegex(ValueError, "divisible"):
      gnn.init(jax.random.PRNGKey(0), make_graph(num_edges=4), training=False)

  def test_combine_sum_requires_matching_dims(self):
    gnn = Config(
        dims=DIMS,
        combine=hetero_gnn.Combine.SUM,
        message=standard.GenericBlockConfig("LAL", dims=2 * DIMS),
    ).make(SCHEMA)
    with self.assertRaisesRegex(ValueError, "same dimension"):
      gnn.init(jax.random.PRNGKey(0), make_graph(num_edges=4), training=False)


if __name__ == "__main__":
  absltest.main()
