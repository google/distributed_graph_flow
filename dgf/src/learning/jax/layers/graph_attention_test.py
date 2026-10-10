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

"""Tests for the attention scores."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.jax.layers import graph_attention
from dgf.src.learning.jax.layers import message_passing_ops
import jax
import jax.numpy as jnp
import numpy as np

DIMS = 8
NUM_HEADS = 2


def make_relation(
    name: str,
    src_nodeset: str,
    dst_nodeset: str,
    node_values: dict[str, jax.Array],
    num_edges: int,
    seed: int,
) -> message_passing_ops.Relation:
  """Creates a relation with random edges between two node sets."""
  num_src_nodes = node_values[src_nodeset].shape[0]
  num_dst_nodes = node_values[dst_nodeset].shape[0]
  k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
  return message_passing_ops.Relation(
      name=name,
      src_nodeset=src_nodeset,
      dst_nodeset=dst_nodeset,
      src_values=node_values[src_nodeset],
      dst_values=node_values[dst_nodeset],
      source_idxs=jax.random.randint(k1, (num_edges,), 0, num_src_nodes),
      target_idxs=jax.random.randint(k2, (num_edges,), 0, num_dst_nodes),
      num_src_nodes=num_src_nodes,
      num_dst_nodes=num_dst_nodes,
      num_edges=num_edges,
      indices_are_sorted=False,
  )


def dense(params: dict[str, jax.Array], x: jax.Array) -> np.ndarray:
  """Applies the parameters of a nn.Dense in numpy."""
  out = np.asarray(x) @ np.asarray(params["kernel"])
  if "bias" in params:
    out = out + np.asarray(params["bias"])
  return out


def leaky_relu(x: np.ndarray, negative_slope: float) -> np.ndarray:
  return np.where(x >= 0, x, negative_slope * x)


def make_relations() -> list[message_passing_ops.Relation]:
  """Relations covering the three "smallest of (N_src, N_dst, E)" regimes of

  the dot-product attention.
  """
  node_values = {
      name: jax.random.normal(jax.random.PRNGKey(seed), (num_nodes, DIMS))
      for seed, (name, num_nodes) in enumerate(
          [("a", 3), ("b", 5), ("c", 2), ("d", 4)]
      )
  }
  return [
      # N_src <= N_dst, N_src <= E.
      make_relation("r_small_src", "a", "b", node_values, 20, seed=10),
      # N_dst < N_src, N_dst <= E.
      make_relation("r_small_dst", "b", "c", node_values, 20, seed=11),
      # E < N_dst < N_src.
      make_relation("r_few_edges", "b", "d", node_values, 3, seed=12),
  ]


class GraphAttentionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.relations = make_relations()

  def apply(
      self, config: graph_attention.GraphAttentionConfig
  ) -> tuple[dict[str, jax.Array], graph_attention.LogitsByRelation]:
    module = config.make()
    variables = module.init(jax.random.PRNGKey(0), self.relations)
    return variables["params"], module.apply(variables, self.relations)

  @parameterized.parameters(
      graph_attention.DotProductAttentionConfig(num_heads=NUM_HEADS, dims=DIMS),
      graph_attention.GatAttentionConfig(num_heads=NUM_HEADS),
      graph_attention.Gatv2AttentionConfig(num_heads=NUM_HEADS, dims=DIMS),
  )
  def test_logits_shape(self, config: graph_attention.GraphAttentionConfig):
    _, logits = self.apply(config)
    self.assertEqual(set(logits), {r.name for r in self.relations})
    for relation in self.relations:
      self.assertEqual(
          logits[relation.name].shape, (relation.num_edges, NUM_HEADS)
      )

  def test_dot_product_matches_reference(self):
    params, logits = self.apply(
        graph_attention.DotProductAttentionConfig(
            num_heads=NUM_HEADS, dims=DIMS
        )
    )
    # The queries and keys are shared per node set.
    self.assertEqual(
        set(params),
        {"q_proj_b", "q_proj_c", "q_proj_d", "k_proj_a", "k_proj_b"}
        | {f"w_att_{r.name}" for r in self.relations},
    )
    head_dims = DIMS // NUM_HEADS
    for relation in self.relations:
      queries = dense(
          params[f"q_proj_{relation.dst_nodeset}"], relation.dst_values
      ).reshape(-1, NUM_HEADS, head_dims)
      keys = dense(
          params[f"k_proj_{relation.src_nodeset}"], relation.src_values
      ).reshape(-1, NUM_HEADS, head_dims)
      w_att = np.asarray(params[f"w_att_{relation.name}"])
      # logits = q^T W_att^T k, per head.
      expected = np.einsum(
          "ehk,hdk,ehd->eh",
          queries[relation.target_idxs],
          w_att,
          keys[relation.source_idxs],
      ) / np.sqrt(head_dims)
      np.testing.assert_allclose(
          logits[relation.name], expected, rtol=1e-5, atol=1e-5
      )

  def test_gat_matches_reference(self):
    config = graph_attention.GatAttentionConfig(
        num_heads=NUM_HEADS, negative_slope=0.1
    )
    params, logits = self.apply(config)
    for relation in self.relations:
      src_scores = dense(
          params[f"att_{relation.name}_src"], relation.src_values
      )
      dst_scores = dense(
          params[f"att_{relation.name}_dst"], relation.dst_values
      )
      expected = leaky_relu(
          src_scores[relation.source_idxs] + dst_scores[relation.target_idxs],
          config.negative_slope,
      )
      np.testing.assert_allclose(
          logits[relation.name], expected, rtol=1e-5, atol=1e-5
      )

  def test_gatv2_matches_reference(self):
    config = graph_attention.Gatv2AttentionConfig(
        num_heads=NUM_HEADS, dims=DIMS, negative_slope=0.1
    )
    params, logits = self.apply(config)
    for relation in self.relations:
      hidden = leaky_relu(
          dense(params[f"att_{relation.name}_src"], relation.src_values)[
              relation.source_idxs
          ]
          + dense(params[f"att_{relation.name}_dst"], relation.dst_values)[
              relation.target_idxs
          ],
          config.negative_slope,
      ).reshape(relation.num_edges, NUM_HEADS, DIMS // NUM_HEADS)
      expected = np.einsum(
          "ehd,hd->eh", hidden, np.asarray(params[f"att_{relation.name}"])
      )
      np.testing.assert_allclose(
          logits[relation.name], expected, rtol=1e-5, atol=1e-5
      )

  @parameterized.parameters(
      graph_attention.DotProductAttentionConfig,
      graph_attention.Gatv2AttentionConfig,
  )
  def test_dims_must_be_divisible_by_num_heads(self, config_cls):
    with self.assertRaisesRegex(ValueError, "divisible"):
      config_cls(num_heads=3, dims=8)

  @parameterized.parameters(
      graph_attention.DotProductAttentionConfig(num_heads=2, dims=16),
      graph_attention.GatAttentionConfig(num_heads=2, negative_slope=0.3),
      graph_attention.Gatv2AttentionConfig(num_heads=2, dims=16),
  )
  def test_json_round_trip(self, config: graph_attention.GraphAttentionConfig):
    restored = type(config).from_json(config.to_json())  # pyrefly: ignore[missing-attribute]
    self.assertEqual(restored, config)

  def test_architecture(self):
    self.assertEqual(
        graph_attention.GatAttentionConfig(num_heads=8).architecture(),
        "GatAttention(heads=8, negative_slope=0.2)",
    )


if __name__ == "__main__":
  absltest.main()
