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

"""Tests for the graph masking layers."""

from absl.testing import absltest
from dgf.src.data import jax_in_memory_graph as jax_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import graph_masking
from dgf.src.util import test_util
from flax import errors as flax_errors
import jax
import jax.numpy as jnp
import numpy as np


def _make_graph(
    num_nodes: int = 5, feat_dim: int = 4
) -> tuple[jax_graph_lib.JaxInMemoryGraph, schema_lib.GraphSchema]:
  """Homogeneous graph whose last node is a sentinel with zero features."""
  schema = schema_lib.GraphSchema(
      node_sets={
          "nodes": schema_lib.NodeSchema(
              features={
                  "embedding": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                      shape=(feat_dim,),
                  )
              }
          )
      },
      edge_sets={
          "edges": schema_lib.EdgeSchema(
              source="nodes",
              target="nodes",
              features={
                  "weight": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32, shape=()
                  )
              },
          )
      },
  )
  real_feats = (
      np.random.RandomState(0).randn(num_nodes - 1, feat_dim).astype(np.float32)
  )
  feat = np.concatenate(
      [real_feats, np.zeros((1, feat_dim), dtype=np.float32)], axis=0
  )
  # Edges: 0->1, 1->2, 2->3, 3->0, 0->2.
  src = np.array([0, 1, 2, 3, 0])
  dst = np.array([1, 2, 3, 0, 2])
  graph = jax_graph_lib.JaxInMemoryGraph(
      node_sets={
          "nodes": jax_graph_lib.JaxInMemoryNodeSet(
              features={"embedding": jnp.asarray(feat)},
              num_nodes=num_nodes,
          )
      },
      edge_sets={
          "edges": jax_graph_lib.JaxInMemoryEdgeSet(
              adjacency=jnp.stack([jnp.asarray(src), jnp.asarray(dst)]),
              features={"weight": jnp.arange(5, dtype=jnp.float32)},
          )
      },
  )
  return graph, schema


def _drop_edges(
    graph: jax_graph_lib.JaxInMemoryGraph,
    schema: schema_lib.GraphSchema,
    seed: int = 0,
    drop_rate: float = 0.5,
    training: bool = True,
) -> jax_graph_lib.JaxInMemoryGraph:
  layer = graph_masking.DropEdgesConfig(drop_rate=drop_rate).make(schema)
  return layer.apply(
      {}, graph, training=training, rngs={"dropout": jax.random.PRNGKey(seed)}
  )


class DropEdgesTest(absltest.TestCase):

  def _drop(self, seed: int, drop_rate: float = 0.5, training: bool = True):
    graph, schema = _make_graph()
    result = _drop_edges(graph, schema, seed, drop_rate, training)
    return graph, result

  def test_zero_rate_returns_same_graph(self):
    graph, result = self._drop(0, drop_rate=0.0)
    self.assertIs(result, graph)

  def test_not_training_returns_same_graph(self):
    graph, result = self._drop(0, drop_rate=1.0, training=False)
    self.assertIs(result, graph)

  def test_shape_preserved(self):
    graph, result = self._drop(42)
    self.assertEqual(
        result.edge_sets["edges"].adjacency.shape,
        graph.edge_sets["edges"].adjacency.shape,
    )

  def test_drop_all_edges_redirects_to_sentinel(self):
    _, result = self._drop(0, drop_rate=1.0)
    np.testing.assert_array_equal(
        result.edge_sets["edges"].adjacency, np.full((2, 5), 4)
    )

  def test_kept_and_dropped_edges(self):
    graph, result = self._drop(7)
    orig = np.asarray(graph.edge_sets["edges"].adjacency)
    new = np.asarray(result.edge_sets["edges"].adjacency)
    sentinel = 4
    is_dropped = new[0] == sentinel
    self.assertGreater(is_dropped.sum(), 0)
    self.assertLess(is_dropped.sum(), 5)
    # Dropped edges are sentinel self-loops.
    np.testing.assert_array_equal(new[1][is_dropped], sentinel)
    # Kept edges are unchanged.
    np.testing.assert_array_equal(new[:, ~is_dropped], orig[:, ~is_dropped])

  def test_node_sets_unchanged(self):
    graph, result = self._drop(0)
    test_util.assert_are_equal(self, result.node_sets, graph.node_sets)

  def test_edge_features_unchanged(self):
    graph, result = self._drop(7)
    test_util.assert_are_equal(
        self,
        result.edge_sets["edges"].features,
        graph.edge_sets["edges"].features,
    )

  def test_deterministic_with_same_key(self):
    _, r1 = self._drop(99)
    _, r2 = self._drop(99)
    np.testing.assert_array_equal(
        r1.edge_sets["edges"].adjacency, r2.edge_sets["edges"].adjacency
    )

  def test_different_keys_give_different_results(self):
    _, r1 = self._drop(0)
    _, r2 = self._drop(12345)
    self.assertFalse(
        np.array_equal(
            r1.edge_sets["edges"].adjacency, r2.edge_sets["edges"].adjacency
        )
    )

  def test_jit_compatible(self):
    graph, schema = _make_graph()
    layer = graph_masking.DropEdgesConfig(drop_rate=0.5).make(schema)

    @jax.jit
    def jitted_drop(rng, g):
      return layer.apply({}, g, training=True, rngs={"dropout": rng})

    result = jitted_drop(jax.random.PRNGKey(0), graph)
    self.assertEqual(result.edge_sets["edges"].adjacency.shape, (2, 5))

  def test_edge_set_missing_from_graph_raises(self):
    graph, schema = _make_graph()
    graph.edge_sets.clear()
    with self.assertRaises(KeyError):
      _drop_edges(graph, schema)

  def test_edge_sets_have_independent_masks(self):
    num_nodes, num_edges = 5, 64
    edge_schema = schema_lib.EdgeSchema(
        source="nodes", target="nodes", features={}
    )
    schema = schema_lib.GraphSchema(
        node_sets={"nodes": schema_lib.NodeSchema(features={})},
        edge_sets={"a": edge_schema, "b": edge_schema},
    )
    # Real nodes are 0..3; node 4 is the sentinel.
    adjacency = jnp.asarray(
        np.random.RandomState(0).randint(0, num_nodes - 1, (2, num_edges))
    )
    graph = jax_graph_lib.JaxInMemoryGraph(
        node_sets={
            "nodes": jax_graph_lib.JaxInMemoryNodeSet(
                features={}, num_nodes=num_nodes
            )
        },
        edge_sets={
            "a": jax_graph_lib.JaxInMemoryEdgeSet(adjacency=adjacency),
            "b": jax_graph_lib.JaxInMemoryEdgeSet(adjacency=adjacency),
        },
    )
    result = _drop_edges(graph, schema)
    self.assertFalse(
        np.array_equal(
            result.edge_sets["a"].adjacency, result.edge_sets["b"].adjacency
        )
    )

  def test_hetero_edges_redirected_to_source_and_target_padding(self):
    # Nodesets of different sizes; the last node of each is padding.
    num_authors, num_papers, num_edges = 6, 4, 32
    schema = schema_lib.GraphSchema(
        node_sets={
            "author": schema_lib.NodeSchema(features={}),
            "paper": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "writes": schema_lib.EdgeSchema(
                source="author", target="paper", features={}
            )
        },
    )
    rng = np.random.RandomState(0)
    adjacency = np.stack([
        rng.randint(0, num_authors - 1, num_edges),
        rng.randint(0, num_papers - 1, num_edges),
    ])
    graph = jax_graph_lib.JaxInMemoryGraph(
        node_sets={
            "author": jax_graph_lib.JaxInMemoryNodeSet(
                features={}, num_nodes=num_authors
            ),
            "paper": jax_graph_lib.JaxInMemoryNodeSet(
                features={}, num_nodes=num_papers
            ),
        },
        edge_sets={
            "writes": jax_graph_lib.JaxInMemoryEdgeSet(
                adjacency=jnp.asarray(adjacency)
            )
        },
    )
    new = np.asarray(_drop_edges(graph, schema).edge_sets["writes"].adjacency)
    padding_edge = np.array([[num_authors - 1], [num_papers - 1]])
    is_dropped = np.all(new == padding_edge, axis=0)
    self.assertGreater(is_dropped.sum(), 0)
    self.assertLess(is_dropped.sum(), num_edges)
    np.testing.assert_array_equal(
        new[:, ~is_dropped], adjacency[:, ~is_dropped]
    )

  def test_architecture(self):
    self.assertEqual(
        graph_masking.DropEdgesConfig(drop_rate=0.5).architecture(),
        "DropEdges(drop_rate=0.5)",
    )


class MaskNodeFeaturesTest(absltest.TestCase):

  def _apply(
      self,
      graph: jax_graph_lib.JaxInMemoryGraph,
      mask_rate: float,
      replace_rate: float = 0.0,
      seed: int = 3,
  ):
    layer = graph_masking.MaskNodeFeaturesConfig(
        nodeset="nodes", mask_rate=mask_rate, replace_rate=replace_rate
    ).make()
    # Use a non-zero mask token to distinguish it from the sentinel features.
    variables = {"params": {"mask_token": jnp.full((1, 8), 7.0)}}
    return layer.apply(
        variables, graph, rngs={"masking": jax.random.PRNGKey(seed)}
    )

  def test_init_creates_zero_mask_token(self):
    graph, _ = _make_graph(num_nodes=5, feat_dim=8)
    layer = graph_masking.MaskNodeFeaturesConfig(
        nodeset="nodes", mask_rate=0.5
    ).make()
    variables = layer.init(
        {"params": jax.random.PRNGKey(0), "masking": jax.random.PRNGKey(1)},
        graph,
    )
    np.testing.assert_array_equal(
        variables["params"]["mask_token"], np.zeros((1, 8))
    )

  def test_requires_masking_rng(self):
    graph, _ = _make_graph(num_nodes=5, feat_dim=8)
    layer = graph_masking.MaskNodeFeaturesConfig(
        nodeset="nodes", mask_rate=0.5
    ).make()
    variables = {"params": {"mask_token": jnp.zeros((1, 8))}}
    with self.assertRaisesRegex(flax_errors.InvalidRngError, "masking"):
      layer.apply(variables, graph, rngs={"dropout": jax.random.PRNGKey(0)})

  def test_mask_rate(self):
    graph, _ = _make_graph(num_nodes=11, feat_dim=8)
    _, is_masked = self._apply(graph, mask_rate=0.4)
    # int(0.4 * 11) = 4 masked nodes.
    self.assertEqual(int(jnp.sum(is_masked)), 4)

  def test_mask_replacement_no_noise(self):
    graph, _ = _make_graph(num_nodes=5, feat_dim=8)
    masked_graph, is_masked = self._apply(graph, mask_rate=0.5)

    orig_x = graph.node_sets["nodes"].features["embedding"]
    new_x = masked_graph.node_sets["nodes"].features["embedding"]
    for i in range(5):
      if is_masked[i]:
        np.testing.assert_allclose(new_x[i], np.full(8, 7.0))
      else:
        np.testing.assert_allclose(new_x[i], orig_x[i])

  def test_mask_replacement_with_noise(self):
    graph, _ = _make_graph(num_nodes=10, feat_dim=8)
    masked_graph, is_masked = self._apply(
        graph, mask_rate=0.8, replace_rate=0.5
    )

    orig_x = np.asarray(graph.node_sets["nodes"].features["embedding"])
    new_x = np.asarray(masked_graph.node_sets["nodes"].features["embedding"])
    num_token = 0
    num_noise = 0
    for i in range(10):
      if not is_masked[i]:
        np.testing.assert_allclose(new_x[i], orig_x[i])
      elif np.allclose(new_x[i], 7.0):
        num_token += 1
      else:
        # The feature is copied from a random donor node.
        self.assertTrue(
            any(np.allclose(new_x[i], orig_x[j]) for j in range(10)),
            f"Masked feature at {i} matches no donor feature",
        )
        num_noise += 1
    self.assertGreater(num_token, 0)
    self.assertGreater(num_noise, 0)

  def test_architecture(self):
    config = graph_masking.MaskNodeFeaturesConfig(
        nodeset="nodes", mask_rate=0.3, replace_rate=0.1
    )
    self.assertEqual(
        config.architecture(),
        "MaskNodeFeatures(nodeset='nodes', mask_rate=0.3, replace_rate=0.1)",
    )


if __name__ == "__main__":
  absltest.main()
