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

"""Tests for graph perturbation operations."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.transform import perturb
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np


class PerturbTest(parameterized.TestCase):

  def test_perturb_node_feature_preserves_marginal_distribution_and_shares_arrays(
      self,
  ):
    graph, schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=50, num_n2_nodes=80, random_seed=0
    )
    nodeset_name = "N1"
    target_feat = "f1"
    other_feat = "f3"

    rng = np.random.default_rng(123)
    perturbed = perturb.perturb_node_feature(
        graph, schema, nodeset_name, target_feat, rng
    )

    orig_vals = graph.node_sets[nodeset_name].features[target_feat]
    pert_vals = perturbed.node_sets[nodeset_name].features[target_feat]
    self.assertEqual(orig_vals.shape, pert_vals.shape)
    self.assertEqual(orig_vals.dtype, pert_vals.dtype)

    # Marginal multiset of rows must be identical.
    test_util.assert_are_equal(self, np.sort(orig_vals), np.sort(pert_vals))
    self.assertFalse(np.array_equal(orig_vals, pert_vals))

    # Unperturbed feature array must be shared (zero-copy).
    self.assertIs(
        graph.node_sets[nodeset_name].features[other_feat],
        perturbed.node_sets[nodeset_name].features[other_feat],
    )

  def test_perturb_node_feature_timeseries_groups(self):
    # A nodeset with a timeseries group "g" (values, timestamps), an ungrouped
    # timeseries "ts" and a static feature "s".
    num_nodes = 20
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=num_nodes,
                features={
                    "g_val": (
                        np.arange(num_nodes * 3, dtype=np.float32).reshape(
                            num_nodes, 3
                        )
                    ),
                    "g_ts": (
                        np.arange(num_nodes * 3, dtype=np.int64).reshape(
                            num_nodes, 3
                        )
                    ),
                    "ts": (
                        np.arange(num_nodes * 2, dtype=np.float32).reshape(
                            num_nodes, 2
                        )
                    ),
                    "s": np.arange(num_nodes, dtype=np.float32),
                },
            )
        },
        edge_sets={},
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n": schema_lib.NodeSchema(
                features={
                    "g_val": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        is_timeseries=True,
                        group="g",
                    ),
                    "g_ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                        is_creation_time=True,
                        group="g",
                    ),
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        is_timeseries=True,
                    ),
                    "s": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    ),
                }
            )
        },
        edge_sets={},
    )
    self.assertEqual(
        perturb.shuffled_feature_names(schema.node_sets["n"].features, "g_val"),
        ["g_val", "g_ts"],
    )
    self.assertEqual(
        perturb.shuffled_feature_names(schema.node_sets["n"].features, "ts"),
        ["ts"],
    )

    # Perturbing a grouped feature shuffles the whole group with the same
    # permutation, and nothing else.
    rng = np.random.default_rng(7)
    perturbed = perturb.perturb_node_feature(graph, schema, "n", "g_val", rng)
    features = graph.node_sets["n"].features
    pert_features = perturbed.node_sets["n"].features
    perm = np.argsort(pert_features["g_val"][:, 0])
    self.assertFalse(np.array_equal(perm, np.arange(num_nodes)))
    test_util.assert_are_equal(
        self, pert_features["g_val"][perm], features["g_val"]
    )
    test_util.assert_are_equal(
        self, pert_features["g_ts"][perm], features["g_ts"]
    )
    for name in ["ts", "s"]:
      self.assertIs(pert_features[name], features[name])

    # Perturbing an ungrouped timeseries shuffles whole sequences.
    perturbed = perturb.perturb_node_feature(graph, schema, "n", "ts", rng)
    pert_ts = perturbed.node_sets["n"].features["ts"]
    self.assertFalse(np.array_equal(pert_ts, features["ts"]))
    test_util.assert_are_equal(
        self, pert_ts[np.argsort(pert_ts[:, 0])], features["ts"]
    )

  def test_maslov_sneppen_rewire_preserves_exact_degrees_and_avoids_self_loops(
      self,
  ):
    rng = np.random.default_rng(42)
    num_nodes = 50
    num_edges = 400
    sources = rng.integers(0, num_nodes, size=num_edges, dtype=np.int32)
    targets = (sources + rng.integers(1, num_nodes, size=num_edges)) % num_nodes
    targets = targets.astype(np.int32)
    adj = np.stack([sources, targets], axis=0)

    rewired = perturb.maslov_sneppen_rewire(
        adj,
        rng,
        num_swaps_per_edge=3,
        allow_self_loops=False,
    )

    self.assertEqual(rewired.shape, adj.shape)
    # Out-degrees of every node must match exactly.
    orig_out_deg = np.bincount(adj[0], minlength=num_nodes)
    new_out_deg = np.bincount(rewired[0], minlength=num_nodes)
    test_util.assert_are_equal(self, orig_out_deg, new_out_deg)

    # In-degrees of every node must match exactly.
    orig_in_deg = np.bincount(adj[1], minlength=num_nodes)
    new_in_deg = np.bincount(rewired[1], minlength=num_nodes)
    test_util.assert_are_equal(self, orig_in_deg, new_in_deg)

    # No self-loops should be introduced.
    self.assertFalse(np.any(rewired[0] == rewired[1]))
    # The adjacency must actually change.
    self.assertFalse(np.array_equal(adj, rewired))

  @parameterized.parameters(
      perturb.EdgePerturbationMode.MASLOV_SNEPPEN,
      perturb.EdgePerturbationMode.DROP,
      "MASLOV_SNEPPEN",
      "DROP",
  )
  def test_perturb_edgeset_modes(self, mode):
    rng = np.random.default_rng(99)
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(
        include_e2=False
    )
    # Use enough edges for the shuffling to be observable.
    num_edges = 30
    edge_ts = np.arange(num_edges, dtype=np.int64) * 10
    edge_weight = np.arange(num_edges, dtype=np.float32)
    graph.edge_sets["e1"] = dataclasses.replace(
        graph.edge_sets["e1"],
        adjacency=rng.integers(0, 4, size=(2, num_edges), dtype=np.int64),
        features={"timestamp": edge_ts.copy(), "w": edge_weight},
    )
    schema.edge_sets["e1"].features["w"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32
    )

    perturbed = perturb.perturb_edgeset(graph, schema, "e1", rng, mode=mode)
    pert_edge = perturbed.edge_sets["e1"]
    mode = perturb.parse_edge_perturbation_mode(mode)

    if mode == perturb.EdgePerturbationMode.DROP:
      self.assertEqual(pert_edge.num_edges(), 0)
      self.assertEqual(len(pert_edge.features["w"]), 0)
      self.assertEqual(len(pert_edge.features["timestamp"]), 0)
    else:
      self.assertEqual(pert_edge.num_edges(), num_edges)
      # Creation timestamp must be preserved in-place when preserve_timestamps=True.
      test_util.assert_are_equal(self, pert_edge.features["timestamp"], edge_ts)
      # Same multiset of edge weights, but shuffled.
      test_util.assert_are_equal(
          self, np.sort(pert_edge.features["w"]), np.sort(edge_weight)
      )
      self.assertFalse(np.array_equal(pert_edge.features["w"], edge_weight))

  def test_perturb_nodeset_heterogeneous(self):
    graph, schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=40, num_n2_nodes=60, random_seed=1
    )
    rng = np.random.default_rng(11)
    # Perturb non-target nodeset "N2" while target is "N1".
    perturbed = perturb.perturb_nodeset(
        graph,
        schema,
        "N2",
        rng,
        target_nodeset="N1",
        target_column="label",
    )
    # Target column on "N1" must be untouched.
    test_util.assert_are_equal(
        self,
        graph.node_sets["N1"].features["label"],
        perturbed.node_sets["N1"].features["label"],
    )
    # N2's feature f2 and incident edge N1_to_N2 must be perturbed.
    self.assertFalse(
        np.array_equal(
            graph.node_sets["N2"].features["f2"],
            perturbed.node_sets["N2"].features["f2"],
        )
    )
    self.assertFalse(
        np.array_equal(
            graph.edge_sets["N1_to_N2"].adjacency,
            perturbed.edge_sets["N1_to_N2"].adjacency,
        )
    )
    # Non-incident edge N1_to_N1 must be untouched.
    self.assertIs(
        graph.edge_sets["N1_to_N1"].adjacency,
        perturbed.edge_sets["N1_to_N1"].adjacency,
    )


if __name__ == "__main__":
  absltest.main()
