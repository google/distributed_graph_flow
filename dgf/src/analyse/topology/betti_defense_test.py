# Copyright 2024 Google LLC.
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
"""Tests for betti_defense."""

from absl.testing import absltest
import numpy as np
from dgf.src.analyse.topology import betti_defense
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib


class BettiDefenseTest(absltest.TestCase):

  def _make_homogeneous_graph(
      self,
      num_nodes: int,
      edges: list[tuple[int, int]],
  ) -> tuple[in_memory_graph.InMemoryGraph, schema_lib.GraphSchema]:
    adjacency = np.array(edges, dtype=np.int32).T if edges else np.zeros(
        (2, 0), dtype=np.int32)
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph.InMemoryNodeSet(
                num_nodes=num_nodes,
                features={},
            )
        },
        edge_sets={
            "edges": in_memory_graph.InMemoryEdgeSet(
                adjacency=adjacency,
                features={},
            )
        },
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "edges": schema_lib.EdgeSchema(
                source="nodes",
                target="nodes",
                features={},
            ),
        },
    )
    return graph, schema

  def test_empty_graph(self):
    graph, schema = self._make_homogeneous_graph(0, [])
    self.assertEqual(betti_defense.calculate_betti_1(graph, schema), 0)

  def test_single_node(self):
    graph, schema = self._make_homogeneous_graph(1, [])
    self.assertEqual(betti_defense.calculate_betti_1(graph, schema), 0)

  def test_tree_path_4(self):
    # 0-1-2-3 : 4 nodes, 3 edges, 1 component -> beta_1 = 0
    graph, schema = self._make_homogeneous_graph(4, [(0, 1), (1, 2), (2, 3)])
    self.assertEqual(betti_defense.calculate_betti_1(graph, schema), 0)

  def test_cycle_5(self):
    # 0-1-2-3-4-0 : 5 nodes, 5 edges, 1 component -> beta_1 = 1
    graph, schema = self._make_homogeneous_graph(
        5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    self.assertEqual(betti_defense.calculate_betti_1(graph, schema), 1)

  def test_two_triangles(self):
    # Two disconnected triangles: 6 nodes, 6 edges, 2 components -> beta_1 = 2
    graph, schema = self._make_homogeneous_graph(
        6,
        [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
    )
    self.assertEqual(betti_defense.calculate_betti_1(graph, schema), 2)

  def test_complete_k4(self):
    # K4: 4 nodes, 6 edges, 1 component -> beta_1 = 3
    graph, schema = self._make_homogeneous_graph(
        4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    self.assertEqual(betti_defense.calculate_betti_1(graph, schema), 3)

  def test_no_edges(self):
    graph, schema = self._make_homogeneous_graph(3, [])
    self.assertEqual(betti_defense.calculate_betti_1(graph, schema), 0)

  def test_is_anomalous(self):
    clean, schema = self._make_homogeneous_graph(4, [(0, 1), (1, 2), (2, 3)])
    anomalous, _ = self._make_homogeneous_graph(
        5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    self.assertFalse(betti_defense.is_anomalous(clean, schema, 0))
    self.assertTrue(betti_defense.is_anomalous(anomalous, schema, 0))
    self.assertFalse(betti_defense.is_anomalous(anomalous, schema, 1))

  def test_self_loops(self):
    # Self-loop adds an edge but doesn't change components.
    # 2 nodes, 2 edges (one real, one self-loop), 1 component -> beta_1 = 1
    graph, schema = self._make_homogeneous_graph(2, [(0, 1), (1, 1)])
    self.assertEqual(betti_defense.calculate_betti_1(graph, schema), 1)

  def test_raises_on_heterogeneous(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(num_nodes=2, features={}),
            "n2": in_memory_graph.InMemoryNodeSet(num_nodes=2, features={}),
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0], [1]], dtype=np.int32),
                features={},
            )
        },
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
            "n2": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1", target="n2", features={}),
        },
    )
    with self.assertRaises(ValueError):
      betti_defense.calculate_betti_1(graph, schema)


if __name__ == "__main__":
  absltest.main()
