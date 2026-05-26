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

import os
import tempfile

from absl.testing import absltest
from dgf.src.analyse.reports import data_model
from dgf.src.analyse.reports import reporter
from dgf.src.analyse.reports import visual_utils
from dgf.src.analyse.topology import global_graph_topology as global_graph_topology_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.util import gen_test_graph
import networkx as nx
import numpy as np


class VisualInspectionTest(absltest.TestCase):

  def test_graph_to_pyvis_data_coloring(self):
    g = nx.Graph()
    g.add_node(0, type="A")
    g.add_node(1, type="A")
    g.add_node(2, type="B")
    g.add_node(3)  # Missing attribute

    data = visual_utils.graph_to_pyvis_data(g, color_by_attribute="type")

    nodes = {n["id"]: n for n in data["nodes"]}

    # 0 and 1 should have same color
    self.assertEqual(nodes[0]["color"], nodes[1]["color"])
    # 0 and 2 should have different colors
    self.assertNotEqual(nodes[0]["color"], nodes[2]["color"])
    # 3 should have a color (grey/hash of None)
    self.assertIn("color", nodes[3])
    # Title should contain the attribute info
    self.assertIn("type: A", nodes[0]["title"])
    # extra_data should contain the raw attributes
    self.assertEqual(nodes[0]["extra_data"], {"type": "A"})

    # Test Edge Attributes
    g.add_edge(0, 1, color="red", weight=5)
    data_with_edges = visual_utils.graph_to_pyvis_data(
        g, color_by_attribute="type"
    )
    edges = data_with_edges["edges"]
    self.assertNotEmpty(edges)
    edge = edges[0]
    self.assertEqual(edge["color"], "red")
    self.assertEqual(edge["value"], 5)
    self.assertEqual(edge["extra_data"]["weight"], 5)
    self.assertEqual(edge["extra_data"]["color"], "red")

  def test_in_memory_graph_to_pyvis_data(self):
    # Create valid InMemoryGraph
    g = gen_test_graph.generate_in_memory_graph(node_ids=True, edge_ids=True)
    schema = gen_test_graph.generate_schema(node_ids=True, edge_ids=True)

    data = visual_utils.graph_to_pyvis_data(g, graph_schema=schema)

    self.assertLen(
        data["nodes"], sum(ns.num_nodes for ns in g.node_sets.values())
    )
    self.assertLen(
        data["edges"], sum(es.num_edges() for es in g.edge_sets.values())
    )

    # Check Legend
    self.assertIn("n1", data["legend"])

    # Check Node Content
    n0 = data["nodes"][0]
    self.assertEqual(
        n0["label"], g.node_sets["n1"].features["#id"][0].decode("utf-8")
    )  # from #id
    self.assertEqual(n0["id"], "n1_0")  # namespaced index
    self.assertEqual(n0["color"], data["legend"]["n1"])

    # Check Edge Content
    e0 = data["edges"][0]
    self.assertEqual(e0["from"], "n1_0")
    self.assertEqual(e0["to"], "n2_0")
    # Check simple scalar extraction
    self.assertEqual(e0["extra_data"]["type"], "e2")

  def test_in_memory_graph_array_features(self):
    ns = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=1, features={"embedding": np.array([[0.1, 0.2]])}
    )
    g = in_memory_graph_lib.InMemoryGraph(node_sets={"users": ns}, edge_sets={})
    data = visual_utils.graph_to_pyvis_data(g)

    # embedding should be stringified
    self.assertIn("embedding", data["nodes"][0]["extra_data"])
    self.assertIsInstance(data["nodes"][0]["extra_data"]["embedding"], str)
    # Check strict equality matches the string conversion format of list or
    # numpy? visual_utils uses str(val.tolist()) which is usually "[0.1, 0.2]"
    self.assertIn("0.1", data["nodes"][0]["extra_data"]["embedding"])

  def test_in_memory_graph_color_by_attribute(self):
    ns = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=2, features={"label": np.array(["A", "B"])}
    )
    g = in_memory_graph_lib.InMemoryGraph(node_sets={"nodes": ns}, edge_sets={})
    data = visual_utils.graph_to_pyvis_data(g, color_by_attribute="label")

    # Legend should contain A and B
    self.assertIn("A", data["legend"])
    self.assertIn("B", data["legend"])

    # Nodes should have different colors
    self.assertNotEqual(data["nodes"][0]["color"], data["nodes"][1]["color"])
    self.assertEqual(data["nodes"][0]["color"], data["legend"]["A"])

  def test_in_memory_graph_heterogeneous(self):
    ns1 = in_memory_graph_lib.InMemoryNodeSet(num_nodes=1)
    ns2 = in_memory_graph_lib.InMemoryNodeSet(num_nodes=1)
    es = in_memory_graph_lib.InMemoryEdgeSet(adjacency=np.array([[0], [0]]))
    g = in_memory_graph_lib.InMemoryGraph(
        node_sets={"users": ns1, "items": ns2}, edge_sets={"users_to_items": es}
    )

    # Define Schema
    schema = schema_lib.GraphSchema(
        node_sets={
            "users": schema_lib.NodeSchema(),
            "items": schema_lib.NodeSchema(),
        },
        edge_sets={
            "users_to_items": schema_lib.EdgeSchema(
                source="users", target="items"
            )
        },
    )

    # This should NOT crash with ID collision
    data = visual_utils.graph_to_pyvis_data(g, graph_schema=schema)

    # Check distinct IDs
    ids = [n["id"] for n in data["nodes"]]
    self.assertLen(ids, 2)
    self.assertIn("items_0", ids)

  def test_in_memory_graph_heterogeneous_color_by_attribute(self):
    ns1 = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=1, features={"label": np.array(["A"])}
    )
    ns2 = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=1, features={"label": np.array(["B"])}
    )
    g = in_memory_graph_lib.InMemoryGraph(
        node_sets={"users": ns1, "items": ns2}, edge_sets={}
    )

    data = visual_utils.graph_to_pyvis_data(g, color_by_attribute="label")

    # Legend should contain "A" and "B", NOT "users" and "items"
    self.assertIn("A", data["legend"])
    self.assertIn("B", data["legend"])
    # "users" and "items" should NOT be in legend if coloring by attribute
    self.assertNotIn("users", data["legend"])
    self.assertNotIn("items", data["legend"])

  def test_graph_to_pyvis_data(self):
    g = nx.karate_club_graph()
    data = visual_utils.graph_to_pyvis_data(g)

    self.assertIn("nodes", data)
    self.assertIn("edges", data)
    self.assertIn("options", data)
    self.assertLen(data["nodes"], 34)
    # PyVis nodes should have 'id', 'label', 'shape', etc.
    self.assertIn("id", data["nodes"][0])

  def test_reporter_integration(self):
    # Create a payload with subgraphs
    g1 = nx.path_graph(5)
    g2 = nx.cycle_graph(5)

    ggt = global_graph_topology_lib.GlobalGraphTopology(
        total_nodes=10,
        total_edges=9,
    )
    payload = data_model.GraphStatsPayload(
        dataset_name="Test Visual Graph",
        feature_dimensionality=128,
        subgraphs=[g1, g2],
        global_graph_topology=ggt,
    )

    # Run reporter
    with tempfile.TemporaryDirectory() as tmp_dir:
      reporter.generate_report(payload, tmp_dir)

      # Check if report.html exists
      html_path = os.path.join(tmp_dir, "report.html")
      self.assertTrue(os.path.exists(html_path))

      # Check content for visual gallery markers
      with open(html_path, "r") as f:
        content = f.read()
        self.assertIn("Visual Inspection Gallery", content)
        self.assertIn("mynetwork_1", content)
        self.assertIn("mynetwork_2", content)
        self.assertIn("vis.Network", content)


if __name__ == "__main__":
  absltest.main()
