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

"""Tests for io.networkx."""

import os
from absl.testing import absltest
from dgf.src.io import networkx as networkx_io_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import networkx as nx


class NetworkXTest(absltest.TestCase):

  def test_graph_to_networkx_and_back(self):
    in_memory_graph = gen_test_graph.generate_in_memory_graph(
        variable_length=False
    )
    schema = gen_test_graph.generate_schema(variable_length=False)

    nx_graph = networkx_io_lib.graph_to_networkx(
        in_memory_graph=in_memory_graph,
        schema=schema,
    )

    recovered_graph, recovered_schema = networkx_io_lib.networkx_to_graph(
        nx_graph=nx_graph,
    )

    test_util.assert_are_equal(self, in_memory_graph, recovered_graph)
    self.assertIn("n1", recovered_schema.node_sets)
    self.assertIn("e1", recovered_schema.edge_sets)

  def test_graph_to_networkx_for_graphml(self):
    """Verifies graph generation behavior when for_graphml is True."""
    in_memory_graph = gen_test_graph.generate_in_memory_graph(
        variable_length=False
    )
    schema = gen_test_graph.generate_schema(variable_length=False)

    nx_graph = networkx_io_lib.graph_to_networkx(
        in_memory_graph=in_memory_graph,
        schema=schema,
        for_graphml=True,
    )

    # Verify specific nodes exist and their types are cleaned
    self.assertIn("n1_0", nx_graph.nodes)
    self.assertIsInstance(nx_graph.nodes["n1_0"]["f1"], str)

    self.assertIn("n2_0", nx_graph.nodes)
    self.assertIn("f3", nx_graph.nodes["n2_0"])
    self.assertIsInstance(nx_graph.nodes["n2_0"]["f3"], int)

    # Verify specific edges explicitly set their own multigraph keys
    self.assertIn("e2_0", nx_graph.get_edge_data("n1_0", "n2_0"))

    # Implicitly verifies all properties are safe for graphml, and edge keys
    # do not collide during graph serialization.
    temp_dir = self.create_tempdir().full_path
    temp_path = os.path.join(temp_dir, "test.graphml")
    nx.write_graphml(nx_graph, temp_path)

    self.assertTrue(
        os.path.exists(temp_path), "Failed to generate the GraphML file."
    )


if __name__ == "__main__":
  absltest.main()
