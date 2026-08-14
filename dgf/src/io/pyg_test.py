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

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.io import pyg as pyg_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np
import torch
import torch_geometric


class PygTest(parameterized.TestCase):

  def test_graph_to_pyg_data(self):
    schema = gen_test_graph.generate_schema(
        variable_length=False, bytes_feature=False
    )
    graph = gen_test_graph.generate_in_memory_graph(
        variable_length=False, bytes_feature=False
    )

    pyg_data = pyg_lib.graph_to_pyg_data(graph, schema)

    self.assertIsInstance(pyg_data, torch_geometric.data.HeteroData)

    # Validate nodes
    self.assertEqual(pyg_data["n1"].num_nodes, graph.node_sets["n1"].num_nodes)
    self.assertEqual(pyg_data["n2"].num_nodes, graph.node_sets["n2"].num_nodes)

    # Validate node features
    np.testing.assert_array_equal(
        pyg_data["n1"].x.numpy(),
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        pyg_data["n2"].x.numpy(),
        np.array([[4, 10], [5, 11]], dtype=np.int64),
    )

    # Validate edges
    np.testing.assert_array_equal(
        pyg_data["n1", "e1", "n1"].edge_index.numpy(),
        graph.edge_sets["e1"].adjacency,
    )
    np.testing.assert_array_equal(
        pyg_data["n1", "e2", "n2"].edge_index.numpy(),
        graph.edge_sets["e2"].adjacency,
    )


if __name__ == "__main__":
  absltest.main()
