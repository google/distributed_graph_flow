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
from dgf.src.analyse import padding as padding_lib
from dgf.src.data import padding as padding_data_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util

test_util.disable_diff_truncation()


class PaddingTest(absltest.TestCase):

  def test_basic(self):
    schema = gen_test_graph.generate_schema()
    graphs = [
        gen_test_graph.generate_in_memory_graph(),
        gen_test_graph.generate_in_memory_graph(),
        gen_test_graph.generate_in_memory_graph(),
    ]
    padding = padding_lib.padding_from_graph_generator(schema, iter(graphs))
    expected_padding = padding_data_lib.Padding(
        node_sets={
            "n1": padding_data_lib.NodeSetPadding(num_nodes=4),
            "n2": padding_data_lib.NodeSetPadding(num_nodes=4),
        },
        edge_sets={
            "e1": padding_data_lib.EdgeSetPadding(num_edges=4),
            "e2": padding_data_lib.EdgeSetPadding(num_edges=4),
        },
    )
    test_util.assert_are_equal(self, padding, expected_padding)

  def test_print_padding(self):
    padding = padding_data_lib.Padding(
        node_sets={
            "n1": padding_data_lib.NodeSetPadding(num_nodes=10),
            "n2": padding_data_lib.NodeSetPadding(num_nodes=20),
        },
        edge_sets={
            "e1": padding_data_lib.EdgeSetPadding(num_edges=100),
        },
    )
    output = padding_lib.print_padding(padding, return_output=True)
    expected_output = """Graph Padding:

Node Sets:
  n1: 10 nodes
  n2: 20 nodes

Edge Sets:
  e1: 100 edges"""
    self.assertEqual(output, expected_output)


if __name__ == "__main__":
  absltest.main()
