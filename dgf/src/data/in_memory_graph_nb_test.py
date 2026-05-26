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

"""Tests for C++ extension for in-memory graph views."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph_nb_test_ext
from dgf.src.util import gen_test_graph


class InMemoryGraphNbTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'without_ids',
          'node_ids': False,
          'edge_ids': False,
      },
      {
          'testcase_name': 'with_ids',
          'node_ids': True,
          'edge_ids': True,
      },
  )
  def test_count_num_nodes(self, node_ids, edge_ids):
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids=node_ids, edge_ids=edge_ids, variable_length=False
    )
    self.assertEqual(in_memory_graph_nb_test_ext.CountNumNodes(graph), 4)

  def test_count_num_edges(self):
    graph = gen_test_graph.generate_in_memory_graph(variable_length=False)
    self.assertEqual(in_memory_graph_nb_test_ext.CountNumEdges(graph), 4)


if __name__ == '__main__':
  absltest.main()
