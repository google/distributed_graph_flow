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

"""Tests for graph filtering."""

from absl.testing import absltest
from dgf.src.data import in_memory_graph
from dgf.src.transform import in_memory_graph_filter
import numpy as np

InMemoryGraph = in_memory_graph.InMemoryGraph
InMemoryNodeSet = in_memory_graph.InMemoryNodeSet
NumNodesPredicate = in_memory_graph_filter.NumNodesPredicate
filter_graphs = in_memory_graph_filter.filter_graphs


class InMemoryGraphFilterTest(absltest.TestCase):

  def test_num_nodes_basic(self):
    graphs = []
    graphs.append(
        InMemoryGraph(
            node_sets={"entity": InMemoryNodeSet(num_nodes=10)}, edge_sets={}
        )
    )
    graphs.append(
        InMemoryGraph(
            node_sets={"entity": InMemoryNodeSet(num_nodes=11)}, edge_sets={}
        )
    )
    graphs.append(
        InMemoryGraph(
            node_sets={"entity": InMemoryNodeSet(num_nodes=2)}, edge_sets={}
        )
    )
    graphs.append(
        InMemoryGraph(
            node_sets={"entity": InMemoryNodeSet(num_nodes=128)}, edge_sets={}
        )
    )
    graphs.append(
        InMemoryGraph(
            node_sets={"entity": InMemoryNodeSet(num_nodes=256)}, edge_sets={}
        )
    )
    graphs.append(
        InMemoryGraph(
            node_sets={"entity": InMemoryNodeSet(num_nodes=512)}, edge_sets={}
        )
    )

    predicates = (
        NumNodesPredicate(upper=5),
        NumNodesPredicate(lower=5, upper=128),
        NumNodesPredicate(lower=128, upper=512),
        NumNodesPredicate(lower=512),
    )

    filtered_results = filter_graphs(graphs, predicates)

    np.testing.assert_allclose([len(r) for r in filtered_results], [1, 2, 2, 1])

    def _num_nodes(graph):
      return graph.node_sets["entity"].num_nodes

    # Same element without order
    self.assertCountEqual([_num_nodes(g) for g in filtered_results[0]], [2])
    self.assertCountEqual(
        [_num_nodes(g) for g in filtered_results[1]], [10, 11]
    )
    self.assertCountEqual(
        [_num_nodes(g) for g in filtered_results[2]], [128, 256]
    )
    self.assertCountEqual([_num_nodes(g) for g in filtered_results[3]], [512])


if __name__ == "__main__":
  absltest.main()
