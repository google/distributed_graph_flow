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

"""Tests for topology_statistics."""

from absl.testing import absltest
from dgf.src.analyse import topology_statistics as topo_stats_lib
from dgf.src.data import histogram as histogram_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util

test_util.disable_diff_truncation()


class TopologyStatisticsTest(absltest.TestCase):

  def test_topology_statistics(self):
    schema = gen_test_graph.generate_schema()
    graph = gen_test_graph.generate_in_memory_graph()

    stats = topo_stats_lib.topology_statistics(graph, schema, num_bins=2)

    expected_stats = statistics_lib.GraphTopologyStatistics(
        node_sets={
            "n1": statistics_lib.NodeSetTopologyStatistics(
                num_nodes=histogram_lib.Histogram(values=[1.0], bins=[2.0, 2.0])
            ),
            "n2": statistics_lib.NodeSetTopologyStatistics(
                num_nodes=histogram_lib.Histogram(values=[1.0], bins=[2.0, 2.0])
            ),
        },
        edge_sets={
            "e1": statistics_lib.EdgeSetTopologyStatistics(
                num_edges=histogram_lib.Histogram(
                    values=[1.0], bins=[2.0, 2.0]
                ),
                in_degree_distribution=histogram_lib.Histogram(
                    values=[2.0], bins=[1.0, 1.0]
                ),
                out_degree_distribution=histogram_lib.Histogram(
                    values=[1.0, 1.0], bins=[0.0, 1.0, 2.0]
                ),
            ),
            "e2": statistics_lib.EdgeSetTopologyStatistics(
                num_edges=histogram_lib.Histogram(
                    values=[1.0], bins=[2.0, 2.0]
                ),
                in_degree_distribution=histogram_lib.Histogram(
                    values=[2.0], bins=[1.0, 1.0]
                ),
                out_degree_distribution=histogram_lib.Histogram(
                    values=[1.0, 1.0], bins=[0.0, 1.0, 2.0]
                ),
            ),
        },
    )

    test_util.assert_are_equal(self, stats, expected_stats)

  def test_topology_statistics_from_graphs(self):
    schema = gen_test_graph.generate_schema()
    graph1 = gen_test_graph.generate_in_memory_graph()
    graph2 = gen_test_graph.generate_in_memory_graph()

    stats = topo_stats_lib.topology_statistics_from_graphs(
        [graph1, graph2], schema, num_bins=2
    )

    expected_stats = statistics_lib.GraphTopologyStatistics(
        node_sets={
            "n1": statistics_lib.NodeSetTopologyStatistics(
                num_nodes=histogram_lib.Histogram(values=[2.0], bins=[2.0, 2.0])
            ),
            "n2": statistics_lib.NodeSetTopologyStatistics(
                num_nodes=histogram_lib.Histogram(values=[2.0], bins=[2.0, 2.0])
            ),
        },
        edge_sets={
            "e1": statistics_lib.EdgeSetTopologyStatistics(
                num_edges=histogram_lib.Histogram(
                    values=[2.0], bins=[2.0, 2.0]
                ),
                in_degree_distribution=histogram_lib.Histogram(
                    values=[4.0], bins=[1.0, 1.0]
                ),
                out_degree_distribution=histogram_lib.Histogram(
                    values=[2.0, 2.0], bins=[0.0, 1.0, 2.0]
                ),
            ),
            "e2": statistics_lib.EdgeSetTopologyStatistics(
                num_edges=histogram_lib.Histogram(
                    values=[2.0], bins=[2.0, 2.0]
                ),
                in_degree_distribution=histogram_lib.Histogram(
                    values=[4.0], bins=[1.0, 1.0]
                ),
                out_degree_distribution=histogram_lib.Histogram(
                    values=[2.0, 2.0], bins=[0.0, 1.0, 2.0]
                ),
            ),
        },
        num_graphs=2,
    )

    test_util.assert_are_equal(self, stats, expected_stats)


if __name__ == "__main__":
  absltest.main()
