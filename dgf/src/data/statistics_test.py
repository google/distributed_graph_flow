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

"""Tests for statistics."""

import math
from absl.testing import absltest
from dgf.src.data import histogram
from dgf.src.data import statistics as statistics_lib
from dgf.src.util import test_util

test_util.disable_diff_truncation()


class StatisticsTest(absltest.TestCase):

  def test_str(self):
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "n1": statistics_lib.FeatureSetStatistics(
                features={
                    "f1": statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=math.inf,
                        maximum=-math.inf,
                        dictionary={
                            "red": statistics_lib.DictionaryItem(
                                index=0, count=1
                            ),
                            "blue": statistics_lib.DictionaryItem(
                                index=1, count=1
                            ),
                        },
                        quantiles=[],
                    ),
                    "f2": statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=0.0,
                        maximum=3.0,
                        dictionary={},
                        quantiles=[0.0, 1.0, 2.0, 3.0],
                    ),
                }
            ),
            "n2": statistics_lib.FeatureSetStatistics(
                features={
                    "f3": statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=4,
                        maximum=5,
                        dictionary={},
                        quantiles=[4.0, 4.3333, 4.6666, 5.0],
                    )
                }
            ),
        }
    )
    self.assertEqual(
        str(stats),
        """\
GraphFeatureStatistics:
  Node Sets (2):
    'n1':
      'f1': count=2, dictionary=(2)['red': 1, 'blue': 1]
      'f2': count=2, min=0.0000, max=3.0000, quantiles=(4)[0.0000, 1.0000, 2.0000, 3.0000]
    'n2':
      'f3': count=2, min=4.0000, max=5.0000, quantiles=(4)[4.0000, 4.3333, 4.6666, 5.0000]
""",
    )

  def test_topology_statistics_repr(self):
    hist1 = histogram.Histogram(values=[1.0, 2.0], bins=[0.0, 1.0, 2.0])
    single_hist_nodes = histogram.Histogram(values=[1.0], bins=[10.0, 10.0])
    single_hist_edges = histogram.Histogram(values=[1.0], bins=[30.0, 30.0])

    node_sets = {
        "nodes1": statistics_lib.NodeSetTopologyStatistics(
            num_nodes=single_hist_nodes
        ),
    }
    edge_sets = {
        "edges1": statistics_lib.EdgeSetTopologyStatistics(
            num_edges=single_hist_edges,
            in_degree_distribution=hist1,
            out_degree_distribution=hist1,
        )
    }
    stats = statistics_lib.GraphTopologyStatistics(
        node_sets=node_sets, edge_sets=edge_sets
    )

    expected_repr = """\
GraphTopologyStatistics (num_graphs=1):
  Node Sets:
    'nodes1':
      num_nodes:
        10 (a single value)
  Edge Sets:
    'edges1':
      num_edges:
        30 (a single value)
      in_degree:
        Count:3 Average:1.1666667 StdDev:0.47140452 Min:0 Max:2
        ----------------------------------
        [0, 1) 1 33.33%  33.33% #####
        [1, 2] 2 66.67% 100.00% ##########
      out_degree:
        Count:3 Average:1.1666667 StdDev:0.47140452 Min:0 Max:2
        ----------------------------------
        [0, 1) 1 33.33%  33.33% #####
        [1, 2] 2 66.67% 100.00% ##########
"""
    self.assertEqual(repr(stats), expected_repr)

  def test_topology_statistics_repr_empty(self):
    stats = statistics_lib.GraphTopologyStatistics(node_sets={}, edge_sets={})
    expected_repr = """\
GraphTopologyStatistics (num_graphs=1):
  Node Sets:
    empty
  Edge Sets:
    empty
"""
    self.assertEqual(repr(stats), expected_repr)


if __name__ == "__main__":
  absltest.main()
