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
from dgf.src.analyse import in_process_feature_statistics as in_process_feature_statistics_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np

test_util.disable_diff_truncation()


class FeatureStatisticsTest(absltest.TestCase):

  def test_basic(self):
    schema = gen_test_graph.generate_schema(
        node_ids=True, semantic=True, variable_length=True
    )
    graphs = [
        # Each graph contains 2 nodes and 2 edges.
        gen_test_graph.generate_in_memory_graph(
            node_ids=True, variable_length=True
        ),
        gen_test_graph.generate_in_memory_graph(
            node_ids=True, variable_length=True
        ),
        gen_test_graph.generate_in_memory_graph(
            node_ids=True, variable_length=True
        ),
    ]
    stats = in_process_feature_statistics_lib.feature_statistics_from_graphs(
        iter(graphs), schema, num_quantiles=4
    )
    expected_stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "n2": statistics_lib.FeatureSetStatistics(
                features={
                    "#id": statistics_lib.FeatureStatistics(count=6),
                    "f3": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=4,
                        maximum=5,
                        quantiles=[4.0, 4.0, 5.0, 5.0],
                    ),
                    "f4": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=10,
                        maximum=11,
                        quantiles=[10.0000, 10.0000, 11.0000, 11.0000],
                    ),
                    "f5": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=11,
                        maximum=14,
                        quantiles=[11.0, 12.0, 13.0, 14.0],
                    ),
                    "f6": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=11,
                        maximum=20,
                        quantiles=[11.0, 14.0, 17.0, 20.0],
                    ),
                }
            ),
            "n1": statistics_lib.FeatureSetStatistics(
                features={
                    "#id": statistics_lib.FeatureStatistics(count=6),
                    "f2": statistics_lib.FeatureStatistics(
                        count=6,
                    ),
                    "f1": statistics_lib.FeatureStatistics(
                        count=6,
                        dictionary={
                            "blue": statistics_lib.DictionaryItem(
                                index=0, count=3
                            ),
                            "red": statistics_lib.DictionaryItem(
                                index=1, count=3
                            ),
                        },
                    ),
                }
            ),
        }
    )
    test_util.assert_are_equal(self, stats, expected_stats)

  def test_variable_length_feature_without_values(self):
    # A timeseries present on all the nodes, but without any value (e.g., all
    # the values are after the seed timestamps).
    schema = schema_lib.GraphSchema(
        node_sets={
            "n": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        shape=(None,),
                        is_timeseries=True,
                    )
                }
            )
        },
        edge_sets={},
    )
    ts = np.empty(2, dtype=object)
    ts[0] = np.zeros((0,), dtype=np.float32)
    ts[1] = np.zeros((0,), dtype=np.float32)
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2, features={"ts": ts}
            )
        },
        edge_sets={},
    )
    stats = in_process_feature_statistics_lib.feature_statistics_from_graphs(
        iter([graph]), schema, num_quantiles=4
    )
    feature_stats = stats.node_sets["n"].features["ts"]
    self.assertEqual(feature_stats.count, 2)
    self.assertEqual(feature_stats.quantiles, [])
    # Printing the statistics used to fail on scalar NaN quantiles.
    self.assertEqual(str(feature_stats), "count=2")


if __name__ == "__main__":
  absltest.main()
