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
from dgf.src.data import statistics as statistics_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util

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


if __name__ == "__main__":
  absltest.main()
