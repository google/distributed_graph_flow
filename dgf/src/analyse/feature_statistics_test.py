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

import functools
import math
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
from dgf.src.analyse import feature_statistics
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam
from dgf.src.io import statistics as statistics_io_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util

test_util.disable_diff_truncation()


class StatisticsTest(parameterized.TestCase):

  def test_feature_statistics(self):
    with tempfile.TemporaryDirectory() as tmpdir:

      # Generate some toy data
      path = os.path.join(tmpdir, "hgraph")
      gen_test_graph.generate_gf_graph(
          path, edge_ids=False, variable_length=True
      )

      with test_pipeline.TestPipeline() as p:
        hgraph = gf_graph_in_beam.read_graph(p, path)

        expected_stats = statistics_lib.GraphFeatureStatistics(
            node_sets={
                "n1": statistics_lib.FeatureSetStatistics(
                    features={
                        "#id": statistics_lib.FeatureStatistics(count=2),
                        "f1": statistics_lib.FeatureStatistics(
                            count=2,
                            dictionary={
                                "blue": statistics_lib.DictionaryItem(
                                    index=0, count=1
                                ),
                                "red": statistics_lib.DictionaryItem(
                                    index=1, count=1
                                ),
                            },
                            quantiles=[],
                        ),
                        "f2": statistics_lib.FeatureStatistics(
                            count=2,
                        ),
                    }
                ),
                "n2": statistics_lib.FeatureSetStatistics(
                    features={
                        "#id": statistics_lib.FeatureStatistics(count=2),
                        "f3": statistics_lib.FeatureStatistics(
                            count=2,
                            minimum=4,
                            maximum=5,
                            quantiles=[4.0, 4.3333, 4.6666, 5.0],
                        ),
                        "f4": statistics_lib.FeatureStatistics(
                            count=2,
                            minimum=10,
                            maximum=11,
                            quantiles=[10.0, 10.3333, 10.6667, 11.0],
                        ),
                        "f5": statistics_lib.FeatureStatistics(
                            count=2,
                            minimum=11,
                            maximum=14,
                            quantiles=[11.0, 12.0, 12.6667, 14.0],
                        ),
                    }
                ),
            }
        )

        pstats = feature_statistics.feature_statistics(hgraph, num_quantiles=4)
        util.assert_that(
            pstats,
            util.equal_to(
                [expected_stats],
                equals_fn=functools.partial(test_util.are_equal, abs_tol=0.001),
            ),
        )

        stats_path = os.path.join(tmpdir, "stats.json")
        statistics_io_lib.write_feature_statistics_beam(pstats, stats_path)

      stats = statistics_io_lib.read_feature_statistics(stats_path)
      test_util.assert_are_equal(self, stats, expected_stats, abs_tol=0.01)

      # Re save the stats.
      stats_path_v2 = os.path.join(tmpdir, "stats_v2.json")
      statistics_io_lib.write_feature_statistics(stats, stats_path_v2)
      reloaded_stats = statistics_io_lib.read_feature_statistics(stats_path_v2)
      test_util.assert_are_equal(self, stats, reloaded_stats, abs_tol=0.01)

  def test_finalize_dictionary(self):
    config = feature_statistics.Config(
        max_num_dictionary_items=2,
        min_dictionary_item_frequency=2,
        dictionary_buffer_size=10,
        reservoir_sampling_buffer_size=1000,
        num_quantiles=10,
    )
    input_dict = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 1,
    }
    final_dict = feature_statistics.finalize_dictionary(input_dict, config)
    self.assertEqual(
        final_dict,
        {
            "b": statistics_lib.DictionaryItem(1, 2),
            "c": statistics_lib.DictionaryItem(0, 3),
        },
    )

  def test_finalize_dictionary_keep_all(self):
    config = feature_statistics.Config(
        max_num_dictionary_items=5,
        min_dictionary_item_frequency=1,
        dictionary_buffer_size=10,
        reservoir_sampling_buffer_size=1000,
        num_quantiles=10,
    )
    input_dict = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 1,
    }
    final_dict = feature_statistics.finalize_dictionary(input_dict, config)
    self.assertEqual(
        final_dict,
        {
            "c": statistics_lib.DictionaryItem(0, 3),
            "b": statistics_lib.DictionaryItem(1, 2),
            "a": statistics_lib.DictionaryItem(2, 1),
            "d": statistics_lib.DictionaryItem(3, 1),
        },
    )

  def test_prune_dictionary_before_wiring(self):
    config = feature_statistics.Config(
        max_num_dictionary_items=10,
        min_dictionary_item_frequency=10,
        dictionary_buffer_size=2,
        reservoir_sampling_buffer_size=1000,
        num_quantiles=10,
    )
    input_dict = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 1,
    }
    final_dict = feature_statistics.prune_dictionary_before_wiring(
        input_dict, config
    )
    self.assertEqual(final_dict, {"b": 2, "c": 3})

  @parameterized.parameters(
      (schema_lib.FeatureSemantic.NUMERICAL, True),
      (schema_lib.FeatureSemantic.CATEGORICAL, True),
      (schema_lib.FeatureSemantic.EMBEDDING, True),
      (schema_lib.FeatureSemantic.PRIMARY_ID, True),
      (schema_lib.FeatureSemantic.TIMESERIES, True),
      (schema_lib.FeatureSemantic.TIMESTAMP, True),
      (schema_lib.FeatureSemantic.UNKNOWN, False),
      (schema_lib.FeatureSemantic.MASK, False),
  )
  def test_filter_feature_schema_by_semantic(self, semantic, should_keep):
    schema = {
        "#f1": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32, semantic=semantic
        )
    }
    filtered_schema = feature_statistics.filter_feature_schema(schema)
    if should_keep:
      self.assertEqual(filtered_schema, schema)
    else:
      self.assertEqual(filtered_schema, {})

  def test_filter_feature_schema_empty(self):
    self.assertEqual(feature_statistics.filter_feature_schema({}), {})


if __name__ == "__main__":
  absltest.main()
