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
import json
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
import numpy as np

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
                            mean=4.5,
                        ),
                        "f4": statistics_lib.FeatureStatistics(
                            count=2,
                            minimum=10,
                            maximum=11,
                            quantiles=[10.0, 10.3333, 10.6667, 11.0],
                            mean=10.5,
                        ),
                        "f5": statistics_lib.FeatureStatistics(
                            count=2,
                            minimum=11,
                            maximum=14,
                            quantiles=[11.0, 12.0, 12.6667, 14.0],
                            mean=12.4,
                        ),
                        "f6": statistics_lib.FeatureStatistics(
                            count=2,
                            minimum=11,
                            maximum=20,
                            quantiles=[11.0, 14.0, 17.0, 20.0],
                            mean=15.5,
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

  def test_merge_accumulators_exact_mean_and_counters(self):
    schema = {
        "num": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_64,
            semantic=schema_lib.FeatureSemantic.NUMERICAL,
            shape=(),
        ),
        "ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_64,
            semantic=schema_lib.FeatureSemantic.TIMESTAMP,
            shape=(),
        ),
        "emb": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(2,),
        ),
        "pk_float": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_64,
            semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
            shape=(),
        ),
        "pk_int": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
            shape=(),
        ),
    }
    config = feature_statistics.Config(
        max_num_dictionary_items=10,
        min_dictionary_item_frequency=1,
        dictionary_buffer_size=20,
        reservoir_sampling_buffer_size=100,
        num_quantiles=4,
    )

    # Accumulator 1: 1 finite value (100.0) and 3 NaNs for 'num'.
    acc1 = feature_statistics.create_accumulator(schema, config)
    feature_statistics.add_to_accumulator(
        schema,
        config,
        acc1,
        {
            "num": np.array(
                [100.0, np.nan, np.nan, np.nan], dtype=np.float64
            ),
            "ts": np.array(
                [1000.0, np.nan, np.inf, -np.inf], dtype=np.float64
            ),
            "emb": np.array(
                [
                    [1.0, np.nan],
                    [np.inf, -np.inf],
                    [2.0, 3.0],
                    [np.nan, np.nan],
                ],
                dtype=np.float32,
            ),
            "pk_float": np.array([1.0, np.nan, np.inf, 4.0], dtype=np.float64),
            "pk_int": np.array([1, 2, 3, 4], dtype=np.int64),
        },
        batch_values=True,
    )

    # Accumulator 2: 3 finite values (0.0, 0.0, 0.0), 1 +inf, 2 -inf for 'num'.
    # Pooled finite mean for 'num' is (100 + 0 + 0 + 0) / 4 = 25.0 (whereas an
    # unweighted average of partial means (100 + 0) / 2 would be 50.0).
    acc2 = feature_statistics.create_accumulator(schema, config)
    feature_statistics.add_to_accumulator(
        schema,
        config,
        acc2,
        {
            "num": np.array(
                [0.0, 0.0, 0.0, np.inf, -np.inf, -np.inf], dtype=np.float64
            ),
            "ts": np.array(
                [2000.0, 3000.0, np.nan, np.nan, np.inf, np.inf],
                dtype=np.float64,
            ),
            "emb": np.array(
                [
                    [4.0, 5.0],
                    [np.nan, np.inf],
                    [6.0, 7.0],
                    [8.0, 9.0],
                    [10.0, 11.0],
                    [12.0, -np.inf],
                ],
                dtype=np.float32,
            ),
            "pk_float": np.array(
                [5.0, -np.inf, np.nan, 8.0, 9.0, 10.0], dtype=np.float64
            ),
            "pk_int": np.array([5, 6, 7, 8, 9, 10], dtype=np.int64),
        },
        batch_values=True,
    )

    merged = feature_statistics.merge_accumulators(schema, config, [acc1, acc2])
    stats = feature_statistics.accumulator_to_feature_stats(
        schema, config, merged
    )

    expected = statistics_lib.FeatureSetStatistics(
        features={
            "num": statistics_lib.FeatureStatistics(
                count=10,
                minimum=0.0,
                maximum=100.0,
                quantiles=[0.0, 0.0, 0.0, 100.0],
                mean=25.0,
                num_nan=3,
                num_posinf=1,
                num_neginf=2,
            ),
            "ts": statistics_lib.FeatureStatistics(
                count=10,
                minimum=1000.0,
                maximum=3000.0,
                quantiles=[],
                mean=2000.0,
                num_nan=3,
                num_posinf=3,
                num_neginf=1,
            ),
            "emb": statistics_lib.FeatureStatistics(
                count=10,
                minimum=math.nan,
                maximum=math.nan,
                quantiles=[],
                mean=math.nan,
                # EMBEDDING is not in the numerical-stats set: no counters.
                num_nan=0,
                num_posinf=0,
                num_neginf=0,
            ),
            # Primary keys never get numerical stats or non-finite counters,
            # even when stored as floats containing NaN / inf.
            "pk_float": statistics_lib.FeatureStatistics(
                count=10,
                minimum=math.nan,
                maximum=math.nan,
                quantiles=[],
                mean=math.nan,
            ),
            "pk_int": statistics_lib.FeatureStatistics(
                count=10,
                minimum=math.nan,
                maximum=math.nan,
                quantiles=[],
                mean=math.nan,
            ),
        }
    )
    test_util.assert_are_equal(self, stats, expected)

  def test_read_legacy_serialized_statistics(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      legacy_path = os.path.join(tmpdir, "legacy_stats.json")
      legacy_payload = {
          "node_sets": {
              "n": {
                  "features": {
                      "x": {
                          "count": 3,
                          "minimum": 1.0,
                          "maximum": 5.0,
                          "dictionary": {},
                          "quantiles": [1.0, 3.0, 5.0],
                      }
                  }
              }
          }
      }
      with open(legacy_path, "w") as f:
        json.dump(legacy_payload, f)

      loaded = statistics_io_lib.read_feature_statistics(legacy_path)
      expected = statistics_lib.GraphFeatureStatistics(
          node_sets={
              "n": statistics_lib.FeatureSetStatistics(
                  features={
                      "x": statistics_lib.FeatureStatistics(
                          count=3,
                          minimum=1.0,
                          maximum=5.0,
                          dictionary={},
                          quantiles=[1.0, 3.0, 5.0],
                          mean=math.nan,
                          num_nan=0,
                          num_posinf=0,
                          num_neginf=0,
                      )
                  }
              )
          }
      )
      test_util.assert_are_equal(self, loaded, expected)


if __name__ == "__main__":
  absltest.main()
