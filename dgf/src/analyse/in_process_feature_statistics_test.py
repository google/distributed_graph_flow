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

import math

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.analyse import feature_statistics as feature_statistics_lib
from dgf.src.analyse import in_process_feature_statistics as in_process_feature_statistics_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np

test_util.disable_diff_truncation()


class FeatureStatisticsTest(parameterized.TestCase):

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
                        mean=4.5,
                    ),
                    "f4": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=10,
                        maximum=11,
                        quantiles=[10.0000, 10.0000, 11.0000, 11.0000],
                        mean=10.5,
                    ),
                    "f5": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=11,
                        maximum=14,
                        quantiles=[11.0, 12.0, 13.0, 14.0],
                        mean=12.4,
                    ),
                    "f6": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=11,
                        maximum=20,
                        quantiles=[11.0, 14.0, 17.0, 20.0],
                        mean=15.5,
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

  def test_mixed_nan_and_inf(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n": schema_lib.NodeSchema(
                features={
                    "num": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        shape=(),
                    ),
                    "f64": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        shape=(),
                    ),
                    "ts_seq": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.TIMESERIES,
                        shape=(None,),
                        is_timeseries=True,
                    ),
                    "timestamp": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        shape=(),
                    ),
                    "emb": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(2,),
                    ),
                }
            )
        },
        edge_sets={},
    )
    # 1_700_000_000_001.0 and 1_700_000_000_003.0 lose precision if cast to
    # float32 (becoming 1700000055296.0); filtering in source precision keeps
    # exact float64 min, max, and mean.
    high_prec_min = 1_700_000_000_001.0
    high_prec_max = 1_700_000_000_003.0
    high_prec_mean = 1_700_000_000_002.0

    ts_seq = np.empty(6, dtype=object)
    ts_seq[0] = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    ts_seq[1] = np.array([np.nan, np.inf], dtype=np.float32)
    ts_seq[2] = np.array([3.0, -np.inf, 4.0], dtype=np.float32)
    ts_seq[3] = np.array([], dtype=np.float32)
    ts_seq[4] = np.array([np.nan], dtype=np.float32)
    ts_seq[5] = np.array([np.inf, -np.inf], dtype=np.float32)

    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=6,
                features={
                    "num": np.array(
                        [2.0, np.nan, 4.0, np.inf, 6.0, -np.inf],
                        dtype=np.float32,
                    ),
                    "f64": np.array(
                        [
                            high_prec_min,
                            np.nan,
                            high_prec_max,
                            np.inf,
                            -np.inf,
                            np.nan,
                        ],
                        dtype=np.float64,
                    ),
                    "ts_seq": ts_seq,
                    "timestamp": np.array(
                        [100.0, 200.0, np.nan, np.nan, np.inf, 300.0],
                        dtype=np.float64,
                    ),
                    "emb": np.array(
                        [
                            [1.0, 2.0],
                            [np.nan, 3.0],
                            [np.inf, -np.inf],
                            [np.nan, np.nan],
                            [4.0, -np.inf],
                            [5.0, 6.0],
                        ],
                        dtype=np.float32,
                    ),
                },
            )
        },
        edge_sets={},
    )
    stats = in_process_feature_statistics_lib.feature_statistics_from_graphs(
        iter([graph]), schema, num_quantiles=3
    )
    expected_stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "n": statistics_lib.FeatureSetStatistics(
                features={
                    "num": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=2.0,
                        maximum=6.0,
                        quantiles=[2.0, 4.0, 6.0],
                        mean=4.0,
                        num_nan=1,
                        num_posinf=1,
                        num_neginf=1,
                    ),
                    "f64": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=high_prec_min,
                        maximum=high_prec_max,
                        quantiles=[
                            float(np.float32(high_prec_min)),
                            float(np.float32(high_prec_mean)),
                            float(np.float32(high_prec_max)),
                        ],
                        mean=high_prec_mean,
                        num_nan=2,
                        num_posinf=1,
                        num_neginf=1,
                    ),
                    "ts_seq": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=1.0,
                        maximum=4.0,
                        quantiles=[1.0, 2.5, 4.0],
                        mean=2.5,
                        num_nan=3,
                        num_posinf=2,
                        num_neginf=2,
                    ),
                    "timestamp": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=100.0,
                        maximum=300.0,
                        quantiles=[],
                        mean=200.0,
                        num_nan=2,
                        num_posinf=1,
                        num_neginf=0,
                    ),
                    "emb": statistics_lib.FeatureStatistics(
                        count=6,
                        minimum=math.nan,
                        maximum=math.nan,
                        quantiles=[],
                        mean=math.nan,
                        # EMBEDDING is not in the numerical-stats set.
                        num_nan=0,
                        num_posinf=0,
                        num_neginf=0,
                    ),
                }
            )
        }
    )
    test_util.assert_are_equal(self, stats, expected_stats)
    # Exact check that float64 min, max, and mean were not truncated to float32.
    f64_stats = stats.node_sets["n"].features["f64"]
    self.assertEqual(f64_stats.minimum, high_prec_min)
    self.assertEqual(f64_stats.maximum, high_prec_max)
    self.assertEqual(f64_stats.mean, high_prec_mean)

  @parameterized.named_parameters(
      (
          "all_nan",
          np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
          4,
          0,
          0,
      ),
      (
          "all_posinf",
          np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
          0,
          4,
          0,
      ),
      (
          "all_neginf",
          np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
          0,
          0,
          4,
      ),
      (
          "all_mixed_nonfinite",
          np.array([np.nan, np.inf, -np.inf, np.nan], dtype=np.float32),
          2,
          1,
          1,
      ),
  )
  def test_all_nonfinite_column(
      self, values, expected_nan, expected_posinf, expected_neginf
  ):
    feature_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    schema = schema_lib.GraphSchema(
        node_sets={"n": schema_lib.NodeSchema(features={"x": feature_schema})},
        edge_sets={},
    )
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=len(values), features={"x": values}
            )
        },
        edge_sets={},
    )
    stats = in_process_feature_statistics_lib.feature_statistics_from_graphs(
        iter([graph]), schema, num_quantiles=4
    )
    expected_feature_stats = statistics_lib.FeatureStatistics(
        count=len(values),
        minimum=math.inf,
        maximum=-math.inf,
        quantiles=[],
        mean=math.nan,
        num_nan=expected_nan,
        num_posinf=expected_posinf,
        num_neginf=expected_neginf,
    )
    feature_stats = stats.node_sets["n"].features["x"]
    test_util.assert_are_equal(self, feature_stats, expected_feature_stats)

    # Verify SoftQuantileNormalizer.create takes the no_observed_value path.
    normalizer = normalize_lib.SoftQuantileNormalizer.create(
        feature_name="x",
        input_schema=feature_schema,
        input_stats=feature_stats,
    )
    np.testing.assert_array_equal(
        normalizer.quantiles, np.asarray([0, 1], dtype=np.float32)
    )

  def test_sparse_column_reservoir_holds_only_finite_values(self):
    feature_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    schema = schema_lib.GraphSchema(
        node_sets={"n": schema_lib.NodeSchema(features={"x": feature_schema})},
        edge_sets={},
    )
    # 990 NaNs and 10 finite values (1.0 .. 10.0) = 99% NaN.
    # With reservoir_sampling_buffer_size=20, if NaNs entered the reservoir,
    # most of the 20 slots would be occupied by NaNs.
    finite_vals = np.arange(1.0, 11.0, dtype=np.float32)
    nan_vals = np.full(990, np.nan, dtype=np.float32)
    values = np.concatenate([nan_vals[:500], finite_vals, nan_vals[500:]])

    config = feature_statistics_lib.Config(
        max_num_dictionary_items=10,
        min_dictionary_item_frequency=1,
        dictionary_buffer_size=20,
        reservoir_sampling_buffer_size=20,
        num_quantiles=4,
    )
    acc = feature_statistics_lib.create_accumulator(
        {"x": feature_schema}, config
    )
    feature_statistics_lib.add_to_accumulator(
        {"x": feature_schema},
        config,
        acc,
        {"x": values},
        batch_values=True,
    )
    reservoir = acc.features["x"].quantiles
    self.assertIsNotNone(reservoir)
    assert reservoir is not None
    self.assertEqual(reservoir._num_in_cache, 10)
    self.assertEqual(reservoir._num_seen, 10)
    np.testing.assert_array_equal(
        reservoir._samples[: reservoir._num_in_cache], finite_vals
    )

    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=len(values), features={"x": values}
            )
        },
        edge_sets={},
    )
    stats = in_process_feature_statistics_lib.feature_statistics_from_graphs(
        iter([graph]),
        schema,
        num_quantiles=4,
        reservoir_sampling_buffer_size=20,
    )
    expected_stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "n": statistics_lib.FeatureSetStatistics(
                features={
                    "x": statistics_lib.FeatureStatistics(
                        count=1000,
                        minimum=1.0,
                        maximum=10.0,
                        quantiles=[1.0, 4.0, 7.0, 10.0],
                        mean=5.5,
                        num_nan=990,
                        num_posinf=0,
                        num_neginf=0,
                    )
                }
            )
        }
    )
    test_util.assert_are_equal(self, stats, expected_stats)


if __name__ == "__main__":
  absltest.main()
