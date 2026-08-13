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

"""Tests for temporal timeseries filtering in sampling."""

from absl.testing import absltest
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.sampling import temporal
from dgf.src.util import temporal as temporal_util
import numpy as np


def _make_sample_graph_and_schema():
  graph = in_memory_graph.InMemoryGraph(
      node_sets={
          "hardware": in_memory_graph.InMemoryNodeSet(
              num_nodes=1,
              features={
                  "time": np.array(
                      [np.array([10, 20, 30, 40, 50])], dtype=np.object_
                  ),
                  "signal": np.array(
                      [np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)],
                      dtype=np.object_,
                  ),
              },
          )
      },
      edge_sets={},
  )
  schema = schema_lib.GraphSchema(
      node_sets={
          "hardware": schema_lib.NodeSchema(
              features={
                  "time": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                      is_timeseries=True,
                      is_creation_time=True,
                      group="time",
                  ),
                  "signal": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                      is_timeseries=True,
                      group="time",
                  ),
              }
          )
      },
      edge_sets={},
  )
  return graph, schema


class TemporalTest(absltest.TestCase):

  def test_filter_timeseries_by_timestamp_scalar(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "hardware": in_memory_graph.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "time": np.array(
                        [
                            np.array([10, 20, 30, 40, 50]),
                            np.array([5, 15, 25, 35]),
                        ],
                        dtype=np.object_,
                    ),
                    "signal": np.array(
                        [
                            np.array(
                                [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
                            ),
                            np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
                        ],
                        dtype=np.object_,
                    ),
                },
            )
        },
        edge_sets={},
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "hardware": schema_lib.NodeSchema(
                features={
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        group="time",
                    ),
                    "signal": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        group="time",
                    ),
                }
            )
        },
        edge_sets={},
    )
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    temporal.extract_features_timeseries(
        graph=graph,
        schema_cache=cache,
        target_timestamp=28,
        max_timeseries_len=2,
    )
    hw_node_set = graph.node_sets["hardware"]
    # Node 0: original <= 28 is [10, 20], max_timeseries_len=2 -> [10, 20]
    np.testing.assert_array_equal(hw_node_set.features["time"][0], [10, 20])
    np.testing.assert_array_equal(hw_node_set.features["signal"][0], [1.0, 2.0])
    # Node 1: original <= 28 is [5, 15, 25], max_timeseries_len=2 -> [15, 25]
    np.testing.assert_array_equal(hw_node_set.features["time"][1], [15, 25])
    np.testing.assert_array_equal(hw_node_set.features["signal"][1], [1.5, 2.5])

  def test_filter_timeseries_skips_non_timestamp_series(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "hardware": in_memory_graph.InMemoryNodeSet(
                num_nodes=1,
                features={
                    "time": np.array(
                        [np.array([10, 20, 30, 40])], dtype=np.object_
                    ),
                    "signal": np.array(
                        [np.array([1.0, 2.0, 3.0, 4.0])], dtype=np.object_
                    ),
                    "waveform": np.array(
                        [np.array([0.1, 0.2, 0.3, 0.4])], dtype=np.object_
                    ),
                },
            )
        },
        edge_sets={},
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "hardware": schema_lib.NodeSchema(
                features={
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        group="time",
                    ),
                    "signal": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        group="time",
                    ),
                    "waveform": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                    ),
                }
            )
        },
        edge_sets={},
    )
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    temporal.extract_features_timeseries(
        graph=graph,
        schema_cache=cache,
        target_timestamp=25,
        max_timeseries_len=32,
    )
    hw_set = graph.node_sets["hardware"]
    np.testing.assert_array_equal(hw_set.features["time"][0], [10, 20])
    np.testing.assert_array_equal(hw_set.features["signal"][0], [1.0, 2.0])
    # waveform has is_timeseries=True but does not point to a timestamp series.
    np.testing.assert_array_equal(
        hw_set.features["waveform"][0], [0.1, 0.2, 0.3, 0.4]
    )

  def test_filter_timeseries_edge_set(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "target_node": in_memory_graph.InMemoryNodeSet(
                num_nodes=1, features={}
            ),
            "source_node": in_memory_graph.InMemoryNodeSet(
                num_nodes=1, features={}
            ),
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0], [0]], dtype=np.int64),
                features={
                    "timestamps": np.array(
                        [np.array([10, 20, 30, 40])], dtype=np.object_
                    ),
                    "edge_sig": np.array(
                        [np.array([1.0, 2.0, 3.0, 4.0])], dtype=np.object_
                    ),
                },
            )
        },
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "target_node": schema_lib.NodeSchema(features={}),
            "source_node": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="source_node",
                target="target_node",
                features={
                    "timestamps": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        group="timestamps",
                    ),
                    "edge_sig": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        group="timestamps",
                    ),
                },
            )
        },
    )
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    temporal.extract_features_timeseries(
        graph=graph,
        schema_cache=cache,
        target_timestamp=20,
        max_timeseries_len=32,
    )
    e1_set = graph.edge_sets["e1"]
    np.testing.assert_array_equal(e1_set.features["timestamps"][0], [10, 20])
    np.testing.assert_array_equal(e1_set.features["edge_sig"][0], [1.0, 2.0])

  def test_filter_timeseries_skips_missing_node_set(self):
    graph = in_memory_graph.InMemoryGraph(node_sets={}, edge_sets={})
    missing_cache = temporal_util.TimeseriesSchemaCache(
        node_sets={
            "missing_node": [
                temporal_util.TimeseriesGroupSpec(
                    timestamp_feature_name="t", feature_names=["t"]
                )
            ]
        },
        edge_sets={},
        has_timeseries=True,
    )
    # Should not raise ValueError for missing node set
    temporal.extract_features_timeseries(
        graph=graph,
        schema_cache=missing_cache,
        target_timestamp=25,
        max_timeseries_len=32,
    )

  def test_extract_timeseries_schema_cache_and_filter(self):
    graph, schema = _make_sample_graph_and_schema()
    cache = temporal_util.extract_timeseries_schema_cache(schema)

    self.assertIn("hardware", cache.node_sets)
    self.assertLen(cache.node_sets["hardware"], 1)
    group = cache.node_sets["hardware"][0]
    self.assertEqual(group.timestamp_feature_name, "time")
    self.assertCountEqual(group.feature_names, ["time", "signal"])

    temporal.extract_features_timeseries(
        graph=graph,
        schema_cache=cache,
        target_timestamp=30,
        max_timeseries_len=32,
    )
    np.testing.assert_array_equal(
        graph.node_sets["hardware"].features["time"][0],
        [10, 20, 30],
    )
    np.testing.assert_array_equal(
        graph.node_sets["hardware"].features["signal"][0],
        [1.0, 2.0, 3.0],
    )

  def test_filter_timeseries_clips_non_timestamp_series(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=1,
                features={
                    "f1": np.array(
                        [np.array([10, 20, 30, 40])], dtype=np.object_
                    ),
                },
            )
        },
        edge_sets={},
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                    ),
                }
            )
        },
        edge_sets={},
    )
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    temporal.extract_features_timeseries(
        graph=graph,
        schema_cache=cache,
        target_timestamp=25,
        max_timeseries_len=2,
    )
    np.testing.assert_array_equal(
        graph.node_sets["n1"].features["f1"][0], [30, 40]
    )

  def test_clip_timeseries_to_max_len(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "f1": np.array(
                        [
                            np.array([10, 20, 30, 40, 50]),
                            np.array([1, 2, 3]),
                        ],
                        dtype=np.object_,
                    ),
                    "f2": np.array(
                        [
                            [1.0, 2.0, 3.0, 4.0, 5.0],
                            [6.0, 7.0, 8.0, 9.0, 10.0],
                        ],
                        dtype=np.float32,
                    ),
                },
            )
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0], [1]], dtype=np.int64),
                features={
                    "f1": np.array(
                        [np.array([100, 200, 300, 400])], dtype=np.object_
                    ),
                },
            )
        },
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                    ),
                    "f2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        is_timeseries=True,
                        shape=(5,),
                    ),
                }
            )
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                    )
                },
            )
        },
    )
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    temporal.extract_features_timeseries(
        graph=graph,
        schema_cache=cache,
        max_timeseries_len=3,
        target_timestamp=None,
    )

    n1 = graph.node_sets["n1"]
    np.testing.assert_array_equal(n1.features["f1"][0], [30, 40, 50])
    np.testing.assert_array_equal(n1.features["f1"][1], [1, 2, 3])
    np.testing.assert_array_equal(
        n1.features["f2"],
        np.array([[3.0, 4.0, 5.0], [8.0, 9.0, 10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        graph.edge_sets["e1"].features["f1"][0], [200, 300, 400]
    )

  def test_clip_timeseries_empty_entities(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=0,
                features={"f1": np.array([], dtype=np.object_)},
            )
        },
        edge_sets={},
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                    ),
                }
            )
        },
        edge_sets={},
    )
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    temporal.extract_features_timeseries(
        graph=graph,
        schema_cache=cache,
        max_timeseries_len=3,
        target_timestamp=None,
    )

  def test_invalid_max_timeseries_len_raises(self):
    graph, schema = _make_sample_graph_and_schema()
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    with self.assertRaisesRegex(
        ValueError, "max_timeseries_len must be positive"
    ):
      temporal.extract_features_timeseries(
          graph=graph, schema_cache=cache, max_timeseries_len=0
      )
    with self.assertRaisesRegex(
        ValueError, "max_timeseries_len must be positive"
    ):
      temporal._compute_group_slices(
          timestamp_values=np.array([10, 20]),
          node_idxs=np.array([0]),
          max_timeseries_len=0,
      )

  def test_compute_group_slices_causal_object_array(self):
    ts_values = np.array(
        [
            np.array([10, 20, 30, 40, 50]),
            np.array([5, 15, 25, 35]),
        ],
        dtype=np.object_,
    )
    node_idxs = np.array([0, 1])
    starts, ends = temporal._compute_group_slices(
        timestamp_values=ts_values,
        node_idxs=node_idxs,
        target_timestamp=28,
        max_timeseries_len=2,
    )
    np.testing.assert_array_equal(starts, np.array([0, 1]))
    np.testing.assert_array_equal(ends, np.array([2, 3]))

  def test_compute_group_slices_causal_dense_array(self):
    ts_values = np.array(
        [[10, 20, 30, 40], [5, 15, 25, 35]],
        dtype=np.int64,
    )
    node_idxs = np.array([1, 0])
    starts, ends = temporal._compute_group_slices(
        timestamp_values=ts_values,
        node_idxs=node_idxs,
        max_timeseries_len=4,
        target_timestamp=20,
    )
    np.testing.assert_array_equal(starts, np.array([0, 0]))
    np.testing.assert_array_equal(ends, np.array([2, 2]))

  def test_compute_group_slices_no_target_timestamp(self):
    ts_values = np.array(
        [np.array([10, 20, 30, 40, 50]), np.array([5, 15, 25])],
        dtype=np.object_,
    )
    starts, ends = temporal._compute_group_slices(
        timestamp_values=ts_values,
        node_idxs=np.array([0, 1]),
        max_timeseries_len=2,
        target_timestamp=None,
    )
    np.testing.assert_array_equal(starts, np.array([3, 1]))
    np.testing.assert_array_equal(ends, np.array([5, 3]))

  def test_compute_group_slices_empty_node_idxs(self):
    ts_values = np.array([np.array([10, 20])], dtype=np.object_)
    empty_idxs = np.empty(0, dtype=np.int64)
    starts, ends = temporal._compute_group_slices(
        timestamp_values=ts_values,
        node_idxs=empty_idxs,
        max_timeseries_len=5,
        target_timestamp=20,
    )
    self.assertEmpty(starts)
    self.assertEmpty(ends)

  def test_crop_timeseries_per_entity_slice_plan(self):
    ts_values = np.array(
        [
            np.array([10, 20, 30, 40, 50]),
            np.array([5, 15, 25, 35]),
        ],
        dtype=np.object_,
    )
    values = np.array(
        [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0.5, 1.5, 2.5, 3.5]),
        ],
        dtype=np.object_,
    )
    starts, ends = temporal._compute_group_slices(
        timestamp_values=ts_values,
        node_idxs=np.array([0, 1]),
        target_timestamp=28,
        max_timeseries_len=2,
    )
    extracted = temporal._crop_timeseries(
        feature_values=values,
        node_idxs=np.array([0, 1]),
        start_indices=starts,
        end_indices=ends,
    )
    self.assertEqual(extracted.dtype, np.object_)
    np.testing.assert_array_equal(extracted[0], [1.0, 2.0])
    np.testing.assert_array_equal(extracted[1], [1.5, 2.5])

  def test_extract_features_timeseries_from_source_graph(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "node_id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    ),
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                    ),
                }
            )
        },
        edge_sets={},
    )
    source_graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=3,
                features={
                    "node_id": np.array([100, 101, 102], dtype=np.int64),
                    "f1": np.array(
                        [
                            np.array([1, 2, 3, 4, 5]),
                            np.array([10, 20, 30]),
                            np.array([100, 200, 300, 400]),
                        ],
                        dtype=np.object_,
                    ),
                },
            )
        },
        edge_sets={},
    )
    sampled_graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=2,
                features={"#idx": np.array([2, 0], dtype=np.int64)},
            )
        },
        edge_sets={},
    )
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    temporal.extract_features_timeseries(
        graph=sampled_graph,
        source_graph=source_graph,
        schema_cache=cache,
        max_timeseries_len=2,
        target_timestamp=None,
    )

    n1 = sampled_graph.node_sets["n1"]
    np.testing.assert_array_equal(n1.features["node_id"], [102, 100])
    np.testing.assert_array_equal(n1.features["f1"][0], [300, 400])
    np.testing.assert_array_equal(n1.features["f1"][1], [4, 5])


if __name__ == "__main__":
  absltest.main()

