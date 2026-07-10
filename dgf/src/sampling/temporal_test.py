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
                  ),
                  "signal": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                      is_timeseries=True,
                      timestamps="time",
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
                    ),
                    "signal": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        timestamps="time",
                    ),
                }
            )
        },
        edge_sets={},
    )
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    temporal.filter_timeseries_by_timestamp(
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
                    ),
                    "signal": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        timestamps="time",
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
    temporal.filter_timeseries_by_timestamp(
        graph=graph, schema_cache=cache, target_timestamp=25
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
                    ),
                    "edge_sig": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        timestamps="timestamps",
                    ),
                },
            )
        },
    )
    cache = temporal_util.extract_timeseries_schema_cache(schema)
    temporal.filter_timeseries_by_timestamp(
        graph=graph,
        schema_cache=cache,
        target_timestamp=20,
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
    temporal.filter_timeseries_by_timestamp(
        graph=graph,
        schema_cache=missing_cache,
        target_timestamp=25,
    )

  def test_extract_timeseries_schema_cache_and_filter(self):
    graph, schema = _make_sample_graph_and_schema()
    cache = temporal_util.extract_timeseries_schema_cache(schema)

    self.assertIn("hardware", cache.node_sets)
    self.assertLen(cache.node_sets["hardware"], 1)
    group = cache.node_sets["hardware"][0]
    self.assertEqual(group.timestamp_feature_name, "time")
    self.assertCountEqual(group.feature_names, ["time", "signal"])

    temporal.filter_timeseries_by_timestamp(
        graph=graph, schema_cache=cache, target_timestamp=30
    )
    np.testing.assert_array_equal(
        graph.node_sets["hardware"].features["time"][0],
        [10, 20, 30],
    )
    np.testing.assert_array_equal(
        graph.node_sets["hardware"].features["signal"][0],
        [1.0, 2.0, 3.0],
    )


if __name__ == "__main__":
  absltest.main()
