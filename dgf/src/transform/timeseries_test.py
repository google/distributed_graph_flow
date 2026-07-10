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

"""Tests for padding and capping timeseries sequence features."""

from typing import Any
from absl.testing import absltest
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.transform import timeseries
import numpy as np


def _make_graph_and_schema(
    features: dict[str, Any],
    feature_schemas: dict[str, schema_lib.FeatureSchema],
    node_set_name: str = "hardware",
    num_nodes: int = 1,
) -> tuple[in_memory_graph.InMemoryGraph, schema_lib.GraphSchema]:
  np_features = {}
  for k, v in features.items():
    if isinstance(v, np.ndarray):
      np_features[k] = v
    else:
      try:
        np_features[k] = np.asarray(v)
      except ValueError:
        np_features[k] = np.asarray(v, dtype=np.object_)
  return (
      in_memory_graph.InMemoryGraph(
          node_sets={
              node_set_name: in_memory_graph.InMemoryNodeSet(
                  num_nodes=num_nodes, features=np_features
              )
          },
          edge_sets={},
      ),
      schema_lib.GraphSchema(
          node_sets={
              node_set_name: schema_lib.NodeSchema(features=feature_schemas)
          },
          edge_sets={},
      ),
  )


def _ts_schema(
    fmt: schema_lib.FeatureFormat = schema_lib.FeatureFormat.FLOAT_32,
    sem: schema_lib.FeatureSemantic = schema_lib.FeatureSemantic.NUMERICAL,
    timestamps: str | None = None,
    shape: schema_lib.Shape = None,
) -> schema_lib.FeatureSchema:
  return schema_lib.FeatureSchema(
      format=fmt,
      semantic=sem,
      is_timeseries=True,
      timestamps=timestamps,
      shape=shape,
  )


class TimeseriesTest(absltest.TestCase):

  def test_capping_and_padding(self):
    # 'time' is np.ndarray, 'signal' is raw Python list (tests both input types)
    graph, schema = _make_graph_and_schema(
        features={
            "time": np.array(
                [np.array([10, 20, 30, 40, 50]), np.array([5, 15])],
                dtype=np.object_,
            ),
            "signal": [[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.5]],
            "id": np.array([101, 102]),
        },
        feature_schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
            ),
            "signal": _ts_schema(timestamps="time"),
            "id": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.INTEGER_64,
                semantic=schema_lib.FeatureSemantic.NUMERICAL,
            ),
        },
        num_nodes=2,
    )
    new_graph, new_schema = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(sequence_length=3),
    )
    hw_val = new_graph.node_sets["hardware"]
    hw_sch = new_schema.node_sets["hardware"]

    # Node 0 (5 steps > K=3): capped to last 3 steps [30, 40, 50]
    np.testing.assert_array_equal(hw_val.features["time"][0], [30, 40, 50])
    np.testing.assert_array_equal(hw_val.features["signal"][0], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(hw_val.features["time_mask"][0], [1, 1, 1])

    # Node 1 (2 steps < K=3): left-padded with 0 to [0, 5, 15]
    np.testing.assert_array_equal(hw_val.features["time"][1], [0, 5, 15])
    np.testing.assert_array_equal(hw_val.features["signal"][1], [0.0, 0.5, 1.5])
    np.testing.assert_array_equal(hw_val.features["time_mask"][1], [0, 1, 1])

    # Static non-timeseries feature preserved unchanged
    np.testing.assert_array_equal(hw_val.features["id"], [101, 102])

    self.assertEqual(hw_sch.features["time"].shape, (3,))
    self.assertTrue(hw_sch.features["time"].is_timeseries)
    self.assertTrue(hw_sch.features["time_mask"].is_timeseries)

  def test_edge_sets_and_non_timeseries(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "user": in_memory_graph.InMemoryNodeSet(
                num_nodes=1, features={"age": np.array([30], dtype=np.int64)}
            ),
        },
        edge_sets={
            "clicks": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0], [0]], dtype=np.int64),
                features={
                    "time": np.array([np.array([100, 200])], dtype=np.object_)
                },
            ),
            "static_edge": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0], [0]], dtype=np.int64),
                features={"weight": np.array([1.0], dtype=np.float32)},
            ),
        },
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "user": schema_lib.NodeSchema(
                features={
                    "age": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                    )
                }
            )
        },
        edge_sets={
            "clicks": schema_lib.EdgeSchema(
                source="user",
                target="user",
                features={
                    "time": _ts_schema(
                        fmt=schema_lib.FeatureFormat.INTEGER_64,
                        sem=schema_lib.FeatureSemantic.TIMESTAMP,
                    )
                },
            ),
            "static_edge": schema_lib.EdgeSchema(
                source="user",
                target="user",
                features={
                    "weight": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                    )
                },
            ),
        },
    )
    new_graph, new_schema = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(sequence_length=2),
    )

    np.testing.assert_array_equal(
        new_graph.node_sets["user"].features["age"], [30]
    )
    np.testing.assert_array_equal(
        new_graph.edge_sets["clicks"].features["time"][0], [100, 200]
    )
    self.assertTrue(
        new_schema.edge_sets["clicks"].features["time"].is_timeseries
    )

  def test_multidimensional_sequence(self):
    graph, schema = _make_graph_and_schema(
        features={
            "emb": np.array(
                [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
                dtype=np.object_,
            )
        },
        feature_schemas={
            "emb": _ts_schema(
                sem=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(None, 2),
            )
        },
    )
    new_graph, new_schema = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(sequence_length=3),
    )
    hw_val = new_graph.node_sets["hardware"]
    hw_sch = new_schema.node_sets["hardware"]

    expected = np.array(
        [[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]], dtype=np.float32
    )
    np.testing.assert_array_equal(hw_val.features["emb"], expected)
    expected_mask = np.array(
        [[[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]], dtype=np.float32
    )
    np.testing.assert_array_equal(hw_val.features["emb_mask"], expected_mask)
    self.assertEqual(hw_sch.features["emb"].shape, (3, 2))
    self.assertEqual(hw_sch.features["emb_mask"].shape, (3, 2))

  def test_empty_sequence(self):
    graph, schema = _make_graph_and_schema(
        features={"time": np.array([np.array([])], dtype=np.object_)},
        feature_schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
            )
        },
    )
    new_graph, _ = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(sequence_length=3),
    )
    hw_val = new_graph.node_sets["hardware"]
    np.testing.assert_array_equal(hw_val.features["time"][0], [0, 0, 0])
    np.testing.assert_array_equal(hw_val.features["time_mask"][0], [0, 0, 0])

  def test_no_timeseries_in_graph(self):
    graph, schema = _make_graph_and_schema(
        features={"age": np.array([30])},
        feature_schemas={
            "age": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.INTEGER_64,
                semantic=schema_lib.FeatureSemantic.NUMERICAL,
            )
        },
    )
    new_graph, _ = timeseries.pad_and_cap_timeseries_features(
        graph, schema, timeseries.PadAndCapTimeseriesConfig()
    )
    self.assertEqual(new_graph.node_sets["hardware"].features["age"][0], 30)
    self.assertNotIn("age_mask", new_graph.node_sets["hardware"].features)

  def test_empty_entity_set(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "user": in_memory_graph.InMemoryNodeSet(
                num_nodes=0, features={"time": np.array([], dtype=np.object_)}
            ),
        },
        edge_sets={},
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "user": schema_lib.NodeSchema(
                features={
                    "time": _ts_schema(
                        fmt=schema_lib.FeatureFormat.INTEGER_64,
                        sem=schema_lib.FeatureSemantic.TIMESTAMP,
                    )
                }
            ),
        },
        edge_sets={},
    )
    new_graph, new_schema = timeseries.pad_and_cap_timeseries_features(
        graph, schema, timeseries.PadAndCapTimeseriesConfig(sequence_length=3)
    )
    user_val = new_graph.node_sets["user"]
    user_sch = new_schema.node_sets["user"]
    self.assertEqual(user_val.num_nodes, 0)
    self.assertEqual(user_val.features["time"].shape, (0, 3))
    self.assertEqual(user_val.features["time_mask"].shape, (0, 3))
    self.assertEqual(user_sch.features["time"].shape, (3,))
    self.assertTrue(user_sch.features["time"].is_timeseries)
    self.assertTrue(user_sch.features["time_mask"].is_timeseries)

  def test_missing_entity_set_raises(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={},
        edge_sets={},
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "absent": schema_lib.NodeSchema(
                features={
                    "time": _ts_schema(fmt=schema_lib.FeatureFormat.INTEGER_64)
                }
            ),
        },
        edge_sets={},
    )
    with self.assertRaises(KeyError):
      timeseries.pad_and_cap_timeseries_features(
          graph, schema, timeseries.PadAndCapTimeseriesConfig(sequence_length=3)
      )

  def test_custom_padding_value(self):
    graph, schema = _make_graph_and_schema(
        features={
            "time": np.array([np.array([10])], dtype=np.object_),
            "signal": [[2.0]],
        },
        feature_schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
            ),
            "signal": _ts_schema(timestamps="time"),
        },
        num_nodes=1,
    )
    new_graph, _ = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(
            sequence_length=3, padding_value=-1
        ),
    )
    hw_val = new_graph.node_sets["hardware"]
    np.testing.assert_array_equal(hw_val.features["time"][0], [-1, -1, 10])
    np.testing.assert_array_equal(
        hw_val.features["signal"][0], [-1.0, -1.0, 2.0]
    )
    np.testing.assert_array_equal(hw_val.features["time_mask"][0], [0, 0, 1])

  def test_fixed_shape_vectorized_path_capping(self):
    # Dense 2D array where all 2 nodes have fixed length T=5 >= K=3
    graph, schema = _make_graph_and_schema(
        features={
            "time": np.array(
                [[10, 20, 30, 40, 50], [100, 200, 300, 400, 500]],
                dtype=np.int64,
            ),
        },
        feature_schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
            ),
        },
        num_nodes=2,
    )
    new_graph, _ = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(sequence_length=3),
    )
    hw_val = new_graph.node_sets["hardware"]
    np.testing.assert_array_equal(
        hw_val.features["time"], [[30, 40, 50], [300, 400, 500]]
    )
    np.testing.assert_array_equal(
        hw_val.features["time_mask"], [[1, 1, 1], [1, 1, 1]]
    )

  def test_fixed_shape_vectorized_path_padding(self):
    # Dense 3D array where all 2 nodes have fixed length T=2 < K=3 and feature
    # dim 2
    graph, schema = _make_graph_and_schema(
        features={
            "emb": np.array(
                [
                    [[1.0, 1.1], [2.0, 2.2]],
                    [[10.0, 10.1], [20.0, 20.2]],
                ],
                dtype=np.float32,
            ),
        },
        feature_schemas={
            "emb": _ts_schema(
                sem=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(None, 2),
            ),
        },
        num_nodes=2,
    )
    new_graph, _ = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(
            sequence_length=3, padding_value=-1.0
        ),
    )
    hw_val = new_graph.node_sets["hardware"]
    expected_emb = np.array(
        [
            [[-1.0, -1.0], [1.0, 1.1], [2.0, 2.2]],
            [[-1.0, -1.0], [10.0, 10.1], [20.0, 20.2]],
        ],
        dtype=np.float32,
    )
    expected_mask = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(hw_val.features["emb"], expected_emb)
    np.testing.assert_array_equal(hw_val.features["emb_mask"], expected_mask)


if __name__ == "__main__":
  absltest.main()
