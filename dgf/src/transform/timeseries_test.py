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

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.transform import timeseries
from dgf.src.util import test_util
import numpy as np


def _make_graph_and_schema(
    values: dict[str, np.ndarray],
    schemas: dict[str, schema_lib.FeatureSchema],
    node_set_name: str = "hardware",
    num_nodes: int = 1,
    edge_values: dict[str, np.ndarray] | None = None,
    edge_schemas: dict[str, schema_lib.FeatureSchema] | None = None,
    edge_set_name: str = "edges",
) -> tuple[in_memory_graph.InMemoryGraph, schema_lib.GraphSchema]:
  edge_sets = {}
  edge_set_schemas = {}
  if edge_values is not None and edge_schemas is not None:
    edge_sets[edge_set_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=np.array([[0], [0]]),
        features=dict(edge_values),
    )
    edge_set_schemas[edge_set_name] = schema_lib.EdgeSchema(
        source=node_set_name,
        target=node_set_name,
        features=edge_schemas,
    )

  return (
      in_memory_graph.InMemoryGraph(
          node_sets={
              node_set_name: in_memory_graph.InMemoryNodeSet(
                  num_nodes=num_nodes, features=dict(values)
              )
          },
          edge_sets=edge_sets,
      ),
      schema_lib.GraphSchema(
          node_sets={node_set_name: schema_lib.NodeSchema(features=schemas)},
          edge_sets=edge_set_schemas,
      ),
  )


def _ts_schema(
    fmt: schema_lib.FeatureFormat = schema_lib.FeatureFormat.FLOAT_32,
    sem: schema_lib.FeatureSemantic = schema_lib.FeatureSemantic.NUMERICAL,
    timestamps: str | None = None,
    shape: schema_lib.Shape = (None,),
) -> schema_lib.FeatureSchema:
  return schema_lib.FeatureSchema(
      format=fmt,
      semantic=sem,
      is_timeseries=True,
      timestamps=timestamps,
      shape=shape,
  )


class TimeseriesTest(parameterized.TestCase):

  def test_capping_and_padding(self):
    # Both 'time' and 'signal' are variable-length sequence object arrays.
    graph, schema = _make_graph_and_schema(
        values={
            "time": np.array(
                [np.array([10, 20, 30, 40, 50]), np.array([5, 15])],
                dtype=np.object_,
            ),
            "signal": np.array(
                [
                    np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
                    np.array([0.5, 1.5], dtype=np.float32),
                ],
                dtype=np.object_,
            ),
            "id": np.array([101, 102]),
        },
        schemas={
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

    expected_features = {
        "time": np.array([[30, 40, 50], [0, 5, 15]], dtype=np.int64),
        "signal": np.array(
            [[3.0, 4.0, 5.0], [0.0, 0.5, 1.5]], dtype=np.float32
        ),
        "time_mask": np.array([[True, True, True], [False, True, True]]),
        "id": np.array([101, 102]),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)

    self.assertEqual(hw_sch.features["time"].shape, (3,))
    self.assertTrue(hw_sch.features["time"].is_timeseries)
    self.assertTrue(hw_sch.features["time_mask"].is_timeseries)
    self.assertEqual(
        hw_sch.features["time_mask"].semantic, schema_lib.FeatureSemantic.MASK
    )

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
        values={
            "emb": np.array(
                [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
                dtype=np.object_,
            )
        },
        schemas={
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

    expected_features = {
        "emb": np.array(
            [[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]], dtype=np.float32
        ),
        "emb_mask": np.array([[False, True, True]]),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)
    self.assertEqual(hw_sch.features["emb"].shape, (3, 2))
    self.assertEqual(hw_sch.features["emb_mask"].shape, (3,))
    self.assertEqual(
        hw_sch.features["emb_mask"].semantic, schema_lib.FeatureSemantic.MASK
    )

  def test_custom_mask_name_is_reused(self):
    graph, schema = _make_graph_and_schema(
        values={
            "emb": np.array(
                [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
                dtype=np.object_,
            ),
            "my_custom_mask": np.array(
                [np.array([True, True], dtype=bool)],
                dtype=np.object_,
            ),
        },
        schemas={
            "emb": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                is_timeseries=True,
                shape=(None, 2),
                timeseries_group="emb_group",
            ),
            "my_custom_mask": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.BOOL,
                semantic=schema_lib.FeatureSemantic.MASK,
                is_timeseries=True,
                shape=(None,),
                timeseries_group="emb_group",
            ),
        },
    )
    new_graph, new_schema = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(sequence_length=3),
    )
    hw_val = new_graph.node_sets["hardware"]
    hw_sch = new_schema.node_sets["hardware"]

    expected_features = {
        "emb": np.array(
            [[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]], dtype=np.float32
        ),
        "my_custom_mask": np.array([[False, True, True]]),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)
    self.assertEqual(hw_sch.features["emb"].shape, (3, 2))
    self.assertEqual(hw_sch.features["my_custom_mask"].shape, (3,))
    self.assertNotIn("emb_mask", hw_sch.features)

  def test_clashing_mask_name_raises(self):
    graph, schema = _make_graph_and_schema(
        values={
            "emb": np.array(
                [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
                dtype=np.object_,
            ),
            "emb_mask": np.array(
                [np.array([42.0], dtype=np.float32)],
                dtype=np.object_,
            ),
        },
        schemas={
            "emb": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                is_timeseries=True,
                shape=(None, 2),
                timeseries_group="emb",
            ),
            # This feature is named "emb_mask", which is the fallback mask name
            # for the group "emb", but its semantic is not MASK.
            "emb_mask": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.NUMERICAL,
                is_timeseries=True,
                shape=(None, 1),
                timeseries_group="emb",
            ),
        },
    )
    with self.assertRaisesRegex(
        ValueError, "clashes with an existing feature"
    ):
      timeseries.pad_and_cap_timeseries_features(
          graph,
          schema,
          timeseries.PadAndCapTimeseriesConfig(sequence_length=3),
      )

  def test_empty_sequence(self):
    graph, schema = _make_graph_and_schema(
        values={"time": np.array([np.array([])], dtype=np.object_)},
        schemas={
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
        values={"age": np.array([30])},
        schemas={
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
    self.assertEqual(
        user_sch.features["time_mask"].semantic, schema_lib.FeatureSemantic.MASK
    )

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
    with self.assertRaises(KeyError):
      timeseries.extract_calendar_features(graph, schema)

  def test_missing_feature_raises(self):
    graph, schema = _make_graph_and_schema(
        values={},
        schemas={
            "absent_feature": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
            )
        },
    )
    with self.assertRaises(KeyError):
      timeseries.pad_and_cap_timeseries_features(
          graph,
          schema,
          timeseries.PadAndCapTimeseriesConfig(sequence_length=3),
      )
    with self.assertRaises(KeyError):
      timeseries.extract_calendar_features(graph, schema)

  def test_custom_padding_value(self):
    graph, schema = _make_graph_and_schema(
        values={
            "time": np.array([np.array([10])], dtype=np.object_),
            "signal": np.array(
                [np.array([2.0], dtype=np.float32)], dtype=np.object_
            ),
        },
        schemas={
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
    expected_features = {
        "time": np.array([[-1, -1, 10]], dtype=np.int64),
        "signal": np.array([[-1.0, -1.0, 2.0]], dtype=np.float32),
        "time_mask": np.array([[False, False, True]]),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)

  def test_fixed_shape_vectorized_path_capping(self):
    # Dense 2D array where all 2 nodes have fixed length T=5 >= K=3
    graph, schema = _make_graph_and_schema(
        values={
            "time": np.array(
                [[10, 20, 30, 40, 50], [100, 200, 300, 400, 500]],
                dtype=np.int64,
            ),
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                shape=(5,),
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
    expected_features = {
        "time": np.array([[30, 40, 50], [300, 400, 500]], dtype=np.int64),
        "time_mask": np.array([[True, True, True], [True, True, True]]),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)

  def test_fixed_shape_vectorized_path_padding(self):
    # Dense 3D array where all 2 nodes have fixed length T=2 < K=3 and feature
    # dim 2
    graph, schema = _make_graph_and_schema(
        values={
            "emb": np.array(
                [
                    [[1.0, 1.1], [2.0, 2.2]],
                    [[10.0, 10.1], [20.0, 20.2]],
                ],
                dtype=np.float32,
            ),
        },
        schemas={
            "emb": _ts_schema(
                sem=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(2, 2),
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
    expected_features = {
        "emb": np.array(
            [
                [[-1.0, -1.0], [1.0, 1.1], [2.0, 2.2]],
                [[-1.0, -1.0], [10.0, 10.1], [20.0, 20.2]],
            ],
            dtype=np.float32,
        ),
        "emb_mask": np.array([
            [False, True, True],
            [False, True, True],
        ]),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)

  def test_compute_calendar_feature(self):
    ts = np.array([65, 3665, 1680000015], dtype=np.int64)
    computed = {
        feat: timeseries._compute_calendar_feature(ts, feat)
        for feat in timeseries._SUPPORTED_CALENDAR_FEATURES
    }
    expected = {
        timeseries.CalendarFeature.SECOND: np.array(
            [5.0, 5.0, 15.0], dtype=np.float32
        ),
        timeseries.CalendarFeature.MINUTE: np.array(
            [1.0, 1.0, 40.0], dtype=np.float32
        ),
        timeseries.CalendarFeature.HOUR: np.array(
            [0.0, 1.0, 10.0], dtype=np.float32
        ),
        timeseries.CalendarFeature.DAY_OF_WEEK: np.array(
            [3.0, 3.0, 1.0], dtype=np.float32
        ),
        timeseries.CalendarFeature.MONTH: np.array(
            [1.0, 1.0, 3.0], dtype=np.float32
        ),
        timeseries.CalendarFeature.YEAR: np.array(
            [1970.0, 1970.0, 2023.0], dtype=np.float32
        ),
    }
    test_util.assert_are_equal(self, computed, expected)

  def test_extract_calendar_features(self):
    graph, schema = _make_graph_and_schema(
        values={
            "time": np.array(
                [np.array([65, 1680000015], dtype=np.int64)], dtype=np.object_
            )
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
            )
        },
    )
    padded_graph, padded_schema = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(sequence_length=2),
    )
    cal_graph, cal_schema = timeseries.extract_calendar_features(
        padded_graph, padded_schema
    )

    hw_val = cal_graph.node_sets["hardware"]
    hw_sch = cal_schema.node_sets["hardware"]

    for cal_k in (
        "second",
        "minute",
        "hour",
        "day_of_week",
        "month",
        "year",
    ):
      feature_name = f"time_{cal_k}"
      self.assertIn(feature_name, hw_val.features)
      fschema = hw_sch.features[feature_name]
      self.assertEqual(
          fschema.semantic,
          schema_lib.FeatureSemantic.NUMERICAL,
      )
      self.assertEqual(fschema.shape, (2,))
      self.assertTrue(fschema.is_timeseries)
      self.assertEqual(fschema.timestamps, "time")

    # 65 -> 1970-01-01 00:01:05 UTC (Thursday=3)
    expected_features = {
        "time": hw_val.features["time"],
        "time_mask": np.array([[True, True]]),
        "time_second": np.array([[5.0, 15.0]], dtype=np.float32),
        "time_minute": np.array([[1.0, 40.0]], dtype=np.float32),
        "time_hour": np.array([[0.0, 10.0]], dtype=np.float32),
        "time_day_of_week": np.array([[3.0, 1.0]], dtype=np.float32),
        "time_month": np.array([[1.0, 3.0]], dtype=np.float32),
        "time_year": np.array([[1970.0, 2023.0]], dtype=np.float32),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)

  def test_extract_calendar_features_requires_fixed_length(self):
    graph, schema = _make_graph_and_schema(
        values={
            "time": np.array(
                [np.array([65, 1680000015], dtype=np.int64)], dtype=np.object_
            )
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
            )
        },
    )
    with self.assertRaisesRegex(
        ValueError,
        "extract_calendar_features requires fixed-length timestamp tensors",
    ):
      timeseries.extract_calendar_features(graph, schema)

  def test_extract_calendar_features_non_timeseries(self):
    graph, schema = _make_graph_and_schema(
        values={"x": np.array([1.0], dtype=np.float32)},
        schemas={
            "x": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.NUMERICAL,
            )
        },
        node_set_name="static_nodes",
        edge_values={"weight": np.array([0.5], dtype=np.float32)},
        edge_schemas={
            "weight": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.NUMERICAL,
            )
        },
        edge_set_name="static_edges",
    )
    cal_graph, cal_schema = timeseries.extract_calendar_features(graph, schema)
    self.assertIn("static_nodes", cal_graph.node_sets)
    self.assertIn("static_edges", cal_graph.edge_sets)
    self.assertEqual(
        cal_schema.node_sets["static_nodes"], schema.node_sets["static_nodes"]
    )
    self.assertEqual(
        cal_schema.edge_sets["static_edges"], schema.edge_sets["static_edges"]
    )

  def test_extract_calendar_features_edge_sets(self):
    graph, schema = _make_graph_and_schema(
        values={},
        schemas={},
        node_set_name="nodes",
        edge_values={"time": np.array([[65, 1680000015]], dtype=np.int64)},
        edge_schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                shape=(2,),
            )
        },
        edge_set_name="ts_edges",
    )
    cal_graph, cal_schema = timeseries.extract_calendar_features(graph, schema)
    self.assertIn("time_hour", cal_graph.edge_sets["ts_edges"].features)
    self.assertIn("time_second", cal_graph.edge_sets["ts_edges"].features)
    np.testing.assert_array_equal(
        cal_graph.edge_sets["ts_edges"].features["time_second"][0], [5, 15]
    )
    self.assertIn("time_hour", cal_schema.edge_sets["ts_edges"].features)

  def test_extract_calendar_features_static_timestamp(self):
    # Non-timeseries timestamp feature.
    graph, schema = _make_graph_and_schema(
        values={"created_at": np.array([65, 1680000015], dtype=np.int64)},
        schemas={
            "created_at": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.INTEGER_64,
                semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                is_timeseries=False,
                shape=(),
            )
        },
    )
    cal_graph, cal_schema = timeseries.extract_calendar_features(graph, schema)
    hw_val = cal_graph.node_sets["hardware"]
    hw_sch = cal_schema.node_sets["hardware"]

    self.assertIn("created_at_hour", hw_val.features)
    fschema = hw_sch.features["created_at_hour"]
    self.assertFalse(fschema.is_timeseries)
    self.assertIsNone(fschema.timestamps)
    np.testing.assert_array_equal(
        hw_val.features["created_at_hour"], [0.0, 10.0]
    )

  def test_extract_calendar_features_parent_timestamp(self):
    # Timestamp feature that references other timestamp feature.
    graph, schema = _make_graph_and_schema(
        values={
            "event_time": np.array([[65, 3665]], dtype=np.int64),
            "master_time": np.array([[65, 3665]], dtype=np.int64),
        },
        schemas={
            "event_time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                timestamps="master_time",
                shape=(2,),
            ),
            "master_time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                shape=(2,),
            ),
        },
    )
    _, cal_schema = timeseries.extract_calendar_features(graph, schema)
    hw_sch = cal_schema.node_sets["hardware"]
    # Calendar feature derived from event_time inherits timestamps="master_time"
    self.assertEqual(
        hw_sch.features["event_time_hour"].timestamps, "master_time"
    )
    # Calendar feature derived from master_time references master_time
    self.assertEqual(
        hw_sch.features["master_time_hour"].timestamps, "master_time"
    )

  def test_extract_timestamp_features(self):
    graph, schema = _make_graph_and_schema(
        values={
            "time": np.array(
                [np.array([100, 250, 300], dtype=np.int64)], dtype=np.object_
            )
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
            )
        },
    )
    padded_graph, padded_schema = timeseries.pad_and_cap_timeseries_features(
        graph,
        schema,
        timeseries.PadAndCapTimeseriesConfig(sequence_length=4),
    )
    delta_graph, delta_schema = timeseries.extract_timestamp_features(
        padded_graph,
        padded_schema,
        seed_timestamp=500,
        config=timeseries.TimestampFeatureConfig(),
    )

    hw_val = delta_graph.node_sets["hardware"]
    hw_sch = delta_schema.node_sets["hardware"]

    # Padded sequence: [0, 100, 250, 300] with mask [0, 1, 1, 1]
    # Seed delta (seed=500): [0, 400, 250, 200]
    expected_features = {
        "time": np.array([[0, 100, 250, 300]], dtype=np.int64),
        "time_mask": np.array([[False, True, True, True]]),
        "time_seed_delta": np.array([[0, 400, 250, 200]], dtype=np.int64),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)

    expected_schemas = {
        "time": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.TIMESTAMP,
            shape=(4,),
            is_timeseries=True,
            timestamps=None,
            timeseries_group="time",
        ),
        "time_mask": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BOOL,
            semantic=schema_lib.FeatureSemantic.MASK,
            shape=(4,),
            is_timeseries=True,
            timestamps=None,
            timeseries_group="time",
        ),
        "time_seed_delta": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.TIMESTAMP,
            shape=(4,),
            is_timeseries=True,
            timestamps="time",
            timeseries_group="time",
        ),
    }
    test_util.assert_are_equal(self, hw_sch.features, expected_schemas)

  def test_extract_timestamp_features_static_timestamp(self):
    values = {"created_at": np.array([65, 1680000015], dtype=np.int64)}
    schemas = {
        "created_at": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.TIMESTAMP,
            is_timeseries=False,
            shape=(),
        )
    }
    new_vals, new_schs = timeseries._extract_feature_set_timestamp_features(
        values, schemas, timeseries.TimestampFeatureConfig(), seed_timestamp=500
    )
    np.testing.assert_array_equal(
        new_vals["created_at_seed_delta"], [435, -1679999515]
    )
    self.assertFalse(new_schs["created_at_seed_delta"].is_timeseries)

  def test_extract_timestamp_features_parent_timestamp(self):
    values = {
        "event_time": np.array([[65, 3665]], dtype=np.int64),
        "master_time": np.array([[65, 3665]], dtype=np.int64),
    }
    schemas = {
        "event_time": _ts_schema(
            sem=schema_lib.FeatureSemantic.TIMESTAMP,
            timestamps="master_time",
            shape=(2,),
        ),
        "master_time": _ts_schema(
            sem=schema_lib.FeatureSemantic.TIMESTAMP, shape=(2,)
        ),
    }
    _, new_schs = timeseries._extract_feature_set_timestamp_features(
        values, schemas, timeseries.TimestampFeatureConfig(), seed_timestamp=500
    )
    self.assertEqual(
        new_schs["event_time_seed_delta"].timestamps, "master_time"
    )
    self.assertEqual(
        new_schs["master_time_seed_delta"].timestamps, "master_time"
    )

  def test_extract_timestamp_features_requires_fixed_length(self):
    graph, schema = _make_graph_and_schema(
        values={
            "time": np.array(
                [np.array([100, 250], dtype=np.int64)], dtype=np.object_
            )
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
            )
        },
    )
    with self.assertRaisesRegex(
        ValueError,
        "extract_timestamp_features requires fixed-length timestamp tensors",
    ):
      timeseries.extract_timestamp_features(
          graph,
          schema,
          seed_timestamp=500,
          config=timeseries.TimestampFeatureConfig(),
      )

  def test_extract_timestamp_features_edge_sets(self):
    graph, schema = _make_graph_and_schema(
        values={},
        schemas={},
        node_set_name="nodes",
        edge_values={"time": np.array([[100, 250]], dtype=np.int64)},
        edge_schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                shape=(2,),
            )
        },
        edge_set_name="ts_edges",
    )
    graph.edge_sets["ts_edges"].features["time_mask"] = np.array(
        [[1, 1]], dtype=np.bool_
    )
    schema.edge_sets["ts_edges"].features["time_mask"] = _ts_schema(
        fmt=schema_lib.FeatureFormat.BOOL,
        sem=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(2,),
    )

    delta_graph, delta_schema = timeseries.extract_timestamp_features(
        graph,
        schema, seed_timestamp=500, config=timeseries.TimestampFeatureConfig(),
    )
    es_val = delta_graph.edge_sets["ts_edges"]
    es_sch = delta_schema.edge_sets["ts_edges"]

    self.assertIn("time_seed_delta", es_val.features)
    self.assertEqual(es_sch.features["time_seed_delta"].timestamps, "time")

  def test_extract_timestamp_features_non_timeseries(self):
    values = {"x": np.array([1.0], dtype=np.float32)}
    schemas = {
        "x": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.NUMERICAL,
        )
    }
    new_vals, new_schs = timeseries._extract_feature_set_timestamp_features(
        values, schemas, timeseries.TimestampFeatureConfig(), seed_timestamp=500
    )
    self.assertEqual(new_vals, values)
    self.assertEqual(new_schs, schemas)

  @parameterized.parameters(
      (np.array([[False, True, True]]), 0, [[0, 400, 250]]),
      (None, 0, [[500, 400, 250]]),
      (np.array([[False, True, True]]), -999, [[-999, 400, 250]]),
  )
  def test_compute_seed_deltas(self, mask, fill_value, expected):
    raw_val = np.array([[0, 100, 250]], dtype=np.int64)
    deltas = timeseries._compute_seed_deltas(raw_val, mask, 500, fill_value)
    np.testing.assert_array_equal(deltas, expected)


if __name__ == "__main__":
  absltest.main()
