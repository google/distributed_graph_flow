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

"""Tests for temporal and timeseries feature extractors."""

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
    group: str | None = None,
    is_creation_time: bool = False,
    shape: schema_lib.Shape = (None,),
) -> schema_lib.FeatureSchema:
  return schema_lib.FeatureSchema(
      format=fmt,
      semantic=sem,
      is_timeseries=True,
      group=group,
      is_creation_time=is_creation_time,
      shape=shape,
  )


class TimeseriesTest(parameterized.TestCase):

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
    padded_graph, padded_schema = _make_graph_and_schema(
        values={
            "time": np.array([[65, 1680000015]], dtype=np.int64),
            "time_mask": np.array([[True, True]], dtype=np.bool_),
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                group="time",
                shape=(2,),
            ),
            "time_mask": _ts_schema(
                fmt=schema_lib.FeatureFormat.BOOL,
                sem=schema_lib.FeatureSemantic.MASK,
                group="time",
                shape=(2,),
            ),
        },
    )

    cal_extractor = timeseries.CalendarFeatureExtractor(padded_schema)
    cal_graph = cal_extractor(padded_graph)
    cal_schema = cal_extractor.output_schema()

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
      self.assertEqual(fschema.group, "time")

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
    extractor = timeseries.CalendarFeatureExtractor(schema)
    with self.assertRaisesRegex(
        AssertionError,
        "CalendarFeatureExtractor requires fixed-length timestamp tensors",
    ):
      extractor(graph)

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
    extractor = timeseries.CalendarFeatureExtractor(schema)
    cal_graph = extractor(graph)
    cal_schema = extractor.output_schema()
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
    extractor = timeseries.CalendarFeatureExtractor(schema)
    cal_graph = extractor(graph)
    cal_schema = extractor.output_schema()
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
    extractor = timeseries.CalendarFeatureExtractor(schema)
    cal_graph = extractor(graph)
    cal_schema = extractor.output_schema()
    hw_val = cal_graph.node_sets["hardware"]
    hw_sch = cal_schema.node_sets["hardware"]

    self.assertIn("created_at_hour", hw_val.features)
    fschema = hw_sch.features["created_at_hour"]
    self.assertFalse(fschema.is_timeseries)
    self.assertIsNone(fschema.group)
    np.testing.assert_array_equal(
        hw_val.features["created_at_hour"], [0.0, 10.0]
    )

  def test_extract_calendar_features_parent_timestamp(self):
    # Timestamp feature with group.
    _, schema = _make_graph_and_schema(
        values={
            "event_time": np.array([[65, 3665]], dtype=np.int64),
            "master_time": np.array([[65, 3665]], dtype=np.int64),
        },
        schemas={
            "event_time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                group="master_time",
                shape=(2,),
            ),
            "master_time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                is_creation_time=True,
                group="master_time",
                shape=(2,),
            ),
        },
    )
    extractor = timeseries.CalendarFeatureExtractor(schema)
    cal_schema = extractor.output_schema()
    hw_sch = cal_schema.node_sets["hardware"]
    self.assertEqual(
        hw_sch.features["event_time_hour"].group, "master_time"
    )
    self.assertEqual(
        hw_sch.features["master_time_hour"].group, "master_time"
    )

  def test_extract_timestamp_features(self):
    padded_graph, padded_schema = _make_graph_and_schema(
        values={
            "time": np.array([[0, 100, 250, 300]], dtype=np.int64),
            "time_mask": np.array([[False, True, True, True]], dtype=np.bool_),
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                group="time",
                shape=(4,),
            ),
            "time_mask": _ts_schema(
                fmt=schema_lib.FeatureFormat.BOOL,
                sem=schema_lib.FeatureSemantic.MASK,
                group="time",
                shape=(4,),
            ),
        },
    )

    ts_extractor = timeseries.TimestampFeatureExtractor(
        padded_schema, config=timeseries.TimestampFeatureExtractorConfig()
    )
    delta_graph = ts_extractor(padded_graph, seed_timestamp=500)
    delta_schema = ts_extractor.output_schema()

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
            group="time",
        ),
        "time_mask": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BOOL,
            semantic=schema_lib.FeatureSemantic.MASK,
            shape=(4,),
            is_timeseries=True,
            group="time",
        ),
        "time_seed_delta": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.TIMEDELTA,
            shape=(4,),
            is_timeseries=True,
            group="time",
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
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "hardware": in_memory_graph.InMemoryNodeSet(
                num_nodes=2, features=values
            )
        },
        edge_sets={},
    )
    schema = schema_lib.GraphSchema(
        node_sets={"hardware": schema_lib.NodeSchema(features=schemas)},
        edge_sets={},
    )
    extractor = timeseries.TimestampFeatureExtractor(schema)
    new_graph = extractor(graph, seed_timestamp=500)
    new_schema = extractor.output_schema()
    np.testing.assert_array_equal(
        new_graph.node_sets["hardware"].features["created_at_seed_delta"],
        [435, -1679999515],
    )
    self.assertFalse(
        new_schema.node_sets["hardware"]
        .features["created_at_seed_delta"]
        .is_timeseries
    )

  def test_extract_timestamp_features_parent_timestamp(self):
    schemas = {
        "event_time": _ts_schema(
            sem=schema_lib.FeatureSemantic.TIMESTAMP,
            group="master_time",
            shape=(2,),
        ),
        "master_time": _ts_schema(
            sem=schema_lib.FeatureSemantic.TIMESTAMP,
            group="master_time",
            shape=(2,),
        ),
    }
    schema = schema_lib.GraphSchema(
        node_sets={"hardware": schema_lib.NodeSchema(features=schemas)},
        edge_sets={},
    )
    extractor = timeseries.TimestampFeatureExtractor(schema)
    new_schema = extractor.output_schema()
    hw_sch = new_schema.node_sets["hardware"]
    self.assertEqual(
        hw_sch.features["event_time_seed_delta"].group, "master_time"
    )
    self.assertEqual(
        hw_sch.features["master_time_seed_delta"].group, "master_time"
    )
    self.assertEqual(
        hw_sch.features["event_time_seed_delta"].semantic,
        schema_lib.FeatureSemantic.TIMEDELTA,
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
    extractor = timeseries.TimestampFeatureExtractor(
        schema, config=timeseries.TimestampFeatureExtractorConfig()
    )
    with self.assertRaisesRegex(
        AssertionError,
        "TimestampFeatureExtractor requires fixed-length timestamp tensors",
    ):
      extractor(graph, seed_timestamp=500)

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
                group="time",
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

    extractor = timeseries.TimestampFeatureExtractor(
        schema, config=timeseries.TimestampFeatureExtractorConfig()
    )
    delta_graph = extractor(graph, seed_timestamp=500)
    delta_schema = extractor.output_schema()
    es_val = delta_graph.edge_sets["ts_edges"]
    es_sch = delta_schema.edge_sets["ts_edges"]

    self.assertIn("time_seed_delta", es_val.features)
    self.assertEqual(es_sch.features["time_seed_delta"].group, "time")
    self.assertEqual(
        es_sch.features["time_seed_delta"].semantic,
        schema_lib.FeatureSemantic.TIMEDELTA,
    )

  def test_extract_timestamp_features_non_timeseries(self):
    values = {"x": np.array([1.0], dtype=np.float32)}
    schemas = {
        "x": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.NUMERICAL,
        )
    }
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "hardware": in_memory_graph.InMemoryNodeSet(
                num_nodes=1, features=values
            )
        },
        edge_sets={},
    )
    schema = schema_lib.GraphSchema(
        node_sets={"hardware": schema_lib.NodeSchema(features=schemas)},
        edge_sets={},
    )
    extractor = timeseries.TimestampFeatureExtractor(schema)
    new_graph = extractor(graph, seed_timestamp=500)
    new_schema = extractor.output_schema()
    self.assertEqual(new_graph.node_sets["hardware"].features, values)
    self.assertEqual(new_schema.node_sets["hardware"].features, schemas)

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
