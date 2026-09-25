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

"""Unit tests for dataset_loader_relbench.py."""

import io
import logging
import os
import unittest
from unittest import mock
import urllib.request
import zipfile

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.analyse import print_schema
from dgf.src.data import schema as schema_lib
from dgf.src.io import dataset_loader
from dgf.src.io import dataset_loader_relbench
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np
import pandas as pd

test_util.disable_diff_truncation()


def _synthetic_spec() -> dataset_loader_relbench.RelbenchDatasetSpec:
  """Returns a 3-table synthetic RelBench spec (`parents`, `children`, `keys_only`)."""
  return dataset_loader_relbench.RelbenchDatasetSpec(
      tables={
          "parents": dataset_loader_relbench.TableSpec(
              primary_key="parentId",
          ),
          "children": dataset_loader_relbench.TableSpec(
              primary_key="childId",
              time_column="event_date",
              foreign_keys={"parentId": "parents"},
          ),
          "keys_only": dataset_loader_relbench.TableSpec(
              primary_key="keyId",
              foreign_keys={"parentId": "parents"},
          ),
      },
      tasks={
          "parent-score": dataset_loader_relbench.TaskSpec(
              entity_column="parentId",
              entity_table="parents",
              time_column="query_date",
              target_column="score",
              is_classification=False,
              hidden_columns=(("children", "leaked_column"),),
          ),
          "parent-flag": dataset_loader_relbench.TaskSpec(
              entity_column="parentId",
              entity_table="parents",
              time_column="query_date",
              target_column="flag",
              is_classification=True,
          ),
      },
  )


def _synthetic_tables() -> dict[str, pd.DataFrame]:
  """Returns synthetic DataFrames matching `_synthetic_spec()`."""
  parents = pd.DataFrame({
      "parentId": pd.Series([0, 1, 2], dtype="Int64"),
      "name": ["alice", "bob", "carol"],
      "rating": np.array([10.5, 20.0, 30.25], dtype=np.float64),
      "created_at": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
  })
  children = pd.DataFrame({
      "childId": pd.Series([0, 1, 2, 3], dtype="Int64"),
      "parentId": pd.Series([0, 2, pd.NA, 1], dtype="Int64"),
      "event_date": pd.to_datetime([
          "2021-01-01",
          "2021-01-02",
          "2021-01-03",
          "2021-01-04",
      ]),
      "value": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
      "leaked_column": np.array([99.0, 98.0, 97.0, 96.0], dtype=np.float64),
  })
  keys_only = pd.DataFrame({
      "keyId": pd.Series([0, 1], dtype="Int64"),
      "parentId": pd.Series([1, 0], dtype="Int64"),
  })
  return {
      "parents": parents,
      "children": children,
      "keys_only": keys_only,
  }


def _synthetic_task_tables() -> dict[str, pd.DataFrame]:
  """Returns synthetic train/valid/test task DataFrames."""
  train = pd.DataFrame({
      "query_date": pd.to_datetime(["2021-02-01", "2021-02-15"]),
      "parentId": np.array([0, 1], dtype=np.int64),
      "score": np.array([1.5, 2.5], dtype=np.float64),
      "flag": np.array([0, 1], dtype=np.int64),
  })
  valid = pd.DataFrame({
      "query_date": pd.to_datetime(["2021-03-01"]),
      "parentId": np.array([2], dtype=np.int64),
      "score": np.array([3.5], dtype=np.float64),
      "flag": np.array([1], dtype=np.int64),
  })
  test = pd.DataFrame({
      "query_date": pd.to_datetime(["2021-04-01", "2021-04-10"]),
      "parentId": np.array([1, 0], dtype=np.int64),
      "score": np.array([4.5, 5.5], dtype=np.float64),
      "flag": np.array([0, 1], dtype=np.int64),
  })
  return {"train": train, "valid": valid, "test": test}


def _mock_rel_f1_tables() -> (
    tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
):
  """Returns small hermetic DataFrames matching the full `rel-f1` schema."""
  spec = dataset_loader_relbench.relbench_dataset_spec("rel-f1")
  tables: dict[str, pd.DataFrame] = {}
  for table_name, table_spec in spec.tables.items():
    columns: dict[str, object] = {}
    if table_spec.primary_key is not None:
      columns[table_spec.primary_key] = pd.Series([0, 1], dtype="Int64")
    if table_spec.time_column is not None:
      columns[table_spec.time_column] = pd.to_datetime(
          ["2004-05-01", "2004-06-01"]
      )
    for foreign_key_column in table_spec.foreign_keys:
      columns[foreign_key_column] = pd.Series([0, 1], dtype="Int64")
    columns["metric"] = np.array([1.0, 2.0], dtype=np.float64)
    tables[table_name] = pd.DataFrame(columns)

  task_tables = {
      "train": pd.DataFrame({
          "date": pd.to_datetime(["2004-01-01", "2004-06-01"]),
          "driverId": np.array([0, 1], dtype=np.int64),
          "position": np.array([1.0, 5.0], dtype=np.float64),
          "did_not_finish": np.array([0, 1], dtype=np.int32),
          "qualifying": np.array([1, 0], dtype=np.int64),
      }),
      "valid": pd.DataFrame({
          "date": pd.to_datetime(["2006-01-01"]),
          "driverId": np.array([0], dtype=np.int64),
          "position": np.array([2.0], dtype=np.float64),
          "did_not_finish": np.array([0], dtype=np.int32),
          "qualifying": np.array([1], dtype=np.int64),
      }),
      "test": pd.DataFrame({
          "date": pd.to_datetime(["2011-01-01"]),
          "driverId": np.array([1], dtype=np.int64),
          "position": np.array([3.0], dtype=np.float64),
          "did_not_finish": np.array([1], dtype=np.int32),
          "qualifying": np.array([0], dtype=np.int64),
      }),
  }
  return tables, task_tables


def _zip_bytes(members: dict[str, pd.DataFrame]) -> bytes:
  """Returns a zip archive containing each DataFrame as a Parquet member."""
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w") as archive:
    for member, df in members.items():
      parquet_buffer = io.BytesIO()
      df.to_parquet(parquet_buffer)
      archive.writestr(member, parquet_buffer.getvalue())
  return buffer.getvalue()


class DatasetLoaderRelbenchTest(parameterized.TestCase):

  def _build_synthetic(self, task: str = "parent-score"):
    graph, schema = dataset_loader_relbench.build_relational_graph(
        tables=_synthetic_tables(),
        task_tables=_synthetic_task_tables(),
        spec=_synthetic_spec(),
        task=task,
    )
    self.assertEmpty(in_memory_graph_validate_lib.issues(graph, schema))
    return graph, schema

  # Structure tests
  def test_node_set_per_table(self):
    graph, schema = self._build_synthetic()
    self.assertCountEqual(
        schema.node_sets.keys(),
        ["parents", "children", "keys_only", "queries"],
    )
    self.assertEqual(graph.node_sets["parents"].num_nodes, 3)
    self.assertEqual(graph.node_sets["children"].num_nodes, 4)
    self.assertEqual(graph.node_sets["keys_only"].num_nodes, 2)
    self.assertEqual(graph.node_sets["queries"].num_nodes, 5)

  def test_only_child_to_parent_edge_sets_exist(self):
    graph, schema = self._build_synthetic()
    expected_edge_sets = {
        "f2p_children_parentId",
        "f2p_keys_only_parentId",
        "f2p_queries_parentId",
    }
    self.assertCountEqual(schema.edge_sets.keys(), expected_edge_sets)
    self.assertCountEqual(graph.edge_sets.keys(), expected_edge_sets)

  def test_edge_set_endpoints(self):
    _, schema = self._build_synthetic()
    self.assertEqual(
        schema.edge_sets["f2p_children_parentId"].source, "children"
    )
    self.assertEqual(
        schema.edge_sets["f2p_children_parentId"].target, "parents"
    )
    self.assertEqual(schema.edge_sets["f2p_queries_parentId"].source, "queries")
    self.assertEqual(schema.edge_sets["f2p_queries_parentId"].target, "parents")

  def test_dangling_foreign_keys_dropped(self):
    graph, _ = self._build_synthetic()
    # Row 2 in `children` has parentId = pd.NA, so only 3 edges are emitted.
    forward = graph.edge_sets["f2p_children_parentId"].adjacency
    np.testing.assert_array_equal(forward[0], [0, 1, 3])
    np.testing.assert_array_equal(forward[1], [0, 2, 1])

  # Conformance tests
  def test_primary_key_not_a_feature(self):
    graph, schema = self._build_synthetic()
    self.assertNotIn("parentId", schema.node_sets["parents"].features)
    self.assertNotIn("parentId", graph.node_sets["parents"].features)
    self.assertNotIn("childId", schema.node_sets["children"].features)
    self.assertNotIn("keyId", schema.node_sets["keys_only"].features)

  def test_foreign_key_not_a_feature(self):
    graph, schema = self._build_synthetic()
    self.assertNotIn("parentId", schema.node_sets["children"].features)
    self.assertNotIn("parentId", graph.node_sets["children"].features)
    self.assertNotIn("parentId", schema.node_sets["keys_only"].features)
    self.assertNotIn("parentId", schema.node_sets["queries"].features)
    self.assertNotIn("parentId", graph.node_sets["queries"].features)

  def test_hidden_columns_removed(self):
    graph, schema = self._build_synthetic(task="parent-score")
    self.assertNotIn("leaked_column", schema.node_sets["children"].features)
    self.assertNotIn("leaked_column", graph.node_sets["children"].features)

    # For `parent-flag`, `leaked_column` is not listed in `hidden_columns`.
    graph_flag, schema_flag = self._build_synthetic(task="parent-flag")
    self.assertIn("leaked_column", schema_flag.node_sets["children"].features)
    self.assertIn("leaked_column", graph_flag.node_sets["children"].features)

  def test_featureless_table_gets_const_feature(self):
    graph, schema = self._build_synthetic()
    const_name = dataset_loader_relbench.CONST_FEATURE
    self.assertIn(const_name, schema.node_sets["keys_only"].features)
    self.assertEqual(
        schema.node_sets["keys_only"].features[const_name].format,
        schema_lib.FeatureFormat.FLOAT_32,
    )
    self.assertEqual(
        schema.node_sets["keys_only"].features[const_name].semantic,
        schema_lib.FeatureSemantic.NUMERICAL,
    )
    np.testing.assert_array_equal(
        graph.node_sets["keys_only"].features[const_name],
        np.ones(2, dtype=np.float32),
    )

  def test_creation_time_is_int64_and_flagged(self):
    graph, schema = self._build_synthetic()
    ct_schema = schema.node_sets["children"].features["creation_time"]
    self.assertEqual(ct_schema.format, schema_lib.FeatureFormat.INTEGER_64)
    self.assertEqual(ct_schema.semantic, schema_lib.FeatureSemantic.TIMESTAMP)
    self.assertTrue(ct_schema.is_creation_time)
    self.assertFalse(ct_schema.is_timeseries)
    self.assertEqual(
        graph.node_sets["children"].features["creation_time"].dtype, np.int64
    )

    # Static table (`parents`) has a non-creation-time datetime feature `created_at`.
    self.assertNotIn("creation_time", schema.node_sets["parents"].features)
    created_at_schema = schema.node_sets["parents"].features["created_at"]
    self.assertEqual(
        created_at_schema.format, schema_lib.FeatureFormat.INTEGER_64
    )
    self.assertEqual(
        created_at_schema.semantic, schema_lib.FeatureSemantic.NUMERICAL
    )
    self.assertFalse(created_at_schema.is_creation_time)

  def test_raw_time_column_not_duplicated_as_feature(self):
    graph, schema = self._build_synthetic()
    self.assertNotIn("event_date", schema.node_sets["children"].features)
    self.assertNotIn("event_date", graph.node_sets["children"].features)
    self.assertNotIn("query_date", schema.node_sets["queries"].features)
    self.assertNotIn("query_date", graph.node_sets["queries"].features)

  # Queries / splits tests
  def test_queries_node_set_has_label_and_split(self):
    graph, schema = self._build_synthetic(task="parent-score")
    q_features = graph.node_sets["queries"].features
    q_schema = schema.node_sets["queries"].features
    self.assertCountEqual(
        q_schema.keys(), ["#id", "creation_time", "score", "#split"]
    )
    self.assertEqual(
        q_schema["score"].format, schema_lib.FeatureFormat.FLOAT_32
    )
    self.assertEqual(
        q_schema["score"].semantic, schema_lib.FeatureSemantic.NUMERICAL
    )
    np.testing.assert_allclose(q_features["score"], [1.5, 2.5, 3.5, 4.5, 5.5])
    np.testing.assert_array_equal(
        q_features["#split"],
        [b"train", b"train", b"valid", b"test", b"test"],
    )

  def test_splits_are_non_empty_and_ordered(self):
    graph, _ = self._build_synthetic()
    q_features = graph.node_sets["queries"].features
    splits = q_features["#split"]
    times = q_features["creation_time"]
    train_times = times[splits == b"train"]
    valid_times = times[splits == b"valid"]
    test_times = times[splits == b"test"]
    self.assertNotEmpty(train_times)
    self.assertNotEmpty(valid_times)
    self.assertNotEmpty(test_times)
    self.assertLessEqual(train_times.max(), valid_times.min())
    self.assertLessEqual(valid_times.max(), test_times.min())

  def test_queries_linked_to_entity_table(self):
    graph, _ = self._build_synthetic()
    forward = graph.edge_sets["f2p_queries_parentId"].adjacency
    np.testing.assert_array_equal(forward[0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(forward[1], [0, 1, 2, 1, 0])

  def test_missing_values_filled_with_constant_and_indicator(self):
    tables = _synthetic_tables()
    tables["children"]["nullable_pos"] = pd.Series(
        [1, pd.NA, 5, 9], dtype="Int64"
    )
    tables["children"]["nullable_float"] = np.array(
        [2.0, np.nan, 6.0, np.inf], dtype=np.float64
    )
    tables["children"]["all_nan_column"] = np.array(
        [np.nan, np.nan, np.nan, np.nan], dtype=np.float64
    )
    tables["children"]["nullable_date"] = pd.to_datetime(
        ["1970-01-02", None, "1970-01-03", None]
    )
    tables["children"]["nullable_str"] = ["a", None, "b", "c"]
    task_tables = _synthetic_task_tables()
    task_tables["train"] = pd.concat(
        [
            task_tables["train"],
            pd.DataFrame({
                "query_date": pd.to_datetime(["2021-02-20"]),
                "parentId": np.array([0], dtype=np.int64),
                "score": np.array([np.nan], dtype=np.float64),
                "flag": np.array([0], dtype=np.int64),
            }),
        ],
        ignore_index=True,
    )
    graph, schema = dataset_loader_relbench.build_relational_graph(
        tables=tables,
        task_tables=task_tables,
        spec=_synthetic_spec(),
        task="parent-score",
    )
    self.assertEmpty(in_memory_graph_validate_lib.issues(graph, schema))
    features = graph.node_sets["children"].features
    feature_schemas = schema.node_sets["children"].features
    suffix = dataset_loader_relbench.MISSING_SUFFIX

    # Values are filled with a constant, not with a statistic of the column.
    np.testing.assert_array_equal(features["nullable_pos"], [1, 0, 5, 9])
    self.assertEqual(
        feature_schemas["nullable_pos"].format,
        schema_lib.FeatureFormat.INTEGER_64,
    )
    np.testing.assert_array_equal(
        features["nullable_float"],
        np.array([2.0, 0.0, 6.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        features["all_nan_column"], np.zeros(4, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        features["nullable_date"], [86400, 0, 172800, 0]
    )
    np.testing.assert_array_equal(
        features["nullable_str"], [b"a", b"", b"b", b"c"]
    )

    # Each column with missing values gets a 0/1 indicator feature.
    expected_indicators = {
        "nullable_pos": [0, 1, 0, 0],
        "nullable_float": [0, 1, 0, 1],
        "all_nan_column": [1, 1, 1, 1],
        "nullable_date": [0, 1, 0, 1],
        "nullable_str": [0, 1, 0, 0],
    }
    for col, expected in expected_indicators.items():
      np.testing.assert_array_equal(features[f"{col}{suffix}"], expected)
      self.assertEqual(
          feature_schemas[f"{col}{suffix}"].format,
          schema_lib.FeatureFormat.INTEGER_64,
      )
    # Columns without missing values get no indicator.
    self.assertNotIn(f"value{suffix}", feature_schemas)
    self.assertNotIn(f"rating{suffix}", schema.node_sets["parents"].features)

    # NaN target row in train split should be dropped.
    self.assertEqual(graph.node_sets["queries"].num_nodes, 5)

  def test_missing_indicator_name_collision_raises(self):
    tables = _synthetic_tables()
    tables["children"]["nullable_float"] = np.array(
        [2.0, np.nan, 6.0, 10.0], dtype=np.float64
    )
    tables["children"][
        f"nullable_float{dataset_loader_relbench.MISSING_SUFFIX}"
    ] = np.zeros(4, dtype=np.float64)
    with self.assertRaisesRegex(ValueError, "Cannot add missing indicator"):
      dataset_loader_relbench.build_relational_graph(
          tables=tables,
          task_tables=_synthetic_task_tables(),
          spec=_synthetic_spec(),
          task="parent-score",
      )

  def test_non_finite_regression_target_raises(self):
    task_tables = _synthetic_task_tables()
    task_tables["test"]["score"] = np.array([np.inf, 1.0], dtype=np.float64)
    with self.assertRaisesRegex(ValueError, "1 non-finite values"):
      dataset_loader_relbench.build_relational_graph(
          tables=_synthetic_tables(),
          task_tables=task_tables,
          spec=_synthetic_spec(),
          task="parent-score",
      )

  # Validation error tests
  def test_missing_declared_column_raises(self):
    tables = _synthetic_tables()
    tables["children"] = tables["children"].drop(columns=["event_date"])
    with self.assertRaisesRegex(
        ValueError, "missing declared time column 'event_date'"
    ):
      dataset_loader_relbench.build_relational_graph(
          tables=tables,
          task_tables=_synthetic_task_tables(),
          spec=_synthetic_spec(),
          task="parent-score",
      )

  def test_non_consecutive_primary_key_raises(self):
    tables = _synthetic_tables()
    tables["parents"]["parentId"] = pd.Series([0, 1, 5], dtype="Int64")
    with self.assertRaisesRegex(ValueError, "not the consecutive range"):
      dataset_loader_relbench.build_relational_graph(
          tables=tables,
          task_tables=_synthetic_task_tables(),
          spec=_synthetic_spec(),
          task="parent-score",
      )

  def test_unresolvable_foreign_key_target_raises(self):
    bad_spec = dataset_loader_relbench.RelbenchDatasetSpec(
        tables={
            "parents": dataset_loader_relbench.TableSpec(
                primary_key="parentId",
                foreign_keys={"parentId": "non_existent_table"},
            ),
        },
        tasks=_synthetic_spec().tasks,
    )
    with self.assertRaisesRegex(ValueError, "non_existent_table"):
      dataset_loader_relbench.validate_spec_against_tables(
          bad_spec, {"parents": _synthetic_tables()["parents"]}
      )

  def test_unknown_dataset_or_task_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown RelBench dataset"):
      dataset_loader_relbench.relbench_dataset_spec("rel-unknown")
    with self.assertRaisesRegex(ValueError, "Unknown RelBench task"):
      dataset_loader_relbench.relbench_cns_name("rel-f1", "unknown-task")
    with self.assertRaisesRegex(ValueError, "Unknown RelBench task"):
      dataset_loader_relbench.build_relational_graph(
          tables=_synthetic_tables(),
          task_tables=_synthetic_task_tables(),
          spec=_synthetic_spec(),
          task="unknown-task",
      )

  @parameterized.parameters(
      ("driver-position", "position", False),
      ("driver-dnf", "did_not_finish", True),
      ("driver-top3", "qualifying", True),
  )
  def test_rel_f1_spec_builds_10_node_sets_and_14_edge_sets(
      self, task_name, target_column, is_classification
  ):
    tables, task_tables = _mock_rel_f1_tables()
    spec = dataset_loader_relbench.relbench_dataset_spec("rel-f1")
    graph, schema = dataset_loader_relbench.build_relational_graph(
        tables=tables,
        task_tables=task_tables,
        spec=spec,
        task=task_name,
    )
    self.assertEmpty(in_memory_graph_validate_lib.issues(graph, schema))
    self.assertLen(schema.node_sets, 10)
    self.assertLen(schema.edge_sets, 14)
    self.assertIn(target_column, schema.node_sets["queries"].features)
    expected_semantic = (
        schema_lib.FeatureSemantic.CATEGORICAL
        if is_classification
        else schema_lib.FeatureSemantic.NUMERICAL
    )
    self.assertEqual(
        schema.node_sets["queries"].features[target_column].semantic,
        expected_semantic,
    )

  def test_download_relbench_tables_and_tasks_from_parquet_dir(self):
    tmpdir = self.create_tempdir().full_path
    db_dir = os.path.join(tmpdir, "db")
    task_dir = os.path.join(tmpdir, "tasks", "driver-position")
    os.makedirs(db_dir)
    os.makedirs(task_dir)

    tables, task_tables = _mock_rel_f1_tables()
    for name, df in tables.items():
      df.to_parquet(os.path.join(db_dir, f"{name}.parquet"))
    task_tables["train"].to_parquet(os.path.join(task_dir, "train.parquet"))
    task_tables["valid"].to_parquet(os.path.join(task_dir, "val.parquet"))
    task_tables["test"].to_parquet(os.path.join(task_dir, "test.parquet"))

    loaded_tables = dataset_loader_relbench.download_relbench_tables(
        "rel-f1", source=tmpdir
    )
    loaded_tasks = dataset_loader_relbench.download_relbench_task_tables(
        "rel-f1", "driver-position", source=tmpdir
    )
    self.assertCountEqual(loaded_tables.keys(), tables.keys())
    self.assertCountEqual(loaded_tasks.keys(), ["train", "valid", "test"])

    graph, schema = dataset_loader_relbench.build_relational_graph(
        tables=loaded_tables,
        task_tables=loaded_tasks,
        spec=dataset_loader_relbench.relbench_dataset_spec("rel-f1"),
        task="driver-position",
    )
    self.assertEmpty(in_memory_graph_validate_lib.issues(graph, schema))

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_download_relbench_tables_and_tasks_from_url_zips(self, mock_urlopen):
    # Same layout as the archives published at relbench.stanford.edu/download.
    tables, task_tables = _mock_rel_f1_tables()
    db_zip = _zip_bytes(
        {f"db/{name}.parquet": df for name, df in tables.items()}
    )
    task_zip = _zip_bytes({
        "driver-position/train.parquet": task_tables["train"],
        "driver-position/val.parquet": task_tables["valid"],
        "driver-position/test.parquet": task_tables["test"],
    })
    url_to_content = {
        "https://example.com/rel-f1/db.zip": db_zip,
        "https://example.com/rel-f1/tasks/driver-position.zip": task_zip,
    }

    def fake_urlopen(url, timeout):
      del timeout
      response = mock.MagicMock()
      response.__enter__.return_value.read.return_value = url_to_content[url]
      return response

    mock_urlopen.side_effect = fake_urlopen

    loaded_tables = dataset_loader_relbench.download_relbench_tables(
        "rel-f1", source="https://example.com/rel-f1"
    )
    loaded_tasks = dataset_loader_relbench.download_relbench_task_tables(
        "rel-f1", "driver-position", source="https://example.com/rel-f1"
    )
    self.assertCountEqual(
        [call.args[0] for call in mock_urlopen.call_args_list],
        url_to_content.keys(),
    )
    self.assertCountEqual(loaded_tables.keys(), tables.keys())
    self.assertCountEqual(loaded_tasks.keys(), ["train", "valid", "test"])
    pd.testing.assert_frame_equal(loaded_tables["circuits"], tables["circuits"])
    np.testing.assert_array_equal(
        loaded_tasks["valid"]["position"], task_tables["valid"]["position"]
    )

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_download_relbench_url_zip_missing_member_raises(self, mock_urlopen):
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = _zip_bytes({})
    mock_urlopen.return_value = response
    with self.assertRaisesRegex(ValueError, "'db/circuits.parquet' not found"):
      dataset_loader_relbench.download_relbench_tables(
          "rel-f1", source="https://example.com/rel-f1"
      )

  @mock.patch.object(
      dataset_loader_relbench, "download_relbench_task_tables", autospec=True
  )
  @mock.patch.object(
      dataset_loader_relbench, "download_relbench_tables", autospec=True
  )
  def test_fetch_relbench_graph_forwards_source(
      self, mock_download_tables, mock_download_tasks
  ):
    tables, task_tables = _mock_rel_f1_tables()
    mock_download_tables.return_value = tables
    mock_download_tasks.return_value = task_tables
    source = f"{dataset_loader_relbench.RELBENCH_WEB_URL}/rel-f1"
    dataset_loader_relbench.fetch_relbench_graph(
        dataset="rel-f1",
        task="driver-dnf",
        cache_dir=None,
        repo="WEB",
        source=source,
    )
    mock_download_tables.assert_called_once_with("rel-f1", source=source)
    mock_download_tasks.assert_called_once_with(
        "rel-f1", "driver-dnf", source=source
    )
    with self.assertRaisesRegex(
        ValueError, "only supported with repo=Repo.WEB"
    ):
      dataset_loader_relbench.fetch_relbench_graph(
          dataset="rel-f1",
          task="driver-dnf",
          cache_dir=None,
          repo="CNS",
          source=source,
      )

  @mock.patch.object(
      dataset_loader_relbench, "download_relbench_task_tables", autospec=True
  )
  @mock.patch.object(
      dataset_loader_relbench, "download_relbench_tables", autospec=True
  )
  def test_fetch_relbench_graph_web_and_cache(
      self, mock_download_tables, mock_download_tasks
  ):
    tables, task_tables = _mock_rel_f1_tables()
    mock_download_tables.return_value = tables
    mock_download_tasks.return_value = task_tables

    tmpdir = self.create_tempdir().full_path
    graph, schema = dataset_loader_relbench.fetch_relbench_graph(
        dataset="rel-f1",
        task="driver-position",
        cache_dir=tmpdir,
        repo="WEB",
    )
    self.assertEmpty(in_memory_graph_validate_lib.issues(graph, schema))
    self.assertLen(graph.node_sets, 10)
    self.assertLen(graph.edge_sets, 14)

    # Second call uses cache.
    graph_cached, _ = dataset_loader_relbench.fetch_relbench_graph(
        dataset="rel-f1",
        task="driver-position",
        cache_dir=tmpdir,
        repo="WEB",
    )
    self.assertEqual(mock_download_tables.call_count, 1)
    self.assertEqual(
        graph_cached.node_sets["queries"].num_nodes,
        graph.node_sets["queries"].num_nodes,
    )

  @mock.patch.object(dataset_loader, "load_from_cns", autospec=True)
  def test_fetch_relbench_graph_cns_delegates_to_load_from_cns(
      self, mock_load_from_cns
  ):
    mock_load_from_cns.return_value = (mock.MagicMock(), mock.MagicMock())
    dataset_loader_relbench.fetch_relbench_graph(
        dataset="rel-f1",
        task="driver-position",
        cache_dir=None,
        repo="CNS",
    )
    mock_load_from_cns.assert_called_once_with(
        name="relbench_f1_driver_position"
    )

  @parameterized.named_parameters(
      (
          "driver_position",
          "rel-f1",
          "driver-position",
          "relbench_f1_driver_position",
      ),
      ("driver_dnf", "rel-f1", "driver-dnf", "relbench_f1_driver_dnf"),
      ("driver_top3", "rel-f1", "driver-top3", "relbench_f1_driver_top3"),
  )
  def test_relbench_cns_name(self, dataset, task, expected):
    self.assertEqual(
        dataset_loader_relbench.relbench_cns_name(dataset, task), expected
    )

  @parameterized.parameters(
      ("driver-position",),
      ("driver-dnf",),
      ("driver-top3",),
  )
  @unittest.skipIf(
      os.environ.get("TEST_STRATEGY") != "local",
      "Manual test that downloads RelBench from the web and only runs with"
      " --test_strategy=local",
  )
  def test_real_relbench_f1(self, task):
    graph, schema = dataset_loader_relbench.fetch_relbench_graph(
        dataset="rel-f1",
        task=task,
        cache_dir=None,
        repo="WEB",
    )
    in_memory_graph_validate_lib.validate_graph(graph, schema)
    logging.info(
        "Schema (%s):\n%s",
        task,
        print_schema.print_schema(schema, return_output=True),
    )
    self.assertLen(schema.node_sets, 10)
    self.assertLen(schema.edge_sets, 14)
    for node_set_name, node_set in graph.node_sets.items():
      for feat_name, values in node_set.features.items():
        if np.issubdtype(values.dtype, np.floating):
          self.assertTrue(
              np.all(np.isfinite(values)),
              f"Found non-finite values in {task} {node_set_name}.{feat_name}",
          )
    for node_set_name, min_nodes in (
        ("drivers", 800),
        ("results", 20000),
        ("queries", 1000),
    ):
      num_nodes = graph.node_sets[node_set_name].num_nodes
      self.assertIsNotNone(num_nodes)
      assert num_nodes is not None
      self.assertGreater(num_nodes, min_nodes)


if __name__ == "__main__":
  absltest.main()
