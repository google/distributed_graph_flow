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

"""IO to load a RelBench relational database into an in-memory graph.

RelBench (Robinson et al., NeurIPS 2024, https://relbench.stanford.edu) is the
reference benchmark for Relational Deep Learning: normalized multi-table
databases with primary/foreign keys and row timestamps, modelled as
heterogeneous temporal graphs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import io
import os
import tempfile
import urllib.request
import zipfile

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import cache as cache_lib
from dgf.src.io import dataset_loader
from dgf.src.io.dataset_loader import Repo
from dgf.src.util import log
import dgf.src.util.filesystem as fs
import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class TableSpec:
  """Primary key, timestamp and foreign keys of one RelBench table.

  Attributes:
    primary_key: Name of the primary key column, or None if the table has none.
    time_column: Name of the row timestamp column, or None if the table is
      static.
    foreign_keys: Maps a foreign key column name to the name of the table it
      points into.
  """

  primary_key: str | None = None
  time_column: str | None = None
  foreign_keys: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TaskSpec:
  """Definition of one RelBench predictive task.

  Attributes:
    entity_column: Foreign key column of the task table (e.g. "driverId").
    entity_table: Table the entity column points into (e.g. "drivers").
    time_column: Timestamp column of the task table.
    target_column: Label column of the task table.
    is_classification: Whether the label is categorical.
    hidden_columns: (table, column) pairs to drop from the graph because they
      would leak the label.
  """

  entity_column: str
  entity_table: str
  time_column: str
  target_column: str
  is_classification: bool
  hidden_columns: Sequence[tuple[str, str]] = ()


@dataclasses.dataclass(frozen=True)
class RelbenchDatasetSpec:
  """Layout of one RelBench dataset."""

  tables: Mapping[str, TableSpec]
  tasks: Mapping[str, TaskSpec]


RELBENCH_DATASETS: dict[str, RelbenchDatasetSpec] = {
    "rel-f1": RelbenchDatasetSpec(
        tables={
            "circuits": TableSpec(primary_key="circuitId"),
            "drivers": TableSpec(primary_key="driverId"),
            "constructors": TableSpec(primary_key="constructorId"),
            "races": TableSpec(
                primary_key="raceId",
                time_column="date",
                foreign_keys={"circuitId": "circuits"},
            ),
            "results": TableSpec(
                primary_key="resultId",
                time_column="date",
                foreign_keys={
                    "raceId": "races",
                    "driverId": "drivers",
                    "constructorId": "constructors",
                },
            ),
            "standings": TableSpec(
                primary_key="driverStandingsId",
                time_column="date",
                foreign_keys={"raceId": "races", "driverId": "drivers"},
            ),
            "constructor_results": TableSpec(
                primary_key="constructorResultsId",
                time_column="date",
                foreign_keys={
                    "raceId": "races",
                    "constructorId": "constructors",
                },
            ),
            "constructor_standings": TableSpec(
                primary_key="constructorStandingsId",
                time_column="date",
                foreign_keys={
                    "raceId": "races",
                    "constructorId": "constructors",
                },
            ),
            "qualifying": TableSpec(
                primary_key="qualifyId",
                time_column="date",
                foreign_keys={
                    "raceId": "races",
                    "driverId": "drivers",
                    "constructorId": "constructors",
                },
            ),
        },
        tasks={
            "driver-position": TaskSpec(
                entity_column="driverId",
                entity_table="drivers",
                time_column="date",
                target_column="position",
                is_classification=False,
            ),
            "driver-dnf": TaskSpec(
                entity_column="driverId",
                entity_table="drivers",
                time_column="date",
                target_column="did_not_finish",
                is_classification=True,
            ),
            "driver-top3": TaskSpec(
                entity_column="driverId",
                entity_table="drivers",
                time_column="date",
                target_column="qualifying",
                is_classification=True,
            ),
        },
    ),
}

QUERIES_NODESET = "queries"
SPLIT_FEATURE = "#split"
CREATION_TIME_FEATURE = "creation_time"
CONST_FEATURE = "__const__"
# Suffix of the 0/1 indicator feature added for each column with missing values.
MISSING_SUFFIX = "#missing"
# Timeout of a single HTTP(S) download of a RelBench archive.
_DOWNLOAD_TIMEOUT_SECONDS = 600

# Public RelBench archives: `<dataset>/db.zip` and `<dataset>/tasks/<task>.zip`.
RELBENCH_WEB_URL = "https://relbench.stanford.edu/download"


def relbench_dataset_spec(dataset: str) -> RelbenchDatasetSpec:
  """Returns the spec of `dataset`, raising on an unknown name."""
  if dataset not in RELBENCH_DATASETS:
    raise ValueError(
        f"Unknown RelBench dataset {dataset!r}. Available datasets:"
        f" {sorted(RELBENCH_DATASETS)}."
    )
  return RELBENCH_DATASETS[dataset]


def relbench_cns_name(dataset: str, task: str) -> str:
  """Returns the CNS_GF_REPO name, e.g. "relbench_f1_driver_position"."""
  spec = relbench_dataset_spec(dataset)
  if task not in spec.tasks:
    raise ValueError(
        f"Unknown RelBench task {task!r} for dataset {dataset!r}. Available"
        f" tasks: {sorted(spec.tasks)}."
    )
  dataset_stem = dataset.removeprefix("rel-").replace("-", "_")
  task_stem = task.replace("-", "_")
  return f"relbench_{dataset_stem}_{task_stem}"


def _read_parquet_file(path: str) -> pd.DataFrame:
  """Reads a Parquet file from a local or CNS path."""
  with fs.open_read(path, binary=True) as f:
    return pd.read_parquet(io.BytesIO(f.read()))


def _download_zip(url: str) -> zipfile.ZipFile:
  """Downloads a zip archive into memory."""
  log.info("Downloading %s", url)
  with urllib.request.urlopen(
      url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
  ) as response:
    return zipfile.ZipFile(io.BytesIO(response.read()))


def _read_parquet_from_zip(
    archive: zipfile.ZipFile, member: str, url: str
) -> pd.DataFrame:
  """Reads the Parquet file `member` of `archive`, raising if it is missing."""
  if member not in archive.namelist():
    raise ValueError(
        f"File {member!r} not found in {url}. Available files:"
        f" {archive.namelist()}."
    )
  return pd.read_parquet(io.BytesIO(archive.read(member)))


def download_relbench_tables(
    dataset: str, source: str | None = None
) -> dict[str, pd.DataFrame]:
  """Reads the Parquet tables of `dataset` into DataFrames.

  Args:
    dataset: Name of the RelBench dataset (e.g. "rel-f1").
    source: Optional root of the dataset. Defaults to
      `{RELBENCH_WEB_URL}/{dataset}`. If it is an http(s) URL, the tables are
      read from the `{source}/db.zip` archive published by RelBench. Otherwise,
      they are read from `{source}/db/<table>.parquet` files.

  Returns:
    Mapping from table name to DataFrame.
  """
  spec = relbench_dataset_spec(dataset)
  root = source or f"{RELBENCH_WEB_URL}/{dataset}"
  if fs.is_url(root):
    url = f"{root}/db.zip"
    with _download_zip(url) as archive:
      return {
          table_name: _read_parquet_from_zip(
              archive, f"db/{table_name}.parquet", url
          )
          for table_name in spec.tables
      }

  tables: dict[str, pd.DataFrame] = {}
  for table_name in spec.tables:
    db_path = f"{root}/db/{table_name}.parquet"
    if not fs.exists(db_path):
      flat_path = f"{root}/{table_name}.parquet"
      if fs.exists(flat_path):
        db_path = flat_path
    tables[table_name] = _read_parquet_file(db_path)
  return tables


def download_relbench_task_tables(
    dataset: str, task: str, source: str | None = None
) -> dict[str, pd.DataFrame]:
  """Reads the train/val/test Parquet task tables of `task`.

  Args:
    dataset: Name of the RelBench dataset (e.g. "rel-f1").
    task: Name of the task (e.g. "driver-position").
    source: Optional root of the dataset. Defaults to
      `{RELBENCH_WEB_URL}/{dataset}`. If it is an http(s) URL, the splits are
      read from the `{source}/tasks/{task}.zip` archive published by RelBench.
      Otherwise, they are read from `{source}/tasks/{task}/<split>.parquet`
      files.

  Returns:
    Mapping from split name ("train", "valid", "test") to DataFrame.
  """
  spec = relbench_dataset_spec(dataset)
  if task not in spec.tasks:
    raise ValueError(
        f"Unknown RelBench task {task!r} for dataset {dataset!r}. Available"
        f" tasks: {sorted(spec.tasks)}."
    )
  root = source or f"{RELBENCH_WEB_URL}/{dataset}"
  file_to_split = [("train", "train"), ("val", "valid"), ("test", "test")]
  if fs.is_url(root):
    url = f"{root}/tasks/{task}.zip"
    with _download_zip(url) as archive:
      return {
          split_name: _read_parquet_from_zip(
              archive, f"{task}/{filename}.parquet", url
          )
          for filename, split_name in file_to_split
      }

  task_tables: dict[str, pd.DataFrame] = {}
  for filename, split_name in file_to_split:
    task_path = f"{root}/tasks/{task}/{filename}.parquet"
    if not fs.exists(task_path):
      flat_path = f"{root}/{filename}.parquet"
      if fs.exists(flat_path):
        task_path = flat_path
    task_tables[split_name] = _read_parquet_file(task_path)
  return task_tables


def validate_spec_against_tables(
    spec: RelbenchDatasetSpec, tables: Mapping[str, pd.DataFrame]
) -> None:
  """Raises if the loaded tables do not match `spec`.

  Checks that every declared table exists, every declared primary key,
  timestamp and foreign key column exists, every foreign key resolves to a
  declared table, and every primary key is the consecutive range 0..N-1.

  Args:
    spec: The dataset specification.
    tables: Mapping of table names to pandas DataFrames.

  Raises:
    ValueError: If any table, column, primary key, or foreign key violates the
      specification.
  """
  for table_name, table_spec in spec.tables.items():
    if table_name not in tables:
      raise ValueError(
          f"Table {table_name!r} declared in spec is missing from loaded"
          f" tables {sorted(tables)}."
      )
    df = tables[table_name]
    if table_spec.primary_key is not None:
      if table_spec.primary_key not in df.columns:
        raise ValueError(
            f"Table {table_name!r} is missing declared primary key column"
            f" {table_spec.primary_key!r} (columns: {list(df.columns)})."
        )
      primary_key_series = df[table_spec.primary_key]
      if bool(primary_key_series.isna().any()):
        raise ValueError(
            f"Primary key column {table_spec.primary_key!r} of table"
            f" {table_name!r} contains null values."
        )
      primary_key_values = primary_key_series.to_numpy(dtype=np.int64)
      expected_primary_key = np.arange(len(df), dtype=np.int64)
      if not np.array_equal(primary_key_values, expected_primary_key):
        raise ValueError(
            f"Primary key column {table_spec.primary_key!r} of table"
            f" {table_name!r} is not the consecutive range 0..{len(df) - 1}."
        )
    if table_spec.time_column is not None:
      if table_spec.time_column not in df.columns:
        raise ValueError(
            f"Table {table_name!r} is missing declared time column"
            f" {table_spec.time_column!r} (columns: {list(df.columns)})."
        )
    for foreign_key_column, parent_table in table_spec.foreign_keys.items():
      if foreign_key_column not in df.columns:
        raise ValueError(
            f"Table {table_name!r} is missing declared foreign key column"
            f" {foreign_key_column!r} (columns: {list(df.columns)})."
        )
      if parent_table not in spec.tables or parent_table not in tables:
        raise ValueError(
            f"Foreign key {table_name}.{foreign_key_column} points to"
            f" undeclared or missing target table {parent_table!r}."
        )
      parent_num_rows = len(tables[parent_table])
      foreign_key_series = df[foreign_key_column]
      valid_foreign_keys = foreign_key_series[
          ~foreign_key_series.isna()
      ].to_numpy(dtype=np.int64)
      if len(valid_foreign_keys) > 0 and (
          np.any(valid_foreign_keys < 0)
          or np.any(valid_foreign_keys >= parent_num_rows)
      ):
        raise ValueError(
            f"Foreign key {table_name}.{foreign_key_column} contains indices"
            f" outside [0, {parent_num_rows}) of target table {parent_table!r}."
        )


def _datetime_to_unix_seconds(
    series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
  """Converts a datetime series to int64 unix seconds and a NaT mask.

  NaT entries are set to 0 in the returned seconds.

  Args:
    series: The datetime series to convert.

  Returns:
    A tuple `(seconds, missing_mask)`.
  """
  datetimes = pd.to_datetime(series)
  missing_mask = datetimes.isna().to_numpy()
  seconds = (
      datetimes.dt.as_unit("s").to_numpy(dtype="datetime64[s]").astype(np.int64)
  )
  seconds = np.where(missing_mask, 0, seconds).astype(np.int64)
  return seconds, missing_mask


def _to_unix_seconds(series: pd.Series) -> np.ndarray:
  """Converts a datetime series to int64 unix seconds, raising on NaT."""
  seconds, missing_mask = _datetime_to_unix_seconds(series)
  if np.any(missing_mask):
    raise ValueError(f"Timestamp column {series.name!r} contains NaT values.")
  return seconds


def _infer_column_feature(
    series: pd.Series,
) -> tuple[np.ndarray, schema_lib.FeatureSchema, np.ndarray]:
  """Converts a pandas column into a numpy array, a schema and a missing mask.

  Missing values are replaced by a constant (0 for numerical and timestamp
  columns, b"" for string columns) rather than by a statistic of the column.
  Statistics such as the median would be computed over all rows, including
  rows from the validation and test periods, and would leak future information
  into the features. Instead, the returned missing mask lets the caller expose
  missingness as an explicit indicator feature.

  Args:
    series: The pandas column to convert.

  Returns:
    A tuple `(values, schema, missing_mask)` where `missing_mask[i]` is True if
    row `i` was missing (NaN, NaT, None, NA or +/-Inf) in `series`.
  """
  dtype = series.dtype
  numerical_int64 = schema_lib.FeatureSchema(
      format=schema_lib.FeatureFormat.INTEGER_64,
      semantic=schema_lib.FeatureSemantic.NUMERICAL,
  )
  if pd.api.types.is_datetime64_any_dtype(dtype):
    values, missing_mask = _datetime_to_unix_seconds(series)
    return values, numerical_int64, missing_mask
  if pd.api.types.is_bool_dtype(dtype):
    missing_mask = series.isna().to_numpy()
    values = series.fillna(False).to_numpy(dtype=np.int64)
    return values, numerical_int64, missing_mask
  if pd.api.types.is_integer_dtype(dtype):
    missing_mask = series.isna().to_numpy()
    values = series.fillna(0).to_numpy(dtype=np.int64)
    return values, numerical_int64, missing_mask
  if pd.api.types.is_float_dtype(dtype):
    raw_values = series.to_numpy(dtype=np.float32, na_value=np.nan)
    missing_mask = ~np.isfinite(raw_values)
    values = np.where(missing_mask, np.float32(0), raw_values).astype(
        np.float32
    )
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )
    return values, schema, missing_mask
  if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
    missing_mask = series.isna().to_numpy()
    values = np.array(
        [
            value
            if isinstance(value, bytes)
            else (value.encode("utf-8") if isinstance(value, str) else b"")
            for value in series
        ],
        dtype=np.bytes_,
    )
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
    )
    return values, schema, missing_mask
  raise ValueError(
      f"Unsupported column dtype {dtype!r} for column {series.name!r}."
  )


def _build_foreign_key_edge_set(
    child_table: str,
    foreign_key_column: str,
    parent_table: str,
    foreign_key_series: pd.Series,
    num_parent_nodes: int,
) -> tuple[str, in_memory_graph_lib.InMemoryEdgeSet, schema_lib.EdgeSchema]:
  """Builds the child -> parent (`f2p_*`) edge set of a foreign key.

  No reverse edge set is created: the DGF graph sampler already traverses
  every edge set in both directions.

  Unlike the other DGF loaders, which name edge sets `<source>_to_<target>`,
  edge sets follow the RelBench / PyG `f2p_<child>_<foreign_key>` convention. A
  child table can hold several foreign keys into the same parent table, so the
  foreign key column must be part of the name to keep edge set names unique.

  Args:
    child_table: Name of the table holding the foreign key.
    foreign_key_column: Name of the foreign key column.
    parent_table: Name of the table the foreign key points into.
    foreign_key_series: Values of the foreign key column. Null values are
      skipped.
    num_parent_nodes: Number of rows of `parent_table`.

  Returns:
    The `(name, edge_set, edge_schema)` of the edge set.
  """
  mask = ~foreign_key_series.isna().to_numpy()
  child_indices = np.arange(len(foreign_key_series), dtype=np.int64)[mask]
  parent_indices = foreign_key_series[mask].to_numpy(dtype=np.int64)
  if len(parent_indices) > 0 and (
      np.any(parent_indices < 0) or np.any(parent_indices >= num_parent_nodes)
  ):
    raise ValueError(
        f"Foreign key {child_table}.{foreign_key_column} contains indices"
        f" outside [0, {num_parent_nodes}) of target table {parent_table!r}."
    )

  name = f"f2p_{child_table}_{foreign_key_column}"
  edge_set = in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=np.stack([child_indices, parent_indices], axis=0),
      features={},
  )
  edge_schema = schema_lib.EdgeSchema(
      source=child_table,
      target=parent_table,
      features={},
  )
  return name, edge_set, edge_schema


def build_relational_graph(
    tables: Mapping[str, pd.DataFrame],
    task_tables: Mapping[str, pd.DataFrame],
    spec: RelbenchDatasetSpec,
    task: str,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Builds the heterogeneous relational graph of a RelBench task.

  Args:
    tables: Mapping from table name to DataFrame.
    task_tables: Mapping from split name ("train", "valid", "test") to
      DataFrame.
    spec: RelBench dataset specification.
    task: Name of the predictive task within `spec.tasks`.

  Returns:
    A tuple of `(InMemoryGraph, GraphSchema)`.
  """
  if task not in spec.tasks:
    raise ValueError(
        f"Unknown RelBench task {task!r}. Available tasks:"
        f" {sorted(spec.tasks)}."
    )
  task_spec = spec.tasks[task]
  validate_spec_against_tables(spec, tables)

  if task_spec.entity_table not in spec.tables:
    raise ValueError(
        f"Task {task!r} entity table {task_spec.entity_table!r} is not a"
        f" declared table in {sorted(spec.tables)}."
    )

  hidden_by_table: dict[str, set[str]] = {}
  for table, column in task_spec.hidden_columns:
    hidden_by_table.setdefault(table, set()).add(column)

  node_sets: dict[str, in_memory_graph_lib.InMemoryNodeSet] = {}
  node_schemas: dict[str, schema_lib.NodeSchema] = {}
  edge_sets: dict[str, in_memory_graph_lib.InMemoryEdgeSet] = {}
  edge_schemas: dict[str, schema_lib.EdgeSchema] = {}

  for table_name, table_spec in spec.tables.items():
    df = tables[table_name]
    num_nodes = len(df)
    excluded_columns: set[str] = set(hidden_by_table.get(table_name, set()))
    if table_spec.primary_key is not None:
      excluded_columns.add(table_spec.primary_key)
    if table_spec.time_column is not None:
      excluded_columns.add(table_spec.time_column)
    excluded_columns.update(table_spec.foreign_keys.keys())

    feature_columns = [
        column for column in df.columns if column not in excluded_columns
    ]

    features: dict[str, np.ndarray] = {
        "#id": dataset_loader.generate_ids(f"{table_name}_", num_nodes),
    }
    feature_schemas: dict[str, schema_lib.FeatureSchema] = {
        "#id": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BYTES,
            semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
        ),
    }

    if table_spec.time_column is not None:
      features[CREATION_TIME_FEATURE] = _to_unix_seconds(
          df[table_spec.time_column]
      )
      feature_schemas[CREATION_TIME_FEATURE] = schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.TIMESTAMP,
          is_creation_time=True,
      )

    if not feature_columns:
      features[CONST_FEATURE] = np.ones(num_nodes, dtype=np.float32)
      feature_schemas[CONST_FEATURE] = schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      )
    else:
      for column in feature_columns:
        column_values, column_schema, missing_mask = _infer_column_feature(
            df[column]
        )
        features[column] = column_values
        feature_schemas[column] = column_schema
        num_missing = int(np.sum(missing_mask))
        if num_missing == 0:
          continue
        missing_column = f"{column}{MISSING_SUFFIX}"
        if missing_column in df.columns:
          raise ValueError(
              f"Cannot add missing indicator {missing_column!r} to table"
              f" {table_name!r}: a column with this name already exists."
          )
        log.info(
            "%s.%s: %d/%d missing values filled with a constant; adding"
            " indicator feature %r.",
            table_name,
            column,
            num_missing,
            num_nodes,
            missing_column,
        )
        features[missing_column] = missing_mask.astype(np.int64)
        feature_schemas[missing_column] = schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.NUMERICAL,
        )

    node_sets[table_name] = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=num_nodes,
        features=features,
    )
    node_schemas[table_name] = schema_lib.NodeSchema(features=feature_schemas)

    for foreign_key_column, parent_table in table_spec.foreign_keys.items():
      name, edge_set, edge_schema = _build_foreign_key_edge_set(
          child_table=table_name,
          foreign_key_column=foreign_key_column,
          parent_table=parent_table,
          foreign_key_series=df[foreign_key_column],
          num_parent_nodes=len(tables[parent_table]),
      )
      edge_sets[name] = edge_set
      edge_schemas[name] = edge_schema

  # Build queries node set from train, valid, test task tables.
  split_order = ("train", "valid", "test")
  split_dfs: list[pd.DataFrame] = []
  split_labels_parts: list[np.ndarray] = []
  required_task_columns = (
      task_spec.entity_column,
      task_spec.time_column,
      task_spec.target_column,
  )
  for split_name in split_order:
    if split_name not in task_tables:
      raise ValueError(
          f"Missing split {split_name!r} in task_tables {sorted(task_tables)}."
      )
    split_df = task_tables[split_name]
    for column in required_task_columns:
      if column not in split_df.columns:
        raise ValueError(
            f"Task split {split_name!r} for task {task!r} is missing required"
            f" column {column!r} (columns: {list(split_df.columns)})."
        )
    num_rows = len(split_df)
    split_df = split_df.dropna(subset=list(required_task_columns)).reset_index(
        drop=True
    )
    num_dropped = num_rows - len(split_df)
    if num_dropped:
      log.info(
          "Dropping %d of the %d %s queries of task %s with a missing entity,"
          " timestamp or target.",
          num_dropped,
          num_rows,
          split_name,
          task,
      )
    split_dfs.append(split_df)
    split_labels_parts.append(
        np.full(len(split_df), split_name.encode("utf-8"), dtype="S5")
    )

  queries_df = pd.concat(split_dfs, ignore_index=True)
  split_labels = np.concatenate(split_labels_parts, axis=0)
  num_queries = len(queries_df)

  if task_spec.is_classification:
    target_values = queries_df[task_spec.target_column].to_numpy(dtype=np.int64)
    target_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        num_categorical_values=dataset_loader.get_num_classes(target_values),
    )
  else:
    target_values = queries_df[task_spec.target_column].to_numpy(
        dtype=np.float32, na_value=np.nan
    )
    num_non_finite = int(np.sum(~np.isfinite(target_values)))
    if num_non_finite > 0:
      raise ValueError(
          f"Target column {task_spec.target_column!r} of task {task!r}"
          f" contains {num_non_finite} non-finite values (after float32"
          " conversion)."
      )
    target_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )

  query_features: dict[str, np.ndarray] = {
      "#id": dataset_loader.generate_ids("query_", num_queries),
      CREATION_TIME_FEATURE: _to_unix_seconds(
          queries_df[task_spec.time_column]
      ),
      task_spec.target_column: target_values,
      SPLIT_FEATURE: split_labels,
  }
  query_feature_schemas: dict[str, schema_lib.FeatureSchema] = {
      "#id": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
      ),
      CREATION_TIME_FEATURE: schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.TIMESTAMP,
          is_creation_time=True,
      ),
      task_spec.target_column: target_schema,
      SPLIT_FEATURE: schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
      ),
  }

  node_sets[QUERIES_NODESET] = in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=num_queries,
      features=query_features,
  )
  node_schemas[QUERIES_NODESET] = schema_lib.NodeSchema(
      features=query_feature_schemas
  )

  name, edge_set, edge_schema = _build_foreign_key_edge_set(
      child_table=QUERIES_NODESET,
      foreign_key_column=task_spec.entity_column,
      parent_table=task_spec.entity_table,
      foreign_key_series=queries_df[task_spec.entity_column],
      num_parent_nodes=len(tables[task_spec.entity_table]),
  )
  edge_sets[name] = edge_set
  edge_schemas[name] = edge_schema

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets=node_sets,
      edge_sets=edge_sets,
  )
  schema = schema_lib.GraphSchema(
      node_sets=node_schemas,
      edge_sets=edge_schemas,
  )
  return graph, schema


def fetch_relbench_graph(
    dataset: str,
    task: str,
    cache_dir: str | None = "AUTO",
    verbose: bool = True,
    repo: Repo | str = Repo.AUTO,
    source: str | None = None,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Downloads and loads a RelBench task into an in-memory graph.

  RelBench (Robinson et al., NeurIPS 2024, https://relbench.stanford.edu) is a
  benchmark of normalized relational databases with primary/foreign keys and row
  timestamps, modelled as heterogeneous temporal graphs.

  Usage example:

  ```python
  graph, schema = dgf.io.fetch_relbench_graph("rel-f1", "driver-position")
  dgf.analyse.print_schema(schema)
  ```

  Args:
    dataset: Name of the RelBench dataset (e.g. "rel-f1").
    task: Name of the predictive task (e.g. "driver-position", "driver-dnf",
      "driver-top3").
    cache_dir: Optional. Directory to cache the graph to avoid re-loading it
      each time. If "AUTO", uses the OS default temporary directory. If None,
      does not cache the graph.
    verbose: Optional. Whether to print cache and load progress.
    repo: Define the source of the data (Repo.AUTO, Repo.CNS, Repo.WEB).
    source: Optional root of the raw RelBench dataset, only used with
      `Repo.WEB`: either an http(s) URL serving the RelBench zip archives (e.g.
      f"{RELBENCH_WEB_URL}/rel-f1") or a directory of Parquet files. If None,
      uses `{RELBENCH_WEB_URL}/{dataset}`.

  Returns:
    An InMemoryGraph instance and its GraphSchema.
  """
  spec = relbench_dataset_spec(dataset)
  cns_name = relbench_cns_name(dataset, task)

  if cache_dir == "AUTO":
    cache_dir = os.path.join(tempfile.gettempdir(), "gf_fetch_relbench")

  if cache_dir is not None:
    fs.makedirs(cache_dir)
    cache_graph_path = os.path.join(cache_dir, f"{cns_name}.cache")
    if verbose:
      log.info(
          "Caching the %s (%s) graph at %s", dataset, task, cache_graph_path
      )
  else:
    cache_graph_path = None

  if isinstance(repo, str):
    repo = Repo(repo)

  if repo == Repo.AUTO:
    repo = Repo.WEB

  if repo == Repo.CNS:
    if source is not None:
      raise ValueError("`source` is only supported with repo=Repo.WEB.")
    loader = functools.partial(dataset_loader.load_from_cns, name=cns_name)
  elif repo == Repo.WEB:

    def loader():
      if verbose:
        log.info(
            "Loading the %s (%s) data from %s",
            dataset,
            task,
            source or f"{RELBENCH_WEB_URL}/{dataset}",
        )
      tables = download_relbench_tables(dataset, source=source)
      task_tables = download_relbench_task_tables(dataset, task, source=source)
      return build_relational_graph(
          tables=tables,
          task_tables=task_tables,
          spec=spec,
          task=task,
      )

  else:
    raise ValueError(f"Unsupported repo for RelBench datasets: {repo}")

  if cache_graph_path is None:
    return loader()
  else:
    return cache_lib.cache(cache_graph_path, loader)
