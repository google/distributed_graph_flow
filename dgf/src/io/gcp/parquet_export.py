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

"""Export BigQuery Graph data from underlying tables to Parquet files."""

import os

from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.io.gcp import bigquery_graph_metadata as bqgm_lib
from dgf.src.util import log

_BQ_TYPE_ARRAY = "ARRAY"
_BQ_TYPE_TIMESTAMP = "TIMESTAMP"
# DGF default values of missing (NULL) values, per BigQuery type.
_BQ_TYPE_TO_DEFAULT_SQL = {
    "BOOL": "FALSE",
    "BYTES": "b''",
    "FLOAT64": "0.0",
    "INT64": "0",
    "STRING": "''",
    _BQ_TYPE_TIMESTAMP: "0",
}


def _endpoint_projection(columns: list[str], target_key: str) -> str:
  """Exports the single column referencing an edge endpoint as `target_key`."""
  if len(columns) != 1:
    raise NotImplementedError(
        f"Composite keys are not supported. Got key columns {columns}."
    )
  return f"`{columns[0]}` AS `{target_key}`"


def _property_projection(property_: bqgm_lib.Property) -> str:
  """Exports a property with timestamps as micros and NULLs as DGF defaults."""
  column = f"`{property_.expression}`"
  data_type = property_.data_type
  if data_type.type_kind == _BQ_TYPE_ARRAY:
    # BigQuery arrays are never NULL.
    assert data_type.array_element_type is not None
    if data_type.array_element_type.type_kind == _BQ_TYPE_TIMESTAMP:
      column = f"ARRAY(SELECT UNIX_MICROS(x) FROM UNNEST({column}) AS x)"
  else:
    if data_type.type_kind == _BQ_TYPE_TIMESTAMP:
      column = f"UNIX_MICROS({column})"
    if data_type.type_kind in _BQ_TYPE_TO_DEFAULT_SQL:
      column = (
          f"COALESCE({column}, {_BQ_TYPE_TO_DEFAULT_SQL[data_type.type_kind]})"
      )
  return f"{column} AS `{property_.name}`"


def create_nodeset_sql(
    node_table: bqgm_lib.NodeTable,
    verbose: int = 1,
) -> str:
  """Creates BQ Graph nodeset data to parquet export SQL from underlying table."""
  bq_table_id = (
      f"`{node_table.data_source_table.project_id}."
      f"{node_table.data_source_table.dataset_id}."
      f"{node_table.data_source_table.table_id}`"
  )

  projections = set()

  for label_and_properties in node_table.label_and_properties:
    if label_and_properties.properties:
      for property_ in label_and_properties.properties:
        projections.add(_property_projection(property_))

  select_clause = ", ".join(projections)
  nodeset_sql = f"""
        SELECT
          {select_clause}
        FROM {bq_table_id}
      """

  if verbose >= 2:
    log.info("Nodeset SQL: %s", nodeset_sql)
  return nodeset_sql


def create_edgeset_sql(
    edge_table: bqgm_lib.EdgeTable,
    verbose: int = 1,
) -> str:
  """Creates BQ Graph edgeset data to parquet export SQL from underlying table."""

  bq_table_id = (
      f"`{edge_table.data_source_table.project_id}."
      f"{edge_table.data_source_table.dataset_id}."
      f"{edge_table.data_source_table.table_id}`"
  )
  projections = set()
  projections.add(
      _endpoint_projection(
          edge_table.source_node_reference.edge_table_columns,
          gf_graph_in_beam_lib.KEY_SOURCE,
      )
  )
  projections.add(
      _endpoint_projection(
          edge_table.destination_node_reference.edge_table_columns,
          gf_graph_in_beam_lib.KEY_TARGET,
      )
  )
  for label_and_properties in edge_table.label_and_properties:
    if label_and_properties.properties:
      for property_ in label_and_properties.properties:
        projections.add(_property_projection(property_))

  select_clause = ", ".join(projections)
  edgeset_sql = f"""
        SELECT
          {select_clause} 
        FROM {bq_table_id}
      """
  if verbose >= 2:
    log.info("Edgeset SQL: %s", edgeset_sql)
  return edgeset_sql


## TODO(tewariy): Error while reading data, error message: Type JSON is not currently supported for parquet exports.
## TODO(tewariy): Add support for combining as JSON.
def create_export_sql(
    export_query: str,
    gcs_prefix: str,
    graph_element_prefix: str,
    graph_element_name: str,
    verbose: int = 1,
) -> str:
  """Creates BQ Graph graph element data to parquet export SQL from underlying table."""
  export_path = os.path.join(
      gcs_prefix,
      graph_element_prefix,
      f"{graph_element_name}-*.parquet",
  )

  export_sql = f"""
      EXPORT DATA
        OPTIONS (
          URI = '{export_path}',
          FORMAT = 'parquet',
          OVERWRITE = TRUE,
          COMPRESSION = 'NONE')
      AS
        {export_query}
      """
  if verbose >= 2:
    log.info("Export SQL: %s", export_sql)
  return export_sql
