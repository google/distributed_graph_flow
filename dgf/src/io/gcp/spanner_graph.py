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

"""Reads Google Cloud Spanner Property Graphs into memory using direct SQL."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import spanner_graph as spanner_graph_lib
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.util import log
from google.cloud import spanner as gcp_spanner
from google.cloud.spanner_v1 import database as gcp_database_lib

# Type of a query parameter of each Spanner type.
PARAM_TYPES: dict[str, Any] = {
    spanner_graph_lib.TYPE_BOOL: gcp_spanner.param_types.BOOL,
    spanner_graph_lib.TYPE_BYTES: gcp_spanner.param_types.BYTES,
    spanner_graph_lib.TYPE_DATE: gcp_spanner.param_types.DATE,
    spanner_graph_lib.TYPE_INT64: gcp_spanner.param_types.INT64,
    spanner_graph_lib.TYPE_STRING: gcp_spanner.param_types.STRING,
    spanner_graph_lib.TYPE_TIMESTAMP: gcp_spanner.param_types.TIMESTAMP,
}


def _execute_query(
    database_or_snapshot: Any,
    query: str,
    params: dict[str, Any] | None = None,
    param_types: dict[str, Any] | None = None,
) -> Any:
  """Executes a Cloud Spanner SQL query on a Database or Snapshot."""
  if isinstance(database_or_snapshot, gcp_database_lib.Database):
    with database_or_snapshot.snapshot() as snapshot:
      # The rows are streamed: Read them before the snapshot is released.
      return list(
          snapshot.execute_sql(query, params=params, param_types=param_types)
      )
  return database_or_snapshot.execute_sql(
      query, params=params, param_types=param_types
  )


def get_metadata(
    database_or_snapshot: Any,
    graph: str,
) -> spanner_graph_metadata_lib.SpannerGraphMetadata:
  """Reads the definition of a property graph of a Cloud Spanner database."""
  rows = _execute_query(
      database_or_snapshot,
      spanner_graph_lib.property_graph_metadata_query(
          "PROPERTY_GRAPH_METADATA_JSON"
      ),
      params={"graph_name": graph},
      param_types={"graph_name": gcp_spanner.param_types.STRING},
  )
  return spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
      spanner_graph_lib.single_property_graph_row(rows, graph)
  )


def read_column_spanner_types_gcp(
    database_or_snapshot: Any,
) -> dict[tuple[str, str], str]:
  """Reads the Spanner type of each (table_name, column_name)."""
  return spanner_graph_lib.column_types_from_rows(
      _execute_query(database_or_snapshot, spanner_graph_lib.COLUMNS_TYPE_QUERY)
  )


def read_table_indexes_gcp(
    database_or_snapshot: Any,
) -> dict[str, list[spanner_graph_lib.TableIndex]]:
  """Reads the primary key and the secondary indexes of each table."""
  return spanner_graph_lib.table_indexes_from_rows(
      _execute_query(
          database_or_snapshot, spanner_graph_lib.INDEX_COLUMNS_QUERY
      )
  )


def graph_data_read_query(
    graph: str,
    graph_element_type: str,
    graph_element_table: (
        spanner_graph_metadata_lib.NodeTable
        | spanner_graph_metadata_lib.EdgeTable
    ),
) -> str:
  """Builds a GQL query string reading Spanner graph elements via Beam."""
  graph_element_labels_string = " & ".join(graph_element_table.label_names)
  has_properties = bool(graph_element_table.property_definitions)
  gql_base_query = gcp_common_lib.gql_base(
      graph_id=graph,
      graph_element_type=graph_element_type,
      graph_element_labels_string=graph_element_labels_string,
      graph_element_table_name=graph_element_table.name,
      omit_json=not has_properties,
  )
  return f"@{{FORCE_INDEX=_base_table}}\n{gql_base_query}"


def read_spanner_graph_schema_gcp(
    project: str,
    instance: str,
    database: str,
    graph: str,
    *,
    verbose: int = 1,
) -> schema_lib.GraphSchema:
  """Reads the DGF schema of a Google Cloud Spanner Property Graph.

  The nodesets and edgesets are named after the element tables of the property
  graph, and the features after its properties. A node table with a composite
  key gets a "#id" BYTES primary key feature holding its key columns joined
  with "/". Arrays declared with a vector_length have a static shape.

  Usage example:

  ```python
  schema = dgf.io.read_spanner_graph_schema_gcp(
      project="my-project",
      instance="my-instance",
      database="my-database",
      graph="my_graph",
  )
  ```

  Args:
    project: GCP project of the Spanner instance.
    instance: Spanner instance.
    database: Spanner database.
    graph: Name of the Spanner property graph.
    verbose: Verbosity (0 = silent, 1 = progress, 2 = debug).

  Returns:
    The DGF schema.
  """
  db = gcp_spanner.Client(project=project).instance(instance).database(database)
  with db.snapshot(multi_use=True) as snapshot:
    snapshot.begin()
    metadata = get_metadata(snapshot, graph)
    column_spanner_types = read_column_spanner_types_gcp(snapshot)

  if verbose >= 2:
    log.info("meta-data:\n%s", metadata)
  return spanner_graph_lib.graph_schema_from_metadata(
      metadata,
      column_spanner_types=column_spanner_types,
  )


def read_spanner_graph_gcp(
    project: str,
    instance: str,
    database: str,
    graph: str,
    *,
    schema: schema_lib.GraphSchema | None = None,
    filters: Mapping[str, Any] | None = None,
    force_index: bool | None = None,
    verbose: int = 1,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a Google Cloud Spanner Property Graph into an in-memory graph.

  The tables of the property graph are read with one SQL query each, all in the
  same snapshot. The whole graph is loaded in memory: To train on a large
  graph, sample it with dgf.sampling.create_spanner_graph_sampler_gcp instead.

  With filters, a table is read fast if the filtered columns are the first
  columns of its primary key or of one of its secondary indexes.

  A node table with a composite key gets a "#id" BYTES primary key feature
  holding its key columns joined with "/". NULL values are read as the DGF
  default values.

  Usage example:

  ```python
  graph, schema = dgf.io.read_spanner_graph_gcp(
      project="my-project",
      instance="my-instance",
      database="my-database",
      graph="my_graph",
  )
  ```

  Args:
    project: GCP project of the Spanner instance.
    instance: Spanner instance.
    database: Spanner database.
    graph: Name of the Spanner property graph.
    schema: The nodesets, edgesets and features to read. If None, reads the
      schema returned by read_spanner_graph_schema_gcp.
    filters: Only the rows whose column equals the value are read, for each
      {column_name: value}, e.g. {"snapshot": my_datetime}. All the tables read
      must have these columns, of type BOOL, DATE, INT64, STRING or TIMESTAMP.
    force_index: Whether to force an index hint on the SQL queries. If None,
      forces a secondary index when one starts with filters. If True, requires
      filters and raises ValueError if a read table has no index starting with
      filters. If False, forces _base_table to disable secondary indexes.
    verbose: Verbosity (0 = silent, 1 = progress, 2 = debug).

  Returns:
    The in-memory graph and its schema.
  """
  db = gcp_spanner.Client(project=project).instance(instance).database(database)
  with db.snapshot(multi_use=True) as snapshot:
    snapshot.begin()

    def execute_fn(
        query: str,
        parameters: Mapping[str, Any],
        parameter_types: Mapping[str, str],
    ) -> Iterable[Sequence[Any]]:
      return snapshot.execute_sql(
          query,
          params=parameters,
          param_types={
              name: PARAM_TYPES[parameter_type]
              for name, parameter_type in parameter_types.items()
          },
      )

    table_indexes = None
    if filters and (force_index is None or force_index):
      table_indexes = read_table_indexes_gcp(snapshot)
    return spanner_graph_lib.read_graph_with_executor(
        metadata=get_metadata(snapshot, graph),
        execute_fn=execute_fn,
        # The Cloud client returns the BYTES values in base64.
        bytes_are_base64=True,
        schema=schema,
        column_spanner_types=read_column_spanner_types_gcp(snapshot),
        filters=filters,
        table_indexes=table_indexes,
        force_index=force_index,
        verbose=verbose,
    )
