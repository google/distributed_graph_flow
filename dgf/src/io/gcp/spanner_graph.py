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

from collections.abc import Iterable, Sequence
import time
from typing import Any

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import spanner_graph as spanner_graph_lib
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.util import log
from google.cloud import spanner as gcp_spanner
from google.cloud.spanner_v1 import database as gcp_database_lib


def _infoschema_query() -> str:
  """Gets the SQL query reading Cloud Spanner property graph metadata."""
  return (
      "SELECT PROPERTY_GRAPH_METADATA_JSON FROM"
      " information_schema.property_graphs WHERE property_graph_name ="
      " @graph_name"
  )


def _execute_query(
    database_or_snapshot: Any,
    query: str,
    params: dict[str, Any] | None = None,
    param_types: dict[str, Any] | None = None,
) -> Any:
  """Executes a Cloud Spanner SQL query on a Database or Snapshot."""
  if isinstance(database_or_snapshot, gcp_database_lib.Database):
    with database_or_snapshot.snapshot() as snapshot:
      return snapshot.execute_sql(query, params=params, param_types=param_types)
  return database_or_snapshot.execute_sql(
      query, params=params, param_types=param_types
  )


def get_metadata(
    database_or_snapshot: Any,
    graph: str,
) -> spanner_graph_metadata_lib.SpannerGraphMetadata:
  """Loads Cloud Spanner property graph metadata."""
  spanner_graph_lib.validate_identifier(graph)
  metadata_query = _infoschema_query()
  log.info("Metadata query: %s (with graph=%s)", metadata_query, graph)

  result_set = _execute_query(
      database_or_snapshot,
      metadata_query,
      params={"graph_name": graph},
      param_types={"graph_name": gcp_spanner.param_types.STRING},
  )
  rows = list(result_set)
  if len(rows) != 1:
    raise ValueError(
        f"Expected one property graph named {graph!r}, found {len(rows)}."
    )
  return spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
      rows[0][0]
  )


def read_column_spanner_types_gcp(
    database_or_snapshot: Any,
) -> dict[tuple[str, str], str]:
  """Reads (table_name, column_name) -> spanner_type from INFORMATION_SCHEMA."""
  rows = _execute_query(
      database_or_snapshot, spanner_graph_lib.COLUMNS_TYPE_QUERY
  )
  return {
      (str(table_name), str(column_name)): str(spanner_type)
      for table_name, column_name, spanner_type in rows
  }


def graph_schema(
    metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
    combine_as_json: bool = False,
    column_spanner_types: dict[tuple[str, str], str] | None = None,
) -> schema_lib.GraphSchema:
  """Builds a DGF GraphSchema from Spanner property graph metadata."""
  return spanner_graph_lib.graph_schema_from_metadata(
      metadata,
      combine_as_json=combine_as_json,
      column_spanner_types=column_spanner_types,
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
    combine_as_json: bool = False,
    verbose: int = 1,
) -> schema_lib.GraphSchema:
  """Reads the DGF GraphSchema of a Google Cloud Spanner Property Graph.

  Composite primary keys on Spanner node tables are represented by a single
  synthesized #id BYTES primary key feature, while individual key columns
  remain available as regular features.

  Usage example:

    schema = dgf.io.read_spanner_graph_schema_gcp(
        project="my-project",
        instance="my-instance",
        database="my-database",
        graph="my_graph",
    )

  Args:
    project: GCP project ID of the Spanner instance.
    instance: Cloud Spanner instance ID.
    database: Cloud Spanner database ID.
    graph: Name of the Spanner property graph.
    combine_as_json: If True, stores all element properties serialized inside a
      single __attributes__ JSON string feature.
    verbose: Logging verbosity level (0 = silent, 1 = progress, 2 = debug).

  Returns:
    The inferred DGF GraphSchema.
  """
  spanner_client = gcp_spanner.Client(project=project)
  db = spanner_client.instance(instance).database(database)
  with db.snapshot(multi_use=True) as snapshot:
    snapshot.begin()
    metadata = get_metadata(snapshot, graph)
    column_spanner_types = read_column_spanner_types_gcp(snapshot)

  if verbose >= 2:
    log.info("meta-data:\n%s", metadata)
  return graph_schema(metadata, combine_as_json, column_spanner_types)


def read_spanner_graph_gcp(
    project: str,
    instance: str,
    database: str,
    graph: str,
    *,
    schema: schema_lib.GraphSchema | None = None,
    combine_as_json: bool = False,
    verbose: int = 1,
    **kwargs: Any,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a Google Cloud Spanner Property Graph into an in-memory DGF graph.

  Executes direct SQL queries against the base tables of the property graph
  within a single multi-use snapshot so all nodesets and edgesets reflect a
  consistent Spanner read timestamp. Multi-column composite primary keys are
  encoded into #id BYTES values by joining escaped components with '/'.

  Usage example:

    graph, schema = dgf.io.read_spanner_graph_gcp(
        project="my-project",
        instance="my-instance",
        database="my-database",
        graph="my_graph",
    )

  Args:
    project: GCP project ID of the Spanner instance.
    instance: Cloud Spanner instance ID.
    database: Cloud Spanner database ID.
    graph: Name of the Spanner property graph.
    schema: Optional DGF GraphSchema selecting which nodesets, edgesets, and
      features to read. If None, the full schema is inferred from Spanner.
    combine_as_json: If True, stores all element properties serialized inside a
      single __attributes__ JSON string feature.
    verbose: Logging verbosity level (0 = silent, 1 = progress, 2 = debug).
    **kwargs: Ignored legacy arguments kept for backwards compatibility.

  Returns:
    A (graph, schema) tuple containing the loaded InMemoryGraph and its
    GraphSchema.
  """
  del kwargs
  start_time = time.time()
  spanner_client = gcp_spanner.Client(project=project)
  db = spanner_client.instance(instance).database(database)

  with db.snapshot(multi_use=True) as snapshot:
    snapshot.begin()
    metadata = get_metadata(snapshot, graph)
    column_spanner_types = read_column_spanner_types_gcp(snapshot)

    def execute_fn(
        query: str,
        bytes_cols: Sequence[int],
        bytes_array_cols: Sequence[int],
    ) -> Iterable[Sequence[Any]]:
      result_set = snapshot.execute_sql(query)
      return (
          spanner_graph_lib.decode_base64_row(row, bytes_cols, bytes_array_cols)
          for row in result_set
      )

    return spanner_graph_lib.read_graph_with_executor(
        metadata=metadata,
        execute_fn=execute_fn,
        schema=schema,
        combine_as_json=combine_as_json,
        column_spanner_types=column_spanner_types,
        verbose=verbose,
        start_time=start_time,
    )


def read_spanner_graph_schema(
    project: str,
    instance: str,
    database: str,
    graph: str,
    *,
    combine_as_json: bool = False,
    verbose: int = 1,
) -> schema_lib.GraphSchema:
  """Reads the DGF GraphSchema of a Google Cloud Spanner Property Graph."""
  return read_spanner_graph_schema_gcp(
      project=project,
      instance=instance,
      database=database,
      graph=graph,
      combine_as_json=combine_as_json,
      verbose=verbose,
  )


def read_spanner_graph(
    project: str,
    instance: str,
    database: str,
    graph: str,
    *,
    schema: schema_lib.GraphSchema | None = None,
    combine_as_json: bool = False,
    verbose: int = 1,
    **kwargs: Any,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a Google Cloud Spanner Property Graph into an in-memory DGF graph."""
  return read_spanner_graph_gcp(
      project=project,
      instance=instance,
      database=database,
      graph=graph,
      schema=schema,
      combine_as_json=combine_as_json,
      verbose=verbose,
      **kwargs,
  )
