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

"""Library for working with Google Cloud Spanner Graphs using direct SQL."""

# TODO(gbm): To remove?

from collections.abc import Iterator
import time
from typing import Any, Dict, List, Optional, Tuple
from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import io_ext
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.util import log
from dgf.src.util import util as util_lib
from google.cloud import spanner as gcp_spanner
from google.cloud.spanner_v1.streamed import StreamedResultSet
import numpy as np

_SPANNER_GRAPH_METADATA_JSON_COLUMN_NAME = "PROPERTY_GRAPH_METADATA_JSON"


def _infoschema_query() -> str:
  """Returns a SQL string to query Spanner property graph metadata."""
  return """
      SELECT PROPERTY_GRAPH_METADATA_JSON FROM information_schema.property_graphs
      WHERE property_graph_name = @graph_name
    """


def _execute_query(
    database_or_snapshot: Any,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    param_types: Optional[Dict[str, Any]] = None,
) -> StreamedResultSet:
  """Executes a Spanner query and returns an iterator of the result rows."""
  if hasattr(database_or_snapshot, "snapshot"):
    with database_or_snapshot.snapshot() as snapshot:
      return snapshot.execute_sql(query, params=params, param_types=param_types)
  else:
    return database_or_snapshot.execute_sql(
        query, params=params, param_types=param_types
    )


def get_metadata(
    database_or_snapshot: Any,
    graph: str,
) -> spanner_graph_metadata_lib.SpannerGraphMetadata:
  """Loads GCP property graph metadata into the metadata object."""
  gcp_common_lib.validate_identifier(graph)
  metadata_query = _infoschema_query()
  log.info("Metadata query: %s (with graph=%s)", metadata_query, graph)

  params = {"graph_name": graph}
  param_types = {"graph_name": gcp_spanner.param_types.STRING}

  ## TODO(tewariy): Use quota_project.
  result_set = _execute_query(
      database_or_snapshot,
      metadata_query,
      params=params,
      param_types=param_types,
  )
  try:
    metadata_row = result_set.one()
    log.info("Metadata rows: %s", metadata_row)
  except ValueError as exc:
    raise ValueError(
        "expected exactly 1 property graph metadata with name:"
        f" {graph}, found {result_set.stats().rowCountExact}"
    ) from exc

  metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(
      metadata_row[0]
  )
  # _check_metadata(database_or_snapshot, graph, metadata)
  return metadata


def _graph_element_table(
    property_definitions: List[spanner_graph_metadata_lib.PropertyDefinition],
    property_types: Dict[str, str],
) -> Dict[str, str]:
  """Returns a Dict of Spanner graph element features."""
  graph_element_table = {}
  if property_definitions:
    for property_definition in property_definitions:
      ## In GoogleSQL Property Graphs, the same property name used in
      # different labels and tables must have the same data type.
      feature_name = property_definition.property_declaration_name
      feature_type = property_types[feature_name]
      graph_element_table[feature_name] = feature_type.strip().upper()
  return graph_element_table


def graph_schema(
    metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
    combine_as_json: bool = False,
) -> schema_lib.GraphSchema:
  """Returns a DGF GraphSchema from a GCP Spanner property graph metadata."""
  node_sets = {}
  for node_table in metadata.node_tables:
    features = gcp_common_lib.infer_feature_set_schema(
        graph_element_table=_graph_element_table(
            node_table.property_definitions, metadata.property_types
        ),
        key_columns=node_table.key_columns,
        combine_as_json=combine_as_json,
    )
    node_sets[node_table.name] = schema_lib.NodeSchema(features=features)

  edge_sets = {}
  for edge_table in metadata.edge_tables:
    features = gcp_common_lib.infer_feature_set_schema(
        graph_element_table=_graph_element_table(
            edge_table.property_definitions, metadata.property_types
        ),
        key_columns=edge_table.key_columns,
        combine_as_json=combine_as_json,
        skip_primary_keys=True,
    )
    edge_sets[edge_table.name] = schema_lib.EdgeSchema(
        source=edge_table.source_node_table.node_table_name,
        target=edge_table.destination_node_table.node_table_name,
        features=features,
    )

  return schema_lib.GraphSchema(
      node_sets=node_sets,
      edge_sets=edge_sets,
  )


def graph_data_read_query(
    graph: str,
    graph_element_type: str,
    graph_element_table: (
        spanner_graph_metadata_lib.NodeTable
        | spanner_graph_metadata_lib.EdgeTable
    ),
) -> str:
  """Returns a GQL string to query Spanner graph elements (nodes or edges)."""
  graph_element_labels_string = " & ".join(graph_element_table.label_names)
  has_properties = (
      graph_element_table.property_definitions is not None
      and len(graph_element_table.property_definitions) > 0
  )
  gql_base_query = gcp_common_lib.gql_base(
      graph_id=graph,
      graph_element_type=graph_element_type,
      graph_element_labels_string=graph_element_labels_string,
      graph_element_table_name=graph_element_table.name,
      omit_json=not has_properties,
  )
  index_hint = "@{FORCE_INDEX=_base_table}"
  spanner_graph_query_string = f"""
      {index_hint}
      {gql_base_query}
    """
  return spanner_graph_query_string


def graph_data_read_sql_query(
    graph_element_table: (
        spanner_graph_metadata_lib.NodeTable
        | spanner_graph_metadata_lib.EdgeTable
    ),
    graph_element_type: str,
) -> str:
  """Returns a SQL string to query Spanner base tables directly, bypassing GQL."""
  table_name = graph_element_table.name

  if graph_element_type == gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE:
    # For nodes, we need 'id'. Assume single key column.
    key_col = graph_element_table.key_columns[0]
    properties = graph_element_table.property_definitions
    if properties:
      # Construct TO_JSON_STRING(JSON_OBJECT('properties', JSON_OBJECT(...))) for properties
      json_args = []
      for prop in properties:
        json_args.append(f"'{prop.property_declaration_name}'")
        json_args.append(prop.value_expression_sql)
      json_object_str = (
          "TO_JSON_STRING(JSON_OBJECT('properties',"
          f" JSON_OBJECT({', '.join(json_args)})))"
      )
      return (
          f"SELECT {key_col} as id, {json_object_str} as graph_element FROM"
          f" {table_name}"
      )
    else:
      return f"SELECT {key_col} as id FROM {table_name}"

  elif graph_element_type == gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE:
    # For edges, we need 'source_id' and 'target_id'. Assume single key columns.
    source_col = graph_element_table.source_node_table.edge_table_columns[0]
    dest_col = graph_element_table.destination_node_table.edge_table_columns[0]
    properties = graph_element_table.property_definitions
    if properties:
      # Construct TO_JSON_STRING(JSON_OBJECT('properties', JSON_OBJECT(...))) for properties
      json_args = []
      for prop in properties:
        json_args.append(f"'{prop.property_declaration_name}'")
        json_args.append(prop.value_expression_sql)
      json_object_str = (
          "TO_JSON_STRING(JSON_OBJECT('properties',"
          f" JSON_OBJECT({', '.join(json_args)})))"
      )
      # We must select a dummy 'id' at index 0 because the client library
      # accesses columns by index when rows are not dicts, expecting:
      # [0]: edge_id, [1]: source_id, [2]: target_id, [3]: graph_element
      return (
          f"SELECT 'dummy_id' as id, {source_col} as source_id, {dest_col} as"
          f" target_id, {json_object_str} as graph_element FROM {table_name}"
      )
    else:
      # We must select a dummy 'id' as index 0 as well.
      return (
          f"SELECT 'dummy_id' as id, {source_col} as source_id, {dest_col} as"
          f" target_id FROM {table_name}"
      )
  else:
    raise ValueError(f"Unknown graph element type: {graph_element_type}")


def read_spanner_graph_schema(
    project: str,
    instance: str,
    database: str,
    graph: str,
    *,
    combine_as_json: bool = False,
    verbose: int = 1,
) -> schema_lib.GraphSchema:
  """Reads the schema of a Spanner Graph."""
  spanner_client = gcp_spanner.Client(project=project)
  db = spanner_client.instance(instance).database(database)
  metadata = get_metadata(db, graph)

  if verbose >= 2:
    log.info("meta-data:\n%s", metadata)

  return graph_schema(metadata, combine_as_json)


def read_spanner_graph(
    project: str,
    instance: str,
    database: str,
    graph: str,
    *,
    schema: Optional[schema_lib.GraphSchema] = None,
    combine_as_json: bool = False,
    verbose: int = 1,
    **kwargs,  # Accept and ignore legacy parallel arguments
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a Spanner Graph sequentially in-process using direct SQL queries on base tables."""
  start_time = time.time()

  spanner_client = gcp_spanner.Client(project=project)
  db = spanner_client.instance(instance).database(database)

  node_sets = {}
  node_id_index_maps = {}
  edge_sets = {}

  # Open a single multi-use snapshot for the entire read operation to ensure consistency
  with db.snapshot(multi_use=True) as snapshot:
    snapshot.begin()
    # Warm up the snapshot and get metadata (still uses Information Schema GQL)
    spanner_graph_metadata = get_metadata(snapshot, graph)

    if verbose >= 2:
      log.info(
          "Consistent Read Timestamp: %s",
          snapshot._transaction_read_timestamp,
      )
      log.info("meta-data:\n%s", spanner_graph_metadata)

    if schema is None:
      schema = graph_schema(spanner_graph_metadata, combine_as_json)

    if verbose >= 2:
      log.info("%s", print_schema_lib.print_schema(schema, return_output=True))

    if verbose >= 1:
      log.info("Reading nodesets and edgesets from Spanner base tables via SQL")

    # Phase 1: Load all node sets sequentially
    for node_table in spanner_graph_metadata.node_tables:
      query_string = graph_data_read_sql_query(
          node_table, gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE
      )
      if verbose >= 2:
        log.info("node SQL query_string:\n%s", query_string)

      result_set = snapshot.execute_sql(query_string)
      in_memory_node_set, node_id_index_map = (
          gcp_common_lib.create_in_memory_node_set(
              nodeset_name=node_table.name,
              graph_schema=schema,
              query_results=result_set,  # StreamedResultSet
              combine_as_json=combine_as_json,
              verbose=verbose,
          )
      )
      node_sets[node_table.name] = in_memory_node_set
      node_id_index_maps[node_table.name] = node_id_index_map

    # Phase 2: Load all edge sets sequentially (only after we have all node maps)
    for edge_table in spanner_graph_metadata.edge_tables:
      query_string = graph_data_read_sql_query(
          edge_table, gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE
      )
      if verbose >= 2:
        log.info("edge SQL query_string:\n%s", query_string)

      result_set = snapshot.execute_sql(query_string)
      in_memory_edge_set = gcp_common_lib.create_in_memory_edge_set(
          edgeset_name=edge_table.name,
          graph_schema=schema,
          query_results=result_set,  # StreamedResultSet
          source_node_id_index_map=node_id_index_maps[
              edge_table.source_node_table.node_table_name
          ],
          target_node_id_index_map=node_id_index_maps[
              edge_table.destination_node_table.node_table_name
          ],
          combine_as_json=combine_as_json,
          verbose=verbose,
      )
      edge_sets[edge_table.name] = in_memory_edge_set

  end_time = time.time()
  log.info(
      "Graph read in memory in %s",
      util_lib.format_duration(end_time - start_time),
  )

  in_memory_graph = in_memory_graph_lib.InMemoryGraph(
      node_sets=node_sets, edge_sets=edge_sets
  )
  return in_memory_graph, schema
