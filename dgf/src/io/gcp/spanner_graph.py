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

"""Library for working with Google Cloud Spanner Graphs."""

# TODO(gbm): To remove?

from collections.abc import Iterator
import concurrent.futures
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


def _infoschema_query(
    graph: str,
) -> str:
  """Returns a SQL string to query Spanner property graph metadata."""
  return f"""
      SELECT PROPERTY_GRAPH_METADATA_JSON FROM information_schema.property_graphs
      WHERE property_graph_name = '{graph}'
    """


def _execute_query(
    project: str, instance: str, database: str, query: str
) -> StreamedResultSet:
  """Executes a Spanner query and returns an iterator of the result rows."""
  spanner_client = gcp_spanner.Client(project=project)

  database = spanner_client.instance(instance).database(database)
  with database.snapshot() as snapshot:
    return snapshot.execute_sql(query)


def _execute_query_to_dict_list(
    project: str, instance: str, database: str, query: str
) -> List[Dict[str, Any]]:
  """Executes a Spanner query and returns a list of dicts."""
  return _execute_query(project, instance, database, query).to_dict_list()


def get_metadata(
    project: str,
    instance: str,
    database: str,
    graph: str,
) -> spanner_graph_metadata_lib.SpannerGraphMetadata:
  """Loads GCP property graph metadata into the metadata object."""
  metadata_query = _infoschema_query(graph)
  log.info("Metadata query: %s", metadata_query)

  ## TODO(tewariy): Use quota_project.
  result_set = _execute_query(project, instance, database, metadata_query)
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
  # _check_metadata(project, instance, database, graph, metadata)
  return metadata


def _graph_element_table(
    property_definitions: List[spanner_graph_metadata_lib.PropertyDefinition],
    property_types: Dict[str, str],
) -> Dict[str, str]:
  """Returns a Dict of Spanner graph element features.

  This represents the nodes and edges metadata in a common format. The common
  function `infer_feature_set_schema()` can then be used to extract
  feature set for graph flow schema for both Spanner and BigQuery.

  Args:
    property_definitions: The PropertyDefinition object from the Spanner graph
      metadata for the given graph element (node or edge).
    property_types: The property types of the Spanner graph.

  Returns:
    A dictionary of Spanner graph element features with feature name as key
    and feature type as value.
  """
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
    spanner_graph_metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
    combine_as_json,
) -> schema_lib.GraphSchema:
  """Returns a GraphFlow schema for the Spanner Graph."""

  node_sets = {}
  for node_table in spanner_graph_metadata.node_tables:
    node_sets[node_table.name] = schema_lib.NodeSchema(
        features=gcp_common_lib.infer_feature_set_schema(
            _graph_element_table(
                node_table.property_definitions,
                spanner_graph_metadata.property_types,
            ),
            node_table.key_columns,
            combine_as_json,
        )
    )

  edge_sets = {}
  for edge_table in spanner_graph_metadata.edge_tables:
    edge_sets[edge_table.name] = schema_lib.EdgeSchema(
        source=edge_table.source_node_table.node_table_name,
        target=edge_table.destination_node_table.node_table_name,
        features=gcp_common_lib.infer_feature_set_schema(
            _graph_element_table(
                edge_table.property_definitions,
                spanner_graph_metadata.property_types,
            ),
            edge_table.key_columns,
            combine_as_json,
            # In spanner, to source/target node fields have to be a property.
            skip_primary_keys=True,
        ),
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
  gql_base_query = gcp_common_lib.gql_base(
      graph_id=graph,
      graph_element_type=graph_element_type,
      graph_element_labels_string=graph_element_labels_string,
      graph_element_table_name=graph_element_table.name,
  )
  index_hint = "@{FORCE_INDEX=_base_table}"
  spanner_graph_query_string = f"""
      {index_hint}
      {gql_base_query}
    """
  return spanner_graph_query_string


def read_spanner_graph_schema(
    project: str,
    instance: str,
    database: str,
    graph: str,
    *,
    combine_as_json: bool = False,
    verbose: int = 1,
) -> schema_lib.GraphSchema:
  """Reads the schema of a Spanner Graph.

  Args:
    project: The GCP project ID of the Spanner Graph.
    instance: The Spanner instance ID of the Spanner Graph.
    database: The Spanner database ID of the Spanner Graph.
    graph: The ID of the Spanner Graph.
    combine_as_json: Whether to combine the features as JSON.
    verbose: Amount of verbose (0: no, 1 or true: a little, 2: a lot).

  Returns:
    A `schema_lib.GraphSchema` object representing the Spanner Graph schema.
  """
  metadata = get_metadata(project, instance, database, graph)

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
    max_workers: int = 10,
    verbose: int = 1,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a Spanner Graph in process and returns a GraphFlow in-memory graph.

  The function reads the graph metadata and data from Spanner,
  and constructs an `InMemoryGraph` object. Node and edge data are
  read in parallel using a ThreadPoolExecutor.

  Usage:
  ```
    import dgf

    graph, schema = dgf.io.read_spanner_graph(
        known_args.spanner_project,
        known_args.spanner_instance,
        known_args.spanner_database,
        known_args.spanner_graph,
    )
    ```

  Args:
    project: The GCP project ID of the Spanner Graph.
    instance: The Spanner instance ID of the Spanner Graph.
    database: The Spanner database ID of the Spanner Graph.
    graph: The ID of the Spanner Graph.
    combine_as_json: Whether to combine the features as JSON.
    max_workers: The maximum number of workers to use for parallel processing.
    verbose: Amount of verbose (0: no, 1 or true: a little, 2: a lot).

  Returns:
    A Tuple of the in memory graph and the schema.
  """

  start_time = time.time()

  spanner_graph_metadata = get_metadata(project, instance, database, graph)

  if verbose >= 2:
    log.info("meta-data:\n%s", spanner_graph_metadata)

  if schema is None:
    schema = graph_schema(spanner_graph_metadata, combine_as_json)

  if verbose >= 2:
    log.info("%s", print_schema_lib.print_schema(schema, return_output=True))

  node_sets = {}
  node_id_index_maps = {}
  edge_sets = {}

  if verbose >= 1:
    log.info("Reading nodesets and edgesets concurrently")

  node_query_futures = {}
  edge_query_futures = {}

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    # Submit all node queries
    for node_table in spanner_graph_metadata.node_tables:
      query_string = graph_data_read_query(
          graph,
          gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE,
          node_table,
      )
      if verbose >= 2:
        log.info("node query_string:\n%s", query_string)
      future = executor.submit(
          _execute_query_to_dict_list,
          project=project,
          instance=instance,
          database=database,
          query=query_string,
      )
      node_query_futures[future] = node_table.name

    # Submit all edge queries
    for edge_table in spanner_graph_metadata.edge_tables:
      query_string = graph_data_read_query(
          graph,
          gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE,
          edge_table,
      )
      if verbose >= 2:
        log.info("edge query_string:\n%s", query_string)
      future = executor.submit(
          _execute_query_to_dict_list,
          project=project,
          instance=instance,
          database=database,
          query=query_string,
      )
      edge_query_futures[future] = edge_table

    # Wait for node queries and build node sets
    node_results = {}
    for future in concurrent.futures.as_completed(node_query_futures):
      node_table_name = node_query_futures[future]
      node_results[node_table_name] = future.result()

    for node_table_name, query_results in node_results.items():
      in_memory_node_set, node_id_index_map = (
          gcp_common_lib.create_in_memory_node_set(
              nodeset_name=node_table_name,
              graph_schema=schema,
              query_results=query_results,
              combine_as_json=combine_as_json,
              verbose=verbose,
          )
      )
      node_sets[node_table_name] = in_memory_node_set
      node_id_index_maps[node_table_name] = node_id_index_map

    # Wait for edge queries
    edge_results = {}
    for future in concurrent.futures.as_completed(edge_query_futures):
      edge_table = edge_query_futures[future]
      edge_results[edge_table.name] = (edge_table, future.result())

    # Build edge sets
    for edge_table_name, (edge_table, query_results) in edge_results.items():
      in_memory_edge_set = gcp_common_lib.create_in_memory_edge_set(
          edgeset_name=edge_table_name,
          graph_schema=schema,
          query_results=query_results,
          source_node_id_index_map=node_id_index_maps[
              edge_table.source_node_table.node_table_name
          ],
          target_node_id_index_map=node_id_index_maps[
              edge_table.destination_node_table.node_table_name
          ],
          combine_as_json=combine_as_json,
          verbose=verbose,
      )
      edge_sets[edge_table_name] = in_memory_edge_set

  if verbose >= 1:
    duration = time.time() - start_time
    log.info(f"Reading Spanner graph took {util_lib.format_duration(duration)}")
  in_memory_graph = in_memory_graph_lib.InMemoryGraph(
      node_sets=node_sets,
      edge_sets=edge_sets,
  )

  optimize_graph_and_schema(in_memory_graph, schema)

  return in_memory_graph, schema


def optimize_graph_and_schema(
    graph: in_memory_graph_lib.InMemoryGraph, schema: schema_lib.GraphSchema
):
  """Applies final optimizations based on both the graph data and schema.

  Optimizations:
  -   If a feature is defined with a variable length in the schema, but all its
      values across the graph elements (nodes/edges) have a consistent, fixed
      shape, the schema is updated to reflect this specific fixed length.

  Args:
    graph: The InMemoryGraph instance containing the loaded graph data.
    schema: The GraphSchema object to be potentially optimized.
  """
  for node_set_name, node_set in graph.node_sets.items():
    optimize_feature_and_feature_schema(
        node_set.features, schema.node_sets[node_set_name].features
    )

  for edge_set_name, edge_set in graph.edge_sets.items():
    optimize_feature_and_feature_schema(
        edge_set.features, schema.edge_sets[edge_set_name].features
    )


def optimize_feature_and_feature_schema(
    features: in_memory_graph_lib.Features, schemas: schema_lib.FeatureSetSchema
):
  """Optimizes feature data and updates the feature schema if possible."""
  for feature_name, feature_schema in schemas.items():
    if feature_schema.is_static_shape():
      continue

    feature_data = features[feature_name]
    assert isinstance(feature_data, np.ndarray) and feature_data.dtype == object

    first_shape = None
    all_same_shape = True
    for item in feature_data:
      assert isinstance(item, np.ndarray)
      current_shape = item.shape
      if first_shape is None:
        first_shape = current_shape
      elif current_shape != first_shape:
        all_same_shape = False
        break
    if all_same_shape:
      log.info(
          "Optimizing feature '%s': Found consistent shape %s, updating schema"
          " and stacking data.",
          feature_name,
          first_shape,
      )
      feature_schema.shape = tuple(first_shape)
      features[feature_name] = np.stack(feature_data)
