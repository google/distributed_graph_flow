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

"""Library for working with Google Cloud BigQuery Graphs."""

import concurrent.futures
import os
from typing import Dict, List, Optional, Tuple, Union
import uuid

from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.data import gf_metadata as gf_metadata_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_constants
from dgf.src.io import graph_in_memory as gf_graph_in_memory_lib
from dgf.src.io import schema as io_schema
from dgf.src.io.gcp import bigquery_graph_metadata as bigquery_graph_metadata_lib
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import parquet_export as parquet_export_lib
from dgf.src.util import filesystem
from dgf.src.util import log
from dgf.src.util import util as util_lib
from google.cloud import bigquery as gcp_bigquery

MAX_SUPPORTED_GF_VERSION = graph_constants.MAX_SUPPORTED_GF_VERSION
FILENAME_SCHEMA = graph_constants.FILENAME_SCHEMA
FILENAME_METADATA = graph_constants.FILENAME_METADATA

BQ_GRAPH_METADATA_JSON_COLUMN_NAME = "PROPERTY_GRAPH_METADATA_JSON"


def _infoschema_query(
    project_id: str,
    dataset_id: str,
    graph_id: str,
) -> str:
  """Returns a SQL string to query BigQuery property graph metadata."""
  return f"""
    SELECT property_graph_metadata_json as PROPERTY_GRAPH_METADATA_JSON
    FROM `{project_id}`.{dataset_id}.INFORMATION_SCHEMA.PROPERTY_GRAPHS
    WHERE property_graph_name = '{graph_id}'
  """


def _execute_query(
    query: str, quota_project_id: str
) -> gcp_bigquery.table.RowIterator:
  """Executes a bigquery query and retruns an iterator of the result rows."""
  bq_client = gcp_bigquery.Client(project=quota_project_id)

  query_job = bq_client.query(query)
  return query_job.result()


def get_metadata(
    project_id: str,
    dataset_id: str,
    graph_id: str,
) -> bigquery_graph_metadata_lib.BigQueryGraphMetadata:
  """Loads GCP property graph metadata into the metadata object."""
  metadata_query = _infoschema_query(project_id, dataset_id, graph_id)

  ## TODO(tewariy): Use quota_project_id.
  metadata_rows = list(_execute_query(metadata_query, project_id))

  if not metadata_rows or len(metadata_rows) != 1:
    raise ValueError(
        "expected exactly 1 property graph metadata with name:"
        f" {graph_id}, found {len(metadata_rows)}"
    )

  bq_meta_data = bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict(
      metadata_rows[0][BQ_GRAPH_METADATA_JSON_COLUMN_NAME]
  )

  _check_metadata(bq_meta_data)
  return bq_meta_data


def _graph_element_table(
    labels_and_properties: bigquery_graph_metadata_lib.LabelAndProperties,
) -> Dict[str, str]:
  """Returns a Dict of BigQuery graph element features.

  This represents the nodes and edges metadata in a common format. The common
  function `infer_feature_set_schema()` can then be used to extract
  feature set for graph flow schema for both Spanner and BigQuery.

  Args:
    labels_and_properties: The LabelAndProperties object from the BigQuery graph
      metadata for the given graph element (node or edge).

  Returns:
    A dictionary of BigQuery graph element features with feature name as key
    and feature type as value.
  """
  graph_element_table = {}
  for label_and_properties in labels_and_properties:
    if label_and_properties.properties:
      for label_property in label_and_properties.properties:
        feature_name = label_property.name
        ## In GoogleSQL Property Graphs, the same property name used in
        # different labels and tables must have the same data type.
        if feature_name not in graph_element_table:
          feature_type = label_property.data_type.resolved_data_type()
          graph_element_table[feature_name] = feature_type
  return graph_element_table


def metadata_to_schema(
    bigquery_graph_metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata,
    combine_as_json: bool,
) -> schema_lib.GraphSchema:
  """Converts BigQuery graph metadata to a GraphFlow schema."""

  node_sets = {}
  for node_table in bigquery_graph_metadata.node_tables:
    node_sets[node_table.name] = schema_lib.NodeSchema(
        features=gcp_common_lib.infer_feature_set_schema(
            _graph_element_table(node_table.label_and_properties),
            node_table.key_columns,
            combine_as_json,
        )
    )

  edge_sets = {}
  for edge_table in bigquery_graph_metadata.edge_tables:
    edge_sets[edge_table.name] = schema_lib.EdgeSchema(
        source=edge_table.source_node_reference.node_table,
        target=edge_table.destination_node_reference.node_table,
        features=gcp_common_lib.infer_feature_set_schema(
            _graph_element_table(edge_table.label_and_properties),
            [],  # Add support for edge ids,
            combine_as_json,
        ),
    )

  return schema_lib.GraphSchema(
      node_sets=node_sets,
      edge_sets=edge_sets,
  )


def _is_source_and_destination_pk_fk_aligned(
    edge_table: bigquery_graph_metadata_lib.EdgeTable,
    node_tables: List[bigquery_graph_metadata_lib.NodeTable],
) -> bool:
  """Returns true if the edge table columns are PK-FK aligned with the node table columns."""
  source_node_table = None
  target_node_table = None
  for node_table in node_tables:
    if node_table.name == edge_table.source_node_reference.node_table:
      source_node_table = node_table
    if node_table.name == edge_table.destination_node_reference.node_table:
      target_node_table = node_table
  if source_node_table is None or target_node_table is None:
    raise ValueError(
        "source or target node table not found for edge table:"
        f" {edge_table.name}"
    )

  source_pk_fk_aligned = gcp_common_lib.is_pk_fk_aligned(
      edge_table.source_node_reference.node_table_columns,
      source_node_table.key_columns,
  )
  target_pk_fk_aligned = gcp_common_lib.is_pk_fk_aligned(
      edge_table.destination_node_reference.node_table_columns,
      target_node_table.key_columns,
  )
  return source_pk_fk_aligned and target_pk_fk_aligned


def _check_metadata(
    bigquery_graph_metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata,
):
  """Raises an error if the graph is not PK-FK aligned."""
  for edge_table in bigquery_graph_metadata.edge_tables:
    if not _is_source_and_destination_pk_fk_aligned(
        edge_table, bigquery_graph_metadata.node_tables
    ):  ## this is not supported, users need to fix the graph schema to make it
      ## PK-FK aligned.
      raise ValueError(
          "Edge table with non PK-FK aligned source and destination node tables"
          " are not supported."
      )


def read_bigquery_graph_schema(
    project: str,
    dataset: str,
    graph: str,
    *,
    combine_as_json: bool = False,
    verbose: Union[int, bool] = True,
):
  """Reads the schema of a BigQuery graph into a GF schema.

  Usage:
    ```python
    import dgf

    schema = dgf.io.read_bigquery_graph_schema(
        project="your-gcp-project",
        dataset="your_bq_dataset",
        graph="your_bq_graph",
    )
    dgf.print.schema(schema)
    ```

  Args:
    project: The GCP project ID of the BigQuery Graph.
    dataset: The BQ dataset ID of the BigQuery Graph.
    graph: The ID of the BigQuery Graph.
    combine_as_json: Whether to combine the features as JSON.
    verbose: Amount of verbose (0: no, 1 or true: a little, 2: a lot).

  Returns:
    A GraphFlow schema object.
  """

  metadata = get_metadata(project, dataset, graph)
  if verbose >= 2:
    log.info("meta-data:\n%s", metadata)
  return metadata_to_schema(metadata, combine_as_json)


def read_bigquery_graph(
    project: str,
    dataset: str,
    graph: str,
    *,
    schema: Optional[schema_lib.GraphSchema] = None,
    work_dir: str,
    combine_as_json: bool = False,
    max_workers: int = 10,
    verbose: Union[int, bool] = True,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a BigQuery Graph in process and returns a GraphFlow in-memory graph.

  Usage:
    ```
    import dgf

    graph, schema = dgf.io.read_bigquery_graph(
      project="biggraphs-poc",
      dataset="arxiv",
      graph="arvix_graph",
      work_dir="gs://my_project/tmp"
    )
    ```

  Args:
    project: The Google Cloud project ID.
    dataset: The BigQuery dataset ID.
    graph: The BigQuery graph ID.
    work_dir: The working directory to use for the temporary storage.
    combine_as_json: Whether to combine the features as JSON.
    max_workers: The maximum number of workers to use for parallel processing.
    verbose: Amount of verbose (0: no, 1 or true: a little, 2: a lot).

  Returns:
    A Tuple of the in memory graph and the schema.
  """

  if isinstance(verbose, bool):
    verbose = int(verbose)

  try:
    work_dir = os.path.join(
        work_dir, f"read_{project}_{dataset}_{graph}_{uuid.uuid4()}"
    )
    export_bigquery_to_disk(
        path=work_dir,
        project=project,
        dataset=dataset,
        graph=graph,
        combine_as_json=combine_as_json,
        max_workers=max_workers,
        verbose=verbose,
        schema=schema,
    )

    in_memory_graph, in_memory_schema = gf_graph_in_memory_lib.read_graph(
        work_dir, verbose=verbose >= 1
    )
  finally:
    try:
      filesystem.remove_paths([work_dir])
    except Exception as e:  # pylint: disable=broad-except-clause
      log.warning(f"Failed to remove temporary work directory {work_dir}: {e}")

  return in_memory_graph, in_memory_schema


def export_bigquery_to_disk(
    path: str,
    project: str,
    dataset: str,
    graph: str,
    *,
    schema: Optional[schema_lib.GraphSchema] = None,
    combine_as_json: bool = False,
    max_workers: int = 10,
    verbose: Union[int, bool] = True,
):
  """Reads a BigQuery Graph in process and returns a GraphFlow in-memory graph.

  Usage:
    ```
    import dgf

    graph, schema = dgf.io.export_bigquery_to_disk(
      path="gs://my_project/my_graph"
      project="biggraphs-poc",
      dataset="arxiv",
      graph="arvix_graph",
    )
    ```

  Args:
    path: The directory to export the BigQuery graph data to.
    project: The Google Cloud project ID.
    dataset: The BigQuery dataset ID.
    graph: The BigQuery graph ID.
    combine_as_json: Whether to combine the features as JSON.
    max_workers: The maximum number of workers to use for parallel processing.
    verbose: Amount of verbose (0: no, 1 or true: a little, 2: a lot).

  Returns:
    A Tuple of the in memory graph and the schema.
  """

  if isinstance(verbose, bool):
    verbose = int(verbose)

  metadata = get_metadata(project, dataset, graph)

  if verbose >= 2:
    log.info("BQ meta-data:\n%s", metadata)

  # TODO(b/328622124): Add feature_shapes, feature_semantics, and
  # num_categorical_values to the function signature and pass them here.
  if schema is None:
    schema = metadata_to_schema(metadata, combine_as_json)

  if verbose >= 2:
    log.info("%s", print_schema_lib.print_schema(schema, return_output=True))

  if verbose >= 1:
    log.info(
        "Exporting nodesets and edgesets from BigQuery Graph: %s:%s:%s to: %s",
        project,
        dataset,
        graph,
        path,
    )

  with util_lib.print_timer("Exporting graph", verbose >= 1):
    # Build a unified list of export tasks to execute.
    export_tasks = []
    for node_table in metadata.node_tables:
      export_tasks.append((
          parquet_export_lib.create_nodeset_sql(node_table),
          gcp_common_lib.GCS_PREFIX_NODESETS,
          node_table.name,
      ))
    for edge_table in metadata.edge_tables:
      export_tasks.append((
          parquet_export_lib.create_edgeset_sql(edge_table),
          gcp_common_lib.GCS_PREFIX_EDGESETS,
          edge_table.name,
      ))

    future_to_table_name = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      for sql, prefix, table_name in export_tasks:
        export_query = parquet_export_lib.create_export_sql(
            sql, path, prefix, table_name
        )
        if verbose >= 2:
          log.info("Query: %s", export_query)
        future = executor.submit(_execute_query, export_query, project)
        future_to_table_name[future] = table_name

      # Wait for all exports to complete and handle potential errors.
      for future in concurrent.futures.as_completed(future_to_table_name):
        table_name = future_to_table_name[future]
        try:
          future.result()
          if verbose >= 1:
            log.info(f"Export completed for node/edge table: {table_name!r}")
        except Exception as e:
          log.warning(f"Export failed for table {table_name!r}: {e}")
          raise

  # Write metadata and schema
  io_schema.write_schema(schema, os.path.join(path, FILENAME_SCHEMA))

  metadata = gf_metadata_lib.GFGraphMetadata(version=MAX_SUPPORTED_GF_VERSION)
  with filesystem.open_write(os.path.join(path, FILENAME_METADATA)) as f:
    f.write(metadata.to_json(indent=2))
