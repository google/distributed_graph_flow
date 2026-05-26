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

"""Library for working with Google Cloud BigQuery Graphs via Apache Beam."""

from typing import Any, Dict

import apache_beam as beam
import apache_beam.io.gcp.bigquery as beam_bigquery_io
from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.io.gcp import bigquery_graph
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import common_beam as gcp_common_beam_lib
from dgf.src.io.gcp import parquet_export as parquet_export_lib


def _wrap_in_graph_element(record: Dict[str, Any]) -> Dict[str, Any]:
  """Wraps the bq graph query result in graph element format."""
  gql_formated_record = {}
  if (
      gcp_common_lib.DGF_GRAPH_ELEMENT_SOURCE_ID_KEY in record
      and gcp_common_lib.DGF_GRAPH_ELEMENT_TARGET_ID_KEY in record
  ):
    gql_formated_record[gcp_common_lib.GRAPH_ELEMENT_SOURCE_ID_KEY] = (
        record.pop(gcp_common_lib.DGF_GRAPH_ELEMENT_SOURCE_ID_KEY)
    )
    gql_formated_record[gcp_common_lib.GRAPH_ELEMENT_TARGET_ID_KEY] = (
        record.pop(gcp_common_lib.DGF_GRAPH_ELEMENT_TARGET_ID_KEY)
    )

  gql_formated_record[gcp_common_lib.GRAPH_ELEMENT_ID_KEY] = record.pop(
      gcp_common_lib.DGF_GRAPH_ELEMENT_ID_KEY
  )
  gql_formated_record[gcp_common_lib.GRAPH_ELEMENT_JSON_KEY] = {
      gcp_common_lib.GRAPH_ELEMENT_PROPERTIES_KEY: record
  }

  return gql_formated_record


def _beam_bq_query_cleanup(query: str) -> str:
  """Replace `#` in the query as its not supported in Beam."""
  query = query.replace(
      gf_graph_in_beam_lib.KEY_SOURCE,
      gcp_common_lib.DGF_GRAPH_ELEMENT_SOURCE_ID_KEY,
  )
  query = query.replace(
      gf_graph_in_beam_lib.KEY_TARGET,
      gcp_common_lib.DGF_GRAPH_ELEMENT_TARGET_ID_KEY,
  )
  return query


def distributed_read_beam(
    project_id: str,
    dataset_id: str,
    graph_id: str,
    p: beam.pvalue.PBegin,
    combine_as_json: bool = False,
) -> distributed_graph_lib.Graph:
  """Read BigQuery Graph via Beam and return a distributed GraphFlow graph.

  Args:
    project_id: The GCP project ID of the BigQuery Graph.
    dataset_id: The BQ dataset ID of the BigQuery Graph.
    graph_id: The ID of the BigQuery Graph.
    p: The Beam pipeline.
    combine_as_json: Whether to combine the features as JSON.

  Returns:
    A distributed GraphFlow graph.
  """
  metadata = bigquery_graph.get_metadata(project_id, dataset_id, graph_id)
  schema = bigquery_graph.metadata_to_schema(metadata, combine_as_json)

  node_sets = {}
  for node_table in metadata.node_tables:
    bq_graph_query_string = parquet_export_lib.create_nodeset_sql(node_table)
    bq_graph_query_string = _beam_bq_query_cleanup(bq_graph_query_string)
    node_pcollection = (
        p
        | f"BigQueryGraphRead_{node_table.name}_{gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE}"
        >> beam_bigquery_io.ReadFromBigQuery(
            query=bq_graph_query_string,
            use_standard_sql=True,
            project=project_id,
        )
        | f"WrapInGraphElement_{node_table.name}_{gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE}"
        >> beam.Map(_wrap_in_graph_element)
        | f"DGFNodes_{node_table.name}_{gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE}"
        >> beam.Map(
            gcp_common_beam_lib.create_distributed_node_set,
            graph_element_name=node_table.name,
            graph_schema=schema,
        )
    )
    node_sets[node_table.name] = node_pcollection

  edge_sets = {}
  for edge_table in metadata.edge_tables:
    bq_graph_query_string = parquet_export_lib.create_edgeset_sql(edge_table)
    bq_graph_query_string = _beam_bq_query_cleanup(bq_graph_query_string)

    edge_pcollection = (
        p
        | f"BigQueryGraphRead_{edge_table.name}_{gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE}"
        >> beam_bigquery_io.ReadFromBigQuery(
            query=bq_graph_query_string,
            use_standard_sql=True,
            project=project_id,
        )
        | f"WrapInGraphElement_{edge_table.name}_{gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE}"
        >> beam.Map(_wrap_in_graph_element)
        | f"DGFEdges_{edge_table.name}_{gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE}"
        >> beam.Map(
            gcp_common_beam_lib.create_distributed_edge_set,
            graph_element_name=edge_table.name,
            graph_schema=schema,
        )
    )
    edge_sets[edge_table.name] = edge_pcollection

  return distributed_graph_lib.Graph(
      schema=schema,
      node_sets=node_sets,
      edge_sets=edge_sets,
  )
