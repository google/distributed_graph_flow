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

"""Tests for bigquery_graph."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.io.gcp import bigquery_graph
from dgf.src.io.gcp import bigquery_graph_metadata as bigquery_graph_metadata_lib


infoschema_query_response_json = {
    "creationTime": "2026-02-14T23:25:51.378016Z",
    "edgeTables": [{
        "dataSourceTable": {
            "datasetId": "ogbn_arxiv_2",
            "projectId": "biggraphs-poc",
            "tableId": "edges",
        },
        "destinationNodeReference": {
            "edgeTableColumns": ["target_id"],
            "nodeTable": "nodes",
            "nodeTableColumns": ["id"],
        },
        "keyColumns": ["id", "target_id"],
        "labelAndProperties": [{
            "label": "edge",
            "properties": [
                {
                    "dataType": {"typeKind": "STRING"},
                    "expression": "source",
                    "name": "source",
                },
                {
                    "dataType": {"typeKind": "STRING"},
                    "expression": "target",
                    "name": "target",
                },
            ],
        }],
        "name": "biggraphs-poc.ogbn_arxiv_2.edges",
        "sourceNodeReference": {
            "edgeTableColumns": ["id"],
            "nodeTable": "nodes",
            "nodeTableColumns": ["id"],
        },
    }],
    "etag": "LLU+8RdCldBSyg46UncjUA==",
    "lastModifiedTime": "2026-02-14T23:25:51.378016Z",
    "nodeTables": [{
        "dataSourceTable": {
            "datasetId": "ogbn_arxiv_2",
            "projectId": "biggraphs-poc",
            "tableId": "nodes",
        },
        "keyColumns": ["id"],
        "labelAndProperties": [{
            "label": "node",
            "properties": [
                {
                    "dataType": {
                        "arrayElementType": {"typeKind": "FLOAT64"},
                        "typeKind": "ARRAY",
                    },
                    "expression": "feat",
                    "name": "feat",
                },
                {
                    "dataType": {"typeKind": "STRING"},
                    "expression": "id",
                    "name": "id",
                },
                {
                    "dataType": {"typeKind": "INT64"},
                    "expression": "labels",
                    "name": "labels",
                },
                {
                    "dataType": {"typeKind": "INT64"},
                    "expression": "year",
                    "name": "year",
                },
            ],
        }],
        "name": "nodes",
    }],
    "propertyGraphReference": {
        "datasetId": "ogbn_arxiv_2",
        "projectId": "biggraphs-poc",
        "propertyGraphId": "ogbn_arxiv_2",
    },
}

infoschema_query_response = {
    "PROPERTY_GRAPH_METADATA_JSON": infoschema_query_response_json
}


class BigqueryGraphTest(parameterized.TestCase):

  def test_infoschema_query(self):
    query = bigquery_graph._infoschema_query("project", "dataset", "graph")
    self.assertIn(
        "FROM `project`.dataset.INFORMATION_SCHEMA.PROPERTY_GRAPHS", query
    )
    self.assertIn("WHERE property_graph_name = 'graph'", query)

  @mock.patch("google.cloud.bigquery.Client")
  def test_execute_query(self, mock_client):
    mock_instance = mock_client.return_value
    mock_query_job = mock_instance.query.return_value
    mock_query_job.result.return_value = infoschema_query_response

    results = bigquery_graph._execute_query("SOME QUERY", "quota-project")

    mock_client.assert_called_once_with(project="quota-project")
    mock_instance.query.assert_called_once_with("SOME QUERY")
    self.assertEqual(results, infoschema_query_response)

  @parameterized.named_parameters(("dict_response", infoschema_query_response))
  @mock.patch("dgf.src.io.gcp.bigquery_graph._execute_query")
  @mock.patch(
      "dgf.src.io.gcp.bigquery_graph_metadata.BigQueryGraphMetadata.from_dict"
  )
  def test_load_metadata(
      self, response_data, mock_from_dict, mock_execute_query
  ):
    mock_execute_query.return_value = iter([response_data])
    mock_metadata = mock.Mock(
        spec=bigquery_graph_metadata_lib.BigQueryGraphMetadata
    )
    mock_metadata.edge_tables = []
    mock_from_dict.return_value = mock_metadata

    metadata = bigquery_graph.get_metadata("project", "dataset", "graph")

    self.assertIsNotNone(metadata)
    mock_execute_query.assert_called_once()
    mock_from_dict.assert_called_once_with(
        response_data["PROPERTY_GRAPH_METADATA_JSON"]
    )

  @mock.patch("dgf.src.io.gcp.bigquery_graph._execute_query")
  def test_load_metadata_error(self, mock_execute_query):
    mock_execute_query.return_value = iter([])
    with self.assertRaisesRegex(
        ValueError, "expected exactly 1 property graph metadata"
    ):
      bigquery_graph.get_metadata("project", "dataset", "graph")

  def test_is_source_and_destination_pk_fk_aligned(self):
    metadata = bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )
    edge_table = metadata.edge_tables[0]
    node_tables = metadata.node_tables
    self.assertTrue(
        bigquery_graph._is_source_and_destination_pk_fk_aligned(
            edge_table, node_tables
        )
    )

  def test_graph_element_table(self):
    node_table = bigquery_graph_metadata_lib.NodeTable.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json["nodeTables"][0]
    )
    table_dict = bigquery_graph._graph_element_table(
        node_table.label_and_properties
    )
    self.assertEqual(
        table_dict,
        {
            "feat": "ARRAY<FLOAT64>",
            "id": "STRING",
            "labels": "INT64",
            "year": "INT64",
        },
    )

  def test_graph_schema(self):
    metadata = bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )
    schema = bigquery_graph.metadata_to_schema(
        metadata,
        combine_as_json=False,
    )
    self.assertIn("nodes", schema.node_sets)
    self.assertIn("biggraphs-poc.ogbn_arxiv_2.edges", schema.edge_sets)

  def test_break_for_non_pk_fk_aligned_graph(self):
    metadata_json = copy.deepcopy(infoschema_query_response_json)
    # Misalign the keys to force non-alignment
    metadata_json["edgeTables"][0]["sourceNodeReference"][
        "nodeTableColumns"
    ] = ["misaligned_id"]
    metadata = bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        metadata_json
    )
    with self.assertRaisesRegex(
        ValueError,
        "Edge table with non PK-FK aligned source and destination node tables"
        " are not supported.",
    ):
      bigquery_graph._check_metadata(metadata)

  @mock.patch("dgf.src.io.gcp.bigquery_graph.get_metadata")
  @mock.patch("dgf.src.io.gcp.bigquery_graph._execute_query")
  @mock.patch("dgf.src.io.gcp.parquet_export.create_export_sql")
  @mock.patch("dgf.src.io.schema.write_schema")
  @mock.patch("dgf.src.util.filesystem.open_write")
  @mock.patch("dgf.src.io.graph_in_memory.read_graph")
  def test_read_bigquery_graph(
      self,
      mock_read_graph,
      mock_open_write,
      mock_write_schema,
      mock_create_export_sql,
      mock_execute_query,
      mock_load_metadata,
  ):
    metadata = bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )
    mock_load_metadata.return_value = metadata
    mock_execute_query.return_value = mock.Mock()
    mock_create_export_sql.return_value = "EXPORT DATA SQL"
    mock_read_graph.return_value = (mock.Mock(), mock.Mock())

    graph, _ = bigquery_graph.read_bigquery_graph(
        "project", "dataset", "graph", work_dir="gs://bucket/prefix"
    )

    self.assertIsNotNone(graph)
    mock_load_metadata.assert_called_once_with("project", "dataset", "graph")
    self.assertEqual(mock_execute_query.call_count, 2)
    mock_create_export_sql.assert_called()
    mock_write_schema.assert_called_once()
    mock_open_write.assert_called_once()
    mock_read_graph.assert_called_once()
    self.assertTrue(
        mock_read_graph.call_args[0][0].startswith(
            "gs://bucket/prefix/read_project_dataset_graph_"
        )
    )
    self.assertEqual(mock_read_graph.call_args[1]["verbose"], 1)


if __name__ == "__main__":
  absltest.main()
