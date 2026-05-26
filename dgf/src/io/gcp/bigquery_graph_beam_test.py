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

"""Tests for bigquery_graph_beam."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from apache_beam.testing import test_pipeline
from dgf.src.io.gcp import bigquery_graph_beam
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


class BigqueryGraphBeamTest(parameterized.TestCase):

  @mock.patch("dgf.src.io.gcp.bigquery_graph.get_metadata")
  @mock.patch("apache_beam.io.gcp.bigquery.ReadFromBigQuery")
  def test_distributed_read_beam(self, mock_read_bq, mock_load_metadata):
    metadata = bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict(
        infoschema_query_response_json
    )
    mock_load_metadata.return_value = metadata

    with test_pipeline.TestPipeline() as p:
      distributed_graph = bigquery_graph_beam.distributed_read_beam(
          "project", "dataset", "graph", p
      )
      self.assertIsNotNone(distributed_graph)
      self.assertIn("nodes", distributed_graph.node_sets)
      self.assertIn(
          "biggraphs-poc.ogbn_arxiv_2.edges", distributed_graph.edge_sets
      )

    self.assertEqual(mock_read_bq.call_count, 2)


if __name__ == "__main__":
  absltest.main()
