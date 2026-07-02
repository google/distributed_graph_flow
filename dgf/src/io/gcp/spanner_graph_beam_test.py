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

"""Tests for spanner_graph_beam."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from apache_beam.testing import test_pipeline
from dgf.src.io.gcp import spanner_graph_beam
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib

infoschema_query_response_json = {
    "catalog": "default",
    "schema": "public",
    "name": "spanner_graph",
    "labels": [
        {"name": "NodeLabel", "propertyDeclarationNames": ["id", "feat"]},
        {"name": "EdgeLabel", "propertyDeclarationNames": ["weight"]},
    ],
    "propertyDeclarations": [
        {"name": "id", "type": "STRING"},
        {"name": "feat", "type": "ARRAY<FLOAT64>"},
        {"name": "weight", "type": "FLOAT64"},
    ],
    "nodeTables": [{
        "name": "NodeTable",
        "baseCatalogName": "default",
        "baseSchemaName": "public",
        "baseTableName": "Nodes",
        "kind": "NODE",
        "keyColumns": ["id"],
        "labelNames": ["NodeLabel"],
        "propertyDefinitions": [
            {"propertyDeclarationName": "id", "valueExpressionSql": "id"},
            {
                "propertyDeclarationName": "feat",
                "valueExpressionSql": "feat",
            },
        ],
    }],
    "edgeTables": [{
        "name": "EdgeTable",
        "baseCatalogName": "default",
        "baseSchemaName": "public",
        "baseTableName": "Edges",
        "kind": "EDGE",
        "keyColumns": ["id"],
        "labelNames": ["EdgeLabel"],
        "sourceNodeTable": {
            "nodeTableName": "NodeTable",
            "nodeTableColumns": ["id"],
            "edgeTableColumns": ["source_id"],
        },
        "destinationNodeTable": {
            "nodeTableName": "NodeTable",
            "nodeTableColumns": ["id"],
            "edgeTableColumns": ["target_id"],
        },
        "propertyDefinitions": [{
            "propertyDeclarationName": "weight",
            "valueExpressionSql": "weight",
        }],
    }],
}


class SpannerGraphBeamTest(parameterized.TestCase):

  @mock.patch("google.cloud.spanner.Client")
  def test_break_for_non_pk_fk_aligned_graph_passes(self, mock_client):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )
    mock_snapshot = (
        mock_client.return_value.instance.return_value.database.return_value.batch_snapshot.return_value
    )
    mock_snapshot.generate_query_batches.return_value = [mock.Mock()]

    spanner_graph_beam.check_metadata(
        "project", "instance", "database", "graph", metadata
    )
    mock_snapshot.generate_query_batches.assert_called_once()

  @mock.patch("google.cloud.spanner.Client")
  def test_break_for_non_pk_fk_aligned_graph_raises(self, mock_client):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )
    mock_snapshot = (
        mock_client.return_value.instance.return_value.database.return_value.batch_snapshot.return_value
    )
    mock_snapshot.generate_query_batches.side_effect = Exception(
        "Spanner error"
    )

    with self.assertRaisesRegex(
        ValueError,
        "Edge table with non PK-FK aligned source and destination node tables"
        " are not supported.",
    ):
      spanner_graph_beam.check_metadata(
          "project", "instance", "database", "graph", metadata
      )

  @mock.patch("google.cloud.spanner.Client")
  @mock.patch("dgf.src.io.gcp.spanner_graph.get_metadata")
  @mock.patch("dgf.src.io.gcp.spanner_graph_beam._generate_read_partitions")
  @mock.patch("apache_beam.io.gcp.spanner.ReadFromSpanner")
  def test_distributed_read_beam(
      self, mock_read_spanner, mock_gen_partitions, mock_load_metadata, mock_client
  ):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )
    mock_load_metadata.return_value = metadata
    mock_gen_partitions.return_value = ([mock.Mock()], True)

    with test_pipeline.TestPipeline() as p:
      dist_graph = spanner_graph_beam.distributed_read_beam(
          "project", "instance", "database", "graph", p
      )
      self.assertIsNotNone(dist_graph)
      self.assertIn("NodeTable", dist_graph.node_sets)
      self.assertIn("EdgeTable", dist_graph.edge_sets)

    self.assertEqual(mock_read_spanner.call_count, 2)


if __name__ == "__main__":
  absltest.main()
