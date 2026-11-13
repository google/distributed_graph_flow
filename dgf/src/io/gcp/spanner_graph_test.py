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

"""Tests for the Google Cloud Spanner Graph reader."""

import base64
import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io.gcp import spanner_graph
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.util import test_util
from google.cloud.spanner_v1 import database as gcp_database_lib
import numpy as np


def _infoschema_response() -> dict[str, Any]:
  """Returns mocked Cloud Spanner PROPERTY_GRAPH_METADATA_JSON content."""
  return {
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
              {"propertyDeclarationName": "feat", "valueExpressionSql": "feat"},
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


class SpannerGraphGcpTest(parameterized.TestCase):

  def test_infoschema_query(self):
    query = spanner_graph._infoschema_query()
    self.assertIn("FROM information_schema.property_graphs", query)
    self.assertIn("WHERE property_graph_name = @graph_name", query)

  def test_execute_query_on_database_and_snapshot(self):
    mock_database = mock.create_autospec(
        gcp_database_lib.Database, instance=True
    )
    mock_snapshot = mock_database.snapshot.return_value.__enter__.return_value
    mock_snapshot.execute_sql.return_value = "RESULT"

    res = spanner_graph._execute_query(
        mock_database, "query", params={"p": 1}, param_types={"p": "INT64"}
    )
    mock_snapshot.execute_sql.assert_called_once_with(
        "query", params={"p": 1}, param_types={"p": "INT64"}
    )
    self.assertEqual(res, "RESULT")

    direct_snapshot = mock.MagicMock()
    direct_snapshot.execute_sql.return_value = "SNAP_RESULT"
    self.assertEqual(
        spanner_graph._execute_query(direct_snapshot, "query"), "SNAP_RESULT"
    )

  @mock.patch("dgf.src.io.gcp.spanner_graph._execute_query")
  def test_load_metadata_and_errors(self, mock_execute_query):
    mock_execute_query.return_value = [[_infoschema_response()]]
    metadata = spanner_graph.get_metadata(mock.Mock(), "graph")
    self.assertEqual(metadata.name, "spanner_graph")

    mock_execute_query.return_value = []
    with self.assertRaisesRegex(ValueError, "Expected one property graph"):
      spanner_graph.get_metadata(mock.Mock(), "graph")

    with self.assertRaisesRegex(ValueError, "Invalid identifier"):
      spanner_graph.get_metadata(mock.Mock(), "invalid-graph-name")

  def test_graph_schema_and_beam_gql_query(self):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        _infoschema_response()
    )
    schema = spanner_graph.graph_schema(metadata, combine_as_json=False)
    self.assertIn("NodeTable", schema.node_sets)
    self.assertIn("EdgeTable", schema.edge_sets)
    self.assertEqual(
        schema.node_sets["NodeTable"].features["id"].semantic,
        schema_lib.FeatureSemantic.PRIMARY_ID,
    )

    gql = spanner_graph.graph_data_read_query(
        "spanner_graph", "NODE", metadata.node_tables[0]
    )
    self.assertIn("@{FORCE_INDEX=_base_table}", gql)
    self.assertIn("GRAPH spanner_graph", gql)

  @mock.patch("google.cloud.spanner.Client")
  def test_read_spanner_graph_gcp_and_alias(self, mock_client):
    mock_database = (
        mock_client.return_value.instance.return_value.database.return_value
    )
    mock_snapshot = mock.MagicMock()
    mock_ctx = mock.MagicMock()
    mock_ctx.__enter__.return_value = mock_snapshot
    mock_database.snapshot.return_value = mock_ctx

    def execute_sql_side_effect(query, params=None, param_types=None):
      del params, param_types
      if "information_schema.property_graphs" in query:
        return [[_infoschema_response()]]
      if "INFORMATION_SCHEMA.COLUMNS" in query:
        return [["Nodes", "feat", "ARRAY<FLOAT64>(vector_length=>2)"]]
      if "FROM `Nodes`" in query:
        return [[base64.b64encode(b"node1"), [1.0, 2.0]]]
      if "FROM `Edges`" in query:
        return [[
            base64.b64encode(b"node1"),
            base64.b64encode(b"node1"),
            0.5,
        ]]
      raise AssertionError(f"Unexpected query: {query}")

    mock_snapshot.execute_sql.side_effect = execute_sql_side_effect

    graph, schema = spanner_graph.read_spanner_graph_gcp(
        "project", "instance", "database", "graph", verbose=0
    )
    alias_graph, alias_schema = spanner_graph.read_spanner_graph(
        "project", "instance", "database", "graph", verbose=0
    )
    schema_only = spanner_graph.read_spanner_graph_schema_gcp(
        "project", "instance", "database", "graph", verbose=0
    )
    test_util.assert_are_equal(self, schema, alias_schema)
    test_util.assert_are_equal(self, schema, schema_only)
    test_util.assert_are_equal(self, graph, alias_graph)
    test_util.assert_are_equal(
        self,
        graph,
        in_memory_graph_lib.InMemoryGraph(
            node_sets={
                "NodeTable": in_memory_graph_lib.InMemoryNodeSet(
                    num_nodes=1,
                    features={
                        "id": np.array([b"node1"]),
                        "feat": np.array([[1.0, 2.0]], dtype=np.float64),
                    },
                )
            },
            edge_sets={
                "EdgeTable": in_memory_graph_lib.InMemoryEdgeSet(
                    adjacency=np.array([[0], [0]], dtype=np.int64),
                    features={"weight": np.array([0.5], dtype=np.float64)},
                )
            },
        ),
    )

  @mock.patch("google.cloud.spanner.Client")
  def test_read_spanner_graph_gcp_combine_as_json(self, mock_client):
    mock_database = (
        mock_client.return_value.instance.return_value.database.return_value
    )
    mock_snapshot = mock.MagicMock()
    mock_ctx = mock.MagicMock()
    mock_ctx.__enter__.return_value = mock_snapshot
    mock_database.snapshot.return_value = mock_ctx

    node_json = json.dumps({"properties": {"id": "node1", "feat": [1.0, 2.0]}})
    edge_json = json.dumps({"properties": {"weight": 0.5}})

    def execute_sql_side_effect(query, params=None, param_types=None):
      del params, param_types
      if "information_schema.property_graphs" in query:
        return [[_infoschema_response()]]
      if "INFORMATION_SCHEMA.COLUMNS" in query:
        return []
      if "FROM `Nodes`" in query:
        return [[base64.b64encode(b"node1"), node_json]]
      if "FROM `Edges`" in query:
        return [[
            base64.b64encode(b"node1"),
            base64.b64encode(b"node1"),
            edge_json,
        ]]
      raise AssertionError(f"Unexpected query: {query}")

    mock_snapshot.execute_sql.side_effect = execute_sql_side_effect

    graph, _ = spanner_graph.read_spanner_graph_gcp(
        "project",
        "instance",
        "database",
        "graph",
        combine_as_json=True,
        verbose=0,
    )
    self.assertEqual(
        json.loads(
            graph.node_sets["NodeTable"].features["__attributes__"][0].tobytes()
        ),
        {"properties": {"id": "node1", "feat": [1.0, 2.0]}},
    )
    self.assertEqual(
        json.loads(
            graph.edge_sets["EdgeTable"].features["__attributes__"][0].tobytes()
        ),
        {"properties": {"weight": 0.5}},
    )


if __name__ == "__main__":
  absltest.main()
