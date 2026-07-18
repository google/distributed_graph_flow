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

"""Tests for spanner_graph."""

import copy
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.io import io_ext
from dgf.src.io.gcp import spanner_graph
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.util import test_util
import numpy as np


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
    "nodeTables": [
        {
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
        }
    ],
    "edgeTables": [
        {
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
            "propertyDefinitions": [
                {
                    "propertyDeclarationName": "weight",
                    "valueExpressionSql": "weight",
                }
            ],
        }
    ],
}


class SpannerGraphTest(parameterized.TestCase):

  def test_infoschema_query(self):
    query = spanner_graph._infoschema_query()
    self.assertIn("FROM information_schema.property_graphs", query)
    self.assertIn("WHERE property_graph_name = @graph_name", query)

  def test_execute_query(self):
    mock_database = mock.MagicMock()
    mock_snapshot = mock_database.snapshot.return_value.__enter__.return_value
    mock_snapshot.execute_sql.return_value = "RESULT"

    res = spanner_graph._execute_query(
        mock_database, "query", params={"p": 1}, param_types={"p": "INT64"}
    )

    mock_snapshot.execute_sql.assert_called_once_with(
        "query", params={"p": 1}, param_types={"p": "INT64"}
    )
    self.assertEqual(res, "RESULT")

  @mock.patch("dgf.src.io.gcp.spanner_graph._execute_query")
  def test_load_metadata(self, mock_execute_query):
    mock_result_set = mock.Mock()
    mock_result_set.one.return_value = [infoschema_query_response_json]
    mock_execute_query.return_value = mock_result_set
    mock_database = mock.Mock()

    metadata = spanner_graph.get_metadata(mock_database, "graph")

    self.assertIsNotNone(metadata)
    self.assertEqual(metadata.name, "spanner_graph")
    mock_execute_query.assert_called_once_with(
        mock_database,
        mock.ANY,
        params={"graph_name": "graph"},
        param_types=mock.ANY,
    )

  @mock.patch("dgf.src.io.gcp.spanner_graph._execute_query")
  def test_load_metadata_error(self, mock_execute_query):
    mock_result_set = mock.Mock()
    mock_result_set.one.side_effect = ValueError("No rows")
    mock_result_set.stats.return_value.rowCountExact = 0
    mock_execute_query.return_value = mock_result_set
    mock_database = mock.Mock()

    with self.assertRaisesRegex(
        ValueError, "expected exactly 1 property graph metadata"
    ):
      spanner_graph.get_metadata(mock_database, "graph")

  def test_load_metadata_invalid_graph_name(self):
    mock_database = mock.Mock()
    with self.assertRaises(ValueError):
      spanner_graph.get_metadata(mock_database, "invalid-graph-name")

  def test_graph_element_table(self):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )
    node_table = metadata.node_tables[0]
    elem_table = spanner_graph._graph_element_table(
        node_table.property_definitions, metadata.property_types
    )
    self.assertEqual(
        elem_table,
        {
            "id": "STRING",
            "feat": "ARRAY<FLOAT64>",
        },
    )

  def test_graph_schema(self):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )
    schema = spanner_graph.graph_schema(
        metadata,
        combine_as_json=False,
    )
    self.assertIn("NodeTable", schema.node_sets)
    self.assertIn("EdgeTable", schema.edge_sets)

  @mock.patch("google.cloud.spanner.Client")
  @mock.patch("dgf.src.io.gcp.spanner_graph.get_metadata")
  def test_direct_read_in_memory(self, mock_load_metadata, mock_client):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )
    mock_load_metadata.return_value = metadata

    mock_instance = mock_client.return_value.instance.return_value
    mock_database = mock_instance.database.return_value

    # Mock the single multi-use snapshot
    mock_snapshot = mock.MagicMock()
    mock_snapshot.begin.return_value = "mock_txn_id"
    mock_snapshot._transaction_read_timestamp = "mock_timestamp"

    # Configure db.snapshot(multi_use=True) to return this mock snapshot
    mock_ctx = mock.MagicMock()
    mock_ctx.__enter__.return_value = mock_snapshot
    mock_database.snapshot.return_value = mock_ctx

    mock_node_res = [(
        "node1",
        json.dumps({"properties": {"id": "node1", "feat": [1.0, 2.0]}}),
    )]
    mock_edge_res = [(
        "edge1",
        "node1",
        "node1",
        json.dumps({"properties": {"weight": 0.5}}),
    )]

    def execute_sql_side_effect(query):
      if "NodeTable" in query:
        return mock_node_res
      elif "EdgeTable" in query:
        return mock_edge_res
      else:
        raise ValueError(f"Unexpected query: {query}")

    mock_snapshot.execute_sql.side_effect = execute_sql_side_effect

    graph, schema = spanner_graph.read_spanner_graph(
        "project", "instance", "database", "graph"
    )

    self.assertIsNotNone(graph)
    self.assertIsNotNone(schema)
    self.assertIn("NodeTable", graph.node_sets)
    self.assertIn("EdgeTable", graph.edge_sets)
    self.assertEqual(graph.node_sets["NodeTable"].num_nodes, 1)
    self.assertEqual(graph.edge_sets["EdgeTable"].num_edges(), 1)

    # Verify db.snapshot was called with multi_use=True
    mock_database.snapshot.assert_called_once_with(multi_use=True)

  def test_graph_data_read_sql_query(self):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        infoschema_query_response_json
    )

    # Case 1: Node table with properties
    node_table_with_props = metadata.node_tables[0]
    query = spanner_graph.graph_data_read_sql_query(
        node_table_with_props, "NODE"
    )
    test_util.assert_golden_string(
        self,
        query,
        "spanner_graph_test_node_query_with_properties.sql",
        strip=True,
    )

    # Case 2: Node table without properties
    node_table_no_props = copy.deepcopy(node_table_with_props)
    node_table_no_props.property_definitions = []
    query = spanner_graph.graph_data_read_sql_query(node_table_no_props, "NODE")
    test_util.assert_golden_string(
        self,
        query,
        "spanner_graph_test_node_query_no_properties.sql",
        strip=True,
    )

    # Case 3: Edge table with properties
    edge_table_with_props = metadata.edge_tables[0]
    query = spanner_graph.graph_data_read_sql_query(
        edge_table_with_props, "EDGE"
    )
    test_util.assert_golden_string(
        self,
        query,
        "spanner_graph_test_edge_query_with_properties.sql",
        strip=True,
    )

    # Case 4: Edge table without properties
    edge_table_no_props = copy.deepcopy(edge_table_with_props)
    edge_table_no_props.property_definitions = []
    query = spanner_graph.graph_data_read_sql_query(edge_table_no_props, "EDGE")
    test_util.assert_golden_string(
        self,
        query,
        "spanner_graph_test_edge_query_no_properties.sql",
        strip=True,
    )


if __name__ == "__main__":
  absltest.main()
