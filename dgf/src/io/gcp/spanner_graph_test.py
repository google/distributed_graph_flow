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
    query = spanner_graph._infoschema_query("graph")
    self.assertIn("FROM information_schema.property_graphs", query)
    self.assertIn("WHERE property_graph_name = 'graph'", query)

  @mock.patch("google.cloud.spanner.Client")
  def test_execute_query(self, mock_client):
    mock_instance = mock_client.return_value.instance.return_value
    mock_database = mock_instance.database.return_value
    mock_snapshot = mock_database.snapshot.return_value.__enter__.return_value
    mock_snapshot.execute_sql.return_value = "RESULT"

    res = spanner_graph._execute_query(
        "project", "instance", "database", "query"
    )

    mock_client.assert_called_once_with(project="project")
    mock_client.return_value.instance.assert_called_once_with("instance")
    mock_instance.database.assert_called_once_with("database")
    mock_snapshot.execute_sql.assert_called_once_with("query")
    self.assertEqual(res, "RESULT")

  @mock.patch("dgf.src.io.gcp.spanner_graph._execute_query")
  def test_load_metadata(self, mock_execute_query):
    mock_result_set = mock.Mock()
    mock_result_set.one.return_value = [infoschema_query_response_json]
    mock_execute_query.return_value = mock_result_set

    metadata = spanner_graph.get_metadata(
        "project", "instance", "database", "graph"
    )

    self.assertIsNotNone(metadata)
    self.assertEqual(metadata.name, "spanner_graph")
    mock_execute_query.assert_called_once()

  @mock.patch("dgf.src.io.gcp.spanner_graph._execute_query")
  def test_load_metadata_error(self, mock_execute_query):
    mock_result_set = mock.Mock()
    mock_result_set.one.side_effect = ValueError("No rows")
    mock_result_set.stats.return_value.rowCountExact = 0
    mock_execute_query.return_value = mock_result_set

    with self.assertRaisesRegex(
        ValueError, "expected exactly 1 property graph metadata"
    ):
      spanner_graph.get_metadata("project", "instance", "database", "graph")

  def test_graph_element_table(self):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(
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
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(
        infoschema_query_response_json
    )
    schema = spanner_graph.graph_schema(
        metadata,
        combine_as_json=False,
    )
    self.assertIn("NodeTable", schema.node_sets)
    self.assertIn("EdgeTable", schema.edge_sets)

  @mock.patch("dgf.src.io.gcp.spanner_graph.get_metadata")
  @mock.patch("dgf.src.io.gcp.spanner_graph._execute_query")
  def test_direct_read_in_memory(self, mock_execute_query, mock_load_metadata):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict(
        infoschema_query_response_json
    )
    mock_load_metadata.return_value = metadata

    mock_node_res = mock.Mock()
    mock_node_res.to_dict_list.return_value = [{
        "id": "node1",
        "graph_element": json.dumps(
            {"properties": {"id": "node1", "feat": [1.0, 2.0]}}
        ),
    }]

    mock_edge_res = mock.Mock()
    mock_edge_res.to_dict_list.return_value = [{
        "id": "edge1",
        "source_id": "node1",
        "target_id": "node1",
        "graph_element": json.dumps({"properties": {"weight": 0.5}}),
    }]

    def execute_query_side_effect(project, instance, database, query):
      del project, instance, database  # Unused
      if "NodeTable" in query:
        return mock_node_res
      elif "EdgeTable" in query:
        return mock_edge_res
      else:
        raise ValueError(f"Unexpected query: {query}")

    mock_execute_query.side_effect = execute_query_side_effect

    graph, schema = spanner_graph.read_spanner_graph(
        "project", "instance", "database", "graph"
    )

    self.assertIsNotNone(graph)
    self.assertIsNotNone(schema)
    self.assertIn("NodeTable", graph.node_sets)
    self.assertIn("EdgeTable", graph.edge_sets)
    self.assertEqual(graph.node_sets["NodeTable"].num_nodes, 1)
    self.assertEqual(graph.edge_sets["EdgeTable"].num_edges(), 1)


if __name__ == "__main__":
  absltest.main()
