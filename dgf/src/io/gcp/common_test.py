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

"""Tests for gcp common library."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.io import io_ext
from dgf.src.io.gcp import common as gcp_common_lib
import numpy as np


class CommonTest(parameterized.TestCase):

  def test_graph_element_feature_definitions(self):
    graph_element_table = {
        "p1": "STRING",
        "p2": "INT64",
        "p3": "ARRAY<FLOAT64>",
        "p4": "TIMESTAMP",
    }
    graph_element_table["p5"] = "ARRAY<INT64>"
    graph_element_table["p6"] = "ARRAY<TIMESTAMP>"

    # Use multi-key to test "#id" generation
    features = gcp_common_lib.infer_feature_set_schema(
        graph_element_table,
        key_columns=["p1", "p2"],
        combine_as_json=False,
    )
    self.assertIn("p1", features)
    self.assertEqual(features["p1"].format, schema_lib.FeatureFormat.BYTES)
    # p1 is part of multi-key, so it should have UNKNOWN semantic in DGF
    self.assertEqual(
        features["p1"].semantic, schema_lib.FeatureSemantic.UNKNOWN
    )

    self.assertIn("p2", features)
    self.assertEqual(features["p2"].format, schema_lib.FeatureFormat.INTEGER_64)
    self.assertEqual(
        features["p2"].semantic, schema_lib.FeatureSemantic.UNKNOWN
    )

    self.assertIn("p3", features)
    self.assertEqual(features["p3"].shape, [None])

    self.assertIn("p4", features)
    self.assertEqual(
        features["p4"].semantic, schema_lib.FeatureSemantic.TIMESTAMP
    )

    # "#id" should be present because of multi-key
    self.assertEqual(
        features["#id"].semantic,
        schema_lib.FeatureSemantic.PRIMARY_ID,
    )

    self.assertIn("p5", features)
    self.assertEqual(features["p5"].format, schema_lib.FeatureFormat.INTEGER_64)
    self.assertEqual(features["p5"].shape, [None])

    self.assertIn("p6", features)
    self.assertEqual(features["p6"].format, schema_lib.FeatureFormat.INTEGER_64)
    self.assertEqual(
        features["p6"].semantic, schema_lib.FeatureSemantic.TIMESERIES
    )
    self.assertEqual(features["p6"].shape, [None])

  def test_graph_element_feature_definitions_single_key(self):
    graph_element_table = {
        "p1": "STRING",
        "p2": "INT64",
    }
    features = gcp_common_lib.infer_feature_set_schema(
        graph_element_table,
        key_columns=["p1"],
        combine_as_json=False,
    )
    self.assertIn("p1", features)
    self.assertEqual(features["p1"].format, schema_lib.FeatureFormat.BYTES)
    self.assertEqual(
        features["p1"].semantic, schema_lib.FeatureSemantic.PRIMARY_ID
    )

    self.assertIn("p2", features)
    self.assertEqual(features["p2"].format, schema_lib.FeatureFormat.INTEGER_64)
    self.assertEqual(
        features["p2"].semantic, schema_lib.FeatureSemantic.UNKNOWN
    )

    self.assertNotIn("#id", features)

  def test_graph_element_feature_definitions_json(self):
    graph_element_table = {"p1": "STRING"}
    features = gcp_common_lib.infer_feature_set_schema(
        graph_element_table,
        key_columns=[],
        combine_as_json=True,
    )
    self.assertIn("__attributes__", features)
    self.assertEqual(
        features["__attributes__"].format, schema_lib.FeatureFormat.BYTES
    )

  @parameterized.named_parameters(
      ("node", gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE, "MATCH (ge:label)"),
      ("edge", gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE, "MATCH -[ge:label]->"),
  )
  def test_gql_base(self, element_type, expected_match):
    gql = gcp_common_lib.gql_base("graph_id", element_type, "label", "table")
    self.assertIn("GRAPH graph_id", gql)
    self.assertIn(expected_match, gql)
    self.assertIn("ELEMENT_DEFINITION_NAME(ge) = 'table'", gql)

  def test_gql_base_invalid_type(self):
    with self.assertRaisesRegex(ValueError, "graph_element_type must be"):
      gcp_common_lib.gql_base("graph_id", "INVALID", "label", "table")

  def test_validate_identifier_valid(self):
    valid_identifiers = ["graph_1", "GraphName", "_graph", "a1", "_"]
    for identifier in valid_identifiers:
      try:
        gcp_common_lib.validate_identifier(identifier)
      except ValueError:
        self.fail(f"validate_identifier failed unexpectedly for {identifier!r}")

  @parameterized.parameters(
      ("graph-1",),
      ("1graph",),
      ("graph id",),
      ("graph'id",),
      ('graph"id',),
      ("graph;id",),
      ("",),
  )
  def test_validate_identifier_invalid(self, identifier):
    with self.assertRaises(ValueError):
      gcp_common_lib.validate_identifier(identifier)

  def test_gql_base_validation(self):
    with self.assertRaises(ValueError):
      gcp_common_lib.gql_base("invalid-graph-id", "NODE", "label", "table")
    with self.assertRaises(ValueError):
      gcp_common_lib.gql_base("graph_id", "NODE", "label", "invalid-table")

  def test_is_semantic_timestamp(self):
    self.assertTrue(gcp_common_lib.is_semantic_timestamp("TIMESTAMP"))
    self.assertFalse(gcp_common_lib.is_semantic_timestamp("STRING"))

  def test_is_semantic_array(self):
    self.assertTrue(gcp_common_lib.is_semantic_array("ARRAY<INT64>"))
    self.assertFalse(gcp_common_lib.is_semantic_array("INT64"))

  def test_is_semantic_timeseries(self):
    self.assertTrue(gcp_common_lib.is_semantic_timeseries("ARRAY<TIMESTAMP>"))
    self.assertFalse(gcp_common_lib.is_semantic_timeseries("ARRAY<INT64>"))
    self.assertFalse(gcp_common_lib.is_semantic_timeseries("TIMESTAMP"))

  def test_parse_timestamp_to_micros(self):
    ts_str = "2023-10-27T10:00:00Z"
    micros = gcp_common_lib.parse_timestamp_to_micros(ts_str)
    expected = 1698400800000000
    self.assertEqual(micros, expected)

  def test_in_memory_node_set(self):
    graph_schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "p_str": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES
                    ),
                    "p_int": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64
                    ),
                    "#id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    ),
                }
            )
        },
        edge_sets={},
    )
    query_results = [
        {
            "id": "node_1",
            "graph_element": json.dumps(
                {"properties": {"p_str": "val1", "p_int": 10}}
            ),
        },
        {
            "id": "node_2",
            "graph_element": json.dumps(
                {"properties": {"p_str": "val2", "p_int": 20}}
            ),
        },
    ]

    node_set, node_id_index_map = gcp_common_lib.create_in_memory_node_set(
        "nodes", graph_schema, query_results, combine_as_json=False, verbose=1
    )

    self.assertEqual(node_set.num_nodes, 2)
    np.testing.assert_array_equal(
        node_set.features["p_str"], [b"val1", b"val2"]
    )
    np.testing.assert_array_equal(node_set.features["p_int"], [10, 20])
    indices, mismatch = node_id_index_map(
        np.array([b"node_1", b"node_2", b"node_3"])
    )
    np.testing.assert_array_equal(indices, [0, 1, -1])
    self.assertEqual(mismatch, 2)

  def test_in_memory_node_set_json_mode(self):
    graph_schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "__attributes__": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.UNKNOWN,
                        shape=(),
                        num_categorical_values=None,
                    )
                }
            )
        },
        edge_sets={},
    )
    query_results = [
        {
            "id": "node_1",
            "graph_element": json.dumps({"properties": {"p1": "val1"}}),
        },
    ]

    node_set, _ = gcp_common_lib.create_in_memory_node_set(
        "nodes", graph_schema, query_results, combine_as_json=True, verbose=1
    )

    self.assertEqual(node_set.num_nodes, 1)
    self.assertIn("__attributes__", node_set.features)
    expected_json = {"properties": {"p1": "val1"}}
    self.assertEqual(
        json.loads(node_set.features["__attributes__"][0].tobytes()),
        expected_json,
    )

  def test_in_memory_edge_set(self):
    graph_schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "edges": schema_lib.EdgeSchema(
                source="nodes",
                target="nodes",
                features={
                    "p_str": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES
                    ),
                    "p_int": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64
                    ),
                },
            )
        },
    )
    query_results = [
        {
            "id": "edge_1",
            "source_id": "node_1",
            "target_id": "node_2",
            "graph_element": json.dumps(
                {"properties": {"p_str": "eval1", "p_int": 100}}
            ),
        },
        {
            "id": "edge_2",
            "source_id": "node_2",
            "target_id": "node_1",
            "graph_element": json.dumps(
                {"properties": {"p_str": "eval2", "p_int": 200}}
            ),
        },
    ]
    source_node_id_index_map = io_ext.ByteIdToIdxMapper(
        np.array([b"node_1", b"node_2"])
    )
    target_node_id_index_map = io_ext.ByteIdToIdxMapper(
        np.array([b"node_1", b"node_2"])
    )

    edge_set = gcp_common_lib.create_in_memory_edge_set(
        "edges",
        graph_schema,
        query_results,
        source_node_id_index_map,
        target_node_id_index_map,
        combine_as_json=False,
        verbose=1,
    )

    self.assertEqual(edge_set.num_edges(), 2)
    np.testing.assert_array_equal(edge_set.adjacency, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(
        edge_set.features["p_str"], [b"eval1", b"eval2"]
    )
    np.testing.assert_array_equal(edge_set.features["p_int"], [100, 200])

  def test_in_memory_edge_set_json_mode(self):
    graph_schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "edges": schema_lib.EdgeSchema(
                source="nodes",
                target="nodes",
                features={
                    "__attributes__": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.UNKNOWN,
                        shape=(),
                        num_categorical_values=None,
                    )
                },
            )
        },
    )
    query_results = [
        {
            "id": "edge_1",
            "source_id": "node_1",
            "target_id": "node_2",
            "graph_element": json.dumps({"properties": {"p1": "val1"}}),
        },
    ]
    source_node_id_index_map = io_ext.ByteIdToIdxMapper(
        np.array([b"node_1", b"node_2"])
    )
    target_node_id_index_map = io_ext.ByteIdToIdxMapper(
        np.array([b"node_1", b"node_2"])
    )

    edge_set = gcp_common_lib.create_in_memory_edge_set(
        "edges",
        graph_schema,
        query_results,
        source_node_id_index_map,
        target_node_id_index_map,
        combine_as_json=True,
        verbose=1,
    )

    self.assertEqual(edge_set.num_edges(), 1)
    self.assertIn("__attributes__", edge_set.features)
    expected_json = {"properties": {"p1": "val1"}}
    self.assertEqual(
        json.loads(edge_set.features["__attributes__"][0].tobytes()),
        expected_json,
    )

  def test_in_memory_edge_set_mismatch_error(self):
    graph_schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "edges": schema_lib.EdgeSchema(
                source="nodes",
                target="nodes",
                features={},
            )
        },
    )
    query_results = [
        {
            "id": "edge_1",
            "source_id": "node_unknown",
            "target_id": "node_2",
            "graph_element": json.dumps({"properties": {}}),
        },
    ]
    source_node_id_index_map = io_ext.ByteIdToIdxMapper(np.array([b"node_1"]))
    target_node_id_index_map = io_ext.ByteIdToIdxMapper(np.array([b"node_2"]))

    with self.assertRaisesRegex(
        ValueError, "Source Node ID 'node_unknown' not found"
    ):
      gcp_common_lib.create_in_memory_edge_set(
          "edges",
          graph_schema,
          query_results,
          source_node_id_index_map,
          target_node_id_index_map,
          combine_as_json=False,
          verbose=1,
      )

  def test_parse_property_value_to_feature_nested(self):
    feature_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
    )
    feature_timeseries = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESERIES,
        shape=[None],
    )
    ts_str = "2023-10-27T10:00:00Z"
    expected_micros = 1698400800000000

    # Test single timestamp
    self.assertEqual(
        gcp_common_lib.parse_property_value_to_feature(feature_ts, ts_str),
        expected_micros,
    )

    # Test timeseries with None
    self.assertEqual(
        gcp_common_lib.parse_property_value_to_feature(
            feature_timeseries, [ts_str, None]
        ),
        [expected_micros, 0],
    )

    # Test array of strings with None
    feature_array_str = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.UNKNOWN,
        shape=[None],
    )
    self.assertEqual(
        gcp_common_lib.parse_property_value_to_feature(
            feature_array_str, ["a", None]
        ),
        ["a", ""],
    )


if __name__ == "__main__":
  absltest.main()
