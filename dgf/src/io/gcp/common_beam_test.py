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

"""Tests for gcp common beam library."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.io.gcp import common_beam as gcp_common_beam_lib


class CommonBeamTest(parameterized.TestCase):

  def test_dgf_node_with_all_types(self):
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
                    "p_float": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    ),
                    "p_bool": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BOOL
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
    graph_element = {
        "id": "node_id",
        "graph_element": {
            "properties": {
                "p_str": "val1",
                "p_int": 42,
                "p_float": 3.14,
                "p_bool": True,
            }
        },
    }
    node = gcp_common_beam_lib.create_distributed_node_set(
        graph_element, "nodes", graph_schema
    )
    self.assertEqual(node.features["p_str"], b"val1")
    self.assertEqual(node.features["p_int"], 42)
    self.assertAlmostEqual(node.features["p_float"], 3.14, places=5)
    self.assertEqual(node.features["p_bool"], True)

    # Test missing properties (currently they are not filled with defaults)
    graph_element_missing = {
        "id": "node_id_missing",
        "graph_element": {"properties": {}},
    }
    node_missing = gcp_common_beam_lib.create_distributed_node_set(
        graph_element_missing, "nodes", graph_schema
    )
    self.assertNotIn("p_str", node_missing.features)
    self.assertNotIn("p_int", node_missing.features)
    self.assertNotIn("p_float", node_missing.features)
    self.assertNotIn("p_bool", node_missing.features)

  def test_dgf_node_with_timestamp(self):
    graph_schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "p_ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
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
    ts_str = "2023-10-27T10:00:00Z"
    graph_element = {
        "id": "node_id",
        "graph_element": {"properties": {"p_ts": ts_str}},
    }
    node = gcp_common_beam_lib.create_distributed_node_set(
        graph_element, "nodes", graph_schema
    )
    expected_micros = 1698400800000000
    self.assertEqual(node.features["p_ts"], expected_micros)

  def test_dgf_node_json_mode(self):
    graph_schema = schema_lib.GraphSchema(node_sets={}, edge_sets={})
    properties = {"p1": "val1", "p2": 42}
    graph_element_content = {"properties": properties}
    graph_element = {
        "id": "node_id",
        "graph_element": graph_element_content,
    }
    node = gcp_common_beam_lib.create_distributed_node_set(
        graph_element, "nodes", graph_schema, combine_as_json=True
    )
    self.assertIn("__attributes__", node.features)
    self.assertEqual(
        json.loads(node.features["__attributes__"].tobytes()),
        graph_element_content,
    )

  def test_dgf_edge_json_mode(self):
    graph_schema = schema_lib.GraphSchema(node_sets={}, edge_sets={})
    properties = {"p1": "val1", "p2": 42}
    graph_element_content = {"properties": properties}
    graph_element = {
        "id": "edge_id",
        "source_id": "s_id",
        "target_id": "t_id",
        "graph_element": graph_element_content,
    }
    edge = gcp_common_beam_lib.create_distributed_edge_set(
        graph_element, "edges", graph_schema, combine_as_json=True
    )
    self.assertIn("__attributes__", edge.features)
    self.assertEqual(
        json.loads(edge.features["__attributes__"].tobytes()),
        graph_element_content,
    )

  def test_dgf_node(self):
    graph_schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "p1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.UNKNOWN,
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
    graph_element = {
        "id": "node_id",
        "graph_element": {"properties": {"p1": "val1"}},
    }
    node = gcp_common_beam_lib.create_distributed_node_set(
        graph_element, "nodes", graph_schema
    )
    self.assertEqual(node.id, b"node_id")
    self.assertIn("p1", node.features)
    self.assertEqual(node.features["p1"], b"val1")
    self.assertEqual(node.features["#id"], b"node_id")

  def test_dgf_edge(self):
    graph_schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "edges": schema_lib.EdgeSchema(
                source="nodes",
                target="nodes",
                features={
                    "p1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.UNKNOWN,
                    ),
                    "#id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    ),
                },
            )
        },
    )
    graph_element = {
        "id": "edge_id",
        "source_id": "s_id",
        "target_id": "t_id",
        "graph_element": {"properties": {"p1": "val1"}},
    }
    edge = gcp_common_beam_lib.create_distributed_edge_set(
        graph_element, "edges", graph_schema
    )
    self.assertEqual(edge.id, b"edge_id")
    self.assertEqual(edge.source, b"s_id")
    self.assertEqual(edge.target, b"t_id")
    self.assertIn("p1", edge.features)
    self.assertEqual(edge.features["p1"], b"val1")
    self.assertEqual(edge.features["#id"], b"edge_id")

  def test_dgf_node_missing_properties_with_defaults(self):
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
                    "p_float": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    ),
                    "p_bool": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BOOL
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
    graph_element = {
        "id": "node_id",
        "graph_element": {
            "properties": {
                "p_str": None,
                "p_int": None,
                "p_float": None,
                "p_bool": None,
            }
        },
    }
    node = gcp_common_beam_lib.create_distributed_node_set(
        graph_element, "nodes", graph_schema
    )
    self.assertEqual(node.features["p_str"], b"")
    self.assertEqual(node.features["p_int"], 0)
    self.assertEqual(node.features["p_float"], 0.0)
    self.assertEqual(node.features["p_bool"], False)


if __name__ == "__main__":
  absltest.main()
