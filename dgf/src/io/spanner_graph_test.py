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

"""Tests for the client-independent Spanner Graph reader core."""

import base64
import datetime

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import spanner_graph as spanner_graph_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as validate_lib
import numpy as np

FeatureFormat = schema_lib.FeatureFormat
FeatureSchema = schema_lib.FeatureSchema
FeatureSemantic = schema_lib.FeatureSemantic


def _prop(name: str, expr: str) -> dict[str, str]:
  """Builds a PropertyDefinition dict entry."""
  return {"propertyDeclarationName": name, "valueExpressionSql": expr}


def _composite_metadata() -> spanner_graph_metadata_lib.SpannerGraphMetadata:
  """Builds composite-key SpannerGraphMetadata with aliased base tables."""
  return spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict({  # pyrefly: ignore[missing-attribute]
      "catalog": "",
      "schema": "",
      "name": "fleet_graph",
      "nodeTables": [
          {
              "name": "Cell",
              "kind": "NODE",
              "baseCatalogName": "",
              "baseSchemaName": "",
              "baseTableName": "Cells",
              "keyColumns": ["cell_id", "snapshot_ts"],
              "labelNames": ["Cell"],
              "propertyDefinitions": [
                  _prop("cell_id", "cell_id"),
                  _prop("snapshot_ts", "snapshot_ts"),
                  _prop("cell_emb", "cell_emb"),
              ],
          },
          {
              "name": "Rack",
              "kind": "NODE",
              "baseCatalogName": "",
              "baseSchemaName": "",
              "baseTableName": "Racks",
              "keyColumns": ["cell_id", "rack_id", "snapshot_ts"],
              "labelNames": ["Rack"],
              "propertyDefinitions": [
                  _prop("rack_id", "rack_id"),
                  _prop("cell_id", "cell_id"),
                  _prop("snapshot_ts", "snapshot_ts"),
                  _prop("machine_count", "machine_count"),
              ],
          },
      ],
      "edgeTables": [{
          "name": "RackPartOfCell",
          "kind": "EDGE",
          "baseCatalogName": "",
          "baseSchemaName": "",
          "baseTableName": "Racks",
          "keyColumns": ["cell_id", "rack_id", "snapshot_ts"],
          "labelNames": ["PART_OF"],
          "propertyDefinitions": [_prop("snapshot_ts", "snapshot_ts")],
          "sourceNodeTable": {
              "nodeTableName": "Rack",
              "edgeTableColumns": ["cell_id", "rack_id", "snapshot_ts"],
              "nodeTableColumns": ["cell_id", "rack_id", "snapshot_ts"],
          },
          # Intentionally reordered foreign-key columns to verify alignment.
          "destinationNodeTable": {
              "nodeTableName": "Cell",
              "edgeTableColumns": ["snapshot_ts", "cell_id"],
              "nodeTableColumns": ["snapshot_ts", "cell_id"],
          },
      }],
      "labels": [],
      "propertyDeclarations": [
          {"name": "cell_id", "type": "STRING"},
          {"name": "rack_id", "type": "STRING"},
          {"name": "snapshot_ts", "type": "TIMESTAMP"},
          {"name": "cell_emb", "type": "ARRAY<FLOAT32>"},
          {"name": "machine_count", "type": "INT64"},
      ],
  })


class SpannerGraphCoreTest(parameterized.TestCase):

  @parameterized.parameters(
      ("STRING(64)", ("STRING", False, None)),
      ("BYTES(MAX)", ("BYTES", False, None)),
      ("FLOAT", ("FLOAT32", False, None)),
      ("DOUBLE", ("FLOAT64", False, None)),
      ("ARRAY<FLOAT>(vector_length=>64)", ("FLOAT32", True, 64)),
      ("ARRAY<FLOAT32>(vector_length => 32)", ("FLOAT32", True, 32)),
      ("ARRAY<STRING(128)>", ("STRING", True, None)),
      ("PROTO<my.pkg.Msg>", ("PROTO", False, None)),
  )
  def test_parse_spanner_type(self, raw_type, expected):
    self.assertEqual(spanner_graph_lib.parse_spanner_type(raw_type), expected)

  def test_composite_key_roundtrip_and_collision_safety(self):
    col_types = ["STRING", "STRING", "TIMESTAMP"]
    ts = datetime.datetime(2026, 11, 11, 12, 5, 0, tzinfo=datetime.timezone.utc)
    expected_micros = 1794398700000000

    key_a = spanner_graph_lib.encode_composite_key(["a/b", "c", ts], col_types)
    key_b = spanner_graph_lib.encode_composite_key(["a", "b/c", ts], col_types)
    key_slash = spanner_graph_lib.encode_composite_key(
        ["a\\", "/b", expected_micros], col_types
    )
    self.assertNotEqual(key_a, key_b)
    self.assertEqual(key_a, rb"a\/b/c/1794398700000000")
    self.assertEqual(key_b, rb"a/b\/c/1794398700000000")
    self.assertEqual(
        spanner_graph_lib.decode_composite_key(key_a, col_types),
        ("a/b", "c", expected_micros),
    )
    self.assertEqual(
        spanner_graph_lib.decode_composite_key(key_b, col_types),
        ("a", "b/c", expected_micros),
    )
    self.assertEqual(
        spanner_graph_lib.decode_composite_key(key_slash, col_types),
        ("a\\", "/b", expected_micros),
    )

    binary_types = ["BYTES", "INT64", "BOOL"]
    encoded_bin = spanner_graph_lib.encode_composite_key(
        [b"\x00\xff/", -42, True], binary_types
    )
    self.assertEqual(encoded_bin, b"00ff2f/-42/true")
    self.assertEqual(
        spanner_graph_lib.decode_composite_key(encoded_bin, binary_types),
        (b"\x00\xff/", -42, True),
    )

  @parameterized.parameters(
      (b"only_one", ["STRING", "STRING"], "has 1 components, expected 2"),
      (b"a/b\\", ["STRING", "STRING"], "Trailing backslash"),
      (rb"a/\nb", ["STRING", "STRING"], "Invalid escape sequence"),
  )
  def test_decode_composite_key_rejects_malformed_inputs(
      self, bad_encoded, col_types, regex
  ):
    with self.assertRaisesRegex(ValueError, regex):
      spanner_graph_lib.decode_composite_key(bad_encoded, col_types)

  def test_schema_inference_composite_keys_and_vector_length(self):
    metadata = _composite_metadata()
    col_types = {
        ("Cells", "cell_emb"): "ARRAY<FLOAT>(vector_length=>4)",
    }
    schema = spanner_graph_lib.graph_schema_from_metadata(
        metadata, column_spanner_types=col_types
    )

    cell_features = schema.node_sets["Cell"].features
    self.assertEqual(
        cell_features["#id"],
        FeatureSchema(
            format=FeatureFormat.BYTES,
            semantic=FeatureSemantic.PRIMARY_ID,
            is_utf8_string=True,
        ),
    )
    self.assertEqual(
        cell_features["snapshot_ts"].semantic, FeatureSemantic.TIMESTAMP
    )
    self.assertEqual(
        cell_features["cell_emb"],
        FeatureSchema(
            format=FeatureFormat.FLOAT_32,
            semantic=FeatureSemantic.EMBEDDING,
            shape=(4,),
        ),
    )
    # The key columns of an edge table are not edge features.
    self.assertEmpty(schema.edge_sets["RackPartOfCell"].features)

  def test_read_graph_with_executor_aligns_keys_and_fills_defaults(self):
    metadata = _composite_metadata()
    col_types = {("Cells", "cell_emb"): "ARRAY<FLOAT>(vector_length=>2)"}
    schema = spanner_graph_lib.graph_schema_from_metadata(
        metadata, column_spanner_types=col_types
    )
    # Assign semantics on remaining UNKNOWN features so strict validation runs.
    schema.node_sets["Cell"].features[
        "cell_id"
    ].semantic = FeatureSemantic.CATEGORICAL
    schema.node_sets["Rack"].features[
        "rack_id"
    ].semantic = FeatureSemantic.CATEGORICAL
    schema.node_sets["Rack"].features[
        "cell_id"
    ].semantic = FeatureSemantic.CATEGORICAL
    schema.node_sets["Rack"].features[
        "machine_count"
    ].semantic = FeatureSemantic.NUMERICAL

    plan = spanner_graph_lib.resolve_read_plan(
        metadata, schema, column_spanner_types=col_types
    )
    # Destination key of RackPartOfCell must follow Cell's key order
    # (cell_id, snapshot_ts) despite reversed order in NodeTableRef.
    self.assertIn(
        "CAST(CONCAT(REPLACE(REPLACE(t.`cell_id`, '\\\\', '\\\\\\\\'), '/',"
        " '\\\\/'), '/', CAST(UNIX_MICROS(t.`snapshot_ts`) AS STRING)) AS"
        " BYTES) AS target_id",
        plan.edge_plans[0].query,
    )

    def fake_execute(query, parameters, parameter_types):
      del parameters, parameter_types
      if "FROM `Cells`" in query:
        return [
            [b"bc/100", "bc", 100, [1.5, 2.5]],
            [b"dz\\/1/100", "dz/1", 100, None],
        ]
      if "FROM `Racks`" in query and "AS source_id" not in query:
        return [
            [b"bc/r0/100", "r0", "bc", 100, 12],
            [b"dz\\/1/r0/100", "r0", "dz/1", 100, None],
        ]
      if "AS source_id" in query:
        return [
            [b"bc/r0/100", b"bc/100"],
            [b"dz\\/1/r0/100", b"dz\\/1/100"],
        ]
      raise AssertionError(f"Unexpected query: {query}")

    graph, returned_schema = spanner_graph_lib.read_graph_with_executor(
        metadata=metadata,
        execute_fn=fake_execute,
        bytes_are_base64=False,
        schema=schema,
        column_spanner_types=col_types,
        verbose=0,
    )
    validate_lib.validate_graph(
        graph, returned_schema, raise_on_error=True, raise_on_warning=True
    )

    expected = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "Cell": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "#id": np.array([b"bc/100", rb"dz\/1/100"]),
                    "cell_id": np.array([b"bc", b"dz/1"]),
                    "snapshot_ts": np.array([100, 100], dtype=np.int64),
                    "cell_emb": np.array(
                        [[1.5, 2.5], [0.0, 0.0]], dtype=np.float32
                    ),
                },
            ),
            "Rack": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "#id": np.array([b"bc/r0/100", rb"dz\/1/r0/100"]),
                    "rack_id": np.array([b"r0", b"r0"]),
                    "cell_id": np.array([b"bc", b"dz/1"]),
                    "snapshot_ts": np.array([100, 100], dtype=np.int64),
                    "machine_count": np.array([12, 0], dtype=np.int64),
                },
            ),
        },
        edge_sets={
            "RackPartOfCell": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 1], [0, 1]], dtype=np.int64)
            )
        },
    )
    test_util.assert_are_equal(self, graph, expected)

  def test_base64_and_raw_bytes_clients_read_the_same_graph(self):
    metadata = _composite_metadata()
    rack_table = metadata.node_tables[1]
    rack_table.property_definitions = [
        spanner_graph_metadata_lib.PropertyDefinition("blob", "blob"),
        spanner_graph_metadata_lib.PropertyDefinition("blobs", "blobs"),
    ]
    assert metadata.property_types is not None
    metadata.property_types["blob"] = "BYTES"
    metadata.property_types["blobs"] = "ARRAY<BYTES>"
    column_types = {
        ("Racks", "cell_id"): "STRING",
        ("Racks", "rack_id"): "STRING",
        ("Racks", "snapshot_ts"): "TIMESTAMP",
    }
    schema = spanner_graph_lib.graph_schema_from_metadata(
        metadata, column_spanner_types=column_types
    )
    del schema.node_sets["Cell"]
    schema.edge_sets.clear()
    rows = [
        [b"bc/r0/100", b"\x00\xff", [b"a", None, b""]],
        [b"bc/r1/100", b"", []],
        [b"bc/r2/100", None, None],
    ]

    def encode(value):
      if isinstance(value, bytes):
        return base64.b64encode(value)
      if isinstance(value, list):
        return [encode(item) for item in value]
      return value

    raw_graph, _ = spanner_graph_lib.read_graph_with_executor(
        metadata=metadata,
        execute_fn=lambda query, parameters, parameter_types: rows,
        bytes_are_base64=False,
        schema=schema,
        column_spanner_types=column_types,
        verbose=0,
    )
    base64_graph, _ = spanner_graph_lib.read_graph_with_executor(
        metadata=metadata,
        execute_fn=lambda query, parameters, parameter_types: [
            encode(row) for row in rows
        ],
        bytes_are_base64=True,
        schema=schema,
        column_spanner_types=column_types,
        verbose=0,
    )
    test_util.assert_are_equal(self, raw_graph, base64_graph)
    rack = raw_graph.node_sets["Rack"]
    self.assertEqual(rack.features["blob"].tolist(), [b"\x00\xff", b"", b""])
    self.assertEqual(rack.features["blobs"][0].tolist(), [b"a", b"", b""])
    self.assertEmpty(rack.features["blobs"][1])
    self.assertEmpty(rack.features["blobs"][2])

  def test_filters(self):
    metadata = _composite_metadata()
    column_types = {
        ("Cells", "cell_id"): "STRING(64)",
        ("Cells", "snapshot_ts"): "TIMESTAMP",
        ("Racks", "cell_id"): "STRING(64)",
        ("Racks", "snapshot_ts"): "TIMESTAMP",
        ("Racks", "machine_count"): "INT64",
        ("Racks", "tags"): "ARRAY<STRING(MAX)>",
    }
    schema = spanner_graph_lib.graph_schema_from_metadata(metadata)
    snapshot = datetime.datetime(2026, 11, 11, tzinfo=datetime.timezone.utc)
    calls = []

    def fake_execute(query, parameters, parameter_types):
      calls.append((query, parameters, parameter_types))
      return []

    spanner_graph_lib.read_graph_with_executor(
        metadata=metadata,
        execute_fn=fake_execute,
        bytes_are_base64=False,
        schema=schema,
        column_spanner_types=column_types,
        filters={"cell_id": "bc", "snapshot_ts": snapshot},
        verbose=0,
    )
    self.assertLen(calls, 3)
    for query, parameters, parameter_types in calls:
      self.assertEndsWith(
          query,
          "t.`cell_id` = @filter_cell_id AND t.`snapshot_ts` ="
          " @filter_snapshot_ts",
      )
      self.assertEqual(
          parameters,
          {"filter_cell_id": "bc", "filter_snapshot_ts": snapshot},
      )
      self.assertEqual(
          parameter_types,
          {"filter_cell_id": "STRING", "filter_snapshot_ts": "TIMESTAMP"},
      )
    self.assertIn("FROM `Cells` AS t WHERE t.`cell_id` =", calls[0][0])
    self.assertIn("IS NOT NULL AND t.`cell_id` =", calls[2][0])

    with self.assertRaisesRegex(ValueError, "'Cells' has no such column"):
      spanner_graph_lib.resolve_read_plan(
          metadata,
          schema,
          column_spanner_types=column_types,
          filters={"machine_count": 1},
      )
    del schema.node_sets["Cell"]
    schema.edge_sets.clear()
    with self.assertRaisesRegex(ValueError, "Cannot filter on column 'tags'"):
      spanner_graph_lib.resolve_read_plan(
          metadata,
          schema,
          column_spanner_types=column_types,
          filters={"tags": "x"},
      )
    with self.assertRaisesRegex(ValueError, "Invalid identifier"):
      spanner_graph_lib.resolve_read_plan(
          metadata,
          schema,
          column_spanner_types=column_types,
          filters={"cell_id = 'bc' OR t.cell_id": "x"},
      )

  def test_null_key_raises(self):
    metadata = _composite_metadata()
    schema = spanner_graph_lib.graph_schema_from_metadata(metadata)
    del schema.node_sets["Rack"]
    schema.edge_sets.clear()
    with self.assertRaisesRegex(ValueError, "'Cell' has a NULL key"):
      spanner_graph_lib.read_graph_with_executor(
          metadata=metadata,
          execute_fn=lambda query, parameters, parameter_types: [
              [None, "bc", 100, None]
          ],
          bytes_are_base64=False,
          schema=schema,
          verbose=0,
      )

  @parameterized.parameters(
      ("INT64", False, "t.`c`"),
      ("TIMESTAMP", False, "UNIX_MICROS(t.`c`)"),
      ("DATE", False, "CAST(t.`c` AS STRING)"),
      ("JSON", False, "IF(t.`c` IS NULL, NULL, TO_JSON_STRING(t.`c`))"),
      ("NUMERIC", False, "CAST(t.`c` AS FLOAT64)"),
      ("PROTO", False, "CAST(t.`c` AS BYTES)"),
      (
          "DATE",
          True,
          (
              "IF(t.`c` IS NULL, NULL, ARRAY(SELECT CAST(t AS STRING) FROM"
              " UNNEST(t.`c`) AS t WITH OFFSET off ORDER BY off))"
          ),
      ),
  )
  def test_to_json_sql(self, type_kind, is_array, expected):
    column = spanner_graph_lib.Column("c", type_kind, is_array)
    self.assertEqual(column.to_json_sql(column.sql("t")), expected)

  def test_derived_property_raises_unless_excluded_from_schema(self):
    metadata = _composite_metadata()
    cell_table = metadata.node_tables[0]
    cell_table.property_definitions = [
        *(cell_table.property_definitions or ()),
        spanner_graph_metadata_lib.PropertyDefinition(
            "machine_count", "LENGTH(cell_id)"
        ),
    ]
    schema = spanner_graph_lib.graph_schema_from_metadata(metadata)
    with self.assertRaisesRegex(
        NotImplementedError, "defined by a plain column"
    ):
      spanner_graph_lib.resolve_read_plan(metadata, schema)

    del schema.node_sets["Cell"].features["machine_count"]
    plan = spanner_graph_lib.resolve_read_plan(metadata, schema)
    self.assertNotIn("LENGTH", plan.node_plans[0].query)

  def test_composite_key_primary_feature_cannot_be_a_property(self):
    metadata = _composite_metadata()
    schema = spanner_graph_lib.graph_schema_from_metadata(metadata)
    cell_features = schema.node_sets["Cell"].features
    del cell_features["#id"]
    cell_features["cell_id"].semantic = FeatureSemantic.PRIMARY_ID
    with self.assertRaisesRegex(ValueError, "cannot be a property"):
      spanner_graph_lib.resolve_read_plan(metadata, schema)

  def test_find_element_table_by_label_or_name(self):
    metadata = _composite_metadata()
    for element_name in ("PART_OF", "RackPartOfCell"):
      table, label, label_is_shared = spanner_graph_lib.find_element_table(
          element_name, metadata.edge_tables
      )
      self.assertEqual(
          (table.name, label, label_is_shared),
          ("RackPartOfCell", "PART_OF", False),
      )

    metadata.edge_tables[0].label_names = ["PART_OF", "IN"]
    _, label, _ = spanner_graph_lib.find_element_table(
        "RackPartOfCell", metadata.edge_tables
    )
    self.assertIsNone(label)

    # A shared label does not identify a table, but the table names do.
    metadata.node_tables[0].label_names = ["Entity"]
    metadata.node_tables[1].label_names = ["Entity"]
    table, label, label_is_shared = spanner_graph_lib.find_element_table(
        "Rack", metadata.node_tables
    )
    self.assertEqual(
        (table.name, label, label_is_shared), ("Rack", "Entity", True)
    )
    with self.assertRaisesRegex(ValueError, "defined on 2 element tables"):
      spanner_graph_lib.find_element_table("Entity", metadata.node_tables)
    with self.assertRaisesRegex(ValueError, "neither a label nor an element"):
      spanner_graph_lib.find_element_table("Machine", metadata.node_tables)

  def test_sql_injection_rejected(self):
    bad_meta = _composite_metadata()
    bad_meta.node_tables[0].base_table_name = "Cells`; DROP TABLE Cells; --"
    schema = spanner_graph_lib.graph_schema_from_metadata(_composite_metadata())
    with self.assertRaisesRegex(ValueError, "Invalid identifier"):
      spanner_graph_lib.resolve_read_plan(bad_meta, schema)


if __name__ == "__main__":
  absltest.main()
