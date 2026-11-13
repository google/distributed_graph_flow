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
    # Edge table keeps snapshot_ts even though it is in base table key_columns.
    edge_features = schema.edge_sets["RackPartOfCell"].features
    self.assertIn("snapshot_ts", edge_features)
    self.assertEqual(
        edge_features["snapshot_ts"].semantic, FeatureSemantic.TIMESTAMP
    )

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

    def fake_execute(query, bytes_cols, bytes_array_cols):
      del bytes_cols, bytes_array_cols
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
            [b"bc/r0/100", b"bc/100", 100],
            [b"dz\\/1/r0/100", b"dz\\/1/100", 100],
        ]
      raise AssertionError(f"Unexpected query: {query}")

    graph, returned_schema = spanner_graph_lib.read_graph_with_executor(
        metadata=metadata,
        execute_fn=fake_execute,
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
                adjacency=np.array([[0, 1], [0, 1]], dtype=np.int64),
                features={
                    "snapshot_ts": np.array([100, 100], dtype=np.int64),
                },
            )
        },
    )
    test_util.assert_are_equal(self, graph, expected)

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
    with self.assertRaisesRegex(
        NotImplementedError, "defined by a plain column"
    ):
      spanner_graph_lib.resolve_read_plan(
          metadata,
          spanner_graph_lib.graph_schema_from_metadata(
              metadata, combine_as_json=True
          ),
          combine_as_json=True,
      )

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
