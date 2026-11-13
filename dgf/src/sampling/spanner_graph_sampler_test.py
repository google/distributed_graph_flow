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

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import spanner_graph_sampler as spanner_graph_sampler_lib
from dgf.src.util import test_util
import numpy as np

FeatureFormat = schema_lib.FeatureFormat
FeatureSchema = schema_lib.FeatureSchema
FeatureSemantic = schema_lib.FeatureSemantic
QueryEngine = spanner_graph_sampler_lib.QueryEngine

_PAPER_PROPERTIES = {"year": 2010, "tag": None, "embedding": [1.0, 2.0]}
# Row of the sampling query for the paper p1, sampled from the seed p1.
_PAPER_ROW = [b"p1", b"p1", "node", "paper", None, None, _PAPER_PROPERTIES]


def _property(name: str, column: str) -> dict[str, str]:
  return {"propertyDeclarationName": name, "valueExpressionSql": column}


def _toy_metadata() -> spanner_graph_metadata_lib.SpannerGraphMetadata:
  """Returns the metadata of a property graph with renamed tables / columns."""
  return spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict({  # pyrefly: ignore[missing-attribute]
      "catalog": "",
      "schema": "",
      "name": "toy_graph",
      "nodeTables": [
          {
              "name": "Papers",
              "kind": "NODE",
              "baseCatalogName": "",
              "baseSchemaName": "",
              "baseTableName": "Papers",
              "keyColumns": ["PaperId"],
              "labelNames": ["paper"],
              "propertyDefinitions": [
                  _property("id", "PaperId"),
                  _property("year", "Year"),
                  _property("tag", "Tag"),
                  _property("embedding", "Embedding"),
              ],
          },
          {
              "name": "Authors",
              "kind": "NODE",
              "baseCatalogName": "",
              "baseSchemaName": "",
              "baseTableName": "Authors",
              "keyColumns": ["AuthorId"],
              "labelNames": ["author"],
              "propertyDefinitions": [_property("name", "AuthorId")],
          },
      ],
      "edgeTables": [{
          "name": "Writes",
          "kind": "EDGE",
          "baseCatalogName": "",
          "baseSchemaName": "",
          "baseTableName": "Writes",
          "keyColumns": ["AuthorId", "PaperId"],
          "labelNames": ["writes"],
          "propertyDefinitions": [_property("position", "Position")],
          "sourceNodeTable": {
              "nodeTableName": "Authors",
              "edgeTableColumns": ["AuthorId"],
              "nodeTableColumns": ["AuthorId"],
          },
          "destinationNodeTable": {
              "nodeTableName": "Papers",
              "edgeTableColumns": ["PaperId"],
              "nodeTableColumns": ["PaperId"],
          },
      }],
      "labels": [],
      "propertyDeclarations": [
          {"name": "id", "type": "BYTES"},
          {"name": "year", "type": "INT64"},
          {"name": "tag", "type": "BYTES"},
          {"name": "embedding", "type": "ARRAY<FLOAT32>"},
          {"name": "name", "type": "STRING"},
          {"name": "position", "type": "INT64"},
      ],
  })


def _toy_schema() -> schema_lib.GraphSchema:
  return schema_lib.GraphSchema(
      node_sets={
          "paper": schema_lib.NodeSchema(
              features={
                  "id": FeatureSchema(
                      format=FeatureFormat.BYTES,
                      semantic=FeatureSemantic.PRIMARY_ID,
                  ),
                  "year": FeatureSchema(
                      format=FeatureFormat.INTEGER_64,
                      semantic=FeatureSemantic.NUMERICAL,
                  ),
                  "tag": FeatureSchema(
                      format=FeatureFormat.BYTES,
                      semantic=FeatureSemantic.CATEGORICAL,
                  ),
                  "embedding": FeatureSchema(
                      format=FeatureFormat.FLOAT_32,
                      semantic=FeatureSemantic.EMBEDDING,
                      shape=(2,),
                  ),
              }
          ),
          "author": schema_lib.NodeSchema(
              features={
                  "name": FeatureSchema(
                      format=FeatureFormat.BYTES,
                      semantic=FeatureSemantic.PRIMARY_ID,
                      is_utf8_string=True,
                  )
              }
          ),
      },
      edge_sets={
          "writes": schema_lib.EdgeSchema(
              source="author",
              target="paper",
              features={
                  "position": FeatureSchema(
                      format=FeatureFormat.INTEGER_64,
                      semantic=FeatureSemantic.NUMERICAL,
                  )
              },
          )
      },
  )


def _table_name_schema() -> schema_lib.GraphSchema:
  """Returns a schema named after the element tables instead of the labels."""
  return schema_lib.GraphSchema(
      node_sets={
          "Papers": schema_lib.NodeSchema(
              features={
                  "id": FeatureSchema(
                      format=FeatureFormat.BYTES,
                      semantic=FeatureSemantic.PRIMARY_ID,
                  )
              }
          ),
          "Authors": schema_lib.NodeSchema(
              features={
                  "name": FeatureSchema(
                      format=FeatureFormat.BYTES,
                      semantic=FeatureSemantic.PRIMARY_ID,
                  )
              }
          ),
      },
      edge_sets={
          "Writes": schema_lib.EdgeSchema(source="Authors", target="Papers")
      },
  )


def _plan() -> config_lib.SamplingPlan:
  return config_lib.simple_sampling_config_to_sampling_plan(
      config_lib.SimpleSamplingConfig(
          seed_nodeset="paper", num_hops=2, hop_width=3, multi_visit=False
      ),
      _toy_schema(),
  )


class SpannerGraphSamplerTest(parameterized.TestCase):

  @parameterized.parameters(
      (QueryEngine.SQL, False, "sql"),
      (QueryEngine.SQL, True, "sql_debug"),
      (QueryEngine.GQL, False, "gql"),
  )
  def test_query(self, engine, debug_sampling, golden_suffix):
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), _toy_schema()
    )
    query = spanner_graph_sampler_lib.generate_query(
        tables=tables,
        plan=_plan(),
        engine=engine,
        graph_name="toy_graph",
        debug_sampling=debug_sampling,
    )
    test_util.assert_golden_string(
        self, query, f"spanner_graph_sampler_test_query_{golden_suffix}.sql"
    )

  def test_missing_label_raises(self):
    schema = _toy_schema()
    schema.node_sets["book"] = schema.node_sets.pop("paper")
    with self.assertRaisesRegex(ValueError, "'book' is neither a label"):
      spanner_graph_sampler_lib.resolve_graph_tables(_toy_metadata(), schema)

  def test_missing_property_raises(self):
    schema = _toy_schema()
    schema.node_sets["paper"].features["color"] = FeatureSchema(
        format=FeatureFormat.INTEGER_64
    )
    with self.assertRaisesRegex(ValueError, "'color' of 'paper' is not a"):
      spanner_graph_sampler_lib.resolve_graph_tables(_toy_metadata(), schema)

  def test_incompatible_type_raises(self):
    schema = _toy_schema()
    schema.node_sets["paper"].features["year"] = FeatureSchema(
        format=FeatureFormat.FLOAT_32
    )
    with self.assertRaisesRegex(ValueError, "not compatible"):
      spanner_graph_sampler_lib.resolve_graph_tables(_toy_metadata(), schema)

  def test_rows_to_in_memory_graphs(self):
    schema = _toy_schema()
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), schema
    )
    paper_1 = {"year": 2010, "tag": "AP8=", "embedding": [1.0, 2.0]}
    paper_2 = {"year": None, "tag": None, "embedding": None}
    rows = [
        # Seed p2.
        [b"p2", b"a0", "node", "author", None, None, None],
        [b"p2", b"p2", "node", "paper", None, None, paper_2],
        [b"p2", b"p1", "node", "paper", None, None, paper_1],
        [b"p2", None, "edge", "writes", b"a0", b"p2", {"position": 3}],
        [b"p2", None, "edge", "writes", b"a0", b"p1", {"position": 4}],
        # A parallel edge.
        [b"p2", None, "edge", "writes", b"a0", b"p2", {"position": 5}],
        # Seed p1.
        [b"p1", b"p1", "node", "paper", None, None, paper_1],
    ]
    graphs = spanner_graph_sampler_lib.rows_to_in_memory_graphs(
        rows, schema, tables, "paper", [b"p2", b"p1"]
    )
    test_util.assert_are_equal(
        self,
        graphs,
        [
            in_memory_graph_lib.InMemoryGraph(
                node_sets={
                    "paper": in_memory_graph_lib.InMemoryNodeSet(
                        num_nodes=2,
                        features={
                            "id": np.array([b"p2", b"p1"]),
                            "year": np.array([0, 2010]),
                            "tag": np.array([b"", b"\x00\xff"], dtype=object),
                            "embedding": np.array(
                                [[0, 0], [1, 2]], dtype=np.float32
                            ),
                        },
                    ),
                    "author": in_memory_graph_lib.InMemoryNodeSet(
                        num_nodes=1, features={"name": np.array([b"a0"])}
                    ),
                },
                edge_sets={
                    "writes": in_memory_graph_lib.InMemoryEdgeSet(
                        adjacency=np.array([[0, 0], [0, 1]]),
                        features={"position": np.array([3, 4])},
                    )
                },
            ),
            in_memory_graph_lib.InMemoryGraph(
                node_sets={
                    "paper": in_memory_graph_lib.InMemoryNodeSet(
                        num_nodes=1,
                        features={
                            "id": np.array([b"p1"]),
                            "year": np.array([2010]),
                            "tag": np.array([b"\x00\xff"], dtype=object),
                            "embedding": np.array([[1, 2]], dtype=np.float32),
                        },
                    ),
                    "author": in_memory_graph_lib.InMemoryNodeSet(
                        num_nodes=0,
                        features={"name": np.zeros(0, dtype=object)},
                    ),
                },
                edge_sets={
                    "writes": in_memory_graph_lib.InMemoryEdgeSet(
                        adjacency=np.zeros((2, 0), dtype=np.int64),
                        features={"position": np.zeros(0, dtype=np.int64)},
                    )
                },
            ),
        ],
    )

  def test_rows_to_in_memory_graphs_missing_seed_raises(self):
    schema = _toy_schema()
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), schema
    )
    with self.assertRaisesRegex(ValueError, "not found"):
      spanner_graph_sampler_lib.rows_to_in_memory_graphs(
          [], schema, tables, "paper", [b"p0"]
      )

  @parameterized.parameters(QueryEngine.SQL, QueryEngine.GQL)
  def test_hop_width_zero_skips_hop(self, engine):
    schema = _toy_schema()
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), schema
    )
    plan = config_lib.SamplingPlan(
        root=config_lib.PlanNode(
            nodeset="paper",
            children=[
                config_lib.PlanEdge(
                    edgeset="writes",
                    reversed=True,
                    hop_width=0,
                    node=config_lib.PlanNode(nodeset="author"),
                )
            ],
        )
    )
    query = spanner_graph_sampler_lib.generate_query(
        tables=tables, plan=plan, engine=engine, graph_name="toy_graph"
    )
    self.assertNotIn("_dgf_hop_1", query)
    self.assertNotIn("Writes", query)

  @parameterized.parameters(
      (-1, ValueError), (1.5, ValueError), ("1; DROP TABLE Papers", TypeError)
  )
  def test_invalid_hop_width_raises(self, invalid_hop_width, error):
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), _toy_schema()
    )
    plan = config_lib.SamplingPlan(
        root=config_lib.PlanNode(
            nodeset="paper",
            children=[
                config_lib.PlanEdge(
                    edgeset="writes",
                    reversed=True,
                    hop_width=invalid_hop_width,  # pyrefly: ignore[bad-argument-type]
                    node=config_lib.PlanNode(nodeset="author"),
                )
            ],
        )
    )
    with self.assertRaises(error):
      spanner_graph_sampler_lib.generate_query(
          tables=tables,
          plan=plan,
          engine=QueryEngine.SQL,
          graph_name="toy_graph",
      )

  def test_resolve_by_element_table_name(self):
    schema = _table_name_schema()
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), schema
    )
    self.assertEqual(tables.node_tables["Papers"].label, "paper")
    self.assertEqual(tables.node_tables["Authors"].label, "author")
    self.assertEqual(tables.edge_tables["Writes"].label, "writes")

  @parameterized.parameters(QueryEngine.SQL, QueryEngine.GQL)
  def test_int32_bool_timestamp_and_variable_length_features(self, engine):
    metadata = spanner_graph_metadata_lib.SpannerGraphMetadata.from_dict({  # pyrefly: ignore[missing-attribute]
        "catalog": "",
        "schema": "",
        "name": "g",
        "nodeTables": [{
            "name": "Events",
            "kind": "NODE",
            "baseCatalogName": "",
            "baseSchemaName": "",
            "baseTableName": "Events",
            "keyColumns": ["EventId"],
            "labelNames": ["event"],
            "propertyDefinitions": [
                _property("id", "EventId"),
                _property("count32", "Count32"),
                _property("active", "Active"),
                _property("created_at", "CreatedAt"),
                _property("history", "History"),
                _property("scores", "Scores"),
            ],
        }],
        "edgeTables": [{
            "name": "Links",
            "kind": "EDGE",
            "baseCatalogName": "",
            "baseSchemaName": "",
            "baseTableName": "Links",
            "keyColumns": ["Src", "Dst"],
            "labelNames": ["links"],
            "propertyDefinitions": [_property("history", "History")],
            "sourceNodeTable": {
                "nodeTableName": "Events",
                "edgeTableColumns": ["Src"],
                "nodeTableColumns": ["EventId"],
            },
            "destinationNodeTable": {
                "nodeTableName": "Events",
                "edgeTableColumns": ["Dst"],
                "nodeTableColumns": ["EventId"],
            },
        }],
        "labels": [],
        "propertyDeclarations": [
            {"name": "id", "type": "BYTES"},
            {"name": "count32", "type": "INT32"},
            {"name": "active", "type": "BOOL"},
            {"name": "created_at", "type": "TIMESTAMP"},
            {"name": "history", "type": "ARRAY<TIMESTAMP>"},
            {"name": "scores", "type": "ARRAY<FLOAT32>"},
        ],
    })
    schema = schema_lib.GraphSchema(
        node_sets={
            "event": schema_lib.NodeSchema(
                features={
                    "id": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.PRIMARY_ID,
                    ),
                    "count32": FeatureSchema(format=FeatureFormat.INTEGER_32),
                    "active": FeatureSchema(format=FeatureFormat.BOOL),
                    "created_at": FeatureSchema(
                        format=FeatureFormat.INTEGER_64
                    ),
                    "history": FeatureSchema(
                        format=FeatureFormat.INTEGER_64, shape=(2,)
                    ),
                    "scores": FeatureSchema(
                        format=FeatureFormat.FLOAT_32, shape=(None,)
                    ),
                }
            )
        },
        edge_sets={
            "links": schema_lib.EdgeSchema(
                source="event",
                target="event",
                features={
                    "history": FeatureSchema(
                        format=FeatureFormat.INTEGER_64, shape=(2,)
                    )
                },
            )
        },
    )
    tables = spanner_graph_sampler_lib.resolve_graph_tables(metadata, schema)
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        config_lib.SimpleSamplingConfig(
            seed_nodeset="event", num_hops=1, hop_width=2, multi_visit=False
        ),
        schema,
    )
    query = spanner_graph_sampler_lib.generate_query(
        tables=tables, plan=plan, engine=engine, graph_name="g"
    )
    self.assertIn("IF(", query)
    self.assertIn("WITH OFFSET off ORDER BY off", query)
    self.assertIn("UNIX_MICROS(n.`CreatedAt`)", query)

    rows = [
        [
            b"e0",
            b"e0",
            "node",
            "event",
            None,
            None,
            {
                "count32": 7,
                "active": True,
                "created_at": 100,
                "history": [10, 20],
                "scores": [1.5, 2.5, 3.5],
            },
        ],
        [
            b"e0",
            b"e1",
            "node",
            "event",
            None,
            None,
            {
                "count32": None,
                "active": None,
                "created_at": None,
                "history": None,
                "scores": None,
            },
        ],
        [b"e0", None, "edge", "links", b"e0", b"e1", {"history": None}],
    ]
    (graph,) = spanner_graph_sampler_lib.rows_to_in_memory_graphs(
        rows, schema, tables, "event", [b"e0"]
    )
    event_features = graph.node_sets["event"].features
    np.testing.assert_array_equal(
        event_features["count32"], np.array([7, 0], dtype=np.int32)
    )
    np.testing.assert_array_equal(event_features["active"], [True, False])
    np.testing.assert_array_equal(event_features["created_at"], [100, 0])
    np.testing.assert_array_equal(event_features["history"], [[10, 20], [0, 0]])
    np.testing.assert_array_equal(
        event_features["scores"][0], np.array([1.5, 2.5, 3.5], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        event_features["scores"][1], np.array([], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        graph.edge_sets["links"].features["history"], [[0, 0]]
    )

  def test_wrong_static_array_length_raises(self):
    schema = _toy_schema()
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), schema
    )
    rows = [[
        b"p1",
        b"p1",
        "node",
        "paper",
        None,
        None,
        {"year": 2010, "tag": None, "embedding": [1.0]},
    ]]
    with self.assertRaisesRegex(
        ValueError, "Feature 'embedding' of 'paper' has length 1, expected 2"
    ):
      spanner_graph_sampler_lib.rows_to_in_memory_graphs(
          rows, schema, tables, "paper", [b"p1"]
      )

  def test_2d_feature_raises(self):
    schema = _toy_schema()
    schema.node_sets["paper"].features["embedding"] = FeatureSchema(
        format=FeatureFormat.FLOAT_32, shape=(2, 2)
    )
    with self.assertRaisesRegex(NotImplementedError, "Only scalar and 1D"):
      spanner_graph_sampler_lib.resolve_graph_tables(_toy_metadata(), schema)

  def test_composite_key_raises(self):
    metadata = _toy_metadata()
    metadata.node_tables[0].key_columns = ["PaperId", "Year"]
    with self.assertRaisesRegex(NotImplementedError, "composite key"):
      spanner_graph_sampler_lib.resolve_graph_tables(metadata, _toy_schema())

  @parameterized.named_parameters(
      ("duplicate_node", [_PAPER_ROW, _PAPER_ROW], "Duplicate node"),
      (
          "edge_to_missing_node",
          [
              _PAPER_ROW,
              [
                  b"p1",
                  None,
                  "edge",
                  "writes",
                  b"a_missing",
                  b"p1",
                  {"position": 1},
              ],
          ],
          "missing from the Spanner node",
      ),
      (
          "unexpected_seed",
          [[b"other", b"p1", "node", "paper", None, None, _PAPER_PROPERTIES]],
          "Unexpected seed_id",
      ),
      (
          "missing_seed_node",
          [[b"p1", b"a0", "node", "author", None, None, None]],
          "Seed nodes .* not found",
      ),
  )
  def test_invalid_rows_raise(self, rows, message):
    schema = _toy_schema()
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), schema
    )
    with self.assertRaisesRegex(ValueError, message):
      spanner_graph_sampler_lib.rows_to_in_memory_graphs(
          rows, schema, tables, "paper", [b"p1"]
      )

  def test_feature_name_injection_rejected(self):
    schema = _toy_schema()
    schema.node_sets["paper"].features["year'); DROP TABLE Papers; --"] = (
        FeatureSchema(format=FeatureFormat.INTEGER_64)
    )
    with self.assertRaisesRegex(ValueError, "Invalid identifier"):
      spanner_graph_sampler_lib.resolve_graph_tables(_toy_metadata(), schema)

  def test_nodeset_name_injection_rejected(self):
    schema = _toy_schema()
    schema.node_sets["paper' OR '1'='1"] = schema.node_sets.pop("paper")
    with self.assertRaisesRegex(ValueError, "Invalid identifier"):
      spanner_graph_sampler_lib.resolve_graph_tables(_toy_metadata(), schema)

  def test_property_defined_by_expression_raises(self):
    metadata = _toy_metadata()
    metadata.node_tables[0].property_definitions = [
        spanner_graph_metadata_lib.PropertyDefinition("id", "PaperId"),
        spanner_graph_metadata_lib.PropertyDefinition("year", "Year + 1"),
    ]
    with self.assertRaisesRegex(
        NotImplementedError,
        "'year' of 'paper' is defined by the expression 'Year \\+ 1'",
    ):
      spanner_graph_sampler_lib.resolve_graph_tables(metadata, _toy_schema())

  def test_table_name_injection_rejected(self):
    metadata = _toy_metadata()
    metadata.node_tables[0].base_table_name = "Papers`; DROP TABLE x"
    with self.assertRaisesRegex(ValueError, "Invalid identifier"):
      spanner_graph_sampler_lib.resolve_graph_tables(metadata, _toy_schema())

  def test_graph_name_injection_rejected(self):
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), _toy_schema()
    )
    with self.assertRaisesRegex(ValueError, "Invalid identifier"):
      spanner_graph_sampler_lib.generate_query(
          tables=tables,
          plan=_plan(),
          engine=QueryEngine.GQL,
          graph_name="toy_graph` )",
      )

  def test_label_on_several_element_tables_raises(self):
    metadata = _toy_metadata()
    metadata.node_tables[1].label_names = ["paper"]
    with self.assertRaisesRegex(
        ValueError, "label 'paper' is defined on 2 element tables"
    ):
      spanner_graph_sampler_lib.resolve_graph_tables(metadata, _toy_schema())

  def test_element_table_with_several_labels_raises(self):
    metadata = _toy_metadata()
    metadata.node_tables[0].label_names = ["paper", "publication"]
    with self.assertRaisesRegex(
        ValueError, "element table 'Papers' has 2 labels"
    ):
      spanner_graph_sampler_lib.resolve_graph_tables(
          metadata, _table_name_schema()
      )

  def test_shared_label_is_only_supported_by_sql_engine(self):
    # Papers and Authors both have the label entity: A GQL query on this label
    # would match both tables.
    metadata = _toy_metadata()
    metadata.node_tables[0].label_names = ["entity"]
    metadata.node_tables[1].label_names = ["entity"]
    schema = _table_name_schema()
    tables = spanner_graph_sampler_lib.resolve_graph_tables(metadata, schema)
    self.assertTrue(tables.node_tables["Papers"].label_is_shared)
    self.assertFalse(tables.edge_tables["Writes"].label_is_shared)
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        config_lib.SimpleSamplingConfig(
            seed_nodeset="Papers", num_hops=1, hop_width=3, multi_visit=False
        ),
        schema,
    )
    query = spanner_graph_sampler_lib.generate_query(
        tables=tables, plan=plan, engine=QueryEngine.SQL, graph_name="toy_graph"
    )
    self.assertIn("FROM `Writes` AS e", query)
    with self.assertRaisesRegex(
        NotImplementedError, "label 'entity' is defined on several"
    ):
      spanner_graph_sampler_lib.generate_query(
          tables=tables,
          plan=plan,
          engine=QueryEngine.GQL,
          graph_name="toy_graph",
      )

  @parameterized.parameters(QueryEngine.SQL, QueryEngine.GQL)
  def test_each_hop_has_a_different_sampling_hash(self, engine):
    tables = spanner_graph_sampler_lib.resolve_graph_tables(
        _toy_metadata(), _toy_schema()
    )
    query = spanner_graph_sampler_lib.generate_query(
        tables=tables, plan=_plan(), engine=engine, graph_name="toy_graph"
    )
    self.assertIn("CONCAT(@sampling_seed, b'1:', p.seed_id,", query)
    self.assertIn("CONCAT(@sampling_seed, b'2:', p.seed_id,", query)

  def test_query_engine_enum_values_match_names(self):
    for member in QueryEngine:
      self.assertEqual(member.value, member.name)
    with self.assertRaises(ValueError):
      QueryEngine("sql")


if __name__ == "__main__":
  absltest.main()
