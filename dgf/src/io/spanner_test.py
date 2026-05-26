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

"""Tests for working with Cloud Spanner databases."""

from typing import Dict, Set, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
from dgf.src.io import spanner
from dgf.src.util import gen_test_graph
from google.cloud import spanner_v1

spanner_emulator = None

_ID_KEY = spanner._DEFAULT_ID_KEY


_CLIENT_EMULATOR_ENV_VAR = "SPANNER_EMULATOR_HOST"


class SpannerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.project_id = "dgf-test-project"
    self.instance_id = "dgf-test-instance"
    self.database_id = "dgf-test-db"

    if spanner_emulator is not None:
      self.spanner_emulator = spanner_emulator.Emulator()
      self.spanner_client = self.spanner_emulator.get_client(
          project=self.project_id
      )
      self.addCleanup(self.spanner_emulator.stop)
    else:
      import os

      if _CLIENT_EMULATOR_ENV_VAR in os.environ:
        from google.cloud import spanner as open_source_spanner

        self.spanner_client = open_source_spanner.Client(
            project=self.project_id
        )
      else:
        self.skipTest("Cloud Spanner Emulator is not available.")

    self.instance = self.spanner_client.instance(self.instance_id)
    self.instance.create().result(timeout=60)

    self.database = self.instance.database(self.database_id)
    self.database.create().result(timeout=60)

    self.hgraph = gen_test_graph.generate_in_memory_graph(
        node_ids=True, edge_ids=True
    )
    self.schema = gen_test_graph.generate_schema(
        node_ids=True, edge_ids=True, semantic=True
    )

    # Adding a primary key id feature instead of #id to the node and edge sets.
    for node_set in self.schema.node_sets.values():
      if "#id" in node_set.features:
        node_set.features["id"] = node_set.features["#id"]
        del node_set.features["#id"]

    for edge_set in self.schema.edge_sets.values():
      if "#id" in edge_set.features:
        edge_set.features["id"] = edge_set.features["#id"]
        del edge_set.features["#id"]

  def test_schema_to_spanner_ddl(self):
    ddl_statements = spanner.schema_to_spanner_ddl(self.schema)

    self.assertIn("CREATE TABLE n1", ddl_statements["n1"])
    self.assertIn("f2 ARRAY<FLOAT32>", ddl_statements["n1"])
    self.assertIn("f1 BYTES(MAX)", ddl_statements["n1"])
    self.assertIn(f"{_ID_KEY} BYTES(MAX) NOT NULL", ddl_statements["n1"])
    self.assertIn(f"PRIMARY KEY ({_ID_KEY})", ddl_statements["n1"])

    self.assertIn("CREATE TABLE n2", ddl_statements["n2"])
    self.assertIn("f3 INT64", ddl_statements["n2"])
    self.assertIn("f4 INT64", ddl_statements["n2"])
    self.assertIn("f5 ARRAY<INT64>", ddl_statements["n2"])
    self.assertIn(f"{_ID_KEY} INT64 NOT NULL", ddl_statements["n2"])
    self.assertIn(f"PRIMARY KEY ({_ID_KEY})", ddl_statements["n2"])

    self.assertIn("CREATE TABLE e1", ddl_statements["e1"])
    self.assertIn("source BYTES(MAX) NOT NULL", ddl_statements["e1"])
    self.assertIn("target BYTES(MAX) NOT NULL", ddl_statements["e1"])
    self.assertIn(f"{_ID_KEY} BYTES(MAX)", ddl_statements["e1"])
    self.assertIn(
        f"PRIMARY KEY (source, target, {_ID_KEY})", ddl_statements["e1"]
    )

    self.assertIn("CREATE TABLE e2", ddl_statements["e2"])
    self.assertIn("source BYTES(MAX) NOT NULL", ddl_statements["e2"])
    self.assertIn("target INT64 NOT NULL", ddl_statements["e2"])
    self.assertIn(f"{_ID_KEY} BYTES(MAX)", ddl_statements["e2"])
    self.assertIn(
        f"PRIMARY KEY (source, target, {_ID_KEY})", ddl_statements["e2"]
    )

  @parameterized.named_parameters(
      ("with_beam", True),
      ("without_beam", False),
  )
  def test_create_graph_tables_from_schema(self, with_beam):
    # TODO(bmayer): Add some edge features to tests.
    if with_beam:
      with test_pipeline.TestPipeline() as p:
        _ = (
            p
            | beam.Create([None])
            | beam.ParDo(
                spanner.CreateSpannerTables(
                    schema=self.schema,
                    project_id=self.project_id,
                    instance_id=self.instance_id,
                    database_id=self.database_id,
                )
            )
        )
    else:
      spanner.create_spanner_tables_from_graph_schema(
          schema=self.schema,
          project_id=self.project_id,
          instance_id=self.instance_id,
          database_id=self.database_id,
      )

    node_set_table_schemas: Dict[str, Set[Tuple[str, str, str]]] = {}
    for node_set_name in self.schema.node_sets.keys():
      with self.database.snapshot() as snapshot:
        results = snapshot.execute_sql(
            "SELECT column_name, spanner_type, is_nullable FROM"
            " information_schema.columns WHERE table_name = @table_name",
            params={"table_name": node_set_name},
            param_types={"table_name": spanner_v1.param_types.STRING},
        )

        node_set_table_schemas[node_set_name] = set()
        for column_name, spanner_type, is_nullable in results:
          node_set_table_schemas[node_set_name].add(
              (column_name, spanner_type, is_nullable)
          )

    self.assertDictEqual(
        node_set_table_schemas,
        {
            "n1": set([
                ("f2", "ARRAY<FLOAT32>", "YES"),
                ("f1", "BYTES(MAX)", "YES"),
                (_ID_KEY, "BYTES(MAX)", "NO"),
            ]),
            "n2": set([
                ("f3", "INT64", "YES"),
                ("f4", "INT64", "YES"),
                ("f5", "ARRAY<INT64>", "YES"),
                (_ID_KEY, "INT64", "NO"),
            ]),
        },
    )

    # edge_set_table_schemas = defaultdict(list)
    edge_set_table_schemas: Dict[str, Set[Tuple[str, str, str]]] = {}
    for edge_set_name in self.schema.edge_sets.keys():
      with self.database.snapshot() as snapshot:
        results = snapshot.execute_sql(
            "SELECT column_name, spanner_type, is_nullable FROM"
            " information_schema.columns WHERE table_name = @table_name",
            params={"table_name": edge_set_name},
            param_types={"table_name": spanner_v1.param_types.STRING},
        )
        edge_set_table_schemas[edge_set_name] = set()
        for column_name, spanner_type, is_nullable in results:
          edge_set_table_schemas[edge_set_name].add(
              (column_name, spanner_type, is_nullable)
          )

    self.assertDictEqual(
        edge_set_table_schemas,
        {
            "e1": set([
                ("source", "BYTES(MAX)", "NO"),
                ("target", "BYTES(MAX)", "NO"),
                (_ID_KEY, "BYTES(MAX)", "NO"),
            ]),
            "e2": set([
                ("source", "BYTES(MAX)", "NO"),
                ("target", "INT64", "NO"),
                (_ID_KEY, "BYTES(MAX)", "NO"),
            ]),
        },
    )


if __name__ == "__main__":
  absltest.main()
