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

import base64
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling.gcp import spanner_graph_sampler as spanner_graph_sampler_lib
from google.cloud import spanner
import numpy as np


def _toy_schema() -> schema_lib.GraphSchema:
  return schema_lib.GraphSchema(
      node_sets={
          "nodes": schema_lib.NodeSchema(
              features={
                  "id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
              }
          )
      },
      edge_sets={
          "edges": schema_lib.EdgeSchema(source="nodes", target="nodes")
      },
  )


def _metadata_dict(key_type: str) -> dict[str, Any]:
  return {
      "catalog": "",
      "schema": "",
      "name": "g",
      "nodeTables": [{
          "name": "Nodes",
          "kind": "NODE",
          "baseCatalogName": "",
          "baseSchemaName": "",
          "baseTableName": "Nodes",
          "keyColumns": ["NodeId"],
          "labelNames": ["nodes"],
          "propertyDefinitions": [{
              "propertyDeclarationName": "id",
              "valueExpressionSql": "NodeId",
          }],
      }],
      "edgeTables": [{
          "name": "Edges",
          "kind": "EDGE",
          "baseCatalogName": "",
          "baseSchemaName": "",
          "baseTableName": "Edges",
          "keyColumns": ["Src", "Dst"],
          "labelNames": ["edges"],
          "sourceNodeTable": {
              "nodeTableName": "Nodes",
              "edgeTableColumns": ["Src"],
              "nodeTableColumns": ["NodeId"],
          },
          "destinationNodeTable": {
              "nodeTableName": "Nodes",
              "edgeTableColumns": ["Dst"],
              "nodeTableColumns": ["NodeId"],
          },
      }],
      "labels": [],
      "propertyDeclarations": [{"name": "id", "type": key_type}],
  }


class _FakeSnapshot:

  def __init__(self, database: "_FakeDatabase"):
    self._database = database

  def __enter__(self) -> "_FakeSnapshot":
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    pass

  def execute_sql(
      self, sql: str, params: dict[str, Any], param_types: dict[str, Any]
  ) -> list[list[Any]]:
    self._database.calls.append((sql, params, param_types))
    if "INFORMATION_SCHEMA.PROPERTY_GRAPHS" in sql:
      return self._database.metadata_rows
    return self._database.sample_rows


class _FakeDatabase:

  def __init__(
      self, metadata_rows: list[list[Any]], sample_rows: list[list[Any]]
  ):
    self.metadata_rows = metadata_rows
    self.sample_rows = sample_rows
    self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

  def snapshot(self) -> _FakeSnapshot:
    return _FakeSnapshot(self)


class _FakeClient:
  """Fake of spanner.Client returning the same database for all the names."""

  def __init__(self, database: _FakeDatabase):
    self._database = database

  def instance(self, instance: str) -> "_FakeClient":
    del instance
    return self

  def database(self, database: str) -> _FakeDatabase:
    del database
    return self._database


class SpannerGraphSamplerGcpTest(parameterized.TestCase):

  @parameterized.parameters(
      (dict(), "multi_visit=True"),
      (dict(multi_visit=False, with_replacement=True), "with_replacement=True"),
      (dict(multi_visit=False, temporal_sampling=True), "temporal_sampling"),
  )
  def test_unsupported_options_raise(self, options, message):
    fake_db = _FakeDatabase(
        metadata_rows=[[_metadata_dict("BYTES")]], sample_rows=[]
    )
    with self.assertRaisesRegex(NotImplementedError, message):
      spanner_graph_sampler_lib.SpannerGraphSamplerGcp(
          database=fake_db,
          graph="g",
          plan=config_lib.SimpleSamplingConfig(seed_nodeset="nodes", **options),
          schema=_toy_schema(),
          debug_sampling=False,
          query_engine="AUTO",
      )

  def test_decode_ids(self):
    row = [
        base64.b64encode(b"seed"),
        None,
        "edge",
        "edges",
        base64.b64encode(b"\x00\xff"),
        base64.b64encode(b"dst"),
        {"weight": 1},
    ]
    self.assertEqual(
        spanner_graph_sampler_lib._decode_ids(row),
        [b"seed", None, "edge", "edges", b"\x00\xff", b"dst", {"weight": 1}],
    )

  def test_sample_bytes_key_debug_omits_sampling_seed(self):
    fake_db = _FakeDatabase(
        metadata_rows=[[_metadata_dict("BYTES")]],
        sample_rows=[[
            base64.b64encode(b"n0"),
            base64.b64encode(b"n0"),
            "node",
            "nodes",
            None,
            None,
            None,
        ]],
    )
    sampler = spanner_graph_sampler_lib.SpannerGraphSamplerGcp(
        database=fake_db,
        graph="g",
        plan=config_lib.SimpleSamplingConfig(
            seed_nodeset="nodes", num_hops=1, hop_width=2, multi_visit=False
        ),
        schema=_toy_schema(),
        debug_sampling=True,
        query_engine="SQL",
    )
    self.assertLen(fake_db.calls, 1)
    self.assertEqual(sampler.sample([]), [])
    self.assertLen(fake_db.calls, 1)

    (sample,) = sampler.sample(["n0"])
    np.testing.assert_array_equal(
        sample.node_sets["nodes"].features["id"], [b"n0"]
    )
    self.assertLen(fake_db.calls, 2)
    _, params, param_types = fake_db.calls[1]
    self.assertEqual(params, {"seed_ids": [base64.b64encode(b"n0")]})
    self.assertEqual(
        param_types,
        {"seed_ids": spanner.param_types.Array(spanner.param_types.BYTES)},
    )

  def test_sample_string_key_uses_spanner_column_type(self):
    # FeatureSchema omits is_utf8_string=True, but Spanner column is STRING.
    fake_db = _FakeDatabase(
        metadata_rows=[[_metadata_dict("STRING")]],
        sample_rows=[[
            base64.b64encode(b"n0"),
            base64.b64encode(b"n0"),
            "node",
            "nodes",
            None,
            None,
            None,
        ]],
    )
    sampler = spanner_graph_sampler_lib.SpannerGraphSamplerGcp(
        database=fake_db,
        graph="g",
        plan=config_lib.SimpleSamplingConfig(
            seed_nodeset="nodes", num_hops=1, hop_width=2, multi_visit=False
        ),
        schema=_toy_schema(),
        debug_sampling=False,
        query_engine="SQL",
    )
    (sample,) = sampler.sample([b"n0"])
    np.testing.assert_array_equal(
        sample.node_sets["nodes"].features["id"], [b"n0"]
    )
    _, params, param_types = fake_db.calls[1]
    self.assertEqual(params["seed_ids"], ["n0"])
    # The Cloud client expects the BYTES parameters in base64.
    self.assertLen(base64.b64decode(params["sampling_seed"], validate=True), 8)
    self.assertEqual(
        param_types,
        {
            "seed_ids": spanner.param_types.Array(spanner.param_types.STRING),
            "sampling_seed": spanner.param_types.BYTES,
        },
    )

  def test_missing_graph_raises(self):
    fake_db = _FakeDatabase(metadata_rows=[], sample_rows=[])
    with self.assertRaisesRegex(ValueError, "Expected one property graph"):
      spanner_graph_sampler_lib.SpannerGraphSamplerGcp(
          database=fake_db,
          graph="missing",
          plan=config_lib.SimpleSamplingConfig(
              seed_nodeset="nodes", num_hops=0, multi_visit=False
          ),
          schema=_toy_schema(),
          debug_sampling=True,
          query_engine="SQL",
      )

  @parameterized.parameters(
      spanner_graph_sampler_lib.create_spanner_graph_sampler_gcp,
      spanner_graph_sampler_lib.create_graph_spanner_sampler,
  )
  def test_factories(self, factory):
    fake_db = _FakeDatabase(
        metadata_rows=[[_metadata_dict("BYTES")]],
        sample_rows=[[
            base64.b64encode(b"n0"),
            base64.b64encode(b"n0"),
            "node",
            "nodes",
            None,
            None,
            None,
        ]],
    )
    with mock.patch.object(
        spanner, "Client", return_value=_FakeClient(fake_db)
    ) as client:
      sampler = factory(
          project="p",
          instance="i",
          database="d",
          graph="g",
          plan=config_lib.SimpleSamplingConfig(
              seed_nodeset="nodes", num_hops=1, hop_width=2, multi_visit=False
          ),
          schema=_toy_schema(),
          debug_sampling=True,
      )
    client.assert_called_once_with(project="p")
    (sample,) = sampler.sample([b"n0"])
    np.testing.assert_array_equal(
        sample.node_sets["nodes"].features["id"], [b"n0"]
    )

  def test_deprecated_create_graph_spanner_sampler_warns(self):
    fake_db = _FakeDatabase(
        metadata_rows=[[_metadata_dict("BYTES")]], sample_rows=[]
    )
    with mock.patch.object(
        spanner, "Client", return_value=_FakeClient(fake_db)
    ):
      with self.assertWarnsRegex(
          DeprecationWarning, "use create_spanner_graph_sampler_gcp"
      ):
        spanner_graph_sampler_lib.create_graph_spanner_sampler(
            project="p",
            instance="i",
            database="d",
            graph="g",
            plan=config_lib.SimpleSamplingConfig(
                seed_nodeset="nodes", num_hops=1, multi_visit=False
            ),
            schema=_toy_schema(),
        )


if __name__ == "__main__":
  absltest.main()
