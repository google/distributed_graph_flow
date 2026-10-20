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
import json
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.io.gcp import bigquery_graph as bigquery_graph_lib
from dgf.src.io.gcp import bigquery_graph_metadata as bigquery_graph_metadata_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling.gcp import bigquery_graph_sampler
from dgf.src.util import test_util
import numpy as np

_PROJECT = "biggraphs-poc"


def _metadata(name: str) -> bigquery_graph_metadata_lib.BigQueryGraphMetadata:
  """Loads the BigQuery Graph metadata saved in `test_data`."""
  path = os.path.join(
      test_util.dgf_test_data_path(),
      f"bigquery_graph_sampler_metadata_{name}.json",
  )
  with open(path) as f:
    return bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
        json.load(f)
    )


def _create_sampler(
    metadata_name: str,
    seed_nodeset: str,
    num_hops: int = 1,
    hop_width: int = 5,
    reverse: bool = False,
    debug_sampling: bool = False,
    schema: schema_lib.GraphSchema | None = None,
) -> bigquery_graph_sampler.BigQueryGraphSampler:
  """Creates a sampler on a test graph without contacting BigQuery."""
  metadata = _metadata(metadata_name)
  if schema is None:
    schema = bigquery_graph_lib.metadata_to_schema(
        metadata, combine_as_json=False
    )
  return bigquery_graph_sampler.create_bigquery_graph_sampler(
      project=_PROJECT,
      dataset=metadata.property_graph_reference.dataset_id,
      graph=metadata.property_graph_reference.property_graph_id,
      plan=config_lib.SimpleSamplingConfig(
          seed_nodeset=seed_nodeset,
          num_hops=num_hops,
          hop_width=hop_width,
          reverse=reverse,
      ),
      schema=schema,
      metadata=metadata,
      debug_sampling=debug_sampling,
      check_optimizations=False,
  )


def _person_row(seed_id: bytes, node_id: bytes, **properties) -> dict:
  """Returns a node row of the `person` nodeset of the string graph."""
  values = {
      "id": node_id.decode(),
      "age": None,
      "score": None,
      "name": None,
      "active": None,
      "emb": [],
      "joined": None,
  } | properties
  return _node_row(seed_id, node_id, "person", values)


def _node_row(
    seed_id: bytes, node_id: bytes, nodeset: str, properties: dict
) -> dict:
  return {
      "seed_id": seed_id,
      "node_id": node_id,
      "element_type": "node",
      "element_class": nodeset,
      "source_id": None,
      "target_id": None,
      "properties_json": json.dumps(properties),
  }


def _edge_row(
    seed_id: bytes,
    edgeset: str,
    source_id: bytes,
    target_id: bytes,
    properties: dict | None = None,
) -> dict:
  return {
      "seed_id": seed_id,
      "node_id": None,
      "element_type": "edge",
      "element_class": edgeset,
      "source_id": source_id,
      "target_id": target_id,
      "properties_json": None if properties is None else json.dumps(properties),
  }


def _mock_query_results(mock_client_cls, rows: list[dict]) -> None:
  mock_client_cls.return_value.query.return_value.result.return_value = rows


class QueryTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="mag_two_hops_undirected",
          metadata_name="mag",
          seed_nodeset="paper",
          num_hops=2,
          reverse=True,
          debug_sampling=False,
          golden_file="bigquery_graph_sampler_test_query_mag.sql",
      ),
      dict(
          testcase_name="string_keys_debug_sampling",
          metadata_name="string",
          seed_nodeset="person",
          num_hops=1,
          reverse=False,
          debug_sampling=True,
          golden_file="bigquery_graph_sampler_test_query_string_debug.sql",
      ),
      dict(
          testcase_name="bytes_keys",
          metadata_name="bytes",
          seed_nodeset="node",
          num_hops=1,
          reverse=True,
          debug_sampling=False,
          golden_file="bigquery_graph_sampler_test_query_bytes.sql",
      ),
  )
  def test_query(
      self,
      metadata_name: str,
      seed_nodeset: str,
      num_hops: int,
      reverse: bool,
      debug_sampling: bool,
      golden_file: str,
  ):
    sampler = _create_sampler(
        metadata_name,
        seed_nodeset,
        num_hops=num_hops,
        hop_width=4,
        reverse=reverse,
        debug_sampling=debug_sampling,
    )
    test_util.assert_golden_string(self, sampler.query, golden_file, strip=True)


class SchemaResolutionTest(absltest.TestCase):

  def test_unsupported_plan_options_raise(self):
    metadata = _metadata("string")
    with self.assertRaisesRegex(NotImplementedError, "with replacement"):
      bigquery_graph_sampler.create_bigquery_graph_sampler(
          project=_PROJECT,
          dataset="dgf_io_tests",
          graph="dgf_sampler_string_graph",
          plan=config_lib.SimpleSamplingConfig(
              seed_nodeset="person", with_replacement=True
          ),
          schema=bigquery_graph_lib.metadata_to_schema(
              metadata, combine_as_json=False
          ),
          metadata=metadata,
          check_optimizations=False,
      )

  def test_composite_key_raises(self):
    with self.assertRaisesRegex(NotImplementedError, "composite key"):
      _create_sampler("composite", "node")

  def test_integer_key_raises(self):
    with self.assertRaisesRegex(
        ValueError, "Only STRING and BYTES primary keys are supported"
    ):
      _create_sampler("int", "node")

  def test_combine_as_json_schema_raises(self):
    schema = bigquery_graph_lib.metadata_to_schema(
        _metadata("string"), combine_as_json=True
    )
    with self.assertRaisesRegex(ValueError, "is not a property"):
      _create_sampler("string", "person", schema=schema)

  def test_feature_not_a_property_raises(self):
    schema = bigquery_graph_lib.metadata_to_schema(
        _metadata("string"), combine_as_json=False
    )
    schema.node_sets["person"].features["missing"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32
    )
    with self.assertRaisesRegex(ValueError, "'missing' .* is not a property"):
      _create_sampler("string", "person", schema=schema)

  def test_incompatible_feature_type_raises(self):
    schema = bigquery_graph_lib.metadata_to_schema(
        _metadata("string"), combine_as_json=False
    )
    schema.node_sets["person"].features["age"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES
    )
    with self.assertRaisesRegex(ValueError, "not compatible"):
      _create_sampler("string", "person", schema=schema)

  def test_missing_primary_key_raises(self):
    schema = bigquery_graph_lib.metadata_to_schema(
        _metadata("string"), combine_as_json=False
    )
    schema.node_sets["item"].features["id"].semantic = (
        schema_lib.FeatureSemantic.UNKNOWN
    )
    with self.assertRaisesRegex(ValueError, "No primary feature"):
      _create_sampler("string", "person", schema=schema)


class SampleTest(absltest.TestCase):

  @mock.patch("google.cloud.bigquery.Client")
  def test_sample_string_graph(self, mock_client_cls):
    _mock_query_results(
        mock_client_cls,
        [
            # The seed node is not the first row.
            _node_row(b"p0", b"i0", "item", {"id": "i0"}),
            _person_row(
                b"p0",
                b"p0",
                age=30,
                score=0.5,
                name="Zoë",
                active=True,
                emb=[1.0, 2.0],
                joined="2020-01-01T00:00:00Z",
            ),
            _person_row(b"p0", b"p1"),
            _edge_row(b"p0", "buys", b"p0", b"i0", {"weight": 1.0}),
            _edge_row(b"p0", "follows", b"p0", b"p1"),
        ],
    )
    sampler = _create_sampler("string", "person")
    (sample,) = sampler.sample(["p0"])

    _, query_kwargs = mock_client_cls.return_value.query.call_args
    parameter, sampling_seed = query_kwargs["job_config"].query_parameters
    self.assertEqual(sampling_seed.name, "sampling_seed")
    self.assertEqual(parameter.array_type, "STRING")
    self.assertEqual(parameter.values, ["p0"])

    person = sample.node_sets["person"]
    self.assertEqual(person.num_nodes, 2)
    np.testing.assert_array_equal(person.features["id"], [b"p0", b"p1"])
    np.testing.assert_array_equal(person.features["age"], [30, 0])
    np.testing.assert_array_equal(person.features["score"], [0.5, 0.0])
    np.testing.assert_array_equal(
        person.features["name"], ["Zoë".encode("utf-8"), b""]
    )
    np.testing.assert_array_equal(person.features["active"], [True, False])
    np.testing.assert_array_equal(
        person.features["joined"], [1577836800000000, 0]
    )
    np.testing.assert_array_equal(person.features["emb"][0], [1.0, 2.0])
    self.assertEmpty(person.features["emb"][1])
    np.testing.assert_array_equal(
        sample.node_sets["item"].features["id"], [b"i0"]
    )
    np.testing.assert_array_equal(
        sample.edge_sets["buys"].adjacency, [[0], [0]]
    )
    np.testing.assert_array_equal(
        sample.edge_sets["buys"].features["weight"], [1.0]
    )
    np.testing.assert_array_equal(
        sample.edge_sets["follows"].adjacency, [[0], [1]]
    )

  @mock.patch("google.cloud.bigquery.Client")
  def test_sample_bytes_graph(self, mock_client_cls):
    seed = b"\x00\xff"
    neighbor = b"\xc3\x28"

    def bytes_node_row(node_id: bytes, blob: bytes | None) -> dict:
      return _node_row(
          seed,
          node_id,
          "node",
          {
              "id": base64.b64encode(node_id).decode(),
              "blob": None if blob is None else base64.b64encode(blob).decode(),
          },
      )

    _mock_query_results(
        mock_client_cls,
        [
            bytes_node_row(neighbor, None),
            bytes_node_row(seed, b"\x01\x02"),
            _edge_row(seed, "edge", neighbor, seed),
        ],
    )
    sampler = _create_sampler("bytes", "node", reverse=True)
    (sample,) = sampler.sample([seed])

    _, query_kwargs = mock_client_cls.return_value.query.call_args
    parameter, sampling_seed = query_kwargs["job_config"].query_parameters
    self.assertEqual(sampling_seed.name, "sampling_seed")
    self.assertEqual(parameter.array_type, "BYTES")
    self.assertEqual(parameter.values, [seed])

    node = sample.node_sets["node"]
    np.testing.assert_array_equal(node.features["id"], [seed, neighbor])
    np.testing.assert_array_equal(node.features["blob"], [b"\x01\x02", b""])
    np.testing.assert_array_equal(sample.edge_sets["edge"].adjacency, [[1], [0]])

  @mock.patch("google.cloud.bigquery.Client")
  def test_sample_preserves_seed_order(self, mock_client_cls):
    _mock_query_results(
        mock_client_cls,
        [_person_row(b"p1", b"p1"), _person_row(b"p0", b"p0")],
    )
    sampler = _create_sampler("string", "person")
    samples = sampler.sample(["p0", "p1"])
    self.assertEqual(
        [sample.node_sets["person"].features["id"][0] for sample in samples],
        [b"p0", b"p1"],
    )

  @mock.patch("google.cloud.bigquery.Client")
  def test_missing_seed_raises(self, mock_client_cls):
    _mock_query_results(mock_client_cls, [_person_row(b"p0", b"p0")])
    sampler = _create_sampler("string", "person")
    with self.assertRaisesRegex(ValueError, r"Seed nodes \[b'unknown'\]"):
      sampler.sample(["p0", "unknown"])

  @mock.patch("google.cloud.bigquery.Client")
  def test_dangling_edge_raises(self, mock_client_cls):
    _mock_query_results(
        mock_client_cls,
        [
            _person_row(b"p0", b"p0"),
            _edge_row(b"p0", "follows", b"p0", b"missing"),
        ],
    )
    sampler = _create_sampler("string", "person")
    with self.assertRaisesRegex(ValueError, "references node b'missing'"):
      sampler.sample(["p0"])

  @mock.patch("google.cloud.bigquery.Client")
  def test_duplicate_node_raises(self, mock_client_cls):
    _mock_query_results(
        mock_client_cls,
        [_person_row(b"p0", b"p0"), _person_row(b"p0", b"p0")],
    )
    sampler = _create_sampler("string", "person")
    with self.assertRaisesRegex(ValueError, "Duplicate node"):
      sampler.sample(["p0"])


class TableClusteringTest(absltest.TestCase):

  @mock.patch("google.cloud.bigquery.Client")
  def test_warns_for_unclustered_tables(self, mock_client_cls):
    mock_client_cls.return_value.get_table.return_value.clustering_fields = None
    metadata = _metadata("string")
    with mock.patch.object(
        bigquery_graph_sampler.log, "warning", autospec=True
    ) as mock_warning:
      bigquery_graph_sampler.create_bigquery_graph_sampler(
          project=_PROJECT,
          dataset="dgf_io_tests",
          graph="dgf_sampler_string_graph",
          plan=config_lib.SimpleSamplingConfig(
              seed_nodeset="person", num_hops=1, hop_width=2
          ),
          schema=bigquery_graph_lib.metadata_to_schema(
              metadata, combine_as_json=False
          ),
          metadata=metadata,
      )
    messages = [call.args[0] for call in mock_warning.call_args_list]
    self.assertLen(messages, 4)  # person, item, buys and follows.
    self.assertIn(
        "CREATE OR REPLACE TABLE `biggraphs-poc.dgf_io_tests.dgf_sampler_buys`"
        " CLUSTER BY `source`, `target`",
        "\n".join(messages),
    )

  @mock.patch("google.cloud.bigquery.Client")
  def test_skips_bytes_keyed_tables(self, mock_client_cls):
    metadata = _metadata("bytes")
    bigquery_graph_sampler.create_bigquery_graph_sampler(
        project=_PROJECT,
        dataset="dgf_io_tests",
        graph="dgf_sampler_bytes_graph",
        plan=config_lib.SimpleSamplingConfig(
            seed_nodeset="node", num_hops=1, hop_width=2
        ),
        schema=bigquery_graph_lib.metadata_to_schema(
            metadata, combine_as_json=False
        ),
        metadata=metadata,
    )
    mock_client_cls.return_value.get_table.assert_not_called()


if __name__ == "__main__":
  absltest.main()
