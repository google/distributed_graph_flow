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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dgf
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io.gcp import bigquery_graph_metadata as bigquery_graph_metadata_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling.gcp import bigquery_graph_sampler
from dgf.src.util import test_util
import numpy as np

_ARXIV_METADATA_DICT = {
    "creationTime": "2026-02-14T23:25:51.378016Z",
    "edgeTables": [{
        "dataSourceTable": {
            "datasetId": "arxiv",
            "projectId": "biggraphs-poc",
            "tableId": "edges",
        },
        "destinationNodeReference": {
            "edgeTableColumns": ["target_id"],
            "nodeTable": "nodes",
            "nodeTableColumns": ["id"],
        },
        "keyColumns": ["id", "target_id"],
        "labelAndProperties": [{
            "label": "edge",
            "properties": [],
        }],
        "name": "edges",
        "sourceNodeReference": {
            "edgeTableColumns": ["id"],
            "nodeTable": "nodes",
            "nodeTableColumns": ["id"],
        },
    }],
    "etag": "etag",
    "lastModifiedTime": "2026-02-14T23:25:51.378016Z",
    "nodeTables": [{
        "dataSourceTable": {
            "datasetId": "arxiv",
            "projectId": "biggraphs-poc",
            "tableId": "nodes",
        },
        "keyColumns": ["id"],
        "labelAndProperties": [{
            "label": "node",
            "properties": [
                {
                    "dataType": {
                        "arrayElementType": {"typeKind": "FLOAT64"},
                        "typeKind": "ARRAY",
                    },
                    "expression": "feat",
                    "name": "feat",
                },
                {
                    "dataType": {"typeKind": "STRING"},
                    "expression": "id",
                    "name": "id",
                },
                {
                    "dataType": {"typeKind": "INT64"},
                    "expression": "labels",
                    "name": "labels",
                },
                {
                    "dataType": {"typeKind": "INT64"},
                    "expression": "year",
                    "name": "year",
                },
            ],
        }],
        "name": "nodes",
    }],
    "propertyGraphReference": {
        "datasetId": "arxiv",
        "projectId": "biggraphs-poc",
        "propertyGraphId": "arvix_graph",
    },
}


def _arxiv_metadata() -> bigquery_graph_metadata_lib.BigQueryGraphMetadata:
  """Returns test BigQueryGraphMetadata for the arxiv graph."""
  return bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict(  # pyrefly: ignore[missing-attribute]
      _ARXIV_METADATA_DICT
  )


def _arxiv_schema(
    static_feat_dim: int | None = None,
) -> schema_lib.GraphSchema:
  """Returns test GraphSchema for the arxiv graph."""
  feat_shape = (static_feat_dim,) if static_feat_dim is not None else (None,)
  return schema_lib.GraphSchema(
      node_sets={
          "nodes": schema_lib.NodeSchema(
              features={
                  "feat": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_64,
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                      shape=feat_shape,
                  ),
                  "id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "labels": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                  ),
                  "year": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          )
      },
      edge_sets={
          "edges": schema_lib.EdgeSchema(
              source="nodes",
              target="nodes",
          )
      },
  )


class BigQueryGraphSamplerTest(parameterized.TestCase):

  def test_generate_cte_query_undirected(self):
    metadata = _arxiv_metadata()
    schema = _arxiv_schema()
    config = config_lib.SimpleSamplingConfig(
        seed_nodeset="nodes",
        num_hops=1,
        hop_width=3,
        reverse=True,
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(config, schema)
    generator = bigquery_graph_sampler.CteQueryGenerator(
        metadata, plan, schema, debug_sampling=False
    )
    query = generator.generate()

    test_util.assert_golden_string(
        self,
        query,
        "bigquery_graph_sampler_test_generate_cte_query_undirected.sql",
        strip=True,
    )

  def test_generate_cte_query_debug_sampling(self):
    metadata = _arxiv_metadata()
    schema = _arxiv_schema()
    config = config_lib.SimpleSamplingConfig(
        seed_nodeset="nodes",
        num_hops=1,
        hop_width=2,
        reverse=False,
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(config, schema)
    generator = bigquery_graph_sampler.CteQueryGenerator(
        metadata, plan, schema, debug_sampling=True
    )
    query = generator.generate()

    test_util.assert_golden_string(
        self,
        query,
        "bigquery_graph_sampler_test_generate_cte_query_debug_sampling.sql",
        strip=True,
    )

  def test_generate_cte_query_heterogeneous_mag(self):
    metadata = bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict({  # pyrefly: ignore[missing-attribute]
        "creationTime": "2026-02-14T23:25:51Z",
        "etag": "etag",
        "lastModifiedTime": "2026-02-14T23:25:51Z",
        "propertyGraphReference": {
            "projectId": "biggraphs-poc",
            "datasetId": "mag",
            "propertyGraphId": "mag_graph",
        },
        "nodeTables": [
            {
                "name": "paper",
                "keyColumns": ["id"],
                "dataSourceTable": {
                    "projectId": "biggraphs-poc",
                    "datasetId": "mag",
                    "tableId": "paper",
                },
                "labelAndProperties": [{
                    "label": "paper",
                    "properties": [
                        {
                            "name": "id",
                            "expression": "id",
                            "dataType": {"typeKind": "BYTES"},
                        },
                        {
                            "name": "labels",
                            "expression": "labels",
                            "dataType": {"typeKind": "INT64"},
                        },
                        {
                            "name": "year",
                            "expression": "year",
                            "dataType": {"typeKind": "INT64"},
                        },
                    ],
                }],
            },
            {
                "name": "author",
                "keyColumns": ["id"],
                "dataSourceTable": {
                    "projectId": "biggraphs-poc",
                    "datasetId": "mag",
                    "tableId": "author",
                },
                "labelAndProperties": [{
                    "label": "author",
                    "properties": [{
                        "name": "id",
                        "expression": "id",
                        "dataType": {"typeKind": "BYTES"},
                    }],
                }],
            },
        ],
        "edgeTables": [
            {
                "name": "cites",
                "keyColumns": ["#source", "#target"],
                "dataSourceTable": {
                    "projectId": "biggraphs-poc",
                    "datasetId": "mag",
                    "tableId": "cites",
                },
                "sourceNodeReference": {
                    "nodeTable": "paper",
                    "nodeTableColumns": ["id"],
                    "edgeTableColumns": ["#source"],
                },
                "destinationNodeReference": {
                    "nodeTable": "paper",
                    "nodeTableColumns": ["id"],
                    "edgeTableColumns": ["#target"],
                },
                "labelAndProperties": [{"label": "cites", "properties": []}],
            },
            {
                "name": "writes",
                "keyColumns": ["#source", "#target"],
                "dataSourceTable": {
                    "projectId": "biggraphs-poc",
                    "datasetId": "mag",
                    "tableId": "writes",
                },
                "sourceNodeReference": {
                    "nodeTable": "author",
                    "nodeTableColumns": ["id"],
                    "edgeTableColumns": ["#source"],
                },
                "destinationNodeReference": {
                    "nodeTable": "paper",
                    "nodeTableColumns": ["id"],
                    "edgeTableColumns": ["#target"],
                },
                "labelAndProperties": [{
                    "label": "writes",
                    "properties": [{
                        "name": "weight",
                        "expression": "weight",
                        "dataType": {"typeKind": "FLOAT64"},
                    }],
                }],
            },
        ],
    })
    schema = schema_lib.GraphSchema(
        node_sets={
            "paper": schema_lib.NodeSchema(
                features={
                    "id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    ),
                    "labels": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                    ),
                    "year": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                    ),
                }
            ),
            "author": schema_lib.NodeSchema(
                features={
                    "id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    )
                }
            ),
        },
        edge_sets={
            "cites": schema_lib.EdgeSchema(source="paper", target="paper"),
            "writes": schema_lib.EdgeSchema(
                source="author",
                target="paper",
                features={
                    "weight": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                    )
                },
            ),
        },
    )
    config = config_lib.SimpleSamplingConfig(
        seed_nodeset="paper",
        num_hops=2,
        hop_width=4,
        reverse=True,
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(config, schema)
    query = bigquery_graph_sampler.CteQueryGenerator(
        metadata, plan, schema, debug_sampling=False
    ).generate()

    test_util.assert_golden_string(
        self,
        query,
        "bigquery_graph_sampler_test_generate_cte_query_heterogeneous_mag.sql",
        strip=True,
    )

  def test_seed_node_placed_at_index_zero_and_preserves_seed_order(self):
    schema = _arxiv_schema(static_feat_dim=2)
    # Simulate out-of-order streaming from BigQuery UNION ALL:
    # Seed "11" arrives before seed "10", and for seed "10" neighbor "10090"
    # arrives before the seed node "10" itself.
    rows = [
        (
            "11",
            "11",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "11", "feat": [1.0, 2.0], "labels": 1,'
                ' "year": 2020}}'
            ),
        ),
        (
            "10",
            "10090",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "10090", "feat": [3.0, 4.0], "labels":'
                ' 2, "year": 2019}}'
            ),
        ),
        (
            "10",
            "10",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "10", "feat": [5.0, 6.0], "labels": 0,'
                ' "year": 2021}}'
            ),
        ),
        ("10", None, "edge", "edges", "10", "10090", None),
        # Duplicate edge from reverse hop should be deduplicated:
        ("10", None, "edge", "edges", "10", "10090", None),
    ]

    samples = bigquery_graph_sampler._cte_result_to_in_memory_graphs(
        rows,
        schema,
        seed_nodeset="nodes",
        seed_ids=["10", "11"],
    )

    self.assertLen(samples, 2)
    # First sample must correspond to "10", with "10" at index 0:
    np.testing.assert_array_equal(
        samples[0].node_sets["nodes"].features["id"],
        np.array([b"10", b"10090"]),
    )
    np.testing.assert_allclose(
        samples[0].node_sets["nodes"].features["feat"],
        np.array([[5.0, 6.0], [3.0, 4.0]]),
    )
    np.testing.assert_array_equal(
        samples[0].edge_sets["edges"].adjacency,
        np.array([[0], [1]], dtype=np.int64),
    )

    # Second sample must correspond to "11", with "11" at index 0:
    np.testing.assert_array_equal(
        samples[1].node_sets["nodes"].features["id"],
        np.array([b"11"]),
    )
    np.testing.assert_array_equal(
        samples[1].edge_sets["edges"].adjacency,
        np.zeros((2, 0), dtype=np.int64),
    )

  @mock.patch("google.cloud.bigquery.Client")
  def test_check_table_clustering_warnings(self, mock_client_cls):
    metadata = _arxiv_metadata()
    schema = _arxiv_schema()
    config = config_lib.SimpleSamplingConfig(
        seed_nodeset="nodes",
        num_hops=1,
        hop_width=2,
        reverse=False,
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(config, schema)

    mock_client = mock_client_cls.return_value
    unclustered_table = mock.Mock()
    unclustered_table.clustering_fields = None
    mock_client.get_table.return_value = unclustered_table

    with mock.patch.object(bigquery_graph_sampler.log, "warning") as mock_warn:
      bigquery_graph_sampler._check_table_clustering(
          mock_client, metadata, plan, schema
      )
      self.assertEqual(mock_warn.call_count, 2)
      self.assertIn(
          "CREATE OR REPLACE TABLE `biggraphs-poc.arxiv.nodes` CLUSTER BY id",
          mock_warn.call_args_list[0][0][0],
      )
      self.assertIn(
          "CREATE OR REPLACE TABLE `biggraphs-poc.arxiv.edges` CLUSTER BY id,"
          " target_id",
          mock_warn.call_args_list[1][0][0],
      )

  @mock.patch("google.cloud.bigquery.Client")
  def test_sample_executes_parameterized_query(self, mock_client_cls):
    metadata = _arxiv_metadata()
    schema = _arxiv_schema()
    config = config_lib.SimpleSamplingConfig(
        seed_nodeset="nodes",
        num_hops=1,
        hop_width=2,
    )

    mock_client = mock_client_cls.return_value
    clustered_table = mock.Mock()
    clustered_table.clustering_fields = ["id", "target_id"]
    mock_client.get_table.return_value = clustered_table

    mock_job = mock.Mock()
    mock_job.result.return_value = [
        (
            "10",
            "10",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "10", "feat": [1.0, 2.0], "labels": 3,'
                ' "year": 2022}}'
            ),
        )
    ]
    mock_client.query.return_value = mock_job

    sampler = bigquery_graph_sampler.create_bigquery_graph_sampler(
        project="biggraphs-poc",
        dataset="arxiv",
        graph="arvix_graph",
        plan=config,
        schema=schema,
        metadata=metadata,
    )
    samples = sampler.sample(["10"])

    self.assertLen(samples, 1)
    mock_client.query.assert_called_once()
    np.testing.assert_array_equal(
        samples[0].node_sets["nodes"].features["id"],
        np.array([b"10"]),
    )

  @mock.patch("google.cloud.bigquery.Client")
  def test_compatibility_with_in_memory_sampler_and_node_model_prediction(
      self, mock_client_cls
  ):
    schema = _arxiv_schema(static_feat_dim=4)
    full_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=4,
                features={
                    "id": np.array([b"0", b"1", b"2", b"3"]),
                    "feat": np.array(
                        [
                            [1.0, 0.0, 0.5, 0.2],
                            [0.0, 1.0, 0.3, 0.8],
                            [0.5, 0.5, 1.0, 0.0],
                            [0.2, 0.9, 0.1, 1.0],
                        ],
                        dtype=np.float64,
                    ),
                    "labels": np.array([0, 1, 0, 1], dtype=np.int64),
                    "year": np.array([2018, 2019, 2020, 2021], dtype=np.int64),
                },
            )
        },
        edge_sets={
            "edges": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array(
                    [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64
                ),
                features={},
            )
        },
    )

    model = dgf.learning.train_node_model(
        graph=full_graph,
        schema=schema,
        target_column="labels",
        num_train_steps=2,
        num_valid_steps=1,
        batch_size=2,
        num_sampling_hops=1,
        sampling_width=10,
        validation_ratio=0.0,
        evaluate_final_model=False,
        fast_compile=True,
        verbose=0,
    )

    sampling_plan = model.data().sampling_plan
    assert sampling_plan is not None
    in_mem_sampler = dgf.sampling.create_sampler(
        full_graph,
        plan=sampling_plan,
        schema=model.data().schema,
        num_threads=1,
    )
    in_mem_samples = in_mem_sampler.sample([0, 1])

    # Construct the exact rows BigQuery CTE query returns for seeds "0" and "1"
    # (1-hop undirected: seed "0" connects to "1" and "3"; seed "1" connects to
    # "2" and "0").
    mock_rows = [
        # Seed "0": put neighbor "3" first to test seed-at-index-0 guarantee.
        (
            "0",
            "3",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "3", "feat": [0.2, 0.9, 0.1, 1.0],'
                ' "labels": 1, "year": 2021}}'
            ),
        ),
        (
            "0",
            "0",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "0", "feat": [1.0, 0.0, 0.5, 0.2],'
                ' "labels": 0, "year": 2018}}'
            ),
        ),
        (
            "0",
            "1",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "1", "feat": [0.0, 1.0, 0.3, 0.8],'
                ' "labels": 1, "year": 2019}}'
            ),
        ),
        ("0", None, "edge", "edges", "0", "1", None),
        ("0", None, "edge", "edges", "3", "0", None),
        # Seed "1":
        (
            "1",
            "1",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "1", "feat": [0.0, 1.0, 0.3, 0.8],'
                ' "labels": 1, "year": 2019}}'
            ),
        ),
        (
            "1",
            "2",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "2", "feat": [0.5, 0.5, 1.0, 0.0],'
                ' "labels": 0, "year": 2020}}'
            ),
        ),
        (
            "1",
            "0",
            "node",
            "nodes",
            None,
            None,
            (
                '{"properties": {"id": "0", "feat": [1.0, 0.0, 0.5, 0.2],'
                ' "labels": 0, "year": 2018}}'
            ),
        ),
        ("1", None, "edge", "edges", "1", "2", None),
        ("1", None, "edge", "edges", "0", "1", None),
    ]

    mock_client = mock_client_cls.return_value
    mock_job = mock.Mock()
    mock_job.result.return_value = mock_rows
    mock_client.query.return_value = mock_job

    bq_sampler = bigquery_graph_sampler.create_bigquery_graph_sampler(
        project="biggraphs-poc",
        dataset="arxiv",
        graph="arvix_graph",
        plan=sampling_plan,
        schema=model.data().schema,
        metadata=_arxiv_metadata(),
        check_optimizations=False,
    )
    bq_samples = bq_sampler.sample(["0", "1"])

    in_mem_preds = model.predict_on_graph_sample_batch(in_mem_samples)
    bq_preds = model.predict_on_graph_sample_batch(bq_samples)

    np.testing.assert_allclose(bq_preds, in_mem_preds, rtol=1e-5, atol=1e-5)
    test_util.assert_are_equal(self, len(bq_samples), len(in_mem_samples))

  def test_bytes_key_and_hash_prefixed_columns(self):
    metadata = bigquery_graph_metadata_lib.BigQueryGraphMetadata.from_dict({  # pyrefly: ignore[missing-attribute]
        "creationTime": "2026-02-14T23:25:51Z",
        "etag": "etag",
        "lastModifiedTime": "2026-02-14T23:25:51Z",
        "propertyGraphReference": {
            "projectId": "biggraphs-poc",
            "datasetId": "arxiv",
            "propertyGraphId": "arvix_graph",
        },
        "nodeTables": [{
            "name": "nodes",
            "keyColumns": ["id"],
            "dataSourceTable": {
                "projectId": "biggraphs-poc",
                "datasetId": "arxiv",
                "tableId": "nodes",
            },
            "labelAndProperties": [{
                "label": "nodes",
                "properties": [{
                    "name": "id",
                    "expression": "id",
                    "dataType": {"typeKind": "BYTES"},
                }],
            }],
        }],
        "edgeTables": [{
            "name": "edges",
            "keyColumns": ["#source", "#target"],
            "dataSourceTable": {
                "projectId": "biggraphs-poc",
                "datasetId": "arxiv",
                "tableId": "edges",
            },
            "sourceNodeReference": {
                "nodeTable": "nodes",
                "nodeTableColumns": ["id"],
                "edgeTableColumns": ["#source"],
            },
            "destinationNodeReference": {
                "nodeTable": "nodes",
                "nodeTableColumns": ["id"],
                "edgeTableColumns": ["#target"],
            },
            "labelAndProperties": [{"label": "edges", "properties": []}],
        }],
    })
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    )
                }
            )
        },
        edge_sets={
            "edges": schema_lib.EdgeSchema(source="nodes", target="nodes")
        },
    )
    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="nodes",
        num_hops=1,
        hop_width=5,
        reverse=False,
    )
    sampler = bigquery_graph_sampler.create_bigquery_graph_sampler(
        project="biggraphs-poc",
        dataset="arxiv",
        graph="arvix_graph",
        plan=plan,
        schema=schema,
        metadata=metadata,
        check_optimizations=False,
    )
    test_util.assert_golden_string(
        self,
        sampler.query,
        "bigquery_graph_sampler_test_generate_cte_query_bytes_key.sql",
        strip=True,
    )


if __name__ == "__main__":
  absltest.main()
