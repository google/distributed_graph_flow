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

r"""Manual integration tests for BigQueryGraphSampler on real BigQuery Graphs.
"""

import copy
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.analyse import schema as schema_analyse_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import cache as cache_lib
from dgf.src.io import graph_in_memory as graph_in_memory_lib
from dgf.src.io.gcp import bigquery_graph as bigquery_graph_lib
from dgf.src.learning.ten_lines import node_prediction_train as node_prediction_train_lib
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.sampling.gcp import bigquery_graph_sampler as bigquery_graph_sampler_lib
from dgf.src.transform import canonicalize as canonicalize_lib
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np

_PROJECT = "biggraphs-poc"
_DATASET = "mag"
_GRAPH = "mag_graph"
_SEED_NODESET = "paper"
_GCS_EXPORT_PATH = "gs://gbm-test/mag_graph_export"
# Graphs created by `testdata/create_sampler_test_graphs.sql`.
_TEST_GRAPHS_DATASET = "dgf_io_tests"
_TEST_GRAPHS_GCS_EXPORT_PATH = "gs://gbm-test/dgf_sampler_test_graphs"


def _assert_samplers_are_equivalent(
    test: absltest.TestCase,
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    bq_sampler: bigquery_graph_sampler_lib.BigQueryGraphSampler,
    plan: sampling_config_lib.SimpleSamplingConfig,
    seed_idxs: list[int],
) -> None:
  """Checks that the BigQuery and in-memory samplers return the same graphs."""
  in_memory_sampler = in_memory_sampler_lib.create_sampler(
      graph, plan=plan, schema=schema, num_threads=1, debug_sampling=True
  )
  primary_key = schema_analyse_lib.primary_feature(
      plan.seed_nodeset, schema.node_sets[plan.seed_nodeset]
  )
  seed_ids = [
      graph.node_sets[plan.seed_nodeset].features[primary_key][idx]
      for idx in seed_idxs
  ]
  bq_samples = bq_sampler.sample(seed_ids)
  in_memory_samples = in_memory_sampler.sample(seed_idxs)

  test.assertLen(bq_samples, len(seed_ids))
  # The in-memory sampler does not return edge features.
  comparison_schema = copy.deepcopy(schema)
  for edgeset_schema in comparison_schema.edge_sets.values():
    edgeset_schema.features = {}
  for seed_id, bq_sample, in_memory_sample in zip(
      seed_ids, bq_samples, in_memory_samples
  ):
    in_memory_graph_validate_lib.validate_graph(
        bq_sample, schema, raise_on_warning=False
    )
    test.assertEqual(
        bq_sample.node_sets[plan.seed_nodeset].features[primary_key][0],
        seed_id,
    )
    test_util.assert_are_equal(
        test,
        canonicalize_lib.canonicalize_graph(bq_sample, comparison_schema),
        canonicalize_lib.canonicalize_graph(
            in_memory_sample, comparison_schema
        ),
    )


class BigQueryGraphSamplerIntegrationTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    def load_exported_graph():
      bigquery_graph_lib.export_bigquery_graph_to_disk(
          path=_GCS_EXPORT_PATH,
          project=_PROJECT,
          dataset=_DATASET,
          graph=_GRAPH,
          verbose=1,
      )
      return graph_in_memory_lib.read_graph(_GCS_EXPORT_PATH)

    cache_path = os.path.join(
        tempfile.gettempdir(), "dgf_bq_mag_integration_cache"
    )
    raw_graph, cls.schema = cache_lib.cache(
        cache_path, load_exported_graph
    )
    cls.full_graph = canonicalize_lib.canonicalize_graph(
        raw_graph, cls.schema
    )
    in_memory_graph_validate_lib.validate_graph(
        cls.full_graph, cls.schema, raise_on_warning=False
    )

  @parameterized.parameters(
      (1, 100),
      (2, 10),
  )
  def test_compare_with_exported_graph_in_memory_sampler(
      self, num_hops: int, hop_width: int
  ):
    plan = sampling_config_lib.SimpleSamplingConfig(
        seed_nodeset=_SEED_NODESET,
        num_hops=num_hops,
        hop_width=hop_width,
    )
    bq_sampler = bigquery_graph_sampler_lib.create_bigquery_graph_sampler(
        project=_PROJECT,
        dataset=_DATASET,
        graph=_GRAPH,
        plan=plan,
        schema=self.schema,
        debug_sampling=True,
    )
    _assert_samplers_are_equivalent(
        self, self.full_graph, self.schema, bq_sampler, plan, seed_idxs=[10, 11]
    )

  def test_exported_graph_model_predictions_match_bigquery_sampler(self):
    schema = copy.deepcopy(self.schema)
    schema.node_sets[_SEED_NODESET] = schema_lib.NodeSchema(
        features={
            "feat": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(128,),
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
    paper_features = dict(self.full_graph.node_sets[_SEED_NODESET].features)
    paper_features["feat"] = np.stack(
        list(paper_features["feat"])
    ).astype(np.float32)
    node_sets = dict(self.full_graph.node_sets)
    node_sets[_SEED_NODESET] = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=self.full_graph.node_sets[_SEED_NODESET].num_nodes,
        features=paper_features,
    )
    train_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets=node_sets,
        edge_sets=self.full_graph.edge_sets,
        timestamp=self.full_graph.timestamp,
    )

    model = node_prediction_train_lib.train_node_model(
        graph=train_graph,
        schema=schema,
        target_column="labels",
        target_nodeset=_SEED_NODESET,
        num_train_steps=5,
        num_valid_steps=2,
        batch_size=8,
        num_sampling_hops=1,
        sampling_width=100,
        evaluate_final_model=False,
        fast_compile=True,
        verbose=1,
    )

    seed_idxs = [10, 11]
    all_ids = train_graph.node_sets[_SEED_NODESET].features["id"]
    seed_ids = [all_ids[idx].decode("utf-8") for idx in seed_idxs]

    sampling_plan = model.data().sampling_plan
    assert sampling_plan is not None
    in_mem_sampler = in_memory_sampler_lib.create_sampler(
        train_graph,
        plan=sampling_plan,
        schema=model.data().schema,
        num_threads=1,
    )
    in_mem_samples = in_mem_sampler.sample(seed_idxs)
    in_mem_preds = model.predict_on_graph_sample_batch(in_mem_samples)

    bq_sampler = bigquery_graph_sampler_lib.create_bigquery_graph_sampler(
        project=_PROJECT,
        dataset=_DATASET,
        graph=_GRAPH,
        plan=sampling_plan,
        schema=model.data().schema,
        debug_sampling=True,
    )
    bq_samples = bq_sampler.sample(seed_ids)
    bq_preds = model.predict_on_graph_sample_batch(bq_samples)

    np.testing.assert_allclose(bq_preds, in_mem_preds, rtol=1e-5, atol=1e-5)


class CornerCaseGraphsTest(parameterized.TestCase):

  @parameterized.parameters(
      ("dgf_sampler_string_graph", "person", [0, 1, 2, 3]),
      ("dgf_sampler_string_graph", "item", [0, 1, 2]),
      ("dgf_sampler_bytes_graph", "node", [0, 1, 2]),
  )
  def test_compare_with_exported_graph_in_memory_sampler(
      self, graph_name: str, seed_nodeset: str, seed_idxs: list[int]
  ):
    export_path = os.path.join(_TEST_GRAPHS_GCS_EXPORT_PATH, graph_name)
    bigquery_graph_lib.export_bigquery_graph_to_disk(
        path=export_path,
        project=_PROJECT,
        dataset=_TEST_GRAPHS_DATASET,
        graph=graph_name,
        verbose=0,
    )
    graph, schema = graph_in_memory_lib.read_graph(export_path)
    plan = sampling_config_lib.SimpleSamplingConfig(
        seed_nodeset=seed_nodeset, num_hops=2, hop_width=10
    )
    bq_sampler = bigquery_graph_sampler_lib.create_bigquery_graph_sampler(
        project=_PROJECT,
        dataset=_TEST_GRAPHS_DATASET,
        graph=graph_name,
        plan=plan,
        schema=schema,
        debug_sampling=True,
    )
    _assert_samplers_are_equivalent(
        self, graph, schema, bq_sampler, plan, seed_idxs
    )

  @parameterized.parameters(
      ("dgf_sampler_composite_graph", NotImplementedError, "composite key"),
      ("dgf_sampler_int_graph", ValueError, "Only STRING and BYTES"),
  )
  def test_unsupported_graph_raises(
      self, graph_name: str, error: type[Exception], message: str
  ):
    with self.assertRaisesRegex(error, message):
      schema = bigquery_graph_lib.read_bigquery_graph_schema(
          project=_PROJECT, dataset=_TEST_GRAPHS_DATASET, graph=graph_name
      )
      bigquery_graph_sampler_lib.create_bigquery_graph_sampler(
          project=_PROJECT,
          dataset=_TEST_GRAPHS_DATASET,
          graph=graph_name,
          plan=sampling_config_lib.SimpleSamplingConfig(seed_nodeset="node"),
          schema=schema,
      )


if __name__ == "__main__":
  absltest.main()
