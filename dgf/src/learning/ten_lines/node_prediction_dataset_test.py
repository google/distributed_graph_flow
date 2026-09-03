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

"""Test converting between heterogeneous graph edge types."""

import os
import unittest.mock
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import jax as jax_io_lib
from dgf.src.io import tf_graph_sample
from dgf.src.learning.ten_lines import node_prediction_dataset
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import gen_test_graph
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np


class GNNDatasetPreparatorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("CacheHost", True, "host"),
      ("CacheDevice", True, "device"),
      ("NoCache", False, "host"),
  )
  def test_in_memory_graph(self, cache_features, cache_device):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    schema = gen_test_graph.generate_schema(True, False, True, False)
    print("schema:\n", schema)
    sampling_config = sampling_config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=2,
        hop_width=3,
        reverse=True,
    )
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, schema
    )
    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_plan=sampling_plan,
        batch_size=2,
        drop_remainder=True,
        shuffle=True,
        cache_normalized_features=cache_features,
        cache_normalized_features_device=cache_device,
    )
    self.assertFalse(preparator.is_prepared())
    preparator.prepare()
    self.assertTrue(preparator.is_prepared())

    num_batches = 0
    num_graphs = 0
    normalized_schema = preparator.get_live().normalizer.output_schema()

    if cache_device == "host":
      generator = preparator.generate()
    else:

      def sanitize(g):
        ng = jax_io_lib.jax_graph_to_graph(g)
        for node_set in ng.node_sets.values():
          for k, v in node_set.features.items():
            if v.dtype == np.int32:
              node_set.features[k] = v.astype(np.int64)
        return ng

      generator = (
          (sanitize(g), offsets) for g, offsets in preparator.generate_jax()
      )

    for graph_sample, merge_offset in generator:
      in_memory_graph_validate_lib.validate_graph(
          graph_sample, normalized_schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(merge_offset["n1"]) - 1
    self.assertEqual(num_batches, 1)
    self.assertEqual(num_graphs, 2)

  def test_in_memory_temporal_graph(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(False)
    sampling_config = sampling_config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=2,
        hop_width=3,
        reverse=True,
        temporal_sampling=True,
    )
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, schema
    )
    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_plan=sampling_plan,
        batch_size=2,
        drop_remainder=True,
        shuffle=True,
        temporal_sampling=True,
        nodeset_timestamp_features={"n1": "timestamp"},
        edgeset_timestamp_features={"e1": "timestamp"},
        cache_normalized_features_device="host",
    )
    self.assertFalse(preparator.is_prepared())
    preparator.prepare()
    self.assertTrue(preparator.is_prepared())

    num_batches = 0
    num_graphs = 0
    normalized_schema = preparator.get_live().normalizer.output_schema()
    for graph_sample, merge_offset in preparator.generate():
      in_memory_graph_validate_lib.validate_graph(
          graph_sample, normalized_schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(merge_offset["n1"]) - 1
    self.assertEqual(num_batches, 2)
    self.assertEqual(num_graphs, 4)

  def test_tf_gnn_samples(self):
    tmpdir = self.create_tempdir().full_path
    path = os.path.join(tmpdir, "samples@5.tfrecord")
    schema = gen_test_graph.generate_schema(
        variable_length=False, semantic=True
    )
    subgraph = gen_test_graph.generate_in_memory_graph(variable_length=False)

    def in_mem_graphs():
      for _ in range(21):
        yield subgraph

    tf_graph_sample.write_tfgnn_graphs(
        in_mem_graphs(),
        path,
        schema=schema,
        container_type="TF_RECORD",
    )

    sampling_config = sampling_config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=2,
        hop_width=3,
        reverse=True,
    )
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, schema
    )
    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=path,
        schema=schema,
        sampling_plan=sampling_plan,
        batch_size=2,
        drop_remainder=True,
        shuffle=True,
    )
    self.assertFalse(preparator.is_prepared())
    preparator.prepare()
    self.assertTrue(preparator.is_prepared())

    num_batches = 0
    num_graphs = 0
    normalized_schema = preparator.get_live().normalizer.output_schema()
    for graph_sample, merge_offset in preparator.generate():
      in_memory_graph_validate_lib.validate_graph(
          graph_sample, normalized_schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(merge_offset["n1"]) - 1
    self.assertEqual(num_batches, 10)
    self.assertEqual(num_graphs, 20)

  def test_in_memory_temporal_graph_with_timestamp_normalization(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(False)
    sampling_config = sampling_config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=2,
        hop_width=3,
        reverse=True,
        temporal_sampling=True,
    )
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, schema
    )

    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_plan=sampling_plan,
        batch_size=2,
        drop_remainder=True,
        shuffle=True,
        temporal_sampling=True,
        nodeset_timestamp_features={"n1": "timestamp"},
        edgeset_timestamp_features={"e1": "timestamp"},
        auto_normalize_config=normalize_lib.AutoNormalizeConfig(
            timestamp_normalize=True,
        ),
    )
    self.assertTrue(preparator.cache_normalized_features)
    preparator.prepare()
    self.assertFalse(preparator.cache_normalized_features)
    self.assertTrue(preparator.is_prepared())

    num_batches = 0
    num_graphs = 0
    normalized_schema = preparator.get_live().normalizer.output_schema()
    for graph_sample, merge_offset in preparator.generate():
      in_memory_graph_validate_lib.validate_graph(
          graph_sample, normalized_schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(merge_offset["n1"]) - 1
      # Verify that timestamp feature has been normalized
      self.assertIn(
          "timestamp_seed_delta_SINUSOID", graph_sample.node_sets["n1"].features
      )
    self.assertEqual(num_batches, 2)
    self.assertEqual(num_graphs, 4)

    # Test generate_jax as well
    jax_batches = 0
    for jax_sample, _ in preparator.generate_jax():
      jax_batches += 1
      self.assertIn(
          "timestamp_seed_delta_SINUSOID",
          jax_sample.node_sets["n1"].features,
      )
    self.assertEqual(jax_batches, 2)

  def test_get_target_nodeset_and_timestamp_feature(self):
    live = unittest.mock.MagicMock()
    live.normalizer.accepted_kwargs = set()
    self.assertEqual(
        node_prediction_dataset._get_target_nodeset_and_timestamp_feature(
            live, schema_lib.GraphSchema(node_sets={}, edge_sets={})
        ),
        (None, None),
    )

    live.normalizer.accepted_kwargs = {"seed_timestamps"}
    live.sampling_plan.root.nodeset = "n"
    schema = schema_lib.GraphSchema(
        node_sets={
            "n": schema_lib.NodeSchema(
                features={
                    "t": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_creation_time=True,
                    ),
                }
            )
        },
        edge_sets={},
    )
    self.assertEqual(
        node_prediction_dataset._get_target_nodeset_and_timestamp_feature(
            live, schema
        ),
        ("n", "t"),
    )

    schema_without_ts = schema_lib.GraphSchema(
        node_sets={"n": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    with self.assertRaises(ValueError):
      node_prediction_dataset._get_target_nodeset_and_timestamp_feature(
          live, schema_without_ts
      )

  def test_normalize_batch_sample(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "t": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_creation_time=True,
                    ),
                }
            ),
        },
        edge_sets={},
    )
    sample = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                features={"t": np.array([10, 20], dtype=np.int64)},
                num_nodes=2,
            ),
        },
        edge_sets={},
    )
    merge_offsets = {"n1": np.array([0, 1, 2], dtype=np.int32)}

    class MockNormalizer:

      def __init__(self, accepted_kwargs):
        self.accepted_kwargs = accepted_kwargs
        self.received_kwargs = None

      def normalize_numpy(self, sample, **kwargs):
        self.received_kwargs = kwargs
        return sample

    class MockSamplingPlan:

      class root:
        nodeset = "n1"

    # 1. Normalizer accepts seed_timestamps
    normalizer_with_ts = MockNormalizer(accepted_kwargs={"seed_timestamps"})
    mock_live_with_ts = node_prediction_dataset.LiveData(
        feature_stats=None,  # pyrefly: ignore[bad-argument-type]
        normalizer=normalizer_with_ts,  # pyrefly: ignore[bad-argument-type]
        padding=None,  # pyrefly: ignore[bad-argument-type]
        sampling_plan=MockSamplingPlan(),  # pyrefly: ignore[bad-argument-type]
        num_nodes_in_seed_nodeset=2,
        sample_generator=None,  # pyrefly: ignore[bad-argument-type]
    )
    out_sample = node_prediction_dataset._normalize_batch_sample(
        sample=sample,
        merge_offsets=merge_offsets,
        live=mock_live_with_ts,
        schema=schema,
    )
    self.assertIs(out_sample, sample)
    self.assertIsNotNone(normalizer_with_ts.received_kwargs)
    assert normalizer_with_ts.received_kwargs is not None
    self.assertIn("seed_timestamps", normalizer_with_ts.received_kwargs)
    np.testing.assert_array_equal(
        normalizer_with_ts.received_kwargs["seed_timestamps"]["n1"],
        np.array([10, 20]),
    )

    # 2. Normalizer does not accept seed_timestamps
    normalizer_without_ts = MockNormalizer(accepted_kwargs=set())
    mock_live_without_ts = node_prediction_dataset.LiveData(
        feature_stats=None,  # pyrefly: ignore[bad-argument-type]
        normalizer=normalizer_without_ts,  # pyrefly: ignore[bad-argument-type]
        padding=None,  # pyrefly: ignore[bad-argument-type]
        sampling_plan=MockSamplingPlan(),  # pyrefly: ignore[bad-argument-type]
        num_nodes_in_seed_nodeset=2,
        sample_generator=None,  # pyrefly: ignore[bad-argument-type]
    )
    out_sample2 = node_prediction_dataset._normalize_batch_sample(
        sample=sample,
        merge_offsets=merge_offsets,
        live=mock_live_without_ts,
        schema=schema,
    )
    self.assertIs(out_sample2, sample)
    self.assertIsNotNone(normalizer_without_ts.received_kwargs)
    self.assertEqual(normalizer_without_ts.received_kwargs, {})

    # 3. Normalizer accepts seed_timestamps but target nodeset has no creation time
    schema_without_ts = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    with self.assertRaisesRegex(
        ValueError,
        "The target nodeset 'n1' must have a creation time feature",
    ):
      node_prediction_dataset._normalize_batch_sample(
          sample=sample,
          merge_offsets=merge_offsets,
          live=mock_live_with_ts,
          schema=schema_without_ts,
      )


if __name__ == "__main__":
  absltest.main()
