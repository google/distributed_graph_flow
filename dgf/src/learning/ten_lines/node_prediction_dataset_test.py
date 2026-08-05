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

"""Tests for node prediction dataset preparation."""

import os
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import jax as jax_io_lib
from dgf.src.io import tf_graph_sample
from dgf.src.learning.ten_lines import node_prediction_dataset
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import gen_test_graph
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import jax
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
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=2,
            hop_width=3,
            reverse=True,
            temporal_sampling=True,
        ),
        schema,
    )
    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_plan=sampling_plan,
        batch_size=2,
        drop_remainder=True,
        shuffle=True,
        temporal_sampling=True,
        cache_normalized_features_device="host",
    )
    self.assertFalse(preparator.is_prepared())
    preparator.prepare()
    batches = list(preparator.generate())
    self.assertLen(batches, 2)
    self.assertEqual(sum(len(offset["n1"]) - 1 for _, offset in batches), 4)

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

  def _create_preparator(
      self,
      graph: in_memory_graph.InMemoryGraph,
      schema: schema_lib.GraphSchema,
      seed_nodeset: str,
  ) -> node_prediction_dataset.GNNDatasetPreparator:
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset=seed_nodeset, num_hops=1, hop_width=2, reverse=True
        ),
        schema,
    )
    return node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_plan=sampling_plan,
        batch_size=2,
        shuffle=True,
        drop_remainder=True,
    )

  def test_compute_train_and_valid_node_idxs_temporal(self):
    graph, schema = (
        gen_test_graph.generate_temporal_timeseries_in_memory_graph()
    )
    train_idx, valid_idx = (
        node_prediction_dataset.compute_train_and_valid_node_idxs(
            graph=graph,
            valid_graph=None,
            graph_format="IN_MEMORY_GRAPH",
            target_nodeset="alerts",
            random_seed=42,
            validation_ratio=0.33,
            train_seed_nodes=None,
            valid_seed_nodes=None,
            max_num_valid_examples=None,
            schema=schema,
            batch_size=2,
        )
    )
    self.assertIsNotNone(train_idx)
    self.assertIsNotNone(valid_idx)
    np.testing.assert_array_equal(train_idx, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(valid_idx, np.array([4, 5]))

  def test_compute_train_and_valid_node_idxs_temporal_unsorted(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "alerts": schema_lib.NodeSchema(
                features={
                    "creation_time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_creation_time=True,
                    ),
                }
            ),
        },
        edge_sets={},
    )
    # Timestamps out of order: index 1 is first (100) and index 3 is last (600)
    alerts_nodes = in_memory_graph.InMemoryNodeSet(
        num_nodes=6,
        features={
            "creation_time": np.array(
                [500, 100, 300, 600, 200, 400], dtype=np.int64
            ),
        },
    )
    graph = in_memory_graph.InMemoryGraph(
        node_sets={"alerts": alerts_nodes}, edge_sets={}
    )
    train_idx, valid_idx = (
        node_prediction_dataset.compute_train_and_valid_node_idxs(
            graph=graph,
            valid_graph=None,
            graph_format="IN_MEMORY_GRAPH",
            target_nodeset="alerts",
            random_seed=42,
            validation_ratio=0.33,
            train_seed_nodes=None,
            valid_seed_nodes=None,
            max_num_valid_examples=None,
            schema=schema,
            batch_size=2,
        )
    )
    self.assertIsNotNone(train_idx)
    self.assertIsNotNone(valid_idx)
    # Chronological sorted order of timestamps:
    # 1 (100), 4 (200), 2 (300), 5 (400) -> train
    # 0 (500), 3 (600) -> valid
    np.testing.assert_array_equal(train_idx, np.array([1, 4, 2, 5]))
    np.testing.assert_array_equal(valid_idx, np.array([0, 3]))

  def test_compute_node_idxs_invalid_target_nodeset_raises(self):
    graph, schema = (
        gen_test_graph.generate_temporal_timeseries_in_memory_graph()
    )
    with self.assertRaisesRegex(
        ValueError, "Target node set 'unknown_nodes' not found in schema."
    ):
      node_prediction_dataset.compute_train_and_valid_node_idxs(
          graph=graph,
          valid_graph=None,
          graph_format="IN_MEMORY_GRAPH",
          target_nodeset="unknown_nodes",
          random_seed=42,
          validation_ratio=0.33,
          train_seed_nodes=None,
          valid_seed_nodes=None,
          max_num_valid_examples=None,
          schema=schema,
          batch_size=2,
      )

  def test_compute_train_and_valid_node_idxs_missing_timestamp_falls_back(self):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    schema = gen_test_graph.generate_schema(True, False, True, False)
    train_idx, valid_idx = (
        node_prediction_dataset.compute_train_and_valid_node_idxs(
            graph=graph,
            valid_graph=None,
            graph_format="IN_MEMORY_GRAPH",
            target_nodeset="n1",
            random_seed=42,
            validation_ratio=0.33,
            train_seed_nodes=None,
            valid_seed_nodes=None,
            max_num_valid_examples=None,
            schema=schema,
            batch_size=1,
        )
    )
    self.assertIsNotNone(train_idx)
    self.assertIsNotNone(valid_idx)

  def test_prepare_datasets_temporal_auto_detection(self):
    graph, schema = (
        gen_test_graph.generate_temporal_timeseries_in_memory_graph()
    )
    train_dataset, valid_dataset = node_prediction_dataset.prepare_datasets(
        graph=graph,
        valid_graph=None,
        schema=schema,
        target_nodeset="alerts",
        random_seed=42,
        batch_size=2,
        num_sampling_hops=1,
        sampling_width=2,
        verbose=0,
        graph_format="IN_MEMORY_GRAPH",
        validation_ratio=0.33,
    )
    self.assertIsNotNone(valid_dataset)
    assert valid_dataset is not None
    train_samples = [sample for sample, _ in train_dataset.generate()]
    self.assertLen(train_samples, 2)
    for sample in train_samples:
      self.assertIn("time_mask", sample.node_sets["hardware"].features)
      self.assertIn(
          "time_seed_delta_SINUSOID", sample.node_sets["hardware"].features
      )
      self.assertIn(
          "creation_time_seed_delta_SINUSOID",
          sample.node_sets["alerts"].features,
      )
      self.assertEqual(
          sample.node_sets["hardware"]
          .features["signal_SOFT_QUANTILE"]
          .shape[1],
          30,
      )

    valid_samples = [sample for sample, _ in valid_dataset.generate()]
    self.assertLen(valid_samples, 1)
    for sample in valid_samples:
      self.assertIn(
          "creation_time_seed_delta_SINUSOID",
          sample.node_sets["alerts"].features,
      )

  def test_prepare_datasets_temporal_jax_generation(self):
    graph, schema = (
        gen_test_graph.generate_temporal_timeseries_in_memory_graph()
    )
    train_dataset, _ = node_prediction_dataset.prepare_datasets(
        graph=graph,
        valid_graph=None,
        schema=schema,
        target_nodeset="alerts",
        random_seed=42,
        batch_size=2,
        num_sampling_hops=1,
        sampling_width=2,
        verbose=0,
        graph_format="IN_MEMORY_GRAPH",
        validation_ratio=0.33,
    )
    batches = list(train_dataset.generate_jax())
    self.assertLen(batches, 2)
    for jax_sample, jax_offsets in batches:
      self.assertIsInstance(jax_sample, jax_in_memory_graph.JaxInMemoryGraph)
      self.assertIn("time_mask", jax_sample.node_sets["hardware"].features)
      self.assertIn(
          "time_seed_delta_SINUSOID",
          jax_sample.node_sets["hardware"].features,
      )
      self.assertIsInstance(
          jax_sample.node_sets["alerts"].features[
              "creation_time_seed_delta_SINUSOID"
          ],
          (jax.Array, np.ndarray),
      )
      self.assertIn("alerts", jax_offsets)
      self.assertIsInstance(jax_offsets["alerts"], (jax.Array, np.ndarray))

  def test_edge_timestamp_propagation_during_prepare_datasets(self):
    graph, schema = (
        gen_test_graph.generate_temporal_timeseries_in_memory_graph()
    )
    self.assertEmpty(schema.edge_sets["alert_to_hw"].features)
    self.assertEmpty(graph.edge_sets["alert_to_hw"].features)

    train_dataset, _ = node_prediction_dataset.prepare_datasets(
        graph=graph,
        valid_graph=None,
        schema=schema,
        target_nodeset="alerts",
        random_seed=42,
        batch_size=2,
        num_sampling_hops=1,
        sampling_width=2,
        verbose=0,
        graph_format="IN_MEMORY_GRAPH",
        validation_ratio=0.33,
    )
    self.assertIn(
        "timestamps",
        train_dataset.schema.edge_sets["alert_to_hw"].features,
    )

  def test_prepare_small_num_samples_for_stats(self):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    schema = gen_test_graph.generate_schema(True, False, True, False)
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config_lib.SimpleSamplingConfig(seed_nodeset="n1"), schema
    )
    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_plan=sampling_plan,
        batch_size=2,
        drop_remainder=True,
        shuffle=True,
        num_samples_for_stats=1,  # Smaller than batch_size
    )
    preparator.prepare()

  def test_get_transform_configs_auto_detection(self):
    graph, schema = (
        gen_test_graph.generate_temporal_timeseries_in_memory_graph()
    )
    transforms = self._create_preparator(
        graph, schema, "alerts"
    )._get_per_sample_transforms()
    self.assertIsNotNone(transforms)
    self.assertIsNotNone(transforms.timeseries_pad_and_cap)
    self.assertIsNotNone(transforms.timedelta_extraction)

  def test_get_per_sample_transforms_non_temporal_returns_none(self):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    schema = gen_test_graph.generate_schema(True, False, True, False)
    transforms = self._create_preparator(
        graph, schema, "n1"
    )._get_per_sample_transforms()
    self.assertIsNone(transforms)

  def test_prepare_datasets_does_not_mutate_caller_auto_normalize_config(self):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    schema = gen_test_graph.generate_schema(True, False, True, False)
    caller_config = normalize_lib.AutoNormalizeConfig(
        keep_raw_features=set(["f2"])
    )
    node_prediction_dataset.prepare_datasets(
        graph=graph,
        valid_graph=None,
        schema=schema,
        target_nodeset="n1",
        random_seed=42,
        batch_size=1,
        num_sampling_hops=1,
        sampling_width=2,
        verbose=0,
        graph_format="IN_MEMORY_GRAPH",
        validation_ratio=0.5,
        keep_raw_features=set(["extra_feature"]),
        auto_normalize_config=caller_config,
    )
    self.assertEqual(caller_config.keep_raw_features, set(["f2"]))

  def test_compute_train_and_valid_node_idxs_numpy_seed_nodes(self):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    train_idx, valid_idx = (
        node_prediction_dataset.compute_train_and_valid_node_idxs(
            graph=graph,
            valid_graph=None,
            graph_format="IN_MEMORY_GRAPH",
            target_nodeset="n1",
            random_seed=42,
            validation_ratio=0.5,
            train_seed_nodes=np.array([0, 1]),
            valid_seed_nodes=np.array([2, 3]),
            max_num_valid_examples=None,
        )
    )
    np.testing.assert_array_equal(train_idx, np.array([0, 1]))
    np.testing.assert_array_equal(valid_idx, np.array([2, 3]))


if __name__ == "__main__":
  absltest.main()
