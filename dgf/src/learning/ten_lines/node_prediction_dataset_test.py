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

  def test_prepare_target_no_creation_time_other_nodesets_have_timestamps(self):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    schema = gen_test_graph.generate_schema(True, False, True, False)
    # n1 (target) has no timestamp; add a timestamp feature to n2.
    schema.node_sets["n2"].features["ts"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
    )
    num_nodes = graph.node_sets["n2"].num_nodes
    assert num_nodes is not None
    graph.node_sets["n2"].features["ts"] = np.zeros(num_nodes, dtype=np.int64)
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1", num_hops=1, hop_width=1
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
        auto_normalize_config=normalize_lib.AutoNormalizeConfig(
            timestamp_normalize=True
        ),
    )
    with self.assertLogs(level="WARNING") as cm:
      preparator.prepare()

    self.assertFalse(preparator.auto_normalize_config.has_seed_timestamps)
    self.assertTrue(preparator.auto_normalize_config.timestamp_normalize)
    self.assertNotIn(
        "seed_timestamps",
        preparator.get_live().normalizer.accepted_kwargs,
    )
    self.assertTrue(
        any("cannot be autonormalized" in text for text in cm.output),
        f"Expected warning not found in: {cm.output}",
    )

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

    # 3. Normalizer accepts seed_timestamps but target nodeset has no creation 
    # time
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

  def test_compute_train_and_valid_node_idxs(self):
    graph, _ = gen_test_graph.generate_temporal_in_memory_graph(False)
    # 1. Temporal split
    train_idx, valid_idx = (
        node_prediction_dataset.compute_train_and_valid_node_idxs(
            graph=graph,
            valid_graph=None,
            graph_format="IN_MEMORY_GRAPH",
            target_nodeset="n1",
            random_seed=42,
            validation_ratio=0.5,
            train_seed_nodes=None,
            valid_seed_nodes=None,
            max_num_valid_examples=None,
            temporal_split=True,
            ts_feature="timestamp",
        )
    )
    self.assertIsNotNone(train_idx)
    self.assertIsNotNone(valid_idx)
    np.testing.assert_array_equal(train_idx, np.array([0, 1]))
    np.testing.assert_array_equal(valid_idx, np.array([2, 3]))

    # 2. Random split (temporal_split=False)
    train_idx, valid_idx = (
        node_prediction_dataset.compute_train_and_valid_node_idxs(
            graph=graph,
            valid_graph=None,
            graph_format="IN_MEMORY_GRAPH",
            target_nodeset="n1",
            random_seed=42,
            validation_ratio=0.5,
            train_seed_nodes=None,
            valid_seed_nodes=None,
            max_num_valid_examples=None,
            temporal_split=False,
        )
    )
    self.assertIsNotNone(train_idx)
    self.assertIsNotNone(valid_idx)
    self.assertLen(train_idx, 2)
    self.assertLen(valid_idx, 2)

    # 3. Error when temporal_split=True but ts_feature is None
    with self.assertRaises(ValueError):
      node_prediction_dataset.compute_train_and_valid_node_idxs(
          graph=graph,
          valid_graph=None,
          graph_format="IN_MEMORY_GRAPH",
          target_nodeset="n1",
          random_seed=42,
          validation_ratio=0.5,
          train_seed_nodes=None,
          valid_seed_nodes=None,
          max_num_valid_examples=None,
          temporal_split=True,
          ts_feature=None,
      )

    # 4. Error when temporal_split=True but ts_feature is not found in graph
    with self.assertRaises(ValueError):
      node_prediction_dataset.compute_train_and_valid_node_idxs(
          graph=graph,
          valid_graph=None,
          graph_format="IN_MEMORY_GRAPH",
          target_nodeset="n1",
          random_seed=42,
          validation_ratio=0.5,
          train_seed_nodes=None,
          valid_seed_nodes=None,
          max_num_valid_examples=None,
          temporal_split=True,
          ts_feature="non_existent_feature",
      )

  def test_prepare_datasets_automatic_temporal_split(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(False)
    train_dataset, valid_dataset = node_prediction_dataset.prepare_datasets(
        graph=graph,
        valid_graph=None,  # pyrefly: ignore[bad-argument-type]
        schema=schema,
        target_nodeset="n1",
        random_seed=42,
        batch_size=2,
        num_sampling_hops=1,
        sampling_width=3,
        verbose=0,
        graph_format="IN_MEMORY_GRAPH",
        validation_ratio=0.5,
        train_seed_nodes=None,
        valid_seed_nodes=None,
        temporal_sampling=True,
        nodeset_timestamp_features={"n1": "timestamp"},
        edgeset_timestamp_features={"e1": "timestamp"},
        num_valid_steps=None,
        cache_valid_dataset=False,
        cache_normalized_features=False,
        cache_normalized_features_device="host",
        sampling_plan=None,
        padding_margin=0.1,
        text_embedder="nnlm-en-dim128",
    )
    self.assertIsNotNone(train_dataset.seed_node_idxs)
    self.assertIsNotNone(valid_dataset)
    assert valid_dataset is not None
    self.assertIsNotNone(valid_dataset.seed_node_idxs)
    np.testing.assert_array_equal(
        np.sort(train_dataset.seed_node_idxs), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        np.sort(valid_dataset.seed_node_idxs), np.array([2, 3])
    )

  def test_padding_margin_scales_padding_size(self):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    schema = gen_test_graph.generate_schema(True, False, True, False)
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1", num_hops=2, hop_width=3, reverse=True
        ),
        schema,
    )
    prep_zero = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_plan=sampling_plan,
        batch_size=2,
        drop_remainder=True,
        shuffle=False,
        padding_margin=0.0,
    )
    prep_zero.prepare()
    prep_large = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_plan=sampling_plan,
        batch_size=2,
        drop_remainder=True,
        shuffle=False,
        padding_margin=20.0,
    )
    prep_large.prepare()
    large_nodes = prep_large.get_live().padding.node_sets["n1"].num_nodes
    zero_nodes = prep_zero.get_live().padding.node_sets["n1"].num_nodes
    assert large_nodes is not None and zero_nodes is not None
    self.assertGreater(large_nodes, zero_nodes)

  def test_target_nodeset_statistics_use_seed_nodes_only(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "static_feat": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        shape=(),
                    ),
                    "ts_feat": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        shape=(None,),
                        is_timeseries=True,
                        group="g",
                    ),
                    "cat_feat": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        shape=(),
                    ),
                    "label": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        shape=(),
                    ),
                }
            ),
            "n2": schema_lib.NodeSchema(
                features={
                    "aux_feat": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        shape=(),
                    ),
                }
            ),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1", features={}),
            "e2": schema_lib.EdgeSchema(source="n2", target="n1", features={}),
        },
    )
    # Nodes 0, 1 are training seeds; nodes 2..66 (65 nodes) are non-seed target
    # nodes reached via 1-hop reverse edges from seeds 0 and 1. Having 67 total
    # sampled n1 nodes makes padding (ceil(67) + 64 = 131 -> rounded to 256)
    # distinguishable from seed-only (2 + 64 = 66 -> rounded to 128).
    num_non_seed = 65
    num_n1 = 2 + num_non_seed
    ts_values = np.empty(num_n1, dtype=np.object_)
    ts_values[0] = np.array([10.0, 20.0], dtype=np.float32)
    ts_values[1] = np.array([30.0], dtype=np.float32)
    for i in range(2, num_n1):
      ts_values[i] = np.array([9999.0, 8888.0, 7777.0], dtype=np.float32)

    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=num_n1,
                features={
                    "static_feat": np.array(
                        [1.0, 2.0] + [1000.0] * num_non_seed, dtype=np.float32
                    ),
                    "ts_feat": ts_values,
                    "cat_feat": np.array(
                        [b"train_a", b"train_b"] + [b"leaked_val"] * num_non_seed
                    ),
                    "label": np.array(
                        [0.1, 0.2] + [500.0] * num_non_seed, dtype=np.float32
                    ),
                },
            ),
            "n2": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "aux_feat": np.array([5.0, 50.0], dtype=np.float32),
                },
            ),
        },
        edge_sets={
            # Reverse sampling from seeds [0, 1] pulls in all 65 non-seed n1 nodes.
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array(
                    [list(range(2, num_n1)), [0] * num_non_seed], dtype=np.int64
                ),
                features={},
            ),
            "e2": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 1], [0, 1]], dtype=np.int64),
                features={},
            ),
        },
    )
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1", num_hops=1, hop_width=100, reverse=True
        ),
        schema,
    )
    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_plan=sampling_plan,
        seed_node_idxs=np.array([0, 1], dtype=np.int64),
        batch_size=2,
        drop_remainder=True,
        shuffle=False,
        padding_margin=0.0,
        target_column="label",
        auto_normalize_config=normalize_lib.AutoNormalizeConfig(
            keep_raw_features={("n1", "label")}
        ),
    )
    preparator.prepare()
    live = preparator.get_live()

    # Target nodeset statistics must only reflect seed nodes [0, 1].
    n1_stats = live.feature_stats.node_sets["n1"].features
    self.assertEqual(n1_stats["static_feat"].count, 2)
    self.assertAlmostEqual(n1_stats["static_feat"].minimum, 1.0)
    self.assertAlmostEqual(n1_stats["static_feat"].maximum, 2.0)
    self.assertLessEqual(max(n1_stats["static_feat"].quantiles), 2.0)

    self.assertEqual(n1_stats["ts_feat"].count, 2)
    self.assertAlmostEqual(n1_stats["ts_feat"].minimum, 10.0)
    self.assertAlmostEqual(n1_stats["ts_feat"].maximum, 30.0)
    self.assertLessEqual(max(n1_stats["ts_feat"].quantiles), 30.0)

    self.assertEqual(n1_stats["cat_feat"].count, 2)
    self.assertCountEqual(
        n1_stats["cat_feat"].dictionary.keys(), ["train_a", "train_b"]
    )
    self.assertEqual(n1_stats["label"].count, 2)
    self.assertAlmostEqual(n1_stats["label"].minimum, 0.1)
    self.assertAlmostEqual(n1_stats["label"].maximum, 0.2)

    # Auxiliary nodeset statistics are unchanged.
    n2_stats = live.feature_stats.node_sets["n2"].features
    self.assertEqual(n2_stats["aux_feat"].count, 2)
    self.assertAlmostEqual(n2_stats["aux_feat"].minimum, 5.0)
    self.assertAlmostEqual(n2_stats["aux_feat"].maximum, 50.0)

    # Padding statistics still account for all 67 sampled n1 nodes (67 + 64
    # margin = 131, rounded up to multiple of 128 -> 256, vs 128 if seed-only).
    self.assertEqual(live.padding.node_sets["n1"].num_nodes, 256)


class RepeatToBatchSizeTest(absltest.TestCase):

  def test_repeats_small_seed_sets(self):
    np.testing.assert_array_equal(
        node_prediction_dataset.repeat_to_batch_size(np.array([1, 2]), 5),
        [1, 2, 1, 2, 1, 2],
    )

  def test_keeps_large_seed_sets(self):
    seeds = np.arange(10)
    self.assertIs(node_prediction_dataset.repeat_to_batch_size(seeds, 5), seeds)
    self.assertIsNone(node_prediction_dataset.repeat_to_batch_size(None, 5))


if __name__ == "__main__":
  absltest.main()
