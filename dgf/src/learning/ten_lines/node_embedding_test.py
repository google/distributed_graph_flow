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

"""Tests for the unsupervised node embedding ten-lines model and trainer."""

import copy
import dataclasses
import os
import tempfile
from typing import Any
import unittest.mock

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.generate import graphs as synthetic_lib
from dgf.src.io import tf as tf_io
from dgf.src.io import tf_graph_sample as tf_graph_sample_lib
from dgf.src.learning.jax import common as jax_common
from dgf.src.learning.jax.layers import graphmae
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.ten_lines import common as common_lib
from dgf.src.learning.ten_lines import dataset
from dgf.src.learning.ten_lines import node_embedding_core_model
from dgf.src.learning.ten_lines import node_embedding_model
from dgf.src.learning.ten_lines import node_embedding_train
from dgf.src.learning.ten_lines import node_prediction_dataset
from dgf.src.learning.ten_lines import node_prediction_model
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import log
from dgf.src.util import test_util
import flax.linen as nn
import jax
import numpy as np
import tensorflow as tf


class NodeEmbeddingModelStandaloneTest(absltest.TestCase):
  """Tests of `NodeEmbeddingModel` logic that do not require training."""

  def setUp(self):
    super().setUp()
    self.graph, _ = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=60, num_n2_nodes=40
    )
    self.data = unittest.mock.MagicMock()
    self.data.core_model_config.graphmae.target_nodeset = "N1"
    self.data.core_model_config.graphmae.get_gnn_dims.return_value = 8
    self.model = node_embedding_model.NodeEmbeddingModel(self.data)

  def _batch(
      self, seed_node_idxs: list[int]
  ) -> node_embedding_model.BatchEmbedding:
    """Returns a batch whose embeddings are filled with the first seed index."""
    return node_embedding_model.BatchEmbedding(
        batch_seed_node_idxs=np.asarray(seed_node_idxs, dtype=np.int32),
        normalized_merged_graph=self.graph,
        merged_seed_node_idxs=np.arange(len(seed_node_idxs), dtype=np.int32),
        embeddings=np.full(
            (len(seed_node_idxs), 8), seed_node_idxs[0], dtype=np.float32
        ),
    )

  def test_registration(self):
    self.assertEqual(
        node_embedding_model.NodeEmbeddingModel.name(), "NodeEmbedding"
    )
    registered = common_lib.REGISTERED_MODELS["NodeEmbedding"]
    self.assertIs(
        registered.model_class, node_embedding_model.NodeEmbeddingModel
    )
    self.assertIs(registered.data_class, node_embedding_model.ModelData)

  def test_properties(self):
    self.assertIs(self.model.data(), self.data)
    self.assertEqual(self.model.target_nodeset, "N1")
    self.assertEqual(self.model.embedding_dim, 8)

  def test_predict_concatenates_batches(self):
    with unittest.mock.patch.object(
        self.model,
        "predict_batch",
        return_value=iter([self._batch([0, 1]), self._batch([2])]),
    ) as predict_batch:
      embeddings = self.model.predict(self.graph, [0, 1, 2], verbose=0)
    np.testing.assert_array_equal(
        predict_batch.call_args.args[1], np.array([0, 1, 2], dtype=np.int32)
    )
    self.assertEqual(embeddings.shape, (3, 8))
    np.testing.assert_array_equal(embeddings[:, 0], [0.0, 0.0, 2.0])

  def test_predict_defaults_to_all_target_nodes(self):
    num_nodes = self.graph.node_sets["N1"].num_nodes
    assert num_nodes is not None
    with unittest.mock.patch.object(
        self.model,
        "predict_batch",
        return_value=iter([self._batch(list(range(num_nodes)))]),
    ) as predict_batch:
      embeddings = self.model.predict(self.graph, verbose=0)
    np.testing.assert_array_equal(
        predict_batch.call_args.args[1], np.arange(num_nodes, dtype=np.int32)
    )
    self.assertEqual(embeddings.shape, (num_nodes, 8))

  def test_predict_empty_seed_node_idxs(self):
    with unittest.mock.patch.object(
        self.model, "predict_batch"
    ) as predict_batch:
      embeddings = self.model.predict(self.graph, [], verbose=0)
    predict_batch.assert_not_called()
    self.assertEqual(embeddings.shape, (0, 8))
    self.assertEqual(embeddings.dtype, np.float32)

  def test_without_sampling_plan_raises(self):
    self.data.sampling_plan = None
    with self.assertRaisesRegex(ValueError, "does not have a sampling plan"):
      next(self.model.predict_batch(self.graph, [0], verbose=0))
    with self.assertRaisesRegex(ValueError, "does not have a sampling plan"):
      self.model.evaluate(self.graph, verbose=0)

  def test_extract_serving_schemata_requires_uuid(self):
    self.model.metadata.uuid = None
    with self.assertRaisesRegex(ValueError, "Model UUID is not set"):
      self.model._extract_serving_schemata()


def _core_model_config(
    pre_mlp: jax_common.GenericLayer, gnn_dims: int
) -> node_embedding_core_model.CoreModelConfig:
  """Returns a `CoreModelConfig` on N1 with the given `pre_mlp` and GNN dims."""
  return node_embedding_core_model.CoreModelConfig(
      embed_graph=preprocess.EmbedGraphConfig(),
      pre_mlp=pre_mlp,
      graphmae=graphmae.GraphMAEConfig(
          encoder_conv=hetero_gnn.HeterogeneousGraphConvolutionConfig(
              dims=gnn_dims, message_pooling="sum"
          ),
          num_encoder_layers=1,
          decoder_conv=hetero_gnn.HeterogeneousGraphConvolutionConfig(
              dims=gnn_dims,
              embedding_feature="latent_embedding",
              message_pooling="sum",
          ),
          target_nodeset="N1",
      ),
  )


class _Dense(nn.Module):
  """A dense layer with the `GenericLayer` call signature."""

  dims: int

  @nn.compact
  def __call__(self, x: jax.Array, training: bool) -> jax.Array:
    del training  # Unused.
    return nn.Dense(self.dims)(x)


@dataclasses.dataclass(frozen=True)
class _DenseConfig:
  """A `GenericLayer` that is not a `standard.GenericBlockConfig`."""

  dims: int

  def make(self, name: str | None = None) -> nn.Module:
    return _Dense(dims=self.dims, name=name)

  def architecture(self) -> str:
    return f"Dense({self.dims})"


class CoreModelConfigTest(absltest.TestCase):
  """Tests of the validation of `node_embedding_core_model.CoreModelConfig`."""

  def setUp(self):
    super().setUp()
    _, self.schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=60, num_n2_nodes=40
    )
    del self.schema.node_sets["N1"].features["label"]

  def _assert_projected_to(
      self, config: node_embedding_core_model.CoreModelConfig, dims: int
  ):
    config.make(self.schema)
    projected_schema = config.projected_schema(self.schema)
    self.assertEqual(
        set(projected_schema.node_sets), set(self.schema.node_sets)
    )
    for nodeset_schema in projected_schema.node_sets.values():
      self.assertEqual(nodeset_schema.features["embedding"].shape, (dims,))
    self.assertEqual(projected_schema.edge_sets, self.schema.edge_sets)

  def test_valid_pre_mlp(self):
    self._assert_projected_to(
        _core_model_config(standard.ingest_feature(8), gnn_dims=8), dims=8
    )
    self._assert_projected_to(
        _core_model_config(
            standard.GenericBlockConfig("L2AL4", dims=2), gnn_dims=8
        ),
        dims=8,
    )

  def test_pre_mlp_not_generic_block(self):
    self._assert_projected_to(
        _core_model_config(
            _DenseConfig(dims=8),  # pyrefly: ignore[bad-argument-type]
            gnn_dims=8,
        ),
        dims=8,
    )

  def test_pre_mlp_dims_mismatch_raises(self):
    with self.assertRaisesRegex(
        ValueError, r"Got output shape \(4,\) for input shape"
    ):
      _core_model_config(standard.ingest_feature(4), gnn_dims=8).make(
          self.schema
      )

  def test_pre_mlp_without_projection_raises(self):
    with self.assertRaisesRegex(
        ValueError, "`pre_mlp` must project the embedding of nodeset"
    ):
      _core_model_config(standard.identity(), gnn_dims=64).make(self.schema)


def _untrained_model(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
) -> node_embedding_model.NodeEmbeddingModel:
  """Returns a `NodeEmbeddingModel` on N1 with randomly initialized weights.

  The model is assembled from the same components as
  `train_node_embedding_model`, but without any training step.

  Args:
    graph: The toy classification graph.
    schema: The schema of `graph`, without the "label" feature.
  """
  dims = 8
  max_degree = max(
      int(np.max(np.bincount(edgeset.adjacency.ravel())))
      for edgeset in graph.edge_sets.values()
  )
  hparams = node_embedding_model.HParam(
      batch_size=8,
      node_embedding_dim=dims,
      num_layers=1,
      num_sampling_hops=1,
      # Larger than the maximum node degree so 1-hop sampling is deterministic.
      sampling_width=max_degree + 1,
  )
  train_dataset, _ = node_prediction_dataset.prepare_datasets(
      graph=graph,
      valid_graph=None,  # pyrefly: ignore[bad-argument-type]
      schema=schema,
      target_nodeset="N1",
      random_seed=hparams.random_seed,
      batch_size=hparams.batch_size,
      num_sampling_hops=hparams.num_sampling_hops,
      sampling_width=hparams.sampling_width,
      verbose=0,
      graph_format=dataset.GraphFormat.AUTO,
      validation_ratio=0.1,
      train_seed_nodes=None,
      valid_seed_nodes=None,
      temporal_sampling=False,
      nodeset_timestamp_features={},
      edgeset_timestamp_features={},
      num_valid_steps=None,
      cache_valid_dataset=False,
      cache_normalized_features=False,
      cache_normalized_features_device="device",
      sampling_plan=None,
      padding_margin=hparams.padding_margin,
      keep_raw_features=set(),
      target_column=None,
      text_embedder="nnlm-en-dim128",
  )
  live = train_dataset.get_live()

  core_model_config = _core_model_config(
      standard.ingest_feature(dims), gnn_dims=dims
  )
  core_model = core_model_config.make(train_dataset.generated_schema())
  sample, merge_offsets = next(iter(train_dataset.generate_jax()))
  model_params = core_model.init(
      {
          "params": jax.random.PRNGKey(0),
          "dropout": jax.random.PRNGKey(1),
          "masking": jax.random.PRNGKey(2),
      },
      (sample, merge_offsets["N1"][:-1]),
      training=True,
  )
  return node_embedding_model.NodeEmbeddingModel(
      node_embedding_model.ModelData(
          core_model_config=core_model_config,
          hparams=hparams,
          schema=schema,
          normalizer_config=live.normalizer.config,
          padding=live.padding,
          sampling_plan=live.sampling_plan,
          feature_stats=live.feature_stats,
          training_stats=node_prediction_model.TrainingStats(
              num_train_seed_nodes=train_dataset.num_nodes_in_seed_nodeset(),
              num_valid_seed_nodes=None,
              train_duration_seconds=0.0,
              num_skipped_train_samples=0,
          ),
          model_params=model_params,
      )
  )


def _single_sample_padding(
    model: node_embedding_model.NodeEmbeddingModel,
    graph: in_memory_graph_lib.InMemoryGraph,
    seed_node_idxs: list[int],
    num_draws: int = 5,
) -> padding_lib.Padding:
  """Returns a padding that fits a single sample, but not a merged pair."""
  data = model.data()
  assert data.sampling_plan is not None
  sampler = in_memory_sampler_lib.create_sampler(
      graph=graph,
      plan=data.sampling_plan,
      schema=data.schema,
      batch_size=len(seed_node_idxs),
  )
  samples = []
  for _ in range(num_draws):
    samples.extend(sampler.sample(np.asarray(seed_node_idxs)))

  node_sets = {
      name: padding_lib.NodeSetPadding(
          num_nodes=max((s.node_sets[name].num_nodes or 0) for s in samples) + 1
      )
      for name in data.schema.node_sets
  }
  edge_sets = {
      name: padding_lib.EdgeSetPadding(
          num_edges=max(s.edge_sets[name].num_edges() for s in samples) + 1
      )
      for name in data.schema.edge_sets
  }
  return padding_lib.Padding(node_sets=node_sets, edge_sets=edge_sets)


def _filter_graph_to_schema(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
) -> in_memory_graph_lib.InMemoryGraph:
  """Returns a shallow copy of `graph` keeping only features in `schema`."""
  node_sets = {
      name: in_memory_graph_lib.InMemoryNodeSet(
          num_nodes=graph.node_sets[name].num_nodes,
          features={
              feat_name: graph.node_sets[name].features[feat_name]
              for feat_name in node_schema.features
          },
      )
      for name, node_schema in schema.node_sets.items()
  }
  edge_sets = {
      name: in_memory_graph_lib.InMemoryEdgeSet(
          adjacency=graph.edge_sets[name].adjacency,
          features={
              feat_name: graph.edge_sets[name].features[feat_name]
              for feat_name in edge_schema.features
          },
      )
      for name, edge_schema in schema.edge_sets.items()
  }
  return in_memory_graph_lib.InMemoryGraph(
      node_sets=node_sets, edge_sets=edge_sets
  )


class NodeEmbeddingModelUntrainedTest(parameterized.TestCase):
  """End-to-end tests of `NodeEmbeddingModel` with random weights."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.graph, schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=300, num_n2_nodes=100, random_seed=0
    )
    del schema.node_sets["N1"].features["label"]
    cls.model = _untrained_model(cls.graph, schema)

  def test_predict(self):
    emb_subset = self.model.predict(self.graph, [0, 1, 2], verbose=0)
    self.assertEqual(emb_subset.shape, (3, 8))
    self.assertEqual(emb_subset.dtype, np.float32)
    self.assertTrue(np.all(np.isfinite(emb_subset)))

    num_n1 = self.graph.node_sets["N1"].num_nodes
    emb_all = self.model.predict(self.graph, verbose=0)
    self.assertEqual(emb_all.shape, (num_n1, 8))
    np.testing.assert_allclose(emb_subset, emb_all[:3], atol=1e-5)

  def test_predict_batch_splits_oversized_batches(self):
    out_degrees = np.bincount(self.graph.edge_sets["N1_to_N2"].adjacency[0])
    seed_node_idxs = [int(i) for i in np.argsort(out_degrees)[-4:]]
    data = self.model.data()
    original_padding = data.padding
    data.padding = _single_sample_padding(
        self.model, self.graph, seed_node_idxs
    )
    try:
      batches = list(
          self.model.predict_batch(
              self.graph, seed_node_idxs=seed_node_idxs, verbose=0
          )
      )
    finally:
      data.padding = original_padding

    self.assertLen(batches, len(seed_node_idxs))
    for batch in batches:
      self.assertIsInstance(batch, node_embedding_model.BatchEmbedding)
      assert batch.batch_seed_node_idxs is not None
      self.assertLen(batch.batch_seed_node_idxs, 1)
      self.assertEqual(batch.embeddings.shape, (1, 8))
    self.assertEqual(
        [
            int(batch.batch_seed_node_idxs[0])
            for batch in batches
            if batch.batch_seed_node_idxs is not None
        ],
        seed_node_idxs,
    )

  @parameterized.named_parameters(
      dict(testcase_name="host", to_host=True, expected_type=np.ndarray),
      dict(testcase_name="device", to_host=False, expected_type=jax.Array),
  )
  def test_predict_batch_array_type(
      self, to_host: bool, expected_type: type[Any]
  ):
    batches = list(
        self.model.predict_batch(
            self.graph, seed_node_idxs=[0, 1, 2], verbose=0, to_host=to_host
        )
    )
    self.assertNotEmpty(batches)
    for batch in batches:
      self.assertIsInstance(batch.embeddings, expected_type)

  def test_cached_normalized_features_match_uncached(self):
    num_n1_nodes = self.graph.node_sets["N1"].num_nodes
    assert num_n1_nodes is not None
    # pylint: disable=protected-access
    self.assertFalse(
        self.model._should_cache_normalized_features(self.graph, 1)
    )
    self.assertTrue(
        self.model._should_cache_normalized_features(self.graph, num_n1_nodes)
    )
    # pylint: enable=protected-access

    # 101 seeds is not a multiple of the batch size (8), and `self.graph` still
    # contains the excluded "label" feature.
    with unittest.mock.patch.object(
        node_embedding_model.NodeEmbeddingModel,
        "_should_cache_normalized_features",
        return_value=True,
    ):
      emb_cached = self.model.predict(
          self.graph, seed_node_idxs=np.arange(101), verbose=0
      )
      eval_cached = self.model.evaluate(self.graph, verbose=0)
    with unittest.mock.patch.object(
        node_embedding_model.NodeEmbeddingModel,
        "_should_cache_normalized_features",
        return_value=False,
    ):
      emb_uncached = self.model.predict(
          self.graph, seed_node_idxs=np.arange(101), verbose=0
      )
      eval_uncached = self.model.evaluate(self.graph, verbose=0)

    self.assertEqual(emb_cached.shape, (101, 8))
    np.testing.assert_allclose(emb_cached, emb_uncached, atol=1e-5)
    self.assertEqual(eval_cached.num_examples, eval_uncached.num_examples)
    assert eval_cached.loss is not None and eval_uncached.loss is not None
    self.assertAlmostEqual(eval_cached.loss, eval_uncached.loss, places=5)

  def test_evaluate(self):
    eval_res = self.model.evaluate(
        self.graph, seed_node_idxs=[0, 1, 2, 3], verbose=0
    )
    self.assertEqual(eval_res.num_examples, 4)
    assert eval_res.loss is not None
    self.assertTrue(np.isfinite(eval_res.loss))
    for metric_key in (
        "recon_loss",
        "var_loss",
        "cov_loss",
        "total_loss",
        "recon_cosine",
        "recon_relative_error",
        "embedding_rank_ratio",
        "target_rank_ratio",
    ):
      self.assertTrue(np.isfinite(eval_res.user_metrics[metric_key]))
    self.assertBetween(eval_res.user_metrics["recon_cosine"], -1.0, 1.0)
    self.assertGreaterEqual(eval_res.user_metrics["recon_relative_error"], 0.0)
    for metric_key in ("embedding_rank_ratio", "target_rank_ratio"):
      self.assertBetween(eval_res.user_metrics[metric_key], 0.0, 1.0 + 1e-5)

  def test_evaluate_without_examples_raises(self):
    with self.assertRaisesRegex(ValueError, "No examples to evaluate"):
      self.model.evaluate_generator(iter([]), verbose=0)

  def test_predict_on_graph_sample_batch(self):
    sampling_plan = self.model.data().sampling_plan
    assert sampling_plan is not None
    sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=sampling_plan,
        schema=self.model.data().schema,
        batch_size=5,
    )
    samples = [sampler.sample(0), sampler.sample(1)]
    embeddings = self.model.predict_on_graph_sample_batch(samples)
    # By convention, the seed node is the first node of the target nodeset of a
    # graph sample.
    expected = np.concatenate(
        [self.model.predict(sample, [0], verbose=0) for sample in samples]
    )
    self.assertEqual(embeddings.shape, (2, 8))
    np.testing.assert_allclose(embeddings, expected, atol=1e-5)

  @parameterized.parameters("TF_GRAPH", "TF_GRAPH_DICT")
  def test_to_tensorflow_function(self, input_format: str):
    consume_tf_graph_dict = input_format == "TF_GRAPH_DICT"
    schema = self.model.data().schema
    sampling_plan = self.model.data().sampling_plan
    assert sampling_plan is not None
    sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=sampling_plan,
        schema=schema,
        batch_size=5,
    )
    sample_1 = _filter_graph_to_schema(sampler.sample(0), schema)
    sample_2 = _filter_graph_to_schema(sampler.sample(1), schema)
    tf_sample_1 = tf_io.graph_to_tf_graph(sample_1, schema=schema)
    tf_sample_2 = tf_io.graph_to_tf_graph(sample_2, schema=schema)

    if consume_tf_graph_dict:
      kwargs_call_1 = {
          **tf_io.tf_graph_to_tf_graph_dict(tf_sample_1),
          "seed_node_idxs": tf.constant([0]),
      }
      kwargs_call_2 = {
          **tf_io.tf_graph_to_tf_graph_dict(tf_sample_2),
          "seed_node_idxs": tf.constant([0]),
      }
    else:
      kwargs_call_1 = {"graph": tf_sample_1, "seed_node_idxs": tf.constant([0])}
      kwargs_call_2 = {"graph": tf_sample_2, "seed_node_idxs": tf.constant([0])}

    tf_embed_fn = self.model.to_tensorflow_function(input_format=input_format)
    emb_sample_1 = tf_embed_fn(**kwargs_call_1)

    expected_1 = self.model.predict(sample_1, [0], verbose=0)
    expected_2 = self.model.predict(sample_2, [0], verbose=0)
    np.testing.assert_allclose(emb_sample_1.numpy(), expected_1, atol=1e-5)

    with tempfile.TemporaryDirectory() as tmpdir:
      tf.saved_model.save(tf_embed_fn, tmpdir)
      loaded = tf.saved_model.load(tmpdir)
      loaded_1 = loaded(**kwargs_call_1)
      loaded_2 = loaded(**kwargs_call_2)

    self.assertIn("serving_default", loaded.signatures)
    if consume_tf_graph_dict:
      input_names = [
          t.name for t in loaded.signatures["serving_default"].inputs
      ]
      self.assertTrue(any("seed_node_idxs" in name for name in input_names))
      # The excluded "label" feature is not part of the serving signature.
      self.assertFalse(any("label" in name for name in input_names))

    np.testing.assert_allclose(loaded_1.numpy(), expected_1, atol=1e-5)
    np.testing.assert_allclose(loaded_2.numpy(), expected_2, atol=1e-5)

  def test_to_tensorflow_function_serialized_tfgnn_graphs(self):
    schema = self.model.data().schema
    sampling_plan = self.model.data().sampling_plan
    assert sampling_plan is not None
    sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=sampling_plan,
        schema=schema,
        batch_size=5,
    )
    samples = [sampler.sample(0), sampler.sample(1)]
    serialized_samples = tf.constant([
        tf_graph_sample_lib.graph_to_tfgnn_graph(
            sample, schema=schema
        ).SerializeToString()
        for sample in samples
    ])
    expected = np.concatenate(
        [self.model.predict(sample, [0], verbose=0) for sample in samples],
        axis=0,
    )

    tf_embed_fn = self.model.to_tensorflow_function(
        input_format="SERIALIZED_TFGNN_GRAPHS"
    )
    embeddings = tf_embed_fn(serialized_samples)
    self.assertEqual(embeddings.shape.as_list(), list(expected.shape))
    np.testing.assert_allclose(embeddings.numpy(), expected, atol=1e-5)

    with tempfile.TemporaryDirectory() as tmpdir:
      tf.saved_model.save(tf_embed_fn, tmpdir)
      loaded = tf.saved_model.load(tmpdir)
      loaded_embeddings = loaded(serialized_samples)

    self.assertIn("serving_default", loaded.signatures)
    input_names = [t.name for t in loaded.signatures["serving_default"].inputs]
    self.assertTrue(any("examples" in name for name in input_names))
    np.testing.assert_allclose(loaded_embeddings.numpy(), expected, atol=1e-5)

  def test_to_tensorflow_function_deprecated_consume_tf_graph_dict(self):
    schema = self.model.data().schema
    sampling_plan = self.model.data().sampling_plan
    assert sampling_plan is not None
    sample = _filter_graph_to_schema(
        in_memory_sampler_lib.create_sampler(
            graph=self.graph,
            plan=sampling_plan,
            schema=schema,
            batch_size=5,
        ).sample(0),
        schema,
    )
    tf_sample = tf_io.graph_to_tf_graph(sample, schema=schema)
    expected = self.model.predict(sample, [0], verbose=0)

    tf_embed_fn = self.model.to_tensorflow_function(consume_tf_graph_dict=True)
    embeddings = tf_embed_fn(
        **tf_io.tf_graph_to_tf_graph_dict(tf_sample),
        seed_node_idxs=tf.constant([0]),
    )
    np.testing.assert_allclose(embeddings.numpy(), expected, atol=1e-5)

    with self.assertRaisesRegex(ValueError, "cannot be set at the same time"):
      self.model.to_tensorflow_function(
          input_format="TF_GRAPH", consume_tf_graph_dict=True
      )

  def test_extract_serving_schemata(self):
    with unittest.mock.patch.object(self.model.metadata, "uuid", "test-uuid"):
      signature, pred_schema = self.model._extract_serving_schemata()
    self.assertEqual(
        pred_schema, {"type": "array", "items": {"type": "number"}}
    )
    self.assertEqual(signature["title"], "NodeEmbedding_N1_test-uuid")
    (input_graph,) = signature["x-google-gnn-input-graphs"]
    self.assertEqual(input_graph["input_node"], "N1")
    self.assertNotEmpty(input_graph["sampling_plan"])
    self.assertIn("gnn_N1_seed_node_idxs", signature["required"])

  def test_describe(self):
    html_str = self.model.describe()._repr_html_()
    self.assertIn("<tr><td>Excluded features</td><td>None</td></tr>", html_str)
    for section in (
        "Objective",
        "Unsupervised Node Embedding (GraphMAE)",
        "Schema",
        "Architecture",
        "Padding",
    ):
      self.assertIn(section, html_str)

  def test_save_and_load(self):
    expected = self.model.predict(self.graph, [0, 1, 2], verbose=0)
    with tempfile.TemporaryDirectory() as tmpdir:
      save_dir = os.path.join(tmpdir, "model")
      self.model.save(save_dir)
      loaded = common_lib.load_model(save_dir)
    assert isinstance(loaded, node_embedding_model.NodeEmbeddingModel)
    self.assertEqual(loaded.target_nodeset, "N1")
    self.assertIsNotNone(self.model.metadata.uuid)
    self.assertEqual(self.model.metadata.uuid, loaded.metadata.uuid)
    test_util.assert_are_equal(self, self.model.data(), loaded.data())
    test_util.assert_are_equal(
        self,
        self.model._get_live().normalized_schema,
        loaded._get_live().normalized_schema,
    )
    np.testing.assert_allclose(
        loaded.predict(self.graph, [0, 1, 2], verbose=0),
        expected,
        atol=1e-6,
    )


# Training arguments for fast unit tests.
RAPID_TRAINING_KWARGS: dict[str, Any] = {
    "num_train_steps": 4,
    "valid_every_n_steps": 4,
    "num_valid_steps": 2,
    "batch_size": 4,
    "sampling_width": 30,
    "num_sampling_hops": 1,
    "padding_margin": 2.0,
    "node_embedding_dim": 8,
    "num_layers": 1,
    "evaluate_final_model": False,
    "fast_compile": True,
    "verbose": 0,
}


def _strip_all_raw_features(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Returns a copy of `(graph, schema)` keeping only `PRIMARY_ID` features."""
  node_sets = {}
  node_schemas = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    nodeset = graph.node_sets[nodeset_name]
    feature_schemas = {
        k: copy.deepcopy(v)
        for k, v in nodeset_schema.features.items()
        if v.semantic == schema_lib.FeatureSemantic.PRIMARY_ID
    }
    node_schemas[nodeset_name] = schema_lib.NodeSchema(
        features=feature_schemas
    )
    node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=nodeset.num_nodes,
        features={k: nodeset.features[k] for k in feature_schemas},
    )
  return (
      in_memory_graph_lib.InMemoryGraph(
          node_sets=node_sets, edge_sets=dict(graph.edge_sets)
      ),
      schema_lib.GraphSchema(
          node_sets=node_schemas, edge_sets=copy.deepcopy(schema.edge_sets)
      ),
  )


class NodeEmbeddingTrainTest(parameterized.TestCase):
  """Tests of `train_node_embedding_model`."""

  def setUp(self):
    super().setUp()
    self.graph, self.schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=60, num_n2_nodes=40
    )

  def test_train_predict_evaluate_and_save_load(self):
    training_kwargs: dict[str, Any] = {
        **RAPID_TRAINING_KWARGS,
        "evaluate_final_model": True,
    }
    model = node_embedding_train.train_node_embedding_model(
        self.graph,
        self.schema,
        target_nodeset="N1",
        exclude_features={"N1": ["label"]},
        **training_kwargs,
    )
    self.assertIsInstance(model, node_embedding_model.NodeEmbeddingModel)
    self.assertEqual(model.target_nodeset, "N1")
    self.assertEqual(model.embedding_dim, 8)
    self.assertEqual(model.data().excluded_features, {"N1": ["label"]})
    self.assertNotIn("label", model.data().schema.node_sets["N1"].features)
    self.assertIsNotNone(model.data().final_evaluation)
    self.assertIsNotNone(model.metadata.trainig_logs)

    embeddings = model.predict(self.graph, [0, 1, 2], verbose=0)
    self.assertEqual(embeddings.shape, (3, 8))
    self.assertTrue(np.all(np.isfinite(embeddings)))

    eval_res = model.evaluate(
        self.graph, seed_node_idxs=[0, 1, 2, 3], verbose=0
    )
    self.assertEqual(eval_res.num_examples, 4)
    for metric_key in ("total_loss", "recon_cosine", "embedding_rank_ratio"):
      self.assertTrue(np.isfinite(eval_res.user_metrics[metric_key]))

    self.assertIn("Training", model.describe()._repr_html_())
    self.assertIn(
        "<tr><td>Excluded features</td><td>N1: label</td></tr>",
        model.describe()._repr_html_(),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      save_dir = os.path.join(tmpdir, "model")
      model.save(save_dir)
      loaded = common_lib.load_model(save_dir)
    assert isinstance(loaded, node_embedding_model.NodeEmbeddingModel)
    self.assertEqual(loaded.data().excluded_features, {"N1": ["label"]})
    np.testing.assert_allclose(
        loaded.predict(self.graph, [0, 1, 2], verbose=0),
        embeddings,
        atol=1e-6,
    )

  def test_exclude_features_prevents_label_leakage(self):
    model = node_embedding_train.train_node_embedding_model(
        self.graph,
        self.schema,
        target_nodeset="N1",
        exclude_features=["label"],
        **RAPID_TRAINING_KWARGS,
    )
    # The embeddings do not depend on the excluded feature.
    corrupted_graph = copy.deepcopy(self.graph)
    corrupted_graph.node_sets["N1"].features["label"] = np.full_like(
        corrupted_graph.node_sets["N1"].features["label"], 999
    )
    np.testing.assert_allclose(
        model.predict(self.graph, [0, 1, 2], verbose=0),
        model.predict(corrupted_graph, [0, 1, 2], verbose=0),
        atol=1e-6,
    )

  def test_single_nodeset_infers_target_nodeset(self):
    num_nodes = self.graph.node_sets["N1"].num_nodes
    assert num_nodes is not None
    schema = schema_lib.GraphSchema(
        node_sets={"only": copy.deepcopy(self.schema.node_sets["N1"])},
        edge_sets={
            "next": schema_lib.EdgeSchema(
                source="only", target="only", features={}
            )
        },
    )
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={"only": copy.deepcopy(self.graph.node_sets["N1"])},
        edge_sets={
            "next": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array(
                    [np.arange(num_nodes), np.roll(np.arange(num_nodes), -1)],
                    dtype=np.int64,
                ),
                features={},
            )
        },
    )
    model = node_embedding_train.train_node_embedding_model(
        graph, schema, exclude_features=["label"], **RAPID_TRAINING_KWARGS
    )
    self.assertEqual(model.target_nodeset, "only")

  @parameterized.parameters(
      "hmpnn", "hgat", "graphsage", "gcn", "gat", "gatv2", "mtalbis"
  )
  def test_architectures(self, architecture: str):
    model = node_embedding_train.train_node_embedding_model(
        self.graph,
        self.schema,
        target_nodeset="N1",
        exclude_features=["label"],
        architecture=architecture,
        **RAPID_TRAINING_KWARGS,
    )
    embeddings = model.predict(self.graph, [0, 1], verbose=0)
    self.assertEqual(embeddings.shape, (2, 8))
    self.assertTrue(np.all(np.isfinite(embeddings)))

  def test_valid_graph_and_manual_seed_nodes(self):
    model = node_embedding_train.train_node_embedding_model(
        self.graph,
        self.schema,
        valid_graph=self.graph,
        target_nodeset="N1",
        exclude_features=["label"],
        train_seed_nodes=list(range(20)),
        valid_seed_nodes=list(range(20, 30)),
        **RAPID_TRAINING_KWARGS,
    )
    self.assertEqual(model.data().training_stats.num_train_seed_nodes, 20)
    self.assertEqual(model.data().training_stats.num_valid_seed_nodes, 10)

  def test_checkpoint_every_n_minutes(self):
    work_dir = self.create_tempdir().full_path
    with unittest.mock.patch.object(
        node_embedding_train.flax_train,
        "train",
        wraps=node_embedding_train.flax_train.train,
    ) as mock_train:
      node_embedding_train.train_node_embedding_model(
          self.graph,
          self.schema,
          target_nodeset="N1",
          exclude_features=["label"],
          work_dir=work_dir,
          checkpoint_every_n_minutes=3.5,
          **RAPID_TRAINING_KWARGS,
      )
    self.assertEqual(
        mock_train.call_args.kwargs["checkpoint_every_n_minutes"], 3.5
    )
    self.assertNotEmpty(
        os.listdir(os.path.join(work_dir, "checkpoint", "checkpoints"))
    )

  def test_automatic_batch_size_and_learning_rate(self):
    model = node_embedding_train.train_node_embedding_model(
        self.graph,
        self.schema,
        target_nodeset="N1",
        exclude_features=["label"],
        num_train_steps=10,
        valid_every_n_steps=10,
        num_valid_steps=1,
        num_sampling_hops=1,
        sampling_width=5,
        node_embedding_dim=8,
        num_layers=1,
        evaluate_final_model=False,
        fast_compile=True,
        verbose=3,
    )
    hparams = model.data().hparams
    self.assertEqual(hparams.batch_size, 128)
    self.assertAlmostEqual(hparams.learning_rate, 5e-4)

  def test_batch_norm_statistics_are_updated_during_training(self):
    node_embedding_dim = RAPID_TRAINING_KWARGS["node_embedding_dim"]

    def preprocess_core_model_config(c):
      return dataclasses.replace(
          c,
          pre_mlp=standard.ingest_feature(
              node_embedding_dim, norm="batch_norm"
          ),
      )

    model = node_embedding_train.train_node_embedding_model(
        self.graph,
        self.schema,
        target_nodeset="N1",
        exclude_features=["label"],
        norm="batch_norm",
        experimental_preprocess_core_model_config=preprocess_core_model_config,
        **RAPID_TRAINING_KWARGS,
    )
    model_params = model.data().model_params
    assert model_params is not None
    running_means = [
        leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(
            model_params["batch_stats"]
        )
        if jax.tree_util.keystr(path).endswith("['mean']")
    ]
    self.assertNotEmpty(running_means)
    for running_mean in running_means:
      self.assertTrue(np.any(running_mean != 0.0))

    embeddings = model.predict(self.graph, [0, 1], verbose=0)
    self.assertEqual(embeddings.shape, (2, node_embedding_dim))

  def test_evaluate_final_model_with_cache_normalized_features(self):
    training_kwargs: dict[str, Any] = {
        **RAPID_TRAINING_KWARGS,
        "cache_normalized_features": True,
        "evaluate_final_model": True,
    }
    model = node_embedding_train.train_node_embedding_model(
        self.graph,
        self.schema,
        target_nodeset="N1",
        exclude_features=["label"],
        **training_kwargs,
    )
    self.assertIsNotNone(model.data().final_evaluation)

  @parameterized.parameters("cnn", "transformer")
  def test_timeseries_features(self, timeseries_encoder: str):
    graph = copy.deepcopy(self.graph)
    schema = copy.deepcopy(self.schema)
    num_n1 = graph.node_sets["N1"].num_nodes
    assert num_n1 is not None
    schema.node_sets["N1"].features["ts_val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        is_timeseries=True,
        shape=(None,),
        group="ts",
    )
    ts_data = np.empty(num_n1, dtype=object)
    for i in range(num_n1):
      ts_data[i] = np.array([float(i), float(i + 1)], dtype=np.float32)
    graph.node_sets["N1"].features["ts_val"] = ts_data

    model = node_embedding_train.train_node_embedding_model(
        graph,
        schema,
        target_nodeset="N1",
        exclude_features=["label"],
        timeseries_encoder=timeseries_encoder,
        timeseries_embedding_dim=8,
        **RAPID_TRAINING_KWARGS,
    )
    embeddings = model.predict(graph, [0, 1, 2], verbose=0)
    self.assertEqual(embeddings.shape, (3, 8))
    self.assertTrue(np.all(np.isfinite(embeddings)))

  def test_skipped_samples_warning_and_error(self):
    orig_prepare = (
        node_embedding_train.node_prediction_dataset.prepare_datasets
    )

    def mock_prepare_warn(*args, **kwargs):
      train_ds, valid_ds = orig_prepare(*args, **kwargs)
      train_ds.get_live().sample_generator.num_skipped_samples = 1
      return train_ds, valid_ds

    with unittest.mock.patch.object(
        node_embedding_train.node_prediction_dataset,
        "prepare_datasets",
        side_effect=mock_prepare_warn,
    ):
      training_kwargs: dict[str, Any] = {
          **RAPID_TRAINING_KWARGS,
          "num_train_steps": 10,
      }
      model_warn = node_embedding_train.train_node_embedding_model(
          self.graph,
          self.schema,
          target_nodeset="N1",
          exclude_features=["label"],
          **training_kwargs,
      )
    self.assertEqual(
        model_warn.data().training_stats.num_skipped_train_samples, 1
    )
    self.assertTrue(
        any(
            msg.severity == log.Severity.WARNING
            and "Skipped 1 out of" in msg.text
            for msg in (model_warn.metadata.captured_logs or [])
        )
    )

    def mock_prepare_fail(*args, **kwargs):
      train_ds, valid_ds = orig_prepare(*args, **kwargs)
      train_ds.get_live().sample_generator.num_skipped_samples = 20
      return train_ds, valid_ds

    with (
        unittest.mock.patch.object(
            node_embedding_train.node_prediction_dataset,
            "prepare_datasets",
            side_effect=mock_prepare_fail,
        ),
        unittest.mock.patch.object(common_lib, "SKIPPED_SAMPLES_CHECK_STEP", 3),
    ):
      training_kwargs = {**RAPID_TRAINING_KWARGS, "num_train_steps": 10}
      with self.assertRaisesRegex(
          merge_lib.InsufficientPaddingError, r"Skipped 20 out of 32 training"
      ):
        node_embedding_train.train_node_embedding_model(
            self.graph,
            self.schema,
            target_nodeset="N1",
            exclude_features=["label"],
            **training_kwargs,
        )

  def test_missing_target_nodeset_raises(self):
    with self.assertRaisesRegex(
        ValueError, "`target_nodeset` must be specified"
    ):
      node_embedding_train.train_node_embedding_model(
          self.graph, self.schema, **RAPID_TRAINING_KWARGS
      )

  def test_unknown_target_nodeset_raises(self):
    with self.assertRaisesRegex(ValueError, "not found in schema"):
      node_embedding_train.train_node_embedding_model(
          self.graph,
          self.schema,
          target_nodeset="unknown",
          **RAPID_TRAINING_KWARGS,
      )

  def test_target_nodeset_without_features_raises(self):
    graph, schema = _strip_all_raw_features(self.graph, self.schema)
    with self.assertRaisesRegex(ValueError, "has no input features"):
      node_embedding_train.train_node_embedding_model(
          graph, schema, target_nodeset="N1", **RAPID_TRAINING_KWARGS
      )

    # UNKNOWN-semantic features are not model inputs either.
    schema_with_unknown = copy.deepcopy(schema)
    graph_with_unknown = copy.deepcopy(graph)
    schema_with_unknown.node_sets["N1"].features["unknown_feat"] = (
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.UNKNOWN,
        )
    )
    graph_with_unknown.node_sets["N1"].features["unknown_feat"] = (
        self.graph.node_sets["N1"].features["f3"]
    )
    with self.assertRaisesRegex(ValueError, "has no input features"):
      node_embedding_train.train_node_embedding_model(
          graph_with_unknown,
          schema_with_unknown,
          target_nodeset="N1",
          **RAPID_TRAINING_KWARGS,
      )

    # Excluding all input features of the target nodeset also raises.
    with self.assertRaisesRegex(ValueError, "has no input features"):
      node_embedding_train.train_node_embedding_model(
          self.graph,
          self.schema,
          target_nodeset="N1",
          exclude_features=["label", "f1", "f3"],
          **RAPID_TRAINING_KWARGS,
      )

  def test_hash_prefixed_feature_with_input_semantic_is_used(self):
    graph, schema = _strip_all_raw_features(self.graph, self.schema)
    schema.node_sets["N1"].features["#f3"] = copy.deepcopy(
        self.schema.node_sets["N1"].features["f3"]
    )
    graph.node_sets["N1"].features["#f3"] = self.graph.node_sets["N1"].features[
        "f3"
    ]
    model = node_embedding_train.train_node_embedding_model(
        graph, schema, target_nodeset="N1", **RAPID_TRAINING_KWARGS
    )
    embeddings = model.predict(graph, [0, 1], verbose=0)
    self.assertEqual(embeddings.shape, (2, 8))
    self.assertTrue(np.all(np.isfinite(embeddings)))

  def test_time_aware_raises(self):
    with self.assertRaisesRegex(ValueError, "time_aware=True"):
      node_embedding_train.train_node_embedding_model(
          self.graph,
          self.schema,
          target_nodeset="N1",
          time_aware=True,
          **RAPID_TRAINING_KWARGS,
      )

  def test_invalid_exclude_features_raises(self):
    with self.assertRaisesRegex(ValueError, "not found in schema"):
      node_embedding_train.train_node_embedding_model(
          self.graph,
          self.schema,
          target_nodeset="N1",
          exclude_features={"unknown": ["label"]},
          **RAPID_TRAINING_KWARGS,
      )
    with self.assertRaisesRegex(
        ValueError,
        "Feature 'unknown_feat' in `exclude_features` not found in nodeset"
        " 'N1'",
    ):
      node_embedding_train.train_node_embedding_model(
          self.graph,
          self.schema,
          target_nodeset="N1",
          exclude_features=["unknown_feat"],
          **RAPID_TRAINING_KWARGS,
      )
    with self.assertRaisesRegex(
        ValueError,
        "Feature 'unknown_feat' in `exclude_features` not found in nodeset"
        " 'N1'",
    ):
      node_embedding_train.train_node_embedding_model(
          self.graph,
          self.schema,
          target_nodeset="N1",
          exclude_features={"N1": ["unknown_feat"]},
          **RAPID_TRAINING_KWARGS,
      )
    with self.assertRaisesRegex(ValueError, "not a bare string"):
      node_embedding_train.train_node_embedding_model(
          self.graph,
          self.schema,
          target_nodeset="N1",
          exclude_features="label",
          **RAPID_TRAINING_KWARGS,
      )


class NodeEmbeddingGraphSamplesWithoutSamplingPlanTest(parameterized.TestCase):
  """Trains on already sampled graph samples without a `sampling_plan`."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.graph, cls.schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=60, num_n2_nodes=40
    )
    cls.sample = synthetic_lib.generate_synthetic_graph(
        cls.schema,
        synthetic_lib.SyntheticGraphConfig(num_nodes=5, num_edges=5),
    )
    cls.samples_path = os.path.join(tempfile.mkdtemp(), "samples@2.tfrecord")
    tf_graph_sample_lib.write_tfgnn_graphs(
        (cls.sample for _ in range(30)),
        cls.samples_path,
        schema=cls.schema,
        container_type="TF_RECORD",
    )
    cls.model = node_embedding_train.train_node_embedding_model(
        graph=cls.samples_path,
        valid_graph=cls.samples_path,
        schema=cls.schema,
        target_nodeset="N1",
        exclude_features=["label"],
        **RAPID_TRAINING_KWARGS,
    )

  def test_no_sampling_plan_and_unknown_seed_counts(self):
    self.assertIsNone(self.model.data().sampling_plan)
    self.assertIsNone(self.model.data().training_stats.num_train_seed_nodes)
    self.assertIsNone(self.model.data().training_stats.num_valid_seed_nodes)

  def test_warning_in_logs(self):
    captured_logs = self.model.metadata.captured_logs
    self.assertIsNotNone(captured_logs)
    warnings = [
        message.text
        for message in (captured_logs or [])
        if message.severity == log.Severity.WARNING
        and "`sampling_plan`" in message.text
    ]
    self.assertNotEmpty(warnings)

  def test_describe(self):
    html_description = self.model.describe()._repr_html_()
    self.assertIn("The sampling plan is not available", html_description)

  def test_predict_and_evaluate_raise(self):
    with self.assertRaisesRegex(ValueError, "does not have a sampling plan"):
      self.model.predict(self.graph, [0, 1, 2], verbose=0)
    with self.assertRaisesRegex(ValueError, "does not have a sampling plan"):
      self.model.evaluate(self.graph, verbose=0)

  def test_predict_on_graph_sample_batch_and_evaluate_generator(self):
    embeddings = self.model.predict_on_graph_sample_batch([self.sample])
    self.assertEqual(embeddings.shape, (1, 8))
    self.assertTrue(np.all(np.isfinite(embeddings)))

    eval_res = self.model.evaluate_generator(
        iter([self.sample] * 4), verbose=0
    )
    self.assertEqual(eval_res.num_examples, 4)

  def test_to_tensorflow_function(self):
    expected = self.model.predict_on_graph_sample_batch([self.sample])
    serialized_samples = tf.constant([
        tf_graph_sample_lib.graph_to_tfgnn_graph(
            self.sample, schema=self.model.data().schema
        ).SerializeToString()
    ])
    tf_embed_fn = self.model.to_tensorflow_function(
        input_format="SERIALIZED_TFGNN_GRAPHS"
    )
    embeddings = tf_embed_fn(serialized_samples)
    np.testing.assert_allclose(embeddings.numpy(), expected, atol=1e-5)

  def test_save_and_load(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      self.model.save(tmpdir)
      restored_model = common_lib.load_model(tmpdir)

    assert isinstance(restored_model, node_embedding_model.NodeEmbeddingModel)
    self.assertIsNone(restored_model.data().sampling_plan)
    with self.assertRaisesRegex(ValueError, "does not have a sampling plan"):
      restored_model.predict(self.graph, [0, 1, 2], verbose=0)

  def test_with_explicit_sampling_plan(self):
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="N1", num_hops=1, hop_width=2, reverse=True
        ),
        self.schema,
    )
    model = node_embedding_train.train_node_embedding_model(
        graph=self.samples_path,
        valid_graph=self.samples_path,
        schema=self.schema,
        target_nodeset="N1",
        exclude_features=["label"],
        sampling_plan=sampling_plan,
        **RAPID_TRAINING_KWARGS,
    )
    test_util.assert_are_equal(self, model.data().sampling_plan, sampling_plan)
    embeddings = model.predict(self.graph, [0, 1, 2], verbose=0)
    self.assertEqual(embeddings.shape, (3, 8))


if __name__ == "__main__":
  absltest.main()
