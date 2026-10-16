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

import os
import tempfile
import unittest.mock

from absl.testing import absltest
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import graphmae
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.ten_lines import common as common_lib
from dgf.src.learning.ten_lines import dataset
from dgf.src.learning.ten_lines import node_embedding_core_model
from dgf.src.learning.ten_lines import node_embedding_model
from dgf.src.learning.ten_lines import node_prediction_dataset
from dgf.src.learning.ten_lines import node_prediction_model
from dgf.src.util import gen_test_graph
import jax
import numpy as np


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

  def test_predict_embedding_concatenates_batches(self):
    with unittest.mock.patch.object(
        self.model,
        "predict_batch",
        return_value=iter([self._batch([0, 1]), self._batch([2])]),
    ) as predict_batch:
      embeddings = self.model.predict_embedding(
          self.graph, [0, 1, 2], verbose=0
      )
    np.testing.assert_array_equal(
        predict_batch.call_args.args[1], np.array([0, 1, 2], dtype=np.int32)
    )
    self.assertEqual(embeddings.shape, (3, 8))
    np.testing.assert_array_equal(embeddings[:, 0], [0.0, 0.0, 2.0])

  def test_predict_embedding_defaults_to_all_target_nodes(self):
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

  def test_predict_embedding_empty_node_idxs(self):
    with unittest.mock.patch.object(
        self.model, "predict_batch"
    ) as predict_batch:
      embeddings = self.model.predict_embedding(self.graph, [], verbose=0)
    predict_batch.assert_not_called()
    self.assertEqual(embeddings.shape, (0, 8))
    self.assertEqual(embeddings.dtype, np.float32)

  def test_predict_embedding_rejects_both_node_idxs_and_seed_node_idxs(self):
    with self.assertRaisesRegex(ValueError, "at most one of"):
      self.model.predict_embedding(
          self.graph, [0], seed_node_idxs=[0], verbose=0
      )

  def test_without_sampling_plan_raises(self):
    self.data.sampling_plan = None
    with self.assertRaisesRegex(ValueError, "does not have a sampling plan"):
      next(self.model.predict_batch(self.graph, [0], verbose=0))
    with self.assertRaisesRegex(ValueError, "sampling_plan` is None"):
      self.model.evaluate(self.graph, verbose=0)

  def test_extract_serving_schemata_requires_uuid(self):
    self.model.metadata.uuid = None
    with self.assertRaisesRegex(ValueError, "Model UUID is not set"):
      self.model._extract_serving_schemata()


def _untrained_model(
    graph: common_lib.Graph,
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
  hparams = node_embedding_model.HParam(
      batch_size=4,
      node_embedding_dim=dims,
      num_layers=1,
      num_sampling_hops=1,
      sampling_width=30,
      padding_margin=2.0,
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
  )
  live = train_dataset.get_live()

  encoder_conv = hetero_gnn.HeterogeneousGraphConvolutionConfig(
      dims=dims, message_pooling="sum"
  )
  decoder_conv = hetero_gnn.HeterogeneousGraphConvolutionConfig(
      dims=dims, embedding_feature="latent_embedding", message_pooling="sum"
  )
  core_model_config = node_embedding_core_model.CoreModelConfig(
      embed_graph=preprocess.EmbedGraphConfig(),
      pre_mlp=standard.ingest_feature(dims),
      graphmae=graphmae.GraphMAEConfig(
          encoder_conv=encoder_conv,
          num_encoder_layers=1,
          decoder_conv=decoder_conv,
          target_nodeset="N1",
      ),
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


class NodeEmbeddingModelUntrainedTest(absltest.TestCase):
  """End-to-end tests of `NodeEmbeddingModel` with random weights."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.graph, schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=60, num_n2_nodes=40
    )
    del schema.node_sets["N1"].features["label"]
    cls.model = _untrained_model(cls.graph, schema)

  def test_predict_embedding(self):
    emb_subset = self.model.predict_embedding(self.graph, [0, 1, 2], verbose=0)
    self.assertEqual(emb_subset.shape, (3, 8))
    self.assertEqual(emb_subset.dtype, np.float32)
    self.assertTrue(np.all(np.isfinite(emb_subset)))

    num_n1 = self.graph.node_sets["N1"].num_nodes
    emb_all = self.model.predict_embedding(self.graph, verbose=0)
    self.assertEqual(emb_all.shape, (num_n1, 8))
    np.testing.assert_allclose(emb_subset, emb_all[:3], atol=1e-5)

  def test_evaluate(self):
    eval_res = self.model.evaluate(
        self.graph, seed_node_idxs=[0, 1, 2, 3], verbose=0
    )
    self.assertEqual(eval_res.num_examples, 4)
    assert eval_res.loss is not None
    self.assertTrue(np.isfinite(eval_res.loss))
    for metric_key in ("recon_loss", "var_loss", "cov_loss", "total_loss"):
      self.assertTrue(np.isfinite(eval_res.user_metrics[metric_key]))

  def test_evaluate_without_examples_raises(self):
    with self.assertRaisesRegex(ValueError, "No examples to evaluate"):
      self.model.evaluate_generator(iter([]), verbose=0)

  def test_describe(self):
    html_str = self.model.describe()._repr_html_()
    for section in (
        "Objective",
        "Unsupervised Node Embedding (GraphMAE)",
        "Schema",
        "Architecture",
        "Padding",
    ):
      self.assertIn(section, html_str)

  def test_save_and_load(self):
    expected = self.model.predict_embedding(self.graph, [0, 1, 2], verbose=0)
    with tempfile.TemporaryDirectory() as tmpdir:
      save_dir = os.path.join(tmpdir, "model")
      self.model.save(save_dir)
      loaded = common_lib.load_model(save_dir)
    assert isinstance(loaded, node_embedding_model.NodeEmbeddingModel)
    self.assertEqual(loaded.target_nodeset, "N1")
    np.testing.assert_allclose(
        loaded.predict_embedding(self.graph, [0, 1, 2], verbose=0),
        expected,
        atol=1e-6,
    )


if __name__ == "__main__":
  absltest.main()
