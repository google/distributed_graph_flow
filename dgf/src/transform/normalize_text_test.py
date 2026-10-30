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

import copy
from unittest import mock

from absl.testing import absltest
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.io import tf as tf_io
from dgf.src.transform import normalize as normalize_lib
from dgf.src.transform import normalize_text
from dgf.src.transform import normalize_text_test_util
import keras_hub
import numpy as np
import tensorflow as tf


def _save_dummy_keras_hub_preset(preset_dir: str, dim: int = 8) -> None:
  """Saves a tiny KerasHub BERT preset into `preset_dir`."""
  vocab = [
      "[PAD]",
      "[UNK]",
      "[CLS]",
      "[SEP]",
      "[MASK]",
      "hello",
      "world",
      "graph",
  ]
  tokenizer = keras_hub.models.BertTokenizer(vocabulary=vocab, lowercase=True)
  preprocessor = keras_hub.models.BertTextClassifierPreprocessor(
      tokenizer=tokenizer, sequence_length=8
  )
  backbone = keras_hub.models.BertBackbone(
      vocabulary_size=len(vocab),
      num_layers=1,
      num_heads=2,
      hidden_dim=dim,
      intermediate_dim=dim * 2,
      max_sequence_length=8,
  )
  preprocessor.save_to_preset(preset_dir)
  backbone.save_to_preset(preset_dir)


class NormalizeTextTest(absltest.TestCase):

  def test_dim_mismatch_raises(self):
    hub_dir = self.create_tempdir("dummy_hub_dim").full_path
    normalize_text_test_util.save_dummy_tf_hub_embedder(hub_dir, dim=8)
    normalizer = normalize_text.TextEmbeddingNormalizer(
        input_feature="title",
        output_feature_name="title_TEXT_EMBEDDING",
        output_shape=(4,),
        embedder_handle=hub_dir,
    )

    with self.assertRaisesRegex(ValueError, "dimension 8, expected 4"):
      _ = normalizer.embedder

  def test_create_and_use_loads_model_once(self):
    hub_dir = self.create_tempdir("dummy_hub_once").full_path
    normalize_text_test_util.save_dummy_tf_hub_embedder(hub_dir, dim=8)
    feature_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.TEXT,
        is_utf8_string=True,
    )

    with mock.patch.object(
        normalize_text,
        "LoadedTextEmbedder",
        wraps=normalize_text.LoadedTextEmbedder,
    ) as loaded_text_embedder:
      normalizer = normalize_text.TextEmbeddingNormalizer.create(
          "title", feature_schema, embedder=hub_dir
      )
      _ = normalizer.embedder

    loaded_text_embedder.assert_called_once()

  def test_graph_normalizer_shares_embedder_across_sets(self):
    hub_dir = self.create_tempdir("dummy_hub_sets").full_path
    normalize_text_test_util.save_dummy_tf_hub_embedder(hub_dir, dim=8)
    feature_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.TEXT,
        is_utf8_string=True,
    )
    node_normalizer = normalize_text.TextEmbeddingNormalizer.create(
        "title", feature_schema, embedder=hub_dir
    )
    edge_normalizer = normalize_text.TextEmbeddingNormalizer.create(
        "label", feature_schema, embedder=hub_dir
    )
    graph_normalizer = normalize_lib.GraphNormalizerConfig(
        nodesets={
            "nodes": normalize_lib.NodeSetNormalizerConfig(
                normalizers=[node_normalizer]
            )
        },
        edgesets={
            "edges": normalize_lib.EdgeSetNormalizerConfig(
                source="nodes", target="nodes", normalizers=[edge_normalizer]
            )
        },
    ).make()

    self.assertLen(graph_normalizer.tensorflow_resources(), 1)
    self.assertIs(node_normalizer.embedder, edge_normalizer.embedder)

  def test_deepcopy_shares_embedder(self):
    hub_dir = self.create_tempdir("dummy_hub_copy").full_path
    normalize_text_test_util.save_dummy_tf_hub_embedder(hub_dir, dim=8)
    normalizer = normalize_text.TextEmbeddingNormalizer.create(
        "title",
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BYTES,
            semantic=schema_lib.FeatureSemantic.TEXT,
            is_utf8_string=True,
        ),
        embedder=hub_dir,
    )

    self.assertIs(
        copy.deepcopy(normalizer).embedder, normalizer.embedder
    )

  def test_builtin_registry(self):
    nnlm_cfg = normalize_text.resolve_text_embedder_config("nnlm-en-dim128")
    self.assertEqual(nnlm_cfg.dim, 128)
    self.assertEqual(
        nnlm_cfg.backend, normalize_text.TextEmbedderBackend.TF_HUB
    )
    bert_cfg = normalize_text.resolve_text_embedder_config(
        "bert-tiny-en-dim128"
    )
    self.assertEqual(bert_cfg.dim, 128)
    self.assertEqual(
        bert_cfg.backend, normalize_text.TextEmbedderBackend.KERAS_HUB
    )

  def test_tf_hub_embedder_numpy_tf_and_saved_model_dedup(self):
    hub_dir = self.create_tempdir("dummy_tf_hub").full_path
    normalize_text_test_util.save_dummy_tf_hub_embedder(hub_dir, dim=8)

    cfg = normalize_text.resolve_text_embedder_config(hub_dir)
    self.assertEqual(cfg.dim, 8)
    self.assertEqual(cfg.backend, normalize_text.TextEmbedderBackend.TF_HUB)

    emb1 = normalize_text.LoadedTextEmbedder(cfg)

    texts = np.array([b"hello world", b"graph neural network", b"hello world"])
    np_out = emb1.embed_numpy(texts)
    self.assertEqual(np_out.shape, (3, 8))
    self.assertEqual(np_out.dtype, np.float32)
    np.testing.assert_allclose(np_out[0], np_out[2])

    texts_2d = texts.reshape(3, 1)
    np_out_2d = emb1.embed_numpy(texts_2d)
    self.assertEqual(np_out_2d.shape, (3, 1, 8))

    class TwoFeatureExportModule(tf.Module):

      def __init__(self, e1, e2):
        super().__init__()
        self._resources = [e1, e2]
        self._e1 = e1
        self._e2 = e2

      @tf.function(
          input_signature=[
              tf.TensorSpec(shape=[None], dtype=tf.string),
              tf.TensorSpec(shape=[None], dtype=tf.string),
          ]
      )
      def __call__(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return self._e1.embed_tensorflow(a) + self._e2.embed_tensorflow(b)

    export_dir = self.create_tempdir("exported_two_features").full_path
    # Two features sharing the same model export a single copy of the weights.
    wrapper = TwoFeatureExportModule(emb1, emb1)
    tf.saved_model.save(wrapper, export_dir)
    reloaded = tf.saved_model.load(export_dir)

    self.assertLen(reloaded.signatures["serving_default"].variables, 1)
    tf_out = reloaded(tf.constant(texts), tf.constant(texts)).numpy()
    np.testing.assert_allclose(tf_out, np_out * 2.0, rtol=1e-5, atol=1e-5)

  def test_embed_numpy_string_types(self):
    hub_dir = self.create_tempdir("dummy_tf_hub_types").full_path
    normalize_text_test_util.save_dummy_tf_hub_embedder(hub_dir, dim=4)
    embedder = normalize_text.LoadedTextEmbedder(
        normalize_text.resolve_text_embedder_config(hub_dir)
    )

    expected = embedder.embed_numpy(np.array([[b"a", b"b"], [b"c", b"a"]]))
    self.assertEqual(expected.shape, (2, 2, 4))
    np.testing.assert_array_equal(expected[0, 0], expected[1, 1])
    # Same result with unicode and object arrays.
    np.testing.assert_array_equal(
        embedder.embed_numpy(np.array([["a", "b"], ["c", "a"]])), expected
    )
    np.testing.assert_array_equal(
        embedder.embed_numpy(
            np.array([["a", b"b"], [b"c", "a"]], dtype=np.object_)
        ),
        expected,
    )
    empty = embedder.embed_numpy(np.array([], np.bytes_))
    self.assertEqual(empty.shape, (0, 4))

  def test_embed_numpy_utf8_bytes(self):
    hub_dir = self.create_tempdir("dummy_tf_hub_utf8").full_path
    normalize_text_test_util.save_dummy_tf_hub_embedder(hub_dir, dim=4)
    embedder = normalize_text.LoadedTextEmbedder(
        normalize_text.resolve_text_embedder_config(hub_dir)
    )

    texts = ["café", "日本語のテキスト", "naïve 🙂"]
    expected = embedder.embed_numpy(
        np.array([t.encode("utf-8") for t in texts], dtype=np.object_)
    )
    self.assertEqual(expected.shape, (3, 4))
    np.testing.assert_array_equal(
        embedder.embed_numpy(np.array(texts)), expected
    )
    np.testing.assert_array_equal(
        embedder.embed_numpy(np.array(texts, dtype=np.object_)), expected
    )

  def test_keras_hub_embedder_numpy_tf_and_saved_model(self):
    preset_dir = self.create_tempdir("dummy_keras_hub").full_path
    _save_dummy_keras_hub_preset(preset_dir, dim=8)

    cfg = normalize_text.resolve_text_embedder_config(preset_dir)
    self.assertEqual(cfg.dim, 8)
    self.assertEqual(cfg.backend, normalize_text.TextEmbedderBackend.KERAS_HUB)

    emb = normalize_text.LoadedTextEmbedder(cfg)
    texts = np.array([b"hello world", b"graph world", b"hello world"])
    np_out = emb.embed_numpy(texts)
    self.assertEqual(np_out.shape, (3, 8))
    np.testing.assert_allclose(np_out[0], np_out[2], rtol=1e-5, atol=1e-5)

    class SingleFeatureExportModule(tf.Module):

      def __init__(self, embedder):
        super().__init__()
        self._embedder = embedder

      @tf.function(
          input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)]
      )
      def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return self._embedder.embed_tensorflow(x)

    export_dir = self.create_tempdir("exported_keras_hub").full_path
    tf.saved_model.save(SingleFeatureExportModule(emb), export_dir)
    reloaded = tf.saved_model.load(export_dir)
    tf_out = reloaded(tf.constant(texts)).numpy()
    np.testing.assert_allclose(tf_out, np_out, rtol=1e-5, atol=1e-5)

  def test_text_embedding_normalizer_and_auto_normalize(self):
    hub_dir = self.create_tempdir("dummy_hub_norm").full_path
    normalize_text_test_util.save_dummy_tf_hub_embedder(hub_dir, dim=8)
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "title": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.TEXT,
                        is_utf8_string=True,
                    ),
                    "abstract": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.TEXT,
                        is_utf8_string=True,
                    ),
                }
            )
        },
        edge_sets={},
    )
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "title": statistics_lib.FeatureStatistics(count=2),
                    "abstract": statistics_lib.FeatureStatistics(count=2),
                }
            )
        }
    )
    normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(text_embedder=hub_dir),
    )

    out_schema = normalizer.output_schema()
    node_features = out_schema.node_sets["nodes"].features
    self.assertIn("title_TEXT_EMBEDDING", node_features)
    self.assertIn("abstract_TEXT_EMBEDDING", node_features)
    title_schema = node_features["title_TEXT_EMBEDDING"]
    self.assertEqual(title_schema.format, schema_lib.FeatureFormat.FLOAT_32)
    self.assertEqual(
        title_schema.semantic, schema_lib.FeatureSemantic.EMBEDDING
    )
    self.assertEqual(title_schema.shape, (8,))

    # JSON serialization only stores the handle, not weights.
    json_str = normalizer.config.to_json()  # pyrefly: ignore[missing-attribute]
    self.assertIn(hub_dir, json_str)
    restored = normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
        json_str
    ).make()

    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "title": np.array([b"first title", b"second title"]),
                    "abstract": np.array(
                        [b"first abstract", b"second abstract"]
                    ),
                },
            )
        },
        edge_sets={},
    )
    np_out = restored.normalize_numpy(graph)
    tf_out = restored.normalize_tensorflow(tf_io.graph_to_tf_graph(graph))
    np.testing.assert_allclose(
        np_out.node_sets["nodes"].features["title_TEXT_EMBEDDING"],
        tf_out.node_sets["nodes"].features["title_TEXT_EMBEDDING"].numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    # Both TEXT features share a single TF resource instance.
    self.assertLen(restored.tensorflow_resources(), 1)
    # Serialization after running normalization still only stores the handle.
    self.assertIn(hub_dir, restored.config.to_json())  # pyrefly: ignore[missing-attribute]
    nice_str = restored.config.nice_print(return_output=True)
    self.assertIn("TextEmbeddingNormalizer", nice_str)
    self.assertIn("dim=8", nice_str)


if __name__ == "__main__":
  absltest.main()
