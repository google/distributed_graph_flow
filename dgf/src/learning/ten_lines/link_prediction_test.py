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

"""Tests for edge prediction."""

import dataclasses
import os
import tempfile
from typing import Literal, Tuple
import unittest.mock
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import tf as tf_io
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.ten_lines import common as common_lib
from dgf.src.learning.ten_lines import link_prediction_model
from dgf.src.learning.ten_lines import link_prediction_train
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.util import filesystem as fs
from dgf.src.util import test_util
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import tensorflow as tf

# Arguments to speed up the train method. Model quality will be poor,
# but training and model inference will run completely.
RAPID_TRAINING_KWARGS = {
    "num_train_steps": 20,
    "valid_every_n_steps": 10,
    "batch_size": 2,
    "sampling_width": 3,
    "num_sampling_hops": 1,
    "max_training_time_seconds": 10,
    "node_embedding_dim": 8,
    "learning_rate": 0.1,
    "num_layers": 2,
}

# Important message for humans and agents
# =======================================
# If runing the unit test repeaterely during development, set TEST_LOCAL_CACHE
# to a local path "/tmp/gf/link_model_cache" and run the test locally (e.g.
# blaze test -c opt --test_strategy=local) for faster iteration: The expensive
# part of the test is the model training which will be cached between test
# calls. Before submitting, set back TEST_LOCAL_CACHE to None.
TEST_LOCAL_CACHE = None


def gen_toy_graph(
    num_nodes_a: int = 500,
    num_nodes_b: int = 500,
    num_categorical_values: int = 200,
    random_seed: int = 42,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Generates a toy dataset for link prediction testing.

  The graph contains two node sets, "A" and "B", and one edge set, "A_to_B".
  Nodes have a numerical feature "f1" and a categorical feature "f2".

  The dataset follows the following edge pattern:
    Nodes A and B can only be connected if:
    1. Their numerical feature "f1" have the same sign.
    2. Their categorical feature "f2" have the exact same value.

  Args:
    num_nodes_a: Number of nodes in node set "A".
    num_nodes_b: Number of nodes in node set "B".
    num_categorical_values: Number of categories for feature "f2".
    random_seed: Seed for reproducibility of features and edge selection.

  Returns:
    A tuple containing the generated InMemoryGraph and its GraphSchema.
  """

  rng = np.random.default_rng(random_seed)

  schema = schema_lib.GraphSchema(
      node_sets={
          "A": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "f1": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                  ),
                  "f2": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_32,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                      num_categorical_values=num_categorical_values,
                  ),
              }
          ),
          "B": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "f1": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                  ),
                  "f2": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_32,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                      num_categorical_values=num_categorical_values,
                  ),
              }
          ),
      },
      edge_sets={
          "A_to_B": schema_lib.EdgeSchema(
              source="A",
              target="B",
              features={},
          ),
      },
  )

  # Features: randomly sampled in [-1, 1]
  f1_a = rng.uniform(-1.0, 1.0, size=num_nodes_a).astype(np.float32)
  f1_b = rng.uniform(-1.0, 1.0, size=num_nodes_b).astype(np.float32)

  # Categorical feature f2: randomly sampled
  f2_a = rng.integers(0, num_categorical_values, size=num_nodes_a).astype(
      np.int32
  )
  f2_b = rng.integers(0, num_categorical_values, size=num_nodes_b).astype(
      np.int32
  )

  # Edges: connect if same sign of 'f1' AND same value of 'f2'
  src_idxs = []
  tgt_idxs = []
  for i in range(num_nodes_a):
    for j in range(num_nodes_b):
      if (f1_a[i] * f1_b[j] > 0) and (f2_a[i] == f2_b[j]):
        src_idxs.append(i)
        tgt_idxs.append(j)

  src_idxs = np.array(src_idxs, dtype=np.int64)
  tgt_idxs = np.array(tgt_idxs, dtype=np.int64)

  # Select a subset of edges
  all_edge_idxs = np.arange(len(src_idxs))
  rng.shuffle(all_edge_idxs)
  num_edges_to_select = min(1000, len(src_idxs))
  selected_edge_idxs = all_edge_idxs[:num_edges_to_select]
  src_idxs = src_idxs[selected_edge_idxs]
  tgt_idxs = tgt_idxs[selected_edge_idxs]

  # Compute and print edge statistics for node A
  degrees = np.bincount(src_idxs, minlength=num_nodes_a)
  logging.info(
      "Node A edge stats: min=%d, max=%d, median=%.1f, mean=%.1f",
      np.min(degrees),
      np.max(degrees),
      np.median(degrees),
      np.mean(degrees),
  )

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "A": in_memory_graph_lib.InMemoryNodeSet(
              features={
                  "f1": f1_a,
                  "f2": f2_a,
                  "#id": np.array(
                      [f"A_{i}".encode() for i in range(num_nodes_a)]
                  ),
              },
              num_nodes=num_nodes_a,
          ),
          "B": in_memory_graph_lib.InMemoryNodeSet(
              features={
                  "f1": f1_b,
                  "f2": f2_b,
                  "#id": np.array(
                      [f"B_{i}".encode() for i in range(num_nodes_b)]
                  ),
              },
              num_nodes=num_nodes_b,
          ),
      },
      edge_sets={
          "A_to_B": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.stack([src_idxs, tgt_idxs], axis=0),
              features={},
          ),
      },
  )
  return graph, schema


def gen_predictible_model_graph():
  schema = schema_lib.GraphSchema(
      node_sets={
          "A": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_32,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "f": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                  ),
              }
          ),
          "B": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_32,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "f": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                  ),
              }
          ),
      },
      edge_sets={
          "A_to_B": schema_lib.EdgeSchema(
              source="A",
              target="B",
              features={},
          ),
      },
  )

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "A": in_memory_graph_lib.InMemoryNodeSet(
              features={
                  "#id": np.array(
                      [
                          0,
                          1,
                          2,
                      ],
                      dtype=np.int32,
                  ),
                  "f": np.array([1, -1, 2], dtype=np.float32),
              },
              num_nodes=3,
          ),
          "B": in_memory_graph_lib.InMemoryNodeSet(
              features={
                  "#id": np.array([0, 1, 2], dtype=np.int32),
                  "f": np.array([1, -1, 2], dtype=np.float32),
              },
              num_nodes=3,
          ),
      },
      edge_sets={
          "A_to_B": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.array([[0, 1, 2, 0], [0, 1, 2, 1]])
          ),
      },
  )
  return graph, schema


class LinkPredictionToyTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.graph, cls.schema = gen_toy_graph()

    def train_model() -> link_prediction_train.LinkPredictionModel:
      # Train model (more steps to learn the pattern)
      return link_prediction_train.train_link_model(
          graph=cls.graph,
          schema=cls.schema,
          target_edgeset="A_to_B",
          # diagnostic_dir="/tmp/gf", # Uncomment to see diagnostics
          **RAPID_TRAINING_KWARGS,
      )

    if TEST_LOCAL_CACHE is None:
      cls.model = train_model()
    else:
      if fs.exists(TEST_LOCAL_CACHE):
        model = common_lib.load_model(TEST_LOCAL_CACHE)
        assert isinstance(model, link_prediction_train.LinkPredictionModel)
        cls.model = model
      else:
        cls.model = train_model()
        cls.model.save(TEST_LOCAL_CACHE)

  def test_interleave_positives_and_negatives(self):
    pos_src = np.array([1, 2])
    pos_trg = np.array([10, 20])
    neg_trg = np.array([[11, 12], [21, 22]])
    full_src, full_trg = (
        link_prediction_model._interleave_positives_and_negatives(
            pos_src, pos_trg, neg_trg
        )
    )

    expected_full_src = np.array([1, 1, 1, 2, 2, 2])
    expected_full_trg = np.array([10, 11, 12, 20, 21, 22])

    test_util.assert_are_equal(self, full_src, expected_full_src)
    test_util.assert_are_equal(self, full_trg, expected_full_trg)

  def test_separate_positives_and_negatives(self):
    probs = np.array([10, 11, 12, 20, 21, 22])
    examples_per_seed_edge = 3

    pos_probs, neg_probs = (
        link_prediction_model._separate_positives_and_negatives(
            probs, examples_per_seed_edge
        )
    )

    expected_pos_probs = np.array([10, 20])
    expected_neg_probs = np.array([11, 12, 21, 22])

    test_util.assert_are_equal(self, pos_probs, expected_pos_probs)
    test_util.assert_are_equal(self, neg_probs, expected_neg_probs)

  def test_model(self):
    self.assertEqual(self.model.data().task.target_edgeset, "A_to_B")

    self.assertAlmostEqual(
        self.model.data().training_stats.num_train_seed_edges, 500, delta=100
    )
    self.assertAlmostEqual(
        self.model.data().training_stats.num_valid_seed_edges, 50, delta=10
    )

  def test_describe(self):
    text_description = repr(self.model.describe())
    html_description = self.model.describe()._repr_html_()  # pylint: disable=attribute-error
    logging.info("Text description:\n%s", text_description)

    for expected in [
        "Objective",
        "Train logs",
        "Schemas",
        "Feature statistics",
        "Padding",
    ]:
      self.assertIn(expected, html_description)

  def test_save_and_load(self):
    # Save and restore the model
    with tempfile.TemporaryDirectory() as tmpdir:
      self.model.save(tmpdir)
      restored_model = link_prediction_train.common.load_model(tmpdir)

    # Check the model class
    assert isinstance(restored_model, link_prediction_train.LinkPredictionModel)

    # Check the model data
    test_util.assert_are_equal(self, self.model.data(), restored_model.data())

    predict_kwards = {
        "graph": self.graph,
        "source_node_idxs": [0, 1],
        "target_node_idxs": [0, 1],
        "all_combinations": False,
    }

    # Check the predictions
    predictions = self.model.predict(**predict_kwards)
    restored_predictions = restored_model.predict(**predict_kwards)
    test_util.assert_are_equal(
        self, predictions, restored_predictions, abs_tol=0.00001
    )

  def test_predict_pairs(self):
    probs = self.model.predict(
        self.graph,
        source_node_idxs=[0, 1],
        target_node_idxs=[0, 1],
        all_combinations=False,
    )
    self.assertEqual(probs.shape, (2,))
    self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

  def test_predict_combinations(self):
    probs_comb = self.model.predict(
        self.graph,
        source_node_idxs=[0, 1],
        target_node_idxs=[0, 1],
        all_combinations=True,
    )
    self.assertEqual(probs_comb.shape, (2, 2))
    self.assertTrue(np.all(probs_comb >= 0.0) and np.all(probs_comb <= 1.0))

  def test_predict_one_to_many(self):
    probs_one_to_many = self.model.predict(
        self.graph,
        source_node_idxs=[0, 1],
        target_node_idxs=[0, 1, 2, 3],
        all_combinations=False,
    )
    self.assertEqual(probs_one_to_many.shape, (4,))

  def test_predict_bad_pairs(self):
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      _ = self.model.predict(
          self.graph,
          source_node_idxs=[0, 1],
          target_node_idxs=[0, 1, 2],
          all_combinations=False,
      )

  def test_predict_mask_edge(self):
    # Find an existing edge
    edge_set = self.graph.edge_sets["A_to_B"]
    src = edge_set.adjacency[0][0]
    trg = edge_set.adjacency[1][0]

    # Prediction with default predict (masks edge)
    prob_masked = self.model.predict(
        self.graph,
        source_node_idxs=[src],
        target_node_idxs=[trg],
        all_combinations=False,
    )

    # Remove the first edge
    new_adjacency = edge_set.adjacency[:, 1:]
    new_edge_set = dataclasses.replace(edge_set, adjacency=new_adjacency)
    new_graph = dataclasses.replace(
        self.graph, edge_sets={"A_to_B": new_edge_set}
    )

    # Prediction on new graph (edge doesn't exist)
    prob_not_exists = self.model.predict(
        new_graph,
        source_node_idxs=[src],
        target_node_idxs=[trg],
        all_combinations=False,
    )

    # They should be equal because predict masks the edge if it exists.
    self.assertAlmostEqual(prob_masked[0], prob_not_exists[0], delta=1e-5)

  def test_predict_multi_batches(self):
    """Make sure predict can split large queries into batches."""

    # Should be split into ~4 batches.
    probs = self.model.predict(
        self.graph,
        source_node_idxs=[0, 1, 3, 4] * self.model.data().hparams.batch_size,
        target_node_idxs=[0, 1, 3, 4] * self.model.data().hparams.batch_size,
        all_combinations=False,
    )
    self.assertEqual(probs.shape, (4 * self.model.data().hparams.batch_size,))
    self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

  def test_predict_predict_batch(self):

    # Should be split into ~4 batches.
    num_batches = 0
    for _ in self.model.predict_batch(
        self.graph,
        source_node_idxs=[0, 1, 3, 4] * self.model.data().hparams.batch_size,
        target_node_idxs=[0, 1, 3, 4] * self.model.data().hparams.batch_size,
        all_combinations=False,
    ):
      num_batches += 1

    self.assertEqual(num_batches, 4)

  def test_predict_batch_insufficient_padding(self):
    """Tests that predict_batch handles InsufficientPaddingError by splitting."""

    original_merge_graphs = link_prediction_model.merge_lib.merge_graphs
    call_count = 0

    def mock_merge_graphs(*args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise link_prediction_model.merge_lib.InsufficientPaddingError(
            "Simulated insufficient padding"
        )
      return original_merge_graphs(*args, **kwargs)

    with unittest.mock.patch.object(
        link_prediction_model.merge_lib,
        "merge_graphs",
        side_effect=mock_merge_graphs,
    ):
      probs = self.model.predict(
          self.graph,
          source_node_idxs=[0, 1],
          target_node_idxs=[0, 1],
          all_combinations=False,
          verbose=0,
      )

      self.assertEqual(probs.shape, (2,))
      self.assertGreater(call_count, 1)

  def test_predict_embedding(self):
    emb_src = self.model.predict_embedding(
        self.graph, node_idxs=[0, 1], encoder="source"
    )
    emb_trg = self.model.predict_embedding(
        self.graph, node_idxs=[0, 1], encoder="target"
    )

    self.assertEqual(
        emb_src.shape, (2, self.model.data().hparams.node_embedding_dim)
    )
    self.assertEqual(
        emb_trg.shape, (2, self.model.data().hparams.node_embedding_dim)
    )

  def test_architecture(self):
    test_util.assert_golden_string(
        self,
        self.model.data().core_model_config.architecture(),
        "link_prediction_architecture.txt",
    )

  @parameterized.parameters(
      {"encoder": "source", "consume_tf_graph_dict": False},
      {"encoder": "source", "consume_tf_graph_dict": True},
      {"encoder": "target", "consume_tf_graph_dict": False},
      {"encoder": "target", "consume_tf_graph_dict": True},
      {"encoder": "both", "consume_tf_graph_dict": False},
      {"encoder": "both", "consume_tf_graph_dict": True},
  )
  def test_to_tensorflow_function(
      self,
      encoder: Literal["source", "target", "both"],
      consume_tf_graph_dict: bool,
  ):
    model = self.model
    graph = self.graph
    schema = model.data().schema

    # Ensure 0 and 1 are not connected to avoid masking issues in test
    edge_set = graph.edge_sets["A_to_B"]
    is_connected = np.any(
        (edge_set.adjacency[0] == 0) & (edge_set.adjacency[1] == 1)
    )
    self.assertFalse(
        is_connected, "Nodes 0 and 1 are connected, choose other nodes for test"
    )

    source_sampler = in_memory_sampler_lib.create_sampler(
        plan=model.data().source_sampling_plan,
        graph=graph,
        schema=schema,
        batch_size=5,
    )
    target_sampler = in_memory_sampler_lib.create_sampler(
        plan=model.data().target_sampling_plan,
        graph=graph,
        schema=schema,
        batch_size=5,
    )

    source_sample = source_sampler.sample(0)
    target_sample = target_sampler.sample(1)

    tf_source_sample = tf_io.graph_to_tf_graph(source_sample, schema=schema)
    tf_target_sample = tf_io.graph_to_tf_graph(target_sample, schema=schema)

    tf_predict_fn = model.to_tensorflow_function(
        encoder=encoder, consume_tf_graph_dict=consume_tf_graph_dict
    )

    if encoder == "source":
      if consume_tf_graph_dict:
        kwargs = {
            **tf_io.tf_graph_to_tf_graph_dict(tf_source_sample),
            "seed_node_idxs": tf.constant([0]),
        }
      else:
        kwargs = {"graph": tf_source_sample, "seed_node_idxs": tf.constant([0])}

      prediction = tf_predict_fn(**kwargs)
      expected_prediction = model.predict_embedding(
          source_sample, [0], encoder="source"
      )
      np.testing.assert_allclose(
          prediction.numpy(), expected_prediction, atol=1e-5
      )

    elif encoder == "target":
      if consume_tf_graph_dict:
        kwargs = {
            **tf_io.tf_graph_to_tf_graph_dict(tf_target_sample),
            "seed_node_idxs": tf.constant([0]),
        }
      else:
        kwargs = {"graph": tf_target_sample, "seed_node_idxs": tf.constant([0])}

      prediction = tf_predict_fn(**kwargs)
      expected_prediction = model.predict_embedding(
          target_sample, [0], encoder="target"
      )
      np.testing.assert_allclose(
          prediction.numpy(), expected_prediction, atol=1e-5
      )

    elif encoder == "both":
      if consume_tf_graph_dict:
        kwargs = {}
        for k, v in tf_io.tf_graph_to_tf_graph_dict(tf_source_sample).items():
          kwargs[f"source_{k}"] = v
        for k, v in tf_io.tf_graph_to_tf_graph_dict(tf_target_sample).items():
          kwargs[f"target_{k}"] = v
        kwargs["source_seed_node_idxs"] = tf.constant([0])
        kwargs["target_seed_node_idxs"] = tf.constant([0])
      else:
        kwargs = {
            "source_graph": tf_source_sample,
            "target_graph": tf_target_sample,
            "source_seed_node_idxs": tf.constant([0]),
            "target_seed_node_idxs": tf.constant([0]),
        }

      prediction = tf_predict_fn(**kwargs)
      expected_prediction = model.predict(
          graph, [0], [1], all_combinations=False, verbose=0
      )
      np.testing.assert_allclose(
          prediction.numpy(), expected_prediction, atol=1e-5
      )

    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
      tf.saved_model.save(tf_predict_fn, tmpdir)
      loaded = tf.saved_model.load(tmpdir)
      loaded_prediction = loaded(**kwargs)
      np.testing.assert_allclose(
          loaded_prediction.numpy(), prediction.numpy(), atol=1e-5
      )

      self.assertIn("serving_default", loaded.signatures)
      signature = loaded.signatures["serving_default"]
      input_names = [t.name for t in signature.inputs]

      if consume_tf_graph_dict:
        h = f"{tf_io.BEGIN_CODE}23{tf_io.END_CODE}"
        u = f"{tf_io.BEGIN_CODE}5f{tf_io.END_CODE}"

        if encoder in ("source", "target"):
          expected_keys_and_shapes = [
              ("nodes_A_reserved_size", ()),
              (f"nodes_A_{h}id", [None]),
              ("nodes_A_f1", [None]),
              ("nodes_A_f2", [None]),
              ("nodes_B_reserved_size", ()),
              (f"nodes_B_{h}id", [None]),
              ("nodes_B_f1", [None]),
              ("nodes_B_f2", [None]),
              (f"edges_A{u}to{u}B_reserved_adjacency", [2, None]),
          ]
          seed_node_idxs_key = "seed_node_idxs"
        elif encoder == "both":
          single_graph_expected = [
              ("nodes_A_reserved_size", ()),
              (f"nodes_A_{h}id", [None]),
              ("nodes_A_f1", [None]),
              ("nodes_A_f2", [None]),
              ("nodes_B_reserved_size", ()),
              (f"nodes_B_{h}id", [None]),
              ("nodes_B_f1", [None]),
              ("nodes_B_f2", [None]),
              (f"edges_A{u}to{u}B_reserved_adjacency", [2, None]),
          ]
          expected_keys_and_shapes = []
          for key, shape in single_graph_expected:
            expected_keys_and_shapes.append((f"source_{key}", shape))
            expected_keys_and_shapes.append((f"target_{key}", shape))

          # Check source and target seed node idxs
          self.assertTrue(
              any("source_seed_node_idxs" in name for name in input_names),
              "source_seed_node_idxs not found in signature inputs:"
              f" {input_names}",
          )
          self.assertTrue(
              any("target_seed_node_idxs" in name for name in input_names),
              "target_seed_node_idxs not found in signature inputs:"
              f" {input_names}",
          )

          # We also need to check their shapes.
          source_seed_tensor = next(
              t for t in signature.inputs if "source_seed_node_idxs" in t.name
          )
          target_seed_tensor = next(
              t for t in signature.inputs if "target_seed_node_idxs" in t.name
          )
          self.assertEqual(source_seed_tensor.shape.as_list(), [None])
          self.assertEqual(target_seed_tensor.shape.as_list(), [None])

          seed_node_idxs_key = None

        if seed_node_idxs_key:
          self.assertTrue(
              any(seed_node_idxs_key in name for name in input_names),
              f"{seed_node_idxs_key} not found in signature inputs:"
              f" {input_names}",
          )
          seed_tensor = next(
              t for t in signature.inputs if seed_node_idxs_key in t.name
          )
          self.assertEqual(seed_tensor.shape.as_list(), [None])

        for key, expected_shape in expected_keys_and_shapes:
          key_lower = key.lower()
          self.assertTrue(
              any(key_lower in name for name in input_names),
              f"Key {key_lower} not found in signature inputs: {input_names}",
          )

          tensor = next(
              (t for t in signature.inputs if key_lower in t.name), None
          )
          self.assertIsNotNone(tensor, f"Tensor for key {key_lower} not found")
          if expected_shape == ():
            self.assertEqual(tensor.shape, ())
          else:
            self.assertEqual(tensor.shape.as_list(), expected_shape)


class LinkPredictionToyStandaloneTest(absltest.TestCase):

  @unittest.skip("The test requires graphviz")
  def test_diagnostic_dir(self):
    graph, schema = gen_toy_graph()
    diagnostic_dir = self.create_tempdir().full_path
    _ = link_prediction_train.train_link_model(
        graph=graph,
        schema=schema,
        target_edgeset="A_to_B",
        diagnostic_dir=diagnostic_dir,
        **RAPID_TRAINING_KWARGS,
    )
    for filename in [
        "negative_target_graph_0.png",
        "positive_source_graph_0.png",
        "positive_target_graph_0.png",
    ]:
      self.assertTrue(fs.exists(os.path.join(diagnostic_dir, filename)))


class LinkPredictionGraphAttentionNetworkToyTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.graph, cls.schema = gen_toy_graph()

    cls.model = link_prediction_train.train_link_model(
        graph=cls.graph,
        schema=cls.schema,
        target_edgeset="A_to_B",
        architecture="heterogeneous_graph_attention_network",
        # diagnostic_dir="/tmp/gf", # Uncomment to see diagnostics
        **RAPID_TRAINING_KWARGS,
    )

  def test_predict_embedding(self):
    emb_src = self.model.predict_embedding(
        self.graph, node_idxs=[0, 1], encoder="source"
    )
    emb_trg = self.model.predict_embedding(
        self.graph, node_idxs=[0, 1], encoder="target"
    )

    self.assertEqual(
        emb_src.shape, (2, self.model.data().hparams.node_embedding_dim)
    )
    self.assertEqual(
        emb_trg.shape, (2, self.model.data().hparams.node_embedding_dim)
    )

  def test_architecture(self):
    test_util.assert_golden_string(
        self,
        self.model.data().core_model_config.architecture(),
        "link_prediction_hetero_gat_architecture.txt",
    )


# This class creates and tests a simple and fully predictable model: The model
# output is equal to the input "embedding" feature.
class LinkPredictionPredictibleModelTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.graph, cls.schema = gen_predictible_model_graph()

    def preprocess_core_model_config(c):
      return dataclasses.replace(
          c,
          encoder_config=dataclasses.replace(
              c.encoder_config,
              pre_mlp=standard.identity(),
          ),
      )

    cls.model = link_prediction_train.train_link_model(
        graph=cls.graph,
        schema=cls.schema,
        target_edgeset="A_to_B",
        num_train_steps=4,
        valid_every_n_steps=2,
        batch_size=2,
        sampling_width=1,
        num_sampling_hops=0,
        node_embedding_dim=1,
        learning_rate=0.1,
        verbose=3,
        num_negative_nodes=2,
        num_layers=0,
        experimental_preprocess_core_model_config=preprocess_core_model_config,
    )

    # Set all the parameter values to zero. Only the residual are remaining.
    cls.model.data().model_params = jax.tree_util.tree_map(
        jnp.zeros_like, cls.model.data().model_params
    )

  def test_model(self):
    self.assertEqual(self.model.data().training_stats.num_train_seed_edges, 2)
    self.assertEqual(self.model.data().training_stats.num_valid_seed_edges, 2)

  def test_predict(self):
    predictions = self.model.predict(
        self.graph, source_node_idxs=[0, 1], target_node_idxs=[0, 1, 1, 2]
    )
    test_util.assert_are_equal(
        self,
        jsp.logit(predictions),
        jnp.array([1.0, -1.0, 1.0, -2.0]),
        abs_tol=0.001,
    )

  def test_predict_combinations(self):
    predictions = self.model.predict(
        self.graph,
        source_node_idxs=[0, 1, 2],
        target_node_idxs=[0, 1, 2],
        all_combinations=True,
    )
    test_util.assert_are_equal(
        self,
        jsp.logit(predictions),
        jnp.array([
            [1.0, -1.0, 2.0],
            [-1.0, 1.0, -2.0],
            [2.0, -2.0, 4.0],
        ]),
        abs_tol=0.001,
    )

  def test_predict_embedding(self):
    emb_src = self.model.predict_embedding(
        self.graph, node_idxs=[0, 1], encoder="source"
    )
    emb_trg = self.model.predict_embedding(
        self.graph, node_idxs=[0, 1], encoder="target"
    )

    test_util.assert_are_equal(
        self, emb_src, np.array([[1.0], [-1.0]]), abs_tol=0.001
    )
    test_util.assert_are_equal(
        self, emb_trg, np.array([[1.0], [-1.0]]), abs_tol=0.001
    )

  def test_evaluate(self):
    evaluation = self.model.evaluate(self.graph, num_negative_nodes=5)
    logging.info("evaluation:\n%s", evaluation)
    self.assertEqual(evaluation.num_examples, 4)
    # Note: The random negative sampling make the evaluation non deterministic.


if __name__ == "__main__":
  absltest.main()
