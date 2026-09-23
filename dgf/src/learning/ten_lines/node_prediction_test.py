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

"""Test for the basic buildable config class."""

import copy
import dataclasses
import os
import tempfile
import unittest
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.generate import graphs as synthetic_lib
from dgf.src.io import tf as tf_io
from dgf.src.io import tf_graph_sample as tf_graph_sample_lib
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.ten_lines import common as common_lib
from dgf.src.learning.ten_lines import node_prediction_model
from dgf.src.learning.ten_lines import node_prediction_train as node_prediction_lib
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.util import filesystem as fs
from dgf.src.util import gen_test_graph
from dgf.src.util import log
from dgf.src.util import test_util
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

# Arguments to speed up the train method. Model quality will be poor,
# but training and model inference will run completely.
RAPID_TRAINING_KWARGS = {
    "num_train_steps": 20,
    "valid_every_n_steps": 10,
    "batch_size": 5,
    "sampling_width": 5,
    "num_sampling_hops": 1,
    "max_training_time_seconds": 5,
    "node_embedding_dim": 8,
    "num_layers": 2,
}

# Important message for humans and agents
# =======================================
# If runing the unit test repeaterely during development, set TEST_LOCAL_CACHE
# to a local path "/tmp/gf/node_model_cache" and run the test locally (e.g.
# blaze test -c opt --test_strategy=local) for faster iteration: The expensive
# part of the test isthe model training which will be cached between test calls.
# Before submitting, set back TEST_LOCAL_CACHE to None.
TEST_LOCAL_CACHE = None


def _sampling_plan(
    model: node_prediction_lib.NodePredictionModel,
) -> sampling_config_lib.SamplingPlan:
  """Returns the sampling plan of a model. Fails if it is not available."""
  sampling_plan = model.data().sampling_plan
  assert sampling_plan is not None
  return sampling_plan


def _gen_graph_real_looking(
    has_timestamp_feature: bool = False,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Generate a toy dataset with real looking features, but without patterns."""
  edge_features = {}
  if has_timestamp_feature:
    edge_features["timestamp"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_creation_time=True,
    )

  schema = schema_lib.GraphSchema(
      node_sets={
          "client": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "city": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                      num_categorical_values=10,
                  ),
                  "age": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
                  "created_at": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                      is_creation_time=True,
                  ),
                  "categorical_label": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                      num_categorical_values=2,
                  ),
              }
          ),
          "transaction": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "date": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                      is_creation_time=True,
                  ),
                  "amount": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
                  "country": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                  ),
              }
          ),
      },
      edge_sets={
          "transation_to_client": schema_lib.EdgeSchema(
              source="transaction",
              target="client",
              features=edge_features,
          ),
      },
  )
  return (
      synthetic_lib.generate_synthetic_graph(
          schema,
          synthetic_lib.SyntheticGraphConfig(num_nodes=1000, num_edges=2000),
      ),
      schema,
  )


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
                  "label": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          ),
      },
      edge_sets={},
  )

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "A": in_memory_graph_lib.InMemoryNodeSet(
              features={
                  "#id": np.array([0, 1, 2], dtype=np.int32),
                  "f": np.array([1, -1, 2], dtype=np.float32),
                  "label": np.array([1, -1, 2], dtype=np.float32),
              },
              num_nodes=3,
          ),
      },
      edge_sets={},
  )
  return graph, schema


def _single_sample_padding(
    model: node_prediction_lib.NodePredictionModel,
    graph: in_memory_graph_lib.InMemoryGraph,
    seed_node_idxs: list[int],
    num_draws: int = 5,
) -> padding_lib.Padding:
  """Returns a padding that fits a single sample, but not a merged pair.

  The padding is the largest sample size observed over `num_draws` independent
  samplings of each seed node (plus one for the sentinel node / edge). Merging
  two samples therefore overflows it, which forces "predict_batch" to split its
  batches.

  Args:
    model: The model whose sampling plan and schema are used for sampling.
    graph: The graph to sample from.
    seed_node_idxs: The seed nodes the padding must accommodate.
    num_draws: How many times each seed node is sampled. Sampling is random, so
      several draws make the observed maximum more stable.

  Returns:
    A padding fitting the largest observed single sample.
  """
  data = model.data()
  schema = node_prediction_model.schema_to_input_feature_schema(
      data.schema, data.task
  )
  sampler = in_memory_sampler_lib.create_sampler(
      graph=graph,
      plan=_sampling_plan(model),
      schema=schema,
      batch_size=len(seed_node_idxs),
  )
  samples = []
  for _ in range(num_draws):
    samples.extend(sampler.sample(np.asarray(seed_node_idxs)))

  node_sets = {}
  for name in schema.node_sets:
    max_num_nodes = max((s.node_sets[name].num_nodes or 0) for s in samples)
    node_sets[name] = padding_lib.NodeSetPadding(num_nodes=max_num_nodes + 1)

  edge_sets = {}
  for name in schema.edge_sets:
    max_num_edges = max(s.edge_sets[name].num_edges() for s in samples)
    edge_sets[name] = padding_lib.EdgeSetPadding(num_edges=max_num_edges + 1)

  return padding_lib.Padding(node_sets=node_sets, edge_sets=edge_sets)


class NodePredictionRealLookingGraphAttentionNetwork(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    # Generate data
    cls.graph, cls.schema = _gen_graph_real_looking()

    cls.model = node_prediction_lib.train_node_model(
        graph=cls.graph,
        schema=cls.schema,
        target_nodeset="client",
        target_column="categorical_label",
        architecture="heterogeneous_graph_attention_network",
        **RAPID_TRAINING_KWARGS,
    )

  def test_predict(self):
    predictions = self.model.predict(graph=self.graph, seed_node_idxs=[0, 1, 2])
    self.assertEqual(predictions.shape, (3, self.model.num_label_classes()))
    self.assertTrue(np.allclose(np.sum(predictions, axis=1), 1.0))

  def test_evaluate(self):
    evaluation = self.model.evaluate(self.graph)
    self.assertEqual(
        evaluation.num_examples,
        self.graph.node_sets["client"].num_nodes,
    )

  def test_architecture(self):
    test_util.assert_golden_string(
        self,
        self.model.data().core_model_config.architecture(),
        "node_prediction_hetero_gat_architecture.txt",
    )


class NodePredictionRealLooking(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    # Generate data
    cls.graph, cls.schema = _gen_graph_real_looking()

    def train_model() -> node_prediction_lib.NodePredictionModel:
      # Train a basic model
      return node_prediction_lib.train_node_model(
          graph=cls.graph,
          schema=cls.schema,
          target_nodeset="client",
          target_column="categorical_label",
          **RAPID_TRAINING_KWARGS,
      )

    if TEST_LOCAL_CACHE is None:
      cls.model = train_model()
    else:
      if fs.exists(TEST_LOCAL_CACHE):
        model = common_lib.load_model(TEST_LOCAL_CACHE)
        assert isinstance(model, node_prediction_lib.NodePredictionModel)
        cls.model = model
      else:
        cls.model = train_model()
        cls.model.save(TEST_LOCAL_CACHE)

  def test_describe(self):
    text_descriptoin = repr(self.model.describe())
    html_description = self.model.describe()._repr_html_()  # pylint: disable=attribute-error

    logging.info("Text description:\n%s", text_descriptoin)
    # TODO(gbm): Golden test describe output.
    for expected in [
        "Objective",
        "Training",
        "Schema",
        "Feature statistics",
        "Graph sampling",
    ]:
      self.assertIn(expected, html_description)

    self.assertIsNotNone(self.model.metadata.captured_logs)
    self.assertIsInstance(self.model.metadata.captured_logs, list)

  def test_predict(self):
    predictions = self.model.predict(graph=self.graph, seed_node_idxs=[0, 1, 2])
    self.assertEqual(predictions.shape, (3, self.model.num_label_classes()))
    self.assertTrue(np.allclose(np.sum(predictions, axis=1), 1.0))

  def test_predict_batch_splits_oversized_batches(self):
    seed_node_idxs = [0, 1, 2, 3]
    data = self.model.data()
    original_padding = data.padding
    data.padding = _single_sample_padding(
        self.model, self.graph, seed_node_idxs
    )
    try:
      batches = list(
          self.model.predict_batch(
              graph=self.graph,
              seed_node_idxs=seed_node_idxs,
              input_features_only=True,
              verbose=0,
          )
      )
    finally:
      data.padding = original_padding

    # The padding only fits a single sample, so the batches are recursively
    # split down to individual samples instead of raising.
    self.assertLen(batches, len(seed_node_idxs))
    for batch in batches:
      self.assertIsInstance(batch, node_prediction_model.BatchPrediction)
      self.assertLen(batch.batch_seed_node_idxs, 1)
      self.assertLen(batch.predictions, 1)
    self.assertEqual(
        [int(batch.batch_seed_node_idxs[0]) for batch in batches],
        seed_node_idxs,
    )

  def test_predict_without_labels(self):
    graph_without_labels = copy.deepcopy(self.graph)
    del graph_without_labels.node_sets["client"].features["categorical_label"]
    predictions = self.model.predict(
        graph=graph_without_labels, seed_node_idxs=[0, 1, 2]
    )
    self.assertEqual(predictions.shape, (3, self.model.num_label_classes()))
    self.assertTrue(np.allclose(np.sum(predictions, axis=1), 1.0))

  def test_save_and_load(self):

    # Save and restore the model
    with tempfile.TemporaryDirectory() as tmpdir:
      self.model.save(tmpdir)
      restored_model = common_lib.load_model(tmpdir)

    # Check the model class
    assert isinstance(restored_model, node_prediction_lib.NodePredictionModel)

    # Check the model UUID
    self.assertIsNotNone(self.model.metadata.uuid)
    # The UUID is restored.
    self.assertEqual(self.model.metadata.uuid, restored_model.metadata.uuid)
    # The UUID is different than one for a new model.
    self.assertNotEqual(
        self.model.metadata.uuid, common_lib.Metadata(name="test").uuid
    )

    # Check the model data
    test_util.assert_are_equal(self, self.model.data(), restored_model.data())  # pylint: disable=protected-access

    # Check the model live resources
    test_util.assert_are_equal(
        self, self.model._get_live().normalized_schema, restored_model._get_live().normalized_schema  # pylint: disable=protected-access
    )

    # Check the predictions (relaxed check to handle non-deterministic sampling)
    predictions = self.model.predict(graph=self.graph, seed_node_idxs=[0, 1, 2])
    restored_predictions = restored_model.predict(
        graph=self.graph, seed_node_idxs=[0, 1, 2]
    )
    self.assertEqual(predictions.shape, restored_predictions.shape)
    self.assertTrue(np.allclose(np.sum(restored_predictions, axis=1), 1.0))

    # Check that predictions on a pre-sampled graph are exactly equal
    sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=_sampling_plan(self.model),
        schema=self.model.data().schema,
        batch_size=5,
    )
    sample = sampler.sample(0)
    predictions_on_sample = self.model.predict(graph=sample, seed_node_idxs=[0])
    restored_predictions_on_sample = restored_model.predict(
        graph=sample, seed_node_idxs=[0]
    )
    np.testing.assert_allclose(
        predictions_on_sample,
        restored_predictions_on_sample,
        rtol=1e-5,
        atol=1e-5,
    )

  @parameterized.parameters(
      {"input_format": "TF_GRAPH"},
      {"input_format": "TF_GRAPH_DICT"},
  )
  def test_to_tensorflow_function(self, input_format: str):
    consume_tf_graph_dict = input_format == "TF_GRAPH_DICT"

    sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=_sampling_plan(self.model),
        schema=self.model.data().schema,
        batch_size=5,
    )
    # Note: We use different samples to test that the traced/frozen model
    # can run on samples with different sizes.
    sample_1 = sampler.sample(0)
    sample_2 = sampler.sample(1)

    tf_sample_1 = tf_io.graph_to_tf_graph(
        sample_1, schema=self.model.data().schema
    )
    tf_sample_2 = tf_io.graph_to_tf_graph(
        sample_2, schema=self.model.data().schema
    )

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

    tf_predict_fn = self.model.to_tensorflow_function(input_format=input_format)

    prediction_sample_1 = tf_predict_fn(**kwargs_call_1)  # pyrefly: ignore[not-callable]

    # Expected prediction
    expected_prediction_sample_1 = self.model.predict(sample_1, [0])
    expected_prediction_sample_2 = self.model.predict(sample_2, [0])

    np.testing.assert_allclose(
        prediction_sample_1.numpy(), expected_prediction_sample_1, atol=1e-5
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      tf.saved_model.save(tf_predict_fn, tmpdir)
      loaded = tf.saved_model.load(tmpdir)
      loaded_prediction_sample_1 = loaded(**kwargs_call_1)
      loaded_prediction_sample_2 = loaded(**kwargs_call_2)

    self.assertIn("serving_default", loaded.signatures)

    for signature_key in loaded.signatures:
      logging.info("\nSignature Key: '%s'", signature_key)
      signature = loaded.signatures[signature_key]
      logging.info("  Inputs:")
      for input_tensor in signature.inputs:
        logging.info(
            "    - %s, Shape: %s, DType: %s",
            input_tensor.name,
            input_tensor.shape,
            input_tensor.dtype,
        )
      logging.info("  Outputs:")
      for output_key, output_tensor in signature.structured_outputs.items():
        logging.info(
            "    - '%s', Name: %s, Shape: %s, DType: %s",
            output_key,
            output_tensor.name,
            output_tensor.shape,
            output_tensor.dtype,
        )

    if consume_tf_graph_dict:
      signature = loaded.signatures["serving_default"]
      input_names = [t.name for t in signature.inputs]

      h = f"{tf_io.BEGIN_CODE}23{tf_io.END_CODE}"
      u = f"{tf_io.BEGIN_CODE}5f{tf_io.END_CODE}"
      expected_keys_and_shapes = [
          ("nodes_client_reserved_size", ()),
          (f"nodes_client_{h}id", [None]),
          ("nodes_client_city", [None]),
          ("nodes_client_age", [None]),
          (f"nodes_client_categorical{u}label", [None]),
          ("nodes_transaction_reserved_size", ()),
          (f"nodes_transaction_{h}id", [None]),
          ("nodes_transaction_date", [None]),
          ("nodes_transaction_amount", [None]),
          (
              f"edges_transation{u}to{u}client_reserved_adjacency",
              [2, None],
          ),
      ]

      for key, expected_shape in expected_keys_and_shapes:
        self.assertTrue(
            any(key in name for name in input_names),
            f"Key {key} not found in signature inputs: {input_names}",
        )

        tensor = next((t for t in signature.inputs if key in t.name), None)
        self.assertIsNotNone(tensor, f"Tensor for key {key} not found")
        if expected_shape == ():
          self.assertEqual(tensor.shape, ())  # pyrefly: ignore[missing-attribute]
        else:
          self.assertEqual(tensor.shape.as_list(), expected_shape)  # pyrefly: ignore[missing-attribute]

      self.assertTrue(
          any("seed_node_idxs" in name for name in input_names),
          f"seed_node_idxs not found in signature inputs: {input_names}",
      )

    np.testing.assert_allclose(
        loaded_prediction_sample_1.numpy(),
        expected_prediction_sample_1,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        loaded_prediction_sample_2.numpy(),
        expected_prediction_sample_2,
        atol=1e-5,
    )

  def test_to_tensorflow_function_serialized_tfgnn_graphs(self):
    schema = self.model.data().schema
    sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=_sampling_plan(self.model),
        schema=schema,
        batch_size=5,
    )
    # Note: We use different samples to test that the traced/frozen model can
    # run on graph samples of different sizes.
    samples = [sampler.sample(0), sampler.sample(1)]
    serialized_samples = tf.constant([
        tf_graph_sample_lib.graph_to_tfgnn_graph(
            sample, schema=schema
        ).SerializeToString()
        for sample in samples
    ])
    # Note: By convention, the seed node is the first node of the target
    # nodeset of a graph sample.
    expected_predictions = np.concatenate(
        [self.model.predict(sample, [0]) for sample in samples], axis=0
    )

    tf_predict_fn = self.model.to_tensorflow_function(
        input_format="SERIALIZED_TFGNN_GRAPHS"
    )

    predictions = tf_predict_fn(serialized_samples)  # pyrefly: ignore[not-callable]
    self.assertEqual(
        predictions.shape.as_list(), list(expected_predictions.shape)
    )
    np.testing.assert_allclose(
        predictions.numpy(), expected_predictions, atol=1e-5
    )

    # The exported function is a valid TF SavedModel.
    with tempfile.TemporaryDirectory() as tmpdir:
      tf.saved_model.save(tf_predict_fn, tmpdir)
      loaded = tf.saved_model.load(tmpdir)
      loaded_predictions = loaded(serialized_samples)

    self.assertIn("serving_default", loaded.signatures)
    input_names = [t.name for t in loaded.signatures["serving_default"].inputs]
    self.assertTrue(
        any("examples" in name for name in input_names),
        f"examples not found in signature inputs: {input_names}",
    )
    np.testing.assert_allclose(
        loaded_predictions.numpy(), expected_predictions, atol=1e-5
    )

  def test_to_tensorflow_function_deprecated_consume_tf_graph_dict(self):
    """The deprecated `consume_tf_graph_dict` argument is still supported."""
    sample = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=_sampling_plan(self.model),
        schema=self.model.data().schema,
        batch_size=5,
    ).sample(0)
    tf_sample = tf_io.graph_to_tf_graph(sample, schema=self.model.data().schema)
    expected_prediction = self.model.predict(sample, [0])

    tf_predict_fn = self.model.to_tensorflow_function(
        consume_tf_graph_dict=True
    )

    prediction = tf_predict_fn(  # pyrefly: ignore[not-callable]
        **tf_io.tf_graph_to_tf_graph_dict(tf_sample),
        seed_node_idxs=tf.constant([0]),
    )
    np.testing.assert_allclose(
        prediction.numpy(), expected_prediction, atol=1e-5
    )

    with self.assertRaisesRegex(ValueError, "cannot be set at the same time"):
      self.model.to_tensorflow_function(
          input_format="TF_GRAPH", consume_tf_graph_dict=True
      )

  def test_evaluate(self):

    evaluation = self.model.evaluate(self.graph)
    logging.info("evaluation:\n%s", evaluation)
    self.assertEqual(
        evaluation.num_examples,
        self.graph.node_sets["client"].num_nodes,
    )
    # Note: The dataset does not contain patterns.
    self.assertGreaterEqual(evaluation.accuracy, 0.0)  # pyrefly: ignore[no-matching-overload]

  def test_label_classes(self):
    classes = self.model.label_classes()
    self.assertLen(classes, 2)
    for c in classes:
      self.assertIsInstance(c, str)
    logging.info("label_classes: %s", classes)

  def test_manual_seed_nodes(self):
    model = node_prediction_lib.train_node_model(
        graph=self.graph,
        schema=self.schema,
        target_nodeset="client",
        target_column="categorical_label",
        train_seed_nodes=list(range(200)),
        valid_seed_nodes=list(range(200, 210)),
        **RAPID_TRAINING_KWARGS,
    )
    self.assertEqual(model.data().training_stats.num_train_seed_nodes, 200)
    self.assertEqual(model.data().training_stats.num_valid_seed_nodes, 10)

  def test_valid_graph(self):
    model = node_prediction_lib.train_node_model(
        graph=self.graph,
        valid_graph=self.graph,
        schema=self.schema,
        target_nodeset="client",
        target_column="categorical_label",
        **RAPID_TRAINING_KWARGS,
    )
    self.assertEqual(
        model.data().training_stats.num_train_seed_nodes,
        self.graph.node_sets["client"].num_nodes,
    )
    self.assertEqual(
        model.data().training_stats.num_valid_seed_nodes,
        self.graph.node_sets["client"].num_nodes,
    )

  def test_valid_graph_and_seed_nodes(self):
    model = node_prediction_lib.train_node_model(
        graph=self.graph,
        valid_graph=self.graph,
        schema=self.schema,
        target_nodeset="client",
        target_column="categorical_label",
        train_seed_nodes=list(range(200)),
        valid_seed_nodes=list(range(200, 210)),
        **RAPID_TRAINING_KWARGS,
    )
    self.assertEqual(model.data().training_stats.num_train_seed_nodes, 200)
    self.assertEqual(model.data().training_stats.num_valid_seed_nodes, 10)

  def test_graph_samples(self):
    tmpdir = self.create_tempdir().full_path
    path = os.path.join(tmpdir, "samples@5.tfrecord")

    subgraph = synthetic_lib.generate_synthetic_graph(
        self.schema,
        synthetic_lib.SyntheticGraphConfig(num_nodes=5, num_edges=5),
    )

    def in_mem_graphs():
      for _ in range(101):
        yield subgraph

    tf_graph_sample_lib.write_tfgnn_graphs(
        in_mem_graphs(),
        path,
        schema=self.schema,
        container_type="TF_RECORD",
    )

    model = node_prediction_lib.train_node_model(
        graph=path,
        valid_graph=path,
        schema=self.schema,
        target_nodeset="client",
        target_column="categorical_label",
        **RAPID_TRAINING_KWARGS,
    )
    self.assertIsNone(model.data().training_stats.num_train_seed_nodes)
    self.assertIsNone(model.data().training_stats.num_valid_seed_nodes)
    # The sampling plan used to generate the samples is unknown.
    self.assertIsNone(model.data().sampling_plan)

  def test_predict_batch_insufficient_padding(self):
    """Tests that predict_batch handles InsufficientPaddingError by splitting."""

    original_graph_merger = node_prediction_model.merge_lib.GraphMerger
    call_count = 0

    class MockGraphMerger(original_graph_merger):

      def __call__(self, *call_args, **call_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
          raise node_prediction_model.merge_lib.InsufficientPaddingError(
              "Simulated insufficient padding"
          )
        return super().__call__(*call_args, **call_kwargs)

    with unittest.mock.patch.object(
        node_prediction_model.merge_lib,
        "GraphMerger",
        side_effect=MockGraphMerger,
    ):
      # In test, batch_size is 5 (from RAPID_TRAINING_KWARGS).
      # We need to call predict with at least 2 examples to trigger splitting.
      # We use 3 examples to be safe.
      predictions = self.model.predict(
          graph=self.graph, seed_node_idxs=[0, 1, 2]
      )

      self.assertEqual(predictions.shape, (3, self.model.num_label_classes()))
      self.assertGreater(call_count, 1)

  def test_architecture(self):
    test_util.assert_golden_string(
        self,
        self.model.data().core_model_config.architecture(),
        "node_prediction_architecture.txt",
    )


class NodePredictionGraphSamplesWithoutSamplingPlanTest(parameterized.TestCase):
  """Trains on already sampled graph samples without a `sampling_plan`.

  In this case, the sampling plan used to generate the samples is unknown: the
  model does not contain any sampling plan, and it can only generate
  predictions from graph samples.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.graph, cls.schema = _gen_graph_real_looking()

    cls.sample = synthetic_lib.generate_synthetic_graph(
        cls.schema,
        synthetic_lib.SyntheticGraphConfig(num_nodes=5, num_edges=5),
    )

    cls.samples_path = os.path.join(tempfile.mkdtemp(), "samples@5.tfrecord")

    def in_mem_graphs():
      for _ in range(101):
        yield cls.sample

    tf_graph_sample_lib.write_tfgnn_graphs(
        in_mem_graphs(),
        cls.samples_path,
        schema=cls.schema,
        container_type="TF_RECORD",
    )

    cls.model = node_prediction_lib.train_node_model(
        graph=cls.samples_path,
        valid_graph=cls.samples_path,
        schema=cls.schema,
        target_nodeset="client",
        target_column="categorical_label",
        **RAPID_TRAINING_KWARGS,
    )

  def test_no_sampling_plan(self):
    self.assertIsNone(self.model.data().sampling_plan)

  def test_warning_in_logs(self):
    captured_logs = self.model.metadata.captured_logs
    self.assertIsNotNone(captured_logs)
    warnings = [
        message.text
        for message in captured_logs  # pyrefly: ignore[bad-argument-type]
        if message.severity == log.Severity.WARNING
        and "`sampling_plan`" in message.text
    ]
    self.assertNotEmpty(warnings)

  def test_describe(self):
    html_description = self.model.describe()._repr_html_()  # pylint: disable=attribute-error
    self.assertIn("The sampling plan is not available", html_description)

  def test_predict_raises(self):
    with self.assertRaisesRegex(ValueError, "does not have a sampling plan"):
      self.model.predict(graph=self.graph, seed_node_idxs=[0, 1, 2])

  def test_evaluate_raises(self):
    with self.assertRaisesRegex(ValueError, "does not have a sampling plan"):
      self.model.evaluate(self.graph)

  def test_predict_on_graph_sample_batch(self):
    predictions = self.model.predict_on_graph_sample_batch([self.sample])
    self.assertEqual(predictions.shape, (1, self.model.num_label_classes()))
    self.assertTrue(np.allclose(np.sum(predictions, axis=1), 1.0))

  def test_evaluate_generator(self):
    evaluation = self.model.evaluate_generator(iter([self.sample] * 4))
    self.assertEqual(evaluation.num_examples, 4)

  def test_to_tensorflow_function(self):
    expected_predictions = self.model.predict_on_graph_sample_batch(
        [self.sample]
    )
    serialized_samples = tf.constant([
        tf_graph_sample_lib.graph_to_tfgnn_graph(
            self.sample, schema=self.model.data().schema
        ).SerializeToString()
    ])

    tf_predict_fn = self.model.to_tensorflow_function(
        input_format="SERIALIZED_TFGNN_GRAPHS"
    )
    predictions = tf_predict_fn(serialized_samples)  # pyrefly: ignore[not-callable]
    np.testing.assert_allclose(
        predictions.numpy(), expected_predictions, atol=1e-5
    )

  def test_save_and_load(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      self.model.save(tmpdir)
      restored_model = common_lib.load_model(tmpdir)

    assert isinstance(restored_model, node_prediction_lib.NodePredictionModel)
    self.assertIsNone(restored_model.data().sampling_plan)
    with self.assertRaisesRegex(ValueError, "does not have a sampling plan"):
      restored_model.predict(graph=self.graph, seed_node_idxs=[0, 1, 2])

  def test_with_explicit_sampling_plan(self):
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="client", num_hops=1, hop_width=2
        ),
        self.schema,
    )
    model = node_prediction_lib.train_node_model(
        graph=self.samples_path,
        valid_graph=self.samples_path,
        schema=self.schema,
        target_nodeset="client",
        target_column="categorical_label",
        sampling_plan=sampling_plan,
        **RAPID_TRAINING_KWARGS,
    )
    test_util.assert_are_equal(self, model.data().sampling_plan, sampling_plan)
    predictions = model.predict(graph=self.graph, seed_node_idxs=[0, 1, 2])
    self.assertEqual(predictions.shape, (3, model.num_label_classes()))


class NodePredictionRealLookingStandaloneTest(parameterized.TestCase):

  @unittest.skip("The test requires graphviz")
  def test_diagnostic_dir(self):
    graph, schema = _gen_graph_real_looking()
    diagnostic_dir = self.create_tempdir().full_path
    _ = node_prediction_lib.train_node_model(
        graph=graph,
        valid_graph=graph,
        schema=schema,
        target_nodeset="client",
        target_column="categorical_label",
        diagnostic_dir=diagnostic_dir,
        **RAPID_TRAINING_KWARGS,
    )
    for filename in ["graph_0.png", "graph_1.png"]:
      self.assertTrue(fs.exists(os.path.join(diagnostic_dir, filename)))

  def test_partial_cover(self):
    graph, schema = _gen_graph_real_looking()
    my_args = {**RAPID_TRAINING_KWARGS}
    del my_args["num_sampling_hops"]
    _ = node_prediction_lib.train_node_model(
        graph=graph,
        valid_graph=graph,
        schema=schema,
        num_sampling_hops=0,  # The transactions table is not visited.
        target_nodeset="client",
        target_column="categorical_label",
        **my_args,  # pyrefly: ignore[potential-bad-keyword-argument]
    )


# This test trains a model on a toy patter. While this is a toy pattern, this
# test take some time to run.
class NodePredictionClassificationToy(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    # Generate data
    graph_kwargs = {
        "num_n1_nodes": 1000,
        "num_n2_nodes": 500,
        "accuracy": 0.8,
    }
    cls.graph_train, cls.schema = gen_test_graph.gen_toy_classification_dataset(
        **graph_kwargs, random_seed=0
    )
    cls.graph_test, _ = gen_test_graph.gen_toy_classification_dataset(
        **graph_kwargs, random_seed=1
    )

    # Train a model
    cls.model = node_prediction_lib.train_node_model(
        graph=cls.graph_train,
        schema=cls.schema,
        target_nodeset="N1",
        target_column="label",
        num_train_steps=100,
        valid_every_n_steps=25,
        num_sampling_hops=1,
    )

  def test_evaluate(self):

    evaluation = self.model.evaluate(self.graph_test)
    logging.info("evaluation:\n%s", evaluation)
    self.assertEqual(
        evaluation.num_examples,
        self.graph_test.node_sets["N1"].num_nodes,
    )
    # TODO(gbm): Stabilize quality.
    self.assertAlmostEqual(evaluation.accuracy, 0.8, delta=0.15)  # pyrefly: ignore[no-matching-overload]
    self.assertIsNotNone(evaluation.auc)
    self.assertGreater(evaluation.auc, 0.5)  # pyrefly: ignore[no-matching-overload]
    self.assertLen(evaluation.per_classes, 2)
    for pc in evaluation.per_classes:
      self.assertIsNotNone(pc.auc())
      self.assertIsNotNone(pc.pr_auc())
      self.assertLen(pc.tp, 10001)

  def test_label_classes_fails_for_integer_labels(self):
    with self.assertRaises(ValueError) as ctx:
      self.model.label_classes()
    self.assertIn("does not have a string dictionary", str(ctx.exception))


class NodePredictionRegressionToy(parameterized.TestCase):

  @parameterized.parameters(("float32", 1), ("int64", 1), ("float32", 3))
  def test_evaluate(self, label_dtype, label_dim):

    # Generate data
    graph_kwargs = {
        "num_n1_nodes": 1000,
        "num_n2_nodes": 500,
        "label_dim": label_dim,
    }
    graph_train, schema = gen_test_graph.gen_toy_regression_dataset(
        **graph_kwargs, random_seed=0, label_dtype=label_dtype
    )
    graph_test, _ = gen_test_graph.gen_toy_regression_dataset(
        **graph_kwargs, random_seed=1, label_dtype=label_dtype
    )

    # Train a model
    model = node_prediction_lib.train_node_model(
        graph=graph_train,
        schema=schema,
        target_nodeset="N1",
        target_column="label",
        num_train_steps=100,
        valid_every_n_steps=25,
        num_sampling_hops=1,
    )

    evaluation = model.evaluate(graph_test)
    logging.info("evaluation:\n%s", evaluation)
    self.assertEqual(
        evaluation.num_examples,
        graph_test.node_sets["N1"].num_nodes,
    )
    # TODO(gbm): Stabilize quality.
    self.assertIsNotNone(evaluation.rmse)
    self.assertLess(evaluation.rmse, 1.5)  # pyrefly: ignore[no-matching-overload]

    with self.assertRaises(ValueError) as ctx:
      model.label_classes()
    self.assertIn(
        "only supported for NODE_CLASSIFICATION tasks", str(ctx.exception)
    )

    # Save and restore the model
    with tempfile.TemporaryDirectory() as tmpdir:
      model.save(tmpdir)
      restored_model = common_lib.load_model(tmpdir)

    # Check the model class
    assert isinstance(restored_model, node_prediction_lib.NodePredictionModel)

    # Check the model data
    test_util.assert_are_equal(self, model.data(), restored_model.data())  # pylint: disable=protected-access

    # Check the predictions
    predictions = model.predict(graph=graph_test, seed_node_idxs=[0, 1, 2])
    restored_predictions = restored_model.predict(
        graph=graph_test, seed_node_idxs=[0, 1, 2]
    )
    test_util.assert_are_equal(
        self, predictions, restored_predictions, abs_tol=0.00001
    )


class NodePredictionRealLookingTemporal(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    # Generate data
    cls.graph, cls.schema = _gen_graph_real_looking(has_timestamp_feature=True)

    # Note: Set "model_path" to a path to cache the model. This allows the
    # test to setup faster / you to iterate faster.
    model_path = None
    # model_path = "/tmp/my_temporal_model"

    def train_model() -> node_prediction_lib.NodePredictionModel:
      # Train a basic model
      return node_prediction_lib.train_node_model(
          graph=cls.graph,
          schema=cls.schema,
          target_nodeset="client",
          target_column="categorical_label",
          time_aware=True,
          **RAPID_TRAINING_KWARGS,
      )

    if model_path is None:
      cls.model = train_model()
    else:
      if fs.exists(model_path):
        model = common_lib.load_model(model_path)
        assert isinstance(model, node_prediction_lib.NodePredictionModel)
        cls.model = model
      else:
        cls.model = train_model()
        cls.model.save(model_path)

  def test_evaluate(self):
    evaluation = self.model.evaluate(self.graph)
    logging.info("evaluation:\n%s", evaluation)

  def test_predict(self):
    predictions = self.model.predict(graph=self.graph, seed_node_idxs=[0, 1, 2])
    self.assertEqual(predictions.shape, (3, self.model.num_label_classes()))
    self.assertTrue(np.allclose(np.sum(predictions, axis=1), 1.0))


class NodePredictionPredictibleModelTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.graph, cls.schema = gen_predictible_model_graph()

    def preprocess_core_model_config(c):
      return dataclasses.replace(
          c,
          pre_mlp=standard.identity(),
      )

    cls.model = node_prediction_lib.train_node_model(
        graph=cls.graph,
        schema=cls.schema,
        target_nodeset="A",
        target_column="label",
        num_train_steps=4,
        valid_every_n_steps=2,
        batch_size=1,
        sampling_width=1,
        num_sampling_hops=0,
        node_embedding_dim=1,
        learning_rate=0.1,
        verbose=3,
        num_layers=0,
        experimental_preprocess_core_model_config=preprocess_core_model_config,
    )

    # Set parameter names "biases" to zero and parameters named "kernel" to 1.
    def set_params(path, leaf):
      path_str = "".join(str(p) for p in path)
      if "kernel" in path_str:
        return jnp.ones_like(leaf)
      elif "bias" in path_str or "biases" in path_str:
        return jnp.zeros_like(leaf)
      elif "scale" in path_str:
        return jnp.ones_like(leaf)
      else:
        raise ValueError(f"Unknown parameter in path: {path_str}")

    cls.model.data().model_params = jax.tree_util.tree_map_with_path(
        set_params, cls.model.data().model_params
    )

  def test_predict(self):
    predictions = self.model.predict(self.graph, seed_node_idxs=[0, 1, 2])
    expected_predictions = np.array([1.0, -1.0, 2.0], dtype=np.float32)
    test_util.assert_are_equal(
        self,
        predictions,
        expected_predictions,
        abs_tol=0.001,
    )


class NodePredictionTemporalValidationTest(absltest.TestCase):

  def test_temporal_training_no_target_nodeset_timestamp_raises(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(
        include_e2=False
    )
    schema.node_sets["n1"].features["timestamp"].is_creation_time = False

    with self.assertRaisesRegex(
        ValueError,
        "The target nodeset 'n1' must have a creation time feature",
    ):
      node_prediction_lib.train_node_model(
          graph=graph,
          schema=schema,
          target_nodeset="n1",
          target_column="timestamp",
          time_aware=True,
      )

  def test_temporal_training_no_edgeset_timestamp_plan_is_temporal(self):
    graph, schema = _gen_graph_real_looking(has_timestamp_feature=True)
    for es_schema in schema.edge_sets.values():
      for feat in es_schema.features.values():
        feat.is_creation_time = False

    model = node_prediction_lib.train_node_model(
        graph=graph,
        schema=schema,
        target_nodeset="client",
        target_column="categorical_label",
        time_aware=True,
        num_train_steps=2,
        valid_every_n_steps=1,
        batch_size=1,
        sampling_width=1,
        num_sampling_hops=1,
        node_embedding_dim=1,
        num_layers=1,
        verbose=0,
    )
    self.assertTrue(model.data().temporal_sampling)
    self.assertTrue(_sampling_plan(model).temporal_sampling)


class NodePredictionTimeseriesTest(absltest.TestCase):

  def _create_timeseries_graph_and_schema(
      self,
  ) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
    """Creates 1 sensor node with a timeseries and 2 query nodes with different creation times."""

    schema = schema_lib.GraphSchema(
        node_sets={
            "query": schema_lib.NodeSchema(
                features={
                    "creation_time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_creation_time=True,
                    ),
                    "label": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                    ),
                }
            ),
            "sensor": schema_lib.NodeSchema(
                features={
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        shape=(None,),
                        group="sensor_ts",
                    ),
                }
            ),
        },
        edge_sets={
            "query_to_sensor": schema_lib.EdgeSchema(
                source="query",
                target="sensor",
                features={},
            ),
            "sensor_to_query": schema_lib.EdgeSchema(
                source="sensor",
                target="query",
                features={},
            ),
        },
    )

    time_data = np.empty(1, dtype=object)
    time_data[0] = np.array([100, 300], dtype=np.int64)

    sensor_nodes = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=1,
        features={"time": time_data},
    )

    # Query 0: creation_time = 120 -> sees t=100 (timedelta = 20), label = 1.0
    # Query 1: creation_time = 180 -> sees t=100 (timedelta = 80), label = 2.0
    query_nodes = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=2,
        features={
            "creation_time": np.array([120, 180], dtype=np.int64),
            "label": np.array([1.0, 2.0], dtype=np.float32),
        },
    )

    query_to_sensor = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=np.array([[0, 1], [0, 0]], dtype=np.int64),
        features={},
    )
    sensor_to_query = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=np.array([[0, 0], [0, 1]], dtype=np.int64),
        features={},
    )

    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={"query": query_nodes, "sensor": sensor_nodes},
        edge_sets={
            "query_to_sensor": query_to_sensor,
            "sensor_to_query": sensor_to_query,
        },
    )
    return graph, schema

  def test_train_and_predict_with_timeseries(self):
    graph, schema = self._create_timeseries_graph_and_schema()

    model = node_prediction_lib.train_node_model(
        graph=graph,
        valid_graph=graph,
        schema=schema,
        target_nodeset="query",
        target_column="label",
        time_aware=True,
        num_train_steps=100,
        batch_size=2,
        sampling_width=2,
        num_sampling_hops=1,
        node_embedding_dim=16,
        num_layers=1,
        learning_rate=0.01,
        verbose=0,
    )

    predictions = model.predict(graph, seed_node_idxs=np.array([0, 1]))
    self.assertEqual(predictions.shape, (2,))
    self.assertTrue(np.all(np.isfinite(predictions)))
    # Both queries see the same event at t=100 with no other features.
    # The only differentiator is the timedelta to creation_time (-20 vs -80).
    self.assertGreater(predictions[1], predictions[0])
    self.assertAlmostEqual(predictions[0], 1.0, delta=0.3)
    self.assertAlmostEqual(predictions[1], 2.0, delta=0.3)

    # Verify predict_on_graph_sample_batch produces matching predictions
    sampler = in_memory_sampler_lib.create_sampler(
        graph=graph,
        plan=_sampling_plan(model),
        schema=schema,
        batch_size=2,
    )
    samples = sampler.sample(
        np.array([0, 1]), seed_timestamps=np.array([120, 180], dtype=np.int64)
    )
    sample_batch_predictions = model.predict_on_graph_sample_batch(samples)
    self.assertEqual(sample_batch_predictions.shape, (2,))
    np.testing.assert_allclose(sample_batch_predictions, predictions, atol=1e-5)

    eval_result = model.evaluate(graph)
    self.assertEqual(eval_result.num_examples, 2)
    self.assertIsNotNone(eval_result.rmse)
    self.assertLess(eval_result.rmse, 0.5)  # pyrefly: ignore[no-matching-overload]


if __name__ == "__main__":
  absltest.main()
