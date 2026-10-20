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

import json
import os

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.learning import early_stopping_monitor
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.ten_lines import common
from dgf.src.util import gen_test_graph
from dgf.src.util import log
from dgf.src.util import test_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp


def _set_checkpoint_device(path: str, device: str) -> None:
  """Rewrites the device on which the weights in "path" were saved."""

  sharding_path = os.path.join(path, common.FILENAME_PARAMS, "_sharding")
  with open(sharding_path, "r") as f:
    shardings = json.load(f)
  for key, sharding in shardings.items():
    sharding = json.loads(sharding)
    sharding["device_str"] = device
    shardings[key] = json.dumps(sharding)
  with open(sharding_path, "w") as f:
    json.dump(shardings, f)


class TenLines(parameterized.TestCase):

  def test_num_model_weights(self):
    params = {
        "layer1": {
            "w": np.ones((10, 5), dtype=np.float32),
            "b": np.zeros((5,), dtype=np.float32),
        },
        "layer2": {
            "w": np.ones((5, 2), dtype=np.int32),
            "b": np.zeros((2,), dtype=np.int32),
        },
    }
    weights = common.num_model_weights(params)
    self.assertEqual(weights, {"float32": 55, "int32": 12})

  def test_num_model_weights_none(self):
    self.assertEqual(common.num_model_weights(None), {})

  @parameterized.named_parameters(
      (
          "hmp",
          "heterogeneous_message_passing",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hmp_alias_hmpnn",
          "hmpnn",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hmp_upper",
          "HETEROGENEOUS_MESSAGE_PASSING",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hmp_enum",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hgat",
          "heterogeneous_graph_attention_network",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      (
          "hgat_alias_hgat",
          "hgat",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      (
          "hgat_alias_han",
          "han",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      (
          "hgat_enum",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      ("sage", "graph_sage", common.Architecture.GRAPH_SAGE),
      ("sage_alias", "GraphSAGE", common.Architecture.GRAPH_SAGE),
      (
          "gcn",
          "graph_convolutional_network",
          common.Architecture.GRAPH_CONVOLUTIONAL_NETWORK,
      ),
      ("gcn_alias", "gcn", common.Architecture.GRAPH_CONVOLUTIONAL_NETWORK),
      ("gat", "GAT", common.Architecture.GRAPH_ATTENTION_NETWORK),
      (
          "gat_long",
          "graph_attention_network",
          common.Architecture.GRAPH_ATTENTION_NETWORK,
      ),
      ("gatv2", "gatv2", common.Architecture.GRAPH_ATTENTION_NETWORK_V2),
      ("mtalbis", "mtalbis", common.Architecture.MT_ALBIS),
      ("mtalbis_alias", "Albis", common.Architecture.MT_ALBIS),
  )
  def test_parse_architecture_success(self, input_val, expected):
    self.assertEqual(common.parse_architecture(input_val), expected)

  @parameterized.parameters(
      (common.Architecture.HETEROGENEOUS_MESSAGE_PASSING, "Aggregation(mean)"),
      (
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
          "DotProductAttention(heads=4, dims=8)",
      ),
      (common.Architecture.GRAPH_SAGE, "Aggregation(mean)"),
      (
          common.Architecture.GRAPH_CONVOLUTIONAL_NETWORK,
          "Aggregation(symmetric)",
      ),
      (common.Architecture.GRAPH_ATTENTION_NETWORK, "GatAttention(heads=8"),
      (
          common.Architecture.GRAPH_ATTENTION_NETWORK_V2,
          "Gatv2Attention(heads=8",
      ),
      (common.Architecture.MT_ALBIS, "RelationAggregation(concat)"),
  )
  def test_build_gnn_config(
      self, architecture: common.Architecture, expected_in_architecture: str
  ):
    hparams = common.HParam(
        node_embedding_dim=8,
        message_aggregation=hetero_gnn.MessageAggregation.MEAN,
        architecture=architecture,
    )
    config = common.build_gnn_config(hparams)
    self.assertIn(expected_in_architecture, config.architecture())

  def test_parse_architecture_invalid_fails(self):
    with self.assertRaisesRegex(ValueError, "Unknown architecture: invalid"):
      common.parse_architecture("invalid")

    with self.assertRaisesRegex(TypeError, "Expected Architecture or str"):
      common.parse_architecture(123)  # pyrefly: ignore[bad-argument-type]

  @parameterized.named_parameters(
      ("cnn", "cnn", common.TimeseriesEncoder.CNN),
      ("cnn_upper", "CNN", common.TimeseriesEncoder.CNN),
      (
          "cnn_enum",
          common.TimeseriesEncoder.CNN,
          common.TimeseriesEncoder.CNN,
      ),
      (
          "transformer",
          "transformer",
          common.TimeseriesEncoder.TRANSFORMER,
      ),
      (
          "transformer_upper",
          "TRANSFORMER",
          common.TimeseriesEncoder.TRANSFORMER,
      ),
      (
          "transformer_enum",
          common.TimeseriesEncoder.TRANSFORMER,
          common.TimeseriesEncoder.TRANSFORMER,
      ),
  )
  def test_parse_timeseries_encoder_success(self, input_val, expected):
    self.assertEqual(common.parse_timeseries_encoder(input_val), expected)

  def test_parse_timeseries_encoder_invalid_fails(self):
    with self.assertRaisesRegex(
        ValueError, "Unknown timeseries encoder: invalid"
    ):
      common.parse_timeseries_encoder("invalid")

    with self.assertRaisesRegex(TypeError, "Expected TimeseriesEncoder or str"):
      common.parse_timeseries_encoder(123)  # pyrefly: ignore[bad-argument-type]

  def test_build_timeseries_encoder_config_uses_max_timeseries_len(self):
    hparams = common.HParam(
        timeseries_encoder=common.TimeseriesEncoder.TRANSFORMER,
        timeseries_embedding_dim=64,
    )
    config = common.build_timeseries_encoder_config(
        hparams, max_timeseries_len=96
    )
    self.assertEqual(
        config.max_timeseries_len, 96  # pyrefly: ignore[missing-attribute]
    )
    self.assertAlmostEqual(
        config.rope_max_wavelength,  # pyrefly: ignore[missing-attribute]
        96.0 ** (16.0 / 14.0),
        places=5,
    )

  @parameterized.named_parameters(
      ("not_divisible_by_num_heads", 30),
      ("odd_head_dim", 36),
  )
  def test_build_timeseries_encoder_config_invalid_transformer_dim(
      self, timeseries_embedding_dim
  ):
    hparams = common.HParam(
        timeseries_encoder=common.TimeseriesEncoder.TRANSFORMER,
        timeseries_embedding_dim=timeseries_embedding_dim,
    )
    with self.assertRaisesRegex(
        ValueError, "timeseries_embedding_dim must be a multiple of"
    ):
      common.build_timeseries_encoder_config(hparams, max_timeseries_len=32)

  def test_check_number_of_seeds_success(self):
    common.check_number_of_seeds(
        batch_size=10, num_training=20, num_validation=15, key="node"
    )
    common.check_number_of_seeds(
        batch_size=10, num_training=20, num_validation=None, key="node"
    )
    common.check_number_of_seeds(
        batch_size=10, num_training=None, num_validation=15, key="node"
    )

  def test_check_number_of_seeds_insufficient_training_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        r"The number of training seed nodes \(5\) is smaller than the batch"
        r" size \(10\)\. Increase the number of training seed nodes or decrease"
        r" the batch size\.",
    ):
      common.check_number_of_seeds(
          batch_size=10, num_training=5, num_validation=15, key="node"
      )

  def test_check_number_of_seeds_insufficient_validation_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        r"The number of validation seed nodes \(5\) is smaller than the batch"
        r" size \(10\)\. Increase the number of validation seed edges or"
        r" decrease"
        r" the batch size\.",
    ):
      common.check_number_of_seeds(
          batch_size=10, num_training=15, num_validation=5, key="edge"
      )

  def test_log_jax_backend(self):
    with log.capture_logs(log_info=True, log_warning=True) as captured:
      common.log_jax_backend(verbose=2)
    self.assertTrue(
        any(
            "Using" in msg.text and "JAX backend" in msg.text
            for msg in captured
        )
    )
    if jax.default_backend().lower() == "cpu":
      self.assertTrue(
          any(
              msg.severity == log.Severity.WARNING
              and "Using CPU JAX backend" in msg.text
              for msg in captured
          )
      )

  @parameterized.named_parameters(
      ("tf_graph", "TF_GRAPH", common.TFFunctionInputFormat.TF_GRAPH),
      ("lower", "tf_graph_dict", common.TFFunctionInputFormat.TF_GRAPH_DICT),
      (
          "serialized",
          "SERIALIZED_TFGNN_GRAPHS",
          common.TFFunctionInputFormat.SERIALIZED_TFGNN_GRAPHS,
      ),
      (
          "enum",
          common.TFFunctionInputFormat.TF_GRAPH_DICT,
          common.TFFunctionInputFormat.TF_GRAPH_DICT,
      ),
  )
  def test_parse_tf_function_input_format_success(self, input_val, expected):
    self.assertEqual(common.parse_tf_function_input_format(input_val), expected)

  def test_parse_tf_function_input_format_invalid_fails(self):
    with self.assertRaisesRegex(ValueError, "Unknown input format: invalid"):
      common.parse_tf_function_input_format("invalid")

    with self.assertRaisesRegex(
        TypeError, "Expected TFFunctionInputFormat or str"
    ):
      common.parse_tf_function_input_format(123)  # pyrefly: ignore[bad-argument-type]

  def test_resolve_tf_function_input_format_default(self):
    self.assertEqual(
        common.resolve_tf_function_input_format(None, None),
        common.DEFAULT_TF_FUNCTION_INPUT_FORMAT,
    )

  def test_resolve_tf_function_input_format(self):
    self.assertEqual(
        common.resolve_tf_function_input_format("TF_GRAPH_DICT", None),
        common.TFFunctionInputFormat.TF_GRAPH_DICT,
    )

  @parameterized.named_parameters(
      ("true", True, common.TFFunctionInputFormat.TF_GRAPH_DICT),
      ("false", False, common.TFFunctionInputFormat.TF_GRAPH),
  )
  def test_resolve_tf_function_input_format_deprecated(
      self, consume_tf_graph_dict, expected
  ):
    with log.capture_logs(log_info=True, log_warning=True) as captured:
      self.assertEqual(
          common.resolve_tf_function_input_format(None, consume_tf_graph_dict),
          expected,
      )
    self.assertTrue(any("deprecated" in msg.text for msg in captured))

  def test_resolve_tf_function_input_format_conflict_fails(self):
    with self.assertRaisesRegex(ValueError, "cannot be set at the same time"):
      common.resolve_tf_function_input_format("TF_GRAPH", True)

  def test_save_and_load_params(self):
    path = self.create_tempdir().full_path
    params = {
        "layer1": {"w": jnp.ones((3, 2)), "b": jnp.zeros((2,))},
        "layer2": {"w": jnp.full((2, 1), 0.5)},
    }
    common.save_params(params, path)
    test_util.assert_are_equal(self, common.load_params(path), params)

  def test_load_params_saved_on_another_accelerator(self):
    # A model trained on a GPU should be loadable on a CPU-only machine.
    path = self.create_tempdir().full_path
    params = {"layer1": {"w": jnp.ones((3, 2))}}
    common.save_params(params, path)
    _set_checkpoint_device(path, "cuda:0")

    # The default orbax restore looks for the device used at save time.
    with self.assertRaisesRegex(ValueError, "Topology mismatch detected"):
      ocp.StandardCheckpointer().restore(
          os.path.join(path, common.FILENAME_PARAMS)
      )

    test_util.assert_are_equal(self, common.load_params(path), params)

  def test_check_skipped_training_samples_below_threshold_passes(self):
    common.check_skipped_training_samples(
        num_skipped_samples=0, num_generated_samples=0
    )
    common.check_skipped_training_samples(
        num_skipped_samples=0, num_generated_samples=100
    )
    common.check_skipped_training_samples(
        num_skipped_samples=10, num_generated_samples=90
    )

  def test_check_skipped_training_samples_fails_above_ten_percent(self):
    with self.assertRaisesRegex(
        common.merge_lib.InsufficientPaddingError,
        r"Skipped 15 out of 100 training samples \(15\.0%\) due to insufficient"
        r" padding",
    ):
      common.check_skipped_training_samples(
          num_skipped_samples=15, num_generated_samples=85
      )

  def test_extract_graph_metrics(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    metrics = common.extract_graph_metrics(graph, schema)
    self.assertIn("num_nodesets", metrics)
    self.assertIn("num_edgesets", metrics)
    self.assertIn("num_features", metrics)
    self.assertIn("num_nodes", metrics)
    self.assertIn("num_edges", metrics)
    self.assertEqual(metrics["num_nodesets"], len(schema.node_sets))
    self.assertEqual(metrics["num_edgesets"], len(schema.edge_sets))
    self.assertGreater(metrics["num_nodes"], 0)
    self.assertGreater(metrics["num_edges"], 0)
    self.assertGreater(metrics["num_features"], 0)

  def test_schema_to_serving_signature_dict(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f_none": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=None,
                    ),
                    "f_none_23": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=(None, 23),
                    ),
                    "f_none_1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=(None, 1),
                    ),
                }
            )
        },
        edge_sets={},
    )
    signature = common.schema_to_serving_signature_dict(
        schema_=schema,
        target_nodeset="n1",
        model_name="test_model",
        model_uuid="test_uuid",
    )

    f_none = signature["gnn_n1_nodes_n1_f_none"]
    self.assertEqual(f_none["shape"], "(None,)")

    f_none_23 = signature["gnn_n1_nodes_n1_f_none_23"]
    self.assertEqual(f_none_23["shape"], "(-1, 23)")

    f_none_1 = signature["gnn_n1_nodes_n1_f_none_1"]
    self.assertEqual(f_none_1["shape"], "(-1, 1)")

  def test_schema_to_serving_signature_dict_link_prediction(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "f_none": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=None,
                    ),
                },
            )
        },
    )
    signature = common.schema_to_serving_signature_dict(
        schema_=schema,
        target_nodeset=None,
        target_edgeset="e1",
        model_name="test_model",
        model_uuid="test_uuid",
    )
    f_none = signature["source_edges_e1_f_none"]
    self.assertEqual(f_none["shape"], "(None,)")


def _training_logs(
    valid_losses: list[float], stop_reason: common.StopReason | None
) -> common.TrainingLogs:
  return common.TrainingLogs(
      train=[],
      valid=[
          common.LogItem(step=(i + 1) * 100, metrics={"loss": loss})
          for i, loss in enumerate(valid_losses)
      ],
      num_train_step=len(valid_losses) * 100,
      stop_reason=stop_reason,
  )


class ConvergenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "still_improving",
          [3.0, 2.0, 1.0],
          common.StopReason.MAX_NUM_TRAIN_STEPS,
          True,
      ),
      (
          "still_improving_time_limit",
          [3.0, 2.0, 1.0],
          common.StopReason.MAX_TRAINING_TIME,
          True,
      ),
      (
          "plateau",
          [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          common.StopReason.MAX_NUM_TRAIN_STEPS,
          False,
      ),
      (
          "overfitting",
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          common.StopReason.MAX_NUM_TRAIN_STEPS,
          False,
      ),
      ("early_stopping", [3.0, 2.0, 1.0], common.StopReason.EARLY_STOPPING, False),
      (
          "dataset_exhausted",
          [3.0, 2.0, 1.0],
          common.StopReason.DATASET_EXHAUSTED,
          False,
      ),
      ("no_validation", [], common.StopReason.MAX_NUM_TRAIN_STEPS, False),
      ("unknown_stop_reason", [3.0, 2.0, 1.0], None, False),
  )
  def test_is_not_converged(self, valid_losses, stop_reason, expected):
    self.assertEqual(
        common.is_not_converged(
            _training_logs(valid_losses, stop_reason), early_stopping=None
        ),
        expected,
    )

  def test_is_not_converged_uses_early_stopping_config(self):
    logs = _training_logs(
        [3.0, 1.0, 0.95, 0.9], common.StopReason.MAX_NUM_TRAIN_STEPS
    )
    # Improvements of 0.05 are not significant: The best loss is reached at the
    # second validation, two validations ago.
    self.assertFalse(
        common.is_not_converged(
            logs,
            early_stopping_monitor.EarlyStoppingMonitorConfig(
                patience=2, min_improvement=0.2
            ),
        )
    )
    self.assertTrue(
        common.is_not_converged(
            logs,
            early_stopping_monitor.EarlyStoppingMonitorConfig(
                patience=3, min_improvement=0.2
            ),
        )
    )

  def test_warn_if_not_converged_max_num_steps(self):
    with log.capture_logs() as logs:
      common.warn_if_not_converged(
          _training_logs([3.0, 2.0], common.StopReason.MAX_NUM_TRAIN_STEPS),
          common.HParam(num_train_steps=200),
      )
    self.assertLen(logs, 1)
    self.assertEqual(logs[0].severity, log.Severity.WARNING)
    self.assertIn("not converged", logs[0].text)
    self.assertIn("`num_train_steps`=200", logs[0].text)
    self.assertIn("increasing `num_train_steps`", logs[0].text)

  def test_warn_if_not_converged_max_training_time(self):
    with log.capture_logs() as logs:
      common.warn_if_not_converged(
          _training_logs([3.0, 2.0], common.StopReason.MAX_TRAINING_TIME),
          common.HParam(max_training_time_seconds=60),
      )
    self.assertLen(logs, 1)
    self.assertIn("`max_training_time_seconds`=60", logs[0].text)
    self.assertIn("increasing `max_training_time_seconds`", logs[0].text)

  def test_warn_if_not_converged_converged(self):
    with log.capture_logs() as logs:
      common.warn_if_not_converged(
          _training_logs([3.0, 2.0], common.StopReason.EARLY_STOPPING),
          common.HParam(),
      )
    self.assertEmpty(logs)

  def test_metadata_json_with_stop_reason(self):
    metadata = common.Metadata(
        name="test",
        trainig_logs=_training_logs(
            [1.0], common.StopReason.MAX_NUM_TRAIN_STEPS
        ),
    )
    restored = common.Metadata.from_json(metadata.to_json())  # pyrefly: ignore[missing-attribute]
    self.assertEqual(
        restored.trainig_logs.stop_reason,
        common.StopReason.MAX_NUM_TRAIN_STEPS,
    )

  def test_metadata_json_without_stop_reason(self):
    # Models saved before the introduction of `stop_reason`.
    restored = common.Metadata.from_json(  # pyrefly: ignore[missing-attribute]
        json.dumps({
            "name": "test",
            "trainig_logs": {"train": [], "valid": [], "num_train_step": 10},
        })
    )
    self.assertIsNone(restored.trainig_logs.stop_reason)


class SelectBatchSizeTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, 20_000, 128),
      (1_000, None, 128),
      (1_175, 20_000, 64),
      (128_000, 20_000, 64),
      (163_263, 20_000, 128),
      (1_000, 100, 128),
  )
  def test_select_batch_size(
      self, num_train_examples, num_train_steps, expected
  ):
    self.assertEqual(
        common.select_batch_size(
            num_train_examples, num_train_steps, batch_size_range=(64, 128)
        ),
        expected,
    )

  def test_resolve_batch_size_and_learning_rate(self):
    batch_size, learning_rate = common.resolve_batch_size_and_learning_rate(
        None, None, 1_000, 20_000, (64, 128), 1e-3
    )
    self.assertEqual(batch_size, 64)
    self.assertAlmostEqual(learning_rate, 0.064)
    self.assertEqual(
        common.resolve_batch_size_and_learning_rate(
            32, 0.1, 1_000, 20_000, (64, 128), 1e-3
        ),
        (32, 0.1),
    )


class RecommendSmallerBatchSizeOnOomTest(absltest.TestCase):

  def test_out_of_memory_recommends_smaller_batch_size(self):
    with self.assertRaisesRegex(MemoryError, "batch_size=64"):
      with common.recommend_smaller_batch_size_on_oom(128):
        raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory")

  def test_other_errors_are_unchanged(self):
    with self.assertRaisesRegex(jax.errors.JaxRuntimeError, "INTERNAL"):
      with common.recommend_smaller_batch_size_on_oom(128):
        raise jax.errors.JaxRuntimeError("INTERNAL: Something else")


if __name__ == "__main__":
  absltest.main()
