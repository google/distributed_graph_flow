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

"""Tests for TimeseriesCNNEncoder layer."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import timeseries_cnn as lib
from dgf.src.learning.jax.layers.registry import registry as layer_registry
import jax
import jax.numpy as jnp


class TimeseriesCNNEncoderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.feature_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        shape=(10, 16),
        is_timeseries=True,
    )
    self.mask_2d_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(5,),
        is_timeseries=True,
    )

  def test_forward_and_parameters(self):
    batch_size, seq_len, in_channels, out_dim = 2, 6, 8, 10
    config = lib.TimeseriesCNNEncoderConfig(
        out_dim=out_dim, conv_channels=32, kernel_size=3, num_layers=2
    )
    encoder = config.make(self.feature_schema)
    x = jnp.ones((batch_size, seq_len, in_channels), dtype=jnp.float32)

    variables = encoder.init(jax.random.PRNGKey(0), x, training=False)
    output = encoder.apply(variables, x, training=False)

    self.assertEqual(output.shape, (batch_size, out_dim))
    params = variables["params"]
    self.assertIn("conv1d_0", params)
    self.assertIn("conv1d_1", params)
    self.assertIn("nad_block_0", params)
    self.assertIn("nad_block_1", params)
    self.assertIn("norm_0", params["nad_block_0"])
    self.assertIn("norm_0", params["nad_block_1"])
    self.assertIn("output_projection", params)
    self.assertEqual(params["conv1d_0"]["kernel"].shape, (3, in_channels, 32))
    self.assertEqual(params["conv1d_1"]["kernel"].shape, (3, 32, 32))
    self.assertEqual(
        params["output_projection"]["kernel"].shape, (32, out_dim)
    )

  def test_variable_sequence_lengths(self):
    config = lib.TimeseriesCNNEncoderConfig(out_dim=16, conv_channels=32)
    encoder = config.make(self.feature_schema)
    x1 = jnp.ones((2, 5, 8), dtype=jnp.float32)
    x2 = jnp.ones((2, 12, 8), dtype=jnp.float32)
    variables = encoder.init(jax.random.PRNGKey(0), x1, training=False)
    out1 = encoder.apply(variables, x1, training=False)
    out2 = encoder.apply(variables, x2, training=False)
    self.assertEqual(out1.shape, (2, 16))
    self.assertEqual(out2.shape, (2, 16))

  @parameterized.named_parameters(
      ("layer_norm", "layer_norm", 0.1),
      ("rms_norm", "rms_norm", 0.0),
      ("batch_norm", "batch_norm", 0.2),
      ("no_norm_no_dropout", None, 0.0),
  )
  def test_norm_and_dropout_configurations(self, norm, dropout_rate):
    config = lib.TimeseriesCNNEncoderConfig(
        out_dim=8,
        conv_channels=16,
        kernel_size=3,
        num_layers=1,
        norm=norm,
        dropout_rate=dropout_rate,
    )
    encoder = config.make(self.feature_schema)
    x = jnp.ones((2, 6, 4), dtype=jnp.float32)
    rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}

    variables = encoder.init(rngs, x, training=True)
    if norm == "batch_norm":
      output, _ = encoder.apply(
          variables, x, training=True, rngs=rngs, mutable=["batch_stats"]
      )
    else:
      output = encoder.apply(variables, x, training=True, rngs=rngs)
    self.assertEqual(output.shape, (2, 8))

  def test_all_true_mask_matches_no_mask(self):
    x = jax.random.normal(jax.random.PRNGKey(42), (2, 5, 4))
    mask = jnp.ones((2, 5), dtype=bool)
    mask_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(5,),
        is_timeseries=True,
    )
    config = lib.TimeseriesCNNEncoderConfig(out_dim=16, conv_channels=32)
    encoder_with_mask = config.make(
        self.feature_schema, mask_schema=mask_schema
    )
    encoder_no_mask = config.make(self.feature_schema, mask_schema=None)

    variables = encoder_no_mask.init(jax.random.PRNGKey(0), x, training=False)
    out_no_mask = encoder_no_mask.apply(variables, x, mask=None, training=False)
    out_with_mask = encoder_with_mask.apply(
        variables, x, mask=mask, training=False
    )

    self.assertTrue(jnp.allclose(out_no_mask, out_with_mask, atol=1e-5))

  def test_mask_anti_contamination(self):
    mask = jnp.array([[True, True, False], [True, False, False]], dtype=bool)
    batch_size, seq_len = mask.shape[0], mask.shape[1]
    in_channels = 2
    x_clean = jax.random.normal(
        jax.random.PRNGKey(7), (batch_size, seq_len, in_channels)
    )

    corruption = jnp.where(~mask[:, :, None], 999.0, 0.0)
    x_corrupted = x_clean + corruption

    mask_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(3,),
        is_timeseries=True,
    )
    config = lib.TimeseriesCNNEncoderConfig(out_dim=16, conv_channels=32)
    encoder = config.make(self.feature_schema, mask_schema=mask_schema)

    variables = encoder.init(
        jax.random.PRNGKey(0), x_clean, mask=mask, training=False
    )
    out_clean = encoder.apply(variables, x_clean, mask=mask, training=False)
    out_corrupted = encoder.apply(
        variables, x_corrupted, mask=mask, training=False
    )

    self.assertTrue(jnp.allclose(out_clean, out_corrupted, atol=1e-5))

  @parameterized.named_parameters(
      ("2d_input_x", (4, 16), None, False),
      ("4d_input_x", (4, 5, 8, 2), None, False),
      ("missing_mask_schema", None, (2, 5), True),
      ("missing_mask_when_schema_present", None, None, False),
      ("2d_mask_wrong_shape", None, (2, 6), False),
      ("3d_mask_with_sequence_schema", None, (2, 5, 4), False),
  )
  def test_invalid_call_inputs(
      self, x_shape, mask_shape, missing_mask_schema
  ):
    config = lib.TimeseriesCNNEncoderConfig(out_dim=16)
    encoder = config.make(
        self.feature_schema,
        mask_schema=None if missing_mask_schema else self.mask_2d_schema,
    )
    x = jnp.ones(x_shape or (2, 5, 4), dtype=jnp.float32)
    mask = jnp.ones(mask_shape, dtype=bool) if mask_shape is not None else None
    with self.assertRaises(AssertionError):
      encoder.init(jax.random.PRNGKey(0), x, mask=mask)

  def test_unsupported_mask_schema_rank(self):
    config = lib.TimeseriesCNNEncoderConfig(out_dim=16)
    invalid_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(5, 4),
        is_timeseries=True,
    )
    encoder = config.make(self.feature_schema, mask_schema=invalid_schema)
    x = jnp.ones((2, 5, 4), dtype=jnp.float32)
    mask = jnp.ones((2, 5), dtype=bool)
    with self.assertRaises(AssertionError):
      encoder.init(jax.random.PRNGKey(0), x, mask=mask)

  def test_mask_schema_none_shape(self):
    config = lib.TimeseriesCNNEncoderConfig(out_dim=16)
    none_shape_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=None,
        is_timeseries=True,
    )
    encoder = config.make(self.feature_schema, mask_schema=none_shape_schema)
    x = jnp.ones((2, 5, 4), dtype=jnp.float32)
    mask = jnp.ones((2, 5), dtype=bool)
    with self.assertRaises(AssertionError):
      encoder.init(jax.random.PRNGKey(0), x, mask=mask)

  @parameterized.named_parameters(
      ("out_dim", {"out_dim": -1}, "out_dim must be positive"),
      (
          "conv_channels",
          {"out_dim": 32, "conv_channels": 0},
          "conv_channels must be positive",
      ),
      (
          "kernel_size",
          {"out_dim": 32, "kernel_size": -1},
          "kernel_size must be positive",
      ),
      (
          "num_layers",
          {"out_dim": 32, "num_layers": 0},
          "num_layers must be positive",
      ),
      (
          "dropout_rate",
          {"out_dim": 32, "dropout_rate": 1.5},
          "dropout_rate must be in",
      ),
  )
  def test_invalid_config(self, kwargs, error_regex):
    with self.assertRaisesRegex(ValueError, error_regex):
      lib.TimeseriesCNNEncoderConfig(**kwargs)

  def test_config_methods_and_registry(self):
    config = lib.TimeseriesCNNEncoderConfig(
        out_dim=32, conv_channels=64, kernel_size=3, num_layers=2
    )
    self.assertEqual(
        config.architecture(),
        "TimeseriesCNNEncoder(num_layers=2, kernel_size=3, channels=64,"
        " out_dim=32)",
    )
    self.assertEqual(
        config.output_schema(),
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(32,),
            is_timeseries=False,
        ),
    )
    # Serialization
    json_str = config.to_json()
    reconstructed = lib.TimeseriesCNNEncoderConfig.from_json(json_str)
    self.assertEqual(config, reconstructed)

    # Registry
    key = "layers.TimeseriesCNNEncoderConfig"
    self.assertIn(key, layer_registry.registered_keys())
    config_cls = layer_registry._registered_classes[key]
    self.assertIsInstance(
        config_cls(out_dim=16), lib.TimeseriesCNNEncoderConfig
    )

    # Make
    encoder = config.make(self.feature_schema, mask_schema=self.mask_2d_schema)
    self.assertEqual(encoder.feature_schema, self.feature_schema)
    self.assertEqual(encoder.mask_schema, self.mask_2d_schema)


if __name__ == "__main__":
  absltest.main()
