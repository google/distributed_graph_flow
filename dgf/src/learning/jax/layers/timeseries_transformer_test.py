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

"""Tests for TimeseriesTransformerEncoder layer."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers import timeseries_transformer as lib
from dgf.src.learning.jax.layers.registry import registry as layer_registry
from dgf.src.util import test_util
import jax
import jax.numpy as jnp
import numpy as np


def _rope(x, positions, max_wavelength):
  """Applies RoPE to `x` at `positions`, building the tables on the fly."""
  sin, cos = lib._rope_sin_cos(positions, x.shape[-1], max_wavelength)
  return lib._apply_rope(x, sin, cos)


class ApplyRopeTest(parameterized.TestCase):
  """Unit tests for the bare RoPE rotation."""

  def test_rotation_preserves_norm(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3, 8))
    positions = jnp.arange(5, dtype=jnp.float32)

    rotated = _rope(x, positions, 100.0)

    self.assertEqual(rotated.shape, x.shape)
    np.testing.assert_allclose(
        jnp.linalg.norm(rotated, axis=-1),
        jnp.linalg.norm(x, axis=-1),
        atol=1e-5,
    )

  def test_position_zero_is_identity(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 3, 8))
    positions = jnp.zeros((1,), dtype=jnp.float32)

    rotated = _rope(x, positions, 100.0)

    np.testing.assert_allclose(rotated, x, atol=1e-6)

  def test_sin_cos_table_shape(self):
    sin, cos = lib._rope_sin_cos(jnp.arange(5, dtype=jnp.float32), 8, 100.0)
    self.assertEqual(sin.shape, (5, 1, 4))
    self.assertEqual(cos.shape, (5, 1, 4))

  def test_dot_product_depends_only_on_offset(self):
    """The whole point of RoPE: <rope(q, i), rope(k, j)> depends on i - j."""
    q = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 1, 8))
    k = jax.random.normal(jax.random.PRNGKey(1), (1, 1, 1, 8))

    def score(i, j):
      qi = _rope(q, jnp.array([float(i)]), 100.0)
      kj = _rope(k, jnp.array([float(j)]), 100.0)
      return jnp.sum(qi * kj)

    # Same offset of 3, different absolute positions.
    self.assertAlmostEqual(float(score(3, 0)), float(score(10, 7)), places=4)
    self.assertAlmostEqual(float(score(3, 0)), float(score(31, 28)), places=4)
    # A different offset gives a different score.
    self.assertNotAlmostEqual(float(score(3, 0)), float(score(5, 0)), places=4)


class DefaultRopeMaxWavelengthTest(absltest.TestCase):
  """Unit tests for `default_rope_max_wavelength`."""

  def test_rotates_slowest_pair_by_one_rad(self):
    for seq_len in (16, 32, 96, 512):
      for head_dim in (4, 8, 16, 32):
        wavelength = lib.default_rope_max_wavelength(seq_len, head_dim)
        self.assertGreater(wavelength, 1.0)
        # Slowest pair has index head_dim // 2 - 1, so fraction = (d - 2) / d
        # and timescale = wavelength ** ((d - 2) / d) == seq_len.
        slowest_timescale = wavelength ** ((head_dim - 2) / head_dim)
        self.assertAlmostEqual(slowest_timescale, float(seq_len), places=5)

  def test_edge_cases(self):
    # head_dim=2 has a single pair (fraction=0), returns max(T, 2).
    self.assertEqual(lib.default_rope_max_wavelength(32, 2), 32.0)
    # max_timeseries_len=1 is clamped to 2 so wavelength stays > 1.
    self.assertGreater(lib.default_rope_max_wavelength(1, 16), 1.0)
    with self.assertRaisesRegex(
        ValueError, "max_timeseries_len must be positive"
    ):
      lib.default_rope_max_wavelength(0, 16)
    with self.assertRaisesRegex(ValueError, "head_dim must be a positive even"):
      lib.default_rope_max_wavelength(32, 3)


class TimeseriesTransformerEncoderTest(parameterized.TestCase):

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

  def _mask_schema(self, seq_len: int) -> schema_lib.FeatureSchema:
    return schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(seq_len,),
        is_timeseries=True,
    )

  def test_forward_and_parameters(self):
    batch_size, seq_len, in_channels, out_dim = 2, 6, 8, 10
    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=out_dim,
        max_timeseries_len=seq_len,
        dims=32,
        num_heads=4,
        num_layers=2,
    )
    encoder = config.make(self.feature_schema)
    x = jnp.ones((batch_size, seq_len, in_channels), dtype=jnp.float32)

    variables = encoder.init(jax.random.PRNGKey(0), x, training=False)
    output = encoder.apply(variables, x, training=False)

    self.assertEqual(output.shape, (batch_size, out_dim))
    params = variables["params"]
    self.assertIn("input_projection", params)
    self.assertIn("attention_norm_0", params)
    self.assertIn("attention_norm_1", params)
    self.assertIn("attention_0", params)
    self.assertIn("attention_1", params)
    self.assertIn("ffn_0", params)
    self.assertIn("ffn_1", params)
    self.assertIn("output_projection", params)
    self.assertEqual(
        params["input_projection"]["kernel"].shape, (in_channels, 32)
    )
    self.assertEqual(params["output_projection"]["kernel"].shape, (32, out_dim))

  def test_variable_sequence_lengths(self):
    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=12, dims=32
    )
    encoder = config.make(self.feature_schema)
    x1 = jnp.ones((2, 5, 8), dtype=jnp.float32)
    x2 = jnp.ones((2, 12, 8), dtype=jnp.float32)

    variables = encoder.init(jax.random.PRNGKey(0), x1, training=False)
    out1 = encoder.apply(variables, x1, training=False)
    out2 = encoder.apply(variables, x2, training=False)

    self.assertEqual(out1.shape, (2, 16))
    self.assertEqual(out2.shape, (2, 16))

  def test_declared_output_schema_matches_actual_output(self):
    """The declared `output_schema` is what callers size their layers with."""
    batch_size, seq_len, in_channels = 2, 6, 8
    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=10, max_timeseries_len=seq_len, dims=32
    )
    encoder = config.make(self.feature_schema)
    x = jnp.ones((batch_size, seq_len, in_channels), dtype=jnp.float32)

    variables = encoder.init(jax.random.PRNGKey(0), x, training=False)
    output = encoder.apply(variables, x, training=False)

    declared_schema = config.output_schema()
    self.assertFalse(declared_schema.is_timeseries)
    self.assertEqual(output.shape, (batch_size,) + declared_schema.shape)
    self.assertEqual(output.dtype, jnp.float32)

  def test_left_padding_invariance(self):
    """Same events with more left padding must give the same embedding.

    This is the property that motivates using RoPE alone: attention scores
    depend only on the offset `i - j`, so right-aligned sequences are encoded
    identically no matter how far the pipeline's `seq_len` cap pads them.
    """
    in_channels, num_valid, num_pad = 4, 4, 3
    events = jax.random.normal(
        jax.random.PRNGKey(3), (2, num_valid, in_channels)
    )

    x_short = events
    mask_short = jnp.ones((2, num_valid), dtype=bool)

    padding = jnp.zeros((2, num_pad, in_channels))
    x_long = jnp.concatenate([padding, events], axis=1)
    mask_long = jnp.concatenate(
        [jnp.zeros((2, num_pad), dtype=bool), mask_short], axis=1
    )

    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=num_valid + num_pad, dims=32
    )
    encoder_short = config.make(
        self.feature_schema, mask_schema=self._mask_schema(num_valid)
    )
    encoder_long = config.make(
        self.feature_schema, mask_schema=self._mask_schema(num_valid + num_pad)
    )

    variables = encoder_short.init(
        jax.random.PRNGKey(0), x_short, mask=mask_short, training=False
    )
    out_short = encoder_short.apply(
        variables, x_short, mask=mask_short, training=False
    )
    out_long = encoder_long.apply(
        variables, x_long, mask=mask_long, training=False
    )

    np.testing.assert_allclose(out_short, out_long, atol=1e-5)

  def test_left_padding_invariance_ignores_padded_content(self):
    """Padding invariance must not depend on the padded slots being zero."""
    in_channels, num_valid, num_pad = 4, 4, 3
    events = jax.random.normal(
        jax.random.PRNGKey(3), (2, num_valid, in_channels)
    )
    garbage = jax.random.normal(
        jax.random.PRNGKey(4), (2, num_pad, in_channels)
    )

    mask = jnp.concatenate(
        [
            jnp.zeros((2, num_pad), dtype=bool),
            jnp.ones((2, num_valid), dtype=bool),
        ],
        axis=1,
    )
    x_zero_pad = jnp.concatenate(
        [jnp.zeros((2, num_pad, in_channels)), events], axis=1
    )
    x_garbage_pad = jnp.concatenate([garbage, events], axis=1)

    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=num_valid + num_pad, dims=32
    )
    encoder = config.make(
        self.feature_schema, mask_schema=self._mask_schema(num_valid + num_pad)
    )

    variables = encoder.init(
        jax.random.PRNGKey(0), x_zero_pad, mask=mask, training=False
    )
    out_zero = encoder.apply(
        variables, x_zero_pad, mask=mask, training=False
    )
    out_garbage = encoder.apply(
        variables, x_garbage_pad, mask=mask, training=False
    )

    np.testing.assert_allclose(out_zero, out_garbage, atol=1e-5)

  def test_entity_with_no_events(self):
    """An all-masked row must stay finite and ignore its padded content."""
    seq_len, in_channels = 5, 4
    mask = jnp.array(
        [[False] * seq_len, [True] * seq_len],
        dtype=bool,
    )
    x_a = jax.random.normal(jax.random.PRNGKey(5), (2, seq_len, in_channels))
    # Perturb only the fully masked row.
    x_b = x_a.at[0].set(
        jax.random.normal(jax.random.PRNGKey(6), (seq_len, in_channels))
    )

    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=seq_len, dims=32
    )
    encoder = config.make(
        self.feature_schema, mask_schema=self._mask_schema(seq_len)
    )

    variables = encoder.init(
        jax.random.PRNGKey(0), x_a, mask=mask, training=False
    )
    out_a = encoder.apply(variables, x_a, mask=mask, training=False)
    out_b = encoder.apply(variables, x_b, mask=mask, training=False)

    # `jax.nn.dot_product_attention` masks attention logits with a large finite
    # negative value rather than -inf, so an all-masked row normalizes to a
    # uniform distribution instead of NaN.
    self.assertTrue(jnp.all(jnp.isfinite(out_a)))
    np.testing.assert_allclose(out_a[0], out_b[0], atol=1e-5)
    # The unmasked row is untouched by the other row's content.
    np.testing.assert_allclose(out_a[1], out_b[1], atol=1e-5)

  def test_rope_makes_the_encoder_order_sensitive(self):
    """Without a position encoding, mean pooling would be order invariant."""
    seq_len, in_channels = 6, 4
    x = jax.random.normal(jax.random.PRNGKey(8), (2, seq_len, in_channels))
    x_shuffled = x[:, ::-1, :]

    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=seq_len, dims=32, pooling="mean"
    )
    encoder = config.make(self.feature_schema)

    variables = encoder.init(jax.random.PRNGKey(0), x, training=False)
    out = encoder.apply(variables, x, training=False)
    out_shuffled = encoder.apply(variables, x_shuffled, training=False)

    self.assertFalse(jnp.allclose(out, out_shuffled, atol=1e-4))

  def test_last_pooling_reads_the_final_timestep(self):
    """The pipeline pads on the left, so index T-1 is the newest event."""
    seq_len, in_channels = 5, 4
    x = jax.random.normal(jax.random.PRNGKey(9), (2, seq_len, in_channels))
    # Change only the final timestep.
    x_new_last = x.at[:, -1, :].set(x[:, -1, :] + 10.0)

    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=seq_len, dims=32, pooling="last"
    )
    encoder = config.make(self.feature_schema)

    variables = encoder.init(jax.random.PRNGKey(0), x, training=False)
    out = encoder.apply(variables, x, training=False)
    out_new_last = encoder.apply(variables, x_new_last, training=False)

    self.assertFalse(jnp.allclose(out, out_new_last, atol=1e-4))

  @parameterized.named_parameters(
      ("layer_norm_last", "layer_norm", "last"),
      ("layer_norm_mean", "layer_norm", "mean"),
      ("rms_norm_last", "rms_norm", "last"),
      ("rms_norm_mean", "rms_norm", "mean"),
  )
  def test_norm_and_pooling_configurations(self, norm, pooling):
    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=8,
        max_timeseries_len=6,
        dims=16,
        num_heads=2,
        num_layers=1,
        norm=norm,
        pooling=pooling,
    )
    encoder = config.make(self.feature_schema)
    x = jnp.ones((2, 6, 4), dtype=jnp.float32)
    rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}

    variables = encoder.init(rngs, x, training=True)
    output = encoder.apply(variables, x, training=True, rngs=rngs)

    self.assertEqual(output.shape, (2, 8))

  def test_all_true_mask_matches_no_mask(self):
    x = jax.random.normal(jax.random.PRNGKey(42), (2, 5, 4))
    mask = jnp.ones((2, 5), dtype=bool)
    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=5, dims=32
    )
    encoder_with_mask = config.make(
        self.feature_schema, mask_schema=self._mask_schema(5)
    )
    encoder_no_mask = config.make(self.feature_schema, mask_schema=None)

    variables = encoder_no_mask.init(jax.random.PRNGKey(0), x, training=False)
    out_no_mask = encoder_no_mask.apply(variables, x, mask=None, training=False)
    out_with_mask = encoder_with_mask.apply(
        variables, x, mask=mask, training=False
    )

    np.testing.assert_allclose(out_no_mask, out_with_mask, atol=1e-5)

  def test_mask_anti_contamination(self):
    mask = jnp.array([[True, True, False], [True, False, False]], dtype=bool)
    batch_size, seq_len = mask.shape[0], mask.shape[1]
    in_channels = 2
    x_clean = jax.random.normal(
        jax.random.PRNGKey(7), (batch_size, seq_len, in_channels)
    )

    corruption = jnp.where(~mask[:, :, None], 999.0, 0.0)
    x_corrupted = x_clean + corruption

    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=seq_len, dims=32
    )
    encoder = config.make(
        self.feature_schema, mask_schema=self._mask_schema(seq_len)
    )

    variables = encoder.init(
        jax.random.PRNGKey(0), x_clean, mask=mask, training=False
    )
    out_clean = encoder.apply(variables, x_clean, mask=mask, training=False)
    out_corrupted = encoder.apply(
        variables, x_corrupted, mask=mask, training=False
    )

    np.testing.assert_allclose(out_clean, out_corrupted, atol=1e-5)

  def test_mask_ignores_non_finite_padded_values(self):
    """NaN or inf in padded slots must not leak (`NaN * 0 = NaN`)."""
    mask = jnp.array([[True, True, False], [True, False, False]], dtype=bool)
    batch_size, seq_len = mask.shape
    in_channels = 2
    x_clean = jax.random.normal(
        jax.random.PRNGKey(7), (batch_size, seq_len, in_channels)
    )
    x_nan = x_clean.at[0, 2, 0].set(jnp.nan).at[1, 1:, :].set(jnp.inf)

    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=seq_len, dims=32
    )
    encoder = config.make(
        self.feature_schema, mask_schema=self._mask_schema(seq_len)
    )

    variables = encoder.init(
        jax.random.PRNGKey(0), x_clean, mask=mask, training=False
    )
    out_clean = encoder.apply(variables, x_clean, mask=mask, training=False)
    out_nan = encoder.apply(variables, x_nan, mask=mask, training=False)

    self.assertTrue(jnp.all(jnp.isfinite(out_nan)))
    np.testing.assert_allclose(out_clean, out_nan, atol=1e-5)

  @parameterized.named_parameters(
      ("2d_input_x", (4, 16), None, False),
      ("4d_input_x", (4, 5, 8, 2), None, False),
      ("missing_mask_schema", None, (2, 5), True),
      ("missing_mask_when_schema_present", None, None, False),
      ("2d_mask_wrong_shape", None, (2, 6), False),
      ("3d_mask", None, (2, 5, 4), False),
  )
  def test_invalid_call_inputs(self, x_shape, mask_shape, missing_mask_schema):
    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=16, max_timeseries_len=5, dims=32
    )
    encoder = config.make(
        self.feature_schema,
        mask_schema=None if missing_mask_schema else self.mask_2d_schema,
    )
    x = jnp.ones(x_shape or (2, 5, 4), dtype=jnp.float32)
    mask = jnp.ones(mask_shape, dtype=bool) if mask_shape is not None else None
    with self.assertRaises(AssertionError):
      encoder.init(jax.random.PRNGKey(0), x, mask=mask)

  @parameterized.named_parameters(
      (
          "out_dim",
          {"out_dim": -1, "max_timeseries_len": 32},
          "out_dim must be positive",
      ),
      (
          "max_timeseries_len",
          {"out_dim": 32, "max_timeseries_len": 0},
          "max_timeseries_len must be positive",
      ),
      (
          "dims",
          {"out_dim": 32, "max_timeseries_len": 32, "dims": 0},
          "dims must be positive",
      ),
      (
          "num_heads",
          {"out_dim": 32, "max_timeseries_len": 32, "num_heads": 0},
          "num_heads must be positive",
      ),
      (
          "num_layers",
          {"out_dim": 32, "max_timeseries_len": 32, "num_layers": 0},
          "num_layers must be positive",
      ),
      (
          "indivisible_dims",
          {"out_dim": 32, "max_timeseries_len": 32, "dims": 10, "num_heads": 4},
          "must be divisible by num_heads",
      ),
      (
          "odd_head_dim",
          {"out_dim": 32, "max_timeseries_len": 32, "dims": 12, "num_heads": 4},
          "must be even",
      ),
      (
          "dropout_rate",
          {"out_dim": 32, "max_timeseries_len": 32, "dropout_rate": 1.5},
          "dropout_rate must be in",
      ),
      (
          "batch_norm_rejected",
          {"out_dim": 32, "max_timeseries_len": 32, "norm": "batch_norm"},
          "norm must be one of",
      ),
      (
          "pooling",
          {"out_dim": 32, "max_timeseries_len": 32, "pooling": "cls"},
          "pooling must be one of",
      ),
      (
          "rope_max_wavelength",
          {
              "out_dim": 32,
              "max_timeseries_len": 32,
              "rope_max_wavelength": 1.0,
          },
          "rope_max_wavelength must be greater than 1",
      ),
  )
  def test_invalid_config(self, kwargs, error_regex):
    with self.assertRaisesRegex(ValueError, error_regex):
      lib.TimeseriesTransformerEncoderConfig(**kwargs)

  def test_rope_max_wavelength_derived_from_max_timeseries_len(self):
    config_32 = lib.TimeseriesTransformerEncoderConfig(
        out_dim=8, dims=64, num_heads=4, max_timeseries_len=32
    )
    config_96 = lib.TimeseriesTransformerEncoderConfig(
        out_dim=8, dims=64, num_heads=4, max_timeseries_len=96
    )
    self.assertAlmostEqual(
        config_32.rope_max_wavelength, 32.0 ** (16.0 / 14.0), places=5
    )
    self.assertAlmostEqual(
        config_96.rope_max_wavelength, 96.0 ** (16.0 / 14.0), places=5
    )
    # Explicit override is preserved.
    config_override = lib.TimeseriesTransformerEncoderConfig(
        out_dim=8, max_timeseries_len=96, rope_max_wavelength=250.0
    )
    self.assertEqual(config_override.rope_max_wavelength, 250.0)

  def test_default_ffn_inherits_the_encoder_norm(self):
    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=8, max_timeseries_len=32, dims=32, norm="rms_norm"
    )
    self.assertIsInstance(config.ffn, standard.GenericBlockConfig)
    self.assertEqual(config.ffn.norm, "rms_norm")
    self.assertEqual(config.ffn.dims, 32)

  def test_architecture(self):
    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=8, max_timeseries_len=32
    )
    self.assertEqual(
        config.architecture(),
        "TimeseriesTransformerEncoder (dims=64, heads=4)\n"
        "Dense(64) # Input projection\n"
        "Transformer Block x2:\n"
        "    X = ...\n"
        "    Norm(layer_norm)\n"
        "    RoPESelfAttention(heads=4, max_wavelength=52.5)\n"
        "    Residual(X)\n"
        "    # Post Attention FFN\n"
        "    X = ...\n"
        "    Norm(layer_norm)\n"
        "    Dense(256)\n"
        "    Activation(silu)\n"
        "    Dense(64)\n"
        "    Dropout(0.1)\n"
        "    Residual(X)\n"
        "Pool(last)\n"
        "Dense(8) # Output projection",
    )

  def test_config_methods_and_registry(self):
    config = lib.TimeseriesTransformerEncoderConfig(
        out_dim=32,
        max_timeseries_len=32,
        dims=64,
        num_heads=4,
        num_layers=2,
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
    # Serialization. This matters because the layer config is stored inside the
    # saved model and rebuilt on `load()`.
    json_str = config.to_json()
    reconstructed = lib.TimeseriesTransformerEncoderConfig.from_json(json_str)
    test_util.assert_are_equal(self, reconstructed, config)

    # Registry
    key = "layers.TimeseriesTransformerEncoderConfig"
    self.assertIn(key, layer_registry.registered_keys())
    config_cls = layer_registry._registered_classes[key]
    self.assertIsInstance(
        config_cls(out_dim=16, max_timeseries_len=32),
        lib.TimeseriesTransformerEncoderConfig,
    )

    # Make
    encoder = config.make(self.feature_schema, mask_schema=self.mask_2d_schema)
    self.assertEqual(encoder.feature_schema, self.feature_schema)
    self.assertEqual(encoder.mask_schema, self.mask_2d_schema)


if __name__ == "__main__":
  absltest.main()
