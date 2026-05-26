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

"""Tests for standard layers."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.jax.layers import standard
import jax
import jax.numpy as jnp


class StandardTest(parameterized.TestCase):

  def test_ingest_feature_architecture_and_execution(self):
    config = standard.ingest_feature(dims=32)
    expected_arch = "Dense(32)\nActivation(silu)\nNorm(layer_norm)"
    self.assertEqual(config.architecture(), expected_arch)

    model = config.make()
    x = jnp.ones((2, 4, 16))
    variables = model.init(jax.random.PRNGKey(0), x, training=True)
    y = model.apply(variables, x, training=True)
    self.assertEqual(y.shape, (2, 4, 32))

  def test_sequential_mlp_architecture_and_execution(self):
    config = standard.sequential_mlp(dims=32, num_layers=3, dropout_rate=0.1)
    expected_arch = (
        "Norm(rms_norm)\n"
        "Dense(32)\n"
        "Activation(silu)\n"
        "Dropout(0.1)\n"
        "Dense(32)\n"
        "Activation(silu)\n"
        "Dropout(0.1)\n"
        "Dense(32)\n"
        "Activation(silu)\n"
        "Dropout(0.1)\n"
        "Dense(32)"
    )
    self.assertEqual(config.architecture(), expected_arch)

    model = config.make()
    x = jnp.ones((2, 4, 16))
    rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
    variables = model.init(rngs, x, training=True)
    y = model.apply(variables, x, training=True, rngs=rngs)
    self.assertEqual(y.shape, (2, 4, 32))

  def test_identity_architecture_and_execution(self):
    config = standard.identity()
    self.assertEqual(config.architecture(), "Identity")

    model = config.make()
    x = jnp.ones((2, 4, 16))
    variables = model.init(jax.random.PRNGKey(0), x, training=True)
    y = model.apply(variables, x, training=True)
    self.assertEqual(y.shape, (2, 4, 16))
    self.assertTrue((x == y).all())


class GenericBlockTest(parameterized.TestCase):

  def test_architecture_representation(self):
    # Valid architecture configurations
    config = standard.GenericBlockConfig(
        config="LNADL",
        dims=16,
        dropout_rate=0.1,
        norm="layer_norm",
        activation="silu",
    )
    expected = (
        "Dense(16)\nNorm(layer_norm)\nActivation(silu)\nDropout(0.1)\nDense(16)"
    )
    self.assertEqual(config.architecture(), expected)

    # With multiplier
    config = standard.GenericBlockConfig(
        config="L4AL", dims=16, activation="relu"
    )
    expected = "Dense(64)\nActivation(relu)\nDense(16)"
    self.assertEqual(config.architecture(), expected)

    # Empty config
    config = standard.GenericBlockConfig(config="", dims=16)
    self.assertEqual(config.architecture(), "Identity")

  def test_generic_block_empty_config_execution(self):
    config = standard.GenericBlockConfig(config="", dims=16)
    self.assertEqual(config.architecture(), "Identity")

    model = config.make()
    x = jnp.ones((2, 4, 16))
    variables = model.init(jax.random.PRNGKey(0), x, training=True)
    y = model.apply(variables, x, training=True)
    self.assertEqual(y.shape, (2, 4, 16))
    self.assertTrue((x == y).all())

  def test_robust_parsing(self):
    # Mixed case, embedded spaces
    config = standard.GenericBlockConfig(
        config="l 4 a l ", dims=16, activation="relu"
    )
    expected = "Dense(64)\nActivation(relu)\nDense(16)"
    self.assertEqual(config.architecture(), expected)

  @parameterized.parameters(
      # Redundant consecutive layers
      ("LL", 16, "layer_norm", "silu", 0.1, "Redundant consecutive layers"),
      ("L4 L", 16, "layer_norm", "silu", 0.1, "Redundant consecutive layers"),
      ("NN", 16, "layer_norm", "silu", 0.1, "Redundant consecutive layers"),
      ("AA", 16, "layer_norm", "silu", 0.1, "Redundant consecutive layers"),
      ("DD", 16, "layer_norm", "silu", 0.1, "Redundant consecutive layers"),
      # Missing parameters
      (
          "N",
          16,
          None,
          "silu",
          0.1,
          (
              "Normalization layer 'N' was requested in config, but 'norm'"
              " parameter is None"
          ),
      ),
      (
          "A",
          16,
          "layer_norm",
          None,
          0.1,
          (
              "Activation layer 'A' was requested in config, but 'activation'"
              " parameter is None"
          ),
      ),
      (
          "D",
          16,
          "layer_norm",
          "silu",
          None,
          (
              "Dropout layer 'D' was requested in config, but 'dropout_rate'"
              " is None"
          ),
      ),
      (
          "D",
          16,
          "layer_norm",
          "silu",
          0.0,
          (
              "Dropout layer 'D' was requested in config, but 'dropout_rate'"
              " is None or <= 0.0"
          ),
      ),
      (
          "D",
          16,
          "layer_norm",
          "silu",
          -0.5,
          (
              "Dropout layer 'D' was requested in config, but 'dropout_rate'"
              " is None or <= 0.0"
          ),
      ),
      (
          "L0",
          16,
          "layer_norm",
          "silu",
          0.1,
          "Linear layer multiplier must be strictly positive",
      ),
      (
          "L-3",
          16,
          "layer_norm",
          "silu",
          0.1,
          (
              "Invalid config string:.*contains unsupported characters or"
              " digits not following 'L'"
          ),
      ),
      # Invalid characters or digits not following 'L'
      (
          "L2X",
          16,
          "layer_norm",
          "silu",
          0.1,
          "unsupported characters or digits not following 'L'",
      ),
      (
          "L234A5",
          16,
          "layer_norm",
          "silu",
          0.1,
          "unsupported characters or digits not following 'L'",
      ),
  )
  def test_validation_errors(
      self, config_str, dims, norm, activation, dropout_rate, error_msg
  ):
    with self.assertRaisesRegex(ValueError, error_msg):
      standard.GenericBlockConfig(
          config=config_str,
          dims=dims,
          norm=norm,
          activation=activation,
          dropout_rate=dropout_rate,
      )

  def test_dims_validation(self):
    with self.assertRaisesRegex(ValueError, "dims must be strictly positive"):
      standard.GenericBlockConfig(config="L", dims=0)
    with self.assertRaisesRegex(ValueError, "dims must be strictly positive"):
      standard.GenericBlockConfig(config="L", dims=-5)

  def test_execution(self):
    config = standard.GenericBlockConfig(
        config="L2 N A D L",
        dims=8,
        norm="batch_norm",
        activation="relu",
        dropout_rate=0.2,
    )
    model = config.make()
    x = jnp.ones((2, 4, 16))
    rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}

    variables = model.init(rngs, x, training=True)
    y, _ = model.apply(
        variables, x, training=True, rngs=rngs, mutable=["batch_stats"]
    )
    self.assertEqual(y.shape, (2, 4, 8))

  def test_architecture_representation_with_residual(self):
    config = standard.GenericBlockConfig(
        config="NL4ALDR",
        dims=16,
        dropout_rate=0.1,
        norm="layer_norm",
        activation="silu",
    )
    expected = (
        "X = ...\n"
        "Norm(layer_norm)\n"
        "Dense(64)\n"
        "Activation(silu)\n"
        "Dense(16)\n"
        "Dropout(0.1)\n"
        "Residual(X)"
    )
    self.assertEqual(config.architecture(), expected)

  def test_execution_with_residual(self):
    config = standard.modern_residual_mlp(dims=8, dropout_rate=0.2)
    model = config.make()
    x = jnp.ones((2, 4, 8))
    rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}

    variables = model.init(rngs, x, training=True)
    y = model.apply(variables, x, training=True, rngs=rngs)
    self.assertEqual(y.shape, (2, 4, 8))


if __name__ == "__main__":
  absltest.main()
