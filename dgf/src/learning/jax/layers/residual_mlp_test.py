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

"""Test of the residual MLP layer."""

from absl.testing import absltest
from dgf.src.learning.jax.layers import residual_mlp
import jax
import jax.numpy as jnp


class ResidualMlpTest(absltest.TestCase):

  def test_basic(self):
    mlp = residual_mlp.ResidualMLP(hidden_dim=8, output_dim=12)
    dummy_input = jnp.ones((4, 8))
    params = mlp.init(jax.random.PRNGKey(42), dummy_input)
    output = mlp.apply(params, dummy_input)
    self.assertEqual(output.shape, (4, 12))

  def test_residual_mlp_v2(self):
    mlp = residual_mlp.ResidualMLPV2Config(dims=16, num_layers=2).make()
    dummy_input = jnp.ones((4, 8))
    variables = mlp.init(jax.random.PRNGKey(42), dummy_input)
    output, _ = mlp.apply(
        variables,
        dummy_input,
        training=True,
        mutable=["batch_stats"],
        rngs={"dropout": jax.random.PRNGKey(43)},
    )
    self.assertEqual(output.shape, (4, 16))


if __name__ == "__main__":
  absltest.main()
