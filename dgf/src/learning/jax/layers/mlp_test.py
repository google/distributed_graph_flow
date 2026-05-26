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

"""Tests for common layers."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.jax.layers import mlp
import flax.linen as nn
import jax


class LayersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="no_norm_single_layer",
          norm_type=None,
          num_layers=1,
          hidden_dim=4,
      ),
      dict(
          testcase_name="no_norm_two_layers",
          norm_type=None,
          num_layers=2,
          hidden_dim=4,
      ),
      dict(
          testcase_name="layer_norm_single_layer",
          norm_type="layer",
          num_layers=1,
          hidden_dim=4,
      ),
      dict(
          testcase_name="layer_norm_three_layers",
          norm_type="layer",
          num_layers=1,
          hidden_dim=12,
      ),
  )
  def test_mlp(
      self,
      norm_type,
      num_layers,
      hidden_dim,
  ):
    model = mlp.MLP(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        norm_type=norm_type,
    )
    B = 10
    D = 16
    dummy_input = jax.random.normal(jax.random.PRNGKey(42), (B, D))
    params = model.init(jax.random.PRNGKey(42), dummy_input)
    output = model.apply(params, dummy_input)
    self.assertEqual(output.shape, (B, hidden_dim))

  @parameterized.named_parameters(
      dict(
          testcase_name="gnn_plus_num_layers_3_hidden_dim_8",
          num_layers=3,
          hidden_dim=8,
      )
  )
  def test_gnn_plus(self, num_layers, hidden_dim):

    class GnnPlusWrapper(nn.Module):
      """A contrived model that handles interleaving for testing."""

      def setup(self):
        self.gnn_plus = mlp.GnnPlus(
            num_layers=num_layers, hidden_dim=hidden_dim
        )

      def __call__(self, x, training=False):
        hprev = x
        hnext = x
        for i in range(self.gnn_plus.num_layers):
          hnext = self.gnn_plus(hprev, hnext, i, training)
        return hnext

    B = 10
    dummy_input = jax.random.normal(jax.random.PRNGKey(42), (B, hidden_dim))
    model = GnnPlusWrapper()
    params = model.init(jax.random.PRNGKey(42), dummy_input)
    output = model.apply(params, dummy_input)
    self.assertEqual(output.shape, (B, hidden_dim))


if __name__ == "__main__":
  absltest.main()
