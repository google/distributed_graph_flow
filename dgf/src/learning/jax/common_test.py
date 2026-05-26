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

"""Tests for common jax utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from dgf.src.learning.jax import common
import jax.numpy as jnp


class JaxUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="relu",
          activation_name="relu",
          input_arr=[-0.9, -0.1, 0.1, 0.2],
          expected_arr=[0.0, 0.0, 0.1, 0.2],
      ),
      dict(
          testcase_name="sigmoid",
          activation_name="sigmoid",
          input_arr=[-0.9, -0.1, 0.1, 0.2],
          expected_arr=[0.2895, 0.4755, 0.5244, 0.5498],
      ),
  )
  def test_get_activation(self, activation_name, input_arr, expected_arr):
    activation_fn = common.get_activation(activation_name)
    chex.assert_trees_all_close(
        activation_fn(jnp.array(input_arr, dtype=jnp.float32)),
        jnp.array(expected_arr, dtype=jnp.float32),
        atol=1e-3
    )

  def test_get_activation_error(self):
    with self.assertRaisesRegex(
        AttributeError, "Activation foo_bar not found in jax.nn or flax.nn"
    ):
      common.get_activation("foo_bar")

  @parameterized.named_parameters(
      dict(
          testcase_name="bfloat16",
          dtype_name="bfloat16",
          expected_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="float32",
          dtype_name="float32",
          expected_dtype=jnp.float32,
      ),
      dict(
          testcase_name="int64",
          dtype_name="int64",
          expected_dtype=jnp.int64,
      ),
      dict(
          testcase_name="int8",
          dtype_name="int8",
          expected_dtype=jnp.int8,
      ),
  )
  def test_jnp_dtype_to_from_string(self, dtype_name, expected_dtype):
    dtype = common.jnp_dtype_from_string(dtype_name)
    self.assertEqual(dtype, expected_dtype)

    name = common.jnp_name_from_dtype(dtype)
    self.assertEqual(name, dtype_name)

  def test_jit_gather_features(self):
    features = {
        "feat1": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "feat2": jnp.array([10, 20, 30]),
    }
    node_idxs = jnp.array([0, 2])
    gathered = common.jit_gather_features(features, node_idxs)
    expected = {
        "feat1": jnp.array([[1.0, 2.0], [5.0, 6.0]]),
        "feat2": jnp.array([10, 30]),
    }
    chex.assert_trees_all_close(gathered, expected)


if __name__ == "__main__":
  absltest.main()
