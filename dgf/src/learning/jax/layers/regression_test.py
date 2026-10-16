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

from absl.testing import absltest
from dgf.src.learning.jax.layers import regression
import jax
import jax.numpy as jnp
import numpy as np

RegressionHeadConfig = regression.RegressionHeadConfig


class RegressionHeadTest(absltest.TestCase):

  def test_no_standardization(self):
    config = RegressionHeadConfig()
    logits = jnp.array([[1.0], [2.0]])
    np.testing.assert_allclose(config.logits_to_predictions(logits), [1.0, 2.0])
    np.testing.assert_allclose(
        config.standardize_labels(jnp.array([3.0, 4.0])), [3.0, 4.0]
    )
    self.assertEqual(config.architecture(), "Dense(1) # Regression head")

  def test_standardization(self):
    config = RegressionHeadConfig(label_mean=10.0, label_std=2.0)
    labels = jnp.array([10.0, 12.0, 6.0])
    standardized = config.standardize_labels(labels)
    np.testing.assert_allclose(standardized, [0.0, 1.0, -2.0])
    np.testing.assert_allclose(
        config.logits_to_predictions(standardized[:, None]), labels
    )
    self.assertEqual(
        config.architecture(), "Dense(1) * 2 + 10 # Regression head"
    )

  def test_initial_predictions_are_the_label_mean(self):
    config = RegressionHeadConfig(label_mean=10.0, label_std=2.0)
    head = config.make()
    x = jnp.ones((3, 4))
    params = head.init(jax.random.PRNGKey(0), x, training=False)
    logits = head.apply(params, x, training=False)
    self.assertEqual(logits.shape, (3, 1))
    np.testing.assert_allclose(config.logits_to_predictions(logits), [10.0] * 3)

  def test_json_round_trip(self):
    config = RegressionHeadConfig(num_outputs=2, label_mean=1.0, label_std=2.0)
    restored = RegressionHeadConfig.from_json(config.to_json())  # pyrefly: ignore[missing-attribute]
    self.assertEqual(restored, config)


if __name__ == "__main__":
  absltest.main()
