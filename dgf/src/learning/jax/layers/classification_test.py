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

"""Tests for the multi-class classification head."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.jax.layers import classification
import jax
import jax.numpy as jnp
import numpy as np


class ClassificationTest(parameterized.TestCase):

  def test_classification_head(self):
    """Tests the classification head."""
    key = jax.random.PRNGKey(0)
    batch_size = 2
    emb_size = 16
    num_classes = 5

    embedding = jnp.ones((batch_size, emb_size))
    head = classification.ClassificationHeadConfig(
        num_classes=num_classes
    ).make()

    params = head.init(key, embedding, training=True)
    logits = head.apply(params, embedding, training=True)

    self.assertEqual(logits.shape, (batch_size, num_classes))

  def test_logits_to_probability(self):
    """Tests the logits to probability conversion."""
    logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    probabilities = classification.ClassificationHead.logits_to_probability(
        logits
    )

    self.assertEqual(probabilities.shape, logits.shape)
    np.testing.assert_allclose(jnp.sum(probabilities, axis=-1), jnp.ones((2,)))
    self.assertTrue(jnp.all(probabilities >= 0))
    self.assertTrue(jnp.all(probabilities <= 1))


if __name__ == "__main__":
  absltest.main()
