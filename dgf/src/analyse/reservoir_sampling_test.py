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
from absl.testing import parameterized
from dgf.src.analyse import reservoir_sampling
import numpy as np


class ReservoirSamplingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.cache_size = 1_000

  def test_initial_is_empty(self):
    r = reservoir_sampling.BatchReservoirSampling(self.cache_size)
    self.assertEqual(r._num_in_cache, 0)
    self.assertEqual(r._num_seen, 0)

  def test_add_small_data(self):
    r = reservoir_sampling.BatchReservoirSampling(self.cache_size)
    r.add(np.random.randn(self.cache_size // 4))
    self.assertEqual(r._num_in_cache, self.cache_size // 4)
    self.assertEqual(r._num_seen, self.cache_size // 4)
    self.assertEqual(r._samples.shape, (self.cache_size,))

  def test_add_multiple_small_data(self):
    r = reservoir_sampling.BatchReservoirSampling(self.cache_size)
    r.add(np.random.randn(self.cache_size // 4))
    r.add(np.random.randn(self.cache_size // 4))
    n = 2 * (self.cache_size // 4)
    self.assertEqual(r._num_in_cache, n)
    self.assertEqual(r._num_seen, n)
    self.assertEqual(r._samples.shape, (self.cache_size,))

  def test_add_large_data(self):
    r = reservoir_sampling.BatchReservoirSampling(self.cache_size)
    r.add(np.random.randn(self.cache_size * 2))
    self.assertEqual(r._num_in_cache, self.cache_size)
    self.assertEqual(r._num_seen, self.cache_size * 2)
    self.assertEqual(r._samples.shape, (self.cache_size,))

  def test_add_multiple_chunks(self):
    r = reservoir_sampling.BatchReservoirSampling(self.cache_size)
    n = 2 * (self.cache_size // 3)
    r.add(np.random.randn(n))
    r.add(np.random.randn(n))
    r.add(np.random.randn(n))
    self.assertEqual(r._num_in_cache, self.cache_size)
    self.assertEqual(r._num_seen, 3 * n)
    self.assertEqual(r._samples.shape, (self.cache_size,))

  @parameterized.parameters(
      (10, 5000, 10_000),
      (200, 5000, 10_000),
      (20_000, 10, 10_000),
  )
  def test_approx_quantiles_small(self, batch_size, num_batches, cache_size):
    r = reservoir_sampling.BatchReservoirSampling(cache_size)
    values = []
    for i in range(num_batches):
      batch = np.random.randn(batch_size) * i / num_batches
      values.append(batch)
      r.add(batch)
    quantiles, thresholds = r.get_quantiles(10)
    expected_quantiles = np.nanquantile(
        np.concatenate(values, axis=0), thresholds
    )
    # The first and last quantiles  (i.e., min/max) are very noisy.
    quantiles = quantiles[1:-1]
    expected_quantiles = expected_quantiles[1:-1]
    max_diff = np.max(np.absolute(quantiles - expected_quantiles))
    self.assertLessEqual(
        max_diff,
        0.05,
        msg=f"quantiles={quantiles} expected_quantiles={expected_quantiles}",
    )

  def test_cache1(self):
    samples = []
    for _ in range(1000):
      sampler = reservoir_sampling.BatchReservoirSampling(cache_size=1)
      sampler.add(np.array([0]))
      sampler.add(np.array([1]))
      samples.append(sampler._samples[0])
    rate = np.mean(samples)
    self.assertAlmostEqual(rate, 0.5, delta=0.122)


if __name__ == "__main__":
  absltest.main()
