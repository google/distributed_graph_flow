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

"""Unit tests for baselines.py."""

from absl.testing import absltest
import numpy as np

from dgf.src.stats.benchmark import baselines


class BaselinesTest(absltest.TestCase):

  def test_clopper_pearson_boundary(self):
    lo_0, _ = baselines.clopper_pearson(0, 10, confidence=0.95)
    self.assertEqual(lo_0, 0.0)

    _, hi_10 = baselines.clopper_pearson(10, 10, confidence=0.95)
    self.assertEqual(hi_10, 1.0)

  def test_student_t_coverage(self):
    rng = np.random.default_rng(20260923)
    reps = 2000
    n = 50
    draws = rng.normal(loc=0.0, scale=1.0, size=(reps, n))
    covered = 0
    for i in range(reps):
      lo, hi = baselines.student_t(draws[i], confidence=0.95)
      if lo <= 0.0 <= hi:
        covered += 1
    cov_rate = covered / reps
    self.assertBetween(cov_rate, 0.94, 0.96)

  def test_bca_bootstrap_symmetric(self):
    rng = np.random.default_rng(20260923)
    sample = rng.normal(loc=5.0, scale=1.0, size=100)
    sample_mean = float(np.mean(sample))
    lo, hi = baselines.bca_bootstrap(sample, confidence=0.95, num_resamples=1000, rng=rng)
    self.assertTrue(lo <= sample_mean <= hi, f"Sample mean {sample_mean} not in [{lo}, {hi}]")


if __name__ == "__main__":
  absltest.main()
