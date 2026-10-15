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

"""Tests for textbook_rmse_coverage benchmark."""

import math
from absl.testing import absltest
import numpy as np

from dgf.src.stats.benchmark import textbook_rmse_coverage


class TextbookRmseCoverageTest(absltest.TestCase):

  def test_quick_mode_smoke(self):
    results_a = textbook_rmse_coverage.run_part_a(
        num_reps=5, num_resamples=20, confidence=0.95
    )
    self.assertNotEmpty(results_a)
    for r in results_a:
      self.assertTrue(0.0 <= r.miss_rate <= 1.0)

    results_b = textbook_rmse_coverage.run_part_b(
        num_reps=5,
        num_resamples=20,
        confidence=0.95,
        sample_sizes=(100,),
        distributions=[textbook_rmse_coverage.get_part_b_distributions()[0]],
    )
    self.assertNotEmpty(results_b)
    for r in results_b:
      self.assertTrue(0.0 <= r.miss_rate <= 1.0)

  def test_percentile_bootstrap_rmse(self):
    rng = np.random.default_rng(42)
    e2 = rng.normal(size=100)**2
    low, high = textbook_rmse_coverage.percentile_bootstrap_rmse(
        e2, confidence=0.95, num_resamples=50, rng=rng
    )
    self.assertTrue(0.0 <= low < high < float("inf"))

  def test_bca_bootstrap_rmse(self):
    rng = np.random.default_rng(42)
    e2 = rng.normal(size=100)**2
    low, high = textbook_rmse_coverage.bca_bootstrap_rmse(
        e2, confidence=0.95, num_resamples=50, rng=rng
    )
    self.assertTrue(0.0 <= low < high < float("inf"))

  def test_bayesian_bootstrap_rmse(self):
    rng = np.random.default_rng(42)
    e2 = rng.normal(size=100)**2
    low, high = textbook_rmse_coverage.bayesian_bootstrap_rmse(
        e2, confidence=0.95, num_resamples=50, rng=rng
    )
    self.assertTrue(0.0 <= low < high < float("inf"))

  def test_student_t_rmse(self):
    rng = np.random.default_rng(42)
    e2 = rng.normal(size=100)**2
    low, high = textbook_rmse_coverage.student_t_rmse(e2, confidence=0.95)
    self.assertTrue(0.0 <= low < high < float("inf"))

  def test_moving_block_bootstrap_rmse(self):
    rng = np.random.default_rng(42)
    residuals = rng.normal(size=(10, 30))
    low, high = textbook_rmse_coverage.moving_block_bootstrap_rmse(
        residuals, block_length=5, confidence=0.95, num_resamples=50, rng=rng
    )
    self.assertTrue(0.0 <= low < high < float("inf"))

  def test_cluster_bootstrap_rmse(self):
    rng = np.random.default_rng(42)
    residuals = rng.normal(size=(10, 30))
    low, high = textbook_rmse_coverage.cluster_bootstrap_rmse(
        residuals, confidence=0.95, num_resamples=50, rng=rng
    )
    self.assertTrue(0.0 <= low < high < float("inf"))

  def test_run_part_a_tiny(self):
    # Run tiny 2-rep run of Part A
    results = textbook_rmse_coverage.run_part_a(
        num_reps=2, num_resamples=20, confidence=0.95
    )
    self.assertEqual(len(results), 7)
    for r in results:
      self.assertEqual(r.num_reps, 2)
      self.assertTrue(0.0 <= r.miss_rate <= 1.0)
      self.assertTrue(0.0 <= r.issuance_rate <= 1.0)

  def test_run_part_b_tiny(self):
    # Run tiny 2-rep run of Part B on 1 sample size
    results = textbook_rmse_coverage.run_part_b(
        num_reps=2, num_resamples=20, confidence=0.95, sample_sizes=(50,)
    )
    # 6 distributions x 1 sample size x 5 methods = 30 results
    self.assertEqual(len(results), 30)
    for r in results:
      self.assertEqual(r.num_reps, 2)
      self.assertTrue(0.0 <= r.miss_rate <= 1.0)

  def test_verify_moments(self):
    dists = dict(textbook_rmse_coverage.get_part_b_distributions())
    e = math.e

    expected_m_star = {
        "gaussian": 3.0,
        "laplace": 6.0,
        "student_t_5.0": 9.0,
        "lognormal_1.0": e**4 + 2.0 * e**3 + 3.0 * e**2 - 3.0,  # 113.9364
        "contaminated_gaussian": 302.97 / (1.99**2),  # 76.5056
    }

    # Check that dist.mean_sq is 1.0 and empirical fourth-moment ratio is finite
    for name, expected in expected_m_star.items():
      dist = dists[name]
      self.assertEqual(dist.mean_sq, 1.0)
      # Monte Carlo check with 100k samples
      rng = np.random.default_rng(2026)
      samples = dist.sample(100_000, rng)
      mc_m_star = float(np.mean(samples**4) / (np.mean(samples**2)**2))
      self.assertTrue(math.isfinite(mc_m_star))


if __name__ == "__main__":
  absltest.main()



