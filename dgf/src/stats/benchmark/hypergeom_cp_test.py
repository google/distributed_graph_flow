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

"""Tests for hypergeom_cp.py."""

from absl.testing import absltest
import numpy as np
import scipy.stats

from dgf.src.stats.benchmark import hypergeom_cp


class HypergeomCpTest(absltest.TestCase):

  def test_quick_mode_smoke(self):
    records = hypergeom_cp.run_full_grid(pools=[50], alphas=[0.05])
    self.assertNotEmpty(records)
    for r in records:
      self.assertIn("max_hyper_ratio", r)
      self.assertTrue(0.0 <= r["max_hyper_lower_ratio"] <= 1.5)
      self.assertTrue(0.0 <= r["max_hyper_upper_ratio"] <= 1.5)

  def test_hypergeom_pmf_sums_to_one(self):
    """Verifies hypergeometric PMF sums to 1 within 1e-12 for multiple configurations."""
    test_cases = [
        (20, 5, 10),
        (50, 15, 20),
        (100, 30, 50),
        (200, 0, 10),
        (200, 200, 50),
    ]
    for pool_n, d, sample_n in test_cases:
      k_arr = np.arange(sample_n + 1)
      pmf = scipy.stats.hypergeom.pmf(k_arr, pool_n, d, sample_n)
      total_prob = float(np.sum(pmf))
      self.assertAlmostEqual(total_prob, 1.0, places=12)

  def test_binomial_reference_ratios_bound(self):
    """Verifies that for N=20, n=5, alpha=0.05, the Binomial reference ratios are <= 1 + 1e-9."""
    pool_n = 20
    sample_n = 5
    alpha = 0.05
    lows, highs = hypergeom_cp.compute_interval_cache(sample_n, alpha)
    rec = hypergeom_cp.evaluate_grid_cell(pool_n, sample_n, alpha, lows, highs)

    self.assertLessEqual(rec["binom_max_lower_ratio"], 1.0 + 1e-9)
    self.assertLessEqual(rec["binom_max_upper_ratio"], 1.0 + 1e-9)

  def test_sample_equals_pool_zero_miss(self):
    """Verifies that when n == N, the hypergeometric miss is exactly 0."""
    pool_n = 20
    sample_n = 20
    alpha = 0.05
    lows, highs = hypergeom_cp.compute_interval_cache(sample_n, alpha)
    rec = hypergeom_cp.evaluate_grid_cell(pool_n, sample_n, alpha, lows, highs)

    self.assertEqual(rec["max_hyper_lower_ratio"], 0.0)
    self.assertEqual(rec["max_hyper_upper_ratio"], 0.0)
    self.assertEqual(rec["max_hyper_ratio"], 0.0)

  def test_determinism(self):
    """Verifies that two runs on a small grid yield bitwise identical results."""
    pools = [10, 20]
    alphas = [0.05, 0.01]

    def run_small():
      res = []
      for p in pools:
        for s in [2, 5]:
          for a in alphas:
            lows, highs = hypergeom_cp.compute_interval_cache(s, a)
            res.append(hypergeom_cp.evaluate_grid_cell(p, s, a, lows, highs))
      return res

    run1 = run_small()
    run2 = run_small()

    self.assertEqual(len(run1), len(run2))
    for r1, r2 in zip(run1, run2):
      self.assertEqual(r1["max_hyper_ratio"], r2["max_hyper_ratio"])
      self.assertEqual(r1["max_hyper_lower_ratio"], r2["max_hyper_lower_ratio"])
      self.assertEqual(r1["max_hyper_upper_ratio"], r2["max_hyper_upper_ratio"])
      self.assertEqual(r1["num_d_exceeding"], r2["num_d_exceeding"])


if __name__ == "__main__":
  absltest.main()
