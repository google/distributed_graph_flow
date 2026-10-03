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

"""Tests for generators.py analytic n_eff formulas."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dgf.src.stats.benchmark import generators


class GeneratorsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("rho03_m10", 0.3, 10),
      ("rho05_m20", 0.5, 20),
      ("rho02_m5", 0.2, 5),
      ("rho08_m50", 0.8, 50),
  )
  def test_analytic_neff_equicorr(self, rho: float, m_cluster: int):
    n = 2000
    reps = 20000
    rng = np.random.default_rng(20260923 + int(rho * 1000) + m_cluster)

    # Chunk into 4 batches of 5000 to manage memory while exercising sample_equicorr
    chunk_size = 5000
    num_chunks = reps // chunk_size

    for metric in ("mse", "mae"):
      sbars = []
      pooled_vars = []

      for _ in range(num_chunks):
        chunk_data = np.empty((chunk_size, n), dtype=np.float64)
        for i in range(chunk_size):
          chunk_data[i] = generators.sample_equicorr(n, rho, m_cluster, rng)

        if metric == "mse":
          s = chunk_data ** 2
        else:
          s = np.abs(chunk_data)

        # per-replicate summand mean
        sbar = np.mean(s, axis=1)
        sbars.append(sbar)
        pooled_vars.append(np.var(s, ddof=1))

      all_sbars = np.concatenate(sbars)
      var_sbar = float(np.var(all_sbars, ddof=1))
      var_s = float(np.mean(pooled_vars))

      n_eff_empirical = var_s / var_sbar
      analytic = generators.analytic_neff_equicorr(n, rho, m_cluster, metric)

      rel_diff = abs(n_eff_empirical - analytic) / analytic
      print(
          f"[{metric.upper()}] rho={rho} m={m_cluster}: "
          f"empirical={n_eff_empirical:.2f}, analytic={analytic:.2f}, "
          f"rel_diff={rel_diff * 100:.2f}%"
      )
      self.assertLess(
          rel_diff,
          0.05,
          f"Relative difference {rel_diff:.4f} exceeds 5% for metric={metric}, "
          f"rho={rho}, m_cluster={m_cluster} (empirical={n_eff_empirical:.2f}, "
          f"analytic={analytic:.2f})",
      )


if __name__ == "__main__":
  absltest.main()
