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

"""Smoke and verification tests for coverage_independent benchmark."""

import math
from absl.testing import absltest
import numpy as np
import scipy.integrate
import scipy.stats

from dgf.src.stats.benchmark import coverage_independent


class CoverageIndependentSmokeTest(absltest.TestCase):

  def test_moments_accuracy(self):
    """Verifies moment formulas for Normal and Laplace against quad integration."""
    for sig in [0.5, 1.0, 3.2]:
      e1_quad, _ = scipy.integrate.quad(
          lambda x: abs(x) * scipy.stats.norm.pdf(x, scale=sig),
          -10 * sig,
          10 * sig,
      )
      f1, f2, f4 = coverage_independent.normal_moments(sig)
      self.assertLessEqual(abs(e1_quad - f1) / f1, 1e-7)

    for b in [0.8, 1.5, 4.0]:
      e1_quad, _ = scipy.integrate.quad(
          lambda x: abs(x) * scipy.stats.laplace.pdf(x, scale=b),
          -np.inf,
          np.inf,
      )
      f1, f2, f4 = coverage_independent.laplace_moments(b)
      self.assertLessEqual(abs(e1_quad - f1) / f1, 1e-7)

  def test_cell_generation(self):
    """Verifies that cell generation produces 106 cells across 6 blocks."""
    spec_rng = np.random.default_rng(coverage_independent.DEFAULT_SEED)
    blocks = [
        ("A", 60),
        ("B", 12),
        ("C", 8),
        ("D", 12),
        ("E", 8),
        ("F", 6),
    ]
    all_cells = []
    for b_id, count in blocks:
      for idx in range(count):
        cell, _ = coverage_independent.draw_cell_spec(b_id, idx, spec_rng)
        all_cells.append(cell)
    self.assertLen(all_cells, 106)

  def test_quick_mode_smoke(self):
    """Runs a quick evaluation across 1 cell per block."""
    spec_rng = np.random.default_rng(coverage_independent.DEFAULT_SEED)
    blocks = ["A", "B", "C", "D", "E", "F"]
    for b_id in blocks:
      cell, _ = coverage_independent.draw_cell_spec(b_id, 0, spec_rng)
      cell["cell_id"] = 0
      rows = coverage_independent.evaluate_cell_reps(
          cell, reps=5, base_seed=coverage_independent.DEFAULT_SEED
      )
      self.assertNotEmpty(rows)
      for r in rows:
        self.assertEqual(r["block"], b_id)
        if b_id == "F":
          self.assertEqual(r["returned"], 0)


if __name__ == "__main__":
  absltest.main()
