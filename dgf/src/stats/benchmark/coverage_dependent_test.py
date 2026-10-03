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

"""Smoke test for coverage_dependent benchmark harness."""

from absl.testing import absltest

from dgf.src.stats.benchmark import coverage_dependent


class CoverageDependentSmokeTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.cells, cls.redraw_counts = coverage_dependent.generate_all_cells(
        coverage_dependent.DEFAULT_SEED
    )

  def test_generate_cells(self):
    """Verifies cell generation produces exactly 136 cells across 6 blocks."""
    self.assertLen(self.cells, 136)
    blocks = set(c["block"] for c in self.cells)
    self.assertEqual(blocks, {"T1", "T2", "T2b", "T3", "G1", "F"})

  def test_quick_mode_smoke(self):
    """Runs quick mode on a tiny config (T1 cell with reps=2) and asserts loose bounds."""
    t1_cell = next(c for c in self.cells if c["block"] == "T1")
    mode_rows, stats, _ = coverage_dependent.evaluate_cell(
        t1_cell,
        reps=2,
        base_seed=coverage_dependent.DEFAULT_SEED,
    )
    self.assertNotEmpty(mode_rows)
    self.assertEqual(stats["name"], t1_cell["name"])
    self.assertEqual(stats["reps"], 2)


if __name__ == "__main__":
  absltest.main()
