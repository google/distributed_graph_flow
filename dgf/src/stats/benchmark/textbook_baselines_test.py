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

"""Smoke test for textbook_baselines benchmark."""

from absl.testing import absltest

from dgf.src.stats.benchmark import textbook_baselines


class TextbookBaselinesSmokeTest(absltest.TestCase):

  def test_get_target_cells(self):
    """Verifies that get_target_cells produces 16 cells (8 T1 + 8 G2)."""
    cells = textbook_baselines.get_target_cells(
        textbook_baselines.DEFAULT_SEED
    )
    self.assertLen(cells, 16)
    t1_cells = [c for c in cells if c["block"] == "T1"]
    g2_cells = [c for c in cells if c["block"] == "G2"]
    self.assertLen(t1_cells, 8)
    self.assertLen(g2_cells, 8)

  def test_quick_eval_t1(self):
    """Smoke tests _eval_cell on one T1 cell with tiny reps and boot_b."""
    cells = textbook_baselines.get_target_cells(
        textbook_baselines.DEFAULT_SEED, block_filter="T1"
    )
    self.assertNotEmpty(cells)
    res = textbook_baselines._eval_cell(
        cells[0],
        base_seed=textbook_baselines.DEFAULT_SEED,
        reps=5,
        boot_b=10,
        cache_dir="/tmp/dgf_cache",
    )
    self.assertIn("student_t", res)
    self.assertIn("bootstrap", res)
    self.assertIn("library", res)


if __name__ == "__main__":
  absltest.main()
