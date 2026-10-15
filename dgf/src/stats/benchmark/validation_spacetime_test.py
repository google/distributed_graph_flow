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

"""Tests for validation_spacetime module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dgf.src.stats.benchmark import validation_spacetime
from dgf.src.stats.independent import interval as _ind_interval

Status = _ind_interval.Status


class ValidationSpacetimeTest(parameterized.TestCase):

  def test_quick_mode_smoke(self):
    """Runs a quick evaluation across 1 cell."""
    cells = validation_spacetime.get_all_cells()
    res = validation_spacetime.run_cell_simulation(cells[0], reps_override=5)
    self.assertEqual(res["cell_id"], cells[0]["cell_id"])
    self.assertIn("coverage_rate", res)

  def test_cell_definitions_count_and_blocks(self):
    """Verifies all 32 cells are defined across the 5 blocks."""
    cells = validation_spacetime.get_all_cells()
    self.assertEqual(len(cells), 32)

    blocks = [c["block"] for c in cells]
    self.assertEqual(blocks.count("ST1"), 8)
    self.assertEqual(blocks.count("ST2"), 4)
    self.assertEqual(blocks.count("ST3a"), 8)
    self.assertEqual(blocks.count("ST3b"), 8)
    self.assertEqual(blocks.count("ST4"), 2)
    self.assertEqual(blocks.count("ST5"), 2)

    cell_ids = [c["cell_id"] for c in cells]
    self.assertEqual(cell_ids, list(range(1, 33)))

  def test_missingness_masks_counts(self):
    """Verifies exact item counts for complete, 10% and 30% missing masks."""
    num_nodes = 40
    num_times = 300

    m_complete = validation_spacetime.generate_missing_mask(
        num_nodes, num_times, 0.0, "complete", seed=1
    )
    self.assertEqual(np.sum(m_complete), 12000)

    m_mcar_10 = validation_spacetime.generate_missing_mask(
        num_nodes, num_times, 0.10, "mcar", seed=2
    )
    self.assertEqual(np.sum(m_mcar_10), 10800)

    m_mcar_30 = validation_spacetime.generate_missing_mask(
        num_nodes, num_times, 0.30, "mcar", seed=3
    )
    self.assertEqual(np.sum(m_mcar_30), 8400)

    m_block_10 = validation_spacetime.generate_missing_mask(
        num_nodes, num_times, 0.10, "block_outage", seed=4
    )
    self.assertEqual(np.sum(m_block_10), 10800)

    m_block_30 = validation_spacetime.generate_missing_mask(
        num_nodes, num_times, 0.30, "block_outage", seed=5
    )
    self.assertEqual(np.sum(m_block_30), 8400)

  @parameterized.named_parameters(
      ("st1_mse", 1),
      ("st2_regimeA_mse", 9),
      ("st3a_mcar10_mse", 13),
      ("st3b_mcar10_mse", 21),
      ("st4_r2_complete", 29),
      ("st5a_drift", 31),
      ("st5b_non_product", 32),
  )
  def test_smoke_cell_simulation(self, cell_id: int):
    """Smoke test running 2 repetitions per representative cell."""
    all_cells = validation_spacetime.get_all_cells()
    cell = [c for c in all_cells if c["cell_id"] == cell_id][0]

    res = validation_spacetime.run_cell_simulation(cell, reps_override=2)
    self.assertEqual(res["cell_id"], cell_id)
    self.assertEqual(res["reps"], 2)
    self.assertIn("coverage_rate", res)
    self.assertIn("mean_rel_width", res)
    self.assertIn("pass_cell", res)


if __name__ == "__main__":
  absltest.main()
