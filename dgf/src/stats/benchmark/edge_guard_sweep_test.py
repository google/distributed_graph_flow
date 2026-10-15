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

"""Smoke tests for edge_guard_sweep benchmark."""

from absl.testing import absltest
from dgf.src.stats.benchmark import edge_guard_sweep


class EdgeGuardSweepTest(absltest.TestCase):

  def test_generate_cells(self):
    cells = edge_guard_sweep.generate_cells()
    self.assertLen(cells, 29)

  def test_quick_mode_smoke(self):
    cell = edge_guard_sweep.generate_cells()[0]
    res = edge_guard_sweep.run_cell_simulation(
        cell, reps=5, base_seed=edge_guard_sweep.DEFAULT_SEED
    )
    self.assertIn("reject_guarded", res)
    self.assertTrue(0.0 <= res["reject_guarded"] <= 1.0)
    self.assertTrue(0.0 <= res["untestable_guard"] <= 1.0)


if __name__ == "__main__":
  absltest.main()
