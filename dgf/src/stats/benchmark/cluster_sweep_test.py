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

"""Smoke tests for cluster_sweep benchmark harness."""

from absl.testing import absltest

from dgf.src.stats.benchmark import cluster_sweep


class ClusterSweepSmokeTest(absltest.TestCase):

  def test_suites_exist(self):
    """Verifies that all three suites are defined and non-empty."""
    self.assertIn("granularity", cluster_sweep.SUITES)
    self.assertIn("calibration", cluster_sweep.SUITES)
    self.assertIn("refutation", cluster_sweep.SUITES)
    self.assertNotEmpty(cluster_sweep.SUITES["granularity"])
    self.assertNotEmpty(cluster_sweep.SUITES["calibration"])
    self.assertNotEmpty(cluster_sweep.SUITES["refutation"])

  def test_quick_mode_smoke(self):
    """Runs one quick cell with tiny reps and asserts valid coverage metrics."""
    cell = cluster_sweep.SUITES["granularity"][0]
    res = cluster_sweep.evaluate_cell(
        cell,
        n=1000,
        reps=20,
        base_seed=cluster_sweep.DEFAULT_SEED,
        include_m_observed=True,
    )
    self.assertIn("miss_rate", res)
    self.assertIn("certified_rate", res)
    self.assertGreaterEqual(res["certified_rate"], 0.9)
    self.assertLessEqual(res["miss_rate"], 0.2)
    self.assertLessEqual(res["num_unrefuted_miss"], res["num_missed"])
    self.assertEqual(
        res["num_refuted"] + res["num_assumption_required"]
        + round(res["certified_rate"] * res["reps"]),
        res["reps"],
    )


if __name__ == "__main__":
  absltest.main()
