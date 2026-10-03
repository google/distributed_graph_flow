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

"""Smoke test for partitioner_robustness benchmark."""

from absl.testing import absltest
from dgf.src.stats.benchmark import partitioner_robustness


class PartitionerRobustnessTest(absltest.TestCase):

  def test_quick_mode_smoke(self):
    results, out_of_band = partitioner_robustness.run_partitioner_sweep(
        n=20000,
        topologies=("torus",),
        cs_list=(50,),
        seeds_list=(0,),
    )
    self.assertLen(results, 1)
    res = results[0]
    self.assertEqual(res["topology"], "torus")
    self.assertEqual(res["cs"], 50)
    self.assertTrue(res["G"] > 0)
    self.assertTrue(0.80 <= res["ratio"] <= (4.0 / 3.0))


if __name__ == "__main__":
  absltest.main()
