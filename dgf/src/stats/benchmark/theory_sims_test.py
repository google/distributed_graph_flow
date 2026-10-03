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

"""Smoke test for theory_sims benchmark binary."""

from absl.testing import absltest
from dgf.src.stats.benchmark import theory_sims


class TheorySimsTest(absltest.TestCase):

  def test_quick_mode_smoke(self):
    records = theory_sims.run_all(mode="quick", seed=theory_sims.DEFAULT_SEED)
    self.assertNotEmpty(records)
    sims = set(r["sim"] for r in records)
    self.assertEqual(sims, {"S1", "S2", "S3", "S4", "S5", "S6", "S7"})


if __name__ == "__main__":
  absltest.main()
