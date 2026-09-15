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

import os
from unittest import mock

from absl.testing import absltest
from dgf.src.gbbs import loader


class ParlayWorkersTest(absltest.TestCase):

  def test_set_num_parlay_workers_takes_effect(self):
    # The scheduler is already running by the time this test starts: it is
    # created when the extension module is imported.
    loader.set_num_parlay_workers(2)
    self.assertEqual(loader.num_parlay_workers(), 2)

    loader.set_num_parlay_workers(1)
    self.assertEqual(loader.num_parlay_workers(), 1)

  def test_set_num_parlay_workers_raises_when_env_var_is_set(self):
    with mock.patch.dict(os.environ, {"PARLAY_NUM_THREADS": "4"}):
      with self.assertRaisesRegex(RuntimeError, "PARLAY_NUM_THREADS is set"):
        loader.set_num_parlay_workers(2)


if __name__ == "__main__":
  absltest.main()
