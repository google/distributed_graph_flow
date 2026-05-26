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

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import dgf
from dgf import beam as dgf_beam


class TestAPITest(parameterized.TestCase):

  def test_empty(self):
    pass

  # Note: We only check the linter.
  def disabled_test_read_in_process(self):
    _ = dgf.io.read_graphai_hgraph("/some/path")

  def disabled_test_read_distributed(self):
    with beam.Pipeline() as pbegin:
      _ = dgf_beam.io.read_graphai_hgraph(pbegin, "/some/path")


if __name__ == "__main__":
  absltest.main()
