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
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from dgf.src.bin import validate_graph
from dgf.src.util import gen_test_graph

flags.FLAGS.set_default("path", "/dummy/path")


class ValidateGraphTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir().full_path
    self.graph_path = os.path.join(self.test_dir, "test_graph")
    gen_test_graph.generate_gf_graph(self.graph_path, edge_ids=False)

  def test_validate_graph_success(self):
    with flagsaver.flagsaver(path=self.graph_path, raise_on_error=True):
      validate_graph.main([])


if __name__ == "__main__":
  absltest.main()
