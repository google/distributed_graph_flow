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
import tempfile
from absl.testing import absltest
from dgf.benchmark import in_process_sampling
from dgf.src.util import gen_test_graph


class InProcessIOTest(absltest.TestCase):

  def test_base(self):
    with tempfile.TemporaryDirectory() as tmpdir:

      # Prepare some data
      work_dir = os.path.join(tmpdir, "workdir")
      os.makedirs(work_dir, exist_ok=True)

      graph_path = os.path.join(tmpdir, "graph")
      gen_test_graph.generate_gf_graph(graph_path, edge_ids=False)

      # Run benchmark
      in_process_sampling.in_process_sampling(
          work_dir=work_dir,
          gf_graph_path=graph_path,
          seed_nodeset="n1",
          list_num_hops=[3],
      )


if __name__ == "__main__":
  absltest.main()
