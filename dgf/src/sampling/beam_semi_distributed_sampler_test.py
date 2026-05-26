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

import functools
import os
import tempfile
from absl.testing import absltest
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
from dgf.src.io import hgraph_in_beam
from dgf.src.sampling import beam_semi_distributed_sampler
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util


test_util.disable_diff_truncation()


class BeamSemiDistributedSamplerTest(absltest.TestCase):

  def test_extract_beam_nodes_ids(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "hgraph")
      gen_test_graph.generate_hgraph(path)

      with test_pipeline.TestPipeline() as p:
        graph = hgraph_in_beam.read_graphai_hgraph(p, path)
        seeds = beam_semi_distributed_sampler.extract_beam_nodes_ids(
            graph, "n1"
        )
        util.assert_that(
            seeds,
            util.equal_to(
                [b"1", b"2"],
                equals_fn=functools.partial(test_util.are_equal, abs_tol=0.001),
            ),
        )


if __name__ == "__main__":
  absltest.main()
