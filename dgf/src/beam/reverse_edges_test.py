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
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
from dgf.src.beam import reverse_edges as reverse_edges_lib
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import hgraph_in_beam
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util

test_util.disable_diff_truncation()


class ReverseEdgesTest(absltest.TestCase):

  def test_reverse_edges(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "hgraph")
      gen_test_graph.generate_hgraph(path, node_id=True)

      with test_pipeline.TestPipeline() as p:
        hgraph = hgraph_in_beam.read_graphai_hgraph(p, path)
        reversed_hgraph = reverse_edges_lib.reverse_edges(hgraph)
        util.assert_that(
            reversed_hgraph.edge_sets["e1"],
            util.equal_to(
                [
                    distributed_graph.Edge(id=b"a", source=b"1", target=b"1"),
                    distributed_graph.Edge(id=b"b", source=b"2", target=b"1"),
                ],
                equals_fn=test_util.are_equal,
            ),
        )
        util.assert_that(
            reversed_hgraph.edge_sets["e2"],
            util.equal_to(
                [
                    distributed_graph.Edge(source=1, target=b"1"),
                    distributed_graph.Edge(source=2, target=b"1"),
                ],
                equals_fn=test_util.are_equal,
            ),
        )

        self.assertSameElements(reversed_hgraph.node_sets.keys(), ["n1", "n2"])

        # Same nodesets as the original graph
        expected_schema = gen_test_graph.generate_schema(
            node_ids=True, semantic=False
        )
        for nodeset_def in expected_schema.node_sets.values():
          nodeset_def.features["#id"].semantic = (
              schema_lib.FeatureSemantic.PRIMARY_ID
          )
        self.assertEqual(
            reversed_hgraph.schema.node_sets,
            expected_schema.node_sets,
        )

        self.assertEqual(
            reversed_hgraph.schema.edge_sets,
            {
                "e2": schema_lib.EdgeSchema(
                    source="n2", target="n1", features={}
                ),
                "e1": schema_lib.EdgeSchema(
                    source="n1", target="n1", features={}
                ),
            },
        )


if __name__ == "__main__":
  absltest.main()
