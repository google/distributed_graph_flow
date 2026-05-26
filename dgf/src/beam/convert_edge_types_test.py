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

"""Test converting between heterogeneous graph edge types."""

from absl.testing import absltest
from apache_beam.testing import test_pipeline
from dgf.src.data import distributed_graph
from dgf.src.util import test_util as gf_test_util

gf_test_util.disable_diff_truncation()
EdgeFormat = distributed_graph.EdgeFormat
Edge = distributed_graph.Edge
AdjacencyList = distributed_graph.AdjacencyList
PEdge = distributed_graph.PEdge


# TODO(bmayer): Complete me.
class ConvertEdgeTypesTest(absltest.TestCase):

  def test_convert_adjacency_to_flat_edges(self):
    with test_pipeline.TestPipeline() as p:
      pass
      # input_hgraph = gf_test_util.generate_fake_citation_graph(
      #     p, edge_format=EdgeFormat.FLAT
      # )
      # self.assertEqual(
      #     input_hgraph.edge_format, distributed_graph.EdgeFormat.FLAT
      # )

      # output_hgraph = convert_edge_types.convert_to_adjacency_list(input_hgraph)

      # self.assertEqual(
      #     output_hgraph.edge_format, distributed_graph.EdgeFormat.ADJACENCY
      # )


if __name__ == "__main__":
  absltest.main()
