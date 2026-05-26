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

"""Tests for io.networkx."""

from absl.testing import absltest
from dgf.src.io import networkx as networkx_io_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util


class NetworkXTest(absltest.TestCase):

  def test_graph_to_networkx_and_back(self):
    in_memory_graph = gen_test_graph.generate_in_memory_graph(
        variable_length=False
    )
    schema = gen_test_graph.generate_schema(variable_length=False)

    nx_graph = networkx_io_lib.graph_to_networkx(
        in_memory_graph=in_memory_graph,
        schema=schema,
    )

    recovered_graph, recovered_schema = networkx_io_lib.networkx_to_graph(
        nx_graph=nx_graph,
    )

    test_util.assert_are_equal(self, in_memory_graph, recovered_graph)
    self.assertIn("n1", recovered_schema.node_sets)
    self.assertIn("e1", recovered_schema.edge_sets)


if __name__ == "__main__":
  absltest.main()
