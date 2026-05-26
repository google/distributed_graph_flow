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
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.plot import network as network_lib
from dgf.src.util import gen_test_graph


def is_test_running_localy():
  """Tries to detect whether the test is running locally or on Forge."""
  return "UNITTEST_ON_BORG" not in os.environ


class PlotInMemoryGraphTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_plot_schema(self, features):
    schema = gen_test_graph.generate_schema()
    p = network_lib.plot_schema(schema, features=features)
    if features:
      golden_value = """\
// Graph Schema
digraph {
	n1 [label=<<b>n1</b><br/>f1<br/>f2> shape=box]
	n2 [label=<<b>n2</b><br/>f3<br/>f4<br/>f5> shape=box]
	n1 -> n1 [label=e1]
	n1 -> n2 [label=e2]
}
"""
    else:
      golden_value = """\
// Graph Schema
digraph {
	n1 [label=n1 shape=ellipse]
	n2 [label=n2 shape=ellipse]
	n1 -> n1 [label=e1]
	n1 -> n2 [label=e2]
}
"""
    self.assertEqual(p.source, golden_value)

    if is_test_running_localy():
      # Save the plot to a temporary file.
      temp_dir = self.create_tempdir()
      temp_file = os.path.join(temp_dir, f"schema_plot_{features}")
      p.render(temp_file, format="png", cleanup=True)

  @parameterized.parameters(True, False)
  def test_plot_graph(self, features):
    graph = gen_test_graph.generate_in_memory_graph(True, True)
    schema = gen_test_graph.generate_schema(variable_length=False)
    p = network_lib.plot_graph(graph, schema, features=features)

    if features:
      golden_value = """\
// In-Memory Graph
digraph {
	graph [rankdir=LR]
	n1_0 [label=<<b>n1_0</b><br/>f1: [b'blue']<br/>f2: [0. 1.]> fillcolor=pink shape=box style=filled]
	n1_1 [label=<<b>n1_1</b><br/>f1: [b'red']<br/>f2: [2. 3.]> fillcolor=pink shape=box style=filled]
	n2_0 [label=<<b>n2_0</b><br/>f3: 4<br/>f4: 10> fillcolor=orange shape=box style=filled]
	n2_1 [label=<<b>n2_1</b><br/>f3: 5<br/>f4: 11> fillcolor=orange shape=box style=filled]
	n1_0 -> n1_0 [label=<<b>e1</b><br/>#id: b'a'> color=green fontcolor=green]
	n1_0 -> n1_1 [label=<<b>e1</b><br/>#id: b'b'> color=green fontcolor=green]
	n1_0 -> n2_0 [label=<<b>e2</b><br/>#id: b'A'> color=blue fontcolor=blue]
	n1_0 -> n2_1 [label=<<b>e2</b><br/>#id: b'B'> color=blue fontcolor=blue]
}
"""
    else:
      golden_value = """\
// In-Memory Graph
digraph {
	graph [rankdir=LR]
	n1_0 [label=n1_0 fillcolor=pink shape=box style=filled]
	n1_1 [label=n1_1 fillcolor=pink shape=box style=filled]
	n2_0 [label=n2_0 fillcolor=orange shape=box style=filled]
	n2_1 [label=n2_1 fillcolor=orange shape=box style=filled]
	n1_0 -> n1_0 [label=e1 color=green fontcolor=green]
	n1_0 -> n1_1 [label=e1 color=green fontcolor=green]
	n1_0 -> n2_0 [label=e2 color=blue fontcolor=blue]
	n1_0 -> n2_1 [label=e2 color=blue fontcolor=blue]
}
"""

    self.assertEqual(p.source, golden_value)

    if is_test_running_localy():
      # Save the plot to a temporary file.
      temp_dir = self.create_tempdir()
      temp_file = os.path.join(temp_dir, f"in_memory_graph_plot_{features}")
      p.render(temp_file, format="png", cleanup=True)


if __name__ == "__main__":
  absltest.main()
