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

"""Tests for dgf.src.plot.pyvis."""

from absl.testing import absltest
from dgf.src.plot import pyvis as pyvis_plot
from dgf.src.util import gen_test_graph


class PyvisTest(absltest.TestCase):

  def test_plot_schema(self):
    schema = gen_test_graph.generate_schema()
    net = pyvis_plot.plot_schema(schema, features=True)

    self.assertIsNotNone(net)
    nodes = net.get_nodes()
    self.assertIn("n1", nodes)
    self.assertIn("n2", nodes)

    html = net.generate_html()
    self.assertIn("n1", html)
    self.assertIn("n2", html)


if __name__ == "__main__":
  absltest.main()
