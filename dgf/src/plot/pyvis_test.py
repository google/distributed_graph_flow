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
from dgf.src.plot import pyvis as pyvis_plot
from dgf.src.util import gen_test_graph


class PyvisTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    pyvis_plot.options.reset()

  def tearDown(self):
    pyvis_plot.options.reset()
    super().tearDown()

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

  def test_plot_schema_with_none_option_fallback(self):
    schema = gen_test_graph.generate_schema()
    pyvis_plot.options.set("height", "500px")

    # Setting height=None instructs plot_schema to omit the kwarg
    # (falling back to pyvis default 600px).
    with pyvis_plot.option_context(height=None):
      net = pyvis_plot.plot_schema(schema)
      self.assertEqual(net.height, "600px")

    net = pyvis_plot.plot_schema(schema)
    self.assertEqual(net.height, "500px")  # Custom height is active again.

  def test_plot_schema_with_default_options_and_override(self):
    schema = gen_test_graph.generate_schema()
    pyvis_plot.options.set("height", "500px")

    net1 = pyvis_plot.plot_schema(schema)
    self.assertEqual(net1.height, "500px")  # Default options applied.

    net2 = pyvis_plot.plot_schema(schema, pyvis_kwargs={"height": "750px"})
    self.assertEqual(net2.height, "750px")  # Call-specific kwargs override.

    self.assertEqual(pyvis_plot.options.get("height"), "500px")  # Unchanged.


if __name__ == "__main__":
  absltest.main()
