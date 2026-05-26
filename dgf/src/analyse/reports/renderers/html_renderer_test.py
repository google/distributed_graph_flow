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
from dgf.src.analyse.reports import data_model
from dgf.src.analyse.reports.renderers import html_renderer
from dgf.src.analyse.topology import global_graph_topology as global_graph_topology_lib
import networkx as nx


class HtmlRendererTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.renderer = html_renderer.HtmlRenderer()

  def test_tojson_filter_registered(self):
    """Verifies that 'tojson' filter is available in Jinja2 environment."""
    self.assertIn("tojson", self.renderer._env.filters)

  def test_render_basic_payload(self):
    """Verifies basic rendering without subgraphs."""
    ggt = global_graph_topology_lib.GlobalGraphTopology(
        total_nodes=10,
        total_edges=20,
    )
    payload = data_model.GraphStatsPayload(
        dataset_name="Test Dataset",
        feature_dimensionality=5,
        global_graph_topology=ggt,
    )
    html = self.renderer.render(payload)
    self.assertIn("Test Dataset", html)
    self.assertIn("10", html)  # Total Nodes

  def test_render_with_auto_subgraph_conversion(self):
    """Verifies subgraphs are auto-converted to visual_gallery_data."""
    g = nx.path_graph(3)
    ggt = global_graph_topology_lib.GlobalGraphTopology(
        total_nodes=3,
        total_edges=2,
    )
    payload = data_model.GraphStatsPayload(
        dataset_name="Visual Test",
        feature_dimensionality=1,
        subgraphs=[g],
        global_graph_topology=ggt,
    )

    # Ensure visual_gallery_data is initially empty
    self.assertIsNone(payload.visual_gallery_data)

    html = self.renderer.render(payload)

    # Verify visual_gallery_data was populated
    self.assertIsNotNone(payload.visual_gallery_data)
    self.assertLen(payload.visual_gallery_data, 1)

    # Verify HTML contains the gallery section/markers
    self.assertIn("Visual Inspection Gallery", html)
    self.assertIn("mynetwork_1", html)
    # Check for safe json output (no escaped quotes)
    self.assertIn('"nodes":', html)
    self.assertNotIn("&#34;", html)

  def test_render_with_explicit_visual_gallery_data(self):
    """Verifies explicit visual_gallery_data is verified."""
    ggt = global_graph_topology_lib.GlobalGraphTopology(
        total_nodes=10,
        total_edges=10,
    )
    payload = data_model.GraphStatsPayload(
        dataset_name="Explicit Visual Data",
        feature_dimensionality=1,
        visual_gallery_data=[
            {"nodes": [{"id": 1, "label": "A"}], "edges": [], "options": {}}
        ],
        global_graph_topology=ggt,
    )
    html = self.renderer.render(payload)
    self.assertIn("mynetwork_1", html)
    self.assertIn('"label": "A"', html)


if __name__ == "__main__":
  absltest.main()
