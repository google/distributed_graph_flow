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

"""Tests for reporter."""

import os
import tempfile

from absl.testing import absltest
from dgf.src.analyse.reports import data_model
from dgf.src.analyse.reports import reporter
from dgf.src.analyse.topology import global_graph_topology as global_graph_topology_lib


class ReporterTest(absltest.TestCase):

  def test_generate_report(self):
    ggt = global_graph_topology_lib.GlobalGraphTopology(
        total_nodes=1000,
        total_edges=5000,
    )
    payload = data_model.GraphStatsPayload(
        dataset_name="Test Dataset",
        task_type="Node Classification",
        feature_dimensionality=128,
        num_classes=5,
        global_graph_topology=ggt,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
      reporter.generate_report(payload, temp_dir)

      html_path = os.path.join(temp_dir, "report.html")
      pdf_path = os.path.join(temp_dir, "report.pdf")

      self.assertTrue(os.path.exists(html_path))
      self.assertTrue(os.path.exists(pdf_path))

      with open(html_path, "r") as f:
        html_content = f.read()
        self.assertIn("Test Dataset", html_content)
        self.assertIn("1,000", html_content)  # Number formatting check

      with open(pdf_path, "rb") as f:
        pdf_content = f.read()
        self.assertTrue(pdf_content.startswith(b"%PDF"))

  def test_generate_report_no_classes(self):
    ggt = global_graph_topology_lib.GlobalGraphTopology(
        total_nodes=1000,
        total_edges=5000,
    )
    payload = data_model.GraphStatsPayload(
        dataset_name="Unsupervised Dataset",
        task_type="Unsupervised Learning",
        feature_dimensionality=128,
        num_classes=None,
        global_graph_topology=ggt,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
      reporter.generate_report(payload, temp_dir)

      html_path = os.path.join(temp_dir, "report.html")
      pdf_path = os.path.join(temp_dir, "report.pdf")

      self.assertTrue(os.path.exists(html_path))
      self.assertTrue(os.path.exists(pdf_path))

      with open(html_path, "r") as f:
        html_content = f.read()
        self.assertIn("Unsupervised Dataset", html_content)
        self.assertNotIn("Num Classes", html_content)

      with open(pdf_path, "rb") as f:
        pdf_content = f.read()
        self.assertTrue(pdf_content.startswith(b"%PDF"))

  def test_generate_report_with_topology(self):
    ggt = global_graph_topology_lib.GlobalGraphTopology(
        total_nodes=100,
        total_edges=200,
        avg_degree=4.0,
        graph_density=0.04,
        num_connected_components=1,
        largest_component_size=100,
        isolated_nodes=0,
        graph_diameter=5.0,
        homophily_ratio=0.5,
        degree_distribution={"1": 10, "2": 50, "5": 40},
    )
    payload = data_model.GraphStatsPayload(
        dataset_name="Topology Dataset",
        task_type="Link Prediction",
        feature_dimensionality=64,
        global_graph_topology=ggt,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
      reporter.generate_report(payload, temp_dir)

      html_path = os.path.join(temp_dir, "report.html")
      self.assertTrue(os.path.exists(html_path))

      with open(html_path, "r") as f:
        html_content = f.read()
        self.assertIn("Topology Dataset", html_content)
        # Verify Chart.js data injection
        self.assertIn(
            'const rawDegreeData = {"1": 10, "2": 50, "5": 40}', html_content
        )
        self.assertIn(
            "const degreeScatterData = Object.entries(rawDegreeData)",
            html_content,
        )
        self.assertIn("const degreeBarCtx", html_content)
        self.assertIn("const degreeScatterCtx", html_content)
        self.assertIn("const degreeBarLogCtx", html_content)
        self.assertIn("const degreeScatterLogCtx", html_content)

  def test_graph_density_auto_calculation(self):
    ggt = global_graph_topology_lib.GlobalGraphTopology(
        total_nodes=5,
        total_edges=4,
    )
    # Max edges for 5 nodes = 5*4/2 = 10. Density = 4/10 = 0.4
    ggt.update_graph_density()  # Auto-calculate density
    self.assertIsNotNone(
        ggt.graph_density, "Graph density should be auto-calculated"
    )
    self.assertAlmostEqual(ggt.graph_density, 0.4)

  def test_graph_density_auto_calculation_single_node(self):
    ggt = global_graph_topology_lib.GlobalGraphTopology(
        total_nodes=1,
        total_edges=0,
    )
    ggt.update_graph_density()
    self.assertIsNone(
        ggt.graph_density, "Density should be None for single node graph"
    )


if __name__ == "__main__":
  absltest.main()
