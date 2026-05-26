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
from dgf.src.data import padding as padding_lib
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import report
from dgf.src.sampling import config as sampling_config_lib


class ReportTest(absltest.TestCase):

  def test_plot_html_training_logs_empty(self):
    logs = common.TrainingLogs(train=[], valid=[])
    html = report.plot_html_training_logs(logs)
    self.assertEqual(html, "<p>No training logs to display.</p>")

  def test_plot_html_training_logs_with_data(self):
    logs = common.TrainingLogs(
        train=[
            common.LogItem(step=0, metrics={"loss": 0.5, "acc": 0.8}),
            common.LogItem(step=1, metrics={"loss": 0.4, "acc": 0.9}),
        ],
        valid=[
            common.LogItem(step=0, metrics={"loss": 0.6, "acc": 0.7}),
            common.LogItem(step=2, metrics={"loss": 0.5, "acc": 0.8}),
        ],
    )
    html = report.plot_html_training_logs(logs)

    self.assertIn("vega", html)
    self.assertIn("loss", html)
    self.assertIn("acc", html)
    self.assertIn("train", html)
    self.assertIn("valid", html)

  def test_html_tabs(self):
    items = [
        ("Summary", "<p>This is the summary</p>"),
        ("Logs", "<div>Log data</div>"),
    ]
    html = report.html_tabs(items)

    self.assertIn("Summary", html)
    self.assertIn("Logs", html)
    self.assertIn("This is the summary", html)
    self.assertIn("Log data", html)
    self.assertIn("tab-btn", html)
    self.assertIn("tab-content", html)
    self.assertIn("style", html)
    self.assertIn("script", html)

  def test_get_common_tabs_padding(self):
    node_padding = padding_lib.NodeSetPadding(num_nodes=10)
    edge_padding = padding_lib.EdgeSetPadding(num_edges=20)
    pad = padding_lib.Padding(
        node_sets={"nodes": node_padding}, edge_sets={"edges": edge_padding}
    )

    padding = {"Default": pad}

    tabs = report.get_common_tabs(
        hparams={"lr": 0.01},
        schemas={},
        padding=padding,
    )

    padding_content = next(c for t, c in tabs if t == "Padding")

    self.assertIn("Default padding", padding_content)
    self.assertIn("Node Sets:", padding_content)
    self.assertIn("nodes: 10 nodes", padding_content)
    self.assertIn("Edge Sets:", padding_content)
    self.assertIn("edges: 20 edges", padding_content)

  def test_get_common_tabs_sampling_plan_label(self):
    plan = sampling_config_lib.SamplingPlan(
        root=sampling_config_lib.PlanNode(nodeset="my_root")
    )
    sampling_plans = {"Default": plan}

    tabs = report.get_common_tabs(
        hparams={"lr": 0.01},
        schemas={},
        sampling_plans=sampling_plans,
    )

    sampling_content = next(c for t, c in tabs if t == "Graph sampling")

    self.assertIn("Default sampling plan", sampling_content)
    self.assertIn("Root: my_root", sampling_content)


if __name__ == "__main__":
  absltest.main()
