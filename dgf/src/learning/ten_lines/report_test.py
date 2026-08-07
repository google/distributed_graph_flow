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

from unittest import mock
from absl.testing import absltest
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import report
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.util import log


class ReportTest(absltest.TestCase):

  def test_plot_html_training_logs_empty(self):
    logs = common.TrainingLogs(train=[], valid=[], num_train_step=0)
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
        num_train_step=2,
    )
    html = report.plot_html_training_logs(logs)

    self.assertIn("vega", html)
    self.assertIn("loss", html)
    self.assertIn("acc", html)
    self.assertIn("train", html)
    self.assertIn("valid", html)

  def test_html_log_messages(self):
    self.assertEqual(
        report.html_log_messages([]), "<i>No logs were captured.</i>"
    )

    logs = [
        log.Message.info("A < B"),
        log.Message.warning("C > D"),
        log.Message.error("E & F"),
    ]
    html = report.html_log_messages(logs)
    self.assertIn("A &lt; B", html)
    self.assertIn("blue", html)
    self.assertIn("C &gt; D", html)
    self.assertIn("orange", html)
    self.assertIn("E &amp; F", html)
    self.assertIn("red", html)

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

  def test_get_common_tabs(self):
    schema = schema_lib.GraphSchema(node_sets={}, edge_sets={})

    node_padding = padding_lib.NodeSetPadding(num_nodes=10)
    edge_padding = padding_lib.EdgeSetPadding(num_edges=20)
    pad = padding_lib.Padding(
        node_sets={"nodes": node_padding}, edge_sets={"edges": edge_padding}
    )

    plan = sampling_config_lib.SamplingPlan(
        root=sampling_config_lib.PlanNode(nodeset="my_root")
    )

    logs = common.TrainingLogs(train=[], valid=[], num_train_step=42)
    mock_stats = mock.MagicMock()

    tabs = report.get_common_tabs(
        hparams={"lr": 0.01},
        schemas={"MySchema": schema},
        feature_stats={"MyStats": mock_stats},
        sampling_plans={"MyPlan": plan},
        training_logs=logs,
        training_stats_summary="My Summary",
        padding={"MyPadding": pad},
        architecture="MyArchitecture",
        num_model_weights={"layer1": 100},
    )

    tabs_dict = dict(tabs)

    # training_logs & training_stats_summary
    self.assertIn("Training", tabs_dict)
    self.assertIn("My Summary", tabs_dict["Training"])

    # hparams
    self.assertIn("Hyper-parameters", tabs_dict)
    self.assertIn("lr", tabs_dict["Hyper-parameters"])

    # schemas
    self.assertIn("Schema", tabs_dict)
    self.assertIn("MySchema", tabs_dict["Schema"])

    # feature_stats
    self.assertIn("Feature statistics", tabs_dict)
    self.assertIn("MyStats", tabs_dict["Feature statistics"])

    # sampling_plans
    self.assertIn("Graph sampling", tabs_dict)
    self.assertIn("MyPlan", tabs_dict["Graph sampling"])
    self.assertIn("my_root", tabs_dict["Graph sampling"])

    # architecture & num_model_weights
    self.assertIn("Architecture", tabs_dict)
    self.assertIn("MyArchitecture", tabs_dict["Architecture"])
    self.assertIn("layer1", tabs_dict["Architecture"])

    # padding
    self.assertIn("Padding", tabs_dict)
    self.assertIn("MyPadding", tabs_dict["Padding"])
    self.assertIn("edges: 20 edges", tabs_dict["Padding"])


if __name__ == "__main__":
  absltest.main()
