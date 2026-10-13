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

"""Tests for the model analysis (model.analyze)."""

import re

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.ten_lines import analyze
from dgf.src.learning.ten_lines import node_prediction_train
from dgf.src.transform import perturb
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import jax
import numpy as np


def _make_item(
    name: str,
    category: analyze.ImportanceCategory,
    delta_loss: float,
    **kwargs,
) -> analyze.ImportanceItem:
  return analyze.ImportanceItem(
      category=category,
      name=name,
      delta_loss=delta_loss,
      delta_metric=delta_loss / 2,
      perturbed_loss=1.0 + delta_loss,
      perturbed_metric=0.9 - delta_loss / 2,
      **kwargs,
  )


def _make_analysis() -> analyze.ModelAnalysis:
  """A small hand-made analysis to test the rendering."""
  return analyze.ModelAnalysis(
      task_type="NODE_CLASSIFICATION",
      target_nodeset="N1",
      target_column="label",
      metric_name="accuracy",
      baseline_metric=0.9,
      baseline_loss=1.0,
      num_seed_nodes=100,
      edge_perturbation_mode=perturb.EdgePerturbationMode.MASLOV_SNEPPEN,
      duration_seconds=12.5,
      node_features=[
          _make_item(
              "N1.f1",
              analyze.ImportanceCategory.NODE_FEATURE,
              0.5,
              gradient_importance=0.1,
          ),
          _make_item("N1.f2", analyze.ImportanceCategory.NODE_FEATURE, 0.0),
      ],
      edgesets=[
          _make_item("N1_to_N2", analyze.ImportanceCategory.EDGESET, 0.2)
      ],
      nodesets=[_make_item("N2", analyze.ImportanceCategory.NODESET, 0.3)],
  )


class ModelAnalysisTest(parameterized.TestCase):

  @parameterized.parameters("NODESET", analyze.ImportanceCategory.NODESET)
  def test_parse_importance_category(self, category):
    self.assertEqual(
        analyze.parse_importance_category(category),
        analyze.ImportanceCategory.NODESET,
    )

  @parameterized.parameters("nodeset", "unknown")
  def test_parse_importance_category_error(self, category):
    with self.assertRaises(ValueError):
      analyze.parse_importance_category(category)

  def test_ranking(self):
    analysis = _make_analysis()
    self.assertEqual(
        [item.name for item in analysis.all_items],
        ["N1.f1", "N2", "N1_to_N2", "N1.f2"],
    )
    self.assertEqual(
        [item.name for item in analysis.features], ["N1.f1", "N1.f2"]
    )

  def test_to_dataframe(self):
    df = _make_analysis().to_dataframe()
    self.assertLen(df, 4)
    self.assertEqual(df["name"].tolist(), ["N1.f1", "N2", "N1_to_N2", "N1.f2"])
    self.assertEqual(df["category"].tolist()[0], "NODE_FEATURE")
    self.assertAlmostEqual(df["delta_loss"].iloc[0], 0.5)

  def test_text(self):
    text = repr(_make_analysis())
    for expected in [
        "ModelAnalysis(",
        "baseline_accuracy=0.9000",
        "edge_perturbation_mode=MASLOV_SNEPPEN",
        "[Features]",
        "[Edgesets]",
        "[Nodesets]",
        "N1.f1",
        "grad_saliency=0.1000",
    ]:
      self.assertIn(expected, text)

  def test_html_golden(self):
    html = _make_analysis().html()
    # Replace the random component ids to make the output deterministic.
    html = re.sub(r"(dgf[-_]analysis|tabs)[-_][0-9a-f]{8}", r"\1-ID", html)
    test_util.assert_golden_string(self, html, "model_analysis.html")

  def test_html(self):
    analysis = _make_analysis()
    html = analysis.html()
    for expected in [
        # Tabs.
        "Component importance",
        "Glossary",
        "tab-btn",
        # Details and importance table.
        "Details",
        "N1.label",
        "Δ Loss",
        "Δ Accuracy",
        "Gradient saliency",
        "dgf-filter-btn",
        "dgf-sortable",
        "N1.f1",
        "N1_to_N2",
        "dgf-badge-NODESET",
        # Glossary.
        "Perturbation",
        "Maslov-Sneppen",
    ]:
      self.assertIn(expected, html)
    for unexpected in ["Edge features", "Perturbed", "Prediction shift"]:
      self.assertNotIn(unexpected, html)

  def test_html_regression(self):
    analysis = _make_analysis()
    analysis.metric_name = "rmse"
    self.assertIn("Δ RMSE", analysis.html())


class AnalyzeNodeModelTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.graph, cls.schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=200,
        num_n2_nodes=300,
        random_seed=0,
        accuracy=0.95,
    )
    cls.model = node_prediction_train.train_node_model(
        cls.graph,
        cls.schema,
        target_nodeset="N1",
        target_column="label",
        num_train_steps=80,
        batch_size=16,
        verbose=0,
    )

  def test_classification(self):
    original_params = [
        np.copy(x) for x in jax.tree.leaves(self.model.data().model_params)
    ]
    analysis = analyze.analyze_node_model(
        self.model,
        self.graph,
        analyze.AnalyzeOptions(num_seed_nodes=200, verbose=0),
    )

    self.assertEqual(analysis.task_type, "NODE_CLASSIFICATION")
    self.assertEqual(analysis.target_nodeset, "N1")
    self.assertEqual(analysis.target_column, "label")
    self.assertEqual(analysis.metric_name, "accuracy")
    self.assertEqual(analysis.num_seed_nodes, 200)
    self.assertEqual(
        analysis.edge_perturbation_mode,
        perturb.EdgePerturbationMode.MASLOV_SNEPPEN,
    )
    self.assertContainsSubset(
        ["N1.f1", "N1.f3", "N2.f2"], [item.name for item in analysis.features]
    )
    self.assertCountEqual(
        [item.name for item in analysis.edgesets], ["N1_to_N2", "N1_to_N1"]
    )
    self.assertCountEqual(
        [item.name for item in analysis.nodesets], ["N1", "N2"]
    )

    # The signal feature N1.f1 is more important than the noise feature N1.f3.
    features = {item.name: item for item in analysis.features}
    for attr in ["delta_loss", "delta_metric"]:
      self.assertGreater(
          getattr(features["N1.f1"], attr),
          getattr(features["N1.f3"], attr),
          msg=attr,
      )
    for name in ["N1.f1", "N1.f3"]:
      self.assertGreater(features[name].gradient_importance or 0.0, 0.0)

    # The rendering works on a real analysis.
    self.assertIn("N1.f1", analysis.html())
    self.assertIn("N1.f1", repr(analysis))

    # The model parameters are not modified.
    test_util.assert_are_equal(
        self,
        original_params,
        [
            np.asarray(x)
            for x in jax.tree.leaves(self.model.data().model_params)
        ],
    )

  @parameterized.parameters(
      perturb.EdgePerturbationMode.MASLOV_SNEPPEN,
      perturb.EdgePerturbationMode.DROP,
      "MASLOV_SNEPPEN",
      "DROP",
  )
  def test_edge_perturbation_modes(self, edge_mode):
    options = analyze.AnalyzeOptions(
        num_seed_nodes=32,
        edge_perturbation_mode=edge_mode,
        compute_gradient_importance=False,
        include_nodesets=False,
        verbose=0,
    )
    analysis = analyze.analyze_node_model(self.model, self.graph, options)
    self.assertEqual(
        analysis.edge_perturbation_mode,
        perturb.parse_edge_perturbation_mode(edge_mode),
    )
    self.assertEqual(analysis.num_seed_nodes, 32)
    self.assertLen(analysis.edgesets, 2)
    self.assertEmpty(analysis.nodesets)
    for item in analysis.features:
      self.assertIsNone(item.gradient_importance)

  def test_regression(self):
    graph, schema = gen_test_graph.gen_toy_regression_dataset(
        num_n1_nodes=200, num_n2_nodes=100, random_seed=3
    )
    model = node_prediction_train.train_node_model(
        graph,
        schema,
        target_nodeset="N1",
        target_column="label",
        num_train_steps=20,
        batch_size=16,
        verbose=0,
    )
    analysis = model.analyze(graph, max_duration=15.0)
    self.assertEqual(analysis.task_type, "NODE_REGRESSION")
    self.assertEqual(analysis.metric_name, "rmse")
    self.assertIn("Δ RMSE", analysis.html())


if __name__ == "__main__":
  absltest.main()
