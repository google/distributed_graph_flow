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

"""Tests for evaluation."""

from absl.testing import absltest
from dgf.src.learning.ten_lines import evaluation as evaluation_lib
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import metrics as sklearn_metrics


class EvaluationTest(absltest.TestCase):

  def test_compute_ranking_metrics_jax(self):
    pos_scores = jnp.array([0.5, 0.8])
    neg_scores = jnp.array([0.2, 0.6, 0.7, 0.9])
    metrics = evaluation_lib.compute_ranking_metrics(pos_scores, neg_scores)

    self.assertIn("mrr", metrics)
    self.assertIn("hit_at_1", metrics)
    self.assertIn("hit_at_5", metrics)
    self.assertIn("auc", metrics)

    # Check types
    self.assertIsInstance(metrics["mrr"], jax.Array)

    # Check values
    self.assertAlmostEqual(float(metrics["mrr"]), 0.5)
    self.assertAlmostEqual(float(metrics["hit_at_1"]), 0.0)
    self.assertAlmostEqual(float(metrics["hit_at_5"]), 1.0)
    self.assertAlmostEqual(float(metrics["auc"]), 0.5)

  def test_compute_ranking_metrics_numpy(self):
    pos_scores = np.array([0.5, 0.8])
    neg_scores = np.array([0.2, 0.6, 0.7, 0.9])
    metrics = evaluation_lib.compute_ranking_metrics(pos_scores, neg_scores)

    self.assertIn("mrr", metrics)
    self.assertIn("hit_at_1", metrics)
    self.assertIn("hit_at_5", metrics)
    self.assertIn("auc", metrics)

    # Check types
    self.assertIsInstance(metrics["mrr"], (np.ndarray, np.generic))

    # Check values
    self.assertAlmostEqual(float(metrics["mrr"]), 0.5)
    self.assertAlmostEqual(float(metrics["hit_at_1"]), 0.0)
    self.assertAlmostEqual(float(metrics["hit_at_5"]), 1.0)
    self.assertAlmostEqual(float(metrics["auc"]), 0.5)

  def test_compute_ranking_metrics_ties(self):
    pos_scores = np.array([0.5, 0.5])
    neg_scores = np.array([0.5] * 16)  # N=8
    metrics = evaluation_lib.compute_ranking_metrics(pos_scores, neg_scores)

    self.assertAlmostEqual(float(metrics["mrr"]), 0.2)
    self.assertAlmostEqual(float(metrics["hit_at_1"]), 1.0 / 9.0)
    self.assertAlmostEqual(float(metrics["hit_at_5"]), 5.0 / 9.0)
    self.assertAlmostEqual(float(metrics["auc"]), 0.5)

  def test_compute_ranking_metrics_nans(self):
    pos_scores = np.array([np.nan, 0.5])
    neg_scores = np.array([0.2] * 16)
    metrics = evaluation_lib.compute_ranking_metrics(pos_scores, neg_scores)

    self.assertTrue(np.isnan(float(metrics["mrr"])))
    self.assertTrue(np.isnan(float(metrics["hit_at_1"])))
    self.assertTrue(np.isnan(float(metrics["hit_at_5"])))
    self.assertTrue(np.isnan(float(metrics["auc"])))

  def test_classification_evaluation_accumulator(self):
    np.random.seed(42)
    num_classes = 3
    num_examples = 1000
    num_bins = 10000

    # Generate random predictions (logits then softmax)
    logits = np.random.randn(num_examples, num_classes)
    predictions = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    predictions = predictions.astype(np.float32)

    # Generate random targets
    targets = np.random.randint(0, num_classes, size=num_examples).astype(
        np.int32
    )

    accumulator = evaluation_lib.ClassificationEvaluationAccumulator(
        num_classes, num_bins
    )
    accumulator.add_predictions(predictions, targets)
    per_classes = accumulator.extract_metrics()

    self.assertLen(per_classes, num_classes)

    for c in range(num_classes):
      pc = per_classes[c]
      self.assertIsNotNone(pc.auc())
      self.assertIsNotNone(pc.pr_auc())
      self.assertLen(pc.tp, num_bins + 1)
      self.assertLen(pc.fp, num_bins + 1)
      self.assertLen(pc.tn, num_bins + 1)
      self.assertLen(pc.fn, num_bins + 1)
      self.assertLen(pc.thresholds, num_bins + 1)

      # Binary targets for class c
      binary_targets = (targets == c).astype(int)
      class_preds = predictions[:, c]

      # Sklearn ROC AUC
      sk_auc = sklearn_metrics.roc_auc_score(binary_targets, class_preds)
      self.assertAlmostEqual(pc.auc(), sk_auc, places=3)

      # Sklearn AP (Average Precision)
      sk_ap = sklearn_metrics.average_precision_score(
          binary_targets, class_preds
      )
      # Davis-Goadrich PR-AUC should be close to AP
      self.assertAlmostEqual(pc.pr_auc(), sk_ap, places=2)

      # Sklearn PR curve (for shape/value checks)
      sk_precision, sk_recall, _ = sklearn_metrics.precision_recall_curve(
          binary_targets, class_preds
      )
      # Sklearn PR AUC (trapezoidal - might be slightly different)
      sk_pr_auc = sklearn_metrics.auc(sk_recall, sk_precision)
      # Compare Davis-Goadrich with trapezoidal, should be close but DG is lower (usually)
      # With binning it might fluctuate slightly, but should be very close.
      self.assertAlmostEqual(pc.pr_auc(), sk_pr_auc, places=2)

  def test_evaluation_populate_from_accumulator(self):
    num_classes = 2
    accumulator = evaluation_lib.ClassificationEvaluationAccumulator(
        num_classes
    )

    # Add some dummy predictions
    preds = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    targets = np.array([1, 0], dtype=np.int32)
    accumulator.add_predictions(preds, targets)

    evaluation = evaluation_lib.Evaluation()
    accumulator.populate_evaluation(evaluation)

    self.assertIsNotNone(evaluation.auc)
    self.assertLen(evaluation.per_classes, num_classes)
    for pc in evaluation.per_classes:
      self.assertIsNotNone(pc.auc())
      self.assertIsNotNone(pc.pr_auc())
      self.assertLen(pc.tp, 10001)
      self.assertLen(pc.fp, 10001)
      self.assertLen(pc.tn, 10001)
      self.assertLen(pc.fn, 10001)
      self.assertLen(pc.thresholds, 10001)
      # Check properties
      self.assertLen(pc.fpr, 10001)
      self.assertLen(pc.tpr, 10001)
      self.assertLen(pc.precision, 10001)
      self.assertLen(pc.recall, 10001)

    # Test JSON serialization
    json_str = evaluation.to_json()
    loaded_eval = evaluation_lib.Evaluation.from_json(json_str)

    self.assertLen(loaded_eval.per_classes, num_classes)
    for i in range(num_classes):
      self.assertEqual(
          loaded_eval.per_classes[i].auc(), evaluation.per_classes[i].auc()
      )
      self.assertEqual(
          loaded_eval.per_classes[i].pr_auc(),
          evaluation.per_classes[i].pr_auc(),
      )
      self.assertLen(loaded_eval.per_classes[i].tp, 10001)  # Serialized

  def test_evaluation_html_with_curves(self):
    num_classes = 2
    accumulator = evaluation_lib.ClassificationEvaluationAccumulator(
        num_classes
    )
    preds = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    targets = np.array([1, 0], dtype=np.int32)
    accumulator.add_predictions(preds, targets)

    evaluation = evaluation_lib.Evaluation()
    accumulator.populate_evaluation(evaluation)
    html_output = evaluation.html()

    self.assertIn("<b>Evaluation</b>", html_output)
    self.assertIn("Per Class Metrics", html_output)
    # Check if Altair chart HTML is present (usually contains vegaEmbed or vg-canvas)
    self.assertIn("vegaEmbed", html_output)

  def test_evaluation_populate_fails_if_not_empty(self):
    num_classes = 2
    accumulator = evaluation_lib.ClassificationEvaluationAccumulator(
        num_classes
    )
    preds = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    targets = np.array([1, 0], dtype=np.int32)
    accumulator.add_predictions(preds, targets)

    evaluation = evaluation_lib.Evaluation()
    accumulator.populate_evaluation(evaluation)

    with self.assertRaises(ValueError):
      accumulator.populate_evaluation(evaluation)


if __name__ == "__main__":
  absltest.main()
