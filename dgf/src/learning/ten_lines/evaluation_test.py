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
from dgf.src.learning.ten_lines import evaluation
import jax
import jax.numpy as jnp
import numpy as np


class EvaluationTest(absltest.TestCase):

  def test_evaluation_html(self):
    eval_obj = evaluation.Evaluation(
        loss=0.1,
        accuracy=0.95,
        num_examples=100,
        num_examples_weighted=100.0,
    )
    html_output = eval_obj.html()
    self.assertIn("<b>Evaluation</b>", html_output)
    self.assertIn("<li><b>Loss:</b> 0.1</li>", html_output)
    self.assertIn("<li><b>Accuracy:</b> 0.95</li>", html_output)
    self.assertIn("<li><b>Num Examples:</b> 100</li>", html_output)
    self.assertIn("<li><b>Num Examples Weighted:</b> 100.0</li>", html_output)
    self.assertNotIn("User Metrics", html_output)

    eval_obj_with_user = evaluation.Evaluation(
        loss=0.2,
        user_metrics={"precision": 0.8, "recall": 0.7},
    )
    html_output_user = eval_obj_with_user.html()
    self.assertIn("<li><b>Loss:</b> 0.2</li>", html_output_user)
    self.assertIn("<li><b>User Metrics:</b><ul>", html_output_user)
    self.assertIn("<li><em>precision</em>: 0.8</li>", html_output_user)
    self.assertIn("<li><em>recall</em>: 0.7</li>", html_output_user)

  def test_evaluation_json(self):
    eval_obj = evaluation.Evaluation(
        loss=0.1,
        accuracy=0.95,
        num_examples=100,
        user_metrics={"f1": 0.85},
    )
    json_str = eval_obj.to_json()
    self.assertIsNotNone(json_str)

    # Test from_json
    loaded_eval = evaluation.Evaluation.from_json(json_str)
    self.assertEqual(loaded_eval.loss, 0.1)
    self.assertEqual(loaded_eval.accuracy, 0.95)
    self.assertEqual(loaded_eval.num_examples, 100)
    self.assertIsNone(loaded_eval.num_examples_weighted)
    self.assertEqual(loaded_eval.user_metrics, {"f1": 0.85})

  def test_compute_ranking_metrics_jax(self):
    pos_scores = jnp.array([0.5, 0.8])
    neg_scores = jnp.array([0.2, 0.6, 0.7, 0.9])
    metrics = evaluation.compute_ranking_metrics(pos_scores, neg_scores)

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
    metrics = evaluation.compute_ranking_metrics(pos_scores, neg_scores)

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
    metrics = evaluation.compute_ranking_metrics(pos_scores, neg_scores)

    self.assertAlmostEqual(float(metrics["mrr"]), 0.2)
    self.assertAlmostEqual(float(metrics["hit_at_1"]), 1.0 / 9.0)
    self.assertAlmostEqual(float(metrics["hit_at_5"]), 5.0 / 9.0)
    self.assertAlmostEqual(float(metrics["auc"]), 0.5)

  def test_compute_ranking_metrics_nans(self):
    pos_scores = np.array([np.nan, 0.5])
    neg_scores = np.array([0.2] * 16)
    metrics = evaluation.compute_ranking_metrics(pos_scores, neg_scores)

    self.assertTrue(np.isnan(float(metrics["mrr"])))
    self.assertTrue(np.isnan(float(metrics["hit_at_1"])))
    self.assertTrue(np.isnan(float(metrics["hit_at_5"])))
    self.assertTrue(np.isnan(float(metrics["auc"])))


if __name__ == "__main__":
  absltest.main()
