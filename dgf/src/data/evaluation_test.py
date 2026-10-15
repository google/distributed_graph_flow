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
from dgf.src.data import evaluation
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
    self.assertIn("<tr><td><b>Loss</b></td><td>0.1</td></tr>", html_output)
    self.assertIn("<tr><td><b>Accuracy</b></td><td>0.95</td></tr>", html_output)
    self.assertIn(
        "<tr><td><b>Num Examples</b></td><td>100</td></tr>", html_output
    )
    self.assertIn(
        "<tr><td><b>Num Examples Weighted</b></td><td>100.0</td></tr>",
        html_output,
    )
    self.assertNotIn("User Metrics", html_output)

    eval_obj_with_user = evaluation.Evaluation(
        loss=0.2,
        user_metrics={"precision": 0.8, "recall": 0.7},
    )
    html_output_user = eval_obj_with_user.html()
    self.assertIn("<tr><td><b>Loss</b></td><td>0.2</td></tr>", html_output_user)
    self.assertIn("<b>User Metrics:</b>", html_output_user)
    self.assertIn("<li><em>precision</em>: 0.8</li>", html_output_user)
    self.assertIn("<li><em>recall</em>: 0.7</li>", html_output_user)

  def test_evaluation_json(self):
    eval_obj = evaluation.Evaluation(
        loss=0.1,
        accuracy=0.95,
        num_examples=100,
        user_metrics={"f1": 0.85},
    )
    json_str = eval_obj.to_json()  # pyrefly: ignore[missing-attribute]
    self.assertIsNotNone(json_str)

    # Test from_json
    loaded_eval = evaluation.Evaluation.from_json(json_str)  # pyrefly: ignore[missing-attribute]
    self.assertEqual(loaded_eval.loss, 0.1)
    self.assertEqual(loaded_eval.accuracy, 0.95)
    self.assertEqual(loaded_eval.num_examples, 100)
    self.assertIsNone(loaded_eval.num_examples_weighted)
    self.assertEqual(loaded_eval.user_metrics, {"f1": 0.85})

  def test_per_class_properties(self):
    pc = evaluation.PerClass(
        auc_value=0.9,
        pr_auc_value=0.8,
        tp=np.array([1, 2]),
        fp=np.array([3, 4]),
        tn=np.array([5, 6]),
        fn=np.array([7, 8]),
        thresholds=np.array([0.5, 0.1]),
    )
    np.testing.assert_array_almost_equal(
        pc.fpr, np.array([3 / (3 + 5 + 1e-9), 4 / (4 + 6 + 1e-9)])
    )
    np.testing.assert_array_almost_equal(
        pc.tpr, np.array([1 / (1 + 7 + 1e-9), 2 / (2 + 8 + 1e-9)])
    )
    np.testing.assert_array_almost_equal(pc.recall, pc.tpr)
    np.testing.assert_array_almost_equal(
        pc.precision, np.array([1 / (1 + 3 + 1e-9), 2 / (2 + 4 + 1e-9)])
    )

  def test_per_class_serialization(self):
    pc = evaluation.PerClass(
        auc_value=0.9,
        pr_auc_value=0.8,
        tp=np.array([1, 2]),
        fp=np.array([3, 4]),
        tn=np.array([5, 6]),
        fn=np.array([7, 8]),
        thresholds=np.array([0.5, 0.1]),
    )
    json_str = pc.to_json()  # pyrefly: ignore[missing-attribute]
    self.assertIsNotNone(json_str)
    self.assertIn('"tp": [1, 2]', json_str)

    loaded = evaluation.PerClass.from_json(json_str)  # pyrefly: ignore[missing-attribute]
    self.assertEqual(loaded.auc_value, 0.9)
    self.assertEqual(loaded.pr_auc_value, 0.8)
    np.testing.assert_array_equal(loaded.tp, np.array([1, 2]))
    np.testing.assert_array_equal(loaded.fp, np.array([3, 4]))
    np.testing.assert_array_equal(loaded.tn, np.array([5, 6]))
    np.testing.assert_array_equal(loaded.fn, np.array([7, 8]))
    np.testing.assert_array_almost_equal(
        loaded.thresholds, np.array([0.5, 0.1])
    )

  def test_evaluation_html_max_plot_points(self):
    pc = evaluation.PerClass(
        auc_value=0.9,
        pr_auc_value=0.8,
        tp=np.arange(100),
        fp=np.arange(100),
        tn=np.arange(100),
        fn=np.arange(100),
        thresholds=np.linspace(0, 0.99, 100),
    )
    eval_obj = evaluation.Evaluation(per_classes=[pc])

    # Downsample to 5. Should not crash and should produce HTML.
    html_5 = eval_obj.html(max_plot_points=5)
    self.assertIsNotNone(html_5)
    self.assertIn("vegaEmbed", html_5)

  def test_downsample_curve_preserves_shape(self):
    # (0, 0) -> (0, 0.9) -> (1, 1) with all but one point on the first segment.
    x = np.concatenate([np.zeros(10000), [1.0]])
    y = np.concatenate([np.linspace(0.0, 0.9, 10000), [1.0]])
    ds_x, ds_y = evaluation._downsample_curve(x, y, 20)
    self.assertLessEqual(len(ds_x), 41)
    self.assertEqual((ds_x[0], ds_y[0]), (0.0, 0.0))
    self.assertEqual((ds_x[-1], ds_y[-1]), (1.0, 1.0))
    # The corner at (0, 0.9) is preserved.
    self.assertEqual((ds_x[-2], ds_y[-2]), (0.0, 0.9))
    # The points are in the original order.
    self.assertTrue(np.all(np.diff(ds_y) > 0))

  def test_evaluation_html_many_classes_and_points(self):
    rng = np.random.default_rng(0)
    per_classes = []
    for _ in range(20):
      tp = np.cumsum(rng.integers(0, 2, 10001))
      fp = np.cumsum(rng.integers(0, 2, 10001))
      per_classes.append(
          evaluation.PerClass(
              auc_value=0.5,
              pr_auc_value=0.5,
              tp=tp,
              fp=fp,
              tn=fp[-1] - fp,
              fn=tp[-1] - tp,
              thresholds=np.linspace(1, 0, 10001),
          )
      )
    html = evaluation.Evaluation(per_classes=per_classes).html()
    self.assertIn("vegaEmbed", html)

  def test_roc_convex_hull(self):
    # A staircase ROC curve.
    fpr = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    tpr = np.array([0.0, 0.5, 0.5, 0.8, 0.8, 1.0])
    hull_fpr, hull_tpr = evaluation._roc_convex_hull(fpr, tpr)
    np.testing.assert_array_equal(hull_fpr, [0.0, 0.0, 0.5, 1.0])
    np.testing.assert_array_equal(hull_tpr, [0.0, 0.5, 0.8, 1.0])

  def test_roc_convex_hull_removes_collinear_and_concave(self):
    fpr = np.array([0.0, 0.25, 0.5, 0.6, 1.0])
    tpr = np.array([0.0, 0.25, 0.5, 0.4, 1.0])
    hull_fpr, hull_tpr = evaluation._roc_convex_hull(fpr, tpr)
    np.testing.assert_array_equal(hull_fpr, [0.0, 1.0])
    np.testing.assert_array_equal(hull_tpr, [0.0, 1.0])

  def test_downsample_curve_small(self):
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.7, 1.0])
    ds_x, ds_y = evaluation._downsample_curve(x, y, 20)
    np.testing.assert_array_equal(ds_x, x)
    np.testing.assert_array_equal(ds_y, y)

  def test_downsample_curve_constant(self):
    x = np.zeros(100)
    y = np.zeros(100)
    ds_x, _ = evaluation._downsample_curve(x, y, 20)
    self.assertLen(ds_x, 2)

  def test_evaluation_html_max_plot_classes(self):
    pc1 = evaluation.PerClass(
        auc_value=0.9,
        pr_auc_value=0.8,
        tp=np.array([1, 2]),
        fp=np.array([3, 4]),
        tn=np.array([5, 6]),
        fn=np.array([7, 8]),
        thresholds=np.array([0.5, 0.1]),
    )
    pc2 = evaluation.PerClass(
        auc_value=0.7,
        pr_auc_value=0.6,
        tp=np.array([1, 2]),
        fp=np.array([3, 4]),
        tn=np.array([5, 6]),
        fn=np.array([7, 8]),
        thresholds=np.array([0.5, 0.1]),
    )
    eval_obj = evaluation.Evaluation(per_classes=[pc1, pc2])

    # Limit to 1 class. Should show warning and only plot 1 class.
    html_1 = eval_obj.html(max_plot_classes=1)
    self.assertIn(
        "Showing plots for the first 1 classes out of 2 total classes.", html_1
    )

    # Limit to 2 classes. Should not show warning.
    html_2 = eval_obj.html(max_plot_classes=2)
    self.assertNotIn("Showing plots for the first", html_2)


if __name__ == "__main__":
  absltest.main()
