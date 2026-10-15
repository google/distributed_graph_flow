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

"""Evaluation classes for models."""

import dataclasses
import uuid
import altair as alt
import dataclasses_json
from dgf.src.util import html_util
import numpy as np
import pandas as pd

# Maximum number of points of each plotted class curve.
_DEFAULT_MAX_PLOT_POINTS = 1000


def _downsample_curve(
    x: np.ndarray, y: np.ndarray, num_points: int
) -> tuple[np.ndarray, np.ndarray]:
  """Selects points uniformly along the length of a 2D curve (not its indices).

  Args:
    x: X coordinates of the curve.
    y: Y coordinates of the curve.
    num_points: Target number of points. Up to `2 * num_points + 1` points are
      returned.

  Returns:
    The downsampled x and y coordinates, in the original order. The first and
    last points are always included.
  """
  if len(x) <= num_points:
    return x, y
  segment_lengths = np.abs(np.diff(x)) + np.abs(np.diff(y))
  cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
  total_length = cumulative_length[-1]
  if total_length == 0:
    indices = np.array([0, len(x) - 1])
  else:
    targets = np.linspace(0.0, total_length, num_points)
    indices = np.searchsorted(cumulative_length, targets, side="left")
    indices = np.clip(indices, 0, len(x) - 1)
    # Also keep the previous points to preserve sharp corners.
    indices = np.concatenate([[0], indices, np.maximum(indices - 1, 0)])
    indices = np.unique(indices)
  return x[indices], y[indices]


def _roc_convex_hull(
    fpr: np.ndarray, tpr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Computes the upper convex hull of a ROC curve (ROCCH).

  Any point on the hull is achievable by randomly interpolating between two
  thresholds (Provost & Fawcett, 2001).

  Args:
    fpr: False positive rates.
    tpr: True positive rates.

  Returns:
    The fpr and tpr of the hull vertices, sorted by increasing fpr.
  """
  points = np.unique(np.stack([fpr, tpr], axis=1), axis=0)
  hull = []
  for p in points:
    while len(hull) >= 2:
      o, a = hull[-2], hull[-1]
      cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
      if cross < 0:
        break
      hull.pop()
    hull.append(p)
  hull = np.array(hull)
  return hull[:, 0], hull[:, 1]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class PerClass:
  """Evaluation metrics for a single class (one-vs-rest)."""

  auc_value: float
  pr_auc_value: float
  tp: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: x.tolist(), decoder=np.array
      )
  )
  fp: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: x.tolist(), decoder=np.array
      )
  )
  tn: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: x.tolist(), decoder=np.array
      )
  )
  fn: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: x.tolist(), decoder=np.array
      )
  )
  thresholds: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: x.tolist(), decoder=np.array
      )
  )

  def auc(self) -> float:
    return self.auc_value

  def pr_auc(self) -> float:
    return self.pr_auc_value

  @property
  def fpr(self) -> np.ndarray:
    return self.fp / (self.fp + self.tn + 1e-9)

  @property
  def tpr(self) -> np.ndarray:
    return self.tp / (self.tp + self.fn + 1e-9)

  @property
  def recall(self) -> np.ndarray:
    return self.tpr

  @property
  def precision(self) -> np.ndarray:
    return self.tp / (self.tp + self.fp + 1e-9)


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Evaluation:
  """A collection of metrics, plots and tables about the quality of a model.

  Attributes:
    loss: Model loss. The loss of the model.
    num_examples: Number of examples (non weighted).
    num_examples_weighted: Number of examples (with weight).
    accuracy: Accuracy of the model.
    rmse: Root mean squared error of the model.
    r2: R-squared of the model.
    mrr: Mean Reciprocal Rank (MRR) of the model.
    auc: Area Under the Curve (AUC) of the model.
    hit_at: Dictionary of Hit@N metrics.
    user_metrics: Additional user-defined metrics.
    per_classes: List of PerClass metrics for multi-class classification.
  """

  loss: float | None = None
  accuracy: float | None = None
  rmse: float | None = None
  r2: float | None = None
  num_examples: int | None = None
  num_examples_weighted: float | None = None
  mrr: float | None = None
  auc: float | None = None
  hit_at: dict[int, float] = dataclasses.field(default_factory=dict)
  user_metrics: dict[str, float] = dataclasses.field(default_factory=dict)
  per_classes: list[PerClass] = dataclasses.field(default_factory=list)

  def _repr_html_(
      self, max_plot_points: int = 1000, max_plot_classes: int = 20
  ) -> str:
    """Html representation of the metrics."""

    html_parts = [
        "<style>",
        html_util.get_table_style(),
        "</style>",
        '<table class="dgf-table">',
        "<tbody>",
    ]

    def _add_metric(name, value):
      if value is not None:
        html_parts.append(f"<tr><td><b>{name}</b></td><td>{value}</td></tr>")

    _add_metric("Loss", self.loss)
    _add_metric("Accuracy", self.accuracy)
    _add_metric("RMSE", self.rmse)
    _add_metric("R2", self.r2)
    _add_metric("Num Examples", self.num_examples)
    _add_metric("Num Examples Weighted", self.num_examples_weighted)
    _add_metric("MRR", self.mrr)
    _add_metric("AUC", self.auc)

    if self.per_classes:
      num_classes = len(self.per_classes)
      html_parts.append(
          '<tr><td style="vertical-align: top;"><b>Per Class Metrics'
          f" [{num_classes}]:</b></td><td><ul>"
      )

      def _add_class_metric(c, pc):
        html_parts.append(
            f"<li><em>Class {c}</em>: AUC={pc.auc():.4f},"
            f" PR-AUC={pc.pr_auc():.4f}</li>"
        )

      max_num_classes_to_display = 5
      if num_classes > max_num_classes_to_display:
        for i in range(3):
          _add_class_metric(i, self.per_classes[i])
        html_parts.append(
            f"<li>... ({num_classes - max_num_classes_to_display} omitted)"
            " ...</li>"
        )
        for i in range(num_classes - 2, num_classes):
          _add_class_metric(i, self.per_classes[i])
      else:
        for c, pc in enumerate(self.per_classes):
          _add_class_metric(c, pc)

      html_parts.append("</ul></td></tr>")

    if self.hit_at:
      html_parts.append(
          '<tr><td style="vertical-align: top;"><b>Hit@N:</b></td><td><ul>'
      )
      for key, value in self.hit_at.items():
        html_parts.append(f"<li><em>@{key}</em>: {value}</li>")
      html_parts.append("</ul></td></tr>")

    if self.user_metrics:
      html_parts.append(
          '<tr><td style="vertical-align: top;"><b>User'
          " Metrics:</b></td><td><ul>"
      )
      for key, value in self.user_metrics.items():
        html_parts.append(f"<li><em>{key}</em>: {value}</li>")
      html_parts.append("</ul></td></tr>")

    html_parts.append("</tbody></table>")

    # Add plots if we have curves
    if self.per_classes and len(self.per_classes[0].tp) > 0:
      if len(self.per_classes) > max_plot_classes:
        html_parts.append(
            f"<p>*Showing plots for the first {max_plot_classes} classes out of"
            f" {len(self.per_classes)} total classes.</p>"
        )
      roc_data = []
      for c, pc in enumerate(self.per_classes[:max_plot_classes]):
        fpr, tpr = _roc_convex_hull(pc.fpr, pc.tpr)
        fpr, tpr = _downsample_curve(fpr, tpr, max_plot_points)
        for i, (f, t) in enumerate(zip(fpr, tpr)):
          roc_data.append({"Class": str(c), "FPR": f, "TPR": t, "Index": i})
      roc_df = pd.DataFrame(roc_data)

      roc_chart = (
          alt.Chart(roc_df)
          .mark_line()
          .encode(
              x=alt.X("FPR", title="False Positive Rate"),
              y=alt.Y("TPR", title="True Positive Rate"),
              color="Class",
              # The default x order is ambiguous for vertical segments.
              order="Index",
          )
          .properties(title="ROC Curve (convex hull)", width=300, height=300)
      )

      pr_data = []
      for c, pc in enumerate(self.per_classes[:max_plot_classes]):
        recall, precision = _downsample_curve(
            pc.recall, pc.precision, max_plot_points
        )
        for i, (r, p) in enumerate(zip(recall, precision)):
          pr_data.append(
              {"Class": str(c), "Recall": r, "Precision": p, "Index": i}
          )
      pr_df = pd.DataFrame(pr_data)

      pr_chart = (
          alt.Chart(pr_df)
          .mark_line()
          .encode(
              x=alt.X("Recall", title="Recall"),
              y=alt.Y("Precision", title="Precision"),
              color="Class",
              order="Index",
          )
          .properties(title="Precision-Recall Curve", width=300, height=300)
      )

      combined_chart = alt.hconcat(roc_chart, pr_chart)
      # The number of points is already bounded by max_plot_points.
      with alt.data_transformers.disable_max_rows():  # pyrefly: ignore[bad-context-manager]
        html_parts.append(
            combined_chart.to_html(
                fullhtml=False, output_div=f"vis-{uuid.uuid4().hex}"
            )
        )

    return "\n".join(html_parts)

  def html(
      self, max_plot_points: int = 1000, max_plot_classes: int = 20
  ) -> str:
    """Html representation of the metrics."""
    return self._repr_html_(max_plot_points, max_plot_classes)
