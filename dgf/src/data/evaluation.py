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
from typing import Dict, List, Optional
import altair as alt
import dataclasses_json
import numpy as np
import pandas as pd


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

  loss: Optional[float] = None
  accuracy: Optional[float] = None
  rmse: Optional[float] = None
  r2: Optional[float] = None
  num_examples: Optional[int] = None
  num_examples_weighted: Optional[float] = None
  mrr: Optional[float] = None
  auc: Optional[float] = None
  hit_at: Dict[int, float] = dataclasses.field(default_factory=dict)
  user_metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
  per_classes: List[PerClass] = dataclasses.field(default_factory=list)

  def _repr_html_(
      self, max_plot_points: int = 1000, max_plot_classes: int = 20
  ) -> str:
    """Html representation of the metrics."""

    html_parts = ["<b>Evaluation</b>", "<ul>"]

    def _add_metric(name, value):
      if value is not None:
        html_parts.append(f"<li><b>{name}:</b> {value}</li>")

    _add_metric("Loss", self.loss)
    _add_metric("Accuracy", self.accuracy)
    _add_metric("RMSE", self.rmse)
    _add_metric("R2", self.r2)
    _add_metric("Num Examples", self.num_examples)
    _add_metric("Num Examples Weighted", self.num_examples_weighted)
    _add_metric("MRR", self.mrr)
    _add_metric("AUC", self.auc)

    if self.per_classes:
      html_parts.append(
          f"<li><b>Per Class Metrics [{len(self.per_classes)}]:</b><ul>"
      )
      for c, pc in enumerate(self.per_classes):
        html_parts.append(
            f"<li><em>Class {c}</em>: AUC={pc.auc():.4f},"
            f" PR-AUC={pc.pr_auc():.4f}</li>"
        )
      html_parts.append("</ul></li>")

    if self.hit_at:
      html_parts.append("<li><b>Hit@N:</b><ul>")
      for key, value in self.hit_at.items():
        html_parts.append(f"<li><em>@{key}</em>: {value}</li>")
      html_parts.append("</ul></li>")

    if self.user_metrics:
      html_parts.append("<li><b>User Metrics:</b><ul>")
      for key, value in self.user_metrics.items():
        html_parts.append(f"<li><em>{key}</em>: {value}</li>")
      html_parts.append("</ul></li>")

    html_parts.append("</ul>")

    # Add plots if we have curves
    if self.per_classes and len(self.per_classes[0].tp) > 0:
      if len(self.per_classes) > max_plot_classes:
        html_parts.append(
            f"<p>*Showing plots for the first {max_plot_classes} classes out of"
            f" {len(self.per_classes)} total classes.</p>"
        )
      num_classes_to_plot = min(len(self.per_classes), max_plot_classes)
      points_per_class = max(10, max_plot_points // num_classes_to_plot)
      roc_data = []
      for c, pc in enumerate(self.per_classes[:max_plot_classes]):
        fpr = pc.fpr
        tpr = pc.tpr
        if len(fpr) > points_per_class:
          indices = np.linspace(0, len(fpr) - 1, points_per_class, dtype=int)
          fpr = fpr[indices]
          tpr = tpr[indices]
        for f, t in zip(fpr, tpr):
          roc_data.append({"Class": str(c), "FPR": f, "TPR": t})
      roc_df = pd.DataFrame(roc_data)

      roc_chart = (
          alt.Chart(roc_df)
          .mark_line()
          .encode(
              x=alt.X("FPR", title="False Positive Rate"),
              y=alt.Y("TPR", title="True Positive Rate"),
              color="Class",
          )
          .properties(title="ROC Curve", width=300, height=300)
      )

      pr_data = []
      for c, pc in enumerate(self.per_classes[:max_plot_classes]):
        precision = pc.precision
        recall = pc.recall
        if len(precision) > points_per_class:
          indices = np.linspace(
              0, len(precision) - 1, points_per_class, dtype=int
          )
          precision = precision[indices]
          recall = recall[indices]
        for r, p in zip(recall, precision):
          pr_data.append({"Class": str(c), "Recall": r, "Precision": p})
      pr_df = pd.DataFrame(pr_data)

      pr_chart = (
          alt.Chart(pr_df)
          .mark_line()
          .encode(
              x=alt.X("Recall", title="Recall"),
              y=alt.Y("Precision", title="Precision"),
              color="Class",
          )
          .properties(title="Precision-Recall Curve", width=300, height=300)
      )

      combined_chart = alt.hconcat(roc_chart, pr_chart)
      html_parts.append(combined_chart.to_html())

    return "\n".join(html_parts)

  def html(
      self, max_plot_points: int = 1000, max_plot_classes: int = 20
  ) -> str:
    """Html representation of the metrics."""
    return self._repr_html_(max_plot_points, max_plot_classes)
