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

"""Evaluation of a model."""

import dataclasses
from typing import Dict, Optional, Union
import dataclasses_json
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Evaluation:
  """A collection of metrics, plots and tables about the quality of a model.

  Usage example:

    ```python
    evaluation = model.evaluate(graph)
    print(evaluation)
    print(evaluation.accuracy)
    evaluation  # Html evaluation in notebook
    ```

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

  def _repr_html_(self) -> str:
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
    return "\n".join(html_parts)

  def html(self) -> str:
    """Html representation of the metrics."""

    return self._repr_html_()


def compute_ranking_metrics(
    pos_scores: Union[jax.Array, np.ndarray],
    neg_scores: Union[jax.Array, np.ndarray],
) -> Dict[str, Union[jax.Array, np.ndarray]]:
  """Computes ranking metrics (MRR, Hit@1, Hit@5, AUC) from scores.

  Scores can be probabilities or logits.

  Args:
    pos_scores: Scores for positive edges, shape (B,).
    neg_scores: Scores for negative edges, shape (B * N,).

  Returns:
    A dictionary containing the computed metrics.
  """
  xp = np if isinstance(pos_scores, np.ndarray) else jnp

  neg_scores_reshaped = neg_scores.reshape(pos_scores.shape[0], -1)

  greater = xp.sum(neg_scores_reshaped > pos_scores[:, None], axis=-1)
  equal = xp.sum(neg_scores_reshaped == pos_scores[:, None], axis=-1)

  mid_ranks = greater + 1 + equal / 2.0
  mrr = xp.mean(1.0 / mid_ranks)

  def hit_at_k(k):
    slots = k - greater
    total_tied = equal + 1
    prob = xp.clip(slots / total_tied, 0.0, 1.0)
    return xp.mean(prob)

  hit_at_1 = hit_at_k(1)
  hit_at_5 = hit_at_k(5)

  auc_matrix = (pos_scores[:, None] > neg_scores_reshaped).astype(
      xp.float32
  ) + 0.5 * (pos_scores[:, None] == neg_scores_reshaped).astype(xp.float32)
  auc = xp.mean(auc_matrix)

  invalid = xp.any(xp.isnan(pos_scores)) | xp.any(xp.isnan(neg_scores))

  return {
      "mrr": xp.where(invalid, xp.nan, mrr),
      "hit_at_1": xp.where(invalid, xp.nan, hit_at_1),
      "hit_at_5": xp.where(invalid, xp.nan, hit_at_5),
      "auc": xp.where(invalid, xp.nan, auc),
  }
