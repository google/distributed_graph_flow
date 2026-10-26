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

"""Evaluation utilities for ten_lines model."""

from typing import Any, Literal
from dgf.src.data import evaluation as evaluation_data_lib
from dgf.src.learning.ten_lines import evaluation_ext
import jax
import jax.numpy as jnp
import numpy as np

Evaluation = evaluation_data_lib.Evaluation
PerClass = evaluation_data_lib.PerClass


class ClassificationEvaluationAccumulator:
  """Accumulator for classification metrics (ROC, PR-ROC, AUC, PR-AUC).

  This class accumulates predictions and targets in a memory-efficient way
  using fixed-bin density histograms, allowing evaluation on large datasets
  where storing all predictions is not feasible.

  For each class (one-vs-rest), it maintains two histograms of size `num_bins`:
  - `pos_histograms`: counts of positive examples in each score bin.
  - `neg_histograms`: counts of negative examples in each score bin.

  Predictions (scores between 0.0 and 1.0) are mapped to bins uniformly
  distributed in the logit space in [-20, 20], which gives a high resolution
  for the small scores of minor classes.

  When `extract_metrics` is called, it computes:
  - Confusion matrix counts (TP, FP, TN, FN) for each threshold. The thresholds
    are the lower bin boundaries.
  - ROC AUC: Computed using the trapezoidal rule on the ROC curve (FPR vs TPR).
    Since TPR and FPR evolve linearly between bins, the trapezoidal rule gives
    an exact calculation of the area under the binned curve.
  - PR-AUC: Computed using the Davis-Goadrich integration method. Linear
    interpolation of Precision-Recall curves mathematically overestimates the
    area because Precision does not evolve linearly between bins (its
    denominator
    TP+FP changes non-linearly). Davis-Goadrich method interpolates linearly
    in the ROC space (TP vs FP) and integrates the corresponding Precision
    exactly over the interval, ensuring accurate PR-AUC calculation.

  The metrics are returned as a list of `PerClass` objects, one for each class.
  """

  def __init__(self, num_classes: int, num_bins: int = 10000):
    self._impl = evaluation_ext.ClassificationEvaluationAccumulator(
        num_classes, num_bins
    )

  def add_predictions(
      self, predictions: np.ndarray, targets: np.ndarray
  ) -> None:
    """Adds predictions to the accumulator."""
    self._impl.add_predictions(predictions, targets)

  def extract_metrics(self) -> list[PerClass]:
    """Extracts metrics from the accumulator."""
    raw_metrics = self._impl.extract_metrics()
    per_classes = []
    for m in raw_metrics:
      per_classes.append(
          PerClass(
              auc_value=m["auc"],
              pr_auc_value=m["pr_auc"],
              tp=m["tp"],
              fp=m["fp"],
              tn=m["tn"],
              fn=m["fn"],
              thresholds=m["thresholds"],
          )
      )
    return per_classes

  def populate_evaluation(self, evaluation: Evaluation) -> None:
    """Populates an Evaluation object with metrics from this accumulator."""
    if evaluation.per_classes:
      raise ValueError("per_classes is already populated.")

    evaluation.per_classes = self.extract_metrics()
    # Compute macro average AUC as default AUC
    if evaluation.auc is None and evaluation.per_classes:
      evaluation.auc = np.mean([pc.auc() for pc in evaluation.per_classes])


def compute_ranking_metrics(
    pos_scores: jax.Array | np.ndarray,
    neg_scores: jax.Array | np.ndarray,
    reduction: Literal["mean", "sum"] = "mean",
) -> dict[str, jax.Array | np.ndarray]:
  """Computes ranking metrics (MRR, Hit@1, Hit@5, AUC) from scores.

  Scores can be probabilities or logits.

  Args:
    pos_scores: Scores for positive edges, shape (B,).
    neg_scores: Scores for negative edges, shape (B * N,).
    reduction: How the per-positive-edge metrics are reduced over the B
      positive edges. "mean" returns the average metrics. "sum" returns the
      sum of the metrics, which is useful to accumulate metrics over batches
      of different sizes without the extra rounding of `mean * B`.

  Returns:
    A dictionary containing the computed metrics.
  """
  xp = np if isinstance(pos_scores, np.ndarray) else jnp
  if reduction == "mean":
    reduce_fn = xp.mean
  elif reduction == "sum":
    reduce_fn = xp.sum
  else:
    raise ValueError(f"Unknown reduction: {reduction!r}")

  neg_scores_reshaped = neg_scores.reshape(pos_scores.shape[0], -1)

  greater = xp.sum(neg_scores_reshaped > pos_scores[:, None], axis=-1)
  equal = xp.sum(neg_scores_reshaped == pos_scores[:, None], axis=-1)

  mid_ranks = greater + 1 + equal / 2.0
  mrr = reduce_fn(1.0 / mid_ranks)

  def hit_at_k(k):
    slots = k - greater
    total_tied = equal + 1
    prob = xp.clip(slots / total_tied, 0.0, 1.0)
    return reduce_fn(prob)

  hit_at_1 = hit_at_k(1)
  hit_at_5 = hit_at_k(5)

  auc_matrix = (pos_scores[:, None] > neg_scores_reshaped).astype(
      xp.float32
  ) + 0.5 * (pos_scores[:, None] == neg_scores_reshaped).astype(xp.float32)
  # Per-positive-edge AUC, reduced over the positive edges.
  auc = reduce_fn(xp.mean(auc_matrix, axis=-1))

  invalid = xp.any(xp.isnan(pos_scores)) | xp.any(xp.isnan(neg_scores))

  return {
      "mrr": xp.where(invalid, xp.nan, mrr),
      "hit_at_1": xp.where(invalid, xp.nan, hit_at_1),
      "hit_at_5": xp.where(invalid, xp.nan, hit_at_5),
      "auc": xp.where(invalid, xp.nan, auc),
  }
