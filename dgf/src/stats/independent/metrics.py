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

"""Metric definitions, summand mapping, and default M declarations."""

from collections.abc import Sequence
from typing import Any
import numpy as np

# Default declared M per metric (see README.md, "Choosing M").
DEFAULT_M: dict[str, float] = {
    "mae": 4.0,
    "mse": 16.0,
    "rmse": 16.0,
    "accuracy": 4.0,
    "r2": 16.0,
}

_AP_ERROR_MSG: str = (
    "average_precision is not supported: no non-vacuous finite-sample"
    " bound exists below n ≈ 1e5 (McDiarmid half-width 0.59 at n=1000, on a"
    " [0,1] metric)."
)


def normalize_metric(metric: str) -> str:
  """Normalizes the metric name and checks that it is supported."""
  m = metric.strip().lower()
  if m in ("average_precision", "averageprecision", "ap"):
    raise ValueError(_AP_ERROR_MSG)
  if m not in DEFAULT_M:
    raise ValueError(
        f"Unknown metric {metric!r}. Supported metrics:"
        f" {sorted(DEFAULT_M.keys())}."
    )
  return m


def get_default_m(metric: str) -> float:
  """Returns the default M parameter for the given metric."""
  m = normalize_metric(metric)
  return DEFAULT_M[m]


# Upper bound on theta implied by the metric's definition, or None if the
# metric is unbounded above. accuracy is a mean of indicators, so theta <= 1.
_METRIC_THETA_MAX: dict[str, float] = {
    "accuracy": 1.0,
}


def get_theta_max(metric: str) -> float | None:
  """Returns the upper bound on theta implied by the metric's definition, or None."""
  m = metric.strip().lower()
  return _METRIC_THETA_MAX.get(m, None)


def extract_r2_summands(data: Any) -> tuple[np.ndarray, np.ndarray]:
  """Extracts per-item summands for R^2: s_a = e^2 and s_b = b_i.

  Non-finite values poison rather than being dropped: any non-finite label or
  prediction becomes a NaN summand in s_a, and poisons any pair in s_b where
  it enters. Malformed input (e.g. data not a 2-tuple) raises ValueError.
  """
  if isinstance(data, tuple) and len(data) == 2:
    y = np.asarray(data[0], dtype=np.float64)
    y_pred = np.asarray(data[1], dtype=np.float64)
    e = y_pred - y
  else:
    raise ValueError(
        "For metric 'r2', raw data must be a tuple of (labels, predictions)."
    )

  invalid_y = ~np.isfinite(y)
  invalid_pred = ~np.isfinite(y_pred)
  invalid_e = invalid_y | invalid_pred
  e_valid = np.where(invalid_e, np.nan, e)
  s_a = e_valid**2

  n = len(y)
  n_b = n // 2
  if n_b > 0:
    y_pairs = y[: 2 * n_b].reshape(n_b, 2)
    pair_invalid = ~np.isfinite(y_pairs[:, 0]) | ~np.isfinite(y_pairs[:, 1])
    diff = y_pairs[:, 1] - y_pairs[:, 0]
    s_b = np.where(pair_invalid, np.nan, 0.5 * (diff**2))
  else:
    s_b = np.empty(0, dtype=np.float64)
  return s_a, s_b


def extract_summands(data: Any, metric: str) -> np.ndarray:
  """Extracts per-item non-negative summands s_i from raw evaluation data.

  Non-finite items (NaN, inf) or non-finite float labels/predictions become
  NaN summands (poisoning the sketch). Finite accuracy values not in {0, 1}
  and malformed input forms raise ValueError.

  Args:
    data: Either a 1D sequence of errors/residuals (or boolean/indicators for
      accuracy), or a two-tuple of (labels, predictions) or (y_true, y_pred).
    metric: The metric name ('mae', 'mse', 'rmse', 'accuracy').

  Returns:
    A 1D float64 numpy array of summands s_i.
  """
  m = normalize_metric(metric)
  if m == "r2":
    raise ValueError("For metric 'r2', call extract_r2_summands instead.")

  # Single-statistic metrics
  if (
      isinstance(data, tuple)
      and len(data) == 2
      and not isinstance(data[0], (int, float, np.number))
  ):
    labels, predictions = np.asarray(data[0]), np.asarray(data[1])
    if m == "accuracy":
      res = (labels == predictions).astype(np.float64)
      if np.issubdtype(labels.dtype, np.floating):
        res[~np.isfinite(labels)] = np.nan
      if np.issubdtype(predictions.dtype, np.floating):
        res[~np.isfinite(predictions)] = np.nan
      return res
    e = (predictions - labels).astype(np.float64)
  else:
    arr = np.asarray(data)
    if m == "accuracy":
      res = arr.astype(np.float64)
      finite_mask = np.isfinite(res)
      if np.any(finite_mask):
        finite_vals = res[finite_mask]
        bad_mask = (finite_vals != 0.0) & (finite_vals != 1.0)
        if np.any(bad_mask):
          bad_val = float(finite_vals[bad_mask][0])
          raise ValueError(
              f"Accuracy value {bad_val} is invalid; accuracy expects 0/1"
              " correctness values; round them (e.g. np.round) or pass"
              " argmax(probabilities) == labels."
          )
      return res
    e = arr.astype(np.float64)

  # mae, mse, rmse: non-finite residuals become NaN summands
  e_valid = np.where(~np.isfinite(e), np.nan, e)
  if m == "mae":
    return np.abs(e_valid)
  elif m in ("mse", "rmse"):
    return e_valid**2
  else:
    raise ValueError(f"Unhandled metric {m}")
