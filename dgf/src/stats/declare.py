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

"""Estimation and declaration of relative variance bound M from archive residuals.

This module provides tools to estimate the relative variance bound M from
historical or archive model residuals, and serialize/deserialize reusable
Declaration records.

An estimate is only as good as the archive's resemblance to the evaluated
data. The mathematical coverage bound holds for the declared M, not for
the estimate itself.

The archive data used to estimate M must never overlap the evaluated sample.
Accidental data leakage invalidates coverage certificates. The
archive_fingerprint identifies the exact archive array; it cannot detect a
partial overlap with the evaluated sample.

Bootstrap upper quantiles tend to understate M under heavy tails. The
diagnostics top1pct_share and m_without_top1pct show how much the tail
drives the estimate. If a few units dominate, declare more than m_upper.
On real residuals, m_upper from one random half covered the other half's M
in 70–95% of splits, below the nominal level; add a margin when
top1pct_share is large.

For MAE, use absolute residuals s = |e|. For MSE and RMSE, use squared
residuals s = e^2. For R^2, use estimate_m_r2 to estimate error and label
bounds jointly. Accuracy on the independent path needs no M bound.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import hashlib
import json
import math
from typing import Any

import numpy as np

__all__ = [
    "Declaration",
    "MEstimate",
    "R2MEstimate",
    "estimate_m",
    "estimate_m_r2",
    "fingerprint",
    "load_declarations",
    "save_declarations",
]


@dataclasses.dataclass(frozen=True)
class MEstimate:
  """Results and diagnostics from estimating M via bootstrap.

  Attributes:
    m_point: Empirical point estimate of M = n * sum(s^2) / (sum(s)^2).
    m_upper: Bootstrap upper quantile at the specified confidence level.
    num_units: Number of resampling units (clusters if clustered, else items).
    num_items: Total number of items n.
    max_unit_share: Share of sum(s^2) contributed by the largest unit.
    top1pct_share: Share of sum(s^2) contributed by the top 1% of units.
    m_without_top1pct: Point estimate of M recomputed without the top 1% of
      units.
    num_skipped_resamples: Number of bootstrap resamples skipped due to sum(s*)
      = 0.
  """

  m_point: float
  m_upper: float
  num_units: int
  num_items: int
  max_unit_share: float
  top1pct_share: float
  m_without_top1pct: float
  num_skipped_resamples: int


@dataclasses.dataclass(frozen=True)
class R2MEstimate:
  """Pair of M estimates for R^2 evaluation (errors and labels).

  Attributes:
    m_errors: M estimate for error summands s = e^2.
    m_labels: M estimate for label variance summands q = (y - y_bar)^2.
  """

  m_errors: MEstimate
  m_labels: MEstimate


@dataclasses.dataclass(frozen=True)
class Declaration:
  """Reusable certified declaration record for evaluation parameters.

  A declaration is a formal claim about evaluated data that refutation checks
  can refute but cannot verify. Reusing a declaration on a new model or new
  period is an engineering judgment that the user makes and should document
  in `source`. The archive data must not overlap the evaluated sample.

  Attributes:
    metric: Metric name ("mae", "mse", "rmse", "accuracy", "r2").
    m: Relative variance bound M on error summands (>= 1.0).
    m_labels: Relative variance bound on label summands (R^2 only).
    kappa: Declared variance inflation factor kappa (>= 1.0).
    kappa_labels: Declared variance inflation factor on labels (R^2 only).
    source: Description of archive source, model version, and rationale.
    archive_fingerprint: SHA-256 fingerprint of the archive summands.
    created: ISO date string when the declaration was created.
    method: Description of estimation method and configuration.
    diagnostics: Numeric diagnostic metrics from estimation.
    format_version: Schema version of the declaration record.
  """

  metric: str
  m: float | None = None
  m_labels: float | None = None
  kappa: float | None = None
  kappa_labels: float | None = None
  source: str = ""
  archive_fingerprint: str = ""
  created: str = ""
  method: str = ""
  diagnostics: Mapping[str, float] = dataclasses.field(default_factory=dict)
  format_version: int = 1

  def kwargs(self) -> dict[str, Any]:
    """Returns dictionary of non-None interval arguments for interval entry points.

    Emits 'm_item' (item-level M) for all metrics (including r2).
    """
    res: dict[str, Any] = {}
    is_r2 = self.metric.strip().lower() == "r2"
    if self.m is not None:
      res["m_item"] = self.m
    if self.kappa is not None:
      res["kappa"] = self.kappa
    if is_r2:
      if self.m_labels is not None:
        res["m_labels"] = self.m_labels
      if self.kappa_labels is not None:
        res["kappa_labels"] = self.kappa_labels
    return res

  def to_json(self) -> str:
    """Serializes the declaration to stable, formatted JSON."""
    data = {
        "archive_fingerprint": self.archive_fingerprint,
        "created": self.created,
        "diagnostics": dict(self.diagnostics),
        "format_version": self.format_version,
        "kappa": self.kappa,
        "kappa_labels": self.kappa_labels,
        "m": self.m,
        "m_labels": self.m_labels,
        "method": self.method,
        "metric": self.metric,
        "source": self.source,
    }
    return json.dumps(data, sort_keys=True, indent=2)

  @classmethod
  def from_json(cls, text: str) -> "Declaration":
    """Deserializes and validates a declaration from JSON."""
    data = json.loads(text)
    format_version = data.get("format_version", 1)
    if format_version != 1:
      raise ValueError(f"Unknown format_version: {format_version}")

    metric = str(data.get("metric", "")).strip().lower()
    if metric not in ("mae", "mse", "rmse", "accuracy", "r2"):
      raise ValueError(f"Unknown metric: {metric}")

    m = data.get("m")
    if m is not None:
      m_val = float(m)
      if not math.isfinite(m_val) or m_val < 1.0:
        raise ValueError(f"m must be finite and >= 1.0, got {m}")
      m = m_val

    raw_kappa = data.get("kappa", 1.0)
    if raw_kappa is None:
      kappa = 1.0
    else:
      kappa_val = float(raw_kappa)
      if not math.isfinite(kappa_val) or kappa_val < 1.0:
        raise ValueError(f"kappa must be finite and >= 1.0, got {raw_kappa}")
      kappa = kappa_val

    m_labels = data.get("m_labels")
    kappa_labels = data.get("kappa_labels")

    if metric != "r2":
      if m_labels is not None:
        raise ValueError(
            f"m_labels is only valid for metric 'r2', got {metric}"
        )
      if kappa_labels is not None:
        raise ValueError(
            f"kappa_labels is only valid for metric 'r2', got {metric}"
        )
    else:
      if m_labels is not None:
        ml_val = float(m_labels)
        if not math.isfinite(ml_val) or ml_val < 1.0:
          raise ValueError(
              f"m_labels must be finite and >= 1.0, got {m_labels}"
          )
        m_labels = ml_val
      if kappa_labels is not None:
        kl_val = float(kappa_labels)
        if not math.isfinite(kl_val) or kl_val < 1.0:
          raise ValueError(
              f"kappa_labels must be finite and >= 1.0, got {kappa_labels}"
          )
        kappa_labels = kl_val

    diag_raw = data.get("diagnostics", {})
    diagnostics = {str(k): float(v) for k, v in diag_raw.items()}

    return cls(
        metric=metric,
        m=m,
        m_labels=m_labels,
        kappa=kappa,
        kappa_labels=kappa_labels,
        source=str(data.get("source", "")),
        archive_fingerprint=str(data.get("archive_fingerprint", "")),
        created=str(data.get("created", "")),
        method=str(data.get("method", "")),
        diagnostics=diagnostics,
        format_version=format_version,
    )

  @classmethod
  def from_estimate(
      cls,
      metric: str,
      est: MEstimate | R2MEstimate,
      *,
      kappa: float | None = None,
      kappa_labels: float | None = None,
      source: str = "",
      archive_fingerprint: str = "",
  ) -> "Declaration":
    """Constructs a Declaration from an MEstimate or R2MEstimate picking m_upper."""
    norm_metric = metric.strip().lower()
    created = datetime.date.today().isoformat()

    if isinstance(est, R2MEstimate):
      if norm_metric != "r2":
        raise ValueError(f"R2MEstimate provided for non-r2 metric: {metric}")
      diag = {
          "errors_m_point": est.m_errors.m_point,
          "errors_m_upper": est.m_errors.m_upper,
          "errors_top1pct_share": est.m_errors.top1pct_share,
          "errors_m_without_top1pct": est.m_errors.m_without_top1pct,
          "labels_m_point": est.m_labels.m_point,
          "labels_m_upper": est.m_labels.m_upper,
          "labels_top1pct_share": est.m_labels.top1pct_share,
          "labels_m_without_top1pct": est.m_labels.m_without_top1pct,
      }
      method_desc = (
          f"estimate_m_r2 units={est.m_errors.num_units} "
          f"items={est.m_errors.num_items}"
      )
      resolved_kl = kappa if kappa_labels is None else kappa_labels
      return cls(
          metric="r2",
          m=est.m_errors.m_upper,
          m_labels=est.m_labels.m_upper,
          kappa=kappa,
          kappa_labels=resolved_kl,
          source=source,
          archive_fingerprint=archive_fingerprint,
          created=created,
          method=method_desc,
          diagnostics=diag,
      )
    elif isinstance(est, MEstimate):
      if norm_metric == "r2":
        raise ValueError("MEstimate provided for r2 metric; use estimate_m_r2")
      diag = {
          "m_point": est.m_point,
          "m_upper": est.m_upper,
          "max_unit_share": est.max_unit_share,
          "top1pct_share": est.top1pct_share,
          "m_without_top1pct": est.m_without_top1pct,
      }
      method_desc = f"estimate_m units={est.num_units} items={est.num_items}"
      return cls(
          metric=norm_metric,
          m=est.m_upper,
          kappa=kappa,
          source=source,
          archive_fingerprint=archive_fingerprint,
          created=created,
          method=method_desc,
          diagnostics=diag,
      )
    else:
      raise TypeError(f"Unsupported estimate type: {type(est)}")


def fingerprint(values: Any) -> str:
  """SHA-256 of little-endian float64 bytes of a 1-D array."""
  arr = np.asarray(values, dtype="<f8").reshape(-1)
  return hashlib.sha256(arr.tobytes()).hexdigest()


def save_declarations(path: str, decls: Mapping[str, Declaration]) -> None:
  """Saves a mapping of named declarations to a JSON file."""
  data = {k: json.loads(d.to_json()) for k, d in decls.items()}
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, sort_keys=True, indent=2)


def load_declarations(path: str) -> dict[str, Declaration]:
  """Loads a mapping of named declarations from a JSON file."""
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  return {k: Declaration.from_json(json.dumps(v)) for k, v in data.items()}


def estimate_m(
    summands: Sequence[float] | np.ndarray,
    *,
    cluster_ids: Sequence[Any] | np.ndarray | None = None,
    level: float = 0.95,
    num_resamples: int = 2000,
    seed: int = 0,
) -> MEstimate:
  """Estimates relative variance bound M via cluster or item bootstrap.

  Args:
    summands: 1-D sequence of non-negative finite loss summands s_i.
    cluster_ids: Optional sequence of cluster identifiers. If provided, units
      are clusters holding (n_c, sum_c s, sum_c s^2). Otherwise units are items.
    level: Bootstrap quantile confidence level in (0, 1). Default is 0.95.
    num_resamples: Number of bootstrap draws (>= 100). Default is 2000.
    seed: Random seed for bootstrap resampling.

  Returns:
    MEstimate containing point estimate, bootstrap upper quantile, and tail
    diagnostics.
  """
  s = np.asarray(summands, dtype=np.float64)
  if s.ndim != 1:
    raise ValueError(f"Summands must be 1-D, got shape {s.shape}.")
  if s.size == 0:
    raise ValueError("Summands array must not be empty.")
  if not np.all(np.isfinite(s)):
    raise ValueError("Summands must be finite.")
  if np.any(s < 0.0):
    raise ValueError("Summands must be non-negative (>= 0).")
  tot_s = float(np.sum(s))
  if tot_s <= 0.0:
    raise ValueError(f"Sum of summands must be strictly positive, got {tot_s}.")

  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")
  if num_resamples < 100:
    raise ValueError(f"num_resamples must be >= 100, got {num_resamples}.")

  n = int(s.size)
  tot_s2 = float(np.sum(s**2))
  m_point = (float(n) * tot_s2) / (tot_s**2)

  if cluster_ids is not None:
    c_arr = np.asarray(cluster_ids)
    if c_arr.shape != s.shape:
      raise ValueError(
          f"cluster_ids length {len(c_arr)} does not match summands length {n}."
      )
    _, inv = np.unique(c_arr, return_inverse=True)
    unit_n = np.bincount(inv).astype(np.int64)
    unit_s = np.bincount(inv, weights=s).astype(np.float64)
    unit_s2 = np.bincount(inv, weights=s**2).astype(np.float64)
  else:
    unit_n = np.ones(n, dtype=np.int64)
    unit_s = s
    unit_s2 = s**2

  num_units = int(unit_n.size)
  max_unit_share = float(np.max(unit_s2)) / tot_s2 if tot_s2 > 0.0 else 0.0

  # Top 1% share and point M without top 1%
  k_top = int(math.ceil(0.01 * num_units))
  sorted_s2 = np.sort(unit_s2)[::-1]
  top1pct_share = (
      float(np.sum(sorted_s2[:k_top])) / tot_s2 if tot_s2 > 0.0 else 0.0
  )

  top_indices = np.argsort(unit_s2)[-k_top:]
  keep_mask = np.ones(num_units, dtype=bool)
  keep_mask[top_indices] = False

  n_rem = int(np.sum(unit_n[keep_mask]))
  s_rem = float(np.sum(unit_s[keep_mask]))
  s2_rem = float(np.sum(unit_s2[keep_mask]))
  if s_rem > 0.0:
    m_without_top1pct = (float(n_rem) * s2_rem) / (s_rem**2)
  else:
    m_without_top1pct = float("nan")

  # Bootstrap resampling of units in memory-bounded blocks
  rng = np.random.default_rng(seed)
  block_size = max(1, (2**22) // num_units)
  boot_m_parts: list[np.ndarray] = []
  num_skipped = 0

  rem = num_resamples
  while rem > 0:
    curr_b = min(rem, block_size)
    rem -= curr_b

    resample_idx = rng.integers(0, num_units, size=(curr_b, num_units))
    b_n = np.sum(unit_n[resample_idx], axis=1)
    b_s = np.sum(unit_s[resample_idx], axis=1)
    b_s2 = np.sum(unit_s2[resample_idx], axis=1)

    valid = b_s > 0.0
    num_skipped += int(np.count_nonzero(~valid))
    if np.any(valid):
      b_m = (b_n[valid].astype(np.float64) * b_s2[valid]) / (b_s[valid] ** 2)
      boot_m_parts.append(b_m)

  if not boot_m_parts:
    raise ValueError("All bootstrap resamples had sum(s*) = 0.")

  boot_m = np.concatenate(boot_m_parts)
  m_upper = float(np.quantile(boot_m, level, method="higher"))

  return MEstimate(
      m_point=float(m_point),
      m_upper=float(m_upper),
      num_units=num_units,
      num_items=n,
      max_unit_share=max_unit_share,
      top1pct_share=top1pct_share,
      m_without_top1pct=float(m_without_top1pct),
      num_skipped_resamples=num_skipped,
  )


def estimate_m_r2(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    cluster_ids: Sequence[Any] | np.ndarray | None = None,
    level: float = 0.95,
    num_resamples: int = 2000,
    seed: int = 0,
) -> R2MEstimate:
  """Estimates relative variance bounds M for R^2 evaluation.

  Computes M estimates for:
  1. Prediction errors: s_e = (y_true - y_pred)^2
  2. Label variation: q_i = (y_i - y_bar)^2 using archive mean y_bar

  Args:
    y_true: 1-D sequence of ground-truth target values.
    y_pred: 1-D sequence of model predictions.
    cluster_ids: Optional cluster identifiers.
    level: Bootstrap quantile confidence level in (0, 1). Default 0.95.
    num_resamples: Number of bootstrap draws (>= 100). Default 2000.
    seed: Random seed for bootstrap draws.

  Returns:
    R2MEstimate with m_errors and m_labels.
  """
  yt = np.asarray(y_true, dtype=np.float64)
  yp = np.asarray(y_pred, dtype=np.float64)
  if yt.ndim != 1 or yp.ndim != 1:
    raise ValueError("y_true and y_pred must be 1-D arrays.")
  if yt.shape != yp.shape:
    raise ValueError(
        f"y_true shape {yt.shape} does not match y_pred shape {yp.shape}."
    )
  if yt.size == 0:
    raise ValueError("Input arrays must not be empty.")
  if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
    raise ValueError("y_true and y_pred must be finite.")

  e = yt - yp
  s_e = e**2

  y_bar = float(np.mean(yt))
  q = (yt - y_bar) ** 2

  m_errors = estimate_m(
      s_e,
      cluster_ids=cluster_ids,
      level=level,
      num_resamples=num_resamples,
      seed=seed,
  )
  m_labels = estimate_m(
      q,
      cluster_ids=cluster_ids,
      level=level,
      num_resamples=num_resamples,
      seed=seed,
  )
  return R2MEstimate(m_errors=m_errors, m_labels=m_labels)
