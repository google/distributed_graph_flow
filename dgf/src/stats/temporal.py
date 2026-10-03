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

"""Temporal confidence intervals via declare-and-refute window aggregation.

Provides the temporal entry point `temporal_interval` for time-ordered
evaluation data, and `temporal_interval_from_shard` for pre-aggregated window
totals. The caller declares kappa (default 1.0, adjacent windows uncorrelated)
and optionally M_c (default per-metric bound). The interval is certified under
these declarations, while statistical refutation tests and split-sample drift
diagnostics check whether the data contradict them.

Estimand:
  The mean loss of a stationary process over the validation period. This is also
  the expected loss on future items only if the future is stationary with it;
  drift is not covered.

Refutation checks and limitations:
  For kappa = 1, consecutive window totals are tested for zero lag-1
  autocorrelation (Fisher-z test). This test cannot detect autocorrelation
  beyond adjacent windows or non-linear dependencies.
  For kappa > 1, a batch-means lower confidence bound tests variance inflation.
  This test is weak when 30 super-batches are shorter than the correlation
  length of the process.
"""

from collections.abc import Sequence
import dataclasses
import enum
import math
from typing import Any

import numpy as np

from dgf.src.stats import cluster_bound as _cluster_bound
from dgf.src.stats import cluster_sketch as _cluster_sketch
from dgf.src.stats import refutation as _refutation
from dgf.src.stats.independent import interval as _interval
from dgf.src.stats.independent import metrics as _metrics

__all__ = [
    "Drift",
    "DriftResult",
    "TemporalResult",
    "temporal_interval",
    "temporal_interval_from_buckets",
    "temporal_interval_from_shard",
]

# Nominal confidence level for refutation tests.
_REFUTATION_LEVEL: float = 0.95

# Number of super-batches for batch-means kappa testing.
_KAPPA_NUM_BATCHES: int = 30
_COUNT_H_MAX: float = 1.1


class Drift(enum.Enum):
  """Outcome of the split-sample distribution drift diagnostic."""

  WARNING = "WARNING"
  NO_WARNING = "NO_WARNING"
  UNTESTABLE = "UNTESTABLE"


@dataclasses.dataclass(frozen=True)
class DriftResult:
  """Result of the split-sample drift diagnostic.

  Attributes:
    outcome: Drift diagnostic classification (WARNING, NO_WARNING, UNTESTABLE).
    first_half: (low, high) confidence interval on the first half of windows.
    second_half: (low, high) confidence interval on the second half of windows.
    first_kappa_check: Refutation check on the first half window totals.
    second_kappa_check: Refutation check on the second half window totals.
    message: Human-readable explanation of drift check outcome.
  """

  outcome: Drift
  first_half: tuple[float, float]
  second_half: tuple[float, float]
  first_kappa_check: _refutation.RefutationResult
  second_kappa_check: _refutation.RefutationResult
  message: str


@dataclasses.dataclass(frozen=True)
class TemporalResult:
  """Confidence interval result for temporal evaluation data.

  Attributes:
    low: Lower confidence bound on the estimand (nan if no interval was issued).
    high: Upper confidence bound on the estimand (nan if no interval was
      issued).
    level: Confidence level, or None when no interval was issued.
    status: Diagnostic status of the interval from the M check.
    kappa_declared: The declared variance inflation factor kappa.
    kappa_check: Outcome and statistics of the kappa refutation test.
    m_declared: Declared relative variance bound M_c for window totals. For R^2,
      this is the error side M_c_A; the label side bound M_c_q is in r2_detail.
    m_observed: Empirical relative variance of window totals. For R^2, this is
      the error side m_hat_e2; the label side m_hat_q is in r2_detail (so status
      can be TAIL_UNRESOLVED while m_observed <= m_declared).
    num_windows: Number of contiguous time windows G.
    mean_window_size: Average number of items per window (n / G).
    effective_n: Effective number of independent windows (num_windows /
      kappa_declared).
    diagnostic: Actionable diagnostic payload (None on invalid input).
    message: Human-readable summary of interval validity and refutation checks.
    drift: Split-sample distribution drift diagnostic.
    count_energy: Diagnostic count energy rho_hat (None when window counts are
      within +-1); does not gate refutation.
    count_h: Count heterogeneity ratio h used to guard refutation checks (None
      on refusal paths).
  """

  low: float
  high: float
  level: float | None
  status: _interval.Status
  kappa_declared: float
  kappa_check: _refutation.RefutationResult
  m_declared: float
  m_observed: float
  num_windows: int
  mean_window_size: float
  effective_n: float
  diagnostic: _cluster_bound.DiagnosticPayload | None
  message: str
  drift: DriftResult
  count_energy: float | None = None
  count_h: float | None = None
  r2_detail: _cluster_bound.R2Detail | None = None
  n_max_over_m_bar: float | None = None
  b_max: int | None = None

  @property

  def refuted(self) -> bool:
    """True if the data contradict a declaration (M or kappa)."""
    return (
        self.status is _interval.Status.TAIL_UNRESOLVED
        or self.kappa_check.outcome is _refutation.Refutation.REFUTED
    )


def _check_kappa_with_guard(
    shard: _cluster_sketch.PartialClusterShard,
    resolved_kappa: float,
    *,
    use_path_edges_for_k1: bool = False,
) -> tuple[_refutation.RefutationResult, float | None, float, str]:
  """Runs kappa check on window residuals with count-only heterogeneity guard.

  The checks run on residuals S_c - n_c θ̂, which remove the count component.
  The guard evaluates the remaining heteroscedasticity for window variances
  proportional to n_c and to n_c²; it is a heuristic for other within-window
  dependence. ±1 counts pass.
  """
  counts = shard.cluster_counts
  totals = shard.cluster_totals
  g = int(len(counts))
  n = int(np.sum(counts))
  m_bar = float(n) / float(g)

  max_c = int(np.max(counts))
  min_c = int(np.min(counts))

  if max_c - min_c <= 1:
    count_energy = None
  else:
    theta_hat = float(np.sum(totals)) / float(n) if n > 0 else 0.0
    num = (theta_hat**2) * float(np.sum((counts - m_bar) ** 2))
    denom = float(np.sum((totals - counts * theta_hat) ** 2))
    if denom == 0.0 or not math.isfinite(denom):
      rho_hat = float("inf")
    else:
      rho_hat = num / denom
    count_energy = float(rho_hat)

  counts_f = counts.astype(np.float64)
  h_vals = []
  for a in (1, 2):
    n_pow = counts_f**a
    num_a = float(np.mean(n_pow[:-1] * n_pow[1:]))
    den_a = float(np.mean(n_pow) ** 2)
    if den_a == 0.0 or not math.isfinite(den_a):
      h_vals.append(float("inf"))
    else:
      h_vals.append(num_a / den_a)
  h = float(max(h_vals))

  if resolved_kappa == 1.0 and use_path_edges_for_k1:
    # Path edges (j, j+1), NO count guard
    theta_hat = float(np.sum(totals)) / float(n) if n > 0 else 0.0
    e = totals - counts_f * theta_hat
    if g < 2:
      path_edges = np.zeros((0, 2), dtype=np.int64)
    else:
      path_edges = np.column_stack(
          [np.arange(g - 1, dtype=np.int64), np.arange(1, g, dtype=np.int64)]
      )
    kappa_check = _refutation.check_uncorrelated_edges(
        e, counts, path_edges, level=_REFUTATION_LEVEL
    )
    if kappa_check.outcome is _refutation.Refutation.UNTESTABLE:
      k_msg = (
          "kappa check untestable"
          f" ({kappa_check.reason or 'fewer than 30 edges'})."
      )
    elif kappa_check.outcome is _refutation.Refutation.REFUTED:
      k_msg = (
          "kappa=1 refuted: lag-1 correlation of window totals is significant"
          f" (T={kappa_check.statistic:.2f} > {kappa_check.threshold:.2f})."
      )
    else:
      k_msg = (
          f"kappa=1 not refuted (T={kappa_check.statistic:.2f} <="
          f" {kappa_check.threshold:.2f})."
      )
    return kappa_check, count_energy, h, k_msg

  if not math.isfinite(h) or h > _COUNT_H_MAX:
    run_check = False
    guard_msg = (
        f"count_h {h:.4g} > {_COUNT_H_MAX:g}: window counts too heterogeneous"
        " for the κ check"
    )
  else:
    run_check = True
    guard_msg = ""

  if not run_check:
    kappa_check = _refutation.RefutationResult(
        _refutation.Refutation.UNTESTABLE, None, None
    )
    k_msg = f"kappa check untestable ({guard_msg})."
  else:
    theta_hat = float(np.sum(totals)) / float(n) if n > 0 else 0.0
    e = totals - counts_f * theta_hat
    if resolved_kappa == 1.0:
      kappa_check = _refutation.check_uncorrelated(
          e, level=_REFUTATION_LEVEL
      )
      if kappa_check.outcome is _refutation.Refutation.UNTESTABLE:
        k_msg = "kappa check untestable (fewer than 5 windows)."
      elif kappa_check.outcome is _refutation.Refutation.REFUTED:
        k_msg = (
            "kappa=1 refuted: lag-1 correlation of window totals is significant"
            f" (z={kappa_check.statistic:.2f} > {kappa_check.threshold:.2f})."
        )
      else:
        k_msg = f"kappa=1 not refuted (z={kappa_check.statistic:.2f} <= {kappa_check.threshold:.2f})."
    else:
      kappa_check = _refutation.check_kappa(
          e,
          kappa_declared=resolved_kappa,
          level=_REFUTATION_LEVEL,
          num_batches=_KAPPA_NUM_BATCHES,
      )
      if kappa_check.outcome is _refutation.Refutation.UNTESTABLE:
        k_msg = f"kappa check untestable (fewer than {_KAPPA_NUM_BATCHES} windows)."
      elif kappa_check.outcome is _refutation.Refutation.REFUTED:
        k_msg = (
            f"kappa={resolved_kappa:g} refuted: lower bound"
            f" {kappa_check.statistic:.2f} > {resolved_kappa:g}."
        )
      else:
        k_msg = (
            f"kappa={resolved_kappa:g} not refuted (lower bound"
            f" {kappa_check.statistic:.2f} <= {resolved_kappa:g})."
        )

  return kappa_check, count_energy, h, k_msg


def _temporal_interval_core(
    shard: _cluster_sketch.PartialClusterShard | _cluster_sketch.R2ClusterShard,
    *,
    metric: str,
    level: float,
    m: float | None,
    m_item: float | None = None,
    kappa: float,
    c_target: float,
    m_labels: float | None = None,
    kappa_labels: float | None = None,
    use_path_edges_for_k1: bool = False,
    b_max: int | None = None,
) -> TemporalResult:
  """Shared core implementation of temporal confidence interval evaluation."""
  if m is not None and m_item is not None:
    raise ValueError("Cannot pass both 'm' and 'm_item'; specify only one.")

  # 1. Validate cluster_ids are 0..G-1
  c_ids = shard.cluster_ids
  if not np.issubdtype(c_ids.dtype, np.integer):
    raise ValueError(f"cluster_ids must be integers, got dtype {c_ids.dtype}.")
  g = int(c_ids.size)
  if g < 2:
    raise ValueError(f"Number of windows G must be >= 2, got {g}.")
  for i in range(g):
    if int(c_ids[i]) != i:
      raise ValueError(
          f"cluster_ids must be contiguous 0..G-1; first gap or mismatch at index {i}, got id {c_ids[i]}."
      )

  resolved_kappa = float(kappa)
  n = int(
      shard.n_items
      if hasattr(shard, "n_items")
      else np.sum(shard.cluster_counts)
  )
  n_max = int(np.max(shard.cluster_counts))
  m_bar = float(n) / float(g)
  n_max_over_m_bar = float(n_max) / m_bar if m_bar > 0 else 1.0

  counts = shard.cluster_counts
  counts_f = counts.astype(np.float64)
  h_vals = []
  for a in (1, 2):
    n_pow = counts_f**a
    num_a = float(np.mean(n_pow[:-1] * n_pow[1:]))
    den_a = float(np.mean(n_pow) ** 2)
    if den_a == 0.0 or not math.isfinite(den_a):
      h_vals.append(float("inf"))
    else:
      h_vals.append(num_a / den_a)
  h = float(max(h_vals))

  if metric == "r2":
    if not isinstance(shard, _cluster_sketch.R2ClusterShard):
      raise ValueError("For metric='r2', shard must be an R2ClusterShard.")

    def run_kappa_check(res_vec, counts_vec, k_val):
      if k_val == 1.0:
        if use_path_edges_for_k1:
          if g < 2:
            path_edges = np.zeros((0, 2), dtype=np.int64)
          else:
            path_edges = np.column_stack([
                np.arange(g - 1, dtype=np.int64),
                np.arange(1, g, dtype=np.int64),
            ])
          return _refutation.check_uncorrelated_edges(
              res_vec, counts_vec, path_edges, level=_REFUTATION_LEVEL
          )
        else:
          if not math.isfinite(h) or h > _COUNT_H_MAX:
            return _refutation.RefutationResult(
                _refutation.Refutation.UNTESTABLE,
                None,
                None,
                reason=f"count_h {h:.4g} > {_COUNT_H_MAX:g}",
            )
          return _refutation.check_uncorrelated(
              res_vec, level=_REFUTATION_LEVEL
          )
      else:
        if not math.isfinite(h) or h > _COUNT_H_MAX:
          return _refutation.RefutationResult(
              _refutation.Refutation.UNTESTABLE,
              None,
              None,
              reason=f"count_h {h:.4g} > {_COUNT_H_MAX:g}",
          )
        return _refutation.check_kappa(
            res_vec,
            kappa_declared=k_val,
            level=_REFUTATION_LEVEL,
            num_batches=_KAPPA_NUM_BATCHES,
        )

    low, high, level_out, status, r2_detail, k_msg, kappa_check = (
        _cluster_bound.compute_r2_interval(
            shard.e2_totals,
            shard.y_totals,
            shard.y2_totals,
            shard.cluster_counts,
            level=level,
            m=m,
            m_item=m_item,
            m_labels=m_labels,
            kappa=resolved_kappa,
            kappa_labels=kappa_labels,
            run_kappa_check_fn=run_kappa_check,
        )
    )

    untestable_ref = _refutation.RefutationResult(
        _refutation.Refutation.UNTESTABLE, None, None
    )
    if status is _interval.Status.ASSUMPTION_REQUIRED:
      drift = DriftResult(
          outcome=Drift.UNTESTABLE,
          first_half=(float("nan"), float("nan")),
          second_half=(float("nan"), float("nan")),
          first_kappa_check=untestable_ref,
          second_kappa_check=untestable_ref,
          message="no interval",
      )
    elif g < 4:
      drift = DriftResult(
          outcome=Drift.UNTESTABLE,
          first_half=(float("nan"), float("nan")),
          second_half=(float("nan"), float("nan")),
          first_kappa_check=untestable_ref,
          second_kappa_check=untestable_ref,
          message="drift check untestable (fewer than 4 windows).",
      )
    else:
      g1 = g // 2
      shard_1 = _cluster_sketch.R2ClusterShard(
          cluster_ids=np.arange(g1, dtype=np.int64),
          e2_totals=shard.e2_totals[:g1],
          y_totals=shard.y_totals[:g1],
          y2_totals=shard.y2_totals[:g1],
          cluster_counts=shard.cluster_counts[:g1],
          ref=shard.ref,
      )
      shard_2 = _cluster_sketch.R2ClusterShard(
          cluster_ids=np.arange(g - g1, dtype=np.int64),
          e2_totals=shard.e2_totals[g1:],
          y_totals=shard.y_totals[g1:],
          y2_totals=shard.y2_totals[g1:],
          cluster_counts=shard.cluster_counts[g1:],
          ref=shard.ref,
      )

      alpha_half = (1.0 - level) / 2.0
      half_level = 1.0 - alpha_half

      def make_half_check_fn(half_g, half_counts_f):
        h_vals_half = []
        for a in (1, 2):
          n_pow = half_counts_f**a
          num_a = float(np.mean(n_pow[:-1] * n_pow[1:]))
          den_a = float(np.mean(n_pow) ** 2)
          if den_a == 0.0 or not math.isfinite(den_a):
            h_vals_half.append(float("inf"))
          else:
            h_vals_half.append(num_a / den_a)
        h_half = float(max(h_vals_half))

        def run_half_check(res_vec, counts_vec, k_val):
          if k_val == 1.0:
            if use_path_edges_for_k1:
              if half_g < 2:
                pe = np.zeros((0, 2), dtype=np.int64)
              else:
                pe = np.column_stack([
                    np.arange(half_g - 1, dtype=np.int64),
                    np.arange(1, half_g, dtype=np.int64),
                ])
              return _refutation.check_uncorrelated_edges(
                  res_vec, counts_vec, pe, level=_REFUTATION_LEVEL
              )
            else:
              if not math.isfinite(h_half) or h_half > _COUNT_H_MAX:
                return _refutation.RefutationResult(
                    _refutation.Refutation.UNTESTABLE,
                    None,
                    None,
                    reason=f"count_h {h_half:.4g} > {_COUNT_H_MAX:g}",
                )
              return _refutation.check_uncorrelated(
                  res_vec, level=_REFUTATION_LEVEL
              )
          else:
            if not math.isfinite(h_half) or h_half > _COUNT_H_MAX:
              return _refutation.RefutationResult(
                  _refutation.Refutation.UNTESTABLE,
                  None,
                  None,
                  reason=f"count_h {h_half:.4g} > {_COUNT_H_MAX:g}",
              )
            return _refutation.check_kappa(
                res_vec,
                kappa_declared=k_val,
                level=_REFUTATION_LEVEL,
                num_batches=_KAPPA_NUM_BATCHES,
            )

        return run_half_check

      low_1, high_1, _, status_1, _, _, k_check_1 = (
          _cluster_bound.compute_r2_interval(
              shard_1.e2_totals,
              shard_1.y_totals,
              shard_1.y2_totals,
              shard_1.cluster_counts,
              level=half_level,
              m=m,
              m_item=m_item,
              m_labels=m_labels,
              kappa=resolved_kappa,
              kappa_labels=kappa_labels,
              run_kappa_check_fn=make_half_check_fn(
                  g1, shard_1.cluster_counts.astype(np.float64)
              ),
          )
      )
      low_2, high_2, _, status_2, _, _, k_check_2 = (
          _cluster_bound.compute_r2_interval(
              shard_2.e2_totals,
              shard_2.y_totals,
              shard_2.y2_totals,
              shard_2.cluster_counts,
              level=half_level,
              m=m,
              m_item=m_item,
              m_labels=m_labels,
              kappa=resolved_kappa,
              kappa_labels=kappa_labels,
              run_kappa_check_fn=make_half_check_fn(
                  g - g1, shard_2.cluster_counts.astype(np.float64)
              ),
          )
      )

      has_int_1 = (
          status_1 is _interval.Status.UNREFUTED
          and math.isfinite(low_1)
          and math.isfinite(high_1)
      )
      has_int_2 = (
          status_2 is _interval.Status.UNREFUTED
          and math.isfinite(low_2)
          and math.isfinite(high_2)
      )

      if not (has_int_1 and has_int_2):
        drift_outcome = Drift.UNTESTABLE
        drift_msg = (
            "drift check untestable (at least one half has no interval)."
        )
      else:
        disjoint = (low_1 > high_2) or (low_2 > high_1)
        if disjoint:
          drift_outcome = Drift.WARNING
          drift_msg = (
              "drift warning: first half interval "
              f"[{low_1:.4g}, {high_1:.4g}] and second half interval "
              f"[{low_2:.4g}, {high_2:.4g}] are disjoint."
          )
        else:
          drift_outcome = Drift.NO_WARNING
          drift_msg = (
              "no drift: first half interval "
              f"[{low_1:.4g}, {high_1:.4g}] and second half interval "
              f"[{low_2:.4g}, {high_2:.4g}] overlap."
          )

      drift = DriftResult(
          outcome=drift_outcome,
          first_half=(low_1, high_1),
          second_half=(low_2, high_2),
          first_kappa_check=k_check_1,
          second_kappa_check=k_check_2,
          message=drift_msg,
      )

    full_message = f"{k_msg} {drift.message}".strip()
    return TemporalResult(
        low=low,
        high=high,
        level=level_out,
        status=status,
        kappa_declared=resolved_kappa,
        kappa_check=kappa_check,
        m_declared=(
            r2_detail.M_c_A
            if r2_detail is not None
            else (
                float(m)
                if m is not None
                else (float(m_item) if m_item is not None else 16.0)
            )
        ),
        m_observed=(
            r2_detail.m_hat_e2 if r2_detail is not None else float("nan")
        ),
        num_windows=g,
        mean_window_size=m_bar,
        effective_n=float(g) / resolved_kappa,
        diagnostic=None,
        message=full_message,
        drift=drift,
        count_energy=None,
        count_h=h,
        r2_detail=r2_detail,
        n_max_over_m_bar=n_max_over_m_bar,
        b_max=b_max,
    )

  # Non-R2 metrics:
  assert isinstance(shard, _cluster_sketch.PartialClusterShard)
  # 2. Kappa refutation check with count guard
  kappa_check, count_energy, count_h, k_msg = _check_kappa_with_guard(
      shard, resolved_kappa, use_path_edges_for_k1=use_path_edges_for_k1
  )

  # 3. Main interval via cluster_interval on shard's sketch
  sk = shard.to_cluster_sketch(kappa_cluster=resolved_kappa)
  c_res = _cluster_bound.cluster_interval(
      sk,
      metric=metric,
      level=level,
      m=m,
      m_item=m_item,
      kappa_cluster=resolved_kappa,
      c_target=c_target,
  )

  # 4. Drift warning via split-sample halves
  untestable_ref = _refutation.RefutationResult(
      _refutation.Refutation.UNTESTABLE, None, None
  )
  if c_res.status is _interval.Status.ASSUMPTION_REQUIRED:
    drift = DriftResult(
        outcome=Drift.UNTESTABLE,
        first_half=(float("nan"), float("nan")),
        second_half=(float("nan"), float("nan")),
        first_kappa_check=untestable_ref,
        second_kappa_check=untestable_ref,
        message="no interval",
    )
  elif g < 4:
    drift = DriftResult(
        outcome=Drift.UNTESTABLE,
        first_half=(float("nan"), float("nan")),
        second_half=(float("nan"), float("nan")),
        first_kappa_check=untestable_ref,
        second_kappa_check=untestable_ref,
        message="drift check untestable (fewer than 4 windows).",
    )
  else:
    g1 = g // 2
    shard_1 = _cluster_sketch.PartialClusterShard(
        cluster_ids=np.arange(g1, dtype=np.int64),
        cluster_totals=shard.cluster_totals[:g1],
        cluster_counts=shard.cluster_counts[:g1],
    )
    shard_2 = _cluster_sketch.PartialClusterShard(
        cluster_ids=np.arange(g - g1, dtype=np.int64),
        cluster_totals=shard.cluster_totals[g1:],
        cluster_counts=shard.cluster_counts[g1:],
    )

    alpha_half = (1.0 - level) / 2.0
    half_level = 1.0 - alpha_half

    sk_1 = shard_1.to_cluster_sketch(kappa_cluster=resolved_kappa)
    sk_2 = shard_2.to_cluster_sketch(kappa_cluster=resolved_kappa)

    res_1 = _cluster_bound.cluster_interval(
        sk_1,
        metric=metric,
        level=half_level,
        m=m,
        m_item=m_item,
        kappa_cluster=resolved_kappa,
        c_target=c_target,
    )
    res_2 = _cluster_bound.cluster_interval(
        sk_2,
        metric=metric,
        level=half_level,
        m=m,
        m_item=m_item,
        kappa_cluster=resolved_kappa,
        c_target=c_target,
    )

    k_check_1, _, _, _ = _check_kappa_with_guard(
        shard_1, resolved_kappa, use_path_edges_for_k1=use_path_edges_for_k1
    )
    k_check_2, _, _, _ = _check_kappa_with_guard(
        shard_2, resolved_kappa, use_path_edges_for_k1=use_path_edges_for_k1
    )

    has_int_1 = (
        res_1.status is _interval.Status.UNREFUTED
        and math.isfinite(res_1.low)
        and math.isfinite(res_1.high)
    )
    has_int_2 = (
        res_2.status is _interval.Status.UNREFUTED
        and math.isfinite(res_2.low)
        and math.isfinite(res_2.high)
    )

    if not (has_int_1 and has_int_2):
      drift_outcome = Drift.UNTESTABLE
      drift_msg = "drift check untestable (at least one half has no interval)."
    else:
      disjoint = (res_1.low > res_2.high) or (res_2.low > res_1.high)
      if disjoint:
        drift_outcome = Drift.WARNING
        drift_msg = (
            "drift warning: first half interval "
            f"[{res_1.low:.4g}, {res_1.high:.4g}] and second half interval "
            f"[{res_2.low:.4g}, {res_2.high:.4g}] are disjoint."
        )
      else:
        drift_outcome = Drift.NO_WARNING
        drift_msg = (
            "no drift: first half interval "
            f"[{res_1.low:.4g}, {res_1.high:.4g}] and second half interval "
            f"[{res_2.low:.4g}, {res_2.high:.4g}] overlap."
        )

    drift = DriftResult(
        outcome=drift_outcome,
        first_half=(res_1.low, res_1.high),
        second_half=(res_2.low, res_2.high),
        first_kappa_check=k_check_1,
        second_kappa_check=k_check_2,
        message=drift_msg,
    )

  diag_msg = c_res.diagnostic.message if c_res.diagnostic is not None else ""
  full_message = f"{diag_msg} {k_msg} {drift.message}".strip()

  return TemporalResult(
      low=c_res.low,
      high=c_res.high,
      level=c_res.level,
      status=c_res.status,
      kappa_declared=resolved_kappa,
      kappa_check=kappa_check,
      m_declared=c_res.m_declared,
      m_observed=c_res.m_observed,
      num_windows=c_res.num_clusters,
      mean_window_size=c_res.mean_cluster_size,
      effective_n=c_res.effective_n,
      diagnostic=c_res.diagnostic,
      message=full_message,
      drift=drift,
      count_energy=count_energy,
      count_h=count_h,
      r2_detail=None,
      n_max_over_m_bar=n_max_over_m_bar,
      b_max=b_max,
  )


def temporal_interval_from_shard(
    shard: _cluster_sketch.PartialClusterShard | _cluster_sketch.R2ClusterShard,
    *,
    metric: str = "mse",
    level: float = 0.95,
    m: float | None = None,
    m_item: float | None = None,
    kappa: float = 1.0,
    c_target: float = 0.2,
    m_labels: float | None = None,
    kappa_labels: float | None = None,
) -> TemporalResult:
  """Computes certified temporal confidence interval from pre-aggregated window totals.

  The shard's cluster totals are expected to be summands of `metric` (e.g. built
  with `PartialClusterShard.from_data(..., metric)` or `R2ClusterShard`).

  Args:
    shard: PartialClusterShard or R2ClusterShard with integer cluster IDs 0..G-1
      in time order.
    metric: Metric name ('mae', 'mse', 'rmse', 'accuracy', 'r2').
    level: Confidence level in (0, 1). Default is 0.95.
    m: Declared relative variance bound M_c for window totals, used unchanged.
      Mutually exclusive with m_item. If both m and m_item are None, uses the
      sketch's default M_c (Lemma I', or 16.0 converted for r2).
    m_item: Item-level bound M (e.g. from declare.estimate_m); M_c is derived
      with Lemma I' from the actual cluster sizes. Mutually exclusive with m.
    kappa: Declared variance inflation factor kappa (>= 1.0). Default is 1.0.
    c_target: Target relative half-width factor in (0, 1). Default is 0.2.
    m_labels: Declared item-level relative variance bound for labels (r2 only),
      converted with Lemma I'. If None, equals converted m_item if m_item given,
      equals unconverted m if m given, or default 16.0 converted.
    kappa_labels: Declared kappa for labels (r2 only).

  Returns:
    TemporalResult with interval endpoints, diagnostics, kappa check, count
    energy, and split-sample drift diagnostic.
  """
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")
  if not (0.0 < c_target < 1.0):
    raise ValueError(f"c_target must be in (0, 1), got {c_target}.")
  if m is not None and m_item is not None:
    raise ValueError("Cannot pass both 'm' and 'm_item'; specify only one.")
  if m is not None and not (math.isfinite(m) and m >= 1.0):
    raise ValueError(f"m must be finite and >= 1.0, got {m}.")
  if m_item is not None and not (math.isfinite(m_item) and m_item >= 1.0):
    raise ValueError(f"m_item must be finite and >= 1.0, got {m_item}.")
  if not (math.isfinite(kappa) and kappa >= 1.0):
    raise ValueError(f"kappa must be finite and >= 1.0, got {kappa}.")
  if m_labels is not None and not (math.isfinite(m_labels) and m_labels >= 1.0):
    raise ValueError(f"m_labels must be finite and >= 1.0, got {m_labels}.")
  if kappa_labels is not None and not (
      math.isfinite(kappa_labels) and kappa_labels >= 1.0
  ):
    raise ValueError(
        f"kappa_labels must be finite and >= 1.0, got {kappa_labels}."
    )

  if metric.strip().lower() == "r2":
    norm_metric = "r2"
    if not isinstance(shard, _cluster_sketch.R2ClusterShard):
      raise ValueError(
          "For metric='r2', shard must be an instance of R2ClusterShard."
      )
  else:
    norm_metric = _metrics.normalize_metric(metric)
    if isinstance(shard, _cluster_sketch.R2ClusterShard):
      raise ValueError("R2ClusterShard is only supported for metric='r2'.")

  return _temporal_interval_core(
      shard,
      metric=norm_metric,
      level=level,
      m=m,
      m_item=m_item,
      kappa=kappa,
      c_target=c_target,
      m_labels=m_labels,
      kappa_labels=kappa_labels,
  )


def temporal_interval(
    data: Any,
    *,
    num_windows: int,
    metric: str = "mse",
    level: float = 0.95,
    m: float | None = None,
    m_item: float | None = None,
    kappa: float = 1.0,
    timestamps: Any = None,
    c_target: float = 0.2,
    m_labels: float | None = None,
    kappa_labels: float | None = None,
) -> TemporalResult:
  """Computes a certified temporal confidence interval with refutation checks.

  The interval width uses only the declared kappa; the lag-1 check is a
  refutation test and never changes the interval.

  Estimand:
    The average expected loss over the evaluated items. It equals the expected
    loss on future items only if the data-generating process stays stationary;
    the drift warning can refute stationarity but never certifies it.

  Args:
    data: 1-D array-like of per-item residuals (0/1 indicators for accuracy).
      For metric='r2', a pair of 1-D arrays (y_true, y_pred). Tuples are
      rejected for non-R2 metrics.
    num_windows: Number of contiguous equal-count windows G in [2, n].
    metric: Metric name ('mae', 'mse', 'rmse', 'accuracy', 'r2').
    level: Confidence level in (0, 1). Default is 0.95.
    m: Declared relative variance bound M_c for window totals, used unchanged.
      Mutually exclusive with m_item. If both m and m_item are None, defaults to
      the window sketch's default_m_c applied to the per-metric item-level M (or
      16.0 converted for r2). On refusals raised before windows are formed (e.g.
      non-finite inputs), the reported m is the item-level M.
    m_item: Item-level bound M (e.g. from declare.estimate_m); M_c is derived
      with Lemma I' from the actual cluster sizes. Mutually exclusive with m.
    kappa: Declared variance inflation factor kappa (>= 1.0). Default is 1.0
      (asserts adjacent windows are uncorrelated).
    timestamps: Optional 1-D sequence of item timestamps. If provided, items are
      reordered by a stable sort on timestamps. If None, data is assumed to be
      in time order.
    c_target: Target relative half-width factor in (0, 1). Default is 0.2.
    m_labels: Declared relative second-moment bound for labels (r2 only).
    kappa_labels: Declared kappa for labels (r2 only).

  Returns:
    TemporalResult containing interval endpoints, diagnostic status, and
    refutation outcomes. Invalid inputs refuse with status ASSUMPTION_REQUIRED
    and NaN bounds without raising.

  Raises:
    ValueError: If data is a tuple for non-R2 or not a pair for R2, data or
      timestamps is not 1-D, data and timestamps have mismatched lengths,
      num_windows is not an int or outside [2, n], kappa is non-finite or < 1.0,
      level or c_target is outside (0, 1), m is non-finite or < 1.0, or metric
      is unsupported.
  """
  # 1. Metric normalization and input type validation
  if metric.strip().lower() == "r2":
    norm_metric = "r2"
    if not isinstance(data, tuple):
      raise ValueError(
          "For metric='r2', data must be a tuple of (y_true, y_pred)."
      )
    if len(data) != 2:
      raise ValueError(
          "For metric='r2', data tuple must have length 2 (y_true, y_pred),"
          f" got {len(data)}."
      )
    y_true_raw, y_pred_raw = data
    y_true = np.asarray(y_true_raw, dtype=np.float64)
    y_pred = np.asarray(y_pred_raw, dtype=np.float64)
    if y_true.ndim != 1 or y_pred.ndim != 1:
      raise ValueError("y_true and y_pred must be 1-D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
      raise ValueError(
          f"Length mismatch: y_true has {y_true.shape[0]} items, y_pred has"
          f" {y_pred.shape[0]}."
      )
    n = int(y_true.shape[0])
    data_arr = None
  else:
    norm_metric = _metrics.normalize_metric(metric)
    if isinstance(data, tuple):
      raise ValueError(
          "Tuples of (labels, predictions) are not accepted in"
          " temporal_interval; provide 1-D per-item residuals."
      )
    data_arr = np.asarray(data)
    if data_arr.ndim != 1:
      raise ValueError(f"data must be 1-D array, got shape {data_arr.shape}.")
    n = int(data_arr.shape[0])
    y_true = None
    y_pred = None

  # 2. Timestamps validation
  ts_arr = None
  if timestamps is not None:
    ts_arr = np.asarray(timestamps)
    if ts_arr.ndim != 1:
      raise ValueError(
          f"timestamps must be 1-D array, got shape {ts_arr.shape}."
      )
    if ts_arr.shape[0] != n:
      raise ValueError(
          f"Length mismatch: data has {n} items, timestamps has"
          f" {ts_arr.shape[0]}."
      )

  # 3. Parameter validation
  if isinstance(num_windows, bool) or not isinstance(
      num_windows, (int, np.integer)
  ):
    raise ValueError(
        f"num_windows must be an integer, got {type(num_windows).__name__}."
    )
  num_windows = int(num_windows)
  if num_windows < 2 or num_windows > n:
    raise ValueError(f"num_windows must be in [2, {n}], got {num_windows}.")

  if not (math.isfinite(kappa) and kappa >= 1.0):
    raise ValueError(f"kappa must be finite and >= 1.0, got {kappa}.")
  resolved_kappa = float(kappa)

  if not 0.0 < level < 1.0:
    raise ValueError(f"level must be in (0, 1), got {level}.")
  if not 0.0 < c_target < 1.0:
    raise ValueError(f"c_target must be in (0, 1), got {c_target}.")
  if m is not None and m_item is not None:
    raise ValueError("Cannot pass both 'm' and 'm_item'; specify only one.")
  if m is not None and not (math.isfinite(m) and m >= 1.0):
    raise ValueError(f"m must be finite and >= 1.0, got {m}.")
  if m_item is not None and not (math.isfinite(m_item) and m_item >= 1.0):
    raise ValueError(f"m_item must be finite and >= 1.0, got {m_item}.")
  if m_labels is not None and not (math.isfinite(m_labels) and m_labels >= 1.0):
    raise ValueError(f"m_labels must be finite and >= 1.0, got {m_labels}.")
  if kappa_labels is not None and not (
      math.isfinite(kappa_labels) and kappa_labels >= 1.0
  ):
    raise ValueError(
        f"kappa_labels must be finite and >= 1.0, got {kappa_labels}."
    )

  if m is not None:
    m_declared = float(m)
  elif m_item is not None:
    m_declared = float(m_item)
  else:
    m_declared = _metrics.get_default_m(norm_metric)

  untestable_ref = _refutation.RefutationResult(
      _refutation.Refutation.UNTESTABLE, None, None
  )
  refused_drift = DriftResult(
      outcome=Drift.UNTESTABLE,
      first_half=(float("nan"), float("nan")),
      second_half=(float("nan"), float("nan")),
      first_kappa_check=untestable_ref,
      second_kappa_check=untestable_ref,
      message="no interval",
  )

  # 4. Check for non-finite timestamps
  if ts_arr is not None and not np.all(np.isfinite(ts_arr)):
    invalid_count = int(np.sum(~np.isfinite(ts_arr)))
    return TemporalResult(
        low=float("nan"),
        high=float("nan"),
        level=None,
        status=_interval.Status.ASSUMPTION_REQUIRED,
        kappa_declared=resolved_kappa,
        kappa_check=untestable_ref,
        m_declared=m_declared,
        m_observed=float("nan"),
        num_windows=num_windows,
        mean_window_size=float(n) / float(num_windows),
        effective_n=float(num_windows) / resolved_kappa,
        diagnostic=None,
        message=(
            f"Refusal: {invalid_count} invalid non-finite value(s) in"
            " timestamps."
        ),
        drift=refused_drift,
        count_energy=None,
        count_h=None,
        r2_detail=None,
    )

  # 5. Extract summands / check invalid values
  if norm_metric == "r2":
    assert y_true is not None and y_pred is not None
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
      invalid_count = int(
          np.sum(~np.isfinite(y_true)) + np.sum(~np.isfinite(y_pred))
      )
      return TemporalResult(
          low=float("nan"),
          high=float("nan"),
          level=None,
          status=_interval.Status.ASSUMPTION_REQUIRED,
          kappa_declared=resolved_kappa,
          kappa_check=untestable_ref,
          m_declared=m_declared,
          m_observed=float("nan"),
          num_windows=num_windows,
          mean_window_size=float(n) / float(num_windows),
          effective_n=float(num_windows) / resolved_kappa,
          diagnostic=None,
          message=f"Refusal: {invalid_count} invalid value(s) in input data.",
          drift=refused_drift,
          count_energy=None,
          count_h=None,
          r2_detail=None,
      )
    if ts_arr is not None:
      order = np.argsort(ts_arr, kind="stable")
      y_true_ord = y_true[order]
      y_pred_ord = y_pred[order]
    else:
      y_true_ord = y_true
      y_pred_ord = y_pred

    indices = np.arange(n, dtype=np.int64)
    window_ids = (indices * num_windows) // n
    shard = _cluster_sketch.R2ClusterShard.from_data(
        y_true_ord, y_pred_ord, window_ids
    )
  else:
    assert data_arr is not None
    if ts_arr is not None:
      order = np.argsort(ts_arr, kind="stable")
      ordered_data = data_arr[order]
    else:
      ordered_data = data_arr

    summands = _metrics.extract_summands(ordered_data, norm_metric)
    invalid_count = int(np.sum(np.isnan(summands)))
    if invalid_count > 0:
      return TemporalResult(
          low=float("nan"),
          high=float("nan"),
          level=None,
          status=_interval.Status.ASSUMPTION_REQUIRED,
          kappa_declared=resolved_kappa,
          kappa_check=untestable_ref,
          m_declared=m_declared,
          m_observed=float("nan"),
          num_windows=num_windows,
          mean_window_size=float(n) / float(num_windows),
          effective_n=float(num_windows) / resolved_kappa,
          diagnostic=None,
          message=f"Refusal: {invalid_count} invalid value(s) in input data.",
          drift=refused_drift,
          count_energy=None,
          count_h=None,
          r2_detail=None,
      )

    indices = np.arange(n, dtype=np.int64)
    window_ids = (indices * num_windows) // n
    shard = _cluster_sketch.PartialClusterShard.from_data(
        ordered_data, window_ids, norm_metric
    )

  return _temporal_interval_core(
      shard,
      metric=norm_metric,
      level=level,
      m=m,
      m_item=m_item,
      kappa=resolved_kappa,
      c_target=c_target,
      m_labels=m_labels,
      kappa_labels=kappa_labels,
  )


def temporal_interval_from_buckets(
    counts: Any,
    totals: Any,
    *,
    num_windows: int,
    metric: str = "mse",
    level: float = 0.95,
    m: float | None = None,
    m_item: float | None = None,
    kappa: float = 1.0,
    c_target: float = 0.2,
    m_labels: float | None = None,
    kappa_labels: float | None = None,
) -> TemporalResult:
  """Computes certified temporal confidence interval from time-bucketed data.

  Buckets arrive in time order. Zero-count buckets are dropped. Non-empty
  buckets are assigned to windows using the mid-count assignment rule:
    w_k = (G * (2 * C_{k-1} + counts[k])) // (2 * n)
  which ensures every window is non-empty and has size within b_max of m_bar
  (Lemma W), provided m_bar >= b_max.

  Args:
    counts: 1-D array-like of bucket counts (counts[k] >= 0).
    totals: For scalar metrics, 1-D array-like of bucket sums of summands. For
      metric='r2', (K, 3) array-like of (sum_e2, sum_y, sum_y2).
    num_windows: Number of contiguous windows G >= 2.
    metric: Metric name ('mae', 'mse', 'rmse', 'accuracy', 'r2').
    level: Confidence level in (0, 1). Default is 0.95.
    m: Declared relative variance bound M_c for window totals, used unchanged.
      Mutually exclusive with m_item. If both m and m_item are None, defaults to
      the sketch's default M_c (or 16.0 converted for r2).
    m_item: Item-level bound M (e.g. from declare.estimate_m); M_c is derived
      with Lemma I' from the actual cluster sizes. Mutually exclusive with m.
    kappa: Declared variance inflation factor kappa (>= 1.0). Default is 1.0.
    c_target: Target relative half-width factor in (0, 1). Default is 0.2.
    m_labels: Declared item-level relative variance bound for labels (r2 only),
      converted with Lemma I'. If None, equals converted m_item if m_item given,
      equals unconverted m if m given, or default 16.0 converted.
    kappa_labels: Declared kappa for labels (r2 only).

  Returns:
    TemporalResult containing interval endpoints, diagnostic status, and
    refutation outcomes.
  """
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")
  if not (0.0 < c_target < 1.0):
    raise ValueError(f"c_target must be in (0, 1), got {c_target}.")
  if m is not None and m_item is not None:
    raise ValueError("Cannot pass both 'm' and 'm_item'; specify only one.")
  if m is not None and not (math.isfinite(m) and m >= 1.0):
    raise ValueError(f"m must be finite and >= 1.0, got {m}.")
  if m_item is not None and not (math.isfinite(m_item) and m_item >= 1.0):
    raise ValueError(f"m_item must be finite and >= 1.0, got {m_item}.")
  if not (math.isfinite(kappa) and kappa >= 1.0):
    raise ValueError(f"kappa must be finite and >= 1.0, got {kappa}.")
  if m_labels is not None and not (math.isfinite(m_labels) and m_labels >= 1.0):
    raise ValueError(f"m_labels must be finite and >= 1.0, got {m_labels}.")
  if kappa_labels is not None and not (
      math.isfinite(kappa_labels) and kappa_labels >= 1.0
  ):
    raise ValueError(
        f"kappa_labels must be finite and >= 1.0, got {kappa_labels}."
    )

  if isinstance(num_windows, bool) or not isinstance(
      num_windows, (int, np.integer)
  ):
    raise ValueError(
        f"num_windows must be an integer, got {type(num_windows).__name__}."
    )
  num_windows = int(num_windows)
  if num_windows < 2:
    raise ValueError(f"num_windows must be >= 2, got {num_windows}.")

  counts_arr = np.asarray(counts, dtype=np.int64)
  if counts_arr.ndim != 1:
    raise ValueError(f"counts must be 1-D, got shape {counts_arr.shape}.")
  if np.any(counts_arr < 0):
    raise ValueError("counts must be non-negative integers.")
  k_buckets = int(counts_arr.size)

  nz_mask = counts_arr > 0
  counts_nz = counts_arr[nz_mask]
  if counts_nz.size == 0:
    raise ValueError("All bucket counts are zero.")

  n = int(np.sum(counts_nz))
  if num_windows > n:
    raise ValueError(
        f"num_windows ({num_windows}) must be <= total count n ({n})."
    )
  b_max = int(np.max(counts_nz))
  m_bar = float(n) / float(num_windows)
  # Lemma W claim 3 needs m_bar >= b_max, so refuse only when m_bar < b_max;
  # m_bar == b_max covers K = G equal buckets (one bucket per window).
  if m_bar < b_max:
    raise ValueError(
        f"m_bar ({m_bar:.4g}) < b_max ({b_max}): some window could be empty;"
        f" suggest fewer windows (num_windows <= {n // b_max}) or finer"
        " buckets."
    )

  if metric.strip().lower() == "r2":
    norm_metric = "r2"
    totals_arr = np.asarray(totals, dtype=np.float64)
    if totals_arr.ndim != 2 or totals_arr.shape != (k_buckets, 3):
      raise ValueError(
          "For metric='r2', totals must be a (K, 3) array of (sum_e2, sum_y,"
          f" sum_y2), got shape {totals_arr.shape}."
      )
    totals_nz = totals_arr[nz_mask]
  else:
    norm_metric = _metrics.normalize_metric(metric)
    totals_arr = np.asarray(totals, dtype=np.float64)
    if totals_arr.ndim != 1 or totals_arr.shape[0] != k_buckets:
      raise ValueError(
          f"For metric '{metric}', totals must be a 1-D array of length K"
          f" ({k_buckets}), got shape {totals_arr.shape}."
      )
    totals_nz = totals_arr[nz_mask]

  # Refusal check for non-finite totals
  if not np.all(np.isfinite(totals_nz)):
    untestable_ref = _refutation.RefutationResult(
        _refutation.Refutation.UNTESTABLE, None, None
    )
    refused_drift = DriftResult(
        outcome=Drift.UNTESTABLE,
        first_half=(float("nan"), float("nan")),
        second_half=(float("nan"), float("nan")),
        first_kappa_check=untestable_ref,
        second_kappa_check=untestable_ref,
        message="no interval",
    )
    if m is not None:
      m_declared = float(m)
    elif m_item is not None:
      m_declared = float(m_item)
    else:
      m_declared = _metrics.get_default_m(norm_metric)
    return TemporalResult(
        low=float("nan"),
        high=float("nan"),
        level=None,
        status=_interval.Status.ASSUMPTION_REQUIRED,
        kappa_declared=float(kappa),
        kappa_check=untestable_ref,
        m_declared=m_declared,
        m_observed=float("nan"),
        num_windows=num_windows,
        mean_window_size=m_bar,
        effective_n=float(num_windows) / float(kappa),
        diagnostic=None,
        message="Refusal: non-finite value(s) in totals.",
        drift=refused_drift,
        count_energy=None,
        count_h=None,
        r2_detail=None,
        n_max_over_m_bar=None,
        b_max=b_max,
    )

  # Lemma W mid-count window assignment rule:
  # w_k = (G * (2 * C_{k-1} + counts[k])) // (2 * n)
  cum_counts = np.cumsum(counts_nz)
  c_prev = np.empty_like(counts_nz)
  c_prev[0] = 0
  c_prev[1:] = cum_counts[:-1]
  w_k = (num_windows * (2 * c_prev + counts_nz)) // (2 * n)

  if norm_metric == "r2":
    w_counts = np.bincount(
        w_k, weights=counts_nz, minlength=num_windows
    ).astype(np.int64)
    w_e2 = np.bincount(
        w_k, weights=totals_nz[:, 0], minlength=num_windows
    ).astype(np.float64)
    w_y = np.bincount(
        w_k, weights=totals_nz[:, 1], minlength=num_windows
    ).astype(np.float64)
    w_y2 = np.bincount(
        w_k, weights=totals_nz[:, 2], minlength=num_windows
    ).astype(np.float64)
    shard = _cluster_sketch.R2ClusterShard(
        cluster_ids=np.arange(num_windows, dtype=np.int64),
        e2_totals=w_e2,
        y_totals=w_y,
        y2_totals=w_y2,
        cluster_counts=w_counts,
    )
  else:
    w_counts = np.bincount(
        w_k, weights=counts_nz, minlength=num_windows
    ).astype(np.int64)
    w_totals = np.bincount(
        w_k, weights=totals_nz, minlength=num_windows
    ).astype(np.float64)
    shard = _cluster_sketch.PartialClusterShard(
        cluster_ids=np.arange(num_windows, dtype=np.int64),
        cluster_totals=w_totals,
        cluster_counts=w_counts,
    )

  return _temporal_interval_core(
      shard,
      metric=norm_metric,
      level=level,
      m=m,
      m_item=m_item,
      kappa=float(kappa),
      c_target=c_target,
      m_labels=m_labels,
      kappa_labels=kappa_labels,
      use_path_edges_for_k1=True,
      b_max=b_max,
  )
