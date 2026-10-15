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

"""Batch-means confidence intervals over cluster totals.

Instead of declaring an `effective_n` for the individual losses, this module
aggregates them into cluster totals and treats *those* as the observations
(`DESIGN.md` §4).

Why this and not an item-level `effective_n`:
  Both give algebraically identical Cantelli widths (`DESIGN.md` §4), so
  aggregating into clusters buys no tightness. What it buys is that the
  *within*-cluster dependence is absorbed into the relative second moment of the
  totals, where `M_hat` can refute a bad declaration. Only the *cross*-cluster
  correlation remains an unchecked assertion, and that is a strictly smaller
  surface of faith.

What is still asserted and not checked here:
  `kappa_cluster`, the declared variance inflation factor of the cluster
  totals: Var(sum_c S_c) <= kappa_cluster * sum_c Var(S_c). A bound on the
  maximum absolute row sum of their correlation matrix is a sufficient
  condition. Passing 1.0 asserts the totals are mutually uncorrelated. The
  temporal and graph entry points run refutation checks on it; this function
  does not. Understating `kappa_cluster` under-covers silently.
"""

from collections.abc import Sequence
import dataclasses
import enum
from fractions import Fraction
import math
from typing import Any

import numpy as np

from dgf.src.stats import cluster_sketch
from dgf.src.stats import refutation as _refutation
from dgf.src.stats.independent import interval as _interval
from dgf.src.stats.independent import metrics as _metrics
from dgf.src.stats.independent import sketch as _sketch

__all__ = [
    "ClusterResult",
    "Diagnosis",
    "DiagnosticPayload",
    "R2Detail",
    "aggregate",
    "cantelli_c",
    "cluster_interval",
    "compute_r2_interval",
    "is_mixing_saturated",
]


# Metrics whose summand is a plain non-negative per-item loss whose mean is the
# estimand. `rmse` is handled by estimating `mse` and taking a square root at the
# end (Lemma D), so it is admissible here too. `r2` is not: it is a ratio of two
# separately-estimated means with different dependence structures, and
# independent bounds already refuse it under `effective_n`.
_SUPPORTED = frozenset({"mae", "mse", "rmse", "accuracy"})


class Diagnosis(enum.Enum):
  """Structured classification of the interval status and width feasibility."""

  CERTIFIED_USEFUL = "certified_useful"
  CERTIFIED_WIDE = "certified_wide"
  RESOLVABLE_SAMPLE_SIZE = "resolvable_sample_size"
  SATURATED_GRAPH_MIXING = "saturated_graph_mixing"
  TAIL_UNRESOLVED_REFUTED = "tail_unresolved_refuted"


@dataclasses.dataclass(frozen=True)
class DiagnosticPayload:
  """Actionable diagnostics payload accompanying every cluster_interval result.

  Attributes:
    diagnosis: High-level classification of the statistical feasibility.
    c_up: Computed Cantelli relative half-width factor sqrt(n_A / nu).
    c_target: Caller's desired target relative half-width factor in (0, 1).
    min_effective_n_exist: Minimum effective cluster count required for
      existence n_A(M_d, alpha) = (M_d - 1)(1 - alpha/2)/(alpha/2).
    target_effective_n: Effective cluster count required to reach c_target:
      nu_target = n_A / (c_target ** 2).
    max_supported_alpha: Tightest nominal alpha supportable by the current nu:
      min(1.0, 2(M_d - 1) / (nu + M_d - 1)).
    required_clusters_exist: Required number of clusters to satisfy existence,
      or None if graph mixing is saturated.
    required_items_exist: Required item count to satisfy existence, or None if
      saturated.
    required_clusters_target: Required number of clusters to achieve c_target,
      or None if saturated.
    required_items_target: Required item count to achieve c_target, or None if
      saturated.
    message: Human-readable diagnostic description and recommended action.
  """

  diagnosis: Diagnosis
  c_up: float
  c_target: float
  min_effective_n_exist: float
  target_effective_n: float
  max_supported_alpha: float
  required_clusters_exist: int | None
  required_items_exist: int | None
  required_clusters_target: int | None
  required_items_target: int | None
  message: str


@dataclasses.dataclass(frozen=True)
class ClusterResult:
  """Interval on the per-item mean loss, computed from cluster totals.

  Attributes:
    low: Lower confidence bound on the per-item estimand.
    high: Upper confidence bound on the per-item estimand.
    level: Confidence level, or None when no interval was issued.
    status: Diagnostic status, forwarded from the underlying independent result.
    m_declared: Relative variance bound declared for the cluster totals. For
      R^2, this is the error side M_c_A; the label side bound M_c_q is in
      r2_detail.
    m_observed: Empirical relative variance of the cluster totals. For R^2, this
      is the error side m_hat_e2; the label side m_hat_q is in r2_detail (so
      status can be TAIL_UNRESOLVED while m_observed <= m_declared).
    num_clusters: Number of clusters G, i.e. the number of observations the
      bound actually saw.
    effective_n: The `G / kappa_cluster` passed to the independent bound.
    mean_cluster_size: n / G, the divisor used to rescale back to per-item.
    diagnostic: Structured actionable diagnostic payload.
  """

  low: float
  high: float
  level: float | None
  status: _interval.Status
  m_declared: float
  m_observed: float
  num_clusters: int
  effective_n: float
  mean_cluster_size: float
  diagnostic: DiagnosticPayload


def aggregate(
    summands: Sequence[float] | np.ndarray,
    cluster_ids: Sequence[int] | Sequence[str] | Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, int]:
  """Sums per-item summands within each cluster.

  Args:
    summands: Non-negative per-item losses, length n.
    cluster_ids: Cluster label per item, length n. Labels need not be contiguous
      or sorted; only the induced partition matters.

  Returns:
    (totals, n): the per-cluster totals in order of first appearance, and the
    original item count.

  Raises:
    ValueError: If the inputs disagree in length or are empty.
  """
  s = np.asarray(summands, dtype=np.float64)
  ids = np.asarray(cluster_ids)
  if s.ndim != 1:
    raise ValueError(f"summands must be 1-D, got shape {s.shape}.")
  if s.shape != ids.shape:
    raise ValueError(
        f"summands and cluster_ids must have equal length, got {s.shape} "
        f"and {ids.shape}."
    )
  if s.size == 0:
    raise ValueError("summands must be non-empty.")
  _, dense = np.unique(ids, return_inverse=True)
  totals = np.bincount(dense, weights=s).astype(np.float64)
  return totals, int(s.size)


def is_mixing_saturated(
    num_clusters: int,
    kappa: float,
    *,
    max_achievable_effective_n: float | None = None,
    n_A: float | None = None,
) -> bool:
  """Checks whether effective cluster count is bounded below requirements by mixing.

  Evaluates whether the topology-constrained maximum achievable effective
  sample size is insufficient to satisfy the sample size requirements under
  declared M, or whether G >= 4 and kappa >= G / 2.

  Args:
    num_clusters: Number of clusters G.
    kappa: Declared variance inflation factor kappa.
    max_achievable_effective_n: Optional upper bound on achievable effective
      cluster count from graph mixing or topology.
    n_A: Optional minimum required sample size under declared M and alpha.

  Returns:
    True if mixing saturation prevents meeting sample size requirements, or
    if G >= 4 and kappa >= G / 2; False otherwise.
  """
  return bool(
      (
          max_achievable_effective_n is not None
          and n_A is not None
          and max_achievable_effective_n <= n_A
      )
      or (num_clusters >= 4 and kappa >= 0.5 * float(num_clusters))
  )


def cluster_interval(
    data: Sequence[float] | np.ndarray | cluster_sketch.ClusterSketch,
    cluster_ids: (
        Sequence[int] | Sequence[str] | Sequence[Any] | np.ndarray | None
    ) = None,
    metric: str = "mse",
    level: float = 0.95,
    m: float | None = None,
    kappa_cluster: float | None = None,
    c_target: float = 0.2,
    max_achievable_effective_n: float | None = None,
    m_item: float | None = None,
    kappa: float | None = None,
) -> ClusterResult:
  """Interval on the per-item mean loss via Cantelli on cluster totals.

  The estimand is the pooled per-item mean `theta = (1/n) * sum_i E[s_i]`, i.e.
  the mean over *all* items, not over any topologically selected subset
  (see `DESIGN.md`, "Design choices").

  Rescaling is exact even for unequal cluster sizes: the pooled cluster mean is
  `(1/G) * sum_c E[S_c] = (n/G) * theta`, so dividing both endpoints by `n/G`
  recovers the per-item estimand. Division by a positive constant is monotone,
  so Lemma D applies and the coverage statement is unchanged.

  Args:
    data: Either a `ClusterSketch` instance or per-item *residuals* `e =
      prediction - label` (or 0/1 indicator for accuracy).
    cluster_ids: Cluster label per item (must be None if data is a
      ClusterSketch).
    metric: One of 'mae', 'mse', 'rmse', 'accuracy'.
    level: Confidence level in (0, 1).
    m: Declared relative variance bound on cluster totals M_c, used unchanged.
      Mutually exclusive with m_item. If both m and m_item are None, uses
      sk.default_m_c(get_default_m(metric)).
    kappa_cluster: Declared variance inflation factor of the cluster totals,
      Var(sum_c S_c) <= kappa_cluster * sum_c Var(S_c) (>= 1.0); a correlation
      row-sum bound is a sufficient condition. If None, defaults to
      data.kappa_cluster when data is a ClusterSketch, or 1.0 when data is an
      array.
    c_target: Target Cantelli relative half-width factor in (0, 1).
    max_achievable_effective_n: Optional upper bound on the effective cluster
      count imposed by graph mixing/topology.
    m_item: Item-level bound M (e.g. from declare.estimate_m); M_c is derived
      with Lemma I' from the actual cluster sizes. Mutually exclusive with m.
    kappa: Alias for kappa_cluster.

  Returns:
    A ClusterResult on the per-item scale with an attached DiagnosticPayload.

  Raises:
    ValueError: On invalid inputs, unsupported metrics, or illegal parameter
    values.
  """
  if m is not None and m_item is not None:
    raise ValueError("Cannot pass both 'm' and 'm_item'; specify only one.")
  if kappa_cluster is not None and kappa is not None:
    raise ValueError(
        "Cannot pass both 'kappa_cluster' and 'kappa'; specify only one."
    )
  k_param = kappa_cluster if kappa_cluster is not None else kappa

  norm_metric = _metrics.normalize_metric(metric)
  if norm_metric not in _SUPPORTED:
    raise ValueError(
        f"cluster_interval does not support metric {norm_metric!r}; supported: "
        f"{sorted(_SUPPORTED)}. r2 is excluded because its two summand "
        "sequences have different dependence structures and the pairing itself "
        "changes them (see DESIGN.md section 9)."
    )

  if not (math.isfinite(c_target) and 0.0 < c_target < 1.0):
    raise ValueError(f"c_target must be in (0, 1), got {c_target}.")

  if max_achievable_effective_n is not None and not (
      math.isfinite(max_achievable_effective_n)
      and max_achievable_effective_n > 0.0
  ):
    raise ValueError(
        "max_achievable_effective_n must be > 0, got"
        f" {max_achievable_effective_n}."
    )

  is_sketch = isinstance(data, cluster_sketch.ClusterSketch)
  if is_sketch:
    if cluster_ids is not None:
      raise ValueError("cluster_ids must be None when passing a ClusterSketch.")
    resolved_kappa = data.kappa_cluster if k_param is None else float(k_param)
    if not (math.isfinite(resolved_kappa) and resolved_kappa >= 1.0):
      raise ValueError(f"kappa_cluster must be >= 1.0, got {resolved_kappa}.")
    if resolved_kappa != data.kappa_cluster:
      sk_cluster = dataclasses.replace(data, kappa_cluster=resolved_kappa)
    else:
      sk_cluster = data
  else:
    if cluster_ids is None:
      raise ValueError("cluster_ids is required when data is an array.")
    resolved_kappa = 1.0 if k_param is None else float(k_param)
    if not (math.isfinite(resolved_kappa) and resolved_kappa >= 1.0):
      raise ValueError(f"kappa_cluster must be >= 1.0, got {resolved_kappa}.")
    sk_cluster = cluster_sketch.ClusterSketch.from_data(
        data, cluster_ids, norm_metric, kappa_cluster=resolved_kappa
    )

  num_clusters = sk_cluster.num_clusters
  nu = sk_cluster.effective_n
  mean_cluster_size = sk_cluster.mean_cluster_size
  if m is not None:
    m_val = float(m)
    if not (math.isfinite(m_val) and m_val >= 1.0):
      raise ValueError(f"m must be finite and >= 1.0, got {m_val}.")
    m_declared = m_val
  elif m_item is not None:
    m_item_val = float(m_item)
    if not (math.isfinite(m_item_val) and m_item_val >= 1.0):
      raise ValueError(f"m_item must be finite and >= 1.0, got {m_item_val}.")
    m_declared = sk_cluster.default_m_c(m_item_val)
  else:
    m_declared = sk_cluster.default_m_c(_metrics.get_default_m(norm_metric))

  # Check nu < 1.0 gracefully: do not crash IndependentSketch validation
  if nu < 1.0:
    status = _interval.Status.ASSUMPTION_REQUIRED
    low = float("nan")
    high = float("nan")
    level_out = None
    m_observed = (
        float((num_clusters * sk_cluster.sum_s2) / (sk_cluster.sum_s**2))
        if sk_cluster.sum_s > 0
        else float("nan")
    )
  else:
    ind_sk = sk_cluster.to_independent_sketch()
    res = _interval.confidence_interval(
        ind_sk, metric="mse", level=level, m=m_declared
    )
    status = res.status
    m_observed = res.m_observed
    level_out = res.level

    low = res.low / mean_cluster_size
    high = res.high / mean_cluster_size

    if norm_metric == "rmse":
      low = math.sqrt(max(0.0, low))
      high = math.sqrt(max(0.0, high))
    theta_max = _metrics.get_theta_max(norm_metric)
    if theta_max is not None:
      high = min(high, theta_max)

  # DiagnosticPayload calculation
  alpha = 1.0 - level
  alpha_prime = alpha / 2.0
  n_A = (m_declared - 1.0) * (1.0 - alpha_prime) / alpha_prime
  nu_target = n_A / (c_target**2)
  c_up = math.sqrt(n_A / nu) if nu > 0 else float("inf")
  max_supported_alpha = min(
      1.0, max(0.0, 2.0 * (m_declared - 1.0) / (nu + m_declared - 1.0))
  )

  is_saturated = is_mixing_saturated(
      num_clusters,
      resolved_kappa,
      max_achievable_effective_n=max_achievable_effective_n,
      n_A=n_A,
  )

  if is_saturated:
    req_clusters_exist = None
    req_items_exist = None
    req_clusters_target = None
    req_items_target = None
  else:
    if status is _interval.Status.TAIL_UNRESOLVED:
      m_req = max(m_declared, m_observed)
      n_A_obs = (m_req - 1.0) * (1.0 - alpha_prime) / alpha_prime
      req_clusters_exist = math.ceil(resolved_kappa * n_A_obs) + 1
      req_items_exist = math.ceil(req_clusters_exist * mean_cluster_size)
      req_clusters_target = (
          math.ceil(resolved_kappa * n_A_obs / (c_target**2)) + 1
      )
      req_items_target = math.ceil(req_clusters_target * mean_cluster_size)
    else:
      req_clusters_exist = math.ceil(resolved_kappa * n_A) + 1
      req_items_exist = math.ceil(req_clusters_exist * mean_cluster_size)
      req_clusters_target = math.ceil(resolved_kappa * nu_target) + 1
      req_items_target = math.ceil(req_clusters_target * mean_cluster_size)

  if status is _interval.Status.ASSUMPTION_REQUIRED:
    if is_saturated:
      diagnosis = Diagnosis.SATURATED_GRAPH_MIXING
      msg = (
          "Refused (saturated graph mixing):"
          f" kappa_cluster={resolved_kappa:.1f} limits effective_n to {nu:.2f}"
          f" <= required n_A={n_A:.2f}."
      )
    else:
      diagnosis = Diagnosis.RESOLVABLE_SAMPLE_SIZE
      msg = (
          f"Refused (insufficient sample size): effective_n={nu:.2f} <"
          f" n_A={n_A:.2f}; requires {req_clusters_exist} clusters (or alpha >="
          f" {max_supported_alpha:.4f})."
      )
  elif status is _interval.Status.TAIL_UNRESOLVED:
    diagnosis = Diagnosis.TAIL_UNRESOLVED_REFUTED
    msg = (
        f"Refuted: observed relative variance m_observed={m_observed:.2f} >"
        f" declared m={m_declared:.2f}; requires declared m >= {m_observed:.2f}"
        f" and {req_clusters_exist} clusters."
    )
  else:  # UNREFUTED
    if c_up <= c_target:
      diagnosis = Diagnosis.CERTIFIED_USEFUL
      msg = (
          f"Certified: relative half-width factor c_up={c_up:.4f} <="
          f" c_target={c_target:.4f}."
      )
    else:
      diagnosis = Diagnosis.CERTIFIED_WIDE
      msg = (
          f"Certified wide: c_up={c_up:.4f} > c_target={c_target:.4f}; "
          f"requires {req_clusters_target} clusters for c_target."
      )

  diagnostic = DiagnosticPayload(
      diagnosis=diagnosis,
      c_up=c_up,
      c_target=c_target,
      min_effective_n_exist=n_A,
      target_effective_n=nu_target,
      max_supported_alpha=max_supported_alpha,
      required_clusters_exist=req_clusters_exist,
      required_items_exist=req_items_exist,
      required_clusters_target=req_clusters_target,
      required_items_target=req_items_target,
      message=msg,
  )

  return ClusterResult(
      low=low,
      high=high,
      level=level_out,
      status=status,
      m_declared=m_declared,
      m_observed=m_observed,
      num_clusters=num_clusters,
      effective_n=nu,
      mean_cluster_size=mean_cluster_size,
      diagnostic=diagnostic,
  )


def cantelli_c(alpha_prime: float, M_c: float, kappa: float, G: int) -> float:
  """Computes the one-sided Cantelli constant c(alpha', M_c, kappa, G).

  c(alpha', M_c, kappa, G) = sqrt((M_c - 1) * kappa * (1 - alpha') / (alpha' *
  G)).
  """
  if alpha_prime <= 0.0 or alpha_prime >= 1.0 or G <= 0:
    return float("inf")
  num = (M_c - 1.0) * kappa * (1.0 - alpha_prime)
  den = alpha_prime * float(G)
  return math.sqrt(max(0.0, num / den))


@dataclasses.dataclass(frozen=True)
class R2Detail:
  """Detailed components of the Claim 7 R^2 confidence interval.

  Attributes:
    L_A: Lower confidence bound on MSE (theta_A).
    U_A: Upper confidence bound on MSE (theta_A).
    L_B: Lower confidence bound on total variance Var(y) (theta_B).
    U_B: Upper confidence bound on total variance Var(y) (theta_B), or +inf.
    D: Denominator margin factor 1 - c_B_upper - chebyshev_term for upper
      variance bound.
    v_hat: Empirical plug-in variance V_hat.
    c_A: Cantelli constant for MSE at alpha/4.
    c_B_lower: Cantelli constant for Q at alpha/4.
    c_B_upper: Cantelli constant for Q at alpha/8.
    chebyshev_term: Chebyshev bound term kappa_labels * n_max / (n * alpha/8).
    e2_kappa_check: Refutation result on E2_c totals.
    q_kappa_check: Refutation result on Q_hat_c totals.
    y_kappa_check: Refutation result on Y_c totals.
    M_c_A: Declared cluster-level bound on relative variance of squared errors.
    m_hat_e2: Empirical relative variance of cluster squared error totals.
    M_c_q: Declared cluster-level bound on relative variance of squared centred
      label totals Q_c.
    m_hat_q: Empirical relative variance of squared centred label totals Q_c.
  """

  L_A: float
  U_A: float
  L_B: float
  U_B: float
  D: float
  v_hat: float
  c_A: float
  c_B_lower: float
  c_B_upper: float
  chebyshev_term: float
  e2_kappa_check: _refutation.RefutationResult
  q_kappa_check: _refutation.RefutationResult
  y_kappa_check: _refutation.RefutationResult
  M_c_A: float
  m_hat_e2: float
  M_c_q: float
  m_hat_q: float


def compute_r2_interval(
    e2_totals: np.ndarray,
    y_totals: np.ndarray,
    y2_totals: np.ndarray,
    cluster_counts: np.ndarray,
    *,
    level: float = 0.95,
    m: float | None = None,
    m_item: float | None = None,
    m_labels: float | None = None,
    kappa: float = 1.0,
    kappa_labels: float | None = None,
    run_kappa_check_fn: Any,
) -> tuple[
    float,
    float,
    float | None,
    _interval.Status,
    R2Detail | None,
    str,
    _refutation.RefutationResult,
]:
  """Computes Claim 7 R^2 confidence interval and refutations from cluster totals.

  Note:
    Callers passing precomputed totals should compute them on labels centred
    by an approximate mean; otherwise relative precision degrades roughly as
    eps * mean^2 / var.

  Args:
    e2_totals: Cluster totals of squared errors.
    y_totals: Cluster totals of true labels.
    y2_totals: Cluster totals of squared true labels.
    cluster_counts: Number of items in each cluster.
    level: Confidence level in (0, 1). Default 0.95.
    m: Cluster-level relative variance bound M_c on squared-error cluster
      totals, used directly with no conversion. Mutually exclusive with m_item.
    m_item: Item-level relative variance bound M on squared errors, converted to
      M_c via Lemma I'. Mutually exclusive with m.
    m_labels: Item-level relative variance bound on labels, converted to M_c,q
      via Lemma I'. If None, equals converted m_item if m_item given, equals
      unconverted m if m given, or default 16.0 converted.
    kappa: Variance inflation factor on squared errors. Default 1.0.
    kappa_labels: Variance inflation factor on labels. If None, defaults to
      kappa.
    run_kappa_check_fn: Callback to perform kappa check refutations.
  """
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")
  alpha = 1.0 - level

  if m is not None and m_item is not None:
    raise ValueError("Cannot pass both 'm' and 'm_item'; specify only one.")

  k_A = float(kappa)
  if not (math.isfinite(k_A) and k_A >= 1.0):
    raise ValueError(f"kappa must be finite and >= 1.0, got {k_A}.")
  k_labels = float(kappa_labels) if kappa_labels is not None else k_A
  if not (math.isfinite(k_labels) and k_labels >= 1.0):
    raise ValueError(f"kappa_labels must be finite and >= 1.0, got {k_labels}.")

  if m is not None:
    m_val = float(m)
    if not (math.isfinite(m_val) and m_val >= 1.0):
      raise ValueError(f"m must be finite and >= 1.0, got {m_val}.")
  else:
    m_val = None

  if m_item is not None:
    m_item_val = float(m_item)
    if not (math.isfinite(m_item_val) and m_item_val >= 1.0):
      raise ValueError(f"m_item must be finite and >= 1.0, got {m_item_val}.")
  else:
    m_item_val = None

  if m_labels is not None:
    m_labels_val = float(m_labels)
    if not (math.isfinite(m_labels_val) and m_labels_val >= 1.0):
      raise ValueError(
          f"m_labels must be finite and >= 1.0, got {m_labels_val}."
      )
  else:
    m_labels_val = None

  untestable_ref = _refutation.RefutationResult(
      _refutation.Refutation.UNTESTABLE, None, None
  )

  # Validate array shapes and values
  e2 = np.asarray(e2_totals, dtype=np.float64)
  y = np.asarray(y_totals, dtype=np.float64)
  y2 = np.asarray(y2_totals, dtype=np.float64)
  counts = np.asarray(cluster_counts, dtype=np.int64)

  if e2.ndim != 1 or y.ndim != 1 or y2.ndim != 1 or counts.ndim != 1:
    raise ValueError("All cluster total arrays must be 1-D.")
  g = int(counts.size)
  if g < 2:
    raise ValueError(f"Number of clusters G must be >= 2, got {g}.")
  if not (e2.size == g and y.size == g and y2.size == g):
    raise ValueError("All cluster total arrays must have the same length G.")
  if np.any(counts < 1):
    raise ValueError("All cluster counts must be >= 1.")

  if not (
      np.all(np.isfinite(e2))
      and np.all(np.isfinite(y))
      and np.all(np.isfinite(y2))
      and np.all(np.isfinite(counts))
  ):
    return (
        float("nan"),
        float("nan"),
        None,
        _interval.Status.ASSUMPTION_REQUIRED,
        None,
        "Refused: non-finite input in cluster totals.",
        untestable_ref,
    )

  n = int(np.sum(counts))
  n_max = int(np.max(counts))

  s_counts_sq = int(np.sum(counts**2))
  if m_val is not None:
    M_c_A = m_val
  elif m_item_val is not None:
    M_c_A = cluster_sketch.default_m_c_from_counts(
        m_item_val, n, g, n_max, s_counts_sq
    )
  else:
    M_c_A = cluster_sketch.default_m_c_from_counts(
        16.0, n, g, n_max, s_counts_sq
    )

  if m_labels_val is not None:
    M_c_q = cluster_sketch.default_m_c_from_counts(
        m_labels_val, n, g, n_max, s_counts_sq
    )
  elif m_item_val is not None:
    M_c_q = M_c_A
  elif m_val is not None:
    M_c_q = m_val
  else:
    M_c_q = cluster_sketch.default_m_c_from_counts(
        16.0, n, g, n_max, s_counts_sq
    )

  alpha_4 = alpha / 4.0
  alpha_8 = alpha / 8.0
  c_A = cantelli_c(alpha_4, M_c_A, k_A, g)
  c_B_lower = cantelli_c(alpha_4, M_c_q, k_A, g)
  c_B_upper = cantelli_c(alpha_8, M_c_q, k_A, g)
  chebyshev_term = (k_labels * float(n_max)) / (float(n) * alpha_8)

  sum_y = float(np.sum(y))
  sum_e2 = float(np.sum(e2))
  y_bar = sum_y / float(n)
  e2_bar = sum_e2 / float(n)

  Q_hat = y2 - 2.0 * y_bar * y + counts.astype(np.float64) * (y_bar**2)
  sum_q = float(np.sum(Q_hat))
  v_hat = max(0.0, sum_q / float(n))

  # Refutation checks on E2_c, Q_hat_c, Y_c
  res_e2 = e2 - counts.astype(np.float64) * e2_bar
  check_e2 = run_kappa_check_fn(res_e2, counts, k_A)

  res_q = Q_hat - counts.astype(np.float64) * (sum_q / float(n))
  check_q = run_kappa_check_fn(res_q, counts, k_A)

  res_y = y - counts.astype(np.float64) * y_bar
  check_y = run_kappa_check_fn(res_y, counts, k_labels)

  checks = (check_e2, check_q, check_y)
  if any(c.outcome is _refutation.Refutation.REFUTED for c in checks):
    overall_outcome = _refutation.Refutation.REFUTED
    ref_stats = [c.statistic for c in checks if c.statistic is not None]
    stat_val = max(ref_stats) if ref_stats else None
    thresh_val = None
    k_msg = "kappa check refuted on at least one R2 sequence."
  elif any(c.outcome is _refutation.Refutation.UNTESTABLE for c in checks):
    overall_outcome = _refutation.Refutation.UNTESTABLE
    stat_val = None
    thresh_val = None
    reasons = [c.reason for c in checks if c.reason]
    k_msg = (
        f"kappa check untestable ({'; '.join(reasons)})"
        if reasons
        else "kappa check untestable."
    )
  else:
    overall_outcome = _refutation.Refutation.NOT_REFUTED
    ref_stats = [c.statistic for c in checks if c.statistic is not None]
    stat_val = max(ref_stats) if ref_stats else None
    thresh_val = None
    k_msg = "kappa check not refuted on any R2 sequence."
  overall_kappa_check = _refutation.RefutationResult(
      overall_outcome, stat_val, thresh_val, reason=k_msg
  )

  # Status from tail checks
  m_hat_e2 = (
      (float(g) * float(np.sum(e2**2))) / (sum_e2**2) if sum_e2 > 0 else 1.0
  )
  m_hat_q = (
      (float(g) * float(np.sum(Q_hat**2))) / (sum_q**2) if sum_q > 0 else 1.0
  )
  if m_hat_e2 > M_c_A or m_hat_q > M_c_q:
    tail_status = _interval.Status.TAIL_UNRESOLVED
  else:
    tail_status = _interval.Status.UNREFUTED

  D = 1.0 - c_B_upper - chebyshev_term

  # Existence condition: c_A >= 1 or c_B_lower >= 1
  if c_A >= 1.0 or c_B_lower >= 1.0:
    refuse_msg = (
        f"Refused (insufficient sample size): c_A={c_A:.2f} or"
        f" c_B_lower={c_B_lower:.2f} >= 1.0."
    )
    detail = R2Detail(
        L_A=float("nan"),
        U_A=float("nan"),
        L_B=float("nan"),
        U_B=float("nan"),
        D=D,
        v_hat=v_hat,
        c_A=c_A,
        c_B_lower=c_B_lower,
        c_B_upper=c_B_upper,
        chebyshev_term=chebyshev_term,
        e2_kappa_check=check_e2,
        q_kappa_check=check_q,
        y_kappa_check=check_y,
        M_c_A=M_c_A,
        m_hat_e2=m_hat_e2,
        M_c_q=M_c_q,
        m_hat_q=m_hat_q,
    )
    return (
        float("nan"),
        float("nan"),
        None,
        _interval.Status.ASSUMPTION_REQUIRED,
        detail,
        refuse_msg,
        overall_kappa_check,
    )

  # Endpoints calculation: compute [L_A, U_A] through Claim 4 MSE path at level 1 - alpha/2
  sk_A = cluster_sketch.ClusterSketch(
      n_items=n,
      num_clusters=g,
      sum_s=sum_e2,
      sum_s2=float(np.sum(e2**2)),
      top_32=tuple(sorted([float(x) for x in e2], reverse=True)[:32]),
      max_cluster_size=n_max,
      sum_count_sq=s_counts_sq,
      kappa_cluster=k_A,
  )
  res_A = cluster_interval(
      sk_A,
      metric="mse",
      level=1.0 - alpha / 2.0,
      m=M_c_A,
      kappa_cluster=k_A,
  )
  L_A = res_A.low
  U_A = res_A.high

  if v_hat == 0.0:
    low = float("-inf")
    high = 1.0
    L_B = 0.0
    U_B = float("inf")
  else:
    L_B = v_hat / (1.0 + c_B_lower)
    U_B = (v_hat / D) if D > 0.0 else float("inf")
    low = 1.0 - (U_A / L_B)
    high = 1.0 if not math.isfinite(U_B) else 1.0 - (L_A / U_B)

  detail = R2Detail(
      L_A=L_A,
      U_A=U_A,
      L_B=L_B,
      U_B=U_B,
      D=D,
      v_hat=v_hat,
      c_A=c_A,
      c_B_lower=c_B_lower,
      c_B_upper=c_B_upper,
      chebyshev_term=chebyshev_term,
      e2_kappa_check=check_e2,
      q_kappa_check=check_q,
      y_kappa_check=check_y,
      M_c_A=M_c_A,
      m_hat_e2=m_hat_e2,
      M_c_q=M_c_q,
      m_hat_q=m_hat_q,
  )
  if tail_status is _interval.Status.TAIL_UNRESOLVED:
    tail_reasons = []
    if m_hat_e2 > M_c_A:
      tail_reasons.append(
          f"squared-error totals m_hat={m_hat_e2:.2f} > M_c={M_c_A:.2f}"
      )
    if m_hat_q > M_c_q:
      tail_reasons.append(
          f"label totals Q_c m_hat={m_hat_q:.2f} > M_c={M_c_q:.2f}"
      )
    out_msg = f"Refuted: {'; '.join(tail_reasons)}. {k_msg}".strip()
  else:
    out_msg = k_msg

  return (low, high, level, tail_status, detail, out_msg, overall_kappa_check)
