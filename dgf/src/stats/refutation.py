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

"""Statistical refutation tests for temporal autocorrelation and variance inflation.

Provides check_uncorrelated and check_kappa primitives to check whether cluster
totals are uncorrelated (kappa = 1) or satisfy an upper bound on variance
inflation (kappa <= kappa_declared).

The public functions are named check_* rather than test_* so that test runners
do not collect them when a test module imports them by name.
"""

import dataclasses
import enum
import math
from typing import Any

import numpy as np
import scipy.stats

__all__ = [
    "Refutation",
    "RefutationResult",
    "check_uncorrelated",
    "kappa_lower_bound",
    "check_kappa",
    "check_uncorrelated_edges",
    "kappa_lower_bound_batches",
    "check_kappa_batches",
]


class Refutation(enum.Enum):
  REFUTED = "refuted"
  NOT_REFUTED = "not_refuted"
  UNTESTABLE = "untestable"


@dataclasses.dataclass(frozen=True)
class RefutationResult:
  """Outcome of a refutation check.

  Attributes:
    outcome: REFUTED, NOT_REFUTED or UNTESTABLE.
    statistic: The test statistic; None when UNTESTABLE.
    threshold: The value the statistic is compared against; None when
      UNTESTABLE.
    reason: Explanation for UNTESTABLE outcome when triggered by a guard; None
      otherwise.
    edge_n_eff: Effective number of edge terms (||y||_1^2 / ||y||_2^2) for the
      edge test; None for other checks or when uncomputed.
  """

  outcome: Refutation
  statistic: float | None
  threshold: float | None
  reason: str | None = None
  edge_n_eff: float | None = None


def _edge_effective_count(y: np.ndarray) -> float:
  """Computes effective number of edge terms (sum |y|)^2 / sum y^2."""
  arr = np.asarray(y, dtype=np.float64)
  if arr.size == 0:
    return 0.0
  sum_sq = float(np.sum(arr**2))
  if sum_sq <= 0.0 or not math.isfinite(sum_sq):
    return 0.0
  sum_abs = float(np.sum(np.abs(arr)))
  return float((sum_abs**2) / sum_sq)


def _compute_lag1_autocorr(y: np.ndarray) -> float | None:
  """Computes sample lag-1 autocorrelation around sample mean."""
  mean_y = float(np.mean(y))
  diff = y - mean_y
  denom = float(np.sum(diff**2))
  if denom <= 0.0 or not np.isfinite(denom):
    return None
  num = float(np.sum(diff[:-1] * diff[1:]))
  return num / denom


def check_uncorrelated(totals: Any, level: float = 0.95) -> RefutationResult:
  """Tests whether consecutive cluster totals have zero lag-1 autocorrelation.

  Computes the one-sided Fisher-z lag-1 test on totals around their sample mean.
  Note that this tests only lag-1 correlation of the totals it is given; it
  cannot detect higher-order autocorrelations, non-linear dependencies, or
  correlations at scales finer than the provided cluster totals.

  Args:
    totals: 1-D array-like of cluster totals.
    level: Confidence level in (0, 1) for the one-sided rejection threshold.
      Default is 0.95 (5% nominal false rejection level).

  Returns:
    RefutationResult with outcome REFUTED, NOT_REFUTED, or UNTESTABLE.
    UNTESTABLE is returned if len(totals) < 5, totals contain non-finite values,
    or totals have zero/non-finite variance.

  Raises:
    ValueError: If totals is not 1-D or level is outside (0, 1).
  """
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")
  arr = np.asarray(totals)
  if arr.ndim != 1:
    raise ValueError(f"totals must be 1-D array, got shape {arr.shape}.")

  g = arr.shape[0]
  if g < 5:
    return RefutationResult(Refutation.UNTESTABLE, None, None)
  if not np.all(np.isfinite(arr)):
    return RefutationResult(Refutation.UNTESTABLE, None, None)

  r1 = _compute_lag1_autocorr(arr)
  if r1 is None:
    return RefutationResult(Refutation.UNTESTABLE, None, None)

  r1_clipped = float(np.clip(r1, -0.9999, 0.9999))
  stat = float(np.arctanh(r1_clipped) * math.sqrt(float(g - 4)))
  thresh = float(scipy.stats.norm.ppf(level))

  outcome = Refutation.REFUTED if stat > thresh else Refutation.NOT_REFUTED
  return RefutationResult(outcome, stat, thresh)


def kappa_lower_bound(
    totals: Any,
    level: float = 0.95,
    num_batches: int = 30,
) -> float | None:
  """Computes a lower confidence bound on the variance inflation factor kappa.

  Uses non-overlapping batch means across num_batches super-batches of size
  m = G // num_batches consecutive totals, dropping any remainder.
  Note that this estimator is biased low when the super-batches are shorter
  than the correlation length of the underlying process.

  Args:
    totals: 1-D array-like of cluster totals.
    level: Confidence level in (0, 1).
    num_batches: Number of super-batches B (must be >= 2).

  Returns:
    Lower bound float, or None if untestable.

  Raises:
    ValueError: If totals is not 1-D, level outside (0, 1), or num_batches < 2.
  """
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")
  if num_batches < 2:
    raise ValueError(f"num_batches must be >= 2, got {num_batches}.")
  arr = np.asarray(totals)
  if arr.ndim != 1:
    raise ValueError(f"totals must be 1-D array, got shape {arr.shape}.")

  g = arr.shape[0]
  m = g // num_batches
  if m < 1:
    return None
  if not np.all(np.isfinite(arr)):
    return None

  s_trunc = arr[: num_batches * m]
  batch_means = np.mean(s_trunc.reshape(num_batches, m), axis=1)
  var_b = float(np.var(batch_means, ddof=1))
  var_g = float(np.var(arr, ddof=1))

  if var_g <= 0.0 or not np.isfinite(var_g) or not np.isfinite(var_b):
    return None

  kappa_hat = float(m * var_b / var_g)
  chi2_thresh = float(scipy.stats.chi2.ppf(level, df=num_batches - 1))
  if chi2_thresh <= 0.0 or not np.isfinite(chi2_thresh):
    return None

  return float((num_batches - 1) * kappa_hat / chi2_thresh)


def check_kappa(
    totals: Any,
    kappa_declared: float,
    level: float = 0.95,
    num_batches: int = 30,
) -> RefutationResult:
  """Tests whether kappa_declared is refuted by batch-means lower confidence bound.

  Refutes when kappa_lower_bound > kappa_declared.
  Note that this test cannot detect variance inflation if the super-batches are
  shorter than the correlation length, as batch-means kappa is biased low in
  that regime.

  Args:
    totals: 1-D array-like of cluster totals.
    kappa_declared: Declared kappa value (must be >= 1.0).
    level: Confidence level in (0, 1).
    num_batches: Number of super-batches B (must be >= 2).

  Returns:
    RefutationResult with outcome REFUTED, NOT_REFUTED, or UNTESTABLE.

  Raises:
    ValueError: On invalid arguments, including a NaN kappa_declared.
  """
  if not kappa_declared >= 1.0:
    raise ValueError(f"kappa_declared must be >= 1.0, got {kappa_declared}.")
  lcb = kappa_lower_bound(totals, level=level, num_batches=num_batches)
  if lcb is None:
    return RefutationResult(Refutation.UNTESTABLE, None, None)

  outcome = (
      Refutation.REFUTED if lcb > kappa_declared else Refutation.NOT_REFUTED
  )
  return RefutationResult(outcome, lcb, float(kappa_declared))


def check_uncorrelated_edges(
    residuals: Any,
    counts: Any,
    edges: Any,
    level: float = 0.95,
    min_effective_edges: float = 10.0,
) -> RefutationResult:
  """Tests whether adjacent cluster errors have zero correlation (self-normalised edge test).

  Null hypothesis:
    Cluster totals are S_c = n_c θ + ε_c with the ε_c independent and mean zero,
    with arbitrary per-cluster variances. The residuals are e_c = S_c − n_c θ̂
    and z_c = e_c / sqrt(n_c).

  Statistic:
    T = (Σ_E z_c z_d − B̂) / sqrt(V′). Because θ is estimated, z = Pζ with
    ζ_c = ε_c / sqrt(n_c) and P the projection orthogonal to
    q = sqrt(n / Σn), so the numerator is the quadratic form ζᵀAζ with
    A = ½PWP (W the cluster adjacency). B̂ is its plug-in null mean and V′ its
    plug-in null variance. Without these corrections the test is biased low
    and over-normalised on dense cluster graphs, where it almost never refutes.

  Asymptotic distribution:
    T is asymptotically standard normal under the null when no small set of
    eigenvalues of the variance-weighted matrix D_tau^{1/2} A D_tau^{1/2}
    dominates (de Jong, 1987). The size is asymptotic, not exact.

  Dominance guard:
    To leading order T <= sqrt(n_eff), where n_eff = (sum_E |y|)^2 / sum_E y^2
    is the effective number of edge terms with y = z_c z_d. Below 10 effective
    edges the normal approximation is unreliable, and skewed residuals can make
    the test refute a true kappa = 1 almost surely, so the test reports
    UNTESTABLE.
    Known limitation: the guard does not cover many edges whose residuals are all
    skewed in the same direction around near-zero-variance clusters.

  One-sided rejection:
    The test is one-sided (positive correlation only), rejecting when T > z_{level},
    since variance inflation kappa > 1 arises from positive edge correlations.

  Args:
    residuals: 1-D float array of cluster residuals e_c of length G.
    counts: 1-D integer array of cluster item counts n_c >= 1 of length G.
    edges: 2-D int array of shape (E, 2) containing cluster indices in [0, G).
    level: Confidence level in (0, 1) for the one-sided threshold (default 0.95).
    min_effective_edges: Minimum effective edge count required to run the test
      (default 10.0). Set to 0.0 to disable the guard.

  Returns:
    RefutationResult with outcome REFUTED, NOT_REFUTED, or UNTESTABLE.
    Returns UNTESTABLE if fewer than 30 edges remain after deduplication and
    dropping self-pairs, or if n_eff < min_effective_edges, or if V' <= 0,
    or if any input is non-finite.

  Raises:
    ValueError: If arrays have incompatible dimensions, indices outside [0, G),
      counts < 1, level outside (0, 1), or min_effective_edges < 0 / non-finite.
  """
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")
  if not (math.isfinite(min_effective_edges) and min_effective_edges >= 0.0):
    raise ValueError(
        f"min_effective_edges must be finite and >= 0, got {min_effective_edges}."
    )

  e = np.asarray(residuals, dtype=np.float64)
  n = np.asarray(counts, dtype=np.int64)
  if e.ndim != 1:
    raise ValueError(f"residuals must be 1-D, got shape {e.shape}.")
  if n.ndim != 1:
    raise ValueError(f"counts must be 1-D, got shape {n.shape}.")
  if e.shape[0] != n.shape[0]:
    raise ValueError(
        f"residuals and counts length mismatch: {e.shape[0]} vs {n.shape[0]}."
    )
  if np.any(n < 1):
    raise ValueError("All counts must be >= 1.")

  g = e.shape[0]
  ed = np.asarray(edges, dtype=np.int64)
  if ed.ndim != 2 or ed.shape[1] != 2:
    raise ValueError(f"edges must have shape (E, 2), got {ed.shape}.")

  if ed.shape[0] == 0:
    return RefutationResult(Refutation.UNTESTABLE, None, None)

  if np.any(ed < 0) or np.any(ed >= g):
    raise ValueError(
        f"Edge cluster indices must be in [0, {g}), got out-of-range index."
    )

  c1 = np.minimum(ed[:, 0], ed[:, 1])
  c2 = np.maximum(ed[:, 0], ed[:, 1])
  non_self = c1 != c2
  c1 = c1[non_self]
  c2 = c2[non_self]
  if c1.size == 0:
    return RefutationResult(Refutation.UNTESTABLE, None, None)

  pair_keys = np.unique(c1 * g + c2)
  if pair_keys.size < 30:
    return RefutationResult(Refutation.UNTESTABLE, None, None)

  if not np.all(np.isfinite(e)) or not np.all(np.isfinite(n)):
    return RefutationResult(Refutation.UNTESTABLE, None, None)

  u = pair_keys // g
  v = pair_keys % g

  z = e / np.sqrt(n.astype(np.float64))
  z_u = z[u]
  z_v = z[v]

  y = z_u * z_v
  if float(np.sum(y**2)) == 0.0:
    return RefutationResult(
        Refutation.UNTESTABLE,
        None,
        None,
        reason="edge statistic has zero variance (all edge products are zero)",
        edge_n_eff=0.0,
    )

  n_eff = _edge_effective_count(y)
  if n_eff < min_effective_edges:
    reason = (
        f"few edges dominate the edge statistic (effective edges {n_eff:.1f} <"
        f" {min_effective_edges:g})"
    )
    return RefutationResult(
        Refutation.UNTESTABLE, None, None, reason=reason, edge_n_eff=n_eff
    )

  # Derivation of the projected quadratic form variance V':
  # Let q_k = sqrt(n_k / N), a unit vector, and zeta_k = eps_k / sqrt(n_k), independent
  # with variances tau_k^2. Then z = P zeta with P = I - q q^T, an orthogonal projection.
  # The numerator is sum_E z_c z_d = zeta^T A zeta, where A = 1/2 P W P and W is the
  # symmetric 0/1 cluster adjacency. The centering bias B_hat is sum_k A_kk z_k^2.
  # The variance of zeta^T A zeta under the null is 2 sum_{k,l} A_kl^2 tau_k^2 tau_l^2,
  # with plug-in tau_k^2 ≈ x_k = z_k^2.
  # With w = W q (w_k = sum_{l~k} q_l), s = q^T W q = 2 sum_E q_c q_d,
  # R_cd = -q_c w_d - w_c q_d + s q_c q_d, and Q_ab = sum_k a_k b_k x_k:
  #   V' = sum_E x_c x_d (1 + 2 R_cd) + 1/2 [ 2 Q_qq Q_ww + s^2 Q_qq^2 + 2 Q_qw^2 - 4 s Q_qq Q_qw ]
  # The fourth-cumulant term sum_k A_kk^2 kappa4_k is omitted on purpose. Its relative
  # size is at most about G / |E| only when the standardised fourth cumulants
  # kappa4_k / tau_k^4 are bounded. That fails for sparse binary losses as
  # theta -> 1, where kappa4_k / tau_k^4 grows like 1 / (n_k (1 - theta)).
  n_total = float(np.sum(n))
  q = np.sqrt(n.astype(np.float64) / n_total)
  w = np.bincount(u, weights=q[v], minlength=g) + np.bincount(
      v, weights=q[u], minlength=g
  )
  s = 2.0 * float(np.sum(q[u] * q[v]))

  q_u = q[u]
  q_v = q[v]
  w_u = w[u]
  w_v = w[v]
  r_uv = -q_u * w_v - w_u * q_v + s * q_u * q_v

  x = z**2
  x_u = x[u]
  x_v = x[v]
  term1 = float(np.sum(x_u * x_v * (1.0 + 2.0 * r_uv)))

  q_qq = float(np.sum((q**2) * x))
  q_ww = float(np.sum((w**2) * x))
  q_qw = float(np.sum(q * w * x))
  term2 = 0.5 * (
      2.0 * q_qq * q_ww
      + (s**2) * (q_qq**2)
      + 2.0 * (q_qw**2)
      - 4.0 * s * q_qq * q_qw
  )
  v_prime = term1 + term2

  # B_hat = sum_k A_kk z_k^2 in e-scale coordinates
  q_sum = float(np.sum(e**2))
  n_u = n[u].astype(np.float64)
  n_v = n[v].astype(np.float64)
  e_u = e[u]
  e_v = e[v]
  b_term = (
      (n_u * n_v * q_sum / (n_total**2))
      - (n_u * (e_v**2) + n_v * (e_u**2)) / n_total
  ) / np.sqrt(n_u * n_v)
  b_hat = float(np.sum(b_term))

  n_stat = float(np.sum(z_u * z_v))

  if (
      v_prime <= 0.0
      or not math.isfinite(v_prime)
      or not math.isfinite(n_stat)
      or not math.isfinite(b_hat)
  ):
    return RefutationResult(Refutation.UNTESTABLE, None, None, edge_n_eff=n_eff)

  t_stat = (n_stat - b_hat) / math.sqrt(v_prime)
  if not math.isfinite(t_stat):
    return RefutationResult(Refutation.UNTESTABLE, None, None, edge_n_eff=n_eff)

  threshold = float(scipy.stats.norm.ppf(level))
  outcome = Refutation.REFUTED if t_stat > threshold else Refutation.NOT_REFUTED
  return RefutationResult(outcome, t_stat, threshold, edge_n_eff=n_eff)


def kappa_lower_bound_batches(
    residuals: Any,
    counts: Any,
    batch_ids: Any,
    level: float = 0.95,
) -> float | None:
  """Computes a lower confidence bound on kappa via super-batch means.

  Theory:
    To first order, under dependence only between adjacent clusters,
    non-negative edge covariances, balanced super-batches and Var(S_c)
    proportional to n_c, the count-normalised estimator kappa_hat_B is biased
    low by the edges cut between super-batches (Proposition B). Refuting when
    the lower confidence bound exceeds kappa_declared is then conservative.
    Outside these conditions this is approximate. The chi^2 pivot step is
    approximate, asymptotically valid under weak dependence.

  Args:
    residuals: 1-D float array of cluster residuals e_c of length G.
    counts: 1-D integer array of cluster item counts n_c >= 1 of length G.
    batch_ids: 1-D array of length G with super-batch integer labels.
    level: Confidence level in (0, 1) for the lower confidence bound.

  Returns:
    Lower confidence bound float, or None if untestable (B < 10, G < 2B,
    denominator <= 0, or non-finite).

  Raises:
    ValueError: If arrays have mismatched lengths, invalid dimensions, counts < 1,
      or level outside (0, 1).
  """
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")

  e = np.asarray(residuals, dtype=np.float64)
  n = np.asarray(counts, dtype=np.int64)
  b = np.asarray(batch_ids)

  if e.ndim != 1 or n.ndim != 1 or b.ndim != 1:
    raise ValueError("residuals, counts, and batch_ids must all be 1-D arrays.")
  if not (len(e) == len(n) == len(b)):
    raise ValueError(
        f"Length mismatch: len(residuals)={len(e)}, len(counts)={len(n)}, "
        f"len(batch_ids)={len(b)}."
    )
  if np.any(n < 1):
    raise ValueError("All counts must be >= 1.")

  if not np.all(np.isfinite(e)) or not np.all(np.isfinite(n)):
    return None

  g = len(e)
  _, dense_b = np.unique(b, return_inverse=True)
  num_b = len(_)

  if num_b < 10 or g < 2 * num_b:
    return None

  t_b = np.bincount(dense_b, weights=e, minlength=num_b).astype(np.float64)
  n_b = np.bincount(dense_b, weights=n, minlength=num_b).astype(np.float64)

  if np.any(n_b <= 0):
    return None

  num = float(np.sum((t_b**2) / n_b)) / float(num_b - 1)
  denom = float(np.sum((e**2) / n.astype(np.float64))) / float(g - 1)

  if denom <= 0.0 or not math.isfinite(denom) or not math.isfinite(num):
    return None

  kappa_hat_b = num / denom
  chi2_thresh = float(scipy.stats.chi2.ppf(level, df=num_b - 1))
  if chi2_thresh <= 0.0 or not math.isfinite(chi2_thresh):
    return None

  lcb = float((num_b - 1) * kappa_hat_b / chi2_thresh)
  return lcb if math.isfinite(lcb) else None


def check_kappa_batches(
    residuals: Any,
    counts: Any,
    batch_ids: Any,
    kappa_declared: float,
    level: float = 0.95,
) -> RefutationResult:
  """Tests whether kappa_declared is refuted by super-batch means LCB.

  Refutes when kappa_lower_bound_batches > kappa_declared.
  Under positive adjacent-cluster dependence, the super-batch estimator is,
  to first order and under balanced super-batches with Var(S_c) proportional
  to n_c, biased low by edges cut across batches (Proposition B), making
  refutations conservative. Outside those conditions the direction of the
  bias is not guaranteed.

  Args:
    residuals: 1-D float array of cluster residuals e_c of length G.
    counts: 1-D integer array of cluster item counts n_c >= 1 of length G.
    batch_ids: 1-D array of length G with super-batch integer labels.
    kappa_declared: Declared kappa value (must be >= 1.0).
    level: Confidence level in (0, 1). Default is 0.95.

  Returns:
    RefutationResult with outcome REFUTED, NOT_REFUTED, or UNTESTABLE.

  Raises:
    ValueError: If kappa_declared < 1.0, or on invalid array dimensions.
  """
  if not (math.isfinite(kappa_declared) and kappa_declared >= 1.0):
    raise ValueError(f"kappa_declared must be >= 1.0, got {kappa_declared}.")

  lcb = kappa_lower_bound_batches(residuals, counts, batch_ids, level=level)
  if lcb is None:
    return RefutationResult(Refutation.UNTESTABLE, None, None)

  outcome = (
      Refutation.REFUTED if lcb > kappa_declared else Refutation.NOT_REFUTED
  )
  return RefutationResult(outcome, lcb, float(kappa_declared))

