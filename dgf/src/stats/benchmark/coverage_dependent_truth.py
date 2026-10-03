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

"""Ground-truth formulas and analytical checks for the coverage_dependent benchmark.

Contents:
- Latent Gaussian field covariance formulas (MSE, RMSE, MAE, Accuracy) and Phi_2.
- Cluster quantities (V_c, V_tot, VIF_true, M_c,true, M_c,var, rho) for AR(1) temporal,
  graph edge-shared noise, and real circular pools.
- Contract predicate under Claim 4.
- T2b equicorrelation latent rho solve.
- Recomputations of count guard h, edge_n_eff, and Lemma I' default_m_c_formula.
- Exact binomial p-value and Clopper-Pearson 99% interval.

Pure numerical functions using numpy and scipy only; no harness logic and no
library calls.
"""

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
import scipy.integrate
import scipy.stats


# ---------------------------------------------------------------------------
# Section 3.1: Latent Gaussian Field Covariances
# ---------------------------------------------------------------------------


def cov_mse(rho: float, sigma: float = 1.0) -> float:
  """Covariance Cov(s_i, s_j) for MSE with latent correlation rho: 2 * sigma^4 * rho^2."""
  return float(2.0 * (sigma**4) * (rho**2))


def cov_mae(rho: float, sigma: float = 1.0) -> float:
  """Covariance Cov(s_i, s_j) for MAE with latent correlation rho:

  (2 * sigma^2 / pi) * (sqrt(1 - rho^2) + rho * arcsin(rho) - 1).
  """
  r = max(-1.0, min(1.0, float(rho)))
  term = math.sqrt(max(0.0, 1.0 - r**2)) + r * math.asin(r) - 1.0
  return float((2.0 * (sigma**2) / math.pi) * term)


def phi2(z: float, rho: float) -> float:
  """Standard bivariate normal CDF Phi_2(z, z; rho).

  Uses the identity:
    Phi_2(z, z; rho) = p^2 + (1 / (2*pi)) * int_0^rho exp(-z^2 / (1 + t)) / sqrt(1 - t^2) dt
  where p = Phi(z). Evaluated by scipy.integrate.quad with tolerance 1e-12.
  """
  r = max(-1.0, min(1.0, float(rho)))
  p = float(scipy.stats.norm.cdf(z))
  if r == 0.0:
    return p**2
  if r == 1.0:
    return p
  if r == -1.0:
    return max(0.0, 2.0 * p - 1.0) if z >= 0 else 0.0

  def integrand(t: float) -> float:
    return math.exp(-(z**2) / (1.0 + t)) / math.sqrt(max(1e-15, 1.0 - t**2))

  integral, _ = scipy.integrate.quad(
      integrand, 0.0, r, epsabs=1e-12, epsrel=1e-12
  )
  return float(p**2 + (1.0 / (2.0 * math.pi)) * integral)


def cov_accuracy(rho: float, p: float) -> float:
  """Covariance Cov(s_i, s_j) for Accuracy with latent correlation rho: Phi_2(z, z; rho) - p^2."""
  r = max(-1.0, min(1.0, float(rho)))
  if r == 0.0:
    return 0.0
  if r == 1.0:
    return float(p * (1.0 - p))
  z = float(scipy.stats.norm.ppf(p))
  return float(phi2(z, r) - p**2)


def item_mean(metric: str, sigma: float = 1.0, p: float = 0.5) -> float:
  """Expected value theta = E[s_i] for latent Gaussian fields."""
  m = metric.lower()
  if m in ("mse", "rmse"):
    return float(sigma**2)
  elif m == "mae":
    return float(sigma * math.sqrt(2.0 / math.pi))
  elif m == "accuracy":
    return float(p)
  else:
    raise ValueError(f"Unsupported metric: {metric}")


def item_variance(metric: str, sigma: float = 1.0, p: float = 0.5) -> float:
  """Variance Var(s_i) = Cov(s_i, s_i) for latent Gaussian fields."""
  m = metric.lower()
  if m in ("mse", "rmse"):
    return float(2.0 * (sigma**4))
  elif m == "mae":
    return float((sigma**2) * (1.0 - 2.0 / math.pi))
  elif m == "accuracy":
    return float(p * (1.0 - p))
  else:
    raise ValueError(f"Unsupported metric: {metric}")


def item_default_m(metric: str, p: float = 0.5) -> float:
  """Item-level second moment ratio bound M_item."""
  m = metric.lower()
  if m in ("mse", "rmse"):
    return 3.0
  elif m == "mae":
    return float(math.pi / 2.0)
  elif m == "accuracy":
    return float(1.0 / p)
  else:
    raise ValueError(f"Unsupported metric: {metric}")


def library_item_default_m(metric: str) -> float:
  """Library's item-level default M per independent.metrics.DEFAULT_M (G7)."""
  m = metric.lower()
  if m in ("mse", "rmse", "r2"):
    return 16.0
  elif m in ("mae", "accuracy"):
    return 4.0
  else:
    raise ValueError(f"Unsupported metric: {metric}")


def summand_covariance(
    metric: str, rho: float, sigma: float = 1.0, p: float = 0.5
) -> float:
  """Covariance Cov(s_i, s_j) given latent correlation rho."""
  m = metric.lower()
  if m in ("mse", "rmse"):
    return cov_mse(rho, sigma)
  elif m == "mae":
    return cov_mae(rho, sigma)
  elif m == "accuracy":
    return cov_accuracy(rho, p)
  else:
    raise ValueError(f"Unsupported metric: {metric}")


# ---------------------------------------------------------------------------
# Section 3.2: Cluster Quantities and Ground Truth
# ---------------------------------------------------------------------------


def temporal_ground_truth(
    metric: str,
    counts: Sequence[int] | np.ndarray,
    phi: float,
    sigma: float = 1.0,
    p: float = 0.5,
) -> dict[str, Any]:
  """Computes exact ground truth for AR(1) temporal latent Gaussian fields.

  In O(n) time using the lag autocovariance sequence C(k).
  """
  c_arr = np.asarray(counts, dtype=np.int64)
  g = int(c_arr.size)
  n = int(np.sum(c_arr))
  theta = item_mean(metric, sigma, p)

  # Precompute lag autocovariances C(k) for k = 0 .. n - 1
  # For AR(1), latent rho(k) = phi^k.
  # For MSE: C(k) = 2 * sigma^4 * (phi^2)^k, geometric sequence.
  # For MAE and Accuracy: evaluate via summand_covariance.
  # If phi == 0.0, C(0) = Var(s_i), C(k) = 0 for k > 0.
  v_item = item_variance(metric, sigma, p)
  c_lags = np.zeros(n, dtype=np.float64)
  c_lags[0] = v_item
  if abs(phi) > 0.0 and n > 1:
    if metric.lower() in ("mse", "rmse"):
      phi2_val = phi**2
      c_lags[1:] = (
          2.0 * (sigma**4) * (phi2_val ** np.arange(1, n, dtype=np.float64))
      )
    else:
      phi_k = phi ** np.arange(1, n, dtype=np.float64)
      for k in range(1, n):
        c_lags[k] = summand_covariance(metric, phi_k[k - 1], sigma, p)

  # Total variance V_tot = n C(0) + 2 * sum_{k=1}^{n-1} (n - k) C(k)
  if n > 1:
    k_weights_tot = (n - np.arange(1, n, dtype=np.float64)).astype(np.float64)
    v_tot = float(n * c_lags[0] + 2.0 * np.sum(k_weights_tot * c_lags[1:]))
  else:
    v_tot = float(c_lags[0])

  # Cluster variances V_c = n_c C(0) + 2 * sum_{k=1}^{n_c - 1} (n_c - k) C(k)
  v_c = np.zeros(g, dtype=np.float64)
  for c_idx, nc in enumerate(c_arr):
    if nc > 1:
      k_weights = (nc - np.arange(1, nc, dtype=np.float64)).astype(np.float64)
      v_c[c_idx] = float(nc * c_lags[0] + 2.0 * np.sum(k_weights * c_lags[1:nc]))
    else:
      v_c[c_idx] = float(c_lags[0])

  sum_v_c = float(np.sum(v_c))
  vif_true = float(v_tot / sum_v_c) if sum_v_c > 0 else 1.0

  mu_c = c_arr.astype(np.float64) * theta
  denom_m = float((n * theta) ** 2) if (n * theta) != 0 else 1.0
  m_c_true = float(g * np.sum(v_c + mu_c**2) / denom_m)
  m_c_var = float(1.0 + g * sum_v_c / denom_m)

  return {
      "theta": theta,
      "n": n,
      "g": g,
      "counts": c_arr,
      "v_tot": v_tot,
      "v_c": v_c,
      "sum_v_c": sum_v_c,
      "vif_true": vif_true,
      "m_c_true": m_c_true,
      "m_c_var": m_c_var,
  }


def t3_drift_ground_truth(
    n: int,
    g: int,
    sigma: float,
    delta: float,
) -> dict[str, Any]:
  """Computes exact ground truth for T3 drift instances.

  Items in the second half have variance sigma^2 * (1 + delta).
  Item estimand theta = sigma^2 * (1 + delta / 2).
  """
  window_ids = (np.arange(n, dtype=np.int64) * g) // n
  counts = np.bincount(window_ids, minlength=g).astype(np.int64)
  n_half = n // 2
  w_first = window_ids[:n_half]
  w_second = window_ids[n_half:]
  n_c1 = np.bincount(w_first, minlength=g).astype(np.float64)
  n_c2 = np.bincount(w_second, minlength=g).astype(np.float64)

  mu_c = n_c1 * (sigma**2) + n_c2 * (sigma**2 * (1.0 + delta))
  theta = float(np.sum(mu_c) / float(n))

  v_c = 2.0 * n_c1 * (sigma**4) + 2.0 * n_c2 * (sigma**4 * ((1.0 + delta) ** 2))
  sum_v_c = float(np.sum(v_c))
  v_tot = sum_v_c
  vif_true = 1.0

  denom_m = float((n * theta) ** 2) if (n * theta) != 0 else 1.0
  m_c_true = float(g * np.sum(v_c + mu_c**2) / denom_m)
  m_c_var = float(1.0 + g * sum_v_c / denom_m)

  return {
      "theta": theta,
      "n": n,
      "g": g,
      "counts": counts,
      "v_tot": v_tot,
      "v_c": v_c,
      "sum_v_c": sum_v_c,
      "vif_true": vif_true,
      "m_c_true": m_c_true,
      "m_c_var": m_c_var,
  }


def graph_ground_truth(
    metric: str,
    counts: Sequence[int] | np.ndarray,
    cluster_edges: np.ndarray,
    node_degrees: np.ndarray,
    internal_edges_endpoints: np.ndarray,
    cross_edges_endpoints: np.ndarray,
    gamma: float,
    sigma: float = 1.0,
    p: float = 0.5,
) -> dict[str, Any]:
  """Computes exact ground truth for graph edge-shared noise latent Gaussian fields.

  In O(n + |E|) time using node degrees and edge endpoints.
  """
  c_arr = np.asarray(counts, dtype=np.int64)
  g = int(c_arr.size)
  n = int(np.sum(c_arr))
  theta = item_mean(metric, sigma, p)
  v_item = item_variance(metric, sigma, p)

  # For each edge (i, j), latent correlation rho_ij = gamma^2 / sqrt((1 + gamma^2 d_i)(1 + gamma^2 d_j))
  # Edge summand covariance = summand_covariance(metric, rho_ij, sigma, p)
  def compute_edge_covs(edge_list: np.ndarray) -> np.ndarray:
    if edge_list.size == 0 or gamma == 0.0:
      return np.zeros(len(edge_list), dtype=np.float64)
    u = edge_list[:, 0]
    v = edge_list[:, 1]
    denom = np.sqrt(
        (1.0 + (gamma**2) * node_degrees[u].astype(np.float64))
        * (1.0 + (gamma**2) * node_degrees[v].astype(np.float64))
    )
    rhos = (gamma**2) / denom
    m = metric.lower()
    if m in ("mse", "rmse"):
      return 2.0 * (sigma**4) * (rhos**2)
    elif m == "mae":
      r = np.clip(rhos, -1.0, 1.0)
      term = np.sqrt(np.maximum(0.0, 1.0 - r**2)) + r * np.arcsin(r) - 1.0
      return (2.0 * (sigma**2) / math.pi) * term
    elif m == "accuracy":
      out = np.zeros(len(rhos), dtype=np.float64)
      for idx, r in enumerate(rhos):
        out[idx] = summand_covariance(metric, float(r), sigma, p)
      return out
    else:
      raise ValueError(f"Unsupported metric: {metric}")

  int_covs = compute_edge_covs(internal_edges_endpoints)
  cross_covs = compute_edge_covs(cross_edges_endpoints)

  # V_tot = n * V_item + 2 * sum_{all edges} Cov(s_i, s_j)
  sum_int_cov = float(np.sum(int_covs))
  sum_cross_cov = float(np.sum(cross_covs))
  v_tot = float(n * v_item + 2.0 * (sum_int_cov + sum_cross_cov))

  # Sum_c V_c = n * V_item + 2 * sum_{internal edges} Cov(s_i, s_j)
  sum_v_c = float(n * v_item + 2.0 * sum_int_cov)

  vif_true = float(v_tot / sum_v_c) if sum_v_c > 0 else 1.0

  denom_m = float((n * theta) ** 2) if (n * theta) != 0 else 1.0
  # M_c,true = G * [ sum_c V_c + theta^2 sum_c n_c^2 ] / (n * theta)^2
  sum_count_sq = float(np.sum(c_arr.astype(np.float64) ** 2))
  m_c_true = float(g * (sum_v_c + (theta**2) * sum_count_sq) / denom_m)
  m_c_var = float(1.0 + g * sum_v_c / denom_m)

  return {
      "theta": theta,
      "n": n,
      "g": g,
      "counts": c_arr,
      "v_tot": v_tot,
      "sum_v_c": sum_v_c,
      "vif_true": vif_true,
      "m_c_true": m_c_true,
      "m_c_var": m_c_var,
  }


def circular_pool_autocovariance(
    pool_summands: np.ndarray,
) -> tuple[float, np.ndarray]:
  """Computes exact circular population mean theta and lag autocovariances C_circ(d)."""
  s = np.asarray(pool_summands, dtype=np.float64)
  n_pop = int(s.size)
  theta = float(np.mean(s))
  centered = s - theta
  # Fast circular convolution via FFT
  f = np.fft.fft(centered)
  ac = np.fft.ifft(f * np.conj(f)).real / float(n_pop)
  return theta, ac.astype(np.float64)


def circular_pool_ground_truth(
    counts: Sequence[int] | np.ndarray,
    theta: float,
    circ_cov: np.ndarray,
) -> dict[str, Any]:
  """Computes exact ground truth for circular pool samples of length n."""
  c_arr = np.asarray(counts, dtype=np.int64)
  g = int(c_arr.size)
  n = int(np.sum(c_arr))

  if n > 1:
    k_weights_tot = (n - np.arange(1, n, dtype=np.float64)).astype(np.float64)
    v_tot = float(n * circ_cov[0] + 2.0 * np.sum(k_weights_tot * circ_cov[1:n]))
  else:
    v_tot = float(n * circ_cov[0])

  v_c = np.zeros(g, dtype=np.float64)
  for c_idx, nc in enumerate(c_arr):
    if nc > 1:
      k_weights = (nc - np.arange(1, nc, dtype=np.float64)).astype(np.float64)
      v_c[c_idx] = float(
          nc * circ_cov[0] + 2.0 * np.sum(k_weights * circ_cov[1:nc])
      )
    else:
      v_c[c_idx] = float(circ_cov[0])

  sum_v_c = float(np.sum(v_c))
  vif_true = float(v_tot / sum_v_c) if sum_v_c > 0 else 1.0

  denom_m = float((n * theta) ** 2) if (n * theta) != 0 else 1.0
  sum_count_sq = float(np.sum(c_arr.astype(np.float64) ** 2))
  m_c_true = float(g * (sum_v_c + (theta**2) * sum_count_sq) / denom_m)
  m_c_var = float(1.0 + g * sum_v_c / denom_m)

  return {
      "theta": theta,
      "n": n,
      "g": g,
      "counts": c_arr,
      "v_tot": v_tot,
      "v_c": v_c,
      "sum_v_c": sum_v_c,
      "vif_true": vif_true,
      "m_c_true": m_c_true,
      "m_c_var": m_c_var,
  }


# ---------------------------------------------------------------------------
# Section 3.4 & 5: Contract, Recomputations, and Test Criteria Helpers
# ---------------------------------------------------------------------------


def is_in_contract(
    m_c_var: float,
    m_c_decl: float,
    vif_true: float,
    kappa_decl: float,
) -> bool:
  """Evaluates whether (cell, side) is in contract under Claim 4."""
  return (m_c_var <= m_c_decl + 1e-12) and (vif_true <= kappa_decl + 1e-12)


def theorem5_rho(
    v_tot: float,
    m_c_decl: float,
    kappa_decl: float,
    n: int,
    theta: float,
    g: int,
) -> float:
  """Computes Claim 5 ratio rho = V_tot / ((M_c,decl - 1) * kappa_decl * (n*theta)^2 / G)."""
  denom = (m_c_decl - 1.0) * kappa_decl * ((float(n) * theta) ** 2) / float(g)
  if denom <= 0.0 or not math.isfinite(denom):
    return float("inf")
  return float(v_tot / denom)


def theorem5_bound(rho: float, alpha: float) -> float:
  """Computes Claim 5 miss bound rho * alpha' / (1 - alpha' + rho * alpha') with alpha' = alpha/2."""
  alpha_prime = alpha / 2.0
  denom = 1.0 - alpha_prime + rho * alpha_prime
  if denom <= 0.0 or not math.isfinite(denom):
    return 1.0
  return float(min(1.0, (rho * alpha_prime) / denom))


def t2b_latent_rho(window_count: int = 100, target_vif: float = 4.0) -> float:
  """Finds latent correlation rho so that within-window summand VIF is exactly target_vif."""
  if window_count <= 1:
    return 0.0
  rho_sq = (target_vif - 1.0) / float(window_count - 1)
  return float(math.sqrt(max(0.0, rho_sq)))


def compute_count_guard_h(counts: Sequence[int] | np.ndarray) -> float:
  """Recomputes count guard h = max(h_1, h_2) independently."""
  c = np.asarray(counts, dtype=np.float64)
  if c.size < 2:
    return 1.0
  h_vals = []
  for a in (1, 2):
    n_pow = c**a
    num_a = float(np.mean(n_pow[:-1] * n_pow[1:]))
    den_a = float(np.mean(n_pow) ** 2)
    if den_a <= 0.0 or not math.isfinite(den_a):
      h_vals.append(float("inf"))
    else:
      h_vals.append(num_a / den_a)
  return float(max(h_vals))


def compute_edge_n_eff(
    residuals: np.ndarray,
    counts: np.ndarray,
    edges: np.ndarray,
) -> float:
  """Recomputes edge_n_eff = (sum_E |y|)^2 / sum_E y^2 on unique cluster edges."""
  e = np.asarray(residuals, dtype=np.float64)
  n = np.asarray(counts, dtype=np.float64)
  if len(edges) == 0:
    return 0.0
  z = e / np.sqrt(n)
  u = edges[:, 0]
  v = edges[:, 1]
  y = z[u] * z[v]
  sum_sq = float(np.sum(y**2))
  if sum_sq <= 0.0 or not math.isfinite(sum_sq):
    return 0.0
  sum_abs = float(np.sum(np.abs(y)))
  return float((sum_abs**2) / sum_sq)


def default_m_c_formula(
    counts: Sequence[int] | np.ndarray,
    m_item: float,
) -> float:
  """Recomputes Lemma I' default M_c formula independently:

  min(r * M, 1 + r*(M - 1) + CV_n^2 + 2 * CV_n * sqrt(r*(M - 1)))
  where r = G * n_max / n, CV_n^2 = G * sum(n_c^2) / n^2 - 1.
  """
  c = np.asarray(counts, dtype=np.float64)
  g = float(c.size)
  n = float(np.sum(c))
  n_max = float(np.max(c))
  sum_sq = float(np.sum(c**2))

  from fractions import Fraction
  ratio1 = Fraction(int(np.max(c)) * int(c.size), int(np.sum(c)))
  ratio2 = Fraction(int(c.size) * int(np.sum(c**2)) - int(np.sum(c))**2, int(np.sum(c))**2)
  if ratio1 == 1 and ratio2 == 0:
    return float(m_item)

  r = float(ratio1)
  cv2 = float(ratio2)
  m = float(m_item)
  branch_a = r * m
  diff = max(0.0, m - 1.0)
  cv = math.sqrt(max(0.0, cv2))
  branch_b = 1.0 + r * diff + cv2 + 2.0 * cv * math.sqrt(max(0.0, r * diff))
  return float(min(branch_a, branch_b))


def clopper_pearson_99(k: int, n: int) -> tuple[float, float]:
  """Computes exact two-sided Clopper-Pearson 99% confidence interval."""
  if n <= 0:
    return 0.0, 1.0
  lo = 0.0 if k == 0 else float(scipy.stats.beta.ppf(0.005, k, n - k + 1))
  hi = 1.0 if k == n else float(scipy.stats.beta.ppf(0.995, k + 1, n - k))
  return lo, hi


def binomial_p_value(misses: int, reps: int, alpha: float) -> float:
  """Exact one-sided binomial test p-value: P(Binomial(reps, alpha) >= misses)."""
  if misses <= 0:
    return 1.0
  return float(scipy.stats.binom.sf(misses - 1, reps, alpha))
