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

"""Finite-sample confidence bounds for a mean under a relative-variance bound.

Implements Algorithm 1 of THEORY_INDEPENDENT.md (§7): given the sufficient statistics of a sample
of non-negative summands s_i and a declared bound E[s^2] <= M E[s]^2, returns a
two-sided interval for theta = E[s] with finite-sample coverage >= 1 - alpha.
"""

import math
from typing import Sequence
import numpy as np

from dgf.src.stats.independent import metrics


_GAMMA_SCALE = 0.35
_SLOPE_INF_MIN = 0.10


def c_cantelli(n: float, M: float, alpha_p: float) -> float:
  """Cantelli constant c such that Pr(sbar <= theta(1-c)) <= alpha_p.

  Under P1 and P2*(M), Var(s) <= (M-1)theta^2.
  Cantelli's inequality gives c = sqrt((M-1)(1-alpha_p)/(n * alpha_p)).
  """
  if n <= 0 or alpha_p <= 0 or alpha_p >= 1.0 or M < 1.0:
    return float("inf")
  return math.sqrt((M - 1.0) * (1.0 - alpha_p) / (float(n) * alpha_p))


def c_bernstein(n: float, M: float, alpha_p: float) -> float:
  """Lemma E: One-sided Bernstein constant c such that Pr(sbar <= theta(1-c)) <= alpha_p.

  Let s_1..s_n >= 0 be i.i.d. with E[s] = theta > 0 and E[s^2] <= M theta^2.
  Put X_i = -(s_i - theta). Then E[X_i] = 0, X_i <= theta (since s_i >= 0),
  and Var(X_i) <= (M - 1) theta^2. Bernstein's inequality for variables bounded above
  by b = theta gives:
    Pr(sbar <= theta(1-c)) <= exp( - n c^2 / (2(M-1) + 2c/3) )
  Setting the RHS to alpha_p with L = ln(1/alpha_p) yields the quadratic:
    n c^2 - (2/3) L c - 2 (M - 1) L = 0
  The unique positive root is:
    c_bernstein = [ (2/3) L + sqrt( (2/3 L)^2 + 8 n (M - 1) L ) ] / (2 n).
  """
  if n <= 0 or alpha_p <= 0 or alpha_p >= 1.0 or M < 1.0:
    return float("inf")
  L = math.log(1.0 / alpha_p)
  two_thirds_L = (2.0 / 3.0) * L
  disc = two_thirds_L ** 2 + 8.0 * float(n) * (M - 1.0) * L
  return (two_thirds_L + math.sqrt(max(0.0, disc))) / (2.0 * float(n))


def n_A(M: float, alpha: float) -> float:
  """Cantelli threshold: c_cantelli < 1 iff n > n_A(M, alpha)."""
  alpha_p = alpha / 2.0
  return (M - 1.0) * (1.0 - alpha_p) / alpha_p


def n_B(M: float, alpha: float) -> float:
  """Lemma B lower-sweep threshold: slope_inf > 0.10 iff n > n_B(M, alpha).

  This is a width threshold, not a validity threshold: above it on the
  independent path (K >= 1) the Lemma B lower sweep is used; at or below it
  the Lemma A' lower endpoint is used, which is equally valid.
  """
  coeff = ((_GAMMA_SCALE + 1.0 / _GAMMA_SCALE) / (1.0 - _SLOPE_INF_MIN)) ** 2
  return 1.0 + coeff * (7.0 / 3.0) * M * math.log(4.0 / alpha)


def n_U(M: float, alpha: float) -> float:
  """Validity threshold of the i.i.d. upper endpoint: c_upper < 1 iff n > n_U.

  c_upper = min(c_cantelli, c_bernstein), so n_U is the smaller of the two
  thresholds at which each constant drops below 1. At or below n_U no finite
  upper endpoint exists and `bound` refuses with ASSUMPTION_REQUIRED.
  """
  alpha_p = alpha / 2.0
  L = math.log(1.0 / alpha_p)
  n_bern = (2.0 * (M - 1.0) + 2.0 / 3.0) * L
  return min(n_A(M, alpha), n_bern)


def _solve_lower_piecewise(
    n: int,
    top_k: np.ndarray,
    gamma: float,
    Lg: float,
    sum_s: float | None = None,
    sum_s2: float | None = None,
) -> float:
  """Exact piecewise quadratic solver for L = inf{ t > 0 : t - G_lo(t) >= 0 }.

  Evaluates the Lemma B lower sweep in data mode (K = n) and sketch mode (K < n)
  via the sketch-mode lower envelope (THEORY_INDEPENDENT.md §5.2).

  Args:
    n: Sample size.
    top_k: Retained summands (K elements), sorted descending.
    gamma: Truncation scale of Lemma B.
    Lg: log(2 / alpha').
    sum_s: Total sum of all n summands (required if K < n).
    sum_s2: Total sum of squares of all n summands (required if K < n).

  Returns:
    The infimum above, or 0.0 if no crossing is found.

  Raises:
    ValueError: If K < 1 or K > n, or if sum_s/sum_s2 are missing when K < n.
  """
  K = len(top_k)
  if K < 1 or K > n:
    raise ValueError(f"Requires 1 <= K <= n, got {K} of {n}.")

  inv_n_n1 = 1.0 / float(n * (n - 1))
  C_L = 2.0 * Lg / float(n)

  if K == n:
    s_rest = 0.0
    q_rest = 0.0
    suf1 = np.zeros(K + 1, dtype=np.float64)
    suf2 = np.zeros(K + 1, dtype=np.float64)
    for j in range(K - 1, -1, -1):
      suf1[j] = suf1[j + 1] + top_k[j]
      suf2[j] = suf2[j + 1] + top_k[j] ** 2
  else:
    if sum_s is None or sum_s2 is None:
      raise ValueError("sum_s and sum_s2 required in sketch mode (K < n).")
    s_K = float(top_k[K - 1])
    n_rest = float(n - K)
    s_rest = max(0.0, float(sum_s) - float(np.sum(top_k)))
    s_rest = min(s_rest, n_rest * s_K)
    q_rest = max(0.0, float(sum_s2) - float(np.sum(top_k ** 2)))
    q_rest = min(max(q_rest, (s_rest ** 2) / n_rest), s_rest * s_K)
    suf1 = np.zeros(K + 1, dtype=np.float64)
    suf2 = np.zeros(K + 1, dtype=np.float64)
    suf1[K] = s_rest
    suf2[K] = q_rest
    for j in range(K - 1, -1, -1):
      suf1[j] = suf1[j + 1] + top_k[j]
      suf2[j] = suf2[j + 1] + top_k[j] ** 2

  def _solve_piece(
      t_lo: float,
      t_hi: float,
      d0: float,
      d1: float,
      v0: float,
      v1: float,
      v2: float,
  ) -> float | None:
    if t_lo >= t_hi:
      return None

    def _h_lo(t: float) -> float:
      return d1 * t - d0 + math.sqrt(C_L * max(0.0, v2 * t * t + v1 * t + v0))

    if t_lo > 0.0:
      if _h_lo(t_lo) >= -1e-12:
        return t_lo
    elif abs(d0) <= 1e-15 and abs(v0) <= 1e-15 and abs(v1) <= 1e-15:
      if d1 + math.sqrt(C_L * max(0.0, v2)) >= 0.0:
        return 0.0
    elif _h_lo(0.0) >= -1e-12:
      return 0.0

    p2 = d1 ** 2 - C_L * v2
    p1 = -2.0 * d0 * d1 - C_L * v1
    p0 = d0 ** 2 - C_L * v0

    roots = []
    if abs(p2) < 1e-15:
      if abs(p1) > 1e-15:
        r = -p0 / p1
        roots.append(r)
    else:
      disc = p1 ** 2 - 4.0 * p2 * p0
      if disc >= 0.0:
        sqrt_d = math.sqrt(max(0.0, disc))
        r1 = (-p1 + sqrt_d) / (2.0 * p2)
        r2 = (-p1 - sqrt_d) / (2.0 * p2)
        roots.extend([r1, r2])

    valid_roots = []
    for r in roots:
      if r <= 1e-12:
        continue
      if t_lo == 0.0:
        in_piece = (r <= t_hi + 1e-11)
      elif math.isinf(t_hi):
        in_piece = (r > t_lo - 1e-11)
      else:
        in_piece = (t_lo - 1e-11 < r <= t_hi + 1e-11)

      if in_piece:
        if d0 - d1 * r >= -1e-9:
          v_val = v2 * r ** 2 + v1 * r + v0
          if v_val >= -1e-9:
            valid_roots.append(r)

    if valid_roots:
      # Control reaches here only with H_lo(t_lo) < 0. By continuity, the
      # infimum of {t > 0 : H_lo(t) >= 0} on this piece is the smallest root
      # in the piece. Convexity is not assumed across all relaxations (sketch
      # sub-pieces can be concave); a spurious smaller root can only widen
      # the interval (lower L), never cause an under-coverage bug.
      return min(valid_roots)

    return None

  s_K = float(top_k[K - 1])

  # Bottom region (piece k = K, t in [0, s_K / gamma])
  if s_K > 0.0:
    if K == n or s_rest <= 0.0 or q_rest <= 0.0:
      d0 = 0.0
      d1 = 1.0 - (float(K) * gamma) / float(n) + (7.0 * gamma * Lg) / (3.0 * float(n - 1))
      v0 = 0.0
      v1 = 0.0
      v2 = float(K) * float(n - K) * (gamma ** 2) * inv_n_n1
      r = _solve_piece(0.0, s_K / gamma, d0, d1, v0, v1, v2)
      if r is not None:
        return r
    else:
      n_rest = float(n - K)
      c1 = s_rest / n_rest
      c_mid = q_rest / (2.0 * s_rest) if s_rest > 0.0 else 0.0
      c2 = q_rest / s_rest if s_rest > 0.0 else 0.0
      raw_knots = sorted([0.0, c1, c_mid, c2, s_K])
      knots = [0.0]
      for ck in raw_knots[1:]:
        ck_clamped = min(s_K, max(0.0, ck))
        if ck_clamped > knots[-1] + 1e-15 * max(1.0, s_K):
          knots.append(ck_clamped)
      if knots[-1] < s_K:
        knots.append(s_K)

      def _y_lo(c_val: float) -> float:
        if c_val <= 0.0:
          return 0.0
        if c_val >= s_K:
          return s_rest
        chord = (s_rest / s_K) * c_val
        if c_val <= c_mid:
          quad = ((s_rest ** 2) / q_rest) * c_val
        else:
          quad = s_rest - q_rest / (4.0 * c_val)
        return min(s_rest, max(chord, quad))

      for m in range(len(knots) - 1):
        c_a = knots[m]
        c_b = knots[m + 1]
        t_lo = c_a / gamma
        t_hi = c_b / gamma
        B1 = (_y_lo(c_b) - _y_lo(c_a)) / (c_b - c_a)
        A1 = max(0.0, _y_lo(c_a) - B1 * c_a)
        k_eff = float(K) + B1
        d0 = A1 / float(n)
        d1 = 1.0 - (k_eff * gamma) / float(n) + (7.0 * gamma * Lg) / (3.0 * float(n - 1))
        c_mp = 0.5 * (c_a + c_b)
        if c_mp <= c1:
          v0 = -(A1 ** 2) * inv_n_n1
          v1 = -2.0 * A1 * k_eff * gamma * inv_n_n1
          v2 = (float(n ** 2) - k_eff ** 2) * (gamma ** 2) * inv_n_n1
        elif c_mp <= c2:
          v0 = -(A1 ** 2) * inv_n_n1
          v1 = (float(n) * s_rest - 2.0 * A1 * k_eff) * gamma * inv_n_n1
          v2 = (float(n * K) - k_eff ** 2) * (gamma ** 2) * inv_n_n1
        else:
          v0 = (float(n) * q_rest - A1 ** 2) * inv_n_n1
          v1 = -2.0 * A1 * k_eff * gamma * inv_n_n1
          v2 = (float(n * K) - k_eff ** 2) * (gamma ** 2) * inv_n_n1

        r = _solve_piece(t_lo, t_hi, d0, d1, v0, v1, v2)
        if r is not None:
          return r

  # Exact pieces k = K - 1 down to 0
  for k in range(K - 1, -1, -1):
    t_lo = top_k[k] / gamma
    t_hi = top_k[k - 1] / gamma if k > 0 else float("inf")
    if t_lo >= t_hi:
      continue
    S1 = float(suf1[k])
    S2 = float(suf2[k])
    v0 = (float(n) * S2 - S1 ** 2) * inv_n_n1
    v1 = -2.0 * float(k) * gamma * S1 * inv_n_n1
    v2 = float(k * (n - k)) * (gamma ** 2) * inv_n_n1
    d0 = S1 / float(n)
    d1 = 1.0 - (float(k) * gamma) / float(n) + (7.0 * gamma * Lg) / (3.0 * float(n - 1))
    r = _solve_piece(t_lo, t_hi, d0, d1, v0, v1, v2)
    if r is not None:
      return r

  return 0.0


def _truncated_moments(
    n: int,
    sum_s: float,
    sum_s2: float,
    top_k: np.ndarray,
    c: float,
) -> tuple[int, float, float]:
  """Returns (cnt, T1, T2) for the sample truncated at threshold `c`.

  T1 = sum_i min(s_i, c), T2 = sum_i min(s_i, c)^2, with `cnt` the number of
  summands above `c`.
  """
  cnt = int(np.sum(top_k > c))
  if len(top_k) == n or cnt < len(top_k):
    if len(top_k) == n:
      rest1 = float(np.sum(top_k[cnt:])) if cnt < n else 0.0
      rest2 = float(np.sum(top_k[cnt:] ** 2)) if cnt < n else 0.0
    else:
      P1k = float(np.sum(top_k[:cnt])) if cnt > 0 else 0.0
      P2k = float(np.sum(top_k[:cnt] ** 2)) if cnt > 0 else 0.0
      rest1 = sum_s - P1k
      rest2 = sum_s2 - P2k
    return cnt, rest1 + cnt * c, rest2 + cnt * (c ** 2)

  # Sketch mode (len(top_k) < n) with c < s_{(K)}
  K = len(top_k)
  s_K = float(top_k[K - 1])
  s_rest = max(0.0, float(sum_s) - float(np.sum(top_k)))
  q_rest = max(0.0, float(sum_s2) - float(np.sum(top_k ** 2)))
  n_rest = float(n - K)
  s_rest = min(s_rest, n_rest * s_K)
  q_rest = min(max(q_rest, (s_rest ** 2) / n_rest), s_rest * s_K)
  if s_rest <= 0.0 or q_rest <= 0.0 or s_K <= 0.0 or c <= 0.0:
    return K, float(K) * c, float(K) * (c ** 2)

  c1 = s_rest / n_rest
  c_mid = q_rest / (2.0 * s_rest) if s_rest > 0.0 else 0.0
  c2 = q_rest / s_rest if s_rest > 0.0 else 0.0
  raw_knots = sorted([0.0, c1, c_mid, c2, s_K])
  knots = [0.0]
  for ck in raw_knots[1:]:
    ck_clamped = min(s_K, max(0.0, ck))
    if ck_clamped > knots[-1] + 1e-15 * max(1.0, s_K):
      knots.append(ck_clamped)
  if knots[-1] < s_K:
    knots.append(s_K)

  def _y_lo(c_val: float) -> float:
    if c_val <= 0.0:
      return 0.0
    if c_val >= s_K:
      return s_rest
    chord = (s_rest / s_K) * c_val
    if c_val <= c_mid:
      quad = ((s_rest ** 2) / q_rest) * c_val
    else:
      quad = s_rest - q_rest / (4.0 * c_val)
    return min(s_rest, max(chord, quad))

  c_clamped = min(s_K, max(0.0, c))
  for m in range(len(knots) - 1):
    c_a = knots[m]
    c_b = knots[m + 1]
    if c_a <= c_clamped <= c_b or m == len(knots) - 2:
      B1 = (_y_lo(c_b) - _y_lo(c_a)) / (c_b - c_a)
      A1 = max(0.0, _y_lo(c_a) - B1 * c_a)
      t1_lo = float(K) * c + (A1 + B1 * c)
      t2_rest = min(n_rest * (c ** 2), s_rest * c, q_rest)
      t2_hi = float(K) * (c ** 2) + t2_rest
      return K, t1_lo, t2_hi

  return K, float(K) * c, float(K) * (c ** 2)


def compute_H_lo(
    t: float,
    n: int,
    sum_s: float,
    sum_s2: float,
    top_k: np.ndarray,
    gamma: float,
    Lg: float,
) -> float:
  """Direct evaluation of t - G_lo(t), used to cross-check the solver in tests."""
  c = gamma * t
  _, T1, T2 = _truncated_moments(n, sum_s, sum_s2, top_k, c)
  Vhat = (n * T2 - T1 ** 2) / float(n * (n - 1))
  Vhat = max(0.0, Vhat)

  G_lo = T1 / float(n) - math.sqrt(2.0 * Vhat * Lg / float(n)) - (7.0 * c * Lg) / (3.0 * (n - 1))
  return t - G_lo


def _lemma_b_params(n: int, alpha: float, M: float) -> tuple[float, float, float]:
  """Returns (gamma, Lg, slope_inf) for the Lemma B lower sweep."""
  alpha_p = alpha / 2.0
  Lg = math.log(2.0 / alpha_p)
  gamma_star = math.sqrt(3.0 * M * float(n - 1) / (7.0 * Lg))
  gamma = _GAMMA_SCALE * gamma_star
  slope_inf = 1.0 - (7.0 * gamma * Lg) / (3.0 * float(n - 1)) - M / gamma
  return gamma, Lg, slope_inf


def _use_lower_sweep(
    n: int, alpha: float, M: float, K: int, effective_n: float | None
) -> bool:
  """Deterministic in (n, alpha, M, K, effective_n) -- never in the data.

  By Lemma C (THEORY_INDEPENDENT.md §5.3), a routing rule that depends only on
  quantities fixed before the data are seen costs no union-bound penalty:
  choosing between the Lemma B sweep and Lemma A' consumes only the single
  per-side budget alpha/2. The sweep is used on the independent path whenever
  K >= 1 and n > n_B(M, alpha) (equivalently, slope_inf > 0.10 at
  gamma = 0.35 * gamma*).
  """
  if effective_n is not None:
    return False
  if K < 1:
    return False
  _, _, slope_inf = _lemma_b_params(n, alpha, M)
  return slope_inf > _SLOPE_INF_MIN


def bound(
    n: int,
    sum_s: float,
    sum_s2: float,
    top_k: Sequence[float] | np.ndarray,
    alpha: float,
    M: float,
    metric: str = "mse",
    effective_n: float | None = None,
) -> tuple[tuple[float, float], str]:
  """Finite-sample confidence interval for theta = E[s] under E[s^2] <= M theta^2.

  Args:
    n: Number of summands.
    sum_s: Sum of the summands s_i >= 0.
    sum_s2: Sum of their squares.
    top_k: The largest summands (the whole sample in data mode, the retained top
      values in sketch mode).
    alpha: Total miss probability; alpha/2 is spent on each side.
    M: Declared relative-variance bound, M >= 1.
    metric: Metric name. `rmse` returns square-rooted endpoints; metrics with a
      definitional maximum (accuracy) have the upper endpoint clamped to it.
    effective_n: If set, the summands are treated as correlated with this
      effective sample size (Claim 1′); only Cantelli-type bounds are used.

  Returns:
    ((low, high), status) with status one of "UNREFUTED", "TAIL_UNRESOLVED" or
    "ASSUMPTION_REQUIRED". On ASSUMPTION_REQUIRED the endpoints are NaN.
  """
  # Degenerate guards
  if n < 2 or not math.isfinite(sum_s) or not math.isfinite(sum_s2) or M < 1.0:
    return (float("nan"), float("nan")), "ASSUMPTION_REQUIRED"

  top_k_arr = np.asarray(top_k, dtype=np.float64)
  if np.any(~np.isfinite(top_k_arr)):
    return (float("nan"), float("nan")), "ASSUMPTION_REQUIRED"
  if top_k_arr.size > n:
    return (float("nan"), float("nan")), "ASSUMPTION_REQUIRED"

  alpha_p = alpha / 2.0
  s_bar = sum_s / float(n)

  # Upper endpoint: the smaller of the Cantelli and one-sided Bernstein
  # (Lemma E) constants. Neither depends on the data, so taking the smaller is
  # a data-independent choice and costs no extra alpha (Lemma C).
  if effective_n is None:
    nu = float(n)
    c_up = min(c_cantelli(n, M, alpha_p), c_bernstein(n, M, alpha_p))
  else:
    nu = float(effective_n)
    c_up = c_cantelli(effective_n, M, alpha_p)  # Bernstein needs independence

  if not (c_up < 1.0):
    return (float("nan"), float("nan")), "ASSUMPTION_REQUIRED"
  U = s_bar / (1.0 - c_up)

  if sum_s == 0.0:
    # If theta > 0, the event {sum_s = 0} is contained in
    # {s_bar <= theta (1 - c_up)}, whose probability is already charged to the
    # upper side (<= alpha'). If theta = 0, [0, 0] covers. Either way returning
    # [0, 0] adds no miss probability.
    return (0.0, 0.0), "UNREFUTED"

  # M_hat uses the raw n: M_hat = n * sum_s2 / (sum_s)^2 is a ratio of two
  # sample means, each unbiased regardless of dependence, so it estimates
  # E[s^2]/theta^2 in the correlated case too. Using n_eff here would be wrong.
  M_hat = (float(n) * sum_s2) / (sum_s ** 2)
  status = "TAIL_UNRESOLVED" if M_hat > M else "UNREFUTED"

  # Lower endpoint: the Lemma B sweep where it applies, else Lemma A'.
  K = len(top_k_arr)
  if _use_lower_sweep(n, alpha, M, K, effective_n):
    gamma, Lg, _ = _lemma_b_params(n, alpha, M)
    top_k_sorted = np.sort(top_k_arr)[::-1]
    Lo = max(
        0.0,
        _solve_lower_piecewise(
            n, top_k_sorted, gamma, Lg, sum_s=sum_s, sum_s2=sum_s2
        ),
    )
  else:
    Lo = s_bar / (1.0 + c_cantelli(nu, M, alpha_p))

  if metric == "rmse":
    return (math.sqrt(max(0.0, Lo)), math.sqrt(max(0.0, U))), status
  theta_max = metrics.get_theta_max(metric)
  if theta_max is not None:
    U = min(U, theta_max)
  return (Lo, U), status
