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

"""Tests for the confidence-interval library: bound, interval, metrics, sketch."""

import dataclasses
import math
import os

from absl.testing import absltest
import numpy as np
import scipy.special
import scipy.stats

from dgf.src.stats.independent import bound
from dgf.src.stats.independent import interval
from dgf.src.stats.independent import metrics
from dgf.src.stats.independent import sketch

# Set DGF_STATS_LONG_TESTS=1 (blaze test --test_env=...) for the full-size runs.
_LONG = os.environ.get("DGF_STATS_LONG_TESTS") == "1"

_SEED = 20260921


def _rng(offset: int) -> np.random.Generator:
  """Independent, reproducible generator per test."""
  return np.random.default_rng(_SEED + offset)


def _bound_of(s, alpha, M, k=32, **kwargs):
  """Runs bound.bound on sample `s`, keeping the top `k` values (None: all)."""
  top = np.sort(s)[::-1]
  if k is not None:
    top = top[:k]
  return bound.bound(
      len(s), float(np.sum(s)), float(np.sum(s**2)), top, alpha, M, **kwargs
  )


def _upper_factor(n, M, alpha):
  """1 / (1 - c_upper) for the i.i.d. path."""
  c_up = min(
      bound.c_cantelli(n, M, alpha / 2.0), bound.c_bernstein(n, M, alpha / 2.0)
  )
  return 1.0 / (1.0 - c_up)


class BoundConstantsTest(absltest.TestCase):
  """Closed-form constants and thresholds of the bound."""

  def test_endpoints_reconstruct_sample_mean(self):
    # U (1 - c_up) == s_bar and, on the Lemma A' branch, L (1 + c_cant) == s_bar.
    rng = _rng(1)
    for _ in range(100):
      n = int(rng.integers(100, 1000))
      s = rng.exponential(scale=2.0, size=n)
      alpha = 0.05
      M = float(rng.uniform(2.0, 5.0))
      alpha_p = alpha / 2.0
      c_cant = bound.c_cantelli(n, M, alpha_p)
      c_up = min(c_cant, bound.c_bernstein(n, M, alpha_p))
      if c_up >= 1.0:
        continue
      s_bar = float(np.mean(s))
      (L, U), _ = _bound_of(s, alpha, M)
      self.assertLessEqual(abs(U * (1.0 - c_up) - s_bar) / s_bar, 1e-12)
      if not bound._use_lower_sweep(n, alpha, M, 32, None):
        self.assertLessEqual(abs(L * (1.0 + c_cant) - s_bar) / s_bar, 1e-12)

  def test_m_equal_one_collapses_to_sample_mean(self):
    rng = _rng(2)
    for _ in range(20):
      s = rng.uniform(1.0, 5.0, size=100)
      s_bar = float(np.mean(s))
      (L, U), _ = _bound_of(s, 0.05, 1.0)
      self.assertLessEqual(max(abs(L - s_bar), abs(U - s_bar)) / s_bar, 1e-12)

    self.assertEqual(bound.c_cantelli(100, 1.0, 0.025), 0.0)
    (lo, hi), st = bound.bound(100, 100.0, 100.0, np.ones(32), 0.05, 1.0)
    self.assertEqual(st, "UNREFUTED")
    self.assertTrue(math.isclose(hi, 1.0, abs_tol=1e-12))
    self.assertTrue(math.isclose(lo, 1.0, abs_tol=1e-12))

  def test_validity_thresholds_n_A_and_n_U(self):
    for M in (2.0, 4.0, 16.0):
      for alpha in (0.05, 0.1):
        nA = bound.n_A(M, alpha)
        nU = bound.n_U(M, alpha)
        alpha_p = alpha / 2.0
        # c_cantelli < 1 iff n > n_A.
        self.assertGreaterEqual(
            bound.c_cantelli(int(math.floor(nA - 0.5)), M, alpha_p), 1.0
        )
        self.assertLess(
            bound.c_cantelli(int(math.ceil(nA + 0.5)), M, alpha_p), 1.0
        )
        # bound refuses iff n <= n_U.
        n_below = int(math.floor(nU - 0.5))
        n_above = int(math.ceil(nU + 0.5))
        _, st_below = bound.bound(
            n_below,
            float(n_below),
            float(n_below * 2),
            np.ones(min(32, n_below)),
            alpha,
            M,
        )
        _, st_above = bound.bound(
            n_above,
            float(n_above),
            float(n_above * 2),
            np.ones(min(32, n_above)),
            alpha,
            M,
        )
        self.assertEqual(st_below, "ASSUMPTION_REQUIRED", (M, alpha))
        self.assertNotEqual(st_above, "ASSUMPTION_REQUIRED", (M, alpha))

  def test_bernstein_constant_solves_its_quadratic(self):
    for n in (50, 100, 500, 2000):
      for alpha in (0.1, 0.05, 0.01, 0.001):
        for M in (1.01, 2.0, 4.0, 16.0):
          L = math.log(2.0 / alpha)
          c = bound.c_bernstein(n, M, alpha / 2.0)
          residual = abs(n * c**2 - (2.0 / 3.0) * L * c - 2.0 * (M - 1.0) * L)
          self.assertLessEqual(residual, 1e-12, (n, alpha, M))

  def test_bernstein_constant_matches_bisection(self):
    for n in (20, 100, 1000, 10000):
      for alpha in (0.1, 0.05, 0.01, 0.001):
        for M in (1.5, 3.0, 16.0):
          L = math.log(2.0 / alpha)
          c_exact = bound.c_bernstein(n, M, alpha / 2.0)
          lo_b, hi_b = 0.0, 100.0
          for _ in range(60):
            mid = 0.5 * (lo_b + hi_b)
            if n * mid**2 - (2.0 / 3.0) * L * mid - 2.0 * (M - 1.0) * L < 0:
              lo_b = mid
            else:
              hi_b = mid
          self.assertLessEqual(abs(c_exact - lo_b) / c_exact, 1e-12)

  def test_bernstein_constant_reference_values(self):
    # c_bernstein at n = 100, alpha' = 0.025, to 4 decimals.
    ms = [1, 1.01, 1.1, 1.5, 2, 3, 4, 8, 16]
    expected = [
        0.0246,
        0.0421,
        0.0991,
        0.2048,
        0.2842,
        0.3966,
        0.4829,
        0.7310,
        1.0643,
    ]
    for m_val, exp_val in zip(ms, expected):
      c_val = bound.c_bernstein(100, m_val, 0.025)
      self.assertLessEqual(abs(round(c_val, 4) - exp_val), 1e-4, m_val)

  def test_n_U_reference_values(self):
    table = [
        (0.10, 4.0, 20),
        (0.10, 16.0, 92),
        (0.05, 4.0, 25),
        (0.05, 16.0, 114),
        (0.01, 4.0, 36),
        (0.01, 16.0, 163),
        (0.001, 4.0, 51),
        (0.001, 16.0, 234),
    ]
    for alpha, M, expected in table:
      self.assertEqual(int(math.ceil(bound.n_U(M, alpha))), expected)

  def test_lower_sweep_threshold_n_B(self):
    for M in (2.0, 4.0, 16.0):
      for alpha in (0.01, 0.001):
        nB = bound.n_B(M, alpha)
        _, _, slope_below = bound._lemma_b_params(  # pylint: disable=protected-access
            int(math.floor(nB)), alpha, M
        )
        _, _, slope_above = bound._lemma_b_params(  # pylint: disable=protected-access
            int(math.ceil(nB)) + 1, alpha, M
        )
        self.assertLessEqual(slope_below, 0.10, (M, alpha))
        self.assertGreater(slope_above, 0.10, (M, alpha))

  def test_upper_factor_monotone_in_n_and_M(self):
    for alpha in (0.05, 0.01):
      for M in (1.5, 4.0, 16.0):
        u = [_upper_factor(n, M, alpha) for n in (200, 400, 800, 1600, 5000)]
        for a, b in zip(u, u[1:]):
          self.assertLessEqual(b, a)
      for n in (500, 2000):
        u = [_upper_factor(n, M, alpha) for M in (1.1, 2.0, 4.0, 8.0, 16.0)]
        for a, b in zip(u, u[1:]):
          self.assertGreaterEqual(b, a)


class BoundTest(absltest.TestCase):
  """Behaviour of bound.bound."""

  def test_scale_invariance_at_alpha_0_05(self):
    rng = _rng(3)
    for kappa in (0.001, 1000.0):
      for _ in range(50):
        s = rng.gamma(shape=2.0, scale=1.5, size=200)
        (L1, U1), _ = _bound_of(s, 0.05, 3.0)
        (L2, U2), _ = _bound_of(s * kappa, 0.05, 3.0)
        self.assertLessEqual(abs(U2 - kappa * U1) / (kappa * U1), 1e-12)
        self.assertLessEqual(abs(L2 - kappa * L1) / (kappa * L1), 1e-12)

  def test_scale_invariance_at_alpha_0_01(self):
    rng = _rng(4)
    for kappa in (0.001, 1000.0):
      for _ in range(20):
        s = rng.exponential(scale=1.0, size=500)
        (L1, U1), _ = _bound_of(s, 0.01, 3.0)
        (L2, U2), _ = _bound_of(s * kappa, 0.01, 3.0)
        if math.isfinite(U1) and math.isfinite(U2) and U1 > 0:
          err_u = abs(U2 - kappa * U1) / (kappa * U1)
          err_l = (
              abs(L2 - kappa * L1) / (kappa * L1)
              if L1 > 0
              else abs(L2 - kappa * L1)
          )
          self.assertLessEqual(max(err_u, err_l), 1e-10)

  def test_sketch_upper_identical_and_sketch_lower_valid(self):
    # The upper endpoint depends only on (n, sum_s), so it is identical in
    # sketch and data mode. The lower endpoint is not: the data-mode Lemma B
    # infimum lives below s_(32)/gamma, which a top-32 sketch cannot evaluate,
    # so sketch mode uses Lemma A'. There only validity is asserted.
    rng = _rng(5)
    for dist in ("gaussian", "laplace", "lognormal"):
      for n in (100, 1000, 5000):
        for _ in range(40):
          if dist == "gaussian":
            s = rng.normal(size=n) ** 2
          elif dist == "laplace":
            s = np.abs(rng.laplace(size=n))
          else:
            s = np.exp(rng.normal(scale=0.5, size=n))
          s_bar = float(np.sum(s)) / n
          (L_sk, U_sk), st_sk = _bound_of(s, 0.01, 6.0, k=32)
          (_, U_dt), _ = _bound_of(s, 0.01, 6.0, k=None)
          if st_sk == "UNREFUTED" and math.isfinite(U_sk):
            self.assertLessEqual(abs(U_sk - U_dt), 1e-12)
            self.assertLessEqual(
                max(0.0 - L_sk, L_sk - s_bar, L_sk - U_sk), 1e-12
            )

  def test_lower_sweep_matches_brute_force_infimum(self):
    # Compares the piecewise solver against the definition
    # L = inf{t > 0 : H_lo(t) >= 0}, evaluated on a grid, including
    # zero-inflated samples (where the infimum can be exactly 0) and n = 5000
    # (where naive prefix-sum subtraction manufactures a spurious root near
    # 1e-11). Comparing sketch mode against data mode cannot catch either
    # failure, so the reference is the definition itself. The grid is
    # geometric so that it resolves the small-t region where the infimum lives.

    def h_lo_grid(grid, s, n, gamma, Lg):
      out = np.empty(len(grid), dtype=np.float64)
      block = max(1, 5_000_000 // max(1, n))
      for lo in range(0, len(grid), block):
        g = grid[lo : lo + block]
        x = np.minimum(s[None, :], (gamma * g)[:, None])
        xbar = x.mean(axis=1)
        vhat = np.maximum(0.0, x.var(axis=1, ddof=1))
        out[lo : lo + block] = (
            g
            - xbar
            + np.sqrt(2.0 * vhat * Lg / float(n))
            + (7.0 * gamma * g * Lg) / (3.0 * float(n - 1))
        )
      return out

    rng = _rng(6)
    for zero_frac in (0.0, 0.5, 0.9, 0.98):
      for n in (1000, 5000):
        for _ in range(2):
          s = np.zeros(n, dtype=np.float64)
          k_plus = max(1, int(round((1.0 - zero_frac) * n)))
          s[:k_plus] = rng.gamma(shape=2.0, scale=1.0, size=k_plus)

          alpha, M = 0.01, 4.0
          if not bound._use_lower_sweep(n, alpha, M, K=n, effective_n=None):  # pylint: disable=protected-access
            continue
          gamma, Lg, _ = bound._lemma_b_params(n, alpha, M)  # pylint: disable=protected-access
          L = bound._solve_lower_piecewise(n, np.sort(s)[::-1], gamma, Lg)  # pylint: disable=protected-access

          # Stage 1: geometric bracket.
          hi = 4.0 * (float(s.max()) / gamma + float(s.mean()))
          coarse = np.geomspace(hi * 1e-13, hi, 4000)
          ok = np.nonzero(h_lo_grid(coarse, s, n, gamma, Lg) >= 0.0)[0]
          if not ok.size:
            continue
          i = int(ok[0])
          if i == 0:
            grid_inf = 0.0
          else:
            # Stage 2: linear refinement inside the bracket.
            fine = np.linspace(coarse[i - 1], coarse[i], 4000)
            ok2 = np.nonzero(h_lo_grid(fine, s, n, gamma, Lg) >= 0.0)[0]
            grid_inf = float(fine[ok2[0]]) if ok2.size else float(coarse[i])

          # Too tight is a coverage failure, too loose a width loss; both are
          # bugs. 1e-4 relative is the grid's own resolution.
          self.assertLessEqual(
              abs(L - grid_inf) / max(1.0, grid_inf), 1e-4, (zero_frac, n)
          )

  def test_degenerate_inputs(self):
    # An all-zero sample at or below n_U must refuse. At M = 2, alpha = 0.05,
    # n_U = 9.84, so n = 5 is in the refusal region. Returning [0, 0] there
    # would fail coverage on the two-point law s in {0, h}: theta = h/M > 0
    # while U = 0, on (1 - 1/M)^n of draws.
    _, st = bound.bound(5, 0.0, 0.0, np.zeros(5), 0.05, 2.0)
    self.assertEqual(st, "ASSUMPTION_REQUIRED")
    # Above n_U the same sample gets [0, 0]: {s_bar = 0} is then contained in
    # the upper-endpoint miss event.
    iv, st = bound.bound(20, 0.0, 0.0, np.zeros(20), 0.05, 2.0)
    self.assertEqual(iv, (0.0, 0.0))
    self.assertEqual(st, "UNREFUTED")

    _, st = bound.bound(1, 1.0, 1.0, np.array([1.0]), 0.05, 2.0)
    self.assertEqual(st, "ASSUMPTION_REQUIRED")
    _, st = bound.bound(10, float("nan"), 1.0, np.ones(10), 0.05, 2.0)
    self.assertEqual(st, "ASSUMPTION_REQUIRED")
    _, st = bound.bound(10, 1.0, float("inf"), np.ones(10), 0.05, 2.0)
    self.assertEqual(st, "ASSUMPTION_REQUIRED")

  def test_endpoints_bracket_sample_mean(self):
    rng = _rng(7)
    for _ in range(100):
      s = rng.exponential(scale=2.0, size=200)
      s_bar = float(np.mean(s))
      for alpha in (0.05, 0.01):
        (L, U), st = _bound_of(s, alpha, 4.0)
        if (
            st != "ASSUMPTION_REQUIRED"
            and math.isfinite(U)
            and math.isfinite(L)
        ):
          self.assertLessEqual(L - s_bar, 1e-12)
          self.assertLessEqual(s_bar - U, 1e-12)

  def test_rmse_endpoints_are_sqrt_of_mse_endpoints(self):
    rng = _rng(8)
    for _ in range(50):
      s = rng.normal(size=300) ** 2
      for alpha in (0.05, 0.01):
        (L_m, U_m), _ = _bound_of(s, alpha, 3.0, metric="mse")
        (L_r, U_r), _ = _bound_of(s, alpha, 3.0, metric="rmse")
        self.assertLessEqual(abs(L_r - math.sqrt(max(0.0, L_m))), 1e-12)
        self.assertLessEqual(abs(U_r - math.sqrt(max(0.0, U_m))), 1e-12)

  def test_heavy_tail_sets_tail_unresolved(self):
    s = np.ones(500)
    s[0] = 50.0
    _, st = _bound_of(s, 0.05, 2.0)
    self.assertEqual(st, "TAIL_UNRESOLVED")

  def test_upper_coverage_on_extremal_two_point_law(self):
    # s in {0, M} with P(s = M) = 1/M attains E[s^2] = M theta^2 with theta = 1.
    # The upper miss rate must not exceed alpha/2 beyond Monte Carlo error.
    rng = _rng(9)
    M = 4.0
    reps, chunk = (20000, 5000) if _LONG else (500, 500)
    alphas = (0.05, 0.001) if _LONG else (0.05,)
    ns = (1000, 5000) if _LONG else (1000,)
    for alpha in alphas:
      for n in ns:
        alpha_p = alpha / 2.0
        factor = _upper_factor(n, M, alpha)
        misses = 0
        for _ in range(reps // chunk):
          draws = (rng.uniform(size=(chunk, n)) < 1.0 / M).astype(
              np.float64
          ) * M
          misses += int(np.sum(np.mean(draws, axis=1) * factor < 1.0))
        miss_rate = misses / reps
        self.assertLessEqual(
            miss_rate,
            alpha_p + 3.0 * math.sqrt(alpha_p * (1.0 - alpha_p) / reps),
            (alpha, n),
        )

  def test_lower_sweep_routing_depends_only_on_design(self):
    for n in (120, 200, 1000, 5000):
      for alpha in (0.1, 0.05, 0.01, 0.001):
        for M in (1.0, 4.0, 16.0):
          for eff in (None, float(n // 2)):
            for k in (32, n):
              _, _, slope_inf = bound._lemma_b_params(n, alpha, M)  # pylint: disable=protected-access
              expected = (
                  eff is None and k >= 1 and slope_inf > 0.10
              )
              self.assertEqual(
                  bound._use_lower_sweep(n, alpha, M, k, eff),  # pylint: disable=protected-access
                  expected,
              )

  def test_bound_is_bit_identical_to_reference_algorithm(self):
    # Algorithm 1 written out inline; bound.bound must match it bit for bit.
    idx = 0
    for n in (120, 200, 1000, 5000, 20000):
      for alpha in (0.1, 0.05, 0.01, 0.001):
        for M in (1.0, 1.5, 4.0, 16.0, 100.0):
          for mode in ("data", "sketch"):
            for correlated in (False, True):
              idx += 1
              rng = np.random.default_rng(20260922 + idx)
              s = rng.gamma(shape=2.0, scale=1.0, size=n)
              sum_s = float(np.sum(s))
              sum_s2 = float(np.sum(s**2))
              top_k = (
                  np.sort(s)[::-1] if mode == "data" else np.sort(s)[::-1][:32]
              )
              eff_n = float(n // 2) if correlated else None

              alpha_p = alpha / 2.0
              s_bar = sum_s / float(n)
              if eff_n is None:
                nu = float(n)
                c_up = min(
                    bound.c_cantelli(n, M, alpha_p),
                    bound.c_bernstein(n, M, alpha_p),
                )
              else:
                nu = float(eff_n)
                c_up = bound.c_cantelli(eff_n, M, alpha_p)

              if not c_up < 1.0:
                L_ref, U_ref, st_ref = (
                    float("nan"),
                    float("nan"),
                    "ASSUMPTION_REQUIRED",
                )
              elif sum_s == 0.0:
                L_ref, U_ref, st_ref = 0.0, 0.0, "UNREFUTED"
              else:
                U_ref = s_bar / (1.0 - c_up)
                M_hat = (float(n) * sum_s2) / (sum_s**2)
                st_ref = "TAIL_UNRESOLVED" if M_hat > M else "UNREFUTED"
                Lg = math.log(2.0 / alpha_p)
                gamma_star = math.sqrt(3.0 * M * float(n - 1) / (7.0 * Lg))
                gamma = 0.35 * gamma_star
                slope_inf = (
                    1.0 - (7.0 * gamma * Lg) / (3.0 * float(n - 1)) - M / gamma
                )
                use_sweep = (
                    eff_n is None
                    and len(top_k) >= 1
                    and slope_inf > 0.10
                )
                if use_sweep:
                  L_ref = max(
                      0.0,
                      bound._solve_lower_piecewise(  # pylint: disable=protected-access
                          n,
                          np.sort(top_k)[::-1],
                          gamma,
                          Lg,
                          sum_s=sum_s,
                          sum_s2=sum_s2,
                      ),
                  )
                else:
                  L_ref = s_bar / (1.0 + bound.c_cantelli(nu, M, alpha_p))

              (L_act, U_act), st_act = bound.bound(
                  n=n,
                  sum_s=sum_s,
                  sum_s2=sum_s2,
                  top_k=top_k,
                  alpha=alpha,
                  M=M,
                  metric="mse",
                  effective_n=eff_n,
              )
              config = (n, alpha, M, mode, correlated)
              self.assertEqual(st_act, st_ref, str(config))
              for ref, act in ((L_ref, L_act), (U_ref, U_act)):
                self.assertEqual(math.isnan(ref), math.isnan(act), str(config))
                if not math.isnan(ref):
                  self.assertEqual(float(act).hex(), float(ref).hex(), str(config))

  def test_sketch_lower_sweep_valid_and_tighter_than_cantelli(self):
    # Under L11, sketch mode runs the piecewise lower sweep whenever
    # slope_inf > 0.10. On non-degenerate distributions, L_sketch must be
    # valid (0 <= L_sketch <= s_bar <= U_sketch) and strictly tighter than Cantelli.
    rng = _rng(11)
    for dist in ("lognormal", "pareto"):
      for n in (1000, 2000):
        for _ in range(5):
          if dist == "lognormal":
            s = np.exp(rng.normal(scale=0.5, size=n))
          else:
            s = rng.pareto(a=3.0, size=n) + 1.0
          alpha, M = 0.01, 6.0
          s_bar = float(np.mean(s))
          c_cant = bound.c_cantelli(n, M, alpha / 2.0)
          L_cant = s_bar / (1.0 + c_cant)

          (L_dt, U_dt), st_dt = _bound_of(s, alpha, M, k=None)
          (L_sk, U_sk), st_sk = _bound_of(s, alpha, M, k=32)
          self.assertIn(st_sk, ("UNREFUTED", "TAIL_UNRESOLVED"))
          self.assertLessEqual(0.0, L_sk)
          self.assertLessEqual(L_sk, s_bar)
          self.assertLessEqual(s_bar, U_sk)
          self.assertLessEqual(L_sk, L_dt + 1e-12)
          if bound._use_lower_sweep(n, alpha, M, 32, None):
            self.assertGreater(L_sk, L_cant)
            gamma, Lg, _ = bound._lemma_b_params(n, alpha, M)
            sum_s = float(np.sum(s))
            sum_s2 = float(np.sum(s**2))
            top_32 = np.sort(s)[::-1][:32]
            grid = np.linspace(max(0.0, L_sk * 0.99), L_sk * 1.05 + 1e-4, 500)
            h_vals = np.array([
                bound.compute_H_lo(t_val, n, sum_s, sum_s2, top_32, gamma, Lg)
                for t_val in grid
            ])
            pos = np.nonzero(h_vals >= -1e-11)[0]
            if pos.size > 0:
              grid_inf = float(grid[pos[0]])
              self.assertLessEqual(L_sk, grid_inf + 1e-4 * max(1.0, grid_inf))

  def test_without_replacement_lower_sweep_coverage(self):
    # Under Claim 1″, uniform sampling without replacement from a finite pool
    # preserving P2*(M) maintains nominal coverage for both endpoints.
    rng = _rng(12)
    N_pool = 2000
    n = 1500
    alpha = 0.05
    M = 4.0
    self.assertTrue(bound._use_lower_sweep(n, alpha, M, 32, None))
    pool = rng.pareto(a=3.0, size=N_pool) + 1.0
    mu = float(np.mean(pool))

    reps = 200 if _LONG else 40
    covered = 0
    for _ in range(reps):
      sample = rng.choice(pool, size=n, replace=False)
      (L, U), st = _bound_of(sample, alpha, M, k=32)
      if st != "ASSUMPTION_REQUIRED":
        if L <= mu <= U:
          covered += 1
    coverage = covered / reps
    alpha_tol = 3.0 * math.sqrt(alpha * (1.0 - alpha) / reps)
    self.assertGreaterEqual(coverage, (1.0 - alpha) - alpha_tol)

  def test_sketch_piece_two_roots_returns_smallest(self):
    # Regression test: on concave sketch sub-pieces where
    # H~_lo has two valid roots, _solve_lower_piecewise must return the
    # smaller root (the infimum) rather than the larger root.
    n = 1000
    alpha = 0.05
    M = 4.0
    gamma, Lg, _ = bound._lemma_b_params(n, alpha, M)
    s_K = 0.001
    top_k = np.full(32, s_K, dtype=np.float64)
    s_rest = 0.00968
    q_rest = 1.063832e-07
    sum_s = float(np.sum(top_k)) + s_rest
    sum_s2 = float(np.sum(top_k**2)) + q_rest

    r_min = 7.50210875691869e-06
    r_max = 1.0161499210330397e-05

    # Verify that H~_lo is 0 at both r_min and r_max on the sub-piece
    h_min = bound.compute_H_lo(r_min, n, sum_s, sum_s2, top_k, gamma, Lg)
    h_max = bound.compute_H_lo(r_max, n, sum_s, sum_s2, top_k, gamma, Lg)
    self.assertAlmostEqual(h_min, 0.0, places=9)
    self.assertAlmostEqual(h_max, 0.0, places=5)

    # In the interior (r_min, r_max), H~_lo is strictly positive (concave)
    r_mid = 0.5 * (r_min + r_max)
    h_mid = bound.compute_H_lo(r_mid, n, sum_s, sum_s2, top_k, gamma, Lg)
    self.assertGreater(h_mid, 1e-10)

    # _solve_lower_piecewise must return the infimum r_min, NOT r_max
    L = bound._solve_lower_piecewise(n, top_k, gamma, Lg, sum_s=sum_s, sum_s2=sum_s2)
    self.assertAlmostEqual(L, r_min, places=9)
    self.assertNotAlmostEqual(L, r_max, places=6)


class IntervalTest(absltest.TestCase):
  """Behaviour of interval.confidence_interval."""

  def test_public_api(self):
    public = sorted(name for name in dir(interval) if not name.startswith("_"))
    self.assertEqual(public, ["Result", "Status", "confidence_interval"])

  def test_default_m_and_status_per_metric(self):
    rng = _rng(10)
    e = rng.normal(size=1000)
    r_mae = interval.confidence_interval(e, metric="mae", level=0.95)
    self.assertEqual(r_mae.m_declared, 4.0)
    self.assertEqual(r_mae.status, interval.Status.UNREFUTED)
    self.assertTrue(0 < r_mae.low < r_mae.high, r_mae)

    r_mse = interval.confidence_interval(e, metric="mse", level=0.95)
    self.assertEqual(r_mse.m_declared, 16.0)
    self.assertEqual(r_mse.status, interval.Status.UNREFUTED)
    self.assertTrue(0 < r_mse.low < r_mse.high, r_mse)

    r_rmse = interval.confidence_interval(e, metric="rmse", level=0.95)
    self.assertEqual(r_rmse.m_declared, 16.0)
    self.assertEqual(r_rmse.status, interval.Status.UNREFUTED)
    self.assertTrue(0 < r_rmse.low < r_rmse.high, r_rmse)
    self.assertLessEqual(abs(r_rmse.low - math.sqrt(r_mse.low)), 1e-12)
    self.assertLessEqual(abs(r_rmse.high - math.sqrt(r_mse.high)), 1e-12)

    y_true = rng.integers(0, 2, size=1000)
    y_pred = y_true.copy()
    y_pred[:50] = 1 - y_pred[:50]  # 95% accuracy
    r_acc = interval.confidence_interval(
        (y_true, y_pred), metric="accuracy", level=0.95
    )
    self.assertEqual(r_acc.m_declared, 4.0)
    self.assertEqual(r_acc.status, interval.Status.UNREFUTED)
    self.assertTrue(0.0 < r_acc.low < r_acc.high, r_acc)

    # n = 3000 gives 1500 disjoint pairs for the Var(y) sequence.
    x = rng.normal(scale=2.0, size=3000)
    y = x + rng.normal(scale=1.0, size=3000)
    r_r2 = interval.confidence_interval((y, x), metric="r2", level=0.95)
    self.assertEqual(r_r2.m_declared, 16.0)
    self.assertEqual(r_r2.status, interval.Status.UNREFUTED)
    self.assertTrue(r_r2.low < r_r2.high <= 1.0, r_r2)

  def test_average_precision_raises(self):
    with self.assertRaises(ValueError) as cm:
      interval.confidence_interval(np.ones(10), metric="average_precision")
    self.assertIn("1e5", str(cm.exception))
    self.assertIn("average_precision is not supported", str(cm.exception))

  def test_sketch_and_data_modes_agree(self):
    # Contract: the sketch either refuses, or it matches data mode exactly on
    # the status and the upper endpoint; the lower endpoint matches exactly when
    # data mode does not use the Lemma B sweep (both modes use Lemma A'), and
    # is merely valid when data mode uses the sweep (which a sketch cannot evaluate).
    rng = _rng(11)
    reps = 2000 if _LONG else 100
    for _ in range(reps):
      n = int(rng.choice([100, 500, 1500]))
      dist = rng.choice(["gaussian", "laplace", "exponential", "lognormal"])
      if dist == "gaussian":
        sample = rng.normal(size=n)
      elif dist == "laplace":
        sample = rng.laplace(size=n)
      elif dist == "exponential":
        sample = rng.exponential(size=n)
      else:
        sample = rng.lognormal(sigma=0.5, size=n)
      met = str(rng.choice(["mae", "mse", "rmse"]))
      lvl = float(rng.choice([0.95, 0.99]))
      res_data = interval.confidence_interval(sample, metric=met, level=lvl)
      res_sk = interval.confidence_interval(
          sketch.from_data(sample, metric=met), metric=met, level=lvl
      )
      if res_sk.status == interval.Status.ASSUMPTION_REQUIRED:
        continue
      self.assertEqual(res_sk.status, res_data.status)
      self.assertLessEqual(abs(res_sk.high - res_data.high), 1e-11)
      alpha = 1.0 - lvl
      if not bound._use_lower_sweep(
          n, alpha, res_data.m_declared, K=n, effective_n=None
      ):
        self.assertLessEqual(abs(res_sk.low - res_data.low), 1e-11)
      else:
        self.assertTrue(0.0 <= res_sk.low <= res_sk.high)

  def test_sketch_and_data_modes_agree_at_small_n(self):
    # n = 100 with the MSE default M = 16 sits between n_U(16, 0.1) = 91.9 and
    # n_U(16, 0.05) = 113.1: both modes must certify at alpha = 0.1 and both
    # must refuse at alpha <= 0.05.
    rng = _rng(12)
    reps = 2000 if _LONG else 100
    for _ in range(reps):
      e = rng.normal(size=100)
      sk = sketch.from_data(e, metric="mse")
      for alpha in (0.10, 0.05, 0.01, 0.001):
        r_dat = interval.confidence_interval(e, metric="mse", level=1.0 - alpha)
        r_sk = interval.confidence_interval(sk, metric="mse", level=1.0 - alpha)
        self.assertEqual(r_dat.status, r_sk.status)
        self.assertEqual(
            r_sk.status == interval.Status.ASSUMPTION_REQUIRED, alpha <= 0.05
        )
        if r_sk.status == interval.Status.ASSUMPTION_REQUIRED:
          continue
        self.assertLessEqual(abs(r_dat.high - r_sk.high), 1e-12)
        if not bound._use_lower_sweep(
            100, alpha, r_dat.m_declared, K=100, effective_n=None
        ):
          self.assertLessEqual(abs(r_dat.low - r_sk.low), 1e-12)
        else:
          self.assertTrue(0.0 <= r_sk.low <= r_sk.high)

  def test_refuses_below_threshold(self):
    rng = _rng(13)
    # n = 50 <= n_U(16, 0.01) = 162.5.
    sk_small = sketch.from_array(rng.normal(size=50) ** 2)
    res_small = interval.confidence_interval(
        sk_small, metric="mse", level=0.99, m=16.0
    )
    self.assertEqual(res_small.status, interval.Status.ASSUMPTION_REQUIRED)
    self.assertIsNone(res_small.level)
    # Correlated path: n_eff = 30 <= n_A(3, 0.05) = 78.
    res_cor = interval.confidence_interval(
        rng.normal(size=500) ** 2,
        metric="mse",
        level=0.95,
        m=3.0,
        effective_n=30.0,
    )
    self.assertEqual(res_cor.status, interval.Status.ASSUMPTION_REQUIRED)
    self.assertIsNone(res_cor.level)

  def test_edge_cases(self):
    rng = _rng(14)
    for data, metric in (
        ([1.0, 2.0, float("inf")], "mae"),
        ([1.0, float("nan"), 3.0], "mse"),
        ([], "rmse"),
        ([2.5], "rmse"),
    ):
      r = interval.confidence_interval(data, metric=metric)
      self.assertEqual(r.status, interval.Status.ASSUMPTION_REQUIRED, data)
      self.assertIsNone(r.level, data)

    # All-zero sample: refuses at n = 4 <= n_U(4, 0.05) = 24.6 ...
    r = interval.confidence_interval([0.0] * 4, metric="mae")
    self.assertEqual(r.status, interval.Status.ASSUMPTION_REQUIRED)
    # ... and is exactly [0, 0] above it.
    r = interval.confidence_interval([0.0] * 200, metric="mae")
    self.assertEqual(r.status, interval.Status.UNREFUTED)
    self.assertEqual((r.low, r.high), (0.0, 0.0))

    # Constant y for R^2 (Var(y) = 0): the interval is [-inf, 1].
    y_const = np.full(3000, 5.0)
    r = interval.confidence_interval(
        (y_const, y_const + rng.normal(scale=0.1, size=3000)),
        metric="r2",
        level=0.95,
    )
    self.assertEqual(r.status, interval.Status.UNREFUTED)
    self.assertEqual(r.low, float("-inf"))
    self.assertEqual(r.high, 1.0)

    # Fewer than 32 values.
    r = interval.confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0], metric="mae")
    self.assertIsInstance(r.status, interval.Status)

    # All values tied.
    r = interval.confidence_interval([2.0] * 200, metric="mae")
    self.assertEqual(r.status, interval.Status.UNREFUTED)
    self.assertTrue(math.isfinite(r.high))

  def test_edge_cases_with_effective_n(self):
    for eff in (None, 50.0):
      r = interval.confidence_interval(
          [1.0], metric="mae", effective_n=1.0 if eff is not None else None
      )
      self.assertEqual(r.status, interval.Status.ASSUMPTION_REQUIRED)
      r = interval.confidence_interval(
          [float("inf")] * 100, metric="mae", effective_n=eff
      )
      self.assertEqual(r.status, interval.Status.ASSUMPTION_REQUIRED)
      r = interval.confidence_interval(
          [float("nan")] * 100, metric="mae", effective_n=eff
      )
      self.assertEqual(r.status, interval.Status.ASSUMPTION_REQUIRED)
      y_c = np.ones(200)
      r = interval.confidence_interval((y_c, y_c), metric="r2", effective_n=eff)
      self.assertIn(
          r.status,
          (interval.Status.ASSUMPTION_REQUIRED, interval.Status.UNREFUTED),
      )

    # All-zero samples. i.i.d. path above n_U answers [0, 0].
    r = interval.confidence_interval([0.0] * 500, metric="mse")
    self.assertEqual(r.status, interval.Status.UNREFUTED)
    self.assertEqual((r.low, r.high), (0.0, 0.0))
    # Correlated path: n_eff = 50 <= n_A(16, 0.05) = 585 refuses ...
    r = interval.confidence_interval(
        [0.0] * 500, metric="mse", effective_n=50.0
    )
    self.assertEqual(r.status, interval.Status.ASSUMPTION_REQUIRED)
    # ... and n_eff = 800 > 585 answers [0, 0].
    r = interval.confidence_interval(
        [0.0] * 1000, metric="mse", effective_n=800.0
    )
    self.assertEqual(r.status, interval.Status.UNREFUTED)
    self.assertEqual((r.low, r.high), (0.0, 0.0))

  def test_level_plumbing_and_monotone_width(self):
    data = _rng(15).normal(size=2000)
    prev_width = 0.0
    for lvl in (0.9, 0.95, 0.99, 0.999):
      r = interval.confidence_interval(data, metric="mae", level=lvl)
      self.assertEqual(r.level, lvl)
      self.assertEqual(r.status, interval.Status.UNREFUTED)
      self.assertGreater(r.high - r.low, prev_width)
      prev_width = r.high - r.low
    for bad in (0.0, 1.0, -0.5, 1.5):
      with self.assertRaises(ValueError):
        interval.confidence_interval(data, metric="mae", level=bad)

  def test_invalid_m_raises(self):
    # NaN used to pass the `m < 1.0` check and reach the bound.
    data = _rng(16).normal(size=2000)
    ci = interval.confidence_interval
    y = _rng(17).normal(size=2000)
    for bad in (0.5, -1.0, float("nan"), float("inf"), float("-inf")):
      with self.assertRaisesRegex(ValueError, "m must be finite and >= 1.0"):
        ci(data, metric="mae", m=bad)
      with self.assertRaisesRegex(ValueError, "m must be finite and >= 1.0"):
        ci(sketch.from_data(data, "mse"), metric="mse", m=bad)
      with self.assertRaisesRegex(ValueError, "m must be finite and >= 1.0"):
        ci((y, y + data), metric="r2", m=bad)

  def test_accuracy_upper_endpoint_clamped_to_one(self):
    # Near-ceiling accuracy at small n: the unclamped upper endpoint is > 1.
    s = (np.random.default_rng(20260922 + 36).uniform(size=120) < 0.99).astype(
        np.float64
    )
    r = interval.confidence_interval(s, metric="accuracy", level=0.95)
    self.assertLessEqual(r.high, 1.0 + 1e-12)
    for p in (0.5, 0.9, 0.99):
      for n in (120, 500, 5000):
        for alpha in (0.05, 0.01):
          rng = np.random.default_rng(int(p * 1000 + n + alpha * 10000))
          s = (rng.uniform(size=n) < p).astype(np.float64)
          r = interval.confidence_interval(
              s, metric="accuracy", level=1 - alpha
          )
          if r.status != interval.Status.ASSUMPTION_REQUIRED:
            self.assertLessEqual(r.high, 1.0 + 1e-12)

  def test_accuracy_lower_endpoint_in_unit_interval(self):
    for p in (0.01, 0.5, 0.99):
      for n in (120, 500, 5000):
        for alpha in (0.05, 0.01):
          rng = np.random.default_rng(int(p * 1000 + n + alpha * 10000) + 7)
          s = (rng.uniform(size=n) < p).astype(np.float64)
          r = interval.confidence_interval(
              s, metric="accuracy", level=1 - alpha
          )
          if r.status != interval.Status.ASSUMPTION_REQUIRED:
            self.assertTrue(0.0 <= r.low <= 1.0 and r.low <= r.high, r)

  def test_unbounded_metrics_not_clamped(self):
    for metric in ("mae", "mse", "rmse"):
      for n in (120, 500, 5000):
        for alpha in (0.05, 0.01):
          e = np.random.default_rng(n + int(alpha * 10000)).normal(size=n) * 2.0
          r = interval.confidence_interval(e, metric=metric, level=1.0 - alpha)
          s = metrics.extract_summands(e, metric)
          (lo, hi), _ = _bound_of(
              s, alpha, metrics.get_default_m(metric), k=None, metric=metric
          )
          if r.status != interval.Status.ASSUMPTION_REQUIRED:
            self.assertLessEqual(abs(r.low - lo), 1e-12)
            self.assertLessEqual(abs(r.high - hi), 1e-12)


class EffectiveNTest(absltest.TestCase):
  """The correlated-data path."""

  def test_invalid_effective_n_raises(self):
    for bad in (150.0, 0.5, -2.0, float("nan"), float("inf")):
      with self.assertRaises(ValueError):
        sketch.IndependentSketch(100, 10.0, 20.0, (), effective_n=bad)
      with self.assertRaises(ValueError):
        interval.confidence_interval(
            np.ones(100), metric="mae", effective_n=bad
        )

  def test_r2_with_effective_n_refuses(self):
    r = interval.confidence_interval(
        (np.ones(100), np.ones(100)), metric="r2", effective_n=50.0
    )
    self.assertEqual(r.status, interval.Status.ASSUMPTION_REQUIRED)
    sk_a = sketch.from_array(np.ones(100), effective_n=50.0)
    sk_b = sketch.from_array(np.ones(50), effective_n=25.0)
    r = interval.confidence_interval(sketch.R2Sketch(sk_a, sk_b), metric="r2")
    self.assertEqual(r.status, interval.Status.ASSUMPTION_REQUIRED)

  def test_r2_sketch_with_effective_n_raises(self):
    with self.assertRaisesRegex(ValueError, "(?i)r2"):
      sketch.from_data(
          (np.ones(100), np.ones(100)), metric="r2", effective_n=50.0
      )

  def test_effective_n_equal_to_n_is_wider_than_iid(self):
    # With effective_n set, only Cantelli is used (Bernstein needs
    # independence), so even effective_n = n is wider.
    rng = _rng(16)
    r_iid = interval.confidence_interval(rng.normal(size=1000), metric="mse")
    r_neff = interval.confidence_interval(
        rng.normal(size=1000), metric="mse", effective_n=1000.0
    )
    self.assertGreater(r_neff.high - r_neff.low, r_iid.high - r_iid.low)
    self.assertGreater(r_neff.high, r_iid.high)

  def test_effective_n_ignores_top_values(self):
    clean = sketch.IndependentSketch(
        5000, 2500.0, 5000.0, tuple(range(32, 0, -1)), effective_n=2000.0
    )
    corrupt = sketch.IndependentSketch(
        5000, 2500.0, 5000.0, tuple([9999.0] * 32), effective_n=2000.0
    )
    r_clean = interval.confidence_interval(clean, metric="mse")
    r_corrupt = interval.confidence_interval(corrupt, metric="mse")
    self.assertEqual(r_clean.status, interval.Status.UNREFUTED)
    self.assertTrue(math.isclose(r_clean.low, r_corrupt.low, rel_tol=1e-12))
    self.assertTrue(math.isclose(r_clean.high, r_corrupt.high, rel_tol=1e-12))

  def test_effective_n_argument_conflicts_with_sketch(self):
    s_none = sketch.IndependentSketch(100, 1.0, 1.0, (), effective_n=None)
    s_set = sketch.IndependentSketch(100, 1.0, 1.0, (), effective_n=50.0)
    with self.assertRaises(ValueError):
      interval.confidence_interval(s_none, metric="mse", effective_n=50.0)
    with self.assertRaises(ValueError):
      interval.confidence_interval(s_set, metric="mse", effective_n=50.0)


class MergeTest(absltest.TestCase):
  """Sketch merge algebra."""

  def test_merge_is_associative_and_matches_whole_sample(self):
    rng = _rng(17)
    for _ in range(50):
      data = rng.exponential(scale=2.0, size=600)
      sk1 = sketch.from_array(data[:200])
      sk2 = sketch.from_array(data[200:400])
      sk3 = sketch.from_array(data[400:])
      m_12_3 = sketch.merge(sketch.merge(sk1, sk2), sk3)
      m_1_23 = sketch.merge(sk1, sketch.merge(sk2, sk3))
      self.assertEqual(m_12_3, m_1_23)
      whole = sketch.from_array(data)
      self.assertEqual(m_12_3.n, whole.n)
      self.assertEqual(m_12_3.top_32, whole.top_32)
      self.assertLessEqual(abs(m_12_3.sum_s - whole.sum_s), 1e-10)
      self.assertLessEqual(abs(m_12_3.sum_s2 - whole.sum_s2), 1e-10)

    r2_1 = sketch.R2Sketch(sketch_a=sk1, sketch_b=sk2)
    r2_2 = sketch.R2Sketch(sketch_a=sk2, sketch_b=sk3)
    r2_3 = sketch.R2Sketch(sketch_a=sk3, sketch_b=sk1)
    self.assertEqual(
        sketch.merge(sketch.merge(r2_1, r2_2), r2_3),
        sketch.merge(r2_1, sketch.merge(r2_2, r2_3)),
    )

  def test_r2_sketch_shard_behaviour(self):
    # Pairing is index-local, so an odd shard drops its last element. This is
    # documented, valid, and NOT shard-invariant; both halves are pinned.
    rng = _rng(18)
    y = rng.normal(size=601)
    p = rng.normal(size=601)

    def merged(cuts):
      out = None
      for lo, hi in cuts:
        part = sketch.from_data((y[lo:hi], p[lo:hi]), metric="r2")
        out = part if out is None else sketch.merge(out, part)
      return out

    # Even shards: invariant.
    self.assertEqual(
        merged([(0, 200), (200, 400), (400, 600)]),
        sketch.from_data((y[:600], p[:600]), metric="r2"),
    )
    # Odd shards: exactly one pair lost per odd shard, and nothing else.
    odd_cuts = [(0, 201), (201, 402), (402, 601)]
    m_odd = merged(odd_cuts)
    whole = sketch.from_data((y, p), metric="r2")
    self.assertIsInstance(m_odd, sketch.R2Sketch)
    self.assertIsInstance(whole, sketch.R2Sketch)
    self.assertEqual(m_odd.sketch_a, whole.sketch_a)
    self.assertEqual(
        m_odd.sketch_b.n, sum((hi - lo) // 2 for lo, hi in odd_cuts)
    )
    self.assertEqual(whole.sketch_b.n, 601 // 2)

  def test_schema_version_mismatch_raises(self):
    with self.assertRaises(ValueError):
      sketch.merge(
          sketch.IndependentSketch(100, 1.0, 1.0, (), schema_version=1),
          sketch.IndependentSketch(100, 1.0, 1.0, (), schema_version=2),
      )

  def test_effective_n_merges_conservative_kappa_max(self):
    rng = _rng(19)
    for _ in range(50):
      # 3 shards of unequal kappa
      n_vals = [100, 200, 300]
      eff_vals = [25.0, 40.0, 50.0]  # kappas: 4.0, 5.0, 6.0
      shards = [
          sketch.IndependentSketch(n, 1.0, 1.0, (), effective_n=eff)
          for n, eff in zip(n_vals, eff_vals)
      ]
      kappas = [float(s.n) / float(s.effective_n or 1.0) for s in shards]
      max_kappa = max(kappas)
      total_n = sum(s.n for s in shards)
      expected_eff = float(total_n) / max_kappa

      # Associativity: (s0 + s1) + s2 == s0 + (s1 + s2)
      m_assoc_1 = sketch.merge(sketch.merge(shards[0], shards[1]), shards[2])
      m_assoc_2 = sketch.merge(shards[0], sketch.merge(shards[1], shards[2]))
      self.assertEqual(m_assoc_1.effective_n, m_assoc_2.effective_n)

      assert m_assoc_1.effective_n is not None
      # Merged kappa equals max kappa_k exactly
      merged_kappa = float(m_assoc_1.n) / m_assoc_1.effective_n
      self.assertAlmostEqual(merged_kappa, max_kappa, places=12)
      self.assertAlmostEqual(m_assoc_1.effective_n, expected_eff, places=12)

      # Commutativity across all permutations
      for perm_idx in ([1, 0, 2], [2, 0, 1], [2, 1, 0]):
        m_perm = sketch.merge(
            sketch.merge(shards[perm_idx[0]], shards[perm_idx[1]]),
            shards[perm_idx[2]],
        )
        assert m_perm.effective_n is not None
        self.assertAlmostEqual(m_assoc_1.effective_n, m_perm.effective_n, places=12)

  def test_counterexample_conservative_merge(self):
    # Shard 1: n1 = 50_000, 100 blocks of 500 identical copies, E[S] = 1, Var(S) = 3
    # Shard 2: n2 = 950_000 i.i.d., E[S2] = 0.001, Var(S2) = 1e-6 * 3
    n1, n_eff1 = 50_000, 100.0  # kappa1 = 500
    n2, n_eff2 = 950_000, 950_000.0  # kappa2 = 1
    s1 = sketch.IndependentSketch(n1, 50_000.0, 200_000.0, (), effective_n=n_eff1)
    s2 = sketch.IndependentSketch(n2, 950.0, 3.8, (), effective_n=n_eff2)
    merged = sketch.merge(s1, s2)

    # Analytical true variance and true max effective_n for M = 80
    n = n1 + n2
    var_s_bar = (n1 / n) ** 2 * (3.0 / 100.0) + (n2 / n) ** 2 * (3e-6 / n2)
    theta = (50_000.0 + 950.0) / float(n)
    m_decl = 80.0
    true_n_eff = (m_decl - 1.0) * (theta**2) / var_s_bar

    # Merged effective_n must be <= true_n_eff (conservative)
    self.assertIsNotNone(merged.effective_n)
    self.assertLessEqual(merged.effective_n, true_n_eff)
    self.assertAlmostEqual(merged.effective_n, float(n) / 500.0, places=9)

    # Reasoning check: under old W-space rule, old_n_eff was ~38536 > true_n_eff (~2735)
    w_old = (float(n1) ** 2 / n_eff1) + (float(n2) ** 2 / n_eff2)
    old_n_eff = float(n) ** 2 / w_old
    self.assertGreater(old_n_eff, true_n_eff)

  def test_merged_effective_n_stays_in_range(self):
    rng = _rng(20)
    for _ in range(100):
      s1 = sketch.IndependentSketch(
          int(rng.integers(10, 1000)),
          1.0,
          1.0,
          (),
          effective_n=float(rng.uniform(1.0, 10.0)),
      )
      s2 = sketch.IndependentSketch(
          int(rng.integers(10, 1000)),
          1.0,
          1.0,
          (),
          effective_n=float(rng.uniform(1.0, 10.0)),
      )
      m = sketch.merge(s1, s2)
      self.assertTrue(1.0 <= m.effective_n <= m.n)

  def test_merge_on_cluster_boundaries_matches_whole(self):
    rng = _rng(21)
    n, m, rho = 1000, 20, 0.5
    z = rng.normal(size=n // m)
    u = rng.normal(size=(n // m, m))
    e = (math.sqrt(rho) * z[:, None] + math.sqrt(1.0 - rho) * u).reshape(-1)
    design_effect = 1.0 + (m - 1) * rho**2
    whole = sketch.from_data(e, metric="mse", effective_n=n / design_effect)
    merged = None
    for k in range(5):  # 5 shards of 4 whole clusters each
      part = sketch.from_data(
          e[k * 200 : (k + 1) * 200],
          metric="mse",
          effective_n=200.0 / design_effect,
      )
      merged = part if merged is None else sketch.merge(merged, part)
    assert isinstance(whole, sketch.IndependentSketch) and whole.effective_n is not None
    assert isinstance(merged, sketch.IndependentSketch) and merged.effective_n is not None
    self.assertLessEqual(abs(whole.effective_n - merged.effective_n), 1e-10)
    # m = 3 so that n_eff = 173.9 > n_A(3, 0.05) = 78 and both certify.
    r_whole = interval.confidence_interval(whole, metric="mse", m=3.0)
    r_merged = interval.confidence_interval(merged, metric="mse", m=3.0)
    self.assertTrue(
        math.isclose(r_whole.low, r_merged.low, rel_tol=1e-8, abs_tol=1e-10)
    )
    self.assertTrue(
        math.isclose(r_whole.high, r_merged.high, rel_tol=1e-8, abs_tol=1e-10)
    )

  def test_merge_refuses_to_mix_iid_and_correlated(self):
    s_none = sketch.IndependentSketch(100, 1.0, 1.0, (), effective_n=None)
    s_set = sketch.IndependentSketch(100, 1.0, 1.0, (), effective_n=50.0)
    with self.assertRaises(ValueError):
      sketch.merge(s_none, s_set)
    self.assertIsNone(sketch.merge(s_none, s_none).effective_n)
    self.assertIsNotNone(sketch.merge(s_set, s_set).effective_n)

  def test_splitting_a_cluster_across_shards_is_optimistic(self):
    # Documents the merge contract's failure mode: two shards that each hold
    # half of every cluster report n_eff = 250 each, and the kappa-max merge
    # yields 500 -- far above the true n / (1 + (m - 1) rho^2) = 30.9.
    true_neff = 1000 / (1.0 + (50 - 1) * 0.8**2)
    merged = sketch.merge(
        sketch.IndependentSketch(500, 1.0, 1.0, (), effective_n=250.0),
        sketch.IndependentSketch(500, 1.0, 1.0, (), effective_n=250.0),
    )
    self.assertGreater(merged.effective_n, true_neff)


class InvalidInputSemanticsTest(absltest.TestCase):
  """Non-finite inputs poison to ASSUMPTION_REQUIRED; out-of-domain values raise ValueError."""

  def test_accuracy_nan_prediction_float_tuple(self):
    # Accuracy with one NaN prediction in a float tuple returns ASSUMPTION_REQUIRED
    # in data, from_data, and merged-sketch mode.
    rng = np.random.default_rng(123)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    preds = labels.copy()
    preds[10] = np.nan
    data = (labels, preds)

    # 1. data mode
    res_data = interval.confidence_interval(data, metric="accuracy")
    self.assertEqual(res_data.status, interval.Status.ASSUMPTION_REQUIRED)

    # 2. from_data mode
    sk = sketch.from_data(data, metric="accuracy")
    assert isinstance(sk, sketch.IndependentSketch)
    self.assertTrue(math.isnan(sk.sum_s))
    self.assertTrue(math.isnan(sk.sum_s2))
    res_sketch = interval.confidence_interval(sk, metric="accuracy")
    self.assertEqual(res_sketch.status, interval.Status.ASSUMPTION_REQUIRED)

    # 3. merged-sketch mode
    sk1 = sketch.from_data((labels[:100], preds[:100]), metric="accuracy")
    sk2 = sketch.from_data((labels[100:], preds[100:]), metric="accuracy")
    sk_merged = sketch.merge(sk1, sk2)
    res_merged = interval.confidence_interval(sk_merged, metric="accuracy")
    self.assertEqual(res_merged.status, interval.Status.ASSUMPTION_REQUIRED)

  def test_accuracy_1d_containing_two(self):
    # Accuracy 1-D containing 2.0 raises ValueError in data and from_data modes.
    n = 200
    arr = np.ones(n, dtype=np.float64)
    arr[5] = 2.0

    with self.assertRaisesRegex(
        ValueError, "accuracy expects 0/1 correctness values"
    ):
      interval.confidence_interval(arr, metric="accuracy")

    with self.assertRaisesRegex(
        ValueError, "accuracy expects 0/1 correctness values"
    ):
      sketch.from_data(arr, metric="accuracy")

  def test_accuracy_1d_containing_minus_one(self):
    # Accuracy 1-D containing -1.0 raises ValueError in data and from_data modes.
    n = 200
    arr = np.ones(n, dtype=np.float64)
    arr[5] = -1.0

    with self.assertRaisesRegex(
        ValueError, "accuracy expects 0/1 correctness values"
    ):
      interval.confidence_interval(arr, metric="accuracy")

    with self.assertRaisesRegex(
        ValueError, "accuracy expects 0/1 correctness values"
    ):
      sketch.from_data(arr, metric="accuracy")

  def test_accuracy_fractional_inputs_raise_value_error(self):
    for bad_val in (0.1, 1.0 - 1e-7, 1.0 + 1e-5):
      arr = np.array([1.0, 0.0, bad_val, 1.0])
      with self.assertRaisesRegex(
          ValueError, "accuracy expects 0/1 correctness values"
      ):
        interval.confidence_interval(arr, metric="accuracy")
      with self.assertRaisesRegex(
          ValueError, "accuracy expects 0/1 correctness values"
      ):
        sketch.from_data(arr, metric="accuracy")

  def test_mae_inf_residual(self):
    # MAE with one inf residual returns ASSUMPTION_REQUIRED in data, from_data, and merged-sketch mode.
    n = 200
    e = np.ones(n, dtype=np.float64)
    e[7] = np.inf

    res_data = interval.confidence_interval(e, metric="mae")
    self.assertEqual(res_data.status, interval.Status.ASSUMPTION_REQUIRED)

    sk = sketch.from_data(e, metric="mae")
    assert isinstance(sk, sketch.IndependentSketch)
    self.assertTrue(math.isnan(sk.sum_s))
    res_sketch = interval.confidence_interval(sk, metric="mae")
    self.assertEqual(res_sketch.status, interval.Status.ASSUMPTION_REQUIRED)

    sk1 = sketch.from_data(e[:100], metric="mae")
    sk2 = sketch.from_data(e[100:], metric="mae")
    sk_merged = sketch.merge(sk1, sk2)
    res_merged = interval.confidence_interval(sk_merged, metric="mae")
    self.assertEqual(res_merged.status, interval.Status.ASSUMPTION_REQUIRED)

  def test_r2_nan_label(self):
    # R^2 with one NaN label returns ASSUMPTION_REQUIRED in data, from_data, and merged-sketch mode.
    n = 200
    y = np.linspace(1.0, 10.0, n)
    y[3] = np.nan
    y_pred = y + 0.1
    data = (y, y_pred)

    res_data = interval.confidence_interval(data, metric="r2")
    self.assertEqual(res_data.status, interval.Status.ASSUMPTION_REQUIRED)

    sk = sketch.from_data(data, metric="r2")
    assert isinstance(sk, sketch.R2Sketch)
    self.assertTrue(math.isnan(sk.sketch_a.sum_s))
    res_sketch = interval.confidence_interval(sk, metric="r2")
    self.assertEqual(res_sketch.status, interval.Status.ASSUMPTION_REQUIRED)

    sk1 = sketch.from_data((y[:100], y_pred[:100]), metric="r2")
    sk2 = sketch.from_data((y[100:], y_pred[100:]), metric="r2")
    sk_merged = sketch.merge(sk1, sk2)
    res_merged = interval.confidence_interval(sk_merged, metric="r2")
    self.assertEqual(res_merged.status, interval.Status.ASSUMPTION_REQUIRED)

  def test_from_array_negative_value(self):
    # from_array with one negative value returns ASSUMPTION_REQUIRED in data, from_data and merged-sketch mode.
    n = 200
    arr = np.ones(n, dtype=np.float64)
    arr[12] = -0.5

    # raw summand sketch via from_array
    sk = sketch.from_array(arr)
    self.assertTrue(math.isnan(sk.sum_s))
    self.assertTrue(math.isnan(sk.sum_s2))
    self.assertEqual(sk.n, n)
    res_sk = interval.confidence_interval(sk, metric="mse")
    self.assertEqual(res_sk.status, interval.Status.ASSUMPTION_REQUIRED)

    # from_data and data mode with accuracy metric raise ValueError on negative value
    with self.assertRaisesRegex(
        ValueError, "accuracy expects 0/1 correctness values"
    ):
      sketch.from_data(arr, metric="accuracy")
    with self.assertRaisesRegex(
        ValueError, "accuracy expects 0/1 correctness values"
    ):
      interval.confidence_interval(arr, metric="accuracy")

    # merged mode
    sk1 = sketch.from_array(arr[:100])
    sk2 = sketch.from_array(arr[100:])
    sk_m = sketch.merge(sk1, sk2)
    res_m = interval.confidence_interval(sk_m, metric="mse")
    self.assertEqual(res_m.status, interval.Status.ASSUMPTION_REQUIRED)

  def test_merge_clean_and_poisoned_sketch(self):
    # A merge of one clean and one poisoned sketch returns ASSUMPTION_REQUIRED.
    clean = sketch.from_array(np.ones(100, dtype=np.float64))
    poisoned = sketch.from_array(
        np.array([1.0, -2.0, 3.0] * 30, dtype=np.float64)
    )
    self.assertTrue(math.isnan(poisoned.sum_s))
    merged = sketch.merge(clean, poisoned)
    self.assertTrue(math.isnan(merged.sum_s))
    self.assertTrue(math.isnan(merged.sum_s2))
    res = interval.confidence_interval(merged, metric="mse")
    self.assertEqual(res.status, interval.Status.ASSUMPTION_REQUIRED)

  def test_integer_label_accuracy_exact(self):
    # Integer-label accuracy on the independent path is the exact binomial
    # interval. (Before 1.1.0 this input gave [0.7079, 1.0].)
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1] * 100, dtype=np.int32)
    preds = labels.copy()
    preds[:50] = 1 - preds[:50]  # 950 correct out of 1000
    r_acc = interval.confidence_interval(
        (labels, preds), metric="accuracy", level=0.95
    )
    self.assertEqual(r_acc.status, interval.Status.UNREFUTED)
    self.assertEqual(r_acc.m_declared, 4.0)
    alpha_tilde_half = (0.05 - (0.05**2) / 4.0) / 2.0
    self.assertAlmostEqual(
        r_acc.low, scipy.stats.beta.ppf(alpha_tilde_half, 950, 51), places=10
    )
    self.assertAlmostEqual(
        r_acc.high,
        scipy.stats.beta.ppf(1.0 - alpha_tilde_half, 951, 50),
        places=10,
    )


class ExactAccuracyTest(absltest.TestCase):
  """Accuracy on the independent path uses the exact binomial interval."""

  def test_per_side_coverage_by_enumeration(self):
    # Exact, not simulated: sum the Binomial pmf over the counts whose interval
    # misses theta, separately for each side.
    for n in (1, 7, 50, 400):
      for level in (0.9, 0.95, 0.99, 0.999):
        alpha = 1.0 - level
        ends = []
        for k in range(n + 1):
          res = interval.confidence_interval(
              np.array([1.0] * k + [0.0] * (n - k)),
              metric="accuracy",
              level=level,
          )
          self.assertEqual(res.status, interval.Status.UNREFUTED)
          ends.append((res.low, res.high))
        for theta in np.linspace(0.0, 1.0, 101):
          pmf = scipy.stats.binom.pmf(np.arange(n + 1), n, theta)
          miss_low = sum(p for p, (lo, _) in zip(pmf, ends) if theta < lo)
          miss_high = sum(p for p, (_, hi) in zip(pmf, ends) if theta > hi)
          self.assertLessEqual(miss_low, alpha / 2 + 1e-12, (n, level, theta))
          self.assertLessEqual(miss_high, alpha / 2 + 1e-12, (n, level, theta))

  def test_edge_counts(self):
    all_wrong = interval.confidence_interval(np.zeros(20), metric="accuracy")
    self.assertEqual(all_wrong.low, 0.0)
    alpha_tilde_half = (0.05 - (0.05**2) / 4.0) / 2.0
    self.assertAlmostEqual(
        all_wrong.high, 1.0 - alpha_tilde_half ** (1.0 / 20), places=12
    )
    all_right = interval.confidence_interval(np.ones(20), metric="accuracy")
    self.assertAlmostEqual(
        all_right.low, alpha_tilde_half ** (1.0 / 20), places=12
    )
    self.assertEqual(all_right.high, 1.0)

  def test_hypergeometric_enumeration(self):
    """Verifies that accuracy interval maintains coverage under sampling without replacement."""
    alphas = (0.5, 0.2, 0.1, 0.0995, 0.099, 0.05, 0.0499, 0.049, 0.01, 0.001)
    n_pools = (10, 20, 50, 100)
    # Precompute intervals per (n, alpha) for n in 1..100
    cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    for alpha in alphas:
      level = 1.0 - alpha
      alpha_tilde = alpha - (alpha**2) / 4.0
      for n in range(1, 101):
        k_vals = np.arange(n + 1)
        lows = np.zeros(n + 1, dtype=np.float64)
        highs = np.ones(n + 1, dtype=np.float64)
        if n > 0:
          lows[1:] = scipy.special.betaincinv(
              k_vals[1:], n - k_vals[1:] + 1, alpha_tilde / 2.0
          )
          highs[:-1] = scipy.special.betaincinv(
              k_vals[:-1] + 1, n - k_vals[:-1], 1.0 - alpha_tilde / 2.0
          )
        cache[(n, alpha)] = (lows, highs)

    for n_pool in n_pools:
      d_vals = np.arange(n_pool + 1)
      thetas = d_vals / float(n_pool)  # shape (N+1,)
      for n in range(1, n_pool + 1):
        k_vals = np.arange(n + 1)
        pmfs = scipy.stats.hypergeom.pmf(
            k_vals[None, :], n_pool, d_vals[:, None], n
        )
        for alpha in alphas:
          lows, highs = cache[(n, alpha)]
          # Vectorized miss check across all D in 0..N
          miss_low = np.sum(pmfs * (thetas[:, None] < lows[None, :]), axis=1)
          miss_high = np.sum(pmfs * (thetas[:, None] > highs[None, :]), axis=1)
          tol = (alpha / 2.0) * (1.0 + 1e-12)
          self.assertTrue(
              np.all(miss_low <= tol),
              f"Lower miss exceeded alpha/2 for N={n_pool}, n={n},"
              f" alpha={alpha}",
          )
          self.assertTrue(
              np.all(miss_high <= tol),
              f"Upper miss exceeded alpha/2 for N={n_pool}, n={n},"
              f" alpha={alpha}",
          )

  def test_hypergeometric_counterexample(self):
    """Verifies that standard i.i.d. Clopper-Pearson undercovers under without-replacement sampling, while accuracy interval covers."""
    n_pool = 100
    d_val = 1
    n = 5
    alpha = 0.0995
    theta = d_val / float(n_pool)  # 0.01

    # Standard i.i.d. Clopper-Pearson lower endpoint for k=1: betaincinv(1, 5, alpha / 2)
    standard_cp_low_k1 = float(scipy.special.betaincinv(1, 5, alpha / 2.0))
    self.assertGreater(standard_cp_low_k1, theta)
    # Pr(K >= 1) under Hypergeom(100, 1, 5) is 5/100 = 0.05 > alpha / 2 = 0.04975
    pr_k_ge_1 = 1.0 - float(scipy.stats.hypergeom.pmf(0, n_pool, d_val, n))
    self.assertAlmostEqual(pr_k_ge_1, 0.05, places=12)
    self.assertGreater(pr_k_ge_1, alpha / 2.0)

    # Accuracy confidence interval covers theta
    arr_k1 = np.array([1.0] + [0.0] * (n - 1))
    res = interval.confidence_interval(
        arr_k1, metric="accuracy", level=1.0 - alpha
    )
    self.assertLessEqual(res.low, theta)

  def test_accuracy_interval_contains_clopper_pearson_and_tight(self):
    """Verifies that at n=1000, k=950, level=0.95 accuracy interval contains exact Clopper-Pearson and is <= 0.5% wider."""
    n = 1000
    k = 950
    level = 0.95
    alpha = 1.0 - level

    cp_low = float(scipy.special.betaincinv(k, n - k + 1, alpha / 2.0))
    cp_high = float(
        scipy.special.betaincinv(k + 1, n - k, 1.0 - alpha / 2.0)
    )
    cp_width = cp_high - cp_low

    arr = np.array([1.0] * k + [0.0] * (n - k))
    res = interval.confidence_interval(arr, metric="accuracy", level=level)
    lib_width = res.high - res.low

    self.assertLessEqual(res.low, cp_low)
    self.assertGreaterEqual(res.high, cp_high)
    self.assertLessEqual(lib_width, 1.005 * cp_width)

  def test_small_n_returns_an_interval(self):
    # The generic bound refuses below n = 25 at M = 4, level 0.95.
    res = interval.confidence_interval(np.array([1.0, 0.0, 1.0]), "accuracy")
    self.assertEqual(res.status, interval.Status.UNREFUTED)
    self.assertEqual(res.level, 0.95)

  def test_m_is_ignored(self):
    arr = np.array([1.0] * 90 + [0.0] * 10)
    a = interval.confidence_interval(arr, "accuracy", m=1.5)
    b = interval.confidence_interval(arr, "accuracy", m=50.0)
    self.assertEqual((a.low, a.high), (b.low, b.high))
    self.assertEqual(a.m_declared, 1.5)
    self.assertEqual(b.status, interval.Status.UNREFUTED)

  def test_sketch_and_merge_match_data(self):
    rng = np.random.default_rng(0)
    arr = (rng.random(1000) < 0.8).astype(np.float64)
    want = interval.confidence_interval(arr, "accuracy")
    sk = sketch.merge(
        sketch.from_data(arr[:300], metric="accuracy"),
        sketch.from_data(arr[300:], metric="accuracy"),
    )
    got = interval.confidence_interval(sk, "accuracy")
    self.assertEqual((got.low, got.high), (want.low, want.high))

  def test_sketch_that_is_not_a_count_is_refused(self):
    base = sketch.from_data(np.array([1.0, 0.0, 1.0, 1.0]), metric="accuracy")
    assert isinstance(base, sketch.IndependentSketch)
    fractional = dataclasses.replace(base, sum_s=2.5, sum_s2=2.5)
    with self.assertRaisesRegex(ValueError, "exact integer"):
      interval.confidence_interval(fractional, "accuracy")
    not_binary = dataclasses.replace(base, sum_s2=2.0)
    with self.assertRaisesRegex(ValueError, "exact integer"):
      interval.confidence_interval(not_binary, "accuracy")

  def test_accuracy_level_guard(self):
    arr = np.array([1.0] * 50 + [0.0] * 50)
    with self.assertRaisesRegex(ValueError, r"requires level >= sqrt\(2\) - 1"):
      interval.confidence_interval(arr, metric="accuracy", level=0.41)
    res = interval.confidence_interval(arr, metric="accuracy", level=0.42)
    self.assertEqual(res.status, interval.Status.UNREFUTED)
    self.assertTrue(math.isfinite(res.low))
    self.assertTrue(math.isfinite(res.high))

  def test_effective_n_keeps_the_generic_bound(self):
    arr = np.array([1.0] * 900 + [0.0] * 100)
    exact = interval.confidence_interval(arr, "accuracy")
    corr = interval.confidence_interval(arr, "accuracy", effective_n=1000.0)
    (low, high), _ = bound.bound(
        n=1000,
        sum_s=900.0,
        sum_s2=900.0,
        top_k=np.sort(arr)[::-1],
        alpha=0.05,
        M=4.0,
        metric="accuracy",
        effective_n=1000.0,
    )
    self.assertEqual((corr.low, corr.high), (low, high))
    self.assertLess(corr.low, exact.low)

  def test_sketch_with_values_exceeding_one_raises(self):
    # [4/3, 1/3, 1/3] has n=3, sum_s=2.0, sum_s2=2.0, passing integer sums,
    # but top_32[0] = 4/3 > 1.0, which must raise ValueError.
    sk = sketch.from_array([4.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    self.assertAlmostEqual(sk.sum_s, 2.0, places=12)
    self.assertAlmostEqual(sk.sum_s2, 2.0, places=12)
    with self.assertRaisesRegex(ValueError, "exceeding 1.0"):
      interval.confidence_interval(sk, metric="accuracy")

    # Correlated path also raises
    sk_corr = dataclasses.replace(sk, effective_n=2.0)
    with self.assertRaisesRegex(ValueError, "exceeding 1.0"):
      interval.confidence_interval(sk_corr, metric="accuracy")


if __name__ == "__main__":
  absltest.main()
