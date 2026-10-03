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

"""Unit tests for statistical refutation primitives (refutation.py)."""

import math
import os

from absl.testing import absltest
import numpy as np

import scipy.signal

from dgf.src.stats import graph
from dgf.src.stats import partitioners
from dgf.src.stats import refutation

# Set DGF_STATS_LONG_TESTS=1 (blaze test --test_env=...) for the full-size runs.
_LONG = os.environ.get("DGF_STATS_LONG_TESTS") == "1"


class RefutationTest(absltest.TestCase):

  def test_check_uncorrelated_size_independent_gaussian(self):
    """Size check on G=2000 independent Gaussian totals: rejection rate near 0.05."""
    g = 2000
    num_seeds = 400 if _LONG else 40
    rejections = 0

    for s in range(num_seeds):
      rng = np.random.default_rng(1000 + s)
      totals = rng.normal(loc=5.0, scale=2.0, size=g)
      res = refutation.check_uncorrelated(totals, level=0.95)
      self.assertEqual(
          res.outcome
          in (refutation.Refutation.REFUTED, refutation.Refutation.NOT_REFUTED),
          True,
      )
      self.assertIsNotNone(res.statistic)
      self.assertIsNotNone(res.threshold)
      if res.outcome == refutation.Refutation.REFUTED:
        rejections += 1

    rate = float(rejections) / float(num_seeds)
    if _LONG:
      print(
          f"check_uncorrelated null false-rejection rate: {rate:.4f}"
          f" ({rejections}/{num_seeds})"
      )
    # Nominal rate is p=0.05. In short mode (num_seeds=40), bounds are p ± 3*sqrt(p*(1-p)/n).
    min_rate = (
        0.02
        if _LONG
        else max(0.0, 0.05 - 3.0 * math.sqrt(0.05 * 0.95 / float(num_seeds)))
    )
    max_rate = (
        0.09
        if _LONG
        else min(1.0, 0.05 + 3.0 * math.sqrt(0.05 * 0.95 / float(num_seeds)))
    )
    self.assertBetween(
        rate,
        min_rate,
        max_rate,
        f"Empirical false rejection rate {rate:.4f} outside [{min_rate:.4f}, {max_rate:.4f}].",
    )

  def test_check_uncorrelated_power_ar1_phi_09(self):
    """Power check: AR(1) with phi=0.9, G=2000, seeds -> rejection >= 95%."""
    g = 2000
    phi = 0.9
    num_seeds = 200 if _LONG else 30
    rejections = 0

    for s in range(num_seeds):
      rng = np.random.default_rng(2000 + s)
      innov = rng.standard_normal(g)
      innov[1:] *= math.sqrt(1.0 - phi**2)
      totals = scipy.signal.lfilter([1.0], [1.0, -phi], innov)
      res = refutation.check_uncorrelated(totals, level=0.95)
      if res.outcome == refutation.Refutation.REFUTED:
        rejections += 1

    power = float(rejections) / float(num_seeds)
    if _LONG:
      print(
          f"check_uncorrelated AR(1) phi=0.9 power: {power:.4f}"
          f" ({rejections}/{num_seeds})"
      )
    # AR(1) phi=0.9 has very high power (near 1.0); in short mode (num_seeds=30) assert >= 0.90
    min_power = 0.95 if _LONG else 0.90
    self.assertGreaterEqual(
        power,
        min_power,
        f"Empirical power {power:.4f} less than {min_power}.",
    )

  def test_check_kappa_independent_totals_kappa_declared_2(self):
    """Verifies check_kappa on independent totals with kappa_declared=2: rejection <= 0.02."""
    g = 2000
    num_seeds = 200 if _LONG else 30
    rejections = 0

    for s in range(num_seeds):
      rng = np.random.default_rng(3000 + s)
      totals = rng.normal(loc=10.0, scale=1.0, size=g)
      res = refutation.check_kappa(totals, kappa_declared=2.0, level=0.95)
      self.assertEqual(
          res.outcome
          in (refutation.Refutation.REFUTED, refutation.Refutation.NOT_REFUTED),
          True,
      )
      if res.outcome == refutation.Refutation.REFUTED:
        rejections += 1

    rate = float(rejections) / float(num_seeds)
    if _LONG:
      print(
          f"check_kappa rejection rate for kappa_declared=2 on i.i.d.: {rate:.4f}"
          f" ({rejections}/{num_seeds})"
      )
    # Under kappa=2 on iid, false rejection is practically zero. In short mode (num_seeds=30), allow up to 1 rejection (0.034)
    bound_rate = 0.02 if _LONG else (1.0 / float(num_seeds) + 1e-6)
    self.assertLessEqual(
        rate,
        bound_rate,
        f"False rejection rate {rate:.4f} exceeded {bound_rate:.4f}.",
    )

  def test_untestable_conditions(self):
    """UNTESTABLE checks: G=4, non-finites (NaN/inf), constant totals, G < num_batches."""
    # 1. G = 4
    arr_g4 = np.array([1.0, 2.0, 3.0, 4.0])
    res_uncorr_g4 = refutation.check_uncorrelated(arr_g4)
    self.assertEqual(res_uncorr_g4.outcome, refutation.Refutation.UNTESTABLE)
    self.assertIsNone(res_uncorr_g4.statistic)
    self.assertIsNone(res_uncorr_g4.threshold)

    # 2. Non-finite values
    arr_nan = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    arr_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0])
    self.assertEqual(
        refutation.check_uncorrelated(arr_nan).outcome,
        refutation.Refutation.UNTESTABLE,
    )
    self.assertEqual(
        refutation.check_uncorrelated(arr_inf).outcome,
        refutation.Refutation.UNTESTABLE,
    )
    self.assertIsNone(refutation.kappa_lower_bound(arr_nan))
    self.assertIsNone(refutation.kappa_lower_bound(arr_inf))
    self.assertEqual(
        refutation.check_kappa(arr_nan, kappa_declared=1.5).outcome,
        refutation.Refutation.UNTESTABLE,
    )
    self.assertEqual(
        refutation.check_kappa(arr_inf, kappa_declared=1.5).outcome,
        refutation.Refutation.UNTESTABLE,
    )

    # 3. Constant totals (zero variance)
    arr_const = np.full(50, 42.0)
    res_uncorr_const = refutation.check_uncorrelated(arr_const)
    self.assertEqual(res_uncorr_const.outcome, refutation.Refutation.UNTESTABLE)
    self.assertIsNone(refutation.kappa_lower_bound(arr_const))
    self.assertEqual(
        refutation.check_kappa(arr_const, kappa_declared=1.0).outcome,
        refutation.Refutation.UNTESTABLE,
    )

    # 4. G < num_batches for check_kappa
    arr_g20 = np.arange(20, dtype=np.float64)
    self.assertIsNone(refutation.kappa_lower_bound(arr_g20, num_batches=30))
    self.assertEqual(
        refutation.check_kappa(
            arr_g20, kappa_declared=1.0, num_batches=30
        ).outcome,
        refutation.Refutation.UNTESTABLE,
    )

  def test_invalid_arguments_raise_value_error(self):
    """ValueError checks: non-1-D input, level outside (0, 1), num_batches < 2, kappa_declared < 1."""
    valid_data = np.arange(100, dtype=np.float64)

    # Non-1-D input
    data_2d = np.ones((10, 10))
    with self.assertRaises(ValueError):
      refutation.check_uncorrelated(data_2d)
    with self.assertRaises(ValueError):
      refutation.kappa_lower_bound(data_2d)
    with self.assertRaises(ValueError):
      refutation.check_kappa(data_2d, kappa_declared=1.0)

    # Level outside (0, 1)
    for bad_lvl in (0.0, 1.0, -0.1, 1.1):
      with self.assertRaises(ValueError):
        refutation.check_uncorrelated(valid_data, level=bad_lvl)
      with self.assertRaises(ValueError):
        refutation.kappa_lower_bound(valid_data, level=bad_lvl)
      with self.assertRaises(ValueError):
        refutation.check_kappa(valid_data, kappa_declared=1.0, level=bad_lvl)

    # num_batches < 2
    for bad_b in (0, 1, -5):
      with self.assertRaises(ValueError):
        refutation.kappa_lower_bound(valid_data, num_batches=bad_b)
      with self.assertRaises(ValueError):
        refutation.check_kappa(
            valid_data, kappa_declared=1.0, num_batches=bad_b
        )

    # kappa_declared < 1.0 or NaN
    for bad_k in (0.99, 0.0, -1.0, float("nan")):
      with self.assertRaises(ValueError):
        refutation.check_kappa(valid_data, kappa_declared=bad_k)

  def test_edge_test_size_heteroscedastic(self):
    """Size check on (a) geometric, (b) BA partitioner cluster graph, and (c) hub graph."""
    num_reps = 2000 if _LONG else 40
    se = math.sqrt(0.05 * 0.95 / float(num_reps))
    max_rate = min(1.0, 0.05 + 3.0 * se)
    min_rate = 0.02 if _LONG else max(0.0, 0.05 - 3.0 * se)

    # (a) Geometric graph G = 500, mean degree ≈ 6
    g_a = 500
    edges_a = _build_geometric_edges(g_a, mean_deg=6.3, seed=100)
    rng_counts = np.random.default_rng(101)
    counts_a = rng_counts.integers(25, 75, size=g_a)

    rej_a = 0
    guard_a = 0
    t_a_list = []
    rng_sim = np.random.default_rng(202601)
    for _ in range(num_reps):
      s_c = rng_sim.lognormal(0.0, 0.5, size=g_a)
      eps = rng_sim.normal(0.0, np.sqrt(counts_a.astype(np.float64)) * s_c)
      theta_hat = float(np.sum(eps)) / float(np.sum(counts_a))
      e = eps - counts_a.astype(np.float64) * theta_hat
      res = refutation.check_uncorrelated_edges(e, counts_a, edges_a, level=0.95)
      if res.statistic is not None:
        t_a_list.append(res.statistic)
      if res.outcome == refutation.Refutation.REFUTED:
        rej_a += 1
      if res.outcome == refutation.Refutation.UNTESTABLE and res.reason is not None:
        guard_a += 1
    rate_a = float(rej_a) / float(num_reps)
    mean_t_a = float(np.mean(t_a_list)) if t_a_list else float("nan")
    std_t_a = float(np.std(t_a_list)) if t_a_list else float("nan")
    if _LONG:
      print(
          f"Edge test size (a) geometric G=500: {rate_a:.4f}"
          f" (rejections={rej_a}/{num_reps}, mean_T={mean_t_a:.4f},"
          f" std_T={std_t_a:.4f})"
      )
      print(f"Edge test mean T on geometric: {mean_t_a:.4f}")
      print(f"Edge test std T on geometric: {std_t_a:.4f}")
      print(
          f"Edge test guard-caused untestable (a) geometric: {guard_a}/{num_reps}"
      )
    self.assertLessEqual(rate_a, max_rate)
    self.assertGreaterEqual(rate_a, min_rate)

    # (b) Cluster graph of partition_graph on Barabási-Albert (n=20000, m=5, t=50)
    ba_edges = _build_ba_edges(20000, 5, seed=200)
    labels_b = partitioners.partition_graph(
        20000, ba_edges, target_cluster_size=50, seed=0
    )
    counts_b = np.bincount(labels_b).astype(np.int64)
    g_b = len(counts_b)
    edges_b = graph.cluster_adjacency(20000, ba_edges, labels_b)

    rej_b = 0
    guard_b = 0
    t_b_list = []
    for _ in range(num_reps):
      s_c = rng_sim.lognormal(0.0, 0.5, size=g_b)
      eps = rng_sim.normal(0.0, np.sqrt(counts_b.astype(np.float64)) * s_c)
      theta_hat = float(np.sum(eps)) / float(np.sum(counts_b))
      e = eps - counts_b.astype(np.float64) * theta_hat
      res = refutation.check_uncorrelated_edges(e, counts_b, edges_b, level=0.95)
      if res.statistic is not None:
        t_b_list.append(res.statistic)
      if res.outcome == refutation.Refutation.REFUTED:
        rej_b += 1
      if res.outcome == refutation.Refutation.UNTESTABLE and res.reason is not None:
        guard_b += 1
    rate_b = float(rej_b) / float(num_reps)
    mean_t_b = float(np.mean(t_b_list)) if t_b_list else float("nan")
    std_t_b = float(np.std(t_b_list)) if t_b_list else float("nan")
    if _LONG:
      print(
          f"Edge test size (b) BA G={g_b}: {rate_b:.4f}"
          f" (rejections={rej_b}/{num_reps}, mean_T={mean_t_b:.4f},"
          f" std_T={std_t_b:.4f})"
      )
      print(f"Edge test mean T on BA: {mean_t_b:.4f}")
      print(f"Edge test std T on BA: {std_t_b:.4f}")
      print(
          f"Edge test guard-caused untestable (b) BA: {guard_b}/{num_reps}"
      )
    # At num_reps=40, SE of sample mean of standard normal is 1/sqrt(40) ~ 0.158, so allow 3*SE ~ 0.50 in short mode
    max_abs_mean_t = 0.1 if _LONG else (3.0 / math.sqrt(float(num_reps)))
    self.assertLessEqual(abs(mean_t_b), max_abs_mean_t)
    self.assertLessEqual(rate_b, max_rate)
    self.assertGreaterEqual(rate_b, min_rate)

    # (c) Hub graph: G=500 geometric base plus 3 hubs each adjacent to 90% of clusters
    g_c = 500
    base_edges = _build_geometric_edges(g_c, mean_deg=6.0, seed=300)
    hub_edges_list = [base_edges]
    rng_hubs = np.random.default_rng(301)
    for hub in (0, 1, 2):
      others = np.delete(np.arange(g_c), hub)
      chosen = rng_hubs.choice(
          others, size=int(0.9 * (g_c - 1)), replace=False
      )
      h_min = np.minimum(hub, chosen)
      h_max = np.maximum(hub, chosen)
      hub_edges_list.append(np.column_stack([h_min, h_max]))
    all_hub_edges = np.vstack(hub_edges_list)
    pair_keys = np.unique(all_hub_edges[:, 0] * g_c + all_hub_edges[:, 1])
    edges_c = np.column_stack([pair_keys // g_c, pair_keys % g_c])
    counts_c = rng_hubs.integers(25, 75, size=g_c)

    rej_c = 0
    guard_c = 0
    t_c_list = []
    for _ in range(num_reps):
      s_c = rng_sim.lognormal(0.0, 0.5, size=g_c)
      eps = rng_sim.normal(0.0, np.sqrt(counts_c.astype(np.float64)) * s_c)
      theta_hat = float(np.sum(eps)) / float(np.sum(counts_c))
      e = eps - counts_c.astype(np.float64) * theta_hat
      res = refutation.check_uncorrelated_edges(e, counts_c, edges_c, level=0.95)
      if res.statistic is not None:
        t_c_list.append(res.statistic)
      if res.outcome == refutation.Refutation.REFUTED:
        rej_c += 1
      if res.outcome == refutation.Refutation.UNTESTABLE and res.reason is not None:
        guard_c += 1
    rate_c = float(rej_c) / float(num_reps)
    mean_t_c = float(np.mean(t_c_list)) if t_c_list else float("nan")
    std_t_c = float(np.std(t_c_list)) if t_c_list else float("nan")
    if _LONG:
      print(
          f"Edge test size (c) Hubs G={g_c}: {rate_c:.4f}"
          f" (rejections={rej_c}/{num_reps}, mean_T={mean_t_c:.4f},"
          f" std_T={std_t_c:.4f})"
      )
      print(f"Edge test mean T on Hubs: {mean_t_c:.4f}")
      print(f"Edge test std T on Hubs: {std_t_c:.4f}")
      print(
          f"Edge test guard-caused untestable (c) Hubs: {guard_c}/{num_reps}"
      )
    self.assertLessEqual(rate_c, max_rate)

  def test_projected_variance_and_bias_exact_algebra(self):
    """Verifies V' == 2 * sum_{k,l} A_kl^2 x_k x_l and B_hat == sum_k A_kk x_k to 1e-10."""
    rng = np.random.default_rng(202609)
    for g in (15, 30, 60):
      counts = rng.integers(10, 50, size=g)
      n_total = float(np.sum(counts))
      q = np.sqrt(counts.astype(np.float64) / n_total)

      pairs = []
      for i in range(g):
        for j in range(i + 1, g):
          if rng.random() < 0.25:
            pairs.append((i, j))
      edges = np.array(pairs, dtype=np.int64)
      u = edges[:, 0]
      v = edges[:, 1]

      z = rng.normal(size=g)
      x = z**2

      # Dense calculation: A = 1/2 P W P
      W = np.zeros((g, g), dtype=np.float64)
      W[u, v] = 1.0
      W[v, u] = 1.0
      P = np.eye(g) - np.outer(q, q)
      A = 0.5 * P @ W @ P

      v_dense = 2.0 * float(np.sum((A**2) * np.outer(x, x)))
      b_dense = float(np.sum(np.diag(A) * x))

      # Vectorized formulas from check_uncorrelated_edges
      w = np.bincount(u, weights=q[v], minlength=g) + np.bincount(
          v, weights=q[u], minlength=g
      )
      s = 2.0 * float(np.sum(q[u] * q[v]))
      q_u = q[u]
      q_v = q[v]
      w_u = w[u]
      w_v = w[v]
      r_uv = -q_u * w_v - w_u * q_v + s * q_u * q_v

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

      e = z * np.sqrt(counts.astype(np.float64))
      q_sum = float(np.sum(e**2))
      n_u = counts[u].astype(np.float64)
      n_v = counts[v].astype(np.float64)
      e_u = e[u]
      e_v = e[v]
      b_term = (
          (n_u * n_v * q_sum / (n_total**2))
          - (n_u * (e_v**2) + n_v * (e_u**2)) / n_total
      ) / np.sqrt(n_u * n_v)
      b_hat = float(np.sum(b_term))

      rel_err_v = abs(v_prime - v_dense) / v_dense
      rel_err_b = abs(b_hat - b_dense) / abs(b_dense)
      self.assertLessEqual(rel_err_v, 1e-10)
      self.assertLessEqual(rel_err_b, 1e-10)

  def test_edge_test_power_and_kappa_w_hat(self):
    """Tests edge-test power (>= 0.90) and kappa_w_hat accuracy (within 5% of true VIF)."""
    g = 500
    edges = _build_geometric_edges(g, mean_deg=6.3, seed=400)
    num_edges = len(edges)
    u = edges[:, 0]
    v = edges[:, 1]

    # Target VIF ≈ 1.7
    # 0.7 = 2 * gamma^2 * |E| / (G + 2 * gamma^2 * |E|) => 0.6 * gamma^2 * |E| = 0.7 * G
    gamma = math.sqrt(0.7 * float(g) / (0.6 * float(num_edges)))
    true_vif = 1.0 + (2.0 * (gamma**2) * float(num_edges)) / (
        float(g) + 2.0 * (gamma**2) * float(num_edges)
    )

    counts = np.ones(g, dtype=np.int64)
    num_reps = 500 if _LONG else 40
    rejections = 0
    kw_estimates = []

    rng = np.random.default_rng(202602)
    for _ in range(num_reps):
      eta = rng.standard_normal(g)
      xi = rng.standard_normal(num_edges)
      shared_per_node = np.bincount(u, weights=xi, minlength=g) + np.bincount(
          v, weights=xi, minlength=g
      )
      eps = eta + gamma * shared_per_node
      e = eps - float(np.mean(eps))

      res = refutation.check_uncorrelated_edges(e, counts, edges, level=0.95)
      if res.outcome == refutation.Refutation.REFUTED:
        rejections += 1

      sum_e2 = float(np.sum(e**2))
      e_sum = float(np.sum(e[u] * e[v]))
      kw = 1.0 + 2.0 * e_sum / sum_e2
      kw_estimates.append(kw)

    power = float(rejections) / float(num_reps)
    mean_kw = float(np.mean(kw_estimates))

    if _LONG:
      print(
          f"Edge test power (VIF={true_vif:.4f}): {power:.4f}"
          f" ({rejections}/{num_reps})"
      )
      print(
          f"kappa_w_hat mean: {mean_kw:.4f} (exact VIF={true_vif:.4f},"
          f" rel_err={abs(mean_kw - true_vif) / true_vif:.4f})"
      )

    # In long mode assert power >= 0.90 and rel_err <= 0.05.
    # In short mode (num_reps=40), assert power >= 0.80 and rel_err <= 0.10.
    min_power = 0.90 if _LONG else 0.80
    max_rel_err = 0.05 if _LONG else 0.10
    self.assertGreaterEqual(power, min_power)
    self.assertLessEqual(abs(mean_kw - true_vif) / true_vif, max_rel_err)

  def test_batch_check_size(self):
    """Verifies batch check size <= 0.05 + 3*SE under true VIF and under independence."""
    g = 2000
    edges = _build_geometric_edges(g, mean_deg=6.0, seed=500)
    num_edges = len(edges)
    u = edges[:, 0]
    v = edges[:, 1]

    target_bs = max(2, int(round(float(g) / 30.0)))
    batch_ids = partitioners.partition_graph(
        g, edges, target_cluster_size=target_bs, seed=0
    )
    b_count = len(np.unique(batch_ids))

    counts = np.ones(g, dtype=np.int64)
    num_reps = 2000 if _LONG else 40
    se = math.sqrt(0.05 * 0.95 / float(num_reps))
    max_rate = min(1.0, 0.05 + 3.0 * se)

    # 1. Under independence: declare kappa = 1.0
    rng = np.random.default_rng(202603)
    rej_indep = 0
    for _ in range(num_reps):
      eps = rng.standard_normal(g)
      e = eps - float(np.mean(eps))
      res = refutation.check_kappa_batches(
          e, counts, batch_ids, kappa_declared=1.0, level=0.95
      )
      if res.outcome == refutation.Refutation.REFUTED:
        rej_indep += 1

    rate_indep = float(rej_indep) / float(num_reps)
    if _LONG:
      print(
          f"Batch check size (indep, declared=1.0, B={b_count}):"
          f" {rate_indep:.4f} ({rej_indep}/{num_reps})"
      )
    self.assertLessEqual(rate_indep, max_rate)

    # 2. Under correlated errors with known exact VIF: declare kappa = exact VIF
    gamma = 0.5
    exact_vif = 1.0 + (2.0 * (gamma**2) * float(num_edges)) / (
        float(g) + 2.0 * (gamma**2) * float(num_edges)
    )

    rej_corr = 0
    for _ in range(num_reps):
      eta = rng.standard_normal(g)
      xi = rng.standard_normal(num_edges)
      shared_per_node = np.bincount(u, weights=xi, minlength=g) + np.bincount(
          v, weights=xi, minlength=g
      )
      eps = eta + gamma * shared_per_node
      e = eps - float(np.mean(eps))
      res = refutation.check_kappa_batches(
          e, counts, batch_ids, kappa_declared=exact_vif, level=0.95
      )
      if res.outcome == refutation.Refutation.REFUTED:
        rej_corr += 1

    rate_corr = float(rej_corr) / float(num_reps)
    if _LONG:
      print(
          f"Batch check size (correlated, exact_vif={exact_vif:.4f},"
          f" B={b_count}): {rate_corr:.4f} ({rej_corr}/{num_reps})"
      )
    self.assertLessEqual(rate_corr, max_rate)

  def test_batch_check_power(self):
    """Verifies batch check power on gamma=1 construction with declared kappa=1.2 (assert >= 0.3)."""
    g = 2000
    edges = _build_geometric_edges(g, mean_deg=6.0, seed=600)
    num_edges = len(edges)
    u = edges[:, 0]
    v = edges[:, 1]

    gamma = 1.0
    true_vif = 1.0 + (2.0 * float(num_edges)) / (
        float(g) + 2.0 * float(num_edges)
    )

    target_bs = max(2, int(round(float(g) / 30.0)))
    batch_ids = partitioners.partition_graph(
        g, edges, target_cluster_size=target_bs, seed=0
    )
    b_count = len(np.unique(batch_ids))

    counts = np.ones(g, dtype=np.int64)
    num_reps = 500 if _LONG else 40
    rej = 0
    rng = np.random.default_rng(202604)

    for _ in range(num_reps):
      eta = rng.standard_normal(g)
      xi = rng.standard_normal(num_edges)
      shared_per_node = np.bincount(u, weights=xi, minlength=g) + np.bincount(
          v, weights=xi, minlength=g
      )
      eps = eta + gamma * shared_per_node
      e = eps - float(np.mean(eps))
      res = refutation.check_kappa_batches(
          e, counts, batch_ids, kappa_declared=1.2, level=0.95
      )
      if res.outcome == refutation.Refutation.REFUTED:
        rej += 1

    power = float(rej) / float(num_reps)
    if _LONG:
      print(
          f"Batch check power (gamma=1, true_vif={true_vif:.4f},"
          f" declared=1.2, B={b_count}): {power:.4f} ({rej}/{num_reps})"
      )
    # Power is ~0.4-0.5. At num_reps=40, assert >= 0.25 (allowing 10/40 rejections)
    min_power = 0.3 if _LONG else 0.25
    self.assertGreaterEqual(power, min_power)

  def test_graph_refutation_degenerate_inputs(self):
    """Verifies degenerate and invalid inputs for edge and batch refutation checks."""
    g = 50
    counts = np.ones(g, dtype=np.int64)
    residuals = np.zeros(g, dtype=np.float64)

    # Fewer than 30 edges -> UNTESTABLE
    edges_few = np.array([[i, i + 1] for i in range(20)], dtype=np.int64)
    res_few = refutation.check_uncorrelated_edges(residuals, counts, edges_few)
    self.assertEqual(res_few.outcome, refutation.Refutation.UNTESTABLE)

    # B < 10 -> UNTESTABLE
    batch_ids_few = np.repeat(np.arange(8), g // 8 + 1)[:g]
    res_batch_few = refutation.check_kappa_batches(
        residuals, counts, batch_ids_few, kappa_declared=1.5
    )
    self.assertEqual(res_batch_few.outcome, refutation.Refutation.UNTESTABLE)

    # Out-of-range edge index -> ValueError
    bad_edges = np.array([[0, g + 5]], dtype=np.int64)
    with self.assertRaises(ValueError):
      refutation.check_uncorrelated_edges(residuals, counts, bad_edges)

    # Duplicates and reversed edges give identical T
    edges_base = np.array([[i, i + 1] for i in range(35)], dtype=np.int64)
    rng = np.random.default_rng(700)
    res_rand = rng.normal(size=g)
    r1 = refutation.check_uncorrelated_edges(res_rand, counts, edges_base)

    # Add reversed edges and duplicates
    rev_edges = np.column_stack([edges_base[:, 1], edges_base[:, 0]])
    edges_dup = np.vstack([edges_base, rev_edges, edges_base[:10]])
    r2 = refutation.check_uncorrelated_edges(res_rand, counts, edges_dup)

    self.assertEqual(r1.outcome, r2.outcome)
    self.assertIsNotNone(r1.statistic)
    self.assertIsNotNone(r2.statistic)
    self.assertIsNotNone(r1.threshold)
    self.assertIsNotNone(r2.threshold)
    assert r1.statistic is not None and r2.statistic is not None
    assert r1.threshold is not None and r2.threshold is not None
    self.assertAlmostEqual(r1.statistic, r2.statistic, places=12)
    self.assertAlmostEqual(r1.threshold, r2.threshold, places=12)

  def test_centering_bias_correction_exact_mean(self):
    """Verifies B_hat equals exact null mean of sum_E z_c z_d within 5% on a 50-cluster graph."""
    rng = np.random.default_rng(123)
    g = 50
    counts = rng.integers(20, 80, size=g)
    sigma2 = counts.astype(np.float64)**1.5
    n_total = float(np.sum(counts))

    edges = _build_geometric_edges(g, mean_deg=6.0, seed=123)
    u = edges[:, 0]
    v = edges[:, 1]
    n_u = counts[u].astype(np.float64)
    n_v = counts[v].astype(np.float64)
    s_u = sigma2[u]
    s_v = sigma2[v]

    # Closed-form exact null mean of sum_E z_c z_d
    exact_cov = (
        (n_u * n_v * np.sum(sigma2) / (n_total**2))
        - (n_u * s_v + n_v * s_u) / n_total
    )
    exact_null_mean = float(np.sum(exact_cov / np.sqrt(n_u * n_v)))

    # Monte Carlo estimation over 20,000 draws in long mode, 200 draws in short mode
    reps = 20000 if _LONG else 200
    sum_b_hat = 0.0
    for _ in range(reps):
      eps = rng.normal(0.0, np.sqrt(sigma2))
      e = eps - counts.astype(np.float64) * (np.sum(eps) / n_total)
      q = float(np.sum(e**2))
      e_u = e[u]
      e_v = e[v]
      b_term = (
          (n_u * n_v * q / (n_total**2))
          - (n_u * (e_v**2) + n_v * (e_u**2)) / n_total
      ) / np.sqrt(n_u * n_v)
      sum_b_hat += float(np.sum(b_term))

    mc_b_mean = sum_b_hat / float(reps)
    rel_err = abs(mc_b_mean - exact_null_mean) / abs(exact_null_mean)
    if _LONG:
      print(
          f"Centering bias B_hat MC mean: {mc_b_mean:.4f},"
          f" exact={exact_null_mean:.4f}, rel_err={rel_err:.4%}"
      )
    # At reps=20000 rel_err <= 0.05. In short mode (reps=200), MC noise allows up to 25% rel_err.
    max_rel_err = 0.05 if _LONG else 0.25
    self.assertLessEqual(rel_err, max_rel_err)

  def test_edge_effective_count(self):
    """Verifies _edge_effective_count matches exact values and scale invariance."""
    self.assertEqual(refutation._edge_effective_count(np.array([1, 1, 1])), 3.0)
    self.assertAlmostEqual(
        refutation._edge_effective_count(np.array([1, -1, 2])),
        16.0 / 6.0,
        places=12,
    )
    self.assertEqual(refutation._edge_effective_count(np.array([])), 0.0)
    self.assertEqual(refutation._edge_effective_count(np.array([0, 0])), 0.0)
    self.assertEqual(refutation._edge_effective_count(np.array([2, 2, 2])), 3.0)

  def test_dominated_matching(self):
    """Dominated matching counterexample: G=1000, 3 active edges refute unguarded but caught by guard."""
    g = 1000
    counts = np.ones(g, dtype=np.int64)
    edges = np.column_stack([np.arange(0, g, 2), np.arange(1, g, 2)]).astype(
        np.int64
    )
    e = np.full(g, -0.006, dtype=np.float64)
    e[:6] = 0.994
    self.assertAlmostEqual(float(np.sum(e)), 0.0, places=12)

    # With min_effective_edges = 0.0: REFUTED, statistic 1.7460 +/- 1e-3
    res_unguarded = refutation.check_uncorrelated_edges(
        e, counts, edges, level=0.95, min_effective_edges=0.0
    )
    self.assertEqual(res_unguarded.outcome, refutation.Refutation.REFUTED)
    self.assertIsNotNone(res_unguarded.statistic)
    self.assertAlmostEqual(res_unguarded.statistic, 1.7460, delta=1e-3)
    if _LONG:
      print(
          f"Dominated matching statistic unguarded: {res_unguarded.statistic:.6f}"
      )

    # With default guard: UNTESTABLE, statistic is None, edge_n_eff = 3.0363 +/- 1e-3, reason contains 'effective edges'
    res_guarded = refutation.check_uncorrelated_edges(
        e, counts, edges, level=0.95
    )
    self.assertEqual(res_guarded.outcome, refutation.Refutation.UNTESTABLE)
    self.assertIsNone(res_guarded.statistic)
    self.assertIsNotNone(res_guarded.edge_n_eff)
    self.assertAlmostEqual(res_guarded.edge_n_eff, 3.0363, delta=1e-3)
    self.assertIsNotNone(res_guarded.reason)
    self.assertIn("effective edges", res_guarded.reason)
    if _LONG:
      print(
          f"Dominated matching edge_n_eff guarded: {res_guarded.edge_n_eff:.6f}"
      )
      print(
          f"Dominated matching reason: {res_guarded.reason}"
      )

  def test_undominated_matching(self):
    """Undominated matching: G=1000, e_c = +/-1 alternating, edge_n_eff == 500, NOT_REFUTED."""
    g = 1000
    counts = np.ones(g, dtype=np.int64)
    edges = np.column_stack([np.arange(0, g, 2), np.arange(1, g, 2)]).astype(
        np.int64
    )
    e = np.array(
        [1.0 if c % 2 == 0 else -1.0 for c in range(g)], dtype=np.float64
    )
    self.assertAlmostEqual(float(np.sum(e)), 0.0, places=12)

    res = refutation.check_uncorrelated_edges(e, counts, edges, level=0.95)
    self.assertIsNotNone(res.edge_n_eff)
    self.assertAlmostEqual(res.edge_n_eff, 500.0, places=9)
    self.assertEqual(res.outcome, refutation.Refutation.NOT_REFUTED)
    self.assertIsNone(res.reason)

  def test_edge_test_min_effective_edges_validation(self):
    """Validation: min_effective_edges of -1.0, nan, or inf raises ValueError."""
    g = 100
    counts = np.ones(g, dtype=np.int64)
    edges = np.column_stack([np.arange(0, g, 2), np.arange(1, g, 2)]).astype(
        np.int64
    )
    e = np.zeros(g, dtype=np.float64)
    for bad_val in (-1.0, float("nan"), float("inf"), float("-inf")):
      with self.assertRaises(ValueError):
        refutation.check_uncorrelated_edges(
            e, counts, edges, min_effective_edges=bad_val
        )

  def test_edge_test_zero_variance_all_edge_products_zero(self):
    """Zero variance: all edge products are zero when only isolated cluster has non-zero residual."""
    g = 100
    counts = np.ones(g, dtype=np.int64)
    # 35 distinct non-self edges spanning clusters 0..70
    edges = np.column_stack([np.arange(35), np.arange(1, 36)]).astype(np.int64)
    # Residuals zero everywhere except cluster 99, which no edge touches
    e = np.zeros(g, dtype=np.float64)
    e[99] = 10.0

    res = refutation.check_uncorrelated_edges(e, counts, edges)
    self.assertEqual(res.outcome, refutation.Refutation.UNTESTABLE)
    self.assertIsNone(res.statistic)
    self.assertIsNone(res.threshold)
    self.assertEqual(
        res.reason,
        "edge statistic has zero variance (all edge products are zero)",
    )
    self.assertEqual(res.edge_n_eff, 0.0)




def _build_geometric_edges(g: int, mean_deg: float, seed: int = 42) -> np.ndarray:
  rng = np.random.default_rng(seed)
  pos = rng.uniform(0.0, 1.0, size=(g, 2))
  r = math.sqrt(mean_deg / (math.pi * float(g)))
  diff = pos[:, None, :] - pos[None, :, :]
  dists = np.sqrt(np.sum(diff**2, axis=-1))
  u, v = np.where((dists < r) & (np.arange(g)[:, None] < np.arange(g)[None, :]))
  return np.column_stack([u, v]).astype(np.int64)


def _build_ba_edges(n: int, m: int, seed: int = 42) -> np.ndarray:
  rng = np.random.default_rng(seed)
  ends = np.empty(2 * n * m + 2 * m, dtype=np.int64)
  ends[:m] = np.arange(m)
  cnt = m
  out = []
  for i in range(m, n):
    t = ends[rng.integers(0, cnt, m)]
    out.append(np.column_stack([np.full(m, i), t]))
    ends[cnt:cnt + m] = t
    ends[cnt + m:cnt + 2 * m] = i
    cnt += 2 * m
  all_edges = np.concatenate(out)
  u = np.minimum(all_edges[:, 0], all_edges[:, 1])
  v = np.maximum(all_edges[:, 0], all_edges[:, 1])
  non_self = u != v
  unique_keys = np.unique(u[non_self] * n + v[non_self])
  return np.column_stack([unique_keys // n, unique_keys % n])


if __name__ == "__main__":
  absltest.main()
