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

"""Tests for the exact ground-truth formulas in coverage_dependent_truth.

Checks:
1. Covariance formulas against quad integration on the bivariate normal
   density, and Phi_2 against scipy.stats.multivariate_normal.
2. Cluster quantities (V_c, V_tot, VIF_true, M_c,true, M_c,var) against the
   dense n x n summand covariance matrix (AR and graph, n <= 300, several metrics).
3. Circular-pool quantities against explicit enumeration of all rotations of a
   length-50 pool.
4. default_m_c_formula, edge_n_eff, and h against hand-computed values on at least
   three inputs each, including the (4,4)+(3,1)x10 block.
5. default_m_c_formula agreement with ClusterSketch.default_m_c to 1e-12 on 20
   random count vectors.
6. The T2b solve reproduces a within-window VIF of 4 to 1e-10.
"""

import math
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import scipy.integrate
import scipy.stats

from dgf.src.stats import cluster_sketch
from dgf.src.stats.benchmark import coverage_dependent_truth as truth


class CoverageDependentTruthTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(20260926)

  # -------------------------------------------------------------------------
  # 1. Section 3.1 Covariance Formulas vs Bivariate Normal Numerical Integrals
  # -------------------------------------------------------------------------

  @parameterized.parameters(0.1, 0.5, 0.8, -0.4)
  def test_cov_mse_against_2d_integral(self, rho):
    """Verifies cov_mse against numerical 1-D quad on conditional Gaussian density."""
    sigma = 1.5
    analytic_cov = truth.cov_mse(rho, sigma)

    def integrand(u1):
      # E[u2^2 - 1 | u1] = rho^2 * (u1^2 - 1)
      e_u2_cond = (sigma**2) * (rho**2) * (u1**2 - 1.0)
      pdf_u1 = scipy.stats.norm.pdf(u1)
      return ((sigma * u1) ** 2 - sigma**2) * e_u2_cond * pdf_u1

    num_cov, _ = scipy.integrate.quad(
        integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-10
    )
    self.assertAlmostEqual(analytic_cov, num_cov, places=9)

  @parameterized.parameters(0.1, 0.5, 0.8, -0.4)
  def test_cov_mae_against_2d_integral(self, rho):
    """Verifies cov_mae against numerical 1-D quad on conditional Gaussian density."""
    sigma = 1.2
    analytic_cov = truth.cov_mae(rho, sigma)
    theta = sigma * math.sqrt(2.0 / math.pi)
    nu = math.sqrt(max(1e-15, 1.0 - rho**2))

    def integrand(u1):
      mu = rho * u1
      # E[|u2| | u1] for u2 ~ N(mu, nu^2)
      e_abs_u2 = mu * math.erf(mu / (nu * math.sqrt(2.0))) + nu * math.sqrt(
          2.0 / math.pi
      ) * math.exp(-(mu**2) / (2.0 * nu**2))
      cond_centered = sigma * e_abs_u2 - theta
      pdf_u1 = scipy.stats.norm.pdf(u1)
      return (sigma * abs(u1) - theta) * cond_centered * pdf_u1

    num_cov, _ = scipy.integrate.quad(
        integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-10
    )
    self.assertAlmostEqual(analytic_cov, num_cov, places=9)

  @parameterized.parameters(
      (0.2, 0.5),
      (0.8, 0.7),
      (-0.3, 0.9),
      (0.6, 0.99),
  )
  def test_phi2_against_multivariate_normal(self, rho, p):
    """Verifies phi2 against scipy.stats.multivariate_normal.cdf."""
    z = float(scipy.stats.norm.ppf(p))
    val_phi2 = truth.phi2(z, rho)
    val_mvn = float(
        scipy.stats.multivariate_normal.cdf(
            [z, z], mean=[0, 0], cov=[[1.0, rho], [rho, 1.0]]
        )
    )
    self.assertAlmostEqual(val_phi2, val_mvn, places=8)

  @parameterized.parameters(0.2, 0.5, 0.8, -0.3)
  def test_cov_accuracy_matches_phi2_minus_p2(self, rho):
    """Verifies cov_accuracy equals phi2(z, z; rho) - p^2."""
    p = 0.85
    z = float(scipy.stats.norm.ppf(p))
    expected = truth.phi2(z, rho) - p**2
    self.assertAlmostEqual(truth.cov_accuracy(rho, p), expected, places=12)

  # -------------------------------------------------------------------------
  # 2. Section 3.2 Cluster Quantities vs Dense n x n Covariance Matrices
  # -------------------------------------------------------------------------

  def test_temporal_ground_truth_against_dense_covariance(self):
    """Verifies temporal_ground_truth against dense n x n summand covariance matrix."""
    n = 120
    counts = np.array([20, 40, 60], dtype=np.int64)
    phi = 0.6
    sigma = 1.3
    p = 0.75

    for metric in ("mse", "mae", "accuracy"):
      gt = truth.temporal_ground_truth(metric, counts, phi, sigma=sigma, p=p)

      # Build dense n x n covariance matrix
      cov_dense = np.zeros((n, n), dtype=np.float64)
      for i in range(n):
        for j in range(n):
          cov_dense[i, j] = truth.summand_covariance(
              metric, phi ** abs(i - j), sigma=sigma, p=p
          )

      # Compare V_tot
      v_tot_dense = float(np.sum(cov_dense))
      np.testing.assert_allclose(gt["v_tot"], v_tot_dense, rtol=1e-10)

      # Compare V_c
      endpoints = np.cumsum([0] + list(counts))
      v_c_dense = np.zeros(len(counts), dtype=np.float64)
      for c in range(len(counts)):
        idx_c = range(endpoints[c], endpoints[c + 1])
        v_c_dense[c] = float(np.sum(cov_dense[np.ix_(idx_c, idx_c)]))

      np.testing.assert_allclose(gt["v_c"], v_c_dense, rtol=1e-10)
      np.testing.assert_allclose(gt["vif_true"], v_tot_dense / np.sum(v_c_dense), rtol=1e-10)

      # Compare M_c,true and M_c,var
      theta = truth.item_mean(metric, sigma, p)
      mu_c = counts.astype(np.float64) * theta
      g = len(counts)
      denom = (n * theta) ** 2
      m_true_dense = g * np.sum(v_c_dense + mu_c**2) / denom
      m_var_dense = 1.0 + g * np.sum(v_c_dense) / denom
      np.testing.assert_allclose(gt["m_c_true"], m_true_dense, rtol=1e-10)
      np.testing.assert_allclose(gt["m_c_var"], m_var_dense, rtol=1e-10)

  def test_graph_ground_truth_against_dense_covariance(self):
    """Verifies graph_ground_truth against dense n x n summand covariance matrix."""
    n = 60
    # 3 clusters of size 20
    counts = np.array([20, 20, 20], dtype=np.int64)
    gamma = 0.5
    sigma = 1.2
    p = 0.8

    # Build small random graph with internal and cross edges
    edges = []
    # Ring of edges
    for i in range(n):
      edges.append([i, (i + 1) % n])
    # Some extra cross-edges
    for i in range(0, n, 5):
      edges.append([i, (i + 17) % n])
    edges_arr = np.array(edges, dtype=np.int64)
    u = np.minimum(edges_arr[:, 0], edges_arr[:, 1])
    v = np.maximum(edges_arr[:, 0], edges_arr[:, 1])
    non_self = u != v
    unique_keys = np.unique(u[non_self] * n + v[non_self])
    all_edges = np.column_stack([unique_keys // n, unique_keys % n])

    cluster_labels = np.repeat([0, 1, 2], 20)
    deg = np.bincount(all_edges[:, 0], minlength=n) + np.bincount(
        all_edges[:, 1], minlength=n
    )

    c_u = cluster_labels[all_edges[:, 0]]
    c_v = cluster_labels[all_edges[:, 1]]
    int_mask = c_u == c_v
    int_edges = all_edges[int_mask]
    cross_edges = all_edges[~int_mask]

    for metric in ("mse", "mae", "accuracy"):
      gt = truth.graph_ground_truth(
          metric,
          counts,
          all_edges,
          deg,
          int_edges,
          cross_edges,
          gamma,
          sigma=sigma,
          p=p,
      )

      # Dense covariance matrix
      cov_dense = np.zeros((n, n), dtype=np.float64)
      v_item = truth.item_variance(metric, sigma, p)
      np.fill_diagonal(cov_dense, v_item)
      for edge_u, edge_v in all_edges:
        denom = math.sqrt(
            (1.0 + (gamma**2) * deg[edge_u]) * (1.0 + (gamma**2) * deg[edge_v])
        )
        rho_uv = (gamma**2) / denom
        cov_val = truth.summand_covariance(metric, rho_uv, sigma, p)
        cov_dense[edge_u, edge_v] = cov_val
        cov_dense[edge_v, edge_u] = cov_val

      v_tot_dense = float(np.sum(cov_dense))
      np.testing.assert_allclose(gt["v_tot"], v_tot_dense, rtol=1e-10)

      # Sum V_c
      v_c_dense = 0.0
      for c in (0, 1, 2):
        idx_c = np.where(cluster_labels == c)[0]
        v_c_dense += float(np.sum(cov_dense[np.ix_(idx_c, idx_c)]))

      np.testing.assert_allclose(gt["sum_v_c"], v_c_dense, rtol=1e-10)
      np.testing.assert_allclose(gt["vif_true"], v_tot_dense / v_c_dense, rtol=1e-10)

  # -------------------------------------------------------------------------
  # 3. Section 3.3 Circular Pool Truth vs Explicit Rotation Enumeration
  # -------------------------------------------------------------------------

  def test_circular_pool_against_explicit_rotations(self):
    """Verifies circular_pool_ground_truth against exhaustive rotation enumeration (N=50)."""
    n_pop = 50
    n = 20
    counts = np.array([5, 10, 5], dtype=np.int64)

    # Random summands
    pool = self.rng.gamma(2.0, 1.0, size=n_pop)
    theta_pop, circ_cov = truth.circular_pool_autocovariance(pool)
    gt = truth.circular_pool_ground_truth(counts, theta_pop, circ_cov)

    # Explicit enumeration of all N rotations
    all_slices = []
    for shift in range(n_pop):
      rot = np.roll(pool, -shift)
      all_slices.append(rot[:n])
    all_slices = np.array(all_slices)  # shape (50, 20)

    # Population covariance between slice item i and slice item j
    # E[(s_i - theta)(s_j - theta)] averaged over all 50 starting points
    mean_slice = np.mean(all_slices, axis=0)
    np.testing.assert_allclose(mean_slice, theta_pop, rtol=1e-10)

    # Covariance of slice sum
    slice_sums = np.sum(all_slices, axis=1)  # shape (50,)
    v_tot_enum = float(np.mean((slice_sums - n * theta_pop) ** 2))
    np.testing.assert_allclose(gt["v_tot"], v_tot_enum, rtol=1e-10)

    # Variances of windows
    endpoints = np.cumsum([0] + list(counts))
    v_c_enum = np.zeros(len(counts), dtype=np.float64)
    for c_idx, nc in enumerate(counts):
      idx_c = range(endpoints[c_idx], endpoints[c_idx + 1])
      c_sums = np.sum(all_slices[:, idx_c], axis=1)
      v_c_enum[c_idx] = float(np.mean((c_sums - nc * theta_pop) ** 2))

    np.testing.assert_allclose(gt["v_c"], v_c_enum, rtol=1e-10)
    np.testing.assert_allclose(gt["vif_true"], v_tot_enum / np.sum(v_c_enum), rtol=1e-10)

  # -------------------------------------------------------------------------
  # 4. Hand-Computed Values for Formulas (default_m_c, edge_n_eff, h)
  # -------------------------------------------------------------------------

  def test_hand_computed_values(self):
    """Verifies default_m_c_formula, edge_n_eff, and count_guard_h against hand computation."""
    # 1. default_m_c_formula
    # Equal counts: exactly M
    self.assertEqual(truth.default_m_c_formula([5, 5, 5], 4.0), 4.0)
    # Sizes [1, 3], M=4 -> branch a gives 1.5 * 4 = 6.0
    self.assertAlmostEqual(truth.default_m_c_formula([1, 3], 4.0), 6.0, places=12)
    # Sizes [1, 3], M=1 -> branch b gives 1.25
    self.assertAlmostEqual(truth.default_m_c_formula([1, 3], 1.0), 1.25, places=12)

    # 2. edge_n_eff
    # [1, 1, 1] on 3 edges -> 3.0
    res_1 = np.array([1.0, 1.0, 1.0, 1.0])
    counts_1 = np.array([1, 1, 1, 1])
    edges_1 = np.array([[0, 1], [1, 2], [2, 3]])
    self.assertAlmostEqual(
        truth.compute_edge_n_eff(res_1, counts_1, edges_1), 3.0, places=12
    )
    # Empty edges -> 0.0
    self.assertEqual(
        truth.compute_edge_n_eff(res_1, counts_1, np.zeros((0, 2), dtype=np.int64)),
        0.0,
    )

    # 3. count_guard_h
    # Equal counts -> 1.0
    self.assertEqual(truth.compute_count_guard_h([10, 10, 10, 10]), 1.0)
    # The (4,4) + (3,1)x10 block from Section 4 T2b
    block_22 = [4, 4] + [3, 1] * 10
    h_val_22 = truth.compute_count_guard_h(block_22)
    c_f = np.array(block_22, dtype=np.float64)
    h1 = float(np.mean(c_f[:-1] * c_f[1:])) / (float(np.mean(c_f)) ** 2)
    c_f2 = c_f**2
    h2 = float(np.mean(c_f2[:-1] * c_f2[1:])) / (float(np.mean(c_f2)) ** 2)
    self.assertAlmostEqual(h1, 0.850, places=2)
    self.assertAlmostEqual(h2, 0.755, places=2)
    self.assertLessEqual(h_val_22, 1.1)

    # Also verify on repeated T2b counts (220 windows)
    counts_t2b = np.array(block_22 * 10, dtype=np.int64) * 25
    self.assertEqual(len(counts_t2b), 220)
    self.assertEqual(int(np.sum(counts_t2b)), 12000)
    self.assertLessEqual(truth.compute_count_guard_h(counts_t2b), 1.1)

  # -------------------------------------------------------------------------
  # 5. default_m_c_formula Agreement with Library ClusterSketch.default_m_c
  # -------------------------------------------------------------------------

  def test_default_m_c_formula_matches_library(self):
    """Guards G7: verifies default_m_c_formula matches ClusterSketch.default_m_c to 1e-12 on 20 random count vectors."""
    for s in range(20):
      rng = np.random.default_rng(2026 + s)
      g = int(rng.integers(5, 50))
      counts = rng.integers(1, 100, size=g)
      totals = rng.uniform(0.1, 10.0, size=g)
      ids = np.arange(g, dtype=np.int64)
      shard = cluster_sketch.PartialClusterShard(ids, totals, counts)
      sk = shard.to_cluster_sketch()

      for m_item in (1.0, 1.5, 3.0, math.pi / 2.0, 16.0):
        val_formula = truth.default_m_c_formula(counts, m_item)
        val_library = sk.default_m_c(m_item)
        self.assertAlmostEqual(val_formula, val_library, places=12)

  # -------------------------------------------------------------------------
  # 6. T2b Latent Rho Solve
  # -------------------------------------------------------------------------

  def test_t2b_latent_rho_reproduces_vif_4(self):
    """Verifies t2b_latent_rho reproduces within-window summand VIF of 4 to 1e-10."""
    w_count = 100
    rho = truth.t2b_latent_rho(window_count=w_count, target_vif=4.0)
    # For MSE with sigma=1, Cov(s_i, s_j) = 2 * rho^2, Var(s_i) = 2
    # Within-window VIF = [w_count * 2 + w_count*(w_count - 1)*Cov] / (w_count * 2)
    cov_val = truth.cov_mse(rho, sigma=1.0)
    v_item = truth.item_variance("mse", sigma=1.0)
    window_var = w_count * v_item + w_count * (w_count - 1) * cov_val
    sum_var = w_count * v_item
    vif_realized = window_var / sum_var
    self.assertAlmostEqual(vif_realized, 4.0, places=10)

  # -------------------------------------------------------------------------
  # 7. T3 Drift Ground Truth Verification (B3 Regression Test)
  # -------------------------------------------------------------------------

  def test_t3_drift_ground_truth_matches_dense(self):
    """B3 regression test: verifies t3_drift_ground_truth matches dense computation."""
    n = 200
    g = 10
    sigma = 1.5
    delta = 0.8
    gt = truth.t3_drift_ground_truth(n, g, sigma, delta)

    # Dense verification
    n_half = n // 2
    item_means = np.concatenate([
        np.full(n_half, sigma**2),
        np.full(n - n_half, sigma**2 * (1.0 + delta)),
    ])
    item_vars = np.concatenate([
        np.full(n_half, 2.0 * (sigma**4)),
        np.full(n - n_half, 2.0 * (sigma**4) * ((1.0 + delta) ** 2)),
    ])
    window_ids = (np.arange(n, dtype=np.int64) * g) // n

    dense_mu_c = np.zeros(g, dtype=np.float64)
    dense_v_c = np.zeros(g, dtype=np.float64)
    for i in range(n):
      w = window_ids[i]
      dense_mu_c[w] += item_means[i]
      dense_v_c[w] += item_vars[i]

    dense_theta = float(np.mean(item_means))
    dense_v_tot = float(np.sum(item_vars))
    denom_m = float((n * dense_theta) ** 2)
    dense_m_c_true = float(g * np.sum(dense_v_c + dense_mu_c**2) / denom_m)
    dense_m_c_var = float(1.0 + g * np.sum(dense_v_c) / denom_m)

    self.assertAlmostEqual(gt["theta"], dense_theta, places=12)
    self.assertAlmostEqual(gt["v_tot"], dense_v_tot, places=12)
    np.testing.assert_allclose(gt["v_c"], dense_v_c, rtol=1e-12)
    self.assertAlmostEqual(gt["m_c_true"], dense_m_c_true, places=12)
    self.assertAlmostEqual(gt["m_c_var"], dense_m_c_var, places=12)
    self.assertAlmostEqual(gt["vif_true"], 1.0, places=12)


if __name__ == "__main__":
  absltest.main()
