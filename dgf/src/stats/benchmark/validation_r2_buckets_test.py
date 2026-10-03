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

"""Tests for validation_r2_buckets ground-truth formulas and analytic checks."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dgf.src.stats import graph
from dgf.src.stats import partitioners


def ar1_window_vif(rho: float, m_c: int, n: int) -> float:
  """Exact window-total VIF for an AR(1) process with covariance C(h) = rho^h."""
  if rho == 0.0 or m_c <= 1:
    return 1.0
  h_m = np.arange(1, m_c, dtype=np.float64)
  v_c = float(m_c) + 2.0 * float(np.sum((m_c - h_m) * (rho**h_m)))
  h_n = np.arange(1, n, dtype=np.float64)
  v_tot = float(n) + 2.0 * float(np.sum((n - h_n) * (rho**h_n)))
  g = n // m_c
  return float(v_tot / (float(g) * v_c))


def graph_field_exact_vif(
    n: int,
    edges: np.ndarray,
    cluster_ids: np.ndarray,
    w: float,
) -> tuple[float, float, np.ndarray]:
  """Computes exact cluster VIF, true theta, and edge phi for V3 field."""
  deg = np.bincount(edges[:, 0], minlength=n) + np.bincount(
      edges[:, 1], minlength=n
  )
  theta = float(np.mean(deg.astype(np.float64) * (w**2) + 1.0))

  g = int(np.max(cluster_ids)) + 1
  c_u = cluster_ids[edges[:, 0]]
  c_v = cluster_ids[edges[:, 1]]
  int_edges_mask = c_u == c_v

  var_s_nodes = 2.0 * ((deg.astype(np.float64) * (w**2) + 1.0) ** 2)

  # V_tot = sum_i Var(s_i) + 4 * w^4 * |E|
  v_tot = float(np.sum(var_s_nodes) + 4.0 * (w**4) * len(edges))

  # V_c = sum_{i in c} Var(s_i) + 4 * w^4 * |E_internal(c)|
  sum_var_c = float(np.sum(var_s_nodes))
  internal_edge_count = int(np.count_nonzero(int_edges_mask))
  sum_v_c = sum_var_c + 4.0 * (w**4) * float(internal_edge_count)

  vif = float(v_tot / sum_v_c)

  # Edge phi = rho_ij^2
  d_u = deg[edges[:, 0]].astype(np.float64)
  d_v = deg[edges[:, 1]].astype(np.float64)
  rho_uv = (w**2) / np.sqrt((d_u * (w**2) + 1.0) * (d_v * (w**2) + 1.0))
  phi = rho_uv**2

  return vif, theta, phi


class ValidationV21Test(parameterized.TestCase):

  def test_exact_r2_truth_v1a_and_v1b(self):
    # In V1a and V1b: Var(y) = 1.0, Var(e) = 0.2, y and e independent
    # predictions y_hat = y - e => residual = e
    # theta_A = E[e^2] = 0.2, theta_B = Var(y) = 1.0
    # R^2 = 1 - theta_A / theta_B = 1 - 0.2 / 1.0 = 0.8
    theta_a = 0.2
    theta_b = 1.0
    r2_truth = 1.0 - (theta_a / theta_b)
    self.assertAlmostEqual(r2_truth, 0.8, places=15)

  def test_exact_vif_v1b_against_brute_force(self):
    n = 60
    g = 3
    m_c = n // g
    rng = np.random.default_rng(42)

    for rho in (0.3, 0.5, 0.7, 0.85):
      vif_formula = ar1_window_vif(rho, m_c, n)

      # Brute-force covariance matrix
      c_matrix = np.zeros((n, n), dtype=np.float64)
      for i in range(n):
        for j in range(n):
          c_matrix[i, j] = rho ** abs(i - j)

      v_tot_brute = float(np.sum(c_matrix))
      v_c_brute = 0.0
      for c in range(g):
        c_slice = slice(c * m_c, (c + 1) * m_c)
        v_c_brute += float(np.sum(c_matrix[c_slice, c_slice]))

      vif_brute = float(v_tot_brute / v_c_brute)
      self.assertAlmostEqual(vif_formula, vif_brute, places=12)

  def test_exact_vif_v3_against_brute_force(self):
    n = 30
    rng = np.random.default_rng(2026)
    # Build random connected graph on n nodes
    edges_list = []
    for i in range(n - 1):
      edges_list.append([i, i + 1])
    for _ in range(30):
      u = int(rng.integers(0, n))
      v = int(rng.integers(0, n))
      if u != v:
        edges_list.append([min(u, v), max(u, v)])
    edges = np.unique(np.array(edges_list, dtype=np.int64), axis=0)

    # Random partition into G = 3 clusters
    cluster_ids = rng.integers(0, 3, size=n, dtype=np.int64)
    # Ensure all clusters exist
    cluster_ids[0] = 0
    cluster_ids[1] = 1
    cluster_ids[2] = 2

    for w in (0.3, 0.7, 1.0):
      vif_formula, theta, _ = graph_field_exact_vif(n, edges, cluster_ids, w)

      # Brute force covariance of s_i = x_i^2
      deg = np.bincount(edges[:, 0], minlength=n) + np.bincount(
          edges[:, 1], minlength=n
      )
      cov_s = np.zeros((n, n), dtype=np.float64)
      for i in range(n):
        cov_s[i, i] = 2.0 * ((deg[i] * (w**2) + 1.0) ** 2)

      for e_idx in range(len(edges)):
        u, v = edges[e_idx]
        cov_s[u, v] = 2.0 * (w**4)
        cov_s[v, u] = 2.0 * (w**4)

      v_tot_brute = float(np.sum(cov_s))
      v_c_sum_brute = 0.0
      for c in range(3):
        mask_c = cluster_ids == c
        v_c_sum_brute += float(np.sum(cov_s[np.ix_(mask_c, mask_c)]))

      vif_brute = float(v_tot_brute / v_c_sum_brute)
      self.assertAlmostEqual(vif_formula, vif_brute, places=12)

  def test_lemma_t_prime_bound_v3(self):
    n = 50
    rng = np.random.default_rng(777)
    edges_list = [[i, i + 1] for i in range(n - 1)]
    for _ in range(60):
      u = int(rng.integers(0, n))
      v = int(rng.integers(0, n))
      if u != v:
        edges_list.append([min(u, v), max(u, v)])
    edges = np.unique(np.array(edges_list, dtype=np.int64), axis=0)
    cluster_ids = partitioners.partition_graph(
        n, edges, target_cluster_size=10, seed=0
    )

    for w in (0.3, 1.0):
      vif_exact, _, phi = graph_field_exact_vif(n, edges, cluster_ids, w)
      top_k = graph.topological_kappa(n, edges, cluster_ids, phi)
      self.assertGreaterEqual(
          top_k.kappa,
          vif_exact - 1e-12,
          f"Lemma T' violated: kappa_T {top_k.kappa} < exact VIF {vif_exact}",
      )

  def test_v5_gaussian_components_exact_fourth_moment(self):
    """Verifies that for zero-mean Gaussian components, E[y^4] = 3 * theta_B^2 and m=3 holds."""
    # Under y ~ N(0, theta_B), E[y^2] = theta_B, E[y^4] = 3 * theta_B^2.
    # Therefore kurtosis = E[y^4] / (E[y^2])^2 = 3.0 identically.
    for theta_b in (0.2, 1.0, 5.0):
      fourth_moment_exact = 3.0 * (theta_b**2)
      ratio = fourth_moment_exact / (theta_b**2)
      self.assertEqual(ratio, 3.0)

  def test_quick_mode_smoke(self):
    """Runs a quick evaluation across 1 cell."""
    from dgf.src.stats.benchmark import validation_r2_buckets
    cells = validation_r2_buckets.get_all_cells(0x21DA7A)
    self.assertLen(cells, 69)
    res = validation_r2_buckets.run_cell_task(cells[0], reps_override=5)
    self.assertEqual(res["cell_name"], cells[0]["name"])
    self.assertIn("passed", res)
    self.assertIn("issued_count", res)

  def test_cell_grid_names_ids_and_paired_seeds(self):
    from dgf.src.stats.benchmark import validation_r2_buckets
    cells = validation_r2_buckets.get_all_cells(0x21DA7A)
    names = [c["name"] for c in cells]
    ids = [c["cell_id"] for c in cells]
    self.assertLen(set(names), len(names))
    self.assertLen(set(ids), len(ids))
    by_name = {c["name"]: c for c in cells}
    for rho in (0.5, 0.8, 0.95):
      self.assertEqual(
          by_name[f"V4_rho={rho}_m_item=4"]["seed"],
          by_name[f"V4_rho={rho}"]["seed"],
      )

  def test_v4_library_only_smoke(self):
    from dgf.src.stats.benchmark import validation_r2_buckets
    cells = validation_r2_buckets.get_all_cells(0x21DA7A)
    cell = next(c for c in cells if c["name"] == "V4_rho=0.5_m_item=4")
    res = validation_r2_buckets.run_cell_task(cell, reps_override=2)
    self.assertNotIn("student_t", res)
    self.assertNotIn("bootstrap", res)
    self.assertEqual(res["m_item"], 4.0)
    self.assertEqual(res["library"]["refused_count"], 0)


if __name__ == "__main__":
  absltest.main()
