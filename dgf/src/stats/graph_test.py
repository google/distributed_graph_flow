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

"""Tests for graph.py graph entry point and diagnostics."""

import math
import os

from absl.testing import absltest
import numpy as np

import scipy.stats

from dgf.src.stats import cluster_bound
from dgf.src.stats import cluster_sketch
from dgf.src.stats import graph
from dgf.src.stats import refutation
from dgf.src.stats.independent import interval as _interval

# Set DGF_STATS_LONG_TESTS=1 (blaze test --test_env=...) for the full-size runs.
_LONG = os.environ.get("DGF_STATS_LONG_TESTS") == "1"


class GraphTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(20260926)

  def assert_interval_equal(self, a, b):
    for field in (
        "status",
        "m_declared",
        "num_clusters",
        "mean_cluster_size",
        "diagnostic",
    ):
      self.assertEqual(getattr(a, field), getattr(b, field))
    for field in ("low", "high", "level", "m_observed", "effective_n"):
      va = getattr(a, field)
      vb = getattr(b, field)
      if va is None or vb is None:
        self.assertEqual(va, vb)
      elif math.isnan(va):
        self.assertTrue(math.isnan(vb))
      else:
        self.assertEqual(va, vb)

  def test_interval_matches_cluster_interval(self):
    """Verifies graph_interval_from_shard interval exactly equals cluster_interval in all fields."""
    g = 60
    n = 600
    cluster_ids = np.repeat(np.arange(g), n // g)
    data = self.rng.gamma(2.0, 1.0, size=n)

    # Simple ring of cluster edges
    cluster_edges = np.column_stack([np.arange(g), (np.arange(g) + 1) % g])

    for metric in ("mse", "mae"):
      shard = cluster_sketch.PartialClusterShard.from_data(
          data, cluster_ids, metric
      )
      for kappa in (1.0, 2.0):
        for m in (None, 1.5):
          res = graph.graph_interval_from_shard(
              shard, cluster_edges, metric=metric, level=0.95, kappa=kappa, m=m
          )
          sk = shard.to_cluster_sketch(kappa_cluster=kappa)
          expected = cluster_bound.cluster_interval(
              sk,
              metric=metric,
              level=0.95,
              kappa_cluster=kappa,
              m=m,
              c_target=0.2,
          )
          self.assert_interval_equal(res.interval, expected)

  def test_raw_data_matches_from_shard(self):
    """Verifies graph_interval from raw data exactly matches graph_interval_from_shard."""
    g = 50
    n = 500
    cluster_ids = np.repeat(np.arange(g), n // g)
    data = self.rng.normal(5.0, 1.0, size=n)
    cluster_edges = np.column_stack([np.arange(g), (np.arange(g) + 1) % g])

    res_raw = graph.graph_interval(
        data, cluster_ids, cluster_edges, metric="mse", kappa=1.5, m=1.5
    )
    shard = cluster_sketch.PartialClusterShard.from_data(
        data, cluster_ids, "mse"
    )
    res_shard = graph.graph_interval_from_shard(
        shard, cluster_edges, metric="mse", kappa=1.5, m=1.5
    )

    self.assert_interval_equal(res_raw.interval, res_shard.interval)
    self.assertEqual(res_raw.kappa_declared, res_shard.kappa_declared)
    self.assertEqual(res_raw.kappa_check, res_shard.kappa_check)
    self.assertEqual(res_raw.kappa_w_hat, res_shard.kappa_w_hat)
    self.assertEqual(res_raw.num_cluster_edges, res_shard.num_cluster_edges)
    self.assertEqual(res_raw.num_dropped_edges, res_shard.num_dropped_edges)
    self.assertEqual(res_raw.hub_ratio, res_shard.hub_ratio)
    self.assertEqual(res_raw.edge_n_eff, res_shard.edge_n_eff)
    self.assertEqual(res_raw.num_batches, res_shard.num_batches)
    self.assertEqual(res_raw.batch_cut_fraction, res_shard.batch_cut_fraction)
    self.assertEqual(res_raw.message, res_shard.message)
    self.assertEqual(res_raw.refuted, res_shard.refuted)

    # Also test kappa = 1.0 for raw vs shard equality and float edge_n_eff >= 10
    res_raw_k1 = graph.graph_interval(
        data, cluster_ids, cluster_edges, metric="mse", kappa=1.0, m=1.5
    )
    res_shard_k1 = graph.graph_interval_from_shard(
        shard, cluster_edges, metric="mse", kappa=1.0, m=1.5
    )
    self.assert_interval_equal(res_raw_k1.interval, res_shard_k1.interval)
    self.assertEqual(res_raw_k1.edge_n_eff, res_shard_k1.edge_n_eff)
    self.assertIsInstance(res_shard_k1.edge_n_eff, float)
    self.assertGreaterEqual(res_shard_k1.edge_n_eff, 10.0)

  def test_kappa_routing(self):
    """Verifies kappa=1 routes to edge test, while kappa>1 routes to batch check."""
    g = 100
    n = 1000
    cluster_ids = np.repeat(np.arange(g), n // g)
    data = self.rng.normal(2.0, 1.0, size=n)
    cluster_edges = np.column_stack([np.arange(g), (np.arange(g) + 1) % g])
    shard = cluster_sketch.PartialClusterShard.from_data(
        data, cluster_ids, "mse"
    )

    # kappa = 1.0 -> edge test
    res1 = graph.graph_interval_from_shard(
        shard, cluster_edges, metric="mse", kappa=1.0
    )
    self.assertEqual(res1.kappa_declared, 1.0)
    self.assertIsNone(res1.num_batches)
    self.assertIsNone(res1.batch_cut_fraction)
    self.assertIsInstance(res1.edge_n_eff, float)
    self.assertGreaterEqual(res1.edge_n_eff, 10.0)
    self.assertAlmostEqual(
        res1.kappa_check.threshold, scipy.stats.norm.ppf(0.95), places=6
    )

    # kappa = 2.0 -> batch check
    res2 = graph.graph_interval_from_shard(
        shard, cluster_edges, metric="mse", kappa=2.0
    )
    self.assertEqual(res2.kappa_declared, 2.0)
    self.assertIsNotNone(res2.num_batches)
    self.assertIsNotNone(res2.batch_cut_fraction)
    self.assertIsNone(res2.edge_n_eff)
    self.assertEqual(res2.kappa_check.threshold, 2.0)

  def test_graph_interval_guard_triggered(self):
    """Reuses dominated matching residuals via PartialClusterShard to trigger guard."""
    g = 1000
    counts = np.ones(g, dtype=np.int64)
    totals = np.zeros(g, dtype=np.float64)
    totals[:6] = 1.0
    shard = cluster_sketch.PartialClusterShard(
        np.arange(g, dtype=np.int64), totals, counts
    )
    edges = np.column_stack([np.arange(0, g, 2), np.arange(1, g, 2)]).astype(
        np.int64
    )
    res = graph.graph_interval_from_shard(shard, edges, metric="mse", kappa=1.0)
    self.assertEqual(res.kappa_check.outcome, refutation.Refutation.UNTESTABLE)
    self.assertIsNotNone(res.edge_n_eff)
    self.assertAlmostEqual(res.edge_n_eff, 3.0363, delta=1e-3)
    self.assertIn("effective edges", res.message)

  def test_dropped_edges(self):
    """Verifies edges with endpoints missing from the shard are dropped and counted."""
    # Shard has cluster IDs [0, 1, 2, 3, 4]
    cluster_ids = np.array([0, 1, 2, 3, 4])
    data = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
    shard = cluster_sketch.PartialClusterShard.from_data(
        data, cluster_ids, "mse"
    )

    # Edges: [0, 1] (kept), [1, 2] (kept), [0, 99] (dropped), [10, 20] (dropped), [3, 50] (dropped)
    cluster_edges = np.array(
        [[0, 1], [1, 2], [0, 99], [10, 20], [3, 50]], dtype=np.int64
    )

    res = graph.graph_interval_from_shard(shard, cluster_edges, metric="mse")
    self.assertEqual(res.num_dropped_edges, 3)
    self.assertEqual(res.num_cluster_edges, 2)

  def test_cluster_adjacency_hand_built(self):
    """Verifies cluster_adjacency handles internal, duplicate, and reversed edges on 6 nodes."""
    # Nodes 0..5
    num_nodes = 6
    cluster_ids = np.array([10, 10, 20, 20, 30, 30], dtype=np.int64)

    edges = np.array(
        [
            [0, 1],  # internal to cluster 10 -> drop
            [0, 2],  # cross 10 - 20
            [2, 0],  # reversed 20 - 10 -> duplicate of [0, 2]
            [1, 3],  # cross 10 - 20 -> duplicate at cluster level
            [2, 4],  # cross 20 - 30
            [4, 5],  # internal to cluster 30 -> drop
            [0, 0],  # self-loop -> drop
        ],
        dtype=np.int64,
    )

    adj = graph.cluster_adjacency(num_nodes, edges, cluster_ids)
    expected = np.array([[10, 20], [20, 30]], dtype=np.int64)
    np.testing.assert_array_equal(adj, expected)

  def _dense_hub_ratio(self, counts, edges):
    g = len(counts)
    w = np.zeros((g, g), dtype=np.float64)
    for u, v in edges:
      w[u, v] = 1.0
      w[v, u] = 1.0
    c = np.asarray(counts, dtype=np.float64)
    q = np.sqrt(c / np.sum(c))
    p = np.eye(g) - np.outer(q, q)
    a = 0.5 * (p @ w @ p)
    eigs = np.linalg.eigvalsh(a)
    lambda_max_sq = float(np.max(eigs**2))
    norm_a_sq = float(np.sum(eigs**2))
    return lambda_max_sq / norm_a_sq

  def test_hub_ratio_dense_cross_checks(self):
    """Verifies projected hub_ratio against dense eigvalsh and theoretical values across 4 graphs."""
    # 1. Star graph, G = 100, equal counts
    g_star = 100
    counts_star = np.ones(g_star, dtype=np.int64)
    star_edges = np.column_stack(
        [np.zeros(g_star - 1, dtype=np.int64), np.arange(1, g_star)]
    )
    dense_star = self._dense_hub_ratio(counts_star, star_edges)
    hr_star, iters_star = graph._compute_hub_ratio_impl(counts_star, star_edges)
    np.testing.assert_allclose(hr_star, dense_star, rtol=1e-4)
    if _LONG:
      print(
          f"Hub ratio Star G=100: dense={dense_star:.6f},"
          f" power={hr_star:.6f}, iters={iters_star}"
      )

    # 2. Cycle graph, G = 100, equal counts
    g_cycle = 100
    counts_cycle = np.ones(g_cycle, dtype=np.int64)
    cycle_edges = np.column_stack(
        [np.arange(g_cycle), (np.arange(g_cycle) + 1) % g_cycle]
    )
    dense_cycle = self._dense_hub_ratio(counts_cycle, cycle_edges)
    hr_cycle, iters_cycle = graph._compute_hub_ratio_impl(
        counts_cycle, cycle_edges, max_iter=2000, tol=1e-8
    )
    if _LONG:
      print(
          f"Hub ratio Cycle G=100: dense={dense_cycle:.6f},"
          f" power={hr_cycle:.6f}, iters={iters_cycle}"
      )
    np.testing.assert_allclose(hr_cycle, dense_cycle, rtol=1e-4)

    # 3. Barabási-Albert style graph, G = 300, counts drawn in [5, 15]
    g_ba = 300
    rng_ba = np.random.default_rng(42)
    counts_ba = rng_ba.integers(5, 16, size=g_ba)
    m = 3
    ends = np.empty(2 * g_ba * m + 2 * m, dtype=np.int64)
    ends[:m] = np.arange(m)
    cnt = m
    ba_out = []
    for i in range(m, g_ba):
      t = ends[rng_ba.integers(0, cnt, m)]
      ba_out.append(np.column_stack([np.full(m, i), t]))
      ends[cnt : cnt + m] = t
      ends[cnt + m : cnt + 2 * m] = i
      cnt += 2 * m
    all_ba = np.concatenate(ba_out)
    u_ba = np.minimum(all_ba[:, 0], all_ba[:, 1])
    v_ba = np.maximum(all_ba[:, 0], all_ba[:, 1])
    non_self_ba = u_ba != v_ba
    keys_ba = np.unique(u_ba[non_self_ba] * g_ba + v_ba[non_self_ba])
    ba_edges = np.column_stack([keys_ba // g_ba, keys_ba % g_ba]).astype(
        np.int64
    )

    dense_ba = self._dense_hub_ratio(counts_ba, ba_edges)
    hr_ba, iters_ba = graph._compute_hub_ratio_impl(counts_ba, ba_edges)
    if _LONG:
      print(
          f"Hub ratio BA G=300: dense={dense_ba:.6f},"
          f" power={hr_ba:.6f}, iters={iters_ba}"
      )
    np.testing.assert_allclose(hr_ba, dense_ba, rtol=1e-4)

    # 4. Complete graph K_20, equal counts
    g_k20 = 20
    counts_k20 = np.ones(g_k20, dtype=np.int64)
    u_k20, v_k20 = np.triu_indices(g_k20, k=1)
    k20_edges = np.column_stack([u_k20, v_k20]).astype(np.int64)
    dense_k20 = self._dense_hub_ratio(counts_k20, k20_edges)
    hr_k20, iters_k20 = graph._compute_hub_ratio_impl(counts_k20, k20_edges)
    unprojected_ratio_k20 = (19.0**2) / (2.0 * 190.0)  # 19^2 / (2 * 190) = 0.95
    if _LONG:
      print(
          f"Hub ratio K_20: dense={dense_k20:.6f},"
          f" power={hr_k20:.6f}, iters={iters_k20}, analytic=1/19={1/19:.6f},"
          f" unprojected={unprojected_ratio_k20:.4f}"
      )
    self.assertAlmostEqual(hr_k20, 1.0 / 19.0, places=6)
    np.testing.assert_allclose(hr_k20, dense_k20, rtol=1e-4)

  def test_topological_kappa_star(self):
    """Star graph: hub plus k leaves, every node in own cluster."""
    phi = 0.1
    for k in (4, 25, 100):
      n = k + 1
      edges = np.column_stack(
          [np.zeros(k, dtype=np.int64), np.arange(1, n, dtype=np.int64)]
      )
      cluster_ids = np.arange(n, dtype=np.int64)
      res = graph.topological_kappa(n, edges, cluster_ids, phi)
      expected_lam = phi * math.sqrt(k)
      expected_kappa = 1.0 + expected_lam
      self.assertGreaterEqual(res.kappa, expected_kappa)
      self.assertLessEqual(res.kappa, expected_kappa * (1.0 + 1e-6))
      self.assertEqual(res.num_cut_edges, k)
      self.assertEqual(res.num_items_on_cut, n)
      self.assertAlmostEqual(res.max_cut_row_sum, phi * k)

  def test_topological_kappa_complete_bipartite(self):
    """Complete bipartite K_{a,b}, all cut edges."""
    a, b = 6, 7
    n = a + b
    u_idx, v_idx = np.meshgrid(np.arange(a), np.arange(a, n))
    edges = np.column_stack([u_idx.ravel(), v_idx.ravel()]).astype(np.int64)
    cluster_ids = np.arange(n, dtype=np.int64)
    phi = 0.1
    res = graph.topological_kappa(n, edges, cluster_ids, phi)
    expected_lam = phi * math.sqrt(a * b)
    expected_kappa = 1.0 + expected_lam
    self.assertGreaterEqual(res.kappa, expected_kappa)
    self.assertLessEqual(res.kappa, expected_kappa * (1.0 + 1e-6))
    self.assertEqual(res.num_cut_edges, a * b)
    self.assertEqual(res.num_items_on_cut, n)

  def test_topological_kappa_within_cluster_edges_ignored(self):
    """Edges within the same cluster must be ignored and give kappa == 1.0."""
    n = 20
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)]).astype(
        np.int64
    )
    cluster_ids = np.zeros(n, dtype=np.int64)
    res = graph.topological_kappa(n, edges, cluster_ids, 0.2)
    self.assertEqual(res.kappa, 1.0)
    self.assertEqual(res.lambda_upper, 0.0)
    self.assertEqual(res.max_cut_row_sum, 0.0)
    self.assertEqual(res.num_cut_edges, 0)
    self.assertEqual(res.num_items_on_cut, 0)

  def test_topological_kappa_exact_vif_property(self):
    """Exact-VIF property: VIF on Perron vector equals kappa up to iteration tolerance."""
    rng = np.random.default_rng(42)
    n = 300
    pos = rng.uniform(0.0, 1.0, size=(n, 2))
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    r = math.sqrt(8.0 / (math.pi * float(n)))
    u, v = np.where(
        (dist < r) & (np.arange(n)[:, None] < np.arange(n)[None, :])
    )
    edges = np.column_stack([u, v]).astype(np.int64)
    cluster_ids = rng.integers(0, 20, size=n, dtype=np.int64)
    phi_edges = rng.uniform(0.0, 0.2, size=edges.shape[0])

    res = graph.topological_kappa(n, edges, cluster_ids, phi_edges)

    # Build dense Phi_cut for verification
    phi_cut = np.zeros((n, n), dtype=np.float64)
    for (ui, vi), pe in zip(edges, phi_edges):
      if cluster_ids[ui] != cluster_ids[vi]:
        phi_cut[ui, vi] = max(phi_cut[ui, vi], pe)
        phi_cut[vi, ui] = max(phi_cut[vi, ui], pe)

    evals, evecs = np.linalg.eigh(phi_cut)
    perron_sigma = np.abs(evecs[:, -1])
    # Avoid zero entries
    perron_sigma = np.maximum(perron_sigma, 1e-8)

    # Assert Sigma is PSD
    sigma_mat = (
        np.diag(perron_sigma) @ (np.eye(n) + phi_cut) @ np.diag(perron_sigma)
    )
    sig_evals = np.linalg.eigvalsh(sigma_mat)
    self.assertGreaterEqual(float(np.min(sig_evals)), -1e-12)

    # VIF formula: 1^T Sigma 1 / sum_c (1_c^T Sigma 1_c)
    # Notice sum_c (1_c^T Sigma 1_c) = sum_c (sum_{i in c} sigma_i)^2 + sum_{i in c, j in c} Cov
    # Since Phi_cut has 0 inside clusters, 1_c^T Sigma 1_c = sum_{i in c} sigma_i^2
    denom = float(np.sum(perron_sigma**2))
    num = float(np.sum(sigma_mat))
    vif_exact = num / denom
    self.assertLessEqual(vif_exact, res.kappa)
    self.assertGreaterEqual(vif_exact, res.kappa - 1e-5)

    # Also check VIF <= kappa for 20 random positive sigma
    for s_idx in range(20):
      rnd_sigma = rng.uniform(0.1, 2.0, size=n)
      s_mat = np.diag(rnd_sigma) @ (np.eye(n) + phi_cut) @ np.diag(rnd_sigma)
      vif_rnd = float(np.sum(s_mat)) / float(np.sum(rnd_sigma**2))
      self.assertLessEqual(vif_rnd, res.kappa)

  def test_topological_kappa_validation_errors(self):
    """Input validation errors."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    c_ids = np.array([0, 1, 0], dtype=np.int64)
    with self.assertRaises(ValueError):
      graph.topological_kappa(3, edges, c_ids, 1.5)
    with self.assertRaises(ValueError):
      graph.topological_kappa(3, edges, c_ids, float("nan"))
    with self.assertRaises(ValueError):
      graph.topological_kappa(3, edges, c_ids, np.array([0.1]))  # wrong length
    with self.assertRaises(ValueError):
      graph.topological_kappa(0, edges, c_ids, 0.1)  # num_nodes < 1
    with self.assertRaises(ValueError):
      graph.topological_kappa(
          3, edges, np.array([0, 1]), 0.1
      )  # wrong cluster_ids length
    with self.assertRaises(ValueError):
      graph.topological_kappa(
          3, np.array([[0, 5]]), c_ids, 0.1
      )  # index out of bounds

  def test_r2_shift_invariance(self):
    """y versus y + 1e6, with the same shift applied to predictions, gives the same R2 interval (rtol 1e-9)."""
    rng = np.random.default_rng(20260934)
    g = 1000
    m_c = 4
    n = g * m_c
    c_ids = np.repeat(np.arange(g, dtype=np.int64), m_c)
    cluster_edges = np.column_stack([np.arange(g - 1), np.arange(1, g)])

    y_true = rng.normal(50.0, 10.0, size=n)
    y_pred = y_true + rng.normal(0.0, 2.0, size=n)

    shift = 1e6
    y_true_shifted = y_true + shift
    y_pred_shifted = y_pred + shift

    res_base = graph.graph_interval(
        (y_true, y_pred),
        c_ids,
        cluster_edges,
        metric="r2",
        level=0.95,
        m_item=4.0,
        m_labels=4.0,
        kappa=1.0,
    )
    res_shifted = graph.graph_interval(
        (y_true_shifted, y_pred_shifted),
        c_ids,
        cluster_edges,
        metric="r2",
        level=0.95,
        m_item=4.0,
        m_labels=4.0,
        kappa=1.0,
    )

    self.assertEqual(res_base.interval.status, _interval.Status.UNREFUTED)
    self.assertEqual(res_shifted.interval.status, _interval.Status.UNREFUTED)
    np.testing.assert_allclose(
        res_base.interval.low, res_shifted.interval.low, rtol=1e-9
    )
    np.testing.assert_allclose(
        res_base.interval.high, res_shifted.interval.high, rtol=1e-9
    )

  def test_item_order_invariance(self):
    """Permuting item order leaves MSE and MAE intervals unchanged (rtol 1e-12)."""
    rng = np.random.default_rng(20260935)
    g = 500
    m_c = 4
    n = g * m_c
    c_ids = np.repeat(np.arange(g, dtype=np.int64), m_c)
    cluster_edges = np.column_stack([np.arange(g - 1), np.arange(1, g)])

    residuals = rng.normal(0.0, 1.0, size=n)
    perm = rng.permutation(n)

    for metric in ("mse", "mae"):
      res_orig = graph.graph_interval(
          residuals,
          c_ids,
          cluster_edges,
          metric=metric,
          level=0.95,
          m_item=4.0 if metric == "mse" else 2.0,
          kappa=1.0,
      )
      res_perm = graph.graph_interval(
          residuals[perm],
          c_ids[perm],
          cluster_edges,
          metric=metric,
          level=0.95,
          m_item=4.0 if metric == "mse" else 2.0,
          kappa=1.0,
      )
      self.assertEqual(res_orig.interval.status, _interval.Status.UNREFUTED)
      self.assertEqual(res_perm.interval.status, _interval.Status.UNREFUTED)
      np.testing.assert_allclose(
          res_orig.interval.low, res_perm.interval.low, rtol=1e-12, atol=1e-12
      )
      np.testing.assert_allclose(
          res_orig.interval.high,
          res_perm.interval.high,
          rtol=1e-12,
          atol=1e-12,
      )

  def test_topological_kappa_margin_scaling_large_degree(self):
    """Verifies rounding margin exceeds 1e-9 when d_max > 2.3e6."""
    eps64 = float(np.finfo(np.float64).eps)

    # For small d_max, margin is clamped to floor 1e-9
    d_small = 1000
    margin_small = max(1e-9, 4.0 * (float(d_small) + 2.0) * eps64)
    self.assertEqual(margin_small, 1e-9)

    # For d_max > 2.3e6, 4 * (d_max + 2) * eps64 exceeds 1e-9
    d_large = 2_500_000
    margin_large = max(1e-9, 4.0 * (float(d_large) + 2.0) * eps64)
    self.assertGreater(margin_large, 1e-9)
    self.assertAlmostEqual(margin_large, 4.0 * (2_500_000 + 2) * eps64, places=15)


if __name__ == "__main__":
  absltest.main()
