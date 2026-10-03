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

"""Tests for spacetime.py (space-time confidence intervals)."""

from collections.abc import Sequence
import math
import os
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import scipy.linalg

from dgf.src.stats import graph as _graph
from dgf.src.stats import refutation as _refutation
from dgf.src.stats import spacetime
from dgf.src.stats import temporal as _temporal
from dgf.src.stats.independent import interval as _interval

# Set DGF_STATS_LONG_TESTS=1 (blaze test --test_env=...) for the full-size runs.
_LONG = os.environ.get("DGF_STATS_LONG_TESTS") == "1"


class SpaceTimeTest(parameterized.TestCase):

  def test_tiny_grid_index_and_edges(self):
    """3 nodes on a path, 4 time steps, max_lag = 1: exact edge counts and observed filtering."""
    # Nodes 0, 1, 2 on a path: spatial edges (0, 1), (1, 2)
    spatial_edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    # 4 time steps: 0, 1, 2, 3
    node_ids = []
    times = []
    for u in [0, 1, 2]:
      for t in [0, 1, 2, 3]:
        node_ids.append(u)
        times.append(t)

    index = spacetime.spacetime_items(node_ids, times)
    self.assertEqual(index.num_nodes, 3)
    self.assertEqual(index.num_times, 4)
    self.assertEqual(index.n_items, 12)
    self.assertTrue(index.is_complete_grid)

    phi_s = 0.5
    phi_t = [0.3]
    # Default dominance product phi_cross = phi_s * phi_t = 0.15
    edges, phi = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=1,
        phi_spatial=phi_s,
        phi_temporal=phi_t,
    )

    # Expected edges:
    # 1. Same-time spatial: 2 spatial edges * 4 time steps = 8 edges
    # 2. Same-node temporal: 3 nodes * 3 lag-1 transitions = 9 edges
    # 3. Cross-lagged: 2 spatial edges * 3 transitions * 2 directions = 12 edges
    # Total = 8 + 9 + 12 = 29 edges
    self.assertEqual(len(edges), 29)
    self.assertEqual(len(phi), 29)

    # Verify edge weights
    # 8 spatial edges with weight 0.5
    # 9 temporal edges with weight 0.3
    # 12 cross edges with weight 0.15
    np.testing.assert_allclose(
        np.sort(phi),
        np.sort(
            np.concatenate([
                np.full(8, 0.5),
                np.full(9, 0.3),
                np.full(12, 0.15),
            ])
        ),
    )

    # Incomplete grid: drop node 1 at time 1 (item index 5)
    inc_nodes = [u for u, t in zip(node_ids, times) if not (u == 1 and t == 1)]
    inc_times = [t for u, t in zip(node_ids, times) if not (u == 1 and t == 1)]
    index_inc = spacetime.spacetime_items(inc_nodes, inc_times)
    self.assertFalse(index_inc.is_complete_grid)
    self.assertEqual(index_inc.n_items, 11)

    edges_inc, phi_inc = spacetime.spacetime_edges(
        index_inc,
        spatial_edges,
        max_lag=1,
        phi_spatial=phi_s,
        phi_temporal=phi_t,
    )
    # Dropping item (1, 1) drops:
    # - 2 same-time spatial edges: (0,1)-(1,1) and (1,1)-(2,1)
    # - 2 temporal edges: (1,0)-(1,1) and (1,1)-(1,2)
    # - 4 cross edges: (0,0)-(1,1), (1,1)-(0,2), (2,0)-(1,1), (1,1)-(2,2)
    # Total dropped = 8 edges; remaining = 29 - 8 = 21 edges
    self.assertEqual(len(edges_inc), 21)
    self.assertEqual(len(phi_inc), 21)

  def test_index_validation(self):
    """Rejects duplicate (node, time) pairs, non-integers, and mismatched lengths."""
    with self.assertRaises(ValueError):
      spacetime.spacetime_items([0, 1], [0])  # length mismatch
    with self.assertRaises(ValueError):
      spacetime.spacetime_items([], [])  # empty
    with self.assertRaises(ValueError):
      spacetime.spacetime_items([0, 0], [1, 1])  # duplicate
    with self.assertRaises(ValueError):
      spacetime.spacetime_items([0, 1], [0.5, 1.0])  # float time
    with self.assertRaises(ValueError):
      spacetime.spacetime_items([0, 1], [True, False])  # bool time

  def test_candidates_complete_grid(self):
    """Verifies partition sizes and G = G_s * G_t on complete grid."""
    spatial_edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    node_ids = []
    times = []
    for u in [0, 1, 2]:
      for t in [0, 1, 2, 3]:
        node_ids.append(u)
        times.append(t)
    index = spacetime.spacetime_items(node_ids, times)

    cands = spacetime.candidate_partitions(
        index,
        spatial_edges,
        target_sizes=[2],
        window_lengths=[2],
        seed=0,
    )

    cand_map = {c.name: c for c in cands}
    self.assertIn("community_ts2", cand_map)
    self.assertIn("window_wl2", cand_map)
    self.assertIn("product_ts2_wl2", cand_map)

    # 3 nodes partitioned with target_size=2 produces 2 communities (sizes 2 and 1)
    c_comm = cand_map["community_ts2"]
    self.assertEqual(c_comm.G_t, 1)
    self.assertEqual(c_comm.G_s, 2)
    self.assertEqual(c_comm.G, 2)

    # 4 time steps partitioned with window_length=2 produces 2 windows
    c_win = cand_map["window_wl2"]
    self.assertEqual(c_win.G_s, 1)
    self.assertEqual(c_win.G_t, 2)
    self.assertEqual(c_win.G, 2)

    # Product partition has G = G_s * G_t = 2 * 2 = 4 blocks
    c_prod = cand_map["product_ts2_wl2"]
    self.assertEqual(c_prod.G_s, 2)
    self.assertEqual(c_prod.G_t, 2)
    self.assertEqual(c_prod.G, 4)
    self.assertEqual(c_prod.G, c_prod.G_s * c_prod.G_t)

  def test_incomplete_grid_route_and_monotonicity(self):
    """Incomplete grid uses fallback and records it; masked topological kappa <= complete."""
    spatial_edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    node_ids = []
    times = []
    for u in [0, 1, 2]:
      for t in [0, 1, 2, 3]:
        node_ids.append(u)
        times.append(t)
    index_full = spacetime.spacetime_items(node_ids, times)
    edges_full, phi_full = spacetime.spacetime_edges(
        index_full,
        spatial_edges,
        max_lag=1,
        phi_spatial=0.5,
        phi_temporal=[0.3],
    )

    # Drop (1, 1)
    inc_nodes = [u for u, t in zip(node_ids, times) if not (u == 1 and t == 1)]
    inc_times = [t for u, t in zip(node_ids, times) if not (u == 1 and t == 1)]
    index_inc = spacetime.spacetime_items(inc_nodes, inc_times)
    edges_inc, phi_inc = spacetime.spacetime_edges(
        index_inc, spatial_edges, max_lag=1, phi_spatial=0.5, phi_temporal=[0.3]
    )

    cands = spacetime.candidate_partitions(
        index_inc, spatial_edges, target_sizes=[2], window_lengths=[2]
    )
    sep_fn = spacetime.separable_kappa(1.2, 1.3)

    # Incomplete grid without fallback raises ValueError
    with self.assertRaisesRegex(ValueError, "fallback_kappa_fn"):
      spacetime.choose_partition(
          cands,
          kappa_fn=sep_fn,
          m_item=2.0,
          is_complete_grid=False,
          fallback_kappa_fn=None,
      )

    # Incomplete grid with fallback succeeds and records route
    fallback_fn = spacetime.topological_kappa_fn(
        index_inc.n_items, edges_inc, phi_inc
    )
    choice = spacetime.choose_partition(
        cands,
        kappa_fn=sep_fn,
        m_item=2.0,
        is_complete_grid=False,
        fallback_kappa_fn=fallback_fn,
        edges=edges_inc,
    )
    self.assertEqual(choice.route, "fallback_topological")
    self.assertIn("Incomplete grid", choice.route_reason)

    # Monotonicity: On the same partition (e.g. singleton clusters), masked kappa <= complete kappa
    cand_full = spacetime.candidate_partitions(
        index_full, spatial_edges, target_sizes=[2], window_lengths=[2]
    )[0]
    tk_full = _graph.topological_kappa(
        index_full.n_items, edges_full, cand_full.cluster_ids, phi=phi_full
    )

    # Mask cluster_ids for incomplete grid
    keep_indices = [
        i
        for i, (u, t) in enumerate(zip(node_ids, times))
        if not (u == 1 and t == 1)
    ]
    cand_inc_ids = cand_full.cluster_ids[keep_indices]
    tk_inc = _graph.topological_kappa(
        index_inc.n_items, edges_inc, cand_inc_ids, phi=phi_inc
    )
    self.assertLessEqual(tk_inc.kappa, tk_full.kappa + 1e-9)

  def test_scoring_hand_computed_and_tie_breaking(self):
    """Hand-computed N_eff, deterministic tie-breaking, and kappa_t dominance."""
    # Hand-computed test:
    # 4 clusters with 10 items each: G=4, n=40, n_max=10, m_bar=10
    c1 = spacetime.Candidate(
        name="cand_1",
        cluster_ids=np.repeat(np.arange(4), 10),
        G=4,
        G_s=2,
        G_t=2,
        n_max=10,
        m_bar=10.0,
    )
    # With m_item=2.0, equal cluster sizes give M_c_default = 2.0
    # r_eff = (2.0 - 1.0) / (2.0 - 1.0) = 1.0
    # With kappa = 2.0: N_eff = G / (kappa * r_eff) = 4 / (2.0 * 1.0) = 2.0
    kappa_fn = lambda c: 2.0
    choice = spacetime.choose_partition(
        [c1], kappa_fn=kappa_fn, m_item=2.0, is_complete_grid=True
    )
    self.assertAlmostEqual(choice.winner_r_eff, 1.0, places=9)
    self.assertAlmostEqual(choice.winner_n_eff, 2.0, places=9)

    # Tie-breaking test:
    # Two candidates with identical N_eff:
    # cand_a has G=4, N_eff=2.0
    # cand_b has G=2, N_eff=2.0 (with kappa=1.0)
    c_a = spacetime.Candidate(
        name="b_cand",
        cluster_ids=np.repeat(np.arange(4), 10),
        G=4,
        G_s=2,
        G_t=2,
        n_max=10,
        m_bar=10.0,
    )
    c_b = spacetime.Candidate(
        name="a_cand",
        cluster_ids=np.repeat(np.arange(2), 20),
        G=2,
        G_s=1,
        G_t=2,
        n_max=20,
        m_bar=20.0,
    )

    def tie_kappa(c: spacetime.Candidate) -> float:
      return 2.0 if c.G == 4 else 1.0

    # Both have N_eff = 2.0. Fewer clusters (c_b with G=2) wins!
    choice_tie = spacetime.choose_partition(
        [c_a, c_b], kappa_fn=tie_kappa, m_item=2.0, is_complete_grid=True
    )
    self.assertEqual(choice_tie.winner.name, "a_cand")

    # Name tie-breaking: same N_eff and same G
    c_c = spacetime.Candidate(
        name="z_name",
        cluster_ids=np.repeat(np.arange(2), 20),
        G=2,
        G_s=1,
        G_t=2,
        n_max=20,
        m_bar=20.0,
    )
    c_d = spacetime.Candidate(
        name="a_name",
        cluster_ids=np.repeat(np.arange(2), 20),
        G=2,
        G_s=1,
        G_t=2,
        n_max=20,
        m_bar=20.0,
    )
    choice_name = spacetime.choose_partition(
        [c_c, c_d], kappa_fn=lambda c: 1.0, m_item=2.0, is_complete_grid=True
    )
    self.assertEqual(choice_name.winner.name, "a_name")

    # With kappa_t large, community-only wins
    spatial_edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    node_ids = []
    times = []
    for u in [0, 1, 2]:
      for t in range(10):
        node_ids.append(u)
        times.append(t)
    idx_grid = spacetime.spacetime_items(node_ids, times)
    cands_grid = spacetime.candidate_partitions(
        idx_grid, spatial_edges, target_sizes=[1], window_lengths=[2]
    )

    # kappa_s = 1.0, kappa_t = 100.0
    sep_large_t = spacetime.separable_kappa(1.0, 100.0)
    choice_large_t = spacetime.choose_partition(
        cands_grid,
        kappa_fn=sep_large_t,
        m_item=2.0,
        is_complete_grid=idx_grid.is_complete_grid,
    )
    self.assertTrue(choice_large_t.winner.name.startswith("community_"))

  def test_end_to_end_synthetic_and_drift(self):
    """Runs end-to-end on synthetic separable data; verifies finite interval and drift warning."""
    rng = np.random.default_rng(42)
    num_nodes = 30
    num_times = 400
    spatial_edges = np.column_stack([
        np.arange(num_nodes - 1, dtype=np.int64),
        np.arange(1, num_nodes, dtype=np.int64),
    ])

    node_ids = []
    times = []
    for u in range(num_nodes):
      for t in range(num_times):
        node_ids.append(u)
        times.append(t)

    index = spacetime.spacetime_items(node_ids, times)
    edges, phi = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=1,
        phi_spatial=0.1,
        phi_temporal=[0.1],
    )

    # Generate synthetic separable Gaussian residuals: AR(1) in time x AR(1) along path
    rho_s = 0.2
    rho_t = 0.2
    c_s = rho_s ** np.abs(
        np.subtract.outer(np.arange(num_nodes), np.arange(num_nodes))
    )
    c_t = rho_t ** np.abs(
        np.subtract.outer(np.arange(num_times), np.arange(num_times))
    )
    chol_s = scipy.linalg.cholesky(c_s, lower=True)
    chol_t = scipy.linalg.cholesky(c_t, lower=True)

    # Reshape matrix of normal draws (num_nodes, num_times)
    z = rng.standard_normal((num_nodes, num_times))
    e_mat = chol_s @ z @ chol_t.T
    residuals = e_mat.flatten()

    # Losses for MSE: s = e^2
    s_items = residuals**2

    cands = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[4]
    )
    sep_fn = spacetime.separable_kappa(1.1, 1.1)
    choice = spacetime.choose_partition(
        cands,
        kappa_fn=sep_fn,
        m_item=3.2,
        is_complete_grid=index.is_complete_grid,
        edges=edges,
    )

    res = spacetime.spacetime_interval(
        s_items,
        index,
        choice,
        metric="mse",
        level=0.95,
        m_item=3.2,
    )

    # Verify certified finite interval
    self.assertEqual(res.status, _interval.Status.UNREFUTED)
    self.assertTrue(math.isfinite(res.low))
    self.assertTrue(math.isfinite(res.high))
    self.assertGreater(res.high, res.low)
    self.assertIsNotNone(res.marginal_checks)

    # Drift check under no drift should be NO_WARNING or UNTESTABLE
    self.assertIsNotNone(res.drift)
    assert res.drift is not None
    self.assertIn(
        res.drift.outcome,
        [_temporal.Drift.NO_WARNING, _temporal.Drift.UNTESTABLE],
    )

    # Now inject linear temporal trend into residuals: e_drift = e + 8.0 * (t / T)
    time_ratios = np.array(
        [t / float(num_times) for t in times], dtype=np.float64
    )
    residuals_drift = residuals + 8.0 * time_ratios
    s_drift = residuals_drift**2

    res_drift = spacetime.spacetime_interval(
        s_drift,
        index,
        choice,
        metric="mse",
        level=0.95,
        m_item=3.2,
    )
    # The window-split drift diagnostic must fire WARNING
    self.assertIsNotNone(res_drift.drift)
    assert res_drift.drift is not None
    self.assertEqual(res_drift.drift.outcome, _temporal.Drift.WARNING)
    self.assertIn("drift warning", res_drift.drift.message)

  def test_tie_breaking_prefer_gt2(self):
    """Tie-breaking prefers G_t >= 2 over G_t = 1 at equal N_eff."""
    # Candidate A: G_t = 1, fewer clusters G = 2, N_eff = 2.0
    c_gt1 = spacetime.Candidate(
        name="cand_gt1",
        cluster_ids=np.repeat(np.arange(2), 20),
        G=2,
        G_s=2,
        G_t=1,
        n_max=20,
        m_bar=20.0,
    )
    # Candidate B: G_t = 2, more clusters G = 4, N_eff = 2.0
    c_gt2 = spacetime.Candidate(
        name="cand_gt2",
        cluster_ids=np.repeat(np.arange(4), 10),
        G=4,
        G_s=2,
        G_t=2,
        n_max=10,
        m_bar=10.0,
    )

    # Both have N_eff = 2.0. Even though c_gt1 has fewer clusters (2 vs 4),
    # c_gt2 wins because G_t >= 2 takes precedence.
    def tie_kappa(c: spacetime.Candidate) -> float:
      return 1.0 if c.G == 2 else 2.0

    choice = spacetime.choose_partition(
        [c_gt1, c_gt2], kappa_fn=tie_kappa, m_item=2.0, is_complete_grid=True
    )
    self.assertEqual(choice.winner.name, "cand_gt2")

    # Also test when G_t = 1 has more clusters (e.g. G=6 vs G=4)
    c_gt1_more = spacetime.Candidate(
        name="cand_gt1_more",
        cluster_ids=np.repeat(np.arange(6), 10),
        G=6,
        G_s=6,
        G_t=1,
        n_max=10,
        m_bar=10.0,
    )
    choice_more = spacetime.choose_partition(
        [c_gt1_more, c_gt2],
        kappa_fn=lambda c: 3.0 if c.G == 6 else 2.0,
        m_item=2.0,
        is_complete_grid=True,
    )
    self.assertEqual(choice_more.winner.name, "cand_gt2")

  def test_vectorized_edges_matches_slow_reference(self):
    """Vectorized spacetime_edges matches slow reference on random graph with 30% missing items."""
    rng = np.random.default_rng(2026)
    num_nodes = 12
    num_times = 40
    # Random connected spatial graph on 12 nodes
    spatial_edge_list = []
    for i in range(num_nodes - 1):
      spatial_edge_list.append((i, i + 1))
    for i in range(num_nodes):
      for j in range(i + 2, num_nodes):
        if rng.random() < 0.25:
          spatial_edge_list.append((i, j))
    spatial_edges = np.array(spatial_edge_list, dtype=np.int64)

    # 30% of items missing
    all_pairs = [(u, t) for u in range(num_nodes) for t in range(num_times)]
    keep_mask = rng.random(len(all_pairs)) >= 0.30
    observed_pairs = [p for p, k in zip(all_pairs, keep_mask) if k]
    node_ids = [p[0] for p in observed_pairs]
    times = [p[1] for p in observed_pairs]

    index = spacetime.spacetime_items(node_ids, times)
    self.assertFalse(index.is_complete_grid)

    phi_s = 0.4
    phi_t = [0.3, 0.15]
    phi_cr = [0.12, 0.05]

    edges_vec, weights_vec = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=2,
        phi_spatial=phi_s,
        phi_temporal=phi_t,
        phi_cross=phi_cr,
    )

    edges_ref, weights_ref = _slow_reference_spacetime_edges(
        index,
        spatial_edges,
        max_lag=2,
        phi_spatial=phi_s,
        phi_temporal=phi_t,
        phi_cross=phi_cr,
    )

    self.assertGreater(len(edges_vec), 0)
    self.assertEqual(len(edges_vec), len(edges_ref))
    np.testing.assert_array_equal(edges_vec, edges_ref)
    np.testing.assert_allclose(weights_vec, weights_ref)

  def test_marginal_checks_window_and_community_recovery(self):
    """Product partition with len(unique_t) % wl != 0 and missing items uses Candidate labels."""
    num_nodes = 4
    num_times = 7  # 7 % 3 = 1 != 0
    spatial_edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)

    all_pairs = [(u, t) for u in range(num_nodes) for t in range(num_times)]
    # Drop (1, 2) and (2, 5) to create missing items
    observed_pairs = [p for p in all_pairs if not (p == (1, 2) or p == (2, 5))]
    node_ids = [p[0] for p in observed_pairs]
    times = [p[1] for p in observed_pairs]

    index = spacetime.spacetime_items(node_ids, times)
    cands = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[3]
    )

    prod_cands = [c for c in cands if c.name.startswith("product_")]
    self.assertNotEmpty(prod_cands)
    prod = prod_cands[0]

    # Verify community_ids and window_ids are present and non-trivial
    self.assertEqual(len(prod.community_ids), index.n_items)
    self.assertEqual(len(prod.window_ids), index.n_items)
    self.assertEqual(prod.G_s, len(np.unique(prod.community_ids)))
    self.assertEqual(prod.G_t, len(np.unique(prod.window_ids)))

    # Evaluate spacetime_interval
    rng = np.random.default_rng(123)
    s_items = rng.uniform(0.5, 2.0, size=index.n_items)
    choice = spacetime.choose_partition(
        [prod],
        kappa_fn=spacetime.separable_kappa(1.2, 1.2),
        m_item=2.0,
        is_complete_grid=True,
    )
    res = spacetime.spacetime_interval(
        s_items, index, choice, metric="mse", level=0.95, m_item=2.0
    )
    self.assertIsNotNone(res.marginal_checks)

  def test_gt_exceeds_half_time_steps_runs(self):
    """Case with G_t > len(unique_t) // 2 runs cleanly without division by zero."""
    num_nodes = 3
    # 3 time steps with wl=1 => G_t = 3 > 3 // 2 = 1
    times_list = [0, 1, 2]
    node_ids = [u for u in range(num_nodes) for _ in times_list]
    times = [t for _ in range(num_nodes) for t in times_list]
    spatial_edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    index = spacetime.spacetime_items(node_ids, times)
    cands = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[1]
    )
    prod = [c for c in cands if c.name.startswith("product_")][0]
    self.assertGreater(prod.G_t, len(index.unique_times) // 2)

    choice = spacetime.choose_partition(
        [prod],
        kappa_fn=spacetime.separable_kappa(1.1, 1.1),
        m_item=2.0,
        is_complete_grid=index.is_complete_grid,
    )
    s_items = np.ones(index.n_items, dtype=np.float64)
    res = spacetime.spacetime_interval(
        s_items, index, choice, metric="mse", level=0.95, m_item=2.0
    )
    self.assertIsNotNone(res.marginal_checks)

  def test_spatial_marginal_trend_robustness(self):
    """Spatial marginal check does not refute under pure linear trend (>=88/100 seeds)."""
    num_nodes = 50
    num_times = 50
    # Path graph on 50 nodes (49 edges >= 30 required by check_uncorrelated_edges)
    spatial_edges = np.column_stack([
        np.arange(num_nodes - 1, dtype=np.int64),
        np.arange(1, num_nodes, dtype=np.int64),
    ])

    wl = 5
    ts = 1

    per_window_refuted = 0
    grand_mean_refuted = 0

    num_seeds = 100 if _LONG else 15
    for seed in range(num_seeds):
      rng = np.random.default_rng(seed)
      # Structured exogenous missingness across quadrants:
      # for (u < N//2, t < T//2) or (u >= N//2, t >= T//2), keep with prob 0.60;
      # on opposite quadrants, keep with prob 0.95.
      half_nodes = num_nodes // 2
      half_times = num_times // 2
      observed = []
      for u in range(num_nodes):
        for t in range(num_times):
          if (u < half_nodes and t < half_times) or (
              u >= half_nodes and t >= half_times
          ):
            prob = 0.60
          else:
            prob = 0.95
          if rng.random() < prob:
            observed.append((u, t))

      node_ids = [p[0] for p in observed]
      times = [p[1] for p in observed]
      n_items = len(node_ids)

      index = spacetime.spacetime_items(node_ids, times)
      edges, _ = spacetime.spacetime_edges(
          index,
          spatial_edges,
          max_lag=1,
          phi_spatial=0.1,
          phi_temporal=[0.1],
      )

      cands = spacetime.candidate_partitions(
          index, spatial_edges, target_sizes=[ts], window_lengths=[wl]
      )
      prod = [c for c in cands if c.name == f"product_ts{ts}_wl{wl}"][0]

      choice = spacetime.choose_partition(
          [prod],
          kappa_fn=spacetime.separable_kappa(1.0, 1.0),
          m_item=2.0,
          is_complete_grid=True,
          edges=edges,
      )

      # Independent noise plus strong linear trend across time
      eps = rng.standard_normal(n_items)
      trend = 20.0 * (np.array(times, dtype=np.float64) / float(num_times))
      s_items = trend + eps

      # 1. Per-window centring (spacetime_interval default)
      res = spacetime.spacetime_interval(
          s_items, index, choice, metric="mse", level=0.95, m_item=2.0
      )
      self.assertIsNotNone(res.marginal_checks)
      assert res.marginal_checks is not None
      self.assertIsNotNone(res.marginal_checks.spatial)
      assert res.marginal_checks.spatial is not None
      self.assertNotEqual(
          res.marginal_checks.spatial.outcome,
          _refutation.Refutation.UNTESTABLE,
          f"Spatial marginal was UNTESTABLE on seed {seed}",
      )
      if res.marginal_checks.spatial.outcome == _refutation.Refutation.REFUTED:
        per_window_refuted += 1

      # 2. Grand-mean centring comparison (for reporting)
      grand_mean = float(np.mean(s_items))
      s_comm = np.bincount(
          prod.community_ids, weights=s_items, minlength=prod.G_s
      ).astype(np.float64)
      n_comm = np.bincount(prod.community_ids, minlength=prod.G_s).astype(
          np.float64
      )
      e_comm_gm = s_comm - n_comm * grand_mean

      # Community edges mapped via winner.community_ids
      cluster_to_comm = np.zeros(prod.G, dtype=np.int64)
      cluster_to_comm[prod.cluster_ids] = prod.community_ids
      self.assertGreater(
          choice.cluster_edges.shape[0],
          0,
          f"Cluster edges empty on seed {seed}",
      )
      c_u = cluster_to_comm[choice.cluster_edges[:, 0]]
      c_v = cluster_to_comm[choice.cluster_edges[:, 1]]
      cross = c_u != c_v
      self.assertTrue(np.any(cross), f"No cross-community edges on seed {seed}")
      raw_e = np.column_stack([c_u[cross], c_v[cross]])
      comm_edges = np.unique(
          np.column_stack([
              np.minimum(raw_e[:, 0], raw_e[:, 1]),
              np.maximum(raw_e[:, 0], raw_e[:, 1]),
          ]),
          axis=0,
      ).astype(np.int64)

      gm_check = _refutation.check_uncorrelated_edges(
          e_comm_gm, n_comm.astype(np.int64), comm_edges, level=0.95
      )
      if gm_check.outcome == _refutation.Refutation.REFUTED:
        grand_mean_refuted += 1

    per_window_not_refuted = num_seeds - per_window_refuted
    if _LONG:
      print(
          f"\nTrend Test: Per-window not refuted: {per_window_not_refuted}/{num_seeds}"
          f" (refuted: {per_window_refuted}), Grand-mean refuted:"
          f" {grand_mean_refuted}/{num_seeds}"
      )
    # Nominal per-window not-refuted rate is >= 0.88. In short mode (num_seeds=15),
    # allow p - 3*sqrt(p*(1-p)/n) = 0.88 - 3*sqrt(0.88*0.12/15) ~ 0.628 (i.e. at least 10/15).
    min_not_refuted_rate = (
        0.88
        if _LONG
        else max(0.0, 0.88 - 3.0 * math.sqrt(0.88 * 0.12 / float(num_seeds)))
    )
    self.assertGreaterEqual(
        float(per_window_not_refuted) / float(num_seeds),
        min_not_refuted_rate,
        f"Per-window centring had {per_window_refuted} refutations > allowed rate.",
    )

  def test_benchmark_timing_207_nodes_2000_steps(self):
    """Measures runtime for space-time edges, max_lag=1, complete grid."""
    rng = np.random.default_rng(2026)
    num_nodes = 207 if _LONG else 30
    num_times = 2000 if _LONG else 50

    # 3-regular-ish graph: ring with random chords
    edges_set = set()
    for i in range(num_nodes):
      edges_set.add((min(i, (i + 1) % num_nodes), max(i, (i + 1) % num_nodes)))
    target_chord_edges = 310 if _LONG else 45
    while len(edges_set) < target_chord_edges:
      u = int(rng.integers(0, num_nodes))
      v = int(rng.integers(0, num_nodes))
      if u != v:
        edges_set.add((min(u, v), max(u, v)))
    spatial_edges = np.array(list(edges_set), dtype=np.int64)

    node_ids = np.repeat(np.arange(num_nodes), num_times)
    times = np.tile(np.arange(num_times), num_nodes)

    t0 = time.perf_counter()
    index = spacetime.spacetime_items(node_ids, times)
    t_items = time.perf_counter() - t0

    t1 = time.perf_counter()
    edges, _ = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=1,
        phi_spatial=0.5,
        phi_temporal=[0.3],
    )
    t_edges = time.perf_counter() - t1

    if _LONG:
      print(
          f"\nTiming Benchmark {num_nodes}x{num_times}, max_lag=1: "
          f"n_items={index.n_items}, total_edges={len(edges)}, "
          f"spacetime_items={t_items:.4f}s, spacetime_edges={t_edges:.4f}s, "
          f"total={t_items + t_edges:.4f}s"
      )
    self.assertEqual(index.n_items, num_nodes * num_times)
    self.assertGreater(len(edges), 0)

  def test_r2_shift_invariance(self):
    """y versus y + 1e6, with the same shift applied to predictions, gives the same R2 interval (rtol 1e-9)."""
    rng = np.random.default_rng(20260938)
    num_nodes = 20
    num_times = 100
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    node_ids = node_grid.ravel()
    times = time_grid.ravel()
    index = spacetime.spacetime_items(node_ids, times)
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )

    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[1]
    )
    edges, phi = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=1,
        phi_spatial=0.0,
        phi_temporal=[0.0],
    )
    kappa_fn = spacetime.separable_kappa(kappa_s=1.0, kappa_t=1.0)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=kappa_fn,
        m_item=4.0,
        is_complete_grid=index.is_complete_grid,
        edges=edges,
    )

    y_true = rng.normal(50.0, 10.0, size=index.n_items)
    y_pred = y_true + rng.normal(0.0, 2.0, size=index.n_items)

    shift = 1e6
    res_base = spacetime.spacetime_interval(
        (y_true, y_pred),
        index,
        choice,
        metric="r2",
        m_item=4.0,
        kappa_labels=1.0,
    )
    res_shifted = spacetime.spacetime_interval(
        (y_true + shift, y_pred + shift),
        index,
        choice,
        metric="r2",
        m_item=4.0,
        kappa_labels=1.0,
    )

    self.assertEqual(res_base.status, _interval.Status.UNREFUTED)
    self.assertEqual(res_shifted.status, _interval.Status.UNREFUTED)
    np.testing.assert_allclose(res_base.low, res_shifted.low, rtol=1e-9)
    np.testing.assert_allclose(res_base.high, res_shifted.high, rtol=1e-9)

  def test_item_order_invariance(self):
    """Permuting item order (with node_ids and times) leaves MSE and MAE intervals unchanged (rtol 1e-12)."""
    rng = np.random.default_rng(20260939)
    num_nodes = 20
    num_times = 100
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    node_ids = node_grid.ravel()
    times = time_grid.ravel()
    index = spacetime.spacetime_items(node_ids, times)
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )

    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[1]
    )
    edges, phi = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=1,
        phi_spatial=0.0,
        phi_temporal=[0.0],
    )
    kappa_fn = spacetime.separable_kappa(kappa_s=1.0, kappa_t=1.0)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=kappa_fn,
        m_item=16.0,
        is_complete_grid=index.is_complete_grid,
        edges=edges,
    )

    residuals = rng.normal(0.0, 1.0, size=index.n_items)
    perm = rng.permutation(index.n_items)

    perm_node_ids = node_ids[perm]
    perm_times = times[perm]
    perm_residuals = residuals[perm]
    perm_index = spacetime.spacetime_items(perm_node_ids, perm_times)

    perm_candidates = spacetime.candidate_partitions(
        perm_index, spatial_edges, target_sizes=[2], window_lengths=[1]
    )
    perm_edges, perm_phi = spacetime.spacetime_edges(
        perm_index,
        spatial_edges,
        max_lag=1,
        phi_spatial=0.0,
        phi_temporal=[0.0],
    )
    perm_choice = spacetime.choose_partition(
        perm_candidates,
        kappa_fn=kappa_fn,
        m_item=16.0,
        is_complete_grid=perm_index.is_complete_grid,
        edges=perm_edges,
    )

    for metric in ("mse", "mae"):
      m_val = 16.0 if metric == "mse" else 2.0
      choice_m = spacetime.choose_partition(
          candidates,
          kappa_fn=kappa_fn,
          m_item=m_val,
          is_complete_grid=index.is_complete_grid,
          edges=edges,
      )
      perm_choice_m = spacetime.choose_partition(
          perm_candidates,
          kappa_fn=kappa_fn,
          m_item=m_val,
          is_complete_grid=perm_index.is_complete_grid,
          edges=perm_edges,
      )
      res_orig = spacetime.spacetime_interval(
          residuals, index, choice_m, metric=metric, m_item=m_val
      )
      res_perm = spacetime.spacetime_interval(
          perm_residuals, perm_index, perm_choice_m, metric=metric, m_item=m_val
      )
      self.assertEqual(res_orig.status, _interval.Status.UNREFUTED)
      self.assertEqual(res_perm.status, _interval.Status.UNREFUTED)
      np.testing.assert_allclose(
          res_orig.low, res_perm.low, rtol=1e-12, atol=1e-12
      )
      np.testing.assert_allclose(
          res_orig.high, res_perm.high, rtol=1e-12, atol=1e-12
      )

  def test_topological_kappa_item_deletion_monotonicity(self):
    """Verifies Claim S2B(4) / Perron-Frobenius principal-submatrix monotonicity.

    Claim in README.md ('Why monotonicity holds only for topological kappa'):
    When items are omitted, the observed cut matrix is a principal submatrix,
    so lambda_max and hence topological kappa on the subset is <= kappa on the
    full set (up to 1e-12).
    """
    rng = np.random.default_rng(20260940)
    num_nodes = 20
    num_times = 100
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    node_ids = node_grid.ravel()
    times = time_grid.ravel()
    index = spacetime.spacetime_items(node_ids, times)
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )

    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[1]
    )
    edges, phi = spacetime.spacetime_edges(
        index, spatial_edges, max_lag=1, phi_spatial=0.2, phi_temporal=[0.2]
    )

    topo_fn = spacetime.topological_kappa_fn(index.n_items, edges, phi)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=topo_fn,
        m_item=4.0,
        is_complete_grid=index.is_complete_grid,
        edges=edges,
    )
    kappa_full = choice.winner_kappa

    # Randomly delete 30% of items (together with all incident edges)
    keep_mask = rng.random(index.n_items) >= 0.3
    sub_node_ids = index.node_ids[keep_mask]
    sub_times = index.times[keep_mask]
    sub_index = spacetime.spacetime_items(sub_node_ids, sub_times)

    sub_candidates = spacetime.candidate_partitions(
        sub_index, spatial_edges, target_sizes=[2], window_lengths=[1]
    )
    sub_edges, sub_phi = spacetime.spacetime_edges(
        sub_index,
        spatial_edges,
        max_lag=1,
        phi_spatial=0.2,
        phi_temporal=[0.2],
    )

    sub_topo_fn = spacetime.topological_kappa_fn(
        sub_index.n_items, sub_edges, sub_phi
    )
    sub_choice = spacetime.choose_partition(
        sub_candidates,
        kappa_fn=sub_topo_fn,
        m_item=4.0,
        is_complete_grid=sub_index.is_complete_grid,
        edges=sub_edges,
    )
    kappa_subset = sub_choice.winner_kappa

    # Perron-Frobenius monotonicity: principal submatrix spectral radius
    # cannot exceed the original matrix, so kappa on subset <= kappa on full set.
    self.assertLessEqual(kappa_subset, kappa_full + 1e-12)

    # Also verify per-candidate monotonicity across all candidate partitions
    for c_full, c_sub in zip(candidates, sub_candidates):
      k_f = topo_fn(c_full)
      k_s = sub_topo_fn(c_sub)
      self.assertLessEqual(k_s, k_f + 1e-12)

  def test_choice_records_m_item(self):
    """Verifies that Choice records m_item passed to choose_partition."""
    rng = np.random.default_rng(20261001)
    num_nodes, num_times = 4, 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack([np.arange(3), np.arange(1, 4)])
    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[2]
    )
    k_fn = spacetime.separable_kappa(kappa_s=1.0, kappa_t=1.0)

    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=k_fn,
        m_item=3.5,
        is_complete_grid=index.is_complete_grid,
    )
    self.assertEqual(choice.m_item, 3.5)

  def test_spacetime_interval_omitted_m_item_uses_choice(self):
    """Verifies that omitted m_item in spacetime_interval uses choice.m_item."""
    rng = np.random.default_rng(20261001)
    num_nodes, num_times = 20, 100
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )
    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[2]
    )
    k_fn = spacetime.separable_kappa(kappa_s=1.0, kappa_t=1.0)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=k_fn,
        m_item=2.5,
        is_complete_grid=index.is_complete_grid,
    )

    residuals = rng.normal(0.0, 1.0, size=index.n_items)
    res_explicit = spacetime.spacetime_interval(
        residuals, index, choice, metric="mse", m_item=2.5
    )
    res_omitted = spacetime.spacetime_interval(
        residuals, index, choice, metric="mse"
    )

    self.assertEqual(res_explicit.status, res_omitted.status)
    self.assertEqual(res_explicit.drift, res_omitted.drift)
    self.assertAlmostEqual(res_explicit.low, res_omitted.low, places=9)
    self.assertAlmostEqual(res_explicit.high, res_omitted.high, places=9)

  def test_spacetime_interval_declaration_kwargs_winner_kappa_greater_than_one(self):
    """Verifies that spacetime_interval(..., **decl.kwargs()) works with winner_kappa > 1."""
    from dgf.src.stats import declare
    rng = np.random.default_rng(20261001)
    num_nodes, num_times = 4, 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack([np.arange(3), np.arange(1, 4)])
    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[2]
    )
    k_fn = spacetime.separable_kappa(kappa_s=1.5, kappa_t=1.5)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=k_fn,
        m_item=3.0,
        is_complete_grid=index.is_complete_grid,
    )
    self.assertGreater(choice.winner_kappa, 1.0)

    decl = declare.Declaration(metric="mse", m=3.0)  # kappa is None by default
    residuals = rng.normal(0.0, 1.0, size=index.n_items)
    res = spacetime.spacetime_interval(
        residuals, index, choice, metric="mse", **decl.kwargs()
    )
    self.assertIsNotNone(res)

  def test_spacetime_interval_mismatched_m_item_raises(self):
    """Verifies that spacetime_interval raises ValueError if m_item differs from choice.m_item."""
    num_nodes, num_times = 4, 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack([np.arange(3), np.arange(1, 4)])
    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[2]
    )
    k_fn = spacetime.separable_kappa(kappa_s=1.0, kappa_t=1.0)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=k_fn,
        m_item=2.5,
        is_complete_grid=index.is_complete_grid,
    )

    residuals = np.zeros(index.n_items)
    with self.assertRaisesRegex(
        ValueError, "must use the m_item passed to choose_partition"
    ):
      spacetime.spacetime_interval(
          residuals, index, choice, metric="mse", m_item=4.0
      )

  def test_spacetime_interval_rejects_m_keyword_argument(self):
    """Verifies that spacetime_interval raises ValueError if 'm' is passed."""
    num_nodes, num_times = 4, 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack([np.arange(3), np.arange(1, 4)])
    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[2]
    )
    k_fn = spacetime.separable_kappa(kappa_s=1.0, kappa_t=1.0)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=k_fn,
        m_item=2.5,
        is_complete_grid=index.is_complete_grid,
    )
    residuals = np.zeros(index.n_items)
    with self.assertRaisesRegex(
        ValueError, "spacetime_interval does not accept 'm'"
    ):
      spacetime.spacetime_interval(
          residuals, index, choice, metric="mse", m=4.0
      )


  def test_spacetime_interval_smaller_kappa_raises_larger_accepted(self):
    """Verifies that kappa override smaller than winner_kappa raises, larger is accepted."""
    rng = np.random.default_rng(20261001)
    num_nodes, num_times = 4, 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack([np.arange(3), np.arange(1, 4)])
    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[2]
    )
    k_fn = spacetime.separable_kappa(kappa_s=1.5, kappa_t=1.5)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=k_fn,
        m_item=3.0,
        is_complete_grid=index.is_complete_grid,
    )

    residuals = rng.normal(0.0, 1.0, size=index.n_items)

    # Smaller kappa raises ValueError
    smaller_k = choice.winner_kappa - 0.2
    with self.assertRaisesRegex(
        ValueError, "cannot be smaller than choice.winner_kappa"
    ):
      spacetime.spacetime_interval(
          residuals, index, choice, metric="mse", kappa=smaller_k
      )

    # Larger kappa is accepted
    larger_k = choice.winner_kappa + 1.0
    res = spacetime.spacetime_interval(
        residuals, index, choice, metric="mse", kappa=larger_k
    )
    self.assertIsNotNone(res)

  def test_spacetime_interval_data_length_mismatch_raises(self):
    """Verifies that data length mismatch against index.n_items raises ValueError."""
    num_nodes, num_times = 4, 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack([np.arange(3), np.arange(1, 4)])
    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[2]
    )
    k_fn = spacetime.separable_kappa(kappa_s=1.0, kappa_t=1.0)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=k_fn,
        m_item=2.0,
        is_complete_grid=index.is_complete_grid,
    )

    # 1. 1-D data length mismatch
    with self.assertRaisesRegex(ValueError, r"Data length.*does not match index\.n_items"):
      spacetime.spacetime_interval(
          np.zeros(index.n_items - 1), index, choice, metric="mse"
      )

    # 2. R2 labels length mismatch
    with self.assertRaisesRegex(ValueError, r"Labels length.*does not match index\.n_items"):
      spacetime.spacetime_interval(
          (np.zeros(index.n_items - 1), np.zeros(index.n_items)),
          index,
          choice,
          metric="r2",
          kappa_labels=1.0,
      )

    # 3. R2 predictions length mismatch
    with self.assertRaisesRegex(ValueError, r"Predictions length.*does not match index\.n_items"):
      spacetime.spacetime_interval(
          (np.zeros(index.n_items), np.zeros(index.n_items + 1)),
          index,
          choice,
          metric="r2",
          kappa_labels=1.0,
      )

  def test_spacetime_interval_r2_requires_explicit_kappa_labels(self):
    """Verifies that metric='r2' requires explicit kappa_labels in spacetime_interval."""
    rng = np.random.default_rng(20261001)
    num_nodes, num_times = 20, 100
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )
    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[2]
    )
    k_fn = spacetime.separable_kappa(kappa_s=1.0, kappa_t=1.0)
    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=k_fn,
        m_item=4.0,
        is_complete_grid=index.is_complete_grid,
    )

    y_true = rng.normal(50.0, 10.0, size=index.n_items)
    y_pred = y_true + rng.normal(0.0, 2.0, size=index.n_items)

    # Without kappa_labels -> raises ValueError
    with self.assertRaisesRegex(ValueError, "kappa_labels must be explicitly passed"):
      spacetime.spacetime_interval(
          (y_true, y_pred), index, choice, metric="r2"
      )

    # With kappa_labels -> succeeds
    res = spacetime.spacetime_interval(
        (y_true, y_pred), index, choice, metric="r2", kappa_labels=1.5
    )
    self.assertEqual(res.status, _interval.Status.UNREFUTED)

  def test_choose_partition_range_route(self):
    """Verifies that kappa functions with is_range=True set route='range'."""
    num_nodes, num_times = 4, 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack([np.arange(3), np.arange(1, 4)])
    candidates = spacetime.candidate_partitions(
        index, spatial_edges, target_sizes=[2], window_lengths=[2]
    )

    def dummy_range_k_fn(c: spacetime.Candidate) -> float:
      return 2.5

    setattr(dummy_range_k_fn, "is_range", True)
    setattr(dummy_range_k_fn, "is_separable", False)

    choice = spacetime.choose_partition(
        candidates,
        kappa_fn=dummy_range_k_fn,
        m_item=2.0,
        is_complete_grid=index.is_complete_grid,
    )
    self.assertEqual(choice.route, "range")
    self.assertEqual(
        choice.route_reason,
        "Range envelope kappa (THEORY_RANGE_ENVELOPE.md, Claim RE1).",
    )


def _slow_reference_spacetime_edges(
    index: spacetime.SpaceTimeIndex,
    spatial_edges: np.ndarray,
    *,
    max_lag: int,
    phi_spatial: float,
    phi_temporal: Sequence[float] | np.ndarray,
    phi_cross: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Pure-Python reference implementation of spacetime_edges."""
  phi_temp_arr = np.asarray(phi_temporal, dtype=np.float64)
  if phi_cross is not None:
    phi_cross_arr = np.asarray(phi_cross, dtype=np.float64)
  else:
    phi_cross_arr = phi_spatial * phi_temp_arr

  s_edges = np.asarray(spatial_edges)
  node_to_idx = {u: i for i, u in enumerate(index.unique_nodes)}

  mapped_spatial_edges = []
  for u, v in s_edges:
    iu = node_to_idx.get(u)
    iv = node_to_idx.get(v)
    if (
        iu is None
        and isinstance(u, (int, np.integer))
        and 0 <= int(u) < index.num_nodes
    ):
      iu = int(u)
    if (
        iv is None
        and isinstance(v, (int, np.integer))
        and 0 <= int(v) < index.num_nodes
    ):
      iv = int(v)
    if iu is not None and iv is not None and iu != iv:
      mapped_spatial_edges.append(
          (index.unique_nodes[iu], index.unique_nodes[iv])
      )

  item_lookup = {
      (u, int(t)): idx
      for idx, (u, t) in enumerate(zip(index.node_ids, index.times))
  }
  edge_dict: dict[tuple[int, int], float] = {}

  def add_edge(idx1: int, idx2: int, weight: float) -> None:
    if idx1 == idx2:
      return
    pair = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
    prev = edge_dict.get(pair)
    if prev is None or weight > prev:
      edge_dict[pair] = weight

  # 1. Same-time spatial edges
  for u, v in mapped_spatial_edges:
    for t in index.unique_times:
      i1 = item_lookup.get((u, int(t)))
      i2 = item_lookup.get((v, int(t)))
      if i1 is not None and i2 is not None:
        add_edge(i1, i2, phi_spatial)

  # 2. Same-node temporal edges
  for idx, (u, t) in enumerate(zip(index.node_ids, index.times)):
    t_int = int(t)
    for lag in range(1, max_lag + 1):
      t_next = t_int + lag
      i_next = item_lookup.get((u, t_next))
      if i_next is not None:
        add_edge(idx, i_next, float(phi_temp_arr[lag - 1]))

  # 3. Cross-lagged edges
  for u, v in mapped_spatial_edges:
    for lag in range(1, max_lag + 1):
      w = float(phi_cross_arr[lag - 1])
      for t in index.unique_times:
        t_int = int(t)
        t_next = t_int + lag
        iu = item_lookup.get((u, t_int))
        iv_next = item_lookup.get((v, t_next))
        if iu is not None and iv_next is not None:
          add_edge(iu, iv_next, w)
        iv = item_lookup.get((v, t_int))
        iu_next = item_lookup.get((u, t_next))
        if iv is not None and iu_next is not None:
          add_edge(iv, iu_next, w)

  if not edge_dict:
    return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)

  pairs = sorted(edge_dict.keys())
  weights = [edge_dict[p] for p in pairs]
  return np.array(pairs, dtype=np.int64), np.array(weights, dtype=np.float64)


if __name__ == "__main__":
  absltest.main()
