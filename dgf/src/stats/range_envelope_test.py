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

"""Unit tests and benchmarks for range-envelope variance inflation (range_envelope.py)."""

import math
import os
import time
from absl.testing import absltest
from absl.testing import parameterized
from unittest import mock
import numpy as np

import scipy.sparse as sp

from dgf.src.stats import range_envelope
from dgf.src.stats import spacetime
from dgf.src.stats.independent import interval as _ind_interval

# Set DGF_STATS_LONG_TESTS=1 (blaze test --test_env=...) for the full-size runs.
_LONG = os.environ.get("DGF_STATS_LONG_TESTS") == "1"

RangeDeclaration = range_envelope.RangeDeclaration
RangeKappa = range_envelope.RangeKappa


def _build_dense_cut_adjacency(
    index: spacetime.SpaceTimeIndex,
    spatial_edges: np.ndarray,
    cluster_ids: np.ndarray,
    declaration: RangeDeclaration,
) -> np.ndarray:
  """Constructs the exact dense cut-adjacency matrix on observed items for testing."""
  n = index.n_items
  num_nodes = index.num_nodes
  r_s = declaration.spatial_range
  r_t = declaration.temporal_range

  # Spatial ball matrix via BFS
  s_ball = range_envelope._compute_spatial_ball_matrix(
      num_nodes, spatial_edges, index.unique_nodes, r_s
  )

  node_indices = np.searchsorted(index.unique_nodes, index.node_ids)
  time_indices = np.searchsorted(index.unique_times, index.times)

  a_cut = np.zeros((n, n), dtype=np.float64)
  for u in range(n):
    c_u = cluster_ids[u]
    i_u = node_indices[u]
    t_u = index.times[u]
    for v in range(u + 1, n):
      c_v = cluster_ids[v]
      if c_u == c_v:
        continue
      i_v = node_indices[v]
      t_v = index.times[v]
      if s_ball[i_u, i_v] == 1.0 and abs(t_u - t_v) <= r_t:
        a_cut[u, v] = 1.0
        a_cut[v, u] = 1.0

  return a_cut


class RangeEnvelopeTest(parameterized.TestCase):

  def test_declaration_properties_and_validation(self):
    """Verifies loss-level ranges, post-init validation, and error types."""
    decl = RangeDeclaration(
        k_hops=1,
        lookback=3,
        horizon=1,
        data_range_s=2,
        data_range_t=2,
        gamma=0.5,
    )
    self.assertEqual(decl.spatial_range, 2 * 1 + 2)  # 4
    self.assertEqual(decl.temporal_range, 3 + 1 + 2)  # 6
    self.assertEqual(decl.gamma, 0.5)

    # Missing gamma is a TypeError
    with self.assertRaises(TypeError):
      decl_cls = getattr(range_envelope, "RangeDeclaration")
      decl_cls(1, 3, 1, 2, 2)

    # Invalid fields raise ValueError
    with self.assertRaises(ValueError):
      RangeDeclaration(1, 3, 1, 2, 2, gamma=-0.1)
    with self.assertRaises(ValueError):
      RangeDeclaration(1, 3, 1, 2, 2, gamma=float("nan"))
    with self.assertRaises(ValueError):
      RangeDeclaration(-1, 3, 1, 2, 2, gamma=0.5)
    with self.assertRaises(ValueError):
      RangeDeclaration(1, -3, 1, 2, 2, gamma=0.5)
    with self.assertRaises(ValueError):
      RangeDeclaration(1, 3, -1, 2, 2, gamma=0.5)
    with self.assertRaises(ValueError):
      RangeDeclaration(1, 3, 1, -2, 2, gamma=0.5)
    with self.assertRaises(ValueError):
      RangeDeclaration(1, 3, 1, 2, -2, gamma=0.5)

  def test_dense_eigenvalue_comparison_small_random(self):
    """Tests both direct and kronecker bounds >= dense lambda_max + 1 + gamma."""
    rng = np.random.default_rng(42)
    num_nodes = 6
    num_times = 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    node_ids = node_grid.ravel()
    times = time_grid.ravel()
    index = spacetime.spacetime_items(node_ids, times)

    # Line graph on 6 nodes
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )

    decl = RangeDeclaration(
        k_hops=1,
        lookback=1,
        horizon=1,
        data_range_s=0,
        data_range_t=0,
        gamma=0.25,
    )
    # spatial_range = 2, temporal_range = 2

    cands = range_envelope.range_candidates(
        index, spatial_edges, decl, target_sizes=[2], window_multipliers=[1, 2]
    )
    self.assertNotEmpty(cands)

    for cand in cands:
      a_cut_dense = _build_dense_cut_adjacency(
          index, spatial_edges, cand.cluster_ids, decl
      )
      eigvals = np.linalg.eigvalsh(a_cut_dense)
      lam_max_dense = float(np.max(eigvals)) if len(eigvals) > 0 else 0.0

      res_dir = range_envelope.range_kappa(
          index, spatial_edges, cand.cluster_ids, decl, method="direct"
      )
      res_kron = range_envelope.range_kappa(
          index, spatial_edges, cand.cluster_ids, decl, method="kronecker"
      )

      # 1. Both certified bounds must be >= 1 + lam_max_dense + gamma
      self.assertGreaterEqual(res_dir.kappa, 1.0 + lam_max_dense + decl.gamma)
      self.assertGreaterEqual(res_kron.kappa, 1.0 + lam_max_dense + decl.gamma)

      # 2. Kronecker <= its own max cut row sum + 1 + gamma
      self.assertLessEqual(
          res_kron.kappa, 1.0 + res_kron.max_cut_row_sum + decl.gamma + 1e-9
      )

      # 3. Direct and Kronecker within 5% of each other
      rel_diff = abs(res_dir.kappa - res_kron.kappa) / max(res_dir.kappa, 1.0)
      self.assertLess(rel_diff, 0.05)

  def test_incomplete_grid_missing_items(self):
    """Tests direct and kronecker agreement on incomplete grid with missing items."""
    num_nodes = 5
    num_times = 8
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    node_ids = node_grid.ravel()
    times = time_grid.ravel()

    # Drop 8 items out of 40 to make an incomplete grid
    rng = np.random.default_rng(123)
    mask = rng.uniform(size=len(node_ids)) > 0.20
    sub_node_ids = node_ids[mask]
    sub_times = times[mask]

    index = spacetime.spacetime_items(sub_node_ids, sub_times)
    self.assertFalse(index.is_complete_grid)

    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )
    decl = RangeDeclaration(
        k_hops=1,
        lookback=1,
        horizon=0,
        data_range_s=1,
        data_range_t=1,
        gamma=0.1,
    )

    cands = range_envelope.range_candidates(
        index, spatial_edges, decl, target_sizes=[2], window_multipliers=[1]
    )
    self.assertNotEmpty(cands)

    for cand in cands:
      res_dir = range_envelope.range_kappa(
          index, spatial_edges, cand.cluster_ids, decl, method="direct"
      )
      res_kron = range_envelope.range_kappa(
          index, spatial_edges, cand.cluster_ids, decl, method="kronecker"
      )

      self.assertGreaterEqual(res_dir.kappa, 1.0 + decl.gamma)
      self.assertGreaterEqual(res_kron.kappa, 1.0 + decl.gamma)
      rel_diff = abs(res_dir.kappa - res_kron.kappa) / max(res_dir.kappa, 1.0)
      self.assertLess(rel_diff, 0.05)

  def test_singletons_partition(self):
    """Tests singletons partition where every item is its own cluster."""
    num_nodes = 4
    num_times = 5
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.array([[0, 1], [1, 2], [2, 3]])

    decl = RangeDeclaration(
        k_hops=1,
        lookback=1,
        horizon=1,
        data_range_s=0,
        data_range_t=0,
        gamma=0.2,
    )
    singletons = np.arange(index.n_items, dtype=np.int64)

    res_dir = range_envelope.range_kappa(
        index, spatial_edges, singletons, decl, method="direct"
    )
    res_kron = range_envelope.range_kappa(
        index, spatial_edges, singletons, decl, method="kronecker"
    )

    self.assertGreater(res_dir.cut_bound, 0.0)
    self.assertGreater(res_kron.cut_bound, 0.0)
    rel_diff = abs(res_dir.kappa - res_kron.kappa) / max(res_dir.kappa, 1.0)
    self.assertLess(rel_diff, 0.05)

  def test_monotonicity_in_declarations(self):
    """Proposition RE3: kappa_range is monotone in each declaration component."""
    num_nodes = 5
    num_times = 6
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )

    base = RangeDeclaration(
        k_hops=1,
        lookback=1,
        horizon=1,
        data_range_s=1,
        data_range_t=1,
        gamma=0.2,
    )
    cand = range_envelope.range_candidates(
        index, spatial_edges, base, target_sizes=[2], window_multipliers=[1]
    )[0]
    c_ids = cand.cluster_ids

    k_base = range_envelope.range_kappa(
        index, spatial_edges, c_ids, base, method="kronecker"
    ).kappa

    # Increasing k_hops
    k_inc = range_envelope.range_kappa(
        index,
        spatial_edges,
        c_ids,
        RangeDeclaration(2, 1, 1, 1, 1, 0.2),
        method="kronecker",
    ).kappa
    self.assertGreaterEqual(k_inc, k_base - 1e-9)

    # Increasing lookback
    k_inc = range_envelope.range_kappa(
        index,
        spatial_edges,
        c_ids,
        RangeDeclaration(1, 2, 1, 1, 1, 0.2),
        method="kronecker",
    ).kappa
    self.assertGreaterEqual(k_inc, k_base - 1e-9)

    # Increasing horizon
    k_inc = range_envelope.range_kappa(
        index,
        spatial_edges,
        c_ids,
        RangeDeclaration(1, 1, 2, 1, 1, 0.2),
        method="kronecker",
    ).kappa
    self.assertGreaterEqual(k_inc, k_base - 1e-9)

    # Increasing data_range_s
    k_inc = range_envelope.range_kappa(
        index,
        spatial_edges,
        c_ids,
        RangeDeclaration(1, 1, 1, 2, 1, 0.2),
        method="kronecker",
    ).kappa
    self.assertGreaterEqual(k_inc, k_base - 1e-9)

    # Increasing data_range_t
    k_inc = range_envelope.range_kappa(
        index,
        spatial_edges,
        c_ids,
        RangeDeclaration(1, 1, 1, 1, 2, 0.2),
        method="kronecker",
    ).kappa
    self.assertGreaterEqual(k_inc, k_base - 1e-9)

    # Increasing gamma
    k_inc = range_envelope.range_kappa(
        index,
        spatial_edges,
        c_ids,
        RangeDeclaration(1, 1, 1, 1, 1, 0.5),
        method="kronecker",
    ).kappa
    self.assertGreaterEqual(k_inc, k_base + 0.3 - 1e-9)

  def test_non_product_partition_raises_for_kronecker(self):
    """Verifies that non-product partition raises ValueError under method='kronecker'."""
    num_nodes = 4
    num_times = 4
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.array([[0, 1], [1, 2], [2, 3]])
    decl = RangeDeclaration(1, 1, 1, 0, 0, 0.1)

    # Construct an explicitly non-product partition:
    # cluster 0 has (node 0, time 0) and (node 1, time 1), but (node 0, time 1) is cluster 1
    # and (node 1, time 0) is cluster 2.
    c_ids = np.arange(index.n_items, dtype=np.int64)
    # item 0 is (node 0, time 0), item 5 is (node 1, time 1)
    c_ids[5] = c_ids[0]  # merge into cluster 0
    # items 1 is (node 0, time 1), item 4 is (node 1, time 0) remain distinct

    with self.assertRaises(ValueError) as ctx:
      range_envelope.range_kappa(
          index, spatial_edges, c_ids, decl, method="kronecker"
      )
    self.assertIn("product partition", str(ctx.exception).lower())

    # Direct method should not raise for non-product partitions
    res_dir = range_envelope.range_kappa(
        index, spatial_edges, c_ids, decl, method="direct"
    )
    self.assertGreaterEqual(res_dir.kappa, 1.0)

  def test_max_edges_raises_for_direct(self):
    """Verifies that method='direct' raises ValueError when edges exceed max_edges."""
    num_nodes = 10
    num_times = 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )
    decl = RangeDeclaration(1, 2, 1, 1, 1, 0.1)
    c_ids = np.zeros(index.n_items, dtype=np.int64)

    with self.assertRaises(ValueError) as ctx:
      range_envelope.range_kappa(
          index, spatial_edges, c_ids, decl, method="direct", max_edges=10
      )
    self.assertIn("exceeds max_edges", str(ctx.exception))
    self.assertIn("kronecker", str(ctx.exception).lower())

  def test_end_to_end_choose_partition_and_spacetime_interval(self):
    """Verifies end-to-end integration with spacetime.choose_partition and spacetime_interval."""
    # 8 nodes x 1000 times = 8000 items (G = 8000 singletons, nu ~ 8000/9.1 = 880 > 585 = n_A).
    num_nodes = 8
    num_times = 1000
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )

    decl = RangeDeclaration(
        k_hops=1,
        lookback=1,
        horizon=0,
        data_range_s=0,
        data_range_t=0,
        gamma=0.1,
    )

    cands = range_envelope.range_candidates(
        index, spatial_edges, decl, target_sizes=[2], window_multipliers=[1, 2]
    )
    self.assertNotEmpty(cands)

    k_fn = range_envelope.range_kappa_fn(
        index, spatial_edges, decl, method="kronecker"
    )

    # m_item=16.0 is the library default for MSE (squared Gaussian residuals have relative second moment 3).
    choice = spacetime.choose_partition(
        cands,
        kappa_fn=k_fn,
        m_item=16.0,
        is_complete_grid=index.is_complete_grid,
    )
    self.assertEqual(choice.route, "range")
    self.assertGreaterEqual(choice.winner_kappa, 1.0 + decl.gamma)

    # Evaluate residuals
    rng = np.random.default_rng(2026)
    residuals = rng.normal(loc=0.0, scale=1.0, size=index.n_items)
    res = spacetime.spacetime_interval(
        residuals,
        index,
        choice,
        metric="mse",
        m_item=16.0,
    )
    self.assertEqual(res.status, _ind_interval.Status.UNREFUTED)
    self.assertGreater(res.high, res.low)
    self.assertGreater(res.low, 0.0)

  def test_benchmark_timings_ring_and_lattice(self):
    """Benchmarks and logs timings for ring 200x500 and lattice 20x20x500."""
    decl = RangeDeclaration(
        k_hops=1,
        lookback=3,
        horizon=1,
        data_range_s=2,
        data_range_t=2,
        gamma=0.5,
    )
    self.assertEqual(decl.spatial_range, 4)
    self.assertEqual(decl.temporal_range, 6)

    # 1. Ring: 200 nodes x 500 steps (long) vs 20 nodes x 50 steps (short)
    num_nodes_ring = 200 if _LONG else 20
    num_times = 500 if _LONG else 50
    ring_edges = np.column_stack(
        [np.arange(num_nodes_ring), (np.arange(num_nodes_ring) + 1) % num_nodes_ring]
    )
    node_grid_r, time_grid_r = np.meshgrid(
        np.arange(num_nodes_ring), np.arange(num_times), indexing="ij"
    )
    index_ring = spacetime.spacetime_items(
        node_grid_r.ravel(), time_grid_r.ravel()
    )

    singletons_ring = np.arange(index_ring.n_items, dtype=np.int64)
    # Window partition: windows of length 25 (or 10 in short mode)
    win_len = 25 if _LONG else 10
    time_indices_r = np.searchsorted(index_ring.unique_times, index_ring.times)
    window_part_ring = (time_indices_r // win_len).astype(np.int64)

    # Time Kronecker on Ring Singletons
    t0 = time.perf_counter()
    k_rk_sing = range_envelope.range_kappa(
        index_ring, ring_edges, singletons_ring, decl, method="kronecker"
    )
    t_kron_ring_sing = time.perf_counter() - t0

    # Time Kronecker on Ring Windows
    t0 = time.perf_counter()
    k_rk_win = range_envelope.range_kappa(
        index_ring, ring_edges, window_part_ring, decl, method="kronecker"
    )
    t_kron_ring_win = time.perf_counter() - t0

    # Time Direct on Ring Singletons (with max_edges headroom)
    t0 = time.perf_counter()
    k_rd_sing = range_envelope.range_kappa(
        index_ring,
        ring_edges,
        singletons_ring,
        decl,
        method="direct",
        max_edges=25_000_000,
    )
    t_dir_ring_sing = time.perf_counter() - t0

    # Time Direct on Ring Windows
    t0 = time.perf_counter()
    k_rd_win = range_envelope.range_kappa(
        index_ring,
        ring_edges,
        window_part_ring,
        decl,
        method="direct",
        max_edges=25_000_000,
    )
    t_dir_ring_win = time.perf_counter() - t0

    if _LONG:
      print(f"\nTiming report ring ({num_nodes_ring} nodes x {num_times} steps = {index_ring.n_items} items):")
      print(
          f"  Kronecker Singletons: kappa={k_rk_sing.kappa:.4f},"
          f" time={t_kron_ring_sing:.3f}s"
      )
      print(
          f"  Direct Singletons:    kappa={k_rd_sing.kappa:.4f},"
          f" time={t_dir_ring_sing:.3f}s"
      )
      print(
          f"  Kronecker Windows:    kappa={k_rk_win.kappa:.4f},"
          f" time={t_kron_ring_win:.3f}s"
      )
      print(
          f"  Direct Windows:       kappa={k_rd_win.kappa:.4f},"
          f" time={t_dir_ring_win:.3f}s"
      )

    # 2. Lattice
    grid_side = 20 if _LONG else 6
    num_nodes_lat = grid_side * grid_side
    lat_edges_list = []
    for r in range(grid_side):
      for c in range(grid_side):
        curr = r * grid_side + c
        if c + 1 < grid_side:
          lat_edges_list.append((curr, r * grid_side + (c + 1)))
        if r + 1 < grid_side:
          lat_edges_list.append((curr, (r + 1) * grid_side + c))
    lat_edges = np.array(lat_edges_list, dtype=np.int64)

    node_grid_l, time_grid_l = np.meshgrid(
        np.arange(num_nodes_lat), np.arange(num_times), indexing="ij"
    )
    index_lat = spacetime.spacetime_items(
        node_grid_l.ravel(), time_grid_l.ravel()
    )

    singletons_lat = np.arange(index_lat.n_items, dtype=np.int64)
    time_indices_l = np.searchsorted(index_lat.unique_times, index_lat.times)
    window_part_lat = (time_indices_l // win_len).astype(np.int64)

    # Time Kronecker on Lattice Singletons
    t0 = time.perf_counter()
    k_lk_sing = range_envelope.range_kappa(
        index_lat, lat_edges, singletons_lat, decl, method="kronecker"
    )
    t_kron_lat_sing = time.perf_counter() - t0

    # Time Kronecker on Lattice Windows
    t0 = time.perf_counter()
    k_lk_win = range_envelope.range_kappa(
        index_lat, lat_edges, window_part_lat, decl, method="kronecker"
    )
    t_kron_lat_win = time.perf_counter() - t0

    # Direct on lattice exceeds max_edges threshold -> raises ValueError
    t0 = time.perf_counter()
    raised_max_edges = False
    t_dir_lat_sing_raise = 0.0
    lat_max_edges = 20_000_000 if _LONG else 100
    try:
      range_envelope.range_kappa(
          index_lat,
          lat_edges,
          singletons_lat,
          decl,
          method="direct",
          max_edges=lat_max_edges,
      )
    except ValueError as e:
      raised_max_edges = True
      t_dir_lat_sing_raise = time.perf_counter() - t0

    self.assertTrue(raised_max_edges)

    if _LONG:
      print(f"\nTiming report lattice ({grid_side}x{grid_side}={num_nodes_lat} nodes x {num_times} steps = {index_lat.n_items} items):")
      print(
          f"  Kronecker Singletons: kappa={k_lk_sing.kappa:.4f},"
          f" time={t_kron_lat_sing:.3f}s"
      )
      print(
          f"  Kronecker Windows:    kappa={k_lk_win.kappa:.4f},"
          f" time={t_kron_lat_win:.3f}s"
      )
      print(
          "  Direct Singletons:    raised ValueError as expected"
          f" in {t_dir_lat_sing_raise:.3f}s"
      )

  def test_candidates_include_singleton_partition(self):
    """Verifies that the singleton partition (ts=1, wl=1) is always among candidates."""
    num_nodes = 10
    num_times = 20
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )
    decl = RangeDeclaration(
        k_hops=1,
        lookback=3,
        horizon=1,
        data_range_s=2,
        data_range_t=2,
        gamma=0.5,
    )
    # 1. Default target sizes and window multipliers
    cands_default = range_envelope.range_candidates(index, spatial_edges, decl)
    names_default = [c.name for c in cands_default]
    self.assertIn("product_ts1_wl1", names_default)
    cand_single = next(c for c in cands_default if c.name == "product_ts1_wl1")
    self.assertEqual(cand_single.G, index.n_items)
    self.assertEqual(cand_single.n_max, 1)
    self.assertEqual(cand_single.m_bar, 1.0)

    # 2. Explicit target sizes not containing 1
    cands_explicit = range_envelope.range_candidates(
        index, spatial_edges, decl, target_sizes=[4, 8]
    )
    names_explicit = [c.name for c in cands_explicit]
    self.assertIn("product_ts1_wl1", names_explicit)
    self.assertIn("product_ts4_wl1", names_explicit)
    self.assertIn("product_ts8_wl1", names_explicit)

  def test_spatial_ball_unobserved_intermediate_nodes_and_string_ids(self):
    """Verifies that unobserved nodes mediate paths and string node IDs work."""
    # 1. Line graph 0-1-2-...-9 where node 5 has no observed items.
    full_edges = np.column_stack([np.arange(9), np.arange(1, 10)])
    observed_nodes = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9])
    # Single observation per node at time 0
    index = spacetime.spacetime_items(observed_nodes, np.zeros(len(observed_nodes), dtype=np.int64))

    # Declaration with spatial range = 2*K + R_s = 2(1) + 0 = 2 hops.
    decl = RangeDeclaration(
        k_hops=1, lookback=0, horizon=0, data_range_s=0, data_range_t=0, gamma=0.0
    )
    s_ball = range_envelope._compute_spatial_ball_matrix(
        index.num_nodes, full_edges, index.unique_nodes, decl.spatial_range
    )
    # Node 4 is at index 4, node 6 is at index 5 in unique_nodes
    idx_4 = int(np.where(index.unique_nodes == 4)[0][0])
    idx_6 = int(np.where(index.unique_nodes == 6)[0][0])
    # Distance in full graph is d(4, 5) + d(5, 6) = 2 <= spatial_range
    self.assertEqual(s_ball[idx_4, idx_6], 1.0)
    self.assertEqual(s_ball[idx_6, idx_4], 1.0)

    # Cut bound with node 4 and node 6 in different clusters
    c_ids = np.arange(len(observed_nodes))
    res_direct = range_envelope.range_kappa(
        index, full_edges, c_ids, decl, method="direct"
    )
    res_kron = range_envelope.range_kappa(
        index, full_edges, c_ids, decl, method="kronecker"
    )
    # Node 4 and node 6 are within range and in distinct clusters -> cut edge exists
    self.assertGreater(res_direct.cut_bound, 0.0)
    self.assertGreater(res_kron.cut_bound, 0.0)
    self.assertAlmostEqual(res_direct.kappa, res_kron.kappa, places=2)

    # 2. String node IDs with unobserved intermediate node: 'A' - 'B' - 'C'
    str_edges = np.array([["node_A", "node_B"], ["node_B", "node_C"]])
    str_observed = np.array(["node_A", "node_C"])
    str_index = spacetime.spacetime_items(str_observed, np.zeros(2, dtype=np.int64))
    str_s_ball = range_envelope._compute_spatial_ball_matrix(
        str_index.num_nodes, str_edges, str_index.unique_nodes, decl.spatial_range
    )
    self.assertEqual(str_s_ball[0, 1], 1.0)
    self.assertEqual(str_s_ball[1, 0], 1.0)

    str_c_ids = np.array([0, 1])
    str_res_direct = range_envelope.range_kappa(
        str_index, str_edges, str_c_ids, decl, method="direct"
    )
    str_res_kron = range_envelope.range_kappa(
        str_index, str_edges, str_c_ids, decl, method="kronecker"
    )
    self.assertGreater(str_res_direct.cut_bound, 0.0)
    self.assertAlmostEqual(str_res_direct.kappa, str_res_kron.kappa, places=4)

  def test_shuffled_item_order_invariance(self):
    """Verifies that shuffled item order produces identical kappa for both methods."""
    num_nodes = 6
    num_times = 10
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    node_ids = node_grid.ravel()
    times = time_grid.ravel()
    index_sorted = spacetime.spacetime_items(node_ids, times)
    spatial_edges = np.column_stack(
        [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
    )

    decl = RangeDeclaration(
        k_hops=1, lookback=2, horizon=0, data_range_s=1, data_range_t=1, gamma=0.25
    )
    cands = range_envelope.range_candidates(
        index_sorted, spatial_edges, decl, target_sizes=[2], window_multipliers=[1]
    )
    cluster_ids_sorted = cands[0].cluster_ids

    res_dir_sorted = range_envelope.range_kappa(
        index_sorted, spatial_edges, cluster_ids_sorted, decl, method="direct"
    )
    res_kron_sorted = range_envelope.range_kappa(
        index_sorted, spatial_edges, cluster_ids_sorted, decl, method="kronecker"
    )

    # Shuffle the items
    rng = np.random.default_rng(42)
    n = index_sorted.n_items
    perm = rng.permutation(n)

    index_shuffled = spacetime.spacetime_items(node_ids[perm], times[perm])
    cluster_ids_shuffled = cluster_ids_sorted[perm]

    res_dir_shuffled = range_envelope.range_kappa(
        index_shuffled, spatial_edges, cluster_ids_shuffled, decl, method="direct"
    )
    res_kron_shuffled = range_envelope.range_kappa(
        index_shuffled, spatial_edges, cluster_ids_shuffled, decl, method="kronecker"
    )

    # Direct method must match to 9 decimal places
    self.assertAlmostEqual(
        res_dir_sorted.cut_bound, res_dir_shuffled.cut_bound, places=9
    )
    self.assertAlmostEqual(res_dir_sorted.kappa, res_dir_shuffled.kappa, places=9)

    # Kronecker method must match
    self.assertAlmostEqual(
        res_kron_sorted.cut_bound, res_kron_shuffled.cut_bound, places=9
    )
    self.assertAlmostEqual(res_kron_sorted.kappa, res_kron_shuffled.kappa, places=9)

  def test_time_step_units_gcd_validation(self):
    """Verifies that time diff gcd > 1 raises ValueError in range_kappa and range_kappa_fn."""
    spatial_edges = np.array([[0, 1]])
    decl = RangeDeclaration(
        k_hops=1, lookback=1, horizon=0, data_range_s=0, data_range_t=0, gamma=0.1
    )

    # 1. Times in steps of 2 -> gcd = 2 > 1 -> raises ValueError
    times_step2 = np.array([0, 2, 4, 6])
    nodes_step2 = np.array([0, 0, 1, 1])
    idx_step2 = spacetime.spacetime_items(nodes_step2, times_step2)
    c_ids_step2 = np.array([0, 1, 2, 3])

    with self.assertRaisesRegex(ValueError, "divide times by g"):
      range_envelope.range_kappa(idx_step2, spatial_edges, c_ids_step2, decl)

    with self.assertRaisesRegex(ValueError, "divide times by g"):
      range_envelope.range_kappa_fn(idx_step2, spatial_edges, decl)

    # 2. Unix seconds with step 60 -> gcd = 60 > 1 -> raises ValueError
    times_unix = np.array([1700000000, 1700000060, 1700000120])
    nodes_unix = np.array([0, 0, 0])
    idx_unix = spacetime.spacetime_items(nodes_unix, times_unix)
    c_ids_unix = np.array([0, 1, 2])

    with self.assertRaisesRegex(ValueError, "divide times by g"):
      range_envelope.range_kappa(idx_unix, spatial_edges, c_ids_unix, decl)

    # 3. Missing time steps (0, 1, 3, 4) -> diffs [1, 2, 1] -> gcd = 1 -> valid
    times_missing = np.array([0, 1, 3, 4])
    nodes_missing = np.array([0, 0, 0, 0])
    idx_missing = spacetime.spacetime_items(nodes_missing, times_missing)
    c_ids_missing = np.array([0, 0, 1, 1])

    res = range_envelope.range_kappa(
        idx_missing, spatial_edges, c_ids_missing, decl
    )
    self.assertGreaterEqual(res.kappa, 1.0)

    # 4. Single timestamp -> valid
    times_single = np.array([10, 10])
    nodes_single = np.array([0, 1])
    idx_single = spacetime.spacetime_items(nodes_single, times_single)
    c_ids_single = np.array([0, 1])
    res_single = range_envelope.range_kappa(
        idx_single, spatial_edges, c_ids_single, decl
    )
    self.assertGreaterEqual(res_single.kappa, 1.0)

  def test_malformed_edges_raise(self):
    """Verifies that malformed edge arrays raise ValueError."""
    decl = RangeDeclaration(
        k_hops=1, lookback=1, horizon=0, data_range_s=0, data_range_t=0, gamma=0.1
    )
    index = spacetime.spacetime_items(np.array([0, 1]), np.array([0, 0]))
    cluster_ids = np.array([0, 1])

    # 1. (E, 3) edges
    with self.assertRaisesRegex(ValueError, r"must have shape \(E, 2\)"):
      range_envelope.range_kappa(
          index, np.zeros((3, 3), dtype=np.int64), cluster_ids, decl
      )

    # 2. 1-D [0, 1] edges
    with self.assertRaisesRegex(ValueError, r"must have shape \(E, 2\)"):
      range_envelope.range_kappa(
          index, np.array([0, 1]), cluster_ids, decl
      )

    # 3. 3-D edges
    with self.assertRaisesRegex(ValueError, r"must have shape \(E, 2\)"):
      range_envelope.range_kappa(
          index, np.zeros((2, 2, 2), dtype=np.int64), cluster_ids, decl
      )

  def test_int_vs_str_node_ids_raise(self):
    """Verifies that node ID type mismatches raise ValueError."""
    decl = RangeDeclaration(
        k_hops=1, lookback=1, horizon=0, data_range_s=0, data_range_t=0, gamma=0.1
    )
    index_int = spacetime.spacetime_items(np.array([0, 1]), np.array([0, 0]))
    cluster_ids = np.array([0, 1])

    # 1. String edges with int index
    str_edges = np.array([["0", "1"]])
    with self.assertRaisesRegex(ValueError, r"dtype kind.*does not match"):
      range_envelope.range_kappa(index_int, str_edges, cluster_ids, decl)

    # 2. Non-matching endpoint IDs (both int, but disjoint sets)
    disjoint_edges = np.array([[100, 101]])
    with self.assertRaisesRegex(
        ValueError, r"spatial_edges node ids do not match node_ids"
    ):
      range_envelope.range_kappa(index_int, disjoint_edges, cluster_ids, decl)

  def test_float_cluster_ids_raise(self):
    """Verifies that non-integer float or non-numeric cluster IDs raise ValueError."""
    decl = RangeDeclaration(
        k_hops=1, lookback=1, horizon=0, data_range_s=0, data_range_t=0, gamma=0.1
    )
    spatial_edges = np.array([[0, 1]])
    index = spacetime.spacetime_items(np.array([0, 1]), np.array([0, 0]))

    # 1. Non-integer float cluster IDs
    with self.assertRaisesRegex(ValueError, "cluster_ids contains non-integer float"):
      range_envelope.range_kappa(
          index, spatial_edges, np.array([0.5, 1.2]), decl
      )

    # 2. Non-finite float cluster IDs
    with self.assertRaisesRegex(ValueError, "cluster_ids contains non-integer float"):
      range_envelope.range_kappa(
          index, spatial_edges, np.array([0.0, np.nan]), decl
      )

    # 3. String cluster IDs
    with self.assertRaisesRegex(ValueError, "cluster_ids must be numeric"):
      range_envelope.range_kappa(
          index, spatial_edges, np.array(["c1", "c2"]), decl
      )

    # 4. Integral float cluster IDs (e.g. 0.0, 1.0) succeed
    res = range_envelope.range_kappa(
        index, spatial_edges, np.array([0.0, 1.0]), decl
    )
    self.assertGreaterEqual(res.kappa, 1.0)

  def test_fail_closed_nan_path_cannot_return_nan(self):
    """Verifies zero-cut kappa is 1+gamma and non-finite row bound raises ValueError."""
    num_nodes = 4
    num_times = 5
    node_grid, time_grid = np.meshgrid(
        np.arange(num_nodes), np.arange(num_times), indexing="ij"
    )
    index = spacetime.spacetime_items(node_grid.ravel(), time_grid.ravel())
    spatial_edges = np.column_stack([np.arange(3), np.arange(1, 4)])
    decl = RangeDeclaration(
        k_hops=1, lookback=1, horizon=0, data_range_s=0, data_range_t=0, gamma=0.2
    )
    cluster_ids = np.zeros(index.n_items, dtype=np.int64)

    # For zero cut (all items in 1 cluster), kappa must be exactly 1 + gamma
    res = range_envelope.range_kappa(index, spatial_edges, cluster_ids, decl)
    self.assertEqual(res.cut_bound, 0.0)
    self.assertEqual(res.kappa, 1.2)

    # Test that invalid ub_rows raises ValueError
    with mock.patch("numpy.max", return_value=float("nan")):
      with self.assertRaises(ValueError):
        range_envelope.range_kappa(index, spatial_edges, cluster_ids, decl)

  def test_independent_oracle_scipy_csgraph_shortest_path(self):
    """Oracle test comparing range_kappa against dense cut adjacency from scipy.csgraph."""
    # 8-node full graph with unobserved node 3
    num_total_nodes = 8
    # Adjacency matrix for scipy csgraph
    adj_matrix = np.zeros((num_total_nodes, num_total_nodes), dtype=np.int64)
    raw_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (2, 5)
    ]
    for u, v in raw_edges:
      adj_matrix[u, v] = 1
      adj_matrix[v, u] = 1
    spatial_edges = np.array(raw_edges, dtype=np.int64)

    # Shortest path distances across all 8 nodes via scipy
    d_full = sp.csgraph.shortest_path(adj_matrix, directed=False)

    # Observed nodes: node 3 is unobserved!
    observed_nodes = np.array([0, 1, 2, 4, 5, 6, 7], dtype=np.int64)
    # Incomplete grid: varying time steps per node
    node_list: list[int] = []
    time_list: list[int] = []
    for u in observed_nodes:
      # Different subsets of times in 0..5
      t_subset = [t for t in range(6) if (u + t) % 3 != 0]
      for t in t_subset:
        node_list.append(int(u))
        time_list.append(int(t))

    # Shuffle the items
    rng = np.random.default_rng(2026)
    n = len(node_list)
    perm = rng.permutation(n)
    nodes_shuffled = np.array(node_list, dtype=np.int64)[perm]
    times_shuffled = np.array(time_list, dtype=np.int64)[perm]

    index = spacetime.spacetime_items(nodes_shuffled, times_shuffled)

    decl = RangeDeclaration(
        k_hops=1, lookback=1, horizon=1, data_range_s=1, data_range_t=1, gamma=0.3
    )
    r_s = decl.spatial_range  # 2*1 + 1 = 3 hops
    r_t = decl.temporal_range  # 1 + 1 + 1 = 3 steps

    # Product partition for Kronecker compatibility
    cands = range_envelope.range_candidates(
        index, spatial_edges, decl, target_sizes=[2], window_multipliers=[1]
    )
    cluster_ids = cands[0].cluster_ids

    # Independent oracle: construct dense A_cut using d_full from scipy
    a_cut_oracle = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
      node_u = index.node_ids[u]
      time_u = index.times[u]
      c_u = cluster_ids[u]
      for v in range(u + 1, n):
        node_v = index.node_ids[v]
        time_v = index.times[v]
        c_v = cluster_ids[v]
        if c_u == c_v:
          continue
        if d_full[node_u, node_v] <= r_s and abs(time_u - time_v) <= r_t:
          a_cut_oracle[u, v] = 1.0
          a_cut_oracle[v, u] = 1.0

    eigvals = np.linalg.eigvalsh(a_cut_oracle)
    lambda_max_oracle = float(np.max(eigvals))
    kappa_oracle = 1.0 + lambda_max_oracle + decl.gamma

    res_direct = range_envelope.range_kappa(
        index, spatial_edges, cluster_ids, decl, method="direct"
    )
    res_kron = range_envelope.range_kappa(
        index, spatial_edges, cluster_ids, decl, method="kronecker"
    )

    # Both methods must be valid upper bounds on lambda_max + 1 + gamma
    self.assertGreaterEqual(res_direct.cut_bound, lambda_max_oracle - 1e-9)
    self.assertGreaterEqual(res_kron.cut_bound, lambda_max_oracle - 1e-9)
    self.assertGreaterEqual(res_direct.kappa, kappa_oracle - 1e-9)
    self.assertGreaterEqual(res_kron.kappa, kappa_oracle - 1e-9)
    # And they should agree closely with each other
    self.assertLess(
        abs(res_direct.kappa - res_kron.kappa) / res_direct.kappa, 0.05
    )


if __name__ == "__main__":
  absltest.main()

