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

"""Tests for distributed-capable size-constrained graph partitioner."""

import math
import os
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dgf.src.stats import partitioners

# Set DGF_STATS_LONG_TESTS=1 (blaze test --test_env=...) for the full-size runs.
_LONG = os.environ.get("DGF_STATS_LONG_TESTS") == "1"


def _make_ring_graph(n: int) -> np.ndarray:
  """Builds edge list for an n-node cycle graph."""
  edges = []
  for i in range(n):
    edges.append([i, (i + 1) % n])
  return np.array(edges, dtype=np.int64)


def _build_grid_2d(rows: int, cols: int) -> np.ndarray:
  """Builds rows x cols 2D torus grid (degree 4)."""
  edges = []
  for r in range(rows):
    for c in range(cols):
      u = r * cols + c
      v_right = r * cols + ((c + 1) % cols)
      edges.append([u, v_right])
      v_down = ((r + 1) % rows) * cols + c
      edges.append([u, v_down])

  all_edges = np.array(edges, dtype=np.int64)
  pairs = np.sort(all_edges, axis=1)
  return np.unique(pairs, axis=0)


def _build_watts_strogatz(
    n: int, k: int, p_rewire: float, seed: int
) -> np.ndarray:
  """Builds ring lattice of n nodes with Watts-Strogatz rewiring."""
  rng = np.random.default_rng(seed)
  edges = []
  for i in range(n):
    for offset in range(1, k + 1):
      j = (i + offset) % n
      if rng.random() < p_rewire:
        rewired = int(rng.integers(0, n))
        while rewired == i:
          rewired = int(rng.integers(0, n))
        edges.append([i, rewired])
      else:
        edges.append([i, j])

  all_edges = np.array(edges, dtype=np.int64)
  all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]
  pairs = np.sort(all_edges, axis=1)
  return np.unique(pairs, axis=0)


def _ba(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
  """Generates Barabási–Albert graph edges."""
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
  return np.concatenate(out)


def _community(
    n: int, block: int, d_in: int, d_out: int, rng: np.random.Generator
) -> np.ndarray:
  """Generates dense SBM graph edges."""
  k_in = n * d_in // 2
  uu = rng.integers(0, n, k_in)
  vv = (uu // block) * block + rng.integers(0, block, k_in)
  vv = np.minimum(vv, n - 1)
  k_out = n * d_out // 2
  e = np.concatenate(
      [np.column_stack([uu, vv]), rng.integers(0, n, (k_out, 2))]
  )
  return e


def _perc_grid(side: int, p: float, rng: np.random.Generator) -> np.ndarray:
  """Generates bond-percolated 2D grid edges."""
  idx = np.arange(side * side).reshape(side, side)
  h = np.column_stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()])
  vt = np.column_stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])
  e = np.concatenate([h, vt])
  return e[rng.random(len(e)) < p]


def _induced(
    n: int, edges: np.ndarray, frac: float, rng: np.random.Generator
) -> tuple[int, np.ndarray]:
  """Induces edges on a random fraction of nodes."""
  keep = rng.random(n) < frac
  new_id = np.cumsum(keep) - 1
  e = edges[keep[edges[:, 0]] & keep[edges[:, 1]]]
  return int(keep.sum()), new_id[e]


def _make_star(n: int) -> np.ndarray:
  """Builds a star graph with 1 hub at node 0 and n-1 leaves."""
  return np.column_stack(
      [np.zeros(n - 1, dtype=np.int64), np.arange(1, n, dtype=np.int64)]
  )


def _make_triangles(num_triangles: int) -> np.ndarray:
  """Builds disjoint triangles."""
  edges = []
  for i in range(num_triangles):
    u = 3 * i
    edges.append([u, u + 1])
    edges.append([u + 1, u + 2])
    edges.append([u, u + 2])
  return np.array(edges, dtype=np.int64)


def _make_clique(n: int) -> np.ndarray:
  """Builds complete graph K_n."""
  u, v = np.triu_indices(n, k=1)
  return np.column_stack([u, v])


class PartitionersTest(parameterized.TestCase):

  def test_partition_metrics_computations(self):
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    # 4 nodes in 2 clusters: {0, 1}, {2, 3}
    cluster_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    m = partitioners.partition_metrics(4, edges, cluster_ids)
    self.assertEqual(m["num_clusters"], 2.0)
    self.assertEqual(m["mean_cluster_size"], 2.0)
    self.assertEqual(m["median_cluster_size"], 2.0)
    self.assertEqual(m["max_cluster_size"], 2.0)
    self.assertEqual(m["min_cluster_size"], 2.0)
    self.assertAlmostEqual(m["size_inflation_factor"], 1.0, places=6)
    # Only edge (1, 2) is cut out of 3 edges
    self.assertAlmostEqual(m["edge_cut_fraction"], 1.0 / 3.0, places=6)

  def test_ring_basic(self):
    n = 1000
    target = 25
    edges = _make_ring_graph(n)
    ids = partitioners.partition_graph(
        n, edges, target_cluster_size=target, seed=42
    )
    m = partitioners.partition_metrics(n, edges, ids)
    cap = math.ceil(1.5 * target)
    self.assertEqual(ids.size, n)
    self.assertLessEqual(m["max_cluster_size"], cap)
    self.assertLessEqual(m["size_inflation_factor"], 1.25)
    self.assertGreaterEqual(m["num_clusters"], math.ceil(n / cap))

  @parameterized.named_parameters(
      ("torus_cs10", "torus", 10),
      ("torus_cs25", "torus", 25),
      ("torus_cs50", "torus", 50),
      ("ws_cs10", "watts_strogatz", 10),
      ("ws_cs25", "watts_strogatz", 25),
      ("ws_cs50", "watts_strogatz", 50),
  )
  def test_cluster_count_and_size_control_n20000(self, topology: str, cs: int):
    """Verifies G in [0.80, 4/3] * n/cs, G >= ceil(n/cap), and max_size <= cap."""
    if not _LONG and not (topology == "torus" and cs == 25):
      return  # Short mode tests a single representative case
    n = 20000
    if topology == "torus":
      edges = _build_grid_2d(100, 200)
    elif topology == "watts_strogatz":
      edges = _build_watts_strogatz(n=20000, k=4, p_rewire=0.02, seed=20260923)
    else:
      raise ValueError(f"Unknown topology: {topology}")

    ids = partitioners.partition_graph(
        n, edges, target_cluster_size=cs, max_size_ratio=1.5, seed=0
    )
    m = partitioners.partition_metrics(n, edges, ids)
    g = int(m["num_clusters"])
    max_c = int(m["max_cluster_size"])
    target_g = n / cs
    ratio = g / target_g
    cap = math.ceil(1.5 * cs)

    if _LONG:
      print(
          f"[{topology} cs={cs}] G={g} (target={target_g:.0f},"
          f" G/target={ratio:.4f}), max_size={max_c} (cap={cap})"
      )

    self.assertBetween(
        ratio,
        0.80,
        4.0 / 3.0,
        f"Topology {topology} cs={cs}: G/target={ratio:.4f} outside [0.80, 4/3]",
    )
    self.assertLessEqual(
        max_c,
        cap,
        f"Topology {topology} cs={cs}: max_size={max_c} exceeded cap={cap}",
    )
    self.assertGreaterEqual(
        g,
        math.ceil(n / cap),
        f"Topology {topology} cs={cs}: G={g} below ceil(n/cap)",
    )

  @parameterized.named_parameters(
      (f"{topo}_cs{cs}_seed{seed}", topo, cs, seed)
      for topo in ("torus", "watts_strogatz")
      for cs in (10, 25, 50)
      for seed in range(5)
  )
  def test_cluster_count_robustness_seeds_0_to_4(
      self, topology: str, cs: int, seed: int
  ):
    """Asserts 0.80 <= G/(n/cs) <= 4/3, G >= ceil(n/cap), and max_cluster_size <= cap."""
    if not _LONG and not (topology == "torus" and cs == 25 and seed == 0):
      return  # Short mode tests a single representative case
    n = 20000
    if topology == "torus":
      edges = _build_grid_2d(100, 200)
    elif topology == "watts_strogatz":
      edges = _build_watts_strogatz(n=20000, k=4, p_rewire=0.02, seed=seed)
    else:
      raise ValueError(f"Unknown topology: {topology}")

    ids = partitioners.partition_graph(
        n, edges, target_cluster_size=cs, max_size_ratio=1.5, seed=seed
    )
    m = partitioners.partition_metrics(n, edges, ids)
    g = int(m["num_clusters"])
    max_c = int(m["max_cluster_size"])
    target_g = n / cs
    ratio = g / target_g
    cap = math.ceil(1.5 * cs)

    if _LONG:
      print(
          f"[{topology} cs={cs} seed={seed}] G={g} (target={target_g:.0f},"
          f" G/target={ratio:.4f}), max_size={max_c} (cap={cap})"
      )

    self.assertBetween(
        ratio,
        0.80,
        4.0 / 3.0,
        f"Topology {topology} cs={cs} seed={seed}: G/target={ratio:.4f} outside [0.80, 4/3]",
    )
    self.assertLessEqual(
        max_c,
        cap,
        f"Topology {topology} cs={cs} seed={seed}: max_size={max_c} exceeded cap={cap}",
    )
    self.assertGreaterEqual(
        g,
        math.ceil(n / cap),
        f"Topology {topology} cs={cs} seed={seed}: G={g} below ceil(n/cap)",
    )

  def _assert_partition_properties(
      self,
      n: int,
      edges: np.ndarray,
      t: int,
      family_name: str,
      seed: int = 42,
  ) -> None:
    t0 = time.perf_counter()
    labels = partitioners.partition_graph(
        n, edges, target_cluster_size=t, seed=seed
    )
    wall_time = time.perf_counter() - t0

    cap = int(math.ceil(1.5 * t))
    min_keep = int(math.ceil(t / 2.0))

    # 1. labels.size == n
    self.assertEqual(labels.size, n)

    # 2. labels are exactly 0..G-1
    unique_labels, counts = np.unique(labels, return_counts=True)
    g = int(len(unique_labels))
    np.testing.assert_array_equal(unique_labels, np.arange(g, dtype=np.int64))

    # 3. max size <= cap
    max_c = int(np.max(counts))
    self.assertLessEqual(max_c, cap)

    # 4. at most one cluster < ceil(t/2)
    small_clusters_count = int(np.count_nonzero(counts < min_keep))
    self.assertLessEqual(small_clusters_count, 1)

    # 5. G >= ceil(n / cap)
    self.assertGreaterEqual(g, int(math.ceil(n / float(cap))))

    # 6. Two calls with the same seed give identical labels
    labels_repeat = partitioners.partition_graph(
        n, edges, target_cluster_size=t, seed=seed
    )
    np.testing.assert_array_equal(labels, labels_repeat)

    # Compute partition metrics for long-mode reporting
    m = partitioners.partition_metrics(n, edges, labels)
    target_g = float(n) / float(t)
    g_ratio = float(g) / target_g
    median_c = float(np.median(counts))
    m_bar = float(n) / float(g)
    n_max_over_mbar = float(max_c) / m_bar
    one_plus_cv2 = float(m["size_inflation_factor"])
    cut = float(m["edge_cut_fraction"])

    if _LONG:
      print(
          f"{family_name} t={t}: G/target={g_ratio:.4f},"
          f" median={median_c:.1f}, n_max/mbar={n_max_over_mbar:.3f},"
          f" 1+CV2={one_plus_cv2:.4f}, cut={cut:.4f}, wall_time={wall_time:.3f}s",
          flush=True,
      )

  @parameterized.named_parameters(
      ("star", "star"),
      ("dense_ba", "dense_ba"),
      ("dense_sbm", "dense_sbm"),
      ("ba_m1_isolated", "ba_m1_isolated"),
      ("no_edges", "no_edges"),
      ("disjoint_triangles", "disjoint_triangles"),
      ("clique_k200", "clique_k200"),
      ("percolated_grid", "percolated_grid"),
      ("n7_t10", "n7_t10"),
  )
  def test_property_graph_families(self, family: str):
    rng = np.random.default_rng(2026)
    if family == "star":
      n = 5000 if _LONG else 1000
      edges = _make_star(n)
      targets = [1, 2, 3, 10, 50, 200] if _LONG else [10]
    elif family == "dense_ba":
      n_raw = 6000 if _LONG else 1000
      raw_edges = _ba(n_raw, 60 if _LONG else 10, rng)
      n, edges = _induced(n_raw, raw_edges, 0.8, rng)
      targets = [1, 2, 3, 10, 50, 200] if _LONG else [10]
    elif family == "dense_sbm":
      n = 6000 if _LONG else 1200
      edges = _community(
          n,
          block=300 if _LONG else 200,
          d_in=100 if _LONG else 20,
          d_out=40 if _LONG else 10,
          rng=rng,
      )
      targets = [1, 2, 3, 10, 50, 200] if _LONG else [10]
    elif family == "ba_m1_isolated":
      n_ba = 8000 if _LONG else 1000
      n_iso = 4500 if _LONG else 500
      n = n_ba + n_iso
      edges = _ba(n_ba, 1, rng)
      targets = [1, 2, 3, 10, 50, 200] if _LONG else [10]
    elif family == "no_edges":
      n = 1000 if _LONG else 200
      edges = np.zeros((0, 2), dtype=np.int64)
      targets = [1, 2, 3, 10, 50, 200] if _LONG else [10]
    elif family == "disjoint_triangles":
      n = 9000 if _LONG else 1500
      edges = _make_triangles(3000 if _LONG else 500)
      targets = [1, 2, 3, 10, 50, 200] if _LONG else [10]
    elif family == "clique_k200":
      n = 200
      edges = _make_clique(n)
      targets = [1, 2, 3, 10, 50, 200] if _LONG else [10]
    elif family == "percolated_grid":
      side = 100 if _LONG else 40
      n = side * side
      edges = _perc_grid(side, 0.55, rng)
      targets = [1, 2, 3, 10, 50, 200] if _LONG else [10]
    elif family == "n7_t10":
      n = 7
      edges = np.array(
          [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=np.int64
      )
      targets = [10]
    else:
      raise ValueError(f"Unknown family: {family}")

    for t in targets:
      if t <= n or family == "n7_t10":
        self._assert_partition_properties(n, edges, t, family)

  def test_edge_hygiene(self):
    """Verifies duplicate, reversed, and self-loop edges give identical labels."""
    rng = np.random.default_rng(42)
    n = 1000
    clean_edges = _ba(n, 10, rng)
    dup_edges = np.concatenate([clean_edges, clean_edges])
    num_dup = len(dup_edges)
    half_idx = rng.choice(num_dup, size=num_dup // 2, replace=False)
    dup_edges[half_idx] = dup_edges[half_idx, ::-1]
    self_loops = np.column_stack([np.arange(100), np.arange(100)])
    messy_edges = np.concatenate([dup_edges, self_loops])

    target = 25
    clean_labels = partitioners.partition_graph(
        n, clean_edges, target_cluster_size=target, seed=123
    )
    messy_labels = partitioners.partition_graph(
        n, messy_edges, target_cluster_size=target, seed=123
    )
    np.testing.assert_array_equal(clean_labels, messy_labels)

  def test_ba_m1_isolated_nodes_regression(self):
    """Verifies that BA m=1 + isolated nodes maintains median >= 25 and 1+CV^2 <= 1.2 at t=50."""
    rng = np.random.default_rng(2026)
    n_ba = 8000
    n_iso = 4500
    n = n_ba + n_iso
    ba_edges = _ba(n_ba, 1, rng)
    target = 50
    labels = partitioners.partition_graph(
        n, ba_edges, target_cluster_size=target, seed=0
    )
    m = partitioners.partition_metrics(n, ba_edges, labels)
    if _LONG:
      print(
          f"BA m=1 + isolated regression: median={m['median_cluster_size']:.1f},"
          f" 1+CV2={m['size_inflation_factor']:.4f}",
          flush=True,
      )
    self.assertGreaterEqual(m["median_cluster_size"], 25.0)
    self.assertLessEqual(m["size_inflation_factor"], 1.2)

  def test_invalid_inputs(self):
    """Verifies ValueError is raised on invalid inputs."""
    edges = np.array([[0, 1]], dtype=np.int64)
    # Node id >= n
    with self.assertRaises(ValueError):
      partitioners.partition_graph(2, np.array([[0, 2]], dtype=np.int64), 1)
    # Node id < 0
    with self.assertRaises(ValueError):
      partitioners.partition_graph(2, np.array([[-1, 0]], dtype=np.int64), 1)
    # max_size_ratio < 1.5
    with self.assertRaises(ValueError):
      partitioners.partition_graph(10, edges, 2, max_size_ratio=1.4)
    # num_nodes <= 0
    with self.assertRaises(ValueError):
      partitioners.partition_graph(0, edges, 2)
    # target_cluster_size <= 0
    with self.assertRaises(ValueError):
      partitioners.partition_graph(10, edges, 0)


if __name__ == "__main__":
  absltest.main()
