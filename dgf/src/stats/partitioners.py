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

"""Distributed-capable, size-constrained graph partitioner for cluster bounds.

Why standard graph partitioners fail the statistical contract:
1. Cluster-size imbalance directly inflates M_hat_G by 1 + CV^2(m_c):
   Even on noise-free constant losses s_i = theta, cluster totals are
   S_c = m_c * theta, so
     M_hat_G = (G * sum_c S_c^2) / (sum_c S_c)^2 = (G * sum_c m_c^2) / n^2
             = 1 + CV^2(m_c).
   Unconstrained modularity methods (e.g. Louvain/Leiden) produce power-law
   cluster sizes where 1 + CV^2(m_c) > 4, causing immediate false refutation
   even on light-tailed errors. Strict size balance (CV(m_c) << 1) is therefore
   a mathematical requirement.
2. Distributable by construction:
   Every step below is a group-by, join, sort or prefix sum over edges,
   cluster pairs or clusters, with O(1) state per node and per cluster. The
   only known issue at scale is key skew on hub clusters in the absorption
   step, which a combiner handles without changing the result.

Algorithm (t = target size, cap = ceil(max_size_ratio * t), k = ceil(t / 2);
a cluster is small if its size is < k):
1. Coarsening: start from singletons. Each round, every cluster below t
   proposes to the neighbour below t that maximises the edge density
   w_ab / (size_a * size_b), subject to size_a + size_b <= cap, under a strict
   total order (density, then a seeded hash, then ids). Mutual proposals
   merge; a cluster that receives proposals absorbs them in order while it is
   below t and within cap. The cluster graph is contracted after each round.
2. Fill: each small cluster joins its most strongly connected non-small
   neighbour if that stays within cap.
3. Pack: the remaining small clusters, sorted by (that neighbour, smallest
   node id), are cut into bins of t by prefix sums of their sizes. A last bin
   below k merges into the previous one if they fit within cap.

Size bounds (for max_size_ratio >= 1.5):
- Every cluster has size <= cap. Steps 1 and 2 check cap on every merge. In
  step 3 every unit is < k and a bin starts below (b + 1) * t, so a bin has
  size at most (t - 1) + (k - 1) <= cap; this needs cap >= t + k - 2, hence
  the ratio requirement.
- At most one cluster is smaller than k. Clusters leaving steps 1 and 2 are
  non-small or are packed; every bin except the last spans from
  < b * t + (k - 1) to >= (b + 1) * t, hence has size >= t - k + 2 >= k.
- Therefore G >= ceil(n / cap), and apart from at most one cluster
  n_max / m_bar <= cap / k (about 3). Measured on synthetic and real graphs:
  1.0 to 1.65.
- Deterministic given the seed; independent of the losses by construction.

Background: size-normalised heavy-edge scores are the METIS-style
normalisation. Pairwise matching alone stalls on complex networks (hubs and
stars), the failure that size-constrained label propagation was introduced
for (Meyerhenke, Sanders and Schulz, "Partitioning Complex Networks via
Size-constrained Clustering", SEA 2014). The absorption rule in step 1 lets a
cluster take several proposers per round for the same reason.
"""

import math
from typing import Any

import numpy as np

__all__ = [
    "partition_graph",
    "partition_metrics",
]


def partition_metrics(
    num_nodes: int,
    edges: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, float]:
  """Computes structural and statistical quality metrics for a graph partitioning.

  Args:
    num_nodes: Total number of vertices n.
    edges: Array of shape (E, 2) containing undirected edge endpoint pairs.
    cluster_ids: 1-D array of length num_nodes containing cluster assignments.

  Returns:
    Dictionary with:
      num_clusters: G
      mean_cluster_size: n / G
      max_cluster_size: max_c m_c
      min_cluster_size: min_c m_c
      size_inflation_factor: 1 + CV^2(m_c) = G * sum(m_c^2) / n^2 (>= 1.0)
      edge_cut_fraction: fraction of edges cut across cluster boundaries
  """
  c_ids = np.asarray(cluster_ids, dtype=np.int64)
  if c_ids.size != num_nodes:
    raise ValueError(
        f"cluster_ids size ({c_ids.size}) must match num_nodes ({num_nodes})."
    )

  _, counts = np.unique(c_ids, return_counts=True)
  num_clusters = int(counts.size)
  mean_size = (
      float(num_nodes) / float(num_clusters) if num_clusters > 0 else 0.0
  )
  max_size = float(np.max(counts)) if counts.size > 0 else 0.0
  min_size = float(np.min(counts)) if counts.size > 0 else 0.0
  size_inflation = float(num_clusters * np.sum(counts**2)) / float(num_nodes**2)

  ed = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
  if ed.size == 0 or ed.shape[0] == 0:
    edge_cut = 0.0
  else:
    u = ed[:, 0]
    v = ed[:, 1]
    cut_count = np.count_nonzero(c_ids[u] != c_ids[v])
    edge_cut = float(cut_count) / float(ed.shape[0])

  return {
      "num_clusters": float(num_clusters),
      "mean_cluster_size": mean_size,
      "median_cluster_size": float(np.median(counts)) if counts.size > 0 else 0.0,
      "max_cluster_size": max_size,
      "min_cluster_size": min_size,
      "size_inflation_factor": size_inflation,
      "edge_cut_fraction": edge_cut,
  }


def _pair_hash(
    seed: int,
    round_idx: int,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
  """Computes deterministic 64-bit splitmix-style hash for pairs (a, b)."""
  with np.errstate(over="ignore"):
    x = np.uint64(seed) ^ (np.uint64(round_idx) * np.uint64(0x9E3779B97F4A7C15))
    x ^= (
        a.astype(np.uint64)
        + np.uint64(0x517CC1B727220A95)
        + (x << np.uint64(6))
        + (x >> np.uint64(2))
    )
    x ^= (
        b.astype(np.uint64)
        + np.uint64(0x9E3779B97F4A7C15)
        + (x << np.uint64(6))
        + (x >> np.uint64(2))
    )
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    h = x ^ (x >> np.uint64(31))
  return h


def _partition_traced(
    num_nodes: int,
    edges: np.ndarray,
    target_cluster_size: int,
    max_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
  """Graph partitioning via contraction coarsening, fill, and prefix packing."""
  n = int(num_nodes)
  t = int(target_cluster_size)
  cap = int(max_size)
  min_keep = int(math.ceil(t / 2.0))

  # 0. Edges: validate, drop self-loops, canonicalize to (min, max), deduplicate
  ed = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
  if ed.size > 0:
    if np.any(ed < 0) or np.any(ed >= n):
      raise ValueError("Edge node ids must be in [0, num_nodes).")
    mask = ed[:, 0] != ed[:, 1]
    ed = ed[mask]
    if ed.size > 0:
      u_min = np.minimum(ed[:, 0], ed[:, 1])
      u_max = np.maximum(ed[:, 0], ed[:, 1])
      keys = u_min * n + u_max
      unique_keys = np.unique(keys)
      pa = unique_keys // n
      pb = unique_keys % n
      w = np.ones(len(unique_keys), dtype=np.float64)
    else:
      pa = np.zeros(0, dtype=np.int64)
      pb = np.zeros(0, dtype=np.int64)
      w = np.zeros(0, dtype=np.float64)
  else:
    pa = np.zeros(0, dtype=np.int64)
    pb = np.zeros(0, dtype=np.int64)
    w = np.zeros(0, dtype=np.float64)

  cluster = np.arange(n, dtype=np.int64)
  size = np.ones(n, dtype=np.int64)
  G = n

  rounds = 0
  merges_per_round: list[int] = []
  g_per_round: list[int] = []
  stop_reason = "round_cap"
  max_rounds = 100

  # 1. Coarsening by graph contraction
  for r in range(max_rounds):
    if pa.size == 0:
      stop_reason = "converged"
      break

    # 1.1 Eligible pairs
    ok = (size[pa] < t) & (size[pb] < t) & (size[pa] + size[pb] <= cap)
    if not np.any(ok):
      stop_reason = "converged"
      break

    pa_ok = pa[ok]
    pb_ok = pb[ok]
    w_ok = w[ok]

    # 1.2 Score
    score = w_ok / (size[pa_ok] * size[pb_ok]).astype(np.float64)

    # 1.3 Strict total order on pairs: (score, h_ab, a, b) largest first
    h_ab = _pair_hash(seed, r, pa_ok, pb_ok)
    # lexsort sorts ascending with primary key last
    pair_order = np.lexsort((pb_ok, pa_ok, h_ab, score))
    # Rank array: higher rank means larger key in total order
    rank = np.empty(len(pa_ok), dtype=np.int64)
    rank[pair_order] = np.arange(len(pa_ok), dtype=np.int64)

    # 1.4 Directed proposals
    src = np.concatenate([pa_ok, pb_ok])
    dst = np.concatenate([pb_ok, pa_ok])
    edge_rank = np.concatenate([rank, rank])

    dir_order = np.lexsort((edge_rank, src))
    s_sorted = src[dir_order]
    d_sorted = dst[dir_order]
    r_sorted = edge_rank[dir_order]

    is_last = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    proposers = s_sorted[is_last]
    targets = d_sorted[is_last]
    prop_rank = r_sorted[is_last]

    prop = np.full(G, -1, dtype=np.int64)
    prop[proposers] = targets
    best_prop_rank = np.full(G, -1, dtype=np.int64)
    best_prop_rank[proposers] = prop_rank

    # 1.5 Mutual pairs
    has_prop = prop >= 0
    idx = np.nonzero(has_prop)[0]
    mutual = np.zeros(G, dtype=bool)
    mutual[idx] = prop[prop[idx]] == idx

    root = np.arange(G, dtype=np.int64)
    m_idx = np.nonzero(mutual)[0]
    root[m_idx] = np.minimum(m_idx, prop[m_idx])

    # 1.6 Receivers
    nm = has_prop & ~mutual
    nm_proposers = np.nonzero(nm)[0]
    received_nm = np.zeros(G, dtype=bool)
    if nm_proposers.size > 0:
      received_nm[prop[nm_proposers]] = True

    # 1.7 Acceptance of remaining proposers
    remaining = np.nonzero(nm & ~received_nm)[0]
    merged_size = np.bincount(root, weights=size, minlength=G).astype(np.int64)
    new_cluster = root.copy()

    if remaining.size > 0:
      target_roots = root[prop[remaining]]
      p_ranks = best_prop_rank[remaining]

      grp_order = np.lexsort((-p_ranks, target_roots))
      p_ord = remaining[grp_order]
      r_ord = target_roots[grp_order]

      s_ord = size[p_ord]
      cs = np.cumsum(s_ord)
      is_start = np.r_[True, r_ord[1:] != r_ord[:-1]]
      grp_offset = np.maximum.accumulate(np.where(is_start, cs - s_ord, 0))
      before_p = cs - s_ord - grp_offset

      base = merged_size[r_ord]
      accepted = (base + before_p < t) & (base + before_p + s_ord <= cap)

      new_cluster[p_ord[accepted]] = r_ord[accepted]
      accepted_count = int(np.count_nonzero(accepted))
    else:
      accepted_count = 0

    mutual_merges_count = len(m_idx) // 2
    merges_this_round = mutual_merges_count + accepted_count
    merges_per_round.append(merges_this_round)
    rounds = r + 1

    if merges_this_round == 0:
      stop_reason = "converged"
      break

    # 1.8 Contract
    unique_roots, dense_map = np.unique(new_cluster, return_inverse=True)
    cluster = dense_map[cluster]
    G_new = len(unique_roots)
    g_per_round.append(G_new)
    size = np.bincount(cluster, minlength=G_new).astype(np.int64)

    new_pa = dense_map[pa]
    new_pb = dense_map[pb]
    mask_edges = new_pa != new_pb
    if not np.any(mask_edges):
      pa = np.zeros(0, dtype=np.int64)
      pb = np.zeros(0, dtype=np.int64)
      w = np.zeros(0, dtype=np.float64)
      G = G_new
      stop_reason = "converged"
      break

    u1 = np.minimum(new_pa[mask_edges], new_pb[mask_edges])
    u2 = np.maximum(new_pa[mask_edges], new_pb[mask_edges])
    w_filtered = w[mask_edges]
    pair_keys = u1 * G_new + u2
    unique_pair_keys, pair_inverse = np.unique(pair_keys, return_inverse=True)
    pa = unique_pair_keys // G_new
    pb = unique_pair_keys % G_new
    w = np.bincount(pair_inverse, weights=w_filtered).astype(np.float64)
    G = G_new

  # 2. Fill
  g_curr = int(len(size))
  small_mask = size < min_keep
  num_filled = 0
  num_units = 0
  num_bins = 0

  if np.any(small_mask):
    anchor = np.full(g_curr, -1, dtype=np.int64)
    anchor_w = np.zeros(g_curr, dtype=np.float64)

    if pa.size > 0:
      src = np.concatenate([pa, pb])
      dst = np.concatenate([pb, pa])
      ww = np.concatenate([w, w])

      valid_edge = small_mask[src] & ~small_mask[dst]
      if np.any(valid_edge):
        s_sm = src[valid_edge]
        d_nsm = dst[valid_edge]
        w_sm = ww[valid_edge]

        ord_anchor = np.lexsort((d_nsm, -w_sm, s_sm))
        s_sm_ord = s_sm[ord_anchor]
        d_nsm_ord = d_nsm[ord_anchor]
        w_sm_ord = w_sm[ord_anchor]

        first_occ = np.r_[True, s_sm_ord[1:] != s_sm_ord[:-1]]
        anchor[s_sm_ord[first_occ]] = d_nsm_ord[first_occ]
        anchor_w[s_sm_ord[first_occ]] = w_sm_ord[first_occ]

    small_indices = np.nonzero(small_mask)[0]
    has_anchor = small_indices[anchor[small_indices] >= 0]
    fill_cluster = np.arange(g_curr, dtype=np.int64)

    if has_anchor.size > 0:
      ord_fill = np.lexsort(
          (has_anchor, -anchor_w[has_anchor], anchor[has_anchor])
      )
      u_ord = has_anchor[ord_fill]
      a_ord = anchor[u_ord]

      s_u = size[u_ord]
      cs_fill = np.cumsum(s_u)
      is_start_fill = np.r_[True, a_ord[1:] != a_ord[:-1]]
      grp_off_fill = np.maximum.accumulate(
          np.where(is_start_fill, cs_fill - s_u, 0)
      )
      before_u = cs_fill - s_u - grp_off_fill

      base_a = size[a_ord]
      accepted_fill = base_a + before_u + s_u <= cap

      fill_cluster[u_ord[accepted_fill]] = a_ord[accepted_fill]
      num_filled = int(np.count_nonzero(accepted_fill))

      unaccepted_u = u_ord[~accepted_fill]
      unanchored = small_indices[anchor[small_indices] < 0]
      units = np.concatenate([unaccepted_u, unanchored])
    else:
      units = small_indices

    num_units = len(units)

    # 3. Pack
    if num_units > 0:
      first_node = np.full(g_curr, n, dtype=np.int64)
      np.minimum.at(first_node, cluster, np.arange(n, dtype=np.int64))

      a_key = np.where(anchor[units] >= 0, anchor[units], g_curr + 1)
      fn = first_node[units]
      ord_pack = np.lexsort((fn, a_key))
      sorted_units = units[ord_pack]

      unit_sizes = size[sorted_units]
      p_j = np.cumsum(unit_sizes) - unit_sizes
      bin_ids = p_j // t

      b_sizes = np.bincount(bin_ids, weights=unit_sizes).astype(np.int64)
      num_bins_initial = len(b_sizes)
      if num_bins_initial >= 2:
        last_b = num_bins_initial - 1
        if b_sizes[last_b] < min_keep and (
            b_sizes[last_b - 1] + b_sizes[last_b] <= cap
        ):
          bin_ids[bin_ids == last_b] = last_b - 1
          num_bins = num_bins_initial - 1
        else:
          num_bins = num_bins_initial
      else:
        num_bins = num_bins_initial

      fill_cluster[sorted_units] = g_curr + bin_ids

    cluster = fill_cluster[cluster]

  # 4. Output: dense labels 0..G-1
  _, final_labels = np.unique(cluster, return_inverse=True)
  trace = {
      "rounds": rounds,
      "stop_reason": stop_reason,
      "g_per_round": g_per_round,
      "merges_per_round": merges_per_round,
      "num_filled": num_filled,
      "num_units": num_units,
      "num_bins": num_bins,
  }
  return final_labels, trace


def partition_graph(
    num_nodes: int,
    edges: np.ndarray,
    target_cluster_size: int,
    max_size_ratio: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
  """Partitions graph vertices into balanced clusters with deterministic size bounds.

  Properties:
    - max_cluster_size <= ceil(max_size_ratio * target_cluster_size).
    - At most one cluster has size < ceil(target_cluster_size / 2).
    - G >= ceil(num_nodes / max_size).

  Args:
    num_nodes: Total number of vertices n > 0.
    edges: Array of shape (E, 2) of 0-indexed undirected edge endpoint pairs.
    target_cluster_size: Desired nominal cluster size t > 0.
    max_size_ratio: Upper cap multiplier max_size = ceil(max_size_ratio * t).
      Must be >= 1.5 to satisfy the packing bound cap >= t + k - 2.
    seed: Random seed for deterministic reproducibility.

  Returns:
    1-D int64 array of length num_nodes with dense cluster labels in 0..G-1.

  Raises:
    ValueError: On invalid inputs (num_nodes <= 0, target_cluster_size <= 0,
      max_size_ratio < 1.5, or edge node IDs outside [0, num_nodes)).
  """
  if num_nodes <= 0:
    raise ValueError(f"num_nodes must be positive, got {num_nodes}.")
  if target_cluster_size <= 0:
    raise ValueError(
        f"target_cluster_size must be positive, got {target_cluster_size}."
    )
  if max_size_ratio < 1.5:
    raise ValueError(
        f"max_size_ratio must be >= 1.5 (got {max_size_ratio}) to satisfy the"
        " packing bound cap >= t + k - 2."
    )

  max_size = int(math.ceil(max_size_ratio * float(target_cluster_size)))

  labels, _ = _partition_traced(
      num_nodes, edges, target_cluster_size, max_size, seed
  )
  return labels
