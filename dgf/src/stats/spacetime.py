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

"""Space-time confidence intervals on graphs with temporal items.

Provides the space-time entry point `spacetime_interval` for spatio-temporal
evaluation data, mapping (node, time-step) items to space-time product blocks.

Theory:
  THEORY_SPACETIME.md (§9):
  - Claim S1: Kronecker factorization and separable VIF kappa = kappa_s *
  kappa_t.
  - Claim S2B: Topological spectral bounds under non-separable correlation
  dominance.
  - Proposition D': Window-split drift check on space-time product partitions.
  - Proposition S5: Design-only candidate partition choice (choose_partition)
    satisfies Lemma R exogeneity.
"""

import dataclasses
import math
from typing import Any, Callable, Sequence

import numpy as np

from dgf.src.stats import cluster_bound as _cluster_bound
from dgf.src.stats import cluster_sketch as _cluster_sketch
from dgf.src.stats import graph as _graph
from dgf.src.stats import partitioners as _partitioners
from dgf.src.stats import refutation as _refutation
from dgf.src.stats import temporal as _temporal
from dgf.src.stats.independent import interval as _interval
from dgf.src.stats.independent import metrics as _metrics

__all__ = [
    "Candidate",
    "Choice",
    "MarginalChecks",
    "SpaceTimeIndex",
    "SpaceTimeResult",
    "candidate_partitions",
    "choose_partition",
    "separable_kappa",
    "spacetime_edges",
    "spacetime_interval",
    "spacetime_items",
    "topological_kappa_fn",
]

_REFUTATION_LEVEL: float = 0.95


@dataclasses.dataclass(frozen=True)
class SpaceTimeIndex:
  """Mapping between (node, time-step) items and dense indices 0..n-1.

  Attributes:
    node_ids: Array of node identifiers per item.
    times: Array of integer time steps per item.
    unique_nodes: Array of unique node identifiers.
    unique_times: Array of unique integer time steps, sorted ascending.
    num_nodes: Total number of unique spatial nodes.
    num_times: Total number of unique time steps.
    n_items: Total number of observed items.
    is_complete_grid: True if n_items == num_nodes * num_times.
  """

  node_ids: np.ndarray
  times: np.ndarray
  unique_nodes: np.ndarray
  unique_times: np.ndarray
  num_nodes: int
  num_times: int
  n_items: int
  is_complete_grid: bool


@dataclasses.dataclass(frozen=True)
class Candidate:
  """Candidate partition for space-time graph evaluation.

  Attributes:
    name: Descriptive identifier of the candidate partition.
    cluster_ids: Array of length n_items mapping each item to a cluster ID in
      0..G-1.
    G: Number of realized non-empty space-time clusters.
    G_s: Number of spatial communities in the partition.
    G_t: Number of time windows in the partition.
    n_max: Maximum items in any single cluster.
    m_bar: Mean items per cluster (n / G).
    community_ids: Array of length n_items mapping each item to a dense
      community index in 0..G_s-1 (all zeros for window-only candidates).
    window_ids: Array of length n_items mapping each item to a dense window
      index in 0..G_t-1 (all zeros for community-only candidates).
  """

  name: str
  cluster_ids: np.ndarray
  G: int
  G_s: int
  G_t: int
  n_max: int
  m_bar: float
  community_ids: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(0, dtype=np.int64)
  )
  window_ids: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(0, dtype=np.int64)
  )

  def __post_init__(self) -> None:
    n = len(self.cluster_ids)
    if len(self.community_ids) != n:
      object.__setattr__(self, "community_ids", np.zeros(n, dtype=np.int64))
    if len(self.window_ids) != n:
      object.__setattr__(self, "window_ids", np.zeros(n, dtype=np.int64))


@dataclasses.dataclass(frozen=True)
class Choice:
  """Outcome of partition selection across candidate partitions.

  Attributes:
    winner: The selected Candidate partition maximizing N_eff.
    winner_kappa: Certified variance inflation factor kappa for the winner.
    winner_r_eff: Size-adjusted imbalance factor r_eff for the winner.
    winner_n_eff: Size-adjusted effective sample size N_eff = G / (kappa *
      r_eff).
    candidates_table: List of diagnostic dictionaries for all scored candidates.
    cluster_edges: Undirected cluster adjacency edges (E_c, 2) for the winner.
    route: Route name ("separable", "topological", "fallback_topological", "range").
    route_reason: Human-readable rationale for the selected kappa route.
    kappa_s: Declared spatial kappa if route is separable, else None.
    kappa_t: Declared temporal kappa if route is separable, else None.
    m_item: Item-level bound passed to choose_partition, or None if not set.
  """

  winner: Candidate
  winner_kappa: float
  winner_r_eff: float
  winner_n_eff: float
  candidates_table: list[dict[str, Any]]
  cluster_edges: np.ndarray
  route: str
  route_reason: str
  kappa_s: float | None = None
  kappa_t: float | None = None
  m_item: float | None = None


@dataclasses.dataclass(frozen=True)
class MarginalChecks:
  """Marginal refutation checks under separability (Claim S1(3)).

  Attributes:
    temporal: Refutation check on marginal window totals against kappa_t.
    spatial: Refutation check on marginal community totals against kappa_s.
  """

  temporal: _refutation.RefutationResult | None
  spatial: _refutation.RefutationResult | None


@dataclasses.dataclass(frozen=True)
class SpaceTimeResult(_graph.GraphResult):
  """Confidence interval result for space-time graph evaluation.

  Attributes:
    marginal_checks: Marginal refutation checks on window and community totals
      under separability (None for non-separable or single-axis partitions).
    drift: Outcome of the split-sample temporal drift diagnostic (Proposition
      D').
    choice: Choice metadata and diagnostics from candidate scoring.
    graph_result: Underlying GraphResult object.
  """

  marginal_checks: MarginalChecks | None = None
  drift: _temporal.DriftResult | None = None
  choice: Choice | None = None
  graph_result: _graph.GraphResult | None = None

  @property
  def low(self) -> float:
    return self.interval.low

  @property
  def high(self) -> float:
    return self.interval.high

  @property
  def status(self) -> Any:
    return self.interval.status


def spacetime_items(
    node_ids: Sequence[Any] | np.ndarray,
    times: Sequence[Any] | np.ndarray,
) -> SpaceTimeIndex:
  """Maps (node, time-step) pairs to dense item indices 0..n-1.

  Args:
    node_ids: Sequence or 1-D array of node identifiers for each observed item.
    times: Sequence or 1-D array of integer time steps for each observed item.

  Returns:
    SpaceTimeIndex containing item mappings and grid completeness status.

  Raises:
    ValueError: If node_ids and times have mismatched lengths, are empty,
      contain non-integer time steps, or contain duplicate (node, time) pairs.
  """
  n_nodes = len(node_ids)
  n_times = len(times)
  if n_nodes != n_times:
    raise ValueError(
        f"Length mismatch: len(node_ids)={n_nodes} != len(times)={n_times}."
    )
  if n_nodes == 0:
    raise ValueError("node_ids and times cannot be empty.")

  # Validate integer time steps (reject floats, booleans, and non-integers)
  times_arr = np.asarray(times)
  if times_arr.dtype == bool or not np.issubdtype(times_arr.dtype, np.integer):
    for idx, t in enumerate(times):
      if isinstance(t, bool) or not isinstance(t, (int, np.integer)):
        raise ValueError(
            f"Time steps must be integers; got {type(t).__name__} ({t!r}) at"
            f" index {idx}."
        )
    raise ValueError("Time steps must be integers.")

  if not isinstance(times, np.ndarray):
    for idx, t in enumerate(times):
      if isinstance(t, bool):
        raise ValueError(
            f"Time steps must be integers; got bool ({t!r}) at index {idx}."
        )

  times_int64 = times_arr.astype(np.int64)
  nodes_arr = np.asarray(node_ids)

  unique_nodes, node_inverse = np.unique(nodes_arr, return_inverse=True)
  unique_times, time_inverse = np.unique(times_int64, return_inverse=True)

  num_nodes = len(unique_nodes)
  num_times = len(unique_times)
  n_items = n_nodes

  key = node_inverse.astype(np.int64) * num_times + time_inverse.astype(
      np.int64
  )

  uniq_keys, key_counts = np.unique(key, return_counts=True)
  if np.any(key_counts > 1):
    dup_k = uniq_keys[np.where(key_counts > 1)[0][0]]
    dup_node = unique_nodes[dup_k // num_times]
    dup_time = unique_times[dup_k % num_times]
    raise ValueError(
        f"Duplicate (node, time) pair found: ({dup_node!r}, {dup_time})."
    )

  is_complete_grid = n_items == (num_nodes * num_times)

  return SpaceTimeIndex(
      node_ids=nodes_arr,
      times=times_int64,
      unique_nodes=unique_nodes,
      unique_times=unique_times,
      num_nodes=num_nodes,
      num_times=num_times,
      n_items=n_items,
      is_complete_grid=is_complete_grid,
  )


def spacetime_edges(
    index: SpaceTimeIndex,
    spatial_edges: np.ndarray,
    *,
    max_lag: int,
    phi_spatial: float,
    phi_temporal: Sequence[float] | np.ndarray,
    phi_cross: Sequence[float] | np.ndarray | None = None,
    max_edges: int = 20_000_000,
) -> tuple[np.ndarray, np.ndarray]:
  """Constructs the item-level strong product on observed items only.

  Edges generated:
    1. Same-time spatial edges: ((u, t), (v, t)) for spatial edges {u, v}
       observed at time t, with weight phi_spatial.
    2. Same-node temporal edges: ((u, t), (u, t + l)) for node u observed at
       times t and t + l, with weight phi_temporal[l - 1].
    3. Cross-lagged edges: ((u, t), (v, t + l)) and ((v, t), (u, t + l)) for
       spatial neighbours {u, v} at lag l = 1..max_lag, with weight
       phi_cross[l - 1]. The default is the dominance product
       phi_spatial * phi_temporal[l - 1].

  Memory and complexity are O(|E_s| * T * max_lag). Raises a clear error if the
  total edges exceed max_edges.

  Args:
    index: SpaceTimeIndex of evaluated items.
    spatial_edges: Undirected spatial edges between node identifiers or node
      indices, shape (E_s, 2).
    max_lag: Maximum temporal lag L >= 1.
    phi_spatial: Correlation upper bound for spatial edges in [0, 1].
    phi_temporal: Sequence of correlation bounds for temporal lags 1..max_lag.
    phi_cross: Sequence of correlation bounds for cross-lagged edges 1..max_lag.
      Defaults to phi_spatial * phi_temporal[l - 1].
    max_edges: Maximum allowed item edges to prevent memory exhaustion.

  Returns:
    Tuple of (edges, phi) where edges has shape (E, 2) and phi has shape (E,).

  Raises:
    ValueError: If parameters are invalid or edge count exceeds max_edges.
  """
  if max_lag < 1:
    raise ValueError(f"max_lag must be >= 1, got {max_lag}.")
  if not (math.isfinite(phi_spatial) and 0.0 <= phi_spatial <= 1.0):
    raise ValueError(
        f"phi_spatial must be finite and in [0, 1], got {phi_spatial}."
    )

  phi_temp_arr = np.asarray(phi_temporal, dtype=np.float64)
  if phi_temp_arr.ndim != 1 or len(phi_temp_arr) != max_lag:
    raise ValueError(
        f"phi_temporal must have length max_lag={max_lag}, got"
        f" {len(phi_temp_arr)}."
    )
  if not (
      np.all(np.isfinite(phi_temp_arr))
      and np.all(phi_temp_arr >= 0.0)
      and np.all(phi_temp_arr <= 1.0)
  ):
    raise ValueError("All phi_temporal values must be finite and in [0, 1].")

  if phi_cross is not None:
    phi_cross_arr = np.asarray(phi_cross, dtype=np.float64)
    if phi_cross_arr.ndim != 1 or len(phi_cross_arr) != max_lag:
      raise ValueError(
          f"phi_cross must have length max_lag={max_lag}, got"
          f" {len(phi_cross_arr)}."
      )
    if not (
        np.all(np.isfinite(phi_cross_arr))
        and np.all(phi_cross_arr >= 0.0)
        and np.all(phi_cross_arr <= 1.0)
    ):
      raise ValueError("All phi_cross values must be finite and in [0, 1].")
  else:
    # Default is the dominance product: phi_spatial * phi_temporal[l-1]
    phi_cross_arr = phi_spatial * phi_temp_arr

  s_edges = np.asarray(spatial_edges)
  if s_edges.ndim != 2 or s_edges.shape[1] != 2:
    raise ValueError(
        f"spatial_edges must have shape (E_s, 2), got {s_edges.shape}."
    )

  # Map spatial edges to dense 0..num_nodes-1 indices
  node_to_idx = {u: i for i, u in enumerate(index.unique_nodes)}
  mapped_u: list[int] = []
  mapped_v: list[int] = []
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
      mapped_u.append(iu)
      mapped_v.append(iv)

  s_u = np.array(mapped_u, dtype=np.int64)
  s_v = np.array(mapped_v, dtype=np.int64)
  num_s_edges = len(s_u)

  # Exact upper bound check:
  # |E_s| * T (spatial) + N * sum_l (T - l) (temporal) + 2 * |E_s| * sum_l (T - l) (cross)
  T = index.num_times
  N = index.num_nodes
  sigma_temp = sum(max(0, T - l) for l in range(1, max_lag + 1))
  est_edges = num_s_edges * T + N * sigma_temp + 2 * num_s_edges * sigma_temp
  if est_edges > max_edges:
    raise ValueError(
        f"Estimated space-time edges ({est_edges}) exceed max_edges="
        f"{max_edges} limit."
    )

  if index.n_items == 0:
    return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)

  # Build lookup for item indices: dense 2D table when N * T <= 50M cells,
  # or searchsorted on 1D key when N * T > 50M cells.
  node_inv = np.searchsorted(index.unique_nodes, index.node_ids).astype(
      np.int64
  )
  time_inv = np.searchsorted(index.unique_times, index.times).astype(np.int64)

  if N * T <= 50_000_000:
    dense_table = np.full((N, T), -1, dtype=np.int64)
    dense_table[node_inv, time_inv] = np.arange(index.n_items, dtype=np.int64)
    item_of: np.ndarray | None = dense_table
    sorted_item_keys = np.zeros(0, dtype=np.int64)
    sorted_item_indices = np.zeros(0, dtype=np.int64)
  else:
    item_of = None
    item_keys = node_inv * T + time_inv
    key_order = np.argsort(item_keys)
    sorted_item_keys = item_keys[key_order]
    sorted_item_indices = np.arange(index.n_items, dtype=np.int64)[key_order]

  def lookup_items(
      query_node_arr: np.ndarray, query_time_arr: np.ndarray
  ) -> np.ndarray:
    if item_of is not None:
      return item_of[query_node_arr, query_time_arr]
    q_keys = query_node_arr * T + query_time_arr
    idx = np.searchsorted(sorted_item_keys, q_keys)
    valid = (idx < len(sorted_item_keys)) & (
        sorted_item_keys[np.minimum(idx, len(sorted_item_keys) - 1)] == q_keys
    )
    res = np.full(len(q_keys), -1, dtype=np.int64)
    res[valid] = sorted_item_indices[idx[valid]]
    return res

  edge_list_u: list[np.ndarray] = []
  edge_list_v: list[np.ndarray] = []
  weight_list: list[np.ndarray] = []

  # 1. Same-time spatial edges
  if num_s_edges > 0:
    if item_of is not None:
      sp_u = item_of[s_u, :].ravel()
      sp_v = item_of[s_v, :].ravel()
    else:
      q_node_u = np.repeat(s_u, T)
      q_time = np.tile(np.arange(T, dtype=np.int64), len(s_u))
      q_node_v = np.repeat(s_v, T)
      sp_u = lookup_items(q_node_u, q_time)
      sp_v = lookup_items(q_node_v, q_time)

    sp_valid = (sp_u >= 0) & (sp_v >= 0)
    if np.any(sp_valid):
      edge_list_u.append(sp_u[sp_valid])
      edge_list_v.append(sp_v[sp_valid])
      weight_list.append(
          np.full(np.sum(sp_valid), phi_spatial, dtype=np.float64)
      )

  # 2. Same-node temporal edges and 3. Cross-lagged edges
  for l in range(1, max_lag + 1):
    w_temp = float(phi_temp_arr[l - 1])
    w_cross = float(phi_cross_arr[l - 1])

    target_times = index.unique_times + l
    cand_t2 = np.searchsorted(index.unique_times, target_times)
    valid_t = (cand_t2 < T) & (
        index.unique_times[np.minimum(cand_t2, T - 1)] == target_times
    )
    t1_arr = np.where(valid_t)[0]
    t2_arr = cand_t2[valid_t]

    if len(t1_arr) == 0:
      continue

    # Same-node temporal
    if item_of is not None:
      temp_u = item_of[:, t1_arr].ravel()
      temp_v = item_of[:, t2_arr].ravel()
    else:
      q_node = np.repeat(np.arange(N, dtype=np.int64), len(t1_arr))
      q_t1 = np.tile(t1_arr, N)
      q_t2 = np.tile(t2_arr, N)
      temp_u = lookup_items(q_node, q_t1)
      temp_v = lookup_items(q_node, q_t2)

    temp_valid = (temp_u >= 0) & (temp_v >= 0)
    if np.any(temp_valid):
      edge_list_u.append(temp_u[temp_valid])
      edge_list_v.append(temp_v[temp_valid])
      weight_list.append(np.full(np.sum(temp_valid), w_temp, dtype=np.float64))

    # Cross-lagged edges
    if num_s_edges > 0:
      if item_of is not None:
        cr_u_A = item_of[s_u[:, None], t1_arr[None, :]].ravel()
        cr_v_A = item_of[s_v[:, None], t2_arr[None, :]].ravel()
        cr_u_B = item_of[s_v[:, None], t1_arr[None, :]].ravel()
        cr_v_B = item_of[s_u[:, None], t2_arr[None, :]].ravel()
      else:
        q_cr_u = np.repeat(s_u, len(t1_arr))
        q_cr_v = np.repeat(s_v, len(t1_arr))
        q_cr_t1 = np.tile(t1_arr, len(s_u))
        q_cr_t2 = np.tile(t2_arr, len(s_u))
        cr_u_A = lookup_items(q_cr_u, q_cr_t1)
        cr_v_A = lookup_items(q_cr_v, q_cr_t2)
        cr_u_B = lookup_items(q_cr_v, q_cr_t1)
        cr_v_B = lookup_items(q_cr_u, q_cr_t2)

      # Direction A: (u, t1) with (v, t2)
      cr_valid_A = (cr_u_A >= 0) & (cr_v_A >= 0)
      if np.any(cr_valid_A):
        edge_list_u.append(cr_u_A[cr_valid_A])
        edge_list_v.append(cr_v_A[cr_valid_A])
        weight_list.append(
            np.full(np.sum(cr_valid_A), w_cross, dtype=np.float64)
        )

      # Direction B: (v, t1) with (u, t2)
      cr_valid_B = (cr_u_B >= 0) & (cr_v_B >= 0)
      if np.any(cr_valid_B):
        edge_list_u.append(cr_u_B[cr_valid_B])
        edge_list_v.append(cr_v_B[cr_valid_B])
        weight_list.append(
            np.full(np.sum(cr_valid_B), w_cross, dtype=np.float64)
        )

  if not edge_list_u:
    return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)

  raw_u = np.concatenate(edge_list_u)
  raw_v = np.concatenate(edge_list_v)
  raw_w = np.concatenate(weight_list)

  not_self = raw_u != raw_v
  raw_u = raw_u[not_self]
  raw_v = raw_v[not_self]
  raw_w = raw_w[not_self]

  if len(raw_u) == 0:
    return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)

  canon_u = np.minimum(raw_u, raw_v)
  canon_v = np.maximum(raw_u, raw_v)

  n_items = index.n_items
  pair_key = canon_u * n_items + canon_v

  order = np.lexsort((raw_w, pair_key))
  sorted_key = pair_key[order]
  sorted_w = raw_w[order]
  sorted_u = canon_u[order]
  sorted_v = canon_v[order]

  _, rev_first_idx = np.unique(sorted_key[::-1], return_index=True)
  last_idx = len(sorted_key) - 1 - rev_first_idx
  last_idx = np.sort(last_idx)

  final_edges = np.column_stack([sorted_u[last_idx], sorted_v[last_idx]])
  final_weights = sorted_w[last_idx]

  if len(final_edges) > max_edges:
    raise ValueError(
        f"Number of space-time edges exceeded max_edges={max_edges} limit."
    )

  return final_edges, final_weights


def candidate_partitions(
    index: SpaceTimeIndex,
    spatial_edges: np.ndarray,
    *,
    target_sizes: Sequence[int],
    window_lengths: Sequence[int],
    seed: int = 0,
) -> list[Candidate]:
  """Constructs candidate partitions over community, window, and product designs.

  Note on design exogeneity (THEORY_SPACETIME §9.5):
    The candidate grid (target_sizes, window_lengths) and seed MUST be fixed
    before looking at evaluated losses, residuals, or labels. Tuning the grid
    based on observed losses invalidates Lemma R.

  Candidates produced:
    1. Community-only (G_t = 1): partitioners.partition_graph on spatial graph,
       broadcast over time.
    2. Window-only (G_s = 1): contiguous time steps in windows of length wl.
    3. Product partitions: community x window blocks for each (ts, wl) pair.

  Args:
    index: SpaceTimeIndex of evaluated items.
    spatial_edges: Spatial edges (E_s, 2).
    target_sizes: Sequence of target spatial community sizes.
    window_lengths: Sequence of temporal window lengths.
    seed: PRNG seed for deterministic spatial graph partitioning.

  Returns:
    List of Candidate partitions.

  Raises:
    ValueError: If target_sizes or window_lengths are empty or non-positive.
  """
  if not target_sizes:
    raise ValueError("target_sizes must be non-empty.")
  if not window_lengths:
    raise ValueError("window_lengths must be non-empty.")
  for ts in target_sizes:
    if ts < 1:
      raise ValueError(f"target_size must be >= 1, got {ts}.")
  for wl in window_lengths:
    if wl < 1:
      raise ValueError(f"window_length must be >= 1, got {wl}.")

  node_to_idx = {u: i for i, u in enumerate(index.unique_nodes)}
  num_nodes = index.num_nodes

  # Map spatial edges to 0..num_nodes-1 indices
  s_edges = np.asarray(spatial_edges)
  s_edges_mapped = []
  if s_edges.ndim == 2 and s_edges.shape[1] == 2 and s_edges.shape[0] > 0:
    for u, v in s_edges:
      idx_u = node_to_idx.get(u)
      idx_v = node_to_idx.get(v)
      if idx_u is None and isinstance(u, (int, np.integer)):
        if 0 <= int(u) < num_nodes:
          idx_u = int(u)
      if idx_v is None and isinstance(v, (int, np.integer)):
        if 0 <= int(v) < num_nodes:
          idx_v = int(v)
      if idx_u is not None and idx_v is not None and idx_u != idx_v:
        s_edges_mapped.append((idx_u, idx_v))

  if s_edges_mapped:
    spatial_edges_idx = np.array(s_edges_mapped, dtype=np.int64)
  else:
    spatial_edges_idx = np.zeros((0, 2), dtype=np.int64)

  # Precompute spatial partitions for each target size
  spatial_partitions: dict[int, np.ndarray] = {}
  for ts in target_sizes:
    sp_c = _partitioners.partition_graph(
        num_nodes, spatial_edges_idx, target_cluster_size=ts, seed=seed
    )
    spatial_partitions[ts] = sp_c

  item_node_indices = np.searchsorted(
      index.unique_nodes, index.node_ids
  ).astype(np.int64)
  item_time_indices = np.searchsorted(index.unique_times, index.times).astype(
      np.int64
  )

  candidates: list[Candidate] = []
  n_items = index.n_items

  def make_candidate(
      name: str,
      dense_cluster_ids: np.ndarray,
      g_s: int,
      g_t: int,
      community_ids: np.ndarray,
      window_ids: np.ndarray,
  ) -> Candidate:
    g = int(np.max(dense_cluster_ids)) + 1 if len(dense_cluster_ids) > 0 else 0
    counts = np.bincount(dense_cluster_ids, minlength=g)
    n_max = int(np.max(counts)) if g > 0 else 0
    m_bar = float(n_items) / float(g) if g > 0 else 0.0
    return Candidate(
        name=name,
        cluster_ids=dense_cluster_ids.astype(np.int64),
        G=g,
        G_s=g_s,
        G_t=g_t,
        n_max=n_max,
        m_bar=m_bar,
        community_ids=community_ids.astype(np.int64),
        window_ids=window_ids.astype(np.int64),
    )

  # Precompute dense community IDs for observed items
  comm_per_ts: dict[int, tuple[np.ndarray, int]] = {}
  for ts in target_sizes:
    sp_c = spatial_partitions[ts]
    comm_raw = sp_c[item_node_indices]
    _, comm_dense = np.unique(comm_raw, return_inverse=True)
    comm_per_ts[ts] = (comm_dense.astype(np.int64), len(_))

  # Precompute dense window IDs for observed items
  win_per_wl: dict[int, tuple[np.ndarray, int]] = {}
  for wl in window_lengths:
    win_raw = item_time_indices // wl
    _, win_dense = np.unique(win_raw, return_inverse=True)
    win_per_wl[wl] = (win_dense.astype(np.int64), len(_))

  # 1. Community-only partitions (G_t = 1)
  for ts in target_sizes:
    c_ids, g_s_count = comm_per_ts[ts]
    w_ids = np.zeros(n_items, dtype=np.int64)
    c_cand = make_candidate(
        f"community_ts{ts}",
        c_ids,
        g_s=g_s_count,
        g_t=1,
        community_ids=c_ids,
        window_ids=w_ids,
    )
    candidates.append(c_cand)

  # 2. Window-only partitions (G_s = 1)
  for wl in window_lengths:
    w_ids, g_t_count = win_per_wl[wl]
    c_ids = np.zeros(n_items, dtype=np.int64)
    c_cand = make_candidate(
        f"window_wl{wl}",
        w_ids,
        g_s=1,
        g_t=g_t_count,
        community_ids=c_ids,
        window_ids=w_ids,
    )
    candidates.append(c_cand)

  # 3. Product partitions (community x window)
  for ts in target_sizes:
    c_ids, g_s_count = comm_per_ts[ts]
    for wl in window_lengths:
      w_ids, g_t_count = win_per_wl[wl]
      raw_prod = c_ids * g_t_count + w_ids
      _, dense_prod = np.unique(raw_prod, return_inverse=True)
      c_cand = make_candidate(
          f"product_ts{ts}_wl{wl}",
          dense_prod.astype(np.int64),
          g_s=g_s_count,
          g_t=g_t_count,
          community_ids=c_ids,
          window_ids=w_ids,
      )
      candidates.append(c_cand)

  return candidates


def separable_kappa(
    kappa_s: float,
    kappa_t: float,
) -> Callable[[Candidate], float]:
  """Constructs a candidate-scoring kappa function under Claim S1 separability.

  On complete product grids, kappa factors exactly as:
    - kappa_s for community-only partitions (G_t = 1),
    - kappa_t for window-only partitions (G_s = 1),
    - kappa_s * kappa_t for product partitions (Claim S1).

  On incomplete grids, this function must not be used alone (Proposition S1B);
  choose_partition requires a fallback_kappa_fn (Claim S2B).

  Args:
    kappa_s: Declared spatial variance inflation factor (>= 1.0).
    kappa_t: Declared temporal variance inflation factor (>= 1.0).

  Returns:
    Callable taking a Candidate and returning its declared kappa.
  """
  if not (math.isfinite(kappa_s) and kappa_s >= 1.0):
    raise ValueError(f"kappa_s must be finite and >= 1.0, got {kappa_s}.")
  if not (math.isfinite(kappa_t) and kappa_t >= 1.0):
    raise ValueError(f"kappa_t must be finite and >= 1.0, got {kappa_t}.")

  k_s = float(kappa_s)
  k_t = float(kappa_t)

  def _kappa_fn(c: Candidate) -> float:
    if c.G_t == 1:
      return k_s
    elif c.G_s == 1:
      return k_t
    else:
      return k_s * k_t

  setattr(_kappa_fn, "is_separable", True)
  setattr(_kappa_fn, "kappa_s", k_s)
  setattr(_kappa_fn, "kappa_t", k_t)
  return _kappa_fn


def topological_kappa_fn(
    num_items: int,
    edges: np.ndarray,
    phi: float | Sequence[float] | np.ndarray,
) -> Callable[[Candidate], float]:
  """Constructs a candidate-scoring kappa function via graph.topological_kappa.

  Valid for any observation pattern (complete or incomplete grid) by
  Perron-Frobenius monotonicity (Claim S2B(4)).

  Args:
    num_items: Total number of observed items n.
    edges: Observed-item edge endpoints (E, 2).
    phi: Per-edge or scalar correlation upper bound.

  Returns:
    Callable taking a Candidate and returning its topological kappa.
  """

  def _kappa_fn(c: Candidate) -> float:
    tk = _graph.topological_kappa(num_items, edges, c.cluster_ids, phi=phi)
    return float(tk.kappa)

  setattr(_kappa_fn, "is_separable", False)
  return _kappa_fn


def choose_partition(
    candidates: Sequence[Candidate],
    *,
    kappa_fn: Callable[[Candidate], float],
    m_item: float,
    metric: str = "mse",
    level: float = 0.95,
    is_complete_grid: bool,
    fallback_kappa_fn: Callable[[Candidate], float] | None = None,
    edges: np.ndarray | None = None,
) -> Choice:
  """Selects the partition maximizing size-adjusted effective sample size N_eff.

  Proposition S5: Partition choice is a deterministic function of the design and
  declarations, never inspecting evaluation losses or residuals.

  Scoring:
    N_eff = G / (kappa * r_eff)
    where r_eff = (M_c_default - 1) / (m_item - 1) for m_item > 1
    (or n_max / m_bar for m_item == 1). M_c_default is computed from cluster
    counts via Lemma I'.

  Tie-breaking:
    1. Higher N_eff (rounded to 12 decimal places).
    2. Prefer G_t >= 2 (so drift and marginal checks can run).
    3. Fewer clusters G.
    4. Candidate name.

  Args:
    candidates: Family of Candidate partitions.
    kappa_fn: Function mapping Candidate to certified kappa.
    m_item: Declared item-level relative variance bound (>= 1.0).
    metric: Evaluation metric name (for default M_c / issuance threshold).
    level: Confidence level in (0, 1).
    is_complete_grid: Whether the observed data form a complete grid.
    fallback_kappa_fn: Fallback kappa function required on incomplete grids when
      kappa_fn is separable.
    edges: Optional item-level edges to compute winner's cluster_edges.

  Returns:
    Choice containing the winner, candidate scores, and adjacency edges.

  Raises:
    ValueError: If candidates is empty, m_item < 1, or separable_kappa is used
      on an incomplete grid without a fallback.
  """
  if not candidates:
    raise ValueError("candidates sequence cannot be empty.")
  if not (math.isfinite(m_item) and m_item >= 1.0):
    raise ValueError(f"m_item must be finite and >= 1.0, got {m_item}.")
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")

  is_sep = bool(getattr(kappa_fn, "is_separable", False))
  is_range = bool(getattr(kappa_fn, "is_range", False))
  kappa_s_val: float | None = getattr(kappa_fn, "kappa_s", None)
  kappa_t_val: float | None = getattr(kappa_fn, "kappa_t", None)

  if is_range:
    effective_kappa_fn = kappa_fn
    route = "range"
    route_reason = (
        "Range envelope kappa (THEORY_RANGE_ENVELOPE.md, Claim RE1)."
    )
  elif not is_complete_grid:
    if is_sep:
      if fallback_kappa_fn is None:
        raise ValueError(
            "separable_kappa cannot be used alone on an incomplete grid"
            " (Proposition S1B); provide fallback_kappa_fn (e.g."
            " topological_kappa_fn via Claim S2B)."
        )
      effective_kappa_fn = fallback_kappa_fn
      route = "fallback_topological"
      route_reason = (
          "Incomplete grid: separable_kappa unproved; routed to fallback"
          " topological_kappa (Claim S2B)."
      )
    else:
      effective_kappa_fn = kappa_fn
      route = "topological"
      route_reason = (
          "Topological kappa on incomplete grid (Claim S2B / Lemma T')."
      )
  else:
    effective_kappa_fn = kappa_fn
    if is_sep:
      route = "separable"
      route_reason = (
          "Complete grid: separable kappa factorization (Claim S1)."
      )
    else:
      route = "topological"
      route_reason = "Topological kappa on complete grid (Claim S2B)."

  alpha = 1.0 - level
  alpha_prime = alpha / 2.0

  candidates_table: list[dict[str, Any]] = []
  scored_candidates: list[tuple[float, int, str, Candidate, float, float]] = []

  for c in candidates:
    k_val = float(effective_kappa_fn(c))
    if not (math.isfinite(k_val) and k_val >= 1.0):
      raise ValueError(f"kappa_fn returned invalid kappa={k_val} for {c.name}.")

    counts = np.bincount(c.cluster_ids, minlength=c.G)
    sum_c_sq = int(np.sum(counts.astype(np.int64) ** 2))
    n_tot = len(c.cluster_ids)

    m_c_def = _cluster_sketch.default_m_c_from_counts(
        m_item, n_tot, c.G, c.n_max, sum_c_sq
    )

    if m_item > 1.0:
      r_eff = float(m_c_def - 1.0) / float(m_item - 1.0)
    else:
      r_eff = float(c.n_max) / float(c.m_bar) if c.m_bar > 0 else 1.0

    n_eff = float(c.G) / (k_val * r_eff) if (k_val * r_eff) > 0 else 0.0

    # Issuance threshold check: c(Pi) < 1 <=> G / kappa > n_A
    n_A = (m_c_def - 1.0) * (1.0 - alpha_prime) / alpha_prime
    clears_issuance = (float(c.G) / k_val) > n_A

    candidates_table.append({
        "name": c.name,
        "G": c.G,
        "G_s": c.G_s,
        "G_t": c.G_t,
        "kappa": k_val,
        "r_eff": r_eff,
        "N_eff": n_eff,
        "M_c_default": m_c_def,
        "n_A": n_A,
        "clears_issuance": clears_issuance,
    })

    scored_candidates.append((n_eff, c.G, c.name, c, k_val, r_eff))

  # Deterministic tie-breaking:
  # 1. Higher N_eff (rounded to 12 decimal places)
  # 2. Prefer G_t >= 2
  # 3. Fewer clusters G
  # 4. Candidate name
  def sort_key(
      item: tuple[float, int, str, Candidate, float, float],
  ) -> tuple[float, int, int, str]:
    n_eff, g, name, c, _, _ = item
    prefer_gt2 = 0 if c.G_t >= 2 else 1
    return (-round(n_eff, 12), prefer_gt2, g, name)

  scored_candidates.sort(key=sort_key)
  winner_n_eff, _, _, winner, winner_kappa, winner_r_eff = scored_candidates[0]

  # Cluster adjacency for the winner
  if edges is not None and edges.shape[0] > 0:
    cluster_edges = _graph.cluster_adjacency(
        len(winner.cluster_ids), edges, winner.cluster_ids
    )
  else:
    cluster_edges = np.zeros((0, 2), dtype=np.int64)

  return Choice(
      winner=winner,
      winner_kappa=winner_kappa,
      winner_r_eff=winner_r_eff,
      winner_n_eff=winner_n_eff,
      candidates_table=candidates_table,
      cluster_edges=cluster_edges,
      route=route,
      route_reason=route_reason,
      kappa_s=kappa_s_val if route == "separable" else None,
      kappa_t=kappa_t_val if route == "separable" else None,
      m_item=float(m_item),
  )


def spacetime_interval(
    data: Any,
    index: SpaceTimeIndex,
    choice: Choice,
    *,
    metric: str = "mse",
    level: float = 0.95,
    m_item: float | None = None,
    kappa: float | None = None,
    **kwargs: Any,
) -> SpaceTimeResult:
  """Computes certified confidence interval for space-time graph evaluation.

  Args:
    data: Evaluated losses/residuals array, or tuple of (labels, predictions).
    index: SpaceTimeIndex mapping items to (node, time).
    choice: Choice object containing the selected partition and cluster edges.
    metric: Evaluation metric name (default 'mse').
    level: Nominal confidence level in (0, 1).
    m_item: Optional item-level bound override.
    kappa: Optional variance inflation factor override.
    **kwargs: Additional keyword arguments forwarded to graph.graph_interval.

  Returns:
    SpaceTimeResult containing interval, marginal checks, drift, and choice.
  """
  if "m" in kwargs:
    raise ValueError(
        "spacetime_interval does not accept 'm'; the cluster-level bound is"
        " derived from m_item via choice.m_item."
    )

  winner = choice.winner

  # 1. Data length must match index.n_items
  norm_metric = _metrics.normalize_metric(metric)
  if norm_metric == "r2":
    if not (isinstance(data, tuple) and len(data) == 2):
      raise ValueError(
          "For metric 'r2', raw data must be a tuple of (labels, predictions)."
      )
    y_true, y_pred = data
    n_true = len(np.asarray(y_true))
    n_pred = len(np.asarray(y_pred))
    if n_true != index.n_items:
      raise ValueError(
          f"Labels length ({n_true}) does not match index.n_items ({index.n_items})."
      )
    if n_pred != index.n_items:
      raise ValueError(
          f"Predictions length ({n_pred}) does not match index.n_items ({index.n_items})."
      )
  else:
    if isinstance(data, tuple):
      raise ValueError(
          f"Tuples are only accepted for metric='r2', got {metric!r}."
      )
    n_data = len(np.asarray(data))
    if n_data != index.n_items:
      raise ValueError(
          f"Data length ({n_data}) does not match index.n_items ({index.n_items})."
      )

  # 2. R2 requires explicit kappa_labels
  if norm_metric == "r2":
    if "kappa_labels" not in kwargs or kwargs["kappa_labels"] is None:
      raise ValueError(
          "For metric 'r2', kappa_labels must be explicitly passed to"
          " spacetime_interval: the space-time kappa bounds the dependence of"
          " the losses only; the label totals need their own declared VIF,"
          " passed as kappa_labels."
      )

  # 3. m_item consistency
  resolved_m_item = m_item
  if resolved_m_item is None and choice.m_item is not None:
    resolved_m_item = choice.m_item
  elif resolved_m_item is not None and choice.m_item is not None:
    if resolved_m_item != choice.m_item:
      raise ValueError(
          f"m_item passed to spacetime_interval ({resolved_m_item}) differs from"
          f" choice.m_item ({choice.m_item}); the interval must use the m_item"
          " passed to choose_partition."
      )

  # 4. kappa override cannot be smaller than winner_kappa
  if kappa is not None:
    if kappa < choice.winner_kappa:
      raise ValueError(
          f"kappa override ({kappa}) cannot be smaller than choice.winner_kappa"
          f" ({choice.winner_kappa}); overrides may only be more conservative."
      )
    resolved_kappa = kappa
  else:
    resolved_kappa = choice.winner_kappa

  # 5. Base graph_interval on the space-time block graph
  graph_res = _graph.graph_interval(
      data,
      cluster_ids=winner.cluster_ids,
      cluster_edges=choice.cluster_edges,
      metric=metric,
      level=level,
      m_item=resolved_m_item,
      kappa=resolved_kappa,
      **kwargs,
  )

  # 6. Extract item-level summands for marginal and drift checks
  if norm_metric == "r2":
    y_true, y_pred = data
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    s_items = (y_true_arr - y_pred_arr) ** 2
  else:
    s_items = _metrics.extract_summands(data, norm_metric)

  n_items = len(s_items)
  grand_mean = float(np.mean(s_items))

  # 3. Marginal checks under separability (Claim S1(3))
  # Run only for product partitions (G_s > 1, G_t > 1) on the separable route
  marginal_checks: MarginalChecks | None = None
  if (
      choice.route == "separable"
      and winner.G_s > 1
      and winner.G_t > 1
      and choice.kappa_s is not None
      and choice.kappa_t is not None
  ):
    # Temporal window totals with grand-mean centring tested against kappa_t
    s_w = np.bincount(
        winner.window_ids, weights=s_items, minlength=winner.G_t
    ).astype(np.float64)
    n_w = np.bincount(winner.window_ids, minlength=winner.G_t).astype(
        np.float64
    )
    e_w = s_w - n_w * grand_mean

    if choice.kappa_t == 1.0:
      t_check = _refutation.check_uncorrelated(e_w, level=_REFUTATION_LEVEL)
    else:
      t_check = _refutation.check_kappa(
          e_w, kappa_declared=choice.kappa_t, level=_REFUTATION_LEVEL
      )

    # Spatial marginal totals with per-window centring tested against kappa_s:
    # e_{a,w} = S_{a,w} - n_{a,w} * (S_{•,w} / n_{•,w}), then sum over w
    window_means = np.where(n_w > 0, s_w / n_w, 0.0)
    e_items_pw = s_items - window_means[winner.window_ids]
    e_comm = np.bincount(
        winner.community_ids, weights=e_items_pw, minlength=winner.G_s
    ).astype(np.float64)
    n_comm = np.bincount(winner.community_ids, minlength=winner.G_s).astype(
        np.float64
    )

    # Community adjacency edges mapped via winner.community_ids
    cluster_to_comm = np.zeros(winner.G, dtype=np.int64)
    cluster_to_comm[winner.cluster_ids] = winner.community_ids
    if choice.cluster_edges.shape[0] > 0:
      c_u = cluster_to_comm[choice.cluster_edges[:, 0]]
      c_v = cluster_to_comm[choice.cluster_edges[:, 1]]
      cross_mask = c_u != c_v
      if np.any(cross_mask):
        raw_comm_edges = np.column_stack([c_u[cross_mask], c_v[cross_mask]])
        comm_min = np.minimum(raw_comm_edges[:, 0], raw_comm_edges[:, 1])
        comm_max = np.maximum(raw_comm_edges[:, 0], raw_comm_edges[:, 1])
        comm_edges = np.unique(
            np.column_stack([comm_min, comm_max]), axis=0
        ).astype(np.int64)
      else:
        comm_edges = np.zeros((0, 2), dtype=np.int64)
    else:
      comm_edges = np.zeros((0, 2), dtype=np.int64)

    if choice.kappa_s == 1.0:
      s_check = _refutation.check_uncorrelated_edges(
          e_comm, n_comm.astype(np.int64), comm_edges, level=_REFUTATION_LEVEL
      )
    else:
      g_s = winner.G_s
      if len(comm_edges) > 0 and g_s >= 20:
        target_size = max(2, int(round(float(g_s) / float(30))))
        batch_ids = _partitioners.partition_graph(
            g_s, comm_edges, target_cluster_size=target_size, seed=0
        )
        s_check = _refutation.check_kappa_batches(
            e_comm,
            n_comm.astype(np.int64),
            batch_ids,
            choice.kappa_s,
            level=_REFUTATION_LEVEL,
        )
      else:
        s_check = _refutation.RefutationResult(
            _refutation.Refutation.UNTESTABLE,
            None,
            None,
            reason="fewer than 10 super-batches or no community edges",
        )

    marginal_checks = MarginalChecks(temporal=t_check, spatial=s_check)

  # 4. Drift diagnostic (Proposition D')
  untestable_ref = _refutation.RefutationResult(
      _refutation.Refutation.UNTESTABLE, None, None
  )
  if winner.G_t == 1:
    drift = _temporal.DriftResult(
        outcome=_temporal.Drift.UNTESTABLE,
        first_half=(float("nan"), float("nan")),
        second_half=(float("nan"), float("nan")),
        first_kappa_check=untestable_ref,
        second_kappa_check=untestable_ref,
        message=(
            "drift check untestable (community-only partition G_t=1 spans all"
            " time)."
        ),
    )
  elif winner.G_t < 4:
    drift = _temporal.DriftResult(
        outcome=_temporal.Drift.UNTESTABLE,
        first_half=(float("nan"), float("nan")),
        second_half=(float("nan"), float("nan")),
        first_kappa_check=untestable_ref,
        second_kappa_check=untestable_ref,
        message="drift check untestable (fewer than 4 windows).",
    )
  else:
    # G_t >= 4: Split by window index at G_t // 2
    w_split = winner.G_t // 2
    early_mask = winner.window_ids < w_split
    late_mask = ~early_mask

    if np.sum(early_mask) == 0 or np.sum(late_mask) == 0:
      drift = _temporal.DriftResult(
          outcome=_temporal.Drift.UNTESTABLE,
          first_half=(float("nan"), float("nan")),
          second_half=(float("nan"), float("nan")),
          first_kappa_check=untestable_ref,
          second_kappa_check=untestable_ref,
          message="drift check untestable (one half has 0 items).",
      )
    else:
      early_clusters = winner.cluster_ids[early_mask]
      late_clusters = winner.cluster_ids[late_mask]

      # Re-index cluster IDs densely for each half
      _, early_dense = np.unique(early_clusters, return_inverse=True)
      g1 = len(_)
      _, late_dense = np.unique(late_clusters, return_inverse=True)
      g2 = len(_)

      s_early = s_items[early_mask]
      s_late = s_items[late_mask]

      t1 = np.bincount(early_dense, weights=s_early, minlength=g1).astype(
          np.float64
      )
      c1 = np.bincount(early_dense, minlength=g1).astype(np.int64)
      t2 = np.bincount(late_dense, weights=s_late, minlength=g2).astype(
          np.float64
      )
      c2 = np.bincount(late_dense, minlength=g2).astype(np.int64)

      shard_1 = _cluster_sketch.PartialClusterShard(
          cluster_ids=np.arange(g1, dtype=np.int64),
          cluster_totals=t1,
          cluster_counts=c1,
      )
      shard_2 = _cluster_sketch.PartialClusterShard(
          cluster_ids=np.arange(g2, dtype=np.int64),
          cluster_totals=t2,
          cluster_counts=c2,
      )

      alpha_half = (1.0 - level) / 2.0
      half_level = 1.0 - alpha_half

      sk_1 = shard_1.to_cluster_sketch(kappa_cluster=resolved_kappa)
      sk_2 = shard_2.to_cluster_sketch(kappa_cluster=resolved_kappa)

      metric_drift = "mse" if norm_metric == "r2" else metric
      res_1 = _cluster_bound.cluster_interval(
          sk_1,
          metric=metric_drift,
          level=half_level,
          m_item=resolved_m_item,
          kappa_cluster=resolved_kappa,
      )
      res_2 = _cluster_bound.cluster_interval(
          sk_2,
          metric=metric_drift,
          level=half_level,
          m_item=resolved_m_item,
          kappa_cluster=resolved_kappa,
      )

      has_int_1 = (
          res_1.status is _interval.Status.UNREFUTED
          and math.isfinite(res_1.low)
          and math.isfinite(res_1.high)
      )
      has_int_2 = (
          res_2.status is _interval.Status.UNREFUTED
          and math.isfinite(res_2.low)
          and math.isfinite(res_2.high)
      )

      if not (has_int_1 and has_int_2):
        drift_outcome = _temporal.Drift.UNTESTABLE
        drift_msg = (
            "drift check untestable (at least one half has no valid interval)."
        )
      else:
        disjoint = (res_1.low > res_2.high) or (res_2.low > res_1.high)
        if disjoint:
          drift_outcome = _temporal.Drift.WARNING
          drift_msg = (
              "drift warning: first half interval "
              f"[{res_1.low:.4g}, {res_1.high:.4g}] and second half interval "
              f"[{res_2.low:.4g}, {res_2.high:.4g}] are disjoint."
          )
        else:
          drift_outcome = _temporal.Drift.NO_WARNING
          drift_msg = (
              "no drift: first half interval "
              f"[{res_1.low:.4g}, {res_1.high:.4g}] and second half interval "
              f"[{res_2.low:.4g}, {res_2.high:.4g}] overlap."
          )

      drift = _temporal.DriftResult(
          outcome=drift_outcome,
          first_half=(res_1.low, res_1.high),
          second_half=(res_2.low, res_2.high),
          first_kappa_check=untestable_ref,
          second_kappa_check=untestable_ref,
          message=drift_msg,
      )

  return SpaceTimeResult(
      interval=graph_res.interval,
      kappa_declared=graph_res.kappa_declared,
      kappa_check=graph_res.kappa_check,
      kappa_w_hat=graph_res.kappa_w_hat,
      num_cluster_edges=graph_res.num_cluster_edges,
      num_dropped_edges=graph_res.num_dropped_edges,
      hub_ratio=graph_res.hub_ratio,
      edge_n_eff=graph_res.edge_n_eff,
      num_batches=graph_res.num_batches,
      batch_cut_fraction=graph_res.batch_cut_fraction,
      message=graph_res.message,
      r2_detail=graph_res.r2_detail,
      marginal_checks=marginal_checks,
      drift=drift,
      choice=choice,
      graph_result=graph_res,
  )
