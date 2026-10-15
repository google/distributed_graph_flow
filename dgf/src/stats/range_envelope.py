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

"""Range-envelope dependence declarations and variance inflation bounds.

Provides user-facing range declarations (RangeDeclaration) and certified
variance-inflation factor bounds (RangeKappa, range_kappa, range_kappa_fn)
derived from model architecture (K hops, lookback L, horizon H) and data
dependence ranges (spatial R_s, temporal R_t) with an out-of-range tail budget
gamma.

Theory:
  THEORY_RANGE_ENVELOPE.md:
  - Lemma RE1: Loss-level range bounding box (valid, possibly conservative) bar_R_s = 2*K + R_s and bar_R_t = L + H + R_t.
  - Claim RE1: Certified bound kappa <= 1 + lambda_max(A_cut) + gamma.
  - Proposition RE3: Monotonicity in declarations and stability under missing data.
"""

from collections.abc import Callable, Sequence
import dataclasses
import math
from typing import Any

import numpy as np

from dgf.src.stats import graph as _graph
from dgf.src.stats import spacetime as _spacetime

Candidate = _spacetime.Candidate
SpaceTimeIndex = _spacetime.SpaceTimeIndex


@dataclasses.dataclass(frozen=True)
class RangeDeclaration:
  """User-facing specification of model receptive field and data dependence range.

  Attributes:
    k_hops: Spatial message-passing hops of the model (K >= 0).
    lookback: Temporal lookback steps of the model (L >= 0).
    horizon: Prediction horizon steps (H >= 0).
    data_range_s: Spatial hops beyond which data noise is independent (R_s >= 0).
    data_range_t: Temporal steps beyond which data noise is independent (R_t >= 0).
    gamma: Out-of-range correlation budget (gamma >= 0.0, required, no default).
  """

  k_hops: int
  lookback: int
  horizon: int
  data_range_s: int
  data_range_t: int
  gamma: float

  def __post_init__(self) -> None:
    for name in ("k_hops", "lookback", "horizon", "data_range_s", "data_range_t"):
      val = getattr(self, name)
      if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer, got {type(val).__name__} ({val!r})."
        )
      if val < 0:
        raise ValueError(f"{name} must be >= 0, got {val}.")

    if isinstance(self.gamma, bool) or not isinstance(
        self.gamma, (int, float, np.floating, np.integer)
    ):
      raise ValueError(
          f"gamma must be a float, got {type(self.gamma).__name__}"
          f" ({self.gamma!r})."
      )
    gamma_f = float(self.gamma)
    if not (math.isfinite(gamma_f) and gamma_f >= 0.0):
      raise ValueError(f"gamma must be finite and >= 0.0, got {gamma_f}.")

  @property
  def spatial_range(self) -> int:
    """Loss-level spatial dependence range bounding box (valid, possibly conservative): 2*K + R_s hops."""
    return 2 * self.k_hops + self.data_range_s

  @property
  def temporal_range(self) -> int:
    """Loss-level temporal dependence range bounding box (valid, possibly conservative): L + H + R_t steps."""
    return self.lookback + self.horizon + self.data_range_t


@dataclasses.dataclass(frozen=True)
class RangeKappa:
  """Certified variance-inflation factor bound under range envelope declaration.

  Attributes:
    kappa: Certified upper bound on 1 + lambda_max(A_cut) + gamma.
    cut_bound: Certified upper bound on lambda_max(A_cut).
    gamma: Declared out-of-range tail budget.
    method: Calculation method ('direct' or 'kronecker').
    max_cut_row_sum: Maximum row sum of the cut adjacency matrix.
    num_items: Total number of items evaluated.
  """

  kappa: float
  cut_bound: float
  gamma: float
  method: str
  max_cut_row_sum: float
  num_items: int


def _compute_spatial_ball_matrix(
    num_nodes: int,
    spatial_edges: Any,
    unique_nodes: np.ndarray,
    spatial_range: int,
) -> np.ndarray:
  """Computes the 0/1 spatial ball matrix S via full-graph BFS up to spatial_range hops.

  Builds the adjacency graph over all node IDs in spatial_edges and unique_nodes
  (allowing unobserved nodes as valid path intermediates), evaluates BFS up to
  spatial_range hops starting from each observed node, and restricts the resulting
  ball matrix to unique_nodes (shape num_nodes x num_nodes).

  Raises:
    ValueError: If spatial_edges shape is not (E, 2), if dtype kinds mismatch,
      or if spatial_edges is non-empty and has no endpoints in unique_nodes.
  """
  s_edges = np.asarray(spatial_edges)
  if s_edges.ndim != 2 or s_edges.shape[1] != 2:
    if s_edges.size == 0 and (s_edges.ndim == 1 or s_edges.ndim == 2):
      s_edges = np.zeros((0, 2), dtype=unique_nodes.dtype)
    else:
      raise ValueError(
          f"spatial_edges must have shape (E, 2), got {s_edges.shape}."
      )

  def _int_kind(k: str) -> str:
    return "i" if k in ("i", "u") else k

  if s_edges.shape[0] > 0 and len(unique_nodes) > 0:
    if _int_kind(s_edges.dtype.kind) != _int_kind(unique_nodes.dtype.kind):
      raise ValueError(
          f"spatial_edges dtype kind ({s_edges.dtype.kind}) does not match"
          f" index.unique_nodes dtype kind ({unique_nodes.dtype.kind})."
      )

  node_to_obs: dict[Any, int] = {}
  adj: dict[Any, list[Any]] = {}

  for i, u in enumerate(unique_nodes):
    u_key = u.item() if hasattr(u, "item") else u
    node_to_obs[u_key] = i
    adj[u_key] = []

  has_match = False
  if s_edges.shape[0] > 0:
    for u, v in s_edges:
      u_key = u.item() if hasattr(u, "item") else u
      v_key = v.item() if hasattr(v, "item") else v
      if u_key not in adj:
        adj[u_key] = []
      if v_key not in adj:
        adj[v_key] = []
      if u_key != v_key:
        adj[u_key].append(v_key)
        adj[v_key].append(u_key)
      if (u_key in node_to_obs) or (v_key in node_to_obs):
        has_match = True

    if not has_match:
      raise ValueError(
          "spatial_edges node ids do not match node_ids; check types (int vs str)"
      )

  s_ball = np.zeros((num_nodes, num_nodes), dtype=np.float64)
  for i, u in enumerate(unique_nodes):
    u_key = u.item() if hasattr(u, "item") else u
    s_ball[i, i] = 1.0
    if spatial_range > 0 and u_key in adj and adj[u_key]:
      visited = {u_key}
      curr_level = [u_key]
      for _ in range(spatial_range):
        next_level = []
        for curr in curr_level:
          for nbr in adj[curr]:
            if nbr not in visited:
              visited.add(nbr)
              next_level.append(nbr)
              obs_j = node_to_obs.get(nbr)
              if obs_j is not None:
                s_ball[i, obs_j] = 1.0
        if not next_level:
          break
        curr_level = next_level

  return s_ball


def _extract_product_partition(
    index: SpaceTimeIndex,
    cluster_ids: np.ndarray,
    node_indices: np.ndarray,
    time_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Verifies and extracts product partition components C_a x W_w from cluster_ids.

  A space-time partition is a product partition if and only if each cluster can
  be factored as a Cartesian product of a spatial community C_a and a contiguous
  temporal window W_w. This function verifies:
  1. No cluster spans multiple spatial communities.
  2. No cluster spans multiple temporal windows.
  3. No two distinct clusters share the same (community, window) pair.
  4. Each temporal window is a contiguous interval of discrete time steps.

  Args:
    index: SpaceTimeIndex of evaluated items.
    cluster_ids: 1-D array of cluster IDs of length n_items.
    node_indices: 1-D array of item node indices in 0..num_nodes-1.
    time_indices: 1-D array of item time indices in 0..num_times-1.

  Returns:
    (node_comm, time_win, win_left, win_right)
    where win_left and win_right are 1-D arrays of shape (num_times,) giving the
    inclusive index bounds of each time step's window.

  Raises:
    ValueError: If cluster_ids is not a product partition.
  """
  n = len(cluster_ids)
  num_nodes = index.num_nodes
  num_times = index.num_times

  if n == 0:
    return (
        np.zeros(num_nodes, dtype=np.int64),
        np.zeros(num_times, dtype=np.int64),
        np.zeros(num_times, dtype=np.int64),
        np.zeros(num_times, dtype=np.int64),
    )

  unique_clusters, cluster_inv = np.unique(cluster_ids, return_inverse=True)
  g_clusters = len(unique_clusters)

  # 1. Disjoint Set Union (DSU) to group nodes appearing in the same cluster
  node_parent = list(range(num_nodes))

  def find_node(x: int) -> int:
    path = []
    while node_parent[x] != x:
      path.append(x)
      x = node_parent[x]
    for p in path:
      node_parent[p] = x
    return x

  def union_node(x: int, y: int) -> None:
    rx, ry = find_node(x), find_node(y)
    if rx != ry:
      node_parent[rx] = ry

  # 2. DSU to group times appearing in the same cluster
  time_parent = list(range(num_times))

  def find_time(x: int) -> int:
    path = []
    while time_parent[x] != x:
      path.append(x)
      x = time_parent[x]
    for p in path:
      time_parent[p] = x
    return x

  def union_time(x: int, y: int) -> None:
    rx, ry = find_time(x), find_time(y)
    if rx != ry:
      time_parent[rx] = ry

  for c in range(g_clusters):
    c_mask = cluster_inv == c
    nodes_c = np.unique(node_indices[c_mask])
    if len(nodes_c) > 1:
      base_node = int(nodes_c[0])
      for other_node in nodes_c[1:]:
        union_node(base_node, int(other_node))

    times_c = np.unique(time_indices[c_mask])
    if len(times_c) > 1:
      base_time = int(times_c[0])
      for other_time in times_c[1:]:
        union_time(base_time, int(other_time))

  # Form dense community and window indices
  node_roots = [find_node(i) for i in range(num_nodes)]
  _, node_comm = np.unique(node_roots, return_inverse=True)

  time_roots = [find_time(t) for t in range(num_times)]
  _, time_win = np.unique(time_roots, return_inverse=True)

  # 3. Verify product consistency: each observed item's (node_comm, time_win)
  # must uniquely determine cluster_ids, with no two clusters sharing (comm, win).
  item_comms = node_comm[node_indices]
  item_wins = time_win[time_indices]
  num_w = int(np.max(time_win)) + 1
  pair_keys = item_comms * int(num_w) + item_wins
  unique_pairs, pair_inv = np.unique(pair_keys, return_inverse=True)

  if len(unique_pairs) != g_clusters:
    raise ValueError(
        "cluster_ids does not represent a valid product partition: multiple"
        f" clusters share the same (community, window) pair (got {g_clusters}"
        f" clusters but {len(unique_pairs)} unique (community, window) pairs)."
        " method='kronecker' requires a product partition; use method='direct'"
        " for non-product partitions."
    )

  # Check that each (comm, win) pair maps bijectively to cluster_inv
  first_c = np.full(len(unique_pairs), -1, dtype=np.int64)
  for u in range(n):
    p_idx = pair_inv[u]
    c_val = cluster_inv[u]
    if first_c[p_idx] == -1:
      first_c[p_idx] = c_val
    elif first_c[p_idx] != c_val:
      raise ValueError(
          "cluster_ids does not represent a valid product partition: items in"
          f" community {item_comms[u]} and window {item_wins[u]} have"
          f" inconsistent clusters {first_c[p_idx]} and {c_val}."
          " method='kronecker' requires a product partition; use"
          " method='direct' for non-product partitions."
      )

  # 4. Verify temporal contiguity of windows
  win_left = np.zeros(num_times, dtype=np.int64)
  win_right = np.zeros(num_times, dtype=np.int64)
  for w in range(num_w):
    w_indices = np.where(time_win == w)[0]
    if len(w_indices) > 0:
      min_idx = int(np.min(w_indices))
      max_idx = int(np.max(w_indices))
      if max_idx - min_idx + 1 != len(w_indices):
        raise ValueError(
            f"Time window {w} is not a contiguous interval (spans indices"
            f" [{min_idx}, {max_idx}] with {len(w_indices)} steps)."
            " method='kronecker' requires contiguous time windows."
        )
      win_left[w_indices] = min_idx
      win_right[w_indices] = max_idx

  return node_comm, time_win, win_left, win_right


def _check_time_step_units(unique_times: np.ndarray) -> None:
  """Validates that evaluation times are integer step counts of the model's time step.

  If index.unique_times has more than one time, computes the differences between
  consecutive times. If the greatest common divisor of these differences is > 1,
  or if differences are not integer step counts, raises ValueError.
  """
  if len(unique_times) > 1:
    diffs = np.diff(unique_times)
    int_diffs: list[int] = []
    for d in diffs:
      d_float = float(d)
      if not (math.isfinite(d_float) and d_float.is_integer() and d_float > 0):
        raise ValueError(
            f"index.unique_times has non-integer or non-positive difference {d}."
            " Times must be integer counts of the model time step. If you evaluate"
            " every g-th step, divide times by g; keeping L, H and R_t in the original"
            " steps is then conservative."
        )
      int_diffs.append(int(round(d_float)))
    g = math.gcd(*int_diffs)
    if g > 1:
      raise ValueError(
          f"index.unique_times has greatest common divisor {g} > 1 of time"
          " differences. Times must be integer counts of the model time step. If you"
          " evaluate every g-th step, divide times by g; keeping L, H and R_t in the"
          " original steps is then conservative."
      )


def range_kappa(
    index: SpaceTimeIndex,
    spatial_edges: Any,
    cluster_ids: Any,
    declaration: RangeDeclaration,
    *,
    method: str = "kronecker",
    max_edges: int = 20_000_000,
) -> RangeKappa:
  """Computes certified variance-inflation factor bound under range envelope.

  Claim RE1 (THEORY_RANGE_ENVELOPE.md):
    kappa <= 1 + lambda_max(A_cut) + gamma
    where A_cut is the cut-adjacency of item pairs within spatial range
    bar_R_s = 2*K + R_s and temporal range bar_R_t = L + H + R_t that belong to
    distinct clusters.

  Note on time units:
    Times must be integer counts of the model's time step; the check cannot
    detect finer units with jitter.

  Args:
    index: SpaceTimeIndex of evaluated items.
    spatial_edges: (E, 2) array of spatial graph edges.
    cluster_ids: 1-D array of length index.n_items mapping items to cluster IDs.
    declaration: RangeDeclaration specifying K, L, H, R_s, R_t, and gamma.
    method: Calculation method: 'kronecker' (fast, matrix-free for product
      partitions) or 'direct' (materializes item-level range edges). Default is
      'kronecker'.
    max_edges: Maximum item-level edges to materialize under method='direct'.
      Default is 20,000,000.

  Returns:
    RangeKappa dataclass with certified kappa, cut_bound, gamma, method,
    max_cut_row_sum, and num_items.

  Raises:
    ValueError: If inputs are invalid, if cluster_ids length != index.n_items,
      if method='direct' exceeds max_edges, or if method='kronecker' receives a
      non-product partition.
  """
  if not isinstance(declaration, RangeDeclaration):
    raise ValueError(
        f"declaration must be a RangeDeclaration, got {type(declaration).__name__}."
    )

  if method not in ("kronecker", "direct"):
    raise ValueError(
        f"method must be 'kronecker' or 'direct', got {method!r}."
    )

  _check_time_step_units(index.unique_times)

  n = index.n_items
  c_arr = np.asarray(cluster_ids)
  if c_arr.ndim != 1 or len(c_arr) != n:
    raise ValueError(
        f"cluster_ids must be 1-D array of length n_items={n}, got shape {c_arr.shape}."
    )
  if not np.issubdtype(c_arr.dtype, np.number):
    raise ValueError(
        f"cluster_ids must be numeric, got dtype {c_arr.dtype}."
    )
  if np.issubdtype(c_arr.dtype, np.floating):
    if not (np.all(np.isfinite(c_arr)) and np.all(np.equal(np.mod(c_arr, 1.0), 0.0))):
      raise ValueError(
          "cluster_ids contains non-integer float values or non-finite values."
      )
  c_ids = c_arr.astype(np.int64)

  if n <= 1:
    return RangeKappa(
        kappa=1.0 + float(declaration.gamma),
        cut_bound=0.0,
        gamma=float(declaration.gamma),
        method=method,
        max_cut_row_sum=0.0,
        num_items=n,
    )

  r_s = declaration.spatial_range
  r_t = declaration.temporal_range
  gamma = float(declaration.gamma)

  # Node and time coordinate mappings
  node_indices = np.searchsorted(index.unique_nodes, index.node_ids).astype(
      np.int64
  )
  time_indices = np.searchsorted(index.unique_times, index.times).astype(
      np.int64
  )

  # Precompute 0/1 spatial ball matrix
  s_ball = _compute_spatial_ball_matrix(
      index.num_nodes, spatial_edges, index.unique_nodes, r_s
  )

  if method == "direct":
    # --------------------------------------------------------------------------
    # Method 1: Direct materialization of item-level range edges
    # --------------------------------------------------------------------------
    # Group items by spatial node, sorted by time within each node
    items_by_node: list[np.ndarray] = []
    node_times: list[np.ndarray] = []
    for i in range(index.num_nodes):
      items_i_raw = np.where(node_indices == i)[0]
      if len(items_i_raw) > 0:
        times_i_raw = index.times[items_i_raw]
        order = np.argsort(times_i_raw, kind="stable")
        items_by_node.append(items_i_raw[order])
        node_times.append(times_i_raw[order])
      else:
        items_by_node.append(np.zeros(0, dtype=np.int64))
        node_times.append(np.zeros(0, dtype=index.times.dtype))

    edge_u_chunks: list[np.ndarray] = []
    edge_v_chunks: list[np.ndarray] = []
    total_edges = 0

    for i in range(index.num_nodes):
      items_i = items_by_node[i]
      if len(items_i) == 0:
        continue
      times_i = node_times[i]

      # Find all spatial neighbor nodes j >= i within spatial_range
      for j in range(i, index.num_nodes):
        if s_ball[i, j] == 0.0:
          continue
        items_j = items_by_node[j]
        if len(items_j) == 0:
          continue
        times_j = node_times[j]

        # For each item in node i, find matching items in node j with |t_u - t_v| <= r_t
        for idx_in_i, u in enumerate(items_i):
          t_u = int(times_i[idx_in_i])
          r_t_int = int(r_t)
          if i == j:
            left = idx_in_i + 1
            right = int(np.searchsorted(times_i, t_u + r_t_int, side="right"))
            if right > left:
              cand_v = items_i[left:right]
            else:
              cand_v = np.zeros(0, dtype=np.int64)
          else:
            t_min = t_u - r_t_int
            t_max = t_u + r_t_int
            left = int(np.searchsorted(times_j, t_min, side="left"))
            right = int(np.searchsorted(times_j, t_max, side="right"))
            if right > left:
              cand_v = items_j[left:right]
            else:
              cand_v = np.zeros(0, dtype=np.int64)

          count_uv = len(cand_v)
          if count_uv > 0:
            total_edges += count_uv
            if total_edges > max_edges:
              raise ValueError(
                  f"Number of range edges ({total_edges}) exceeds"
                  f" max_edges={max_edges}. Consider using method='kronecker'"
                  " instead."
              )
            edge_u_chunks.append(np.full(count_uv, u, dtype=np.int64))
            edge_v_chunks.append(cand_v.astype(np.int64))

    if edge_u_chunks:
      range_edges = np.column_stack([
          np.concatenate(edge_u_chunks),
          np.concatenate(edge_v_chunks),
      ])
    else:
      range_edges = np.zeros((0, 2), dtype=np.int64)

    res = _graph.topological_kappa(
        num_nodes=n, edges=range_edges, cluster_ids=c_ids, phi=1.0
    )
    cut_bound = float(res.lambda_upper)
    if not math.isfinite(cut_bound):
      cut_bound = float(res.max_cut_row_sum)
    kappa_val = float(res.kappa + gamma)
    if not (math.isfinite(kappa_val) and kappa_val >= 1.0 + gamma):
      raise ValueError(
          f"Computed kappa ({kappa_val}) is invalid; must be finite and >= 1 + gamma ({1.0 + gamma})."
      )
    return RangeKappa(
        kappa=kappa_val,
        cut_bound=cut_bound,
        gamma=gamma,
        method="direct",
        max_cut_row_sum=float(res.max_cut_row_sum),
        num_items=n,
    )

  # ----------------------------------------------------------------------------
  # Method 2: Kronecker matrix-free cut operator
  # ----------------------------------------------------------------------------
  # Verify product partition structure and extract community / window mappings
  node_comm, _, win_left, win_right = _extract_product_partition(
      index, c_ids, node_indices, time_indices
  )

  num_times = index.num_times
  num_nodes = index.num_nodes

  # Compute temporal window indices for full band and within-window band
  full_left = np.zeros(num_times, dtype=np.int64)
  full_right = np.zeros(num_times, dtype=np.int64)
  r_t_int = int(r_t)
  for tau in range(num_times):
    t_curr = int(index.unique_times[tau])
    full_left[tau] = int(
        np.searchsorted(index.unique_times, t_curr - r_t_int, side="left")
    )
    full_right[tau] = (
        int(np.searchsorted(index.unique_times, t_curr + r_t_int, side="right")) - 1
    )

  # Within-community spatial ball: S_within[i, j] = S[i, j] * 1{node_comm[i] == node_comm[j]}
  comm_mask = node_comm[:, None] == node_comm[None, :]
  s_within = s_ball * comm_mask.astype(np.float64)

  # Precompute clamped temporal band bounds within windows
  win_band_left = np.maximum(win_left, full_left)
  win_band_right = np.minimum(win_right, full_right)

  def apply_cut_operator(x_vec: np.ndarray) -> tuple[np.ndarray, float]:
    """Evaluates the matrix-free cut adjacency operator y = A_cut x and prefix sum bound P."""
    x_grid = np.zeros((num_nodes, num_times), dtype=np.float64)
    x_grid[node_indices, time_indices] = x_vec

    # 1. Full range operator
    y1_full = s_ball @ x_grid
    prefix_full = np.pad(np.cumsum(y1_full, axis=1), ((0, 0), (1, 0)))
    y2_full = prefix_full[:, full_right + 1] - prefix_full[:, full_left]

    # 2. Within-cluster operator
    y1_within = s_within @ x_grid
    prefix_within = np.pad(np.cumsum(y1_within, axis=1), ((0, 0), (1, 0)))
    y2_within = (
        prefix_within[:, win_band_right + 1] - prefix_within[:, win_band_left]
    )

    # Cut is difference (self-terms cancel identically)
    y_cut_grid = y2_full - y2_within
    y_out = y_cut_grid[node_indices, time_indices]
    p_max = float(np.max(prefix_full[:, -1]))
    return np.maximum(y_out, 0.0), p_max

  # 1. Compute exact max cut row sum from all-ones vector
  ones = np.ones(n, dtype=np.float64)
  cut_row_sums, _ = apply_cut_operator(ones)
  ub_rows = float(np.max(cut_row_sums))

  if not math.isfinite(ub_rows):
    raise ValueError(f"Max cut row sum is non-finite: {ub_rows}.")

  if ub_rows >= (1 << 53):
    raise ValueError(f"Row sums exceed 2^53: {ub_rows}.")

  if ub_rows <= 0.0:
    return RangeKappa(
        kappa=1.0 + gamma,
        cut_bound=0.0,
        gamma=gamma,
        method="kronecker",
        max_cut_row_sum=0.0,
        num_items=n,
    )

  # 2. Run shifted power iteration to estimate the Perron eigenvector v
  # (A_cut + mu*I breaks potential bipartite oscillation)
  mu = ub_rows
  v = np.ones(n, dtype=np.float64)
  for _ in range(50):
    v_next, _ = apply_cut_operator(v)
    v_next = v_next + mu * v
    m_v = float(np.max(v_next))
    if not math.isfinite(m_v) or m_v <= 0.0:
      break
    v = v_next / m_v

  m_v = float(np.max(v))
  if math.isfinite(m_v) and m_v > 0.0:
    v_norm = v / m_v
  else:
    v_norm = np.ones(n, dtype=np.float64)

  # 3. Collatz-Wielandt bound with float rigor
  # Use only x_pert (strictly positive, min >= 1e-3 after normalizing max to 1)
  x_pert = v_norm + 1e-3
  y_pert, p_pert = apply_cut_operator(x_pert)

  eps64 = float(np.finfo(np.float64).eps)
  err = 4.0 * (num_nodes + num_times + 2) * eps64 * p_pert

  quotients = (y_pert + err) / x_pert
  if np.all(np.isfinite(quotients)):
    ub_cw = float(np.max(quotients))
  else:
    ub_cw = ub_rows

  if not math.isfinite(ub_cw):
    ub_cw = ub_rows

  # Certified bound = min(ub_cw, ub_rows) * (1 + 1e-9). ub_cw already includes
  # the explicit float64 rounding margin err, and ub_rows is an exact integer.
  cut_bound = float(min(ub_cw, ub_rows) * (1.0 + 1e-9))
  kappa_val = float(1.0 + cut_bound + gamma)

  if not (math.isfinite(kappa_val) and kappa_val >= 1.0 + gamma):
    raise ValueError(
        f"Computed kappa ({kappa_val}) is invalid; must be finite and >= 1 + gamma ({1.0 + gamma})."
    )

  return RangeKappa(
      kappa=kappa_val,
      cut_bound=cut_bound,
      gamma=gamma,
      method="kronecker",
      max_cut_row_sum=ub_rows,
      num_items=n,
  )


def range_kappa_fn(
    index: SpaceTimeIndex,
    spatial_edges: Any,
    declaration: RangeDeclaration,
    *,
    method: str = "kronecker",
    max_edges: int = 20_000_000,
) -> Callable[[Candidate], float]:
  """Constructs a candidate scoring kappa_fn for choose_partition.

  The returned callable takes a Candidate and returns certified kappa >= 1.0,
  compatible with spacetime.choose_partition.

  Args:
    index: SpaceTimeIndex of evaluated items.
    spatial_edges: Spatial edges (E, 2).
    declaration: RangeDeclaration.
    method: Calculation method ('kronecker' or 'direct'). Default is
      'kronecker'.
    max_edges: Maximum edges to materialize under method='direct'.

  Returns:
    Callable[[Candidate], float] returning certified kappa.
  """
  if not isinstance(declaration, RangeDeclaration):
    raise ValueError(
        f"declaration must be a RangeDeclaration, got {type(declaration).__name__}."
    )

  _check_time_step_units(index.unique_times)

  def _kappa_fn(c: Candidate) -> float:
    res = range_kappa(
        index,
        spatial_edges,
        c.cluster_ids,
        declaration,
        method=method,
        max_edges=max_edges,
    )
    return float(res.kappa)

  setattr(_kappa_fn, "is_range", True)
  setattr(_kappa_fn, "is_separable", False)
  return _kappa_fn


def range_candidates(
    index: SpaceTimeIndex,
    spatial_edges: Any,
    declaration: RangeDeclaration,
    *,
    target_sizes: Sequence[int] | None = None,
    window_multipliers: Sequence[int] | None = None,
    seed: int = 0,
) -> list[Candidate]:
  """Constructs candidate partitions tailored to the range envelope declaration.

  Composes spatial community partitions from partitioners.partition_graph
  with temporal windows of length w * (temporal_range + 1) for w in
  {1, 2, 4, 8, ...} up to index.num_times, delegating candidate generation
  to spacetime.candidate_partitions.

  Args:
    index: SpaceTimeIndex of evaluated items.
    spatial_edges: Spatial edges (E, 2).
    declaration: RangeDeclaration.
    target_sizes: Optional sequence of target spatial community sizes. If None,
      defaults to geometric sequence [1, 2, 4, 8, 16, 32, 64] filtered to
      <= num_nodes.
    window_multipliers: Optional sequence of integer multipliers w >= 1. If
      None, defaults to powers of two [1, 2, 4, 8, ...].
    seed: PRNG seed for deterministic spatial graph partitioning.

  Returns:
    List of Candidate partitions.

  Raises:
    ValueError: If declaration is invalid.
  """
  if not isinstance(declaration, RangeDeclaration):
    raise ValueError(
        f"declaration must be a RangeDeclaration, got {type(declaration).__name__}."
    )

  num_times = index.num_times
  w_base = declaration.temporal_range + 1

  if window_multipliers is not None:
    multipliers = list(window_multipliers)
  else:
    multipliers = []
    w_curr = 1
    while w_curr * w_base <= num_times:
      multipliers.append(w_curr)
      w_curr *= 2
    if not multipliers:
      multipliers = [1]

  window_lengths: list[int] = [1]
  for m in multipliers:
    wl = m * w_base
    if wl <= num_times:
      window_lengths.append(wl)
  if not window_lengths:
    window_lengths = [1, num_times]

  # Deduplicate while preserving order
  seen_wl = set()
  dedup_wl: list[int] = []
  for wl in window_lengths:
    if wl not in seen_wl:
      seen_wl.add(wl)
      dedup_wl.append(wl)

  if target_sizes is not None:
    ts_list = list(target_sizes)
  else:
    ts_cands = [1, 2, 4, 8, 16, 32, 64]
    ts_list = [ts for ts in ts_cands if ts <= index.num_nodes]
    if not ts_list:
      ts_list = [1]
    if index.num_nodes not in ts_list:
      ts_list.append(index.num_nodes)

  if 1 not in ts_list:
    ts_list = [1] + ts_list

  seen_ts = set()
  dedup_ts: list[int] = []
  for ts in ts_list:
    if ts not in seen_ts:
      seen_ts.add(ts)
      dedup_ts.append(ts)

  return _spacetime.candidate_partitions(
      index,
      spatial_edges,
      target_sizes=dedup_ts,
      window_lengths=dedup_wl,
      seed=seed,
  )
