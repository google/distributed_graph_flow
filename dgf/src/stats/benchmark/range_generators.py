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

"""Synthetic space-time data generators with exact finite dependency ranges.

Provides synthetic data generators on graphs (ring and 2D lattice) with exact
finite-range noise fields, common shocks, and diffuse global dependence,
along with diagnostics for empirical loss correlations and out-of-range row sums.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any
import numpy as np


@dataclasses.dataclass(frozen=True)
class GraphStructure:
  """Graph topology representation for space-time benchmarking.

  Attributes:
    num_nodes: Total number of spatial nodes n.
    edges: Undirected edge array of shape (E, 2).
    dist_matrix: All-pairs shortest path hop distance matrix of shape (n, n).
    kind: Name of the graph topology ('ring', 'grid_2d').
  """

  num_nodes: int
  edges: np.ndarray
  dist_matrix: np.ndarray
  kind: str


def make_ring_graph(num_nodes: int) -> GraphStructure:
  """Constructs an undirected ring graph with num_nodes vertices.

  Hop distance between nodes u and v is min(|u - v|, num_nodes - |u - v|).

  Args:
    num_nodes: Number of vertices in the ring (>= 3).

  Returns:
    GraphStructure containing edges and exact distance matrix.
  """
  if num_nodes < 3:
    raise ValueError(f"Ring graph requires at least 3 nodes, got {num_nodes}.")

  edges = np.array(
      [[i, (i + 1) % num_nodes] for i in range(num_nodes)], dtype=np.int64
  )

  # Exact analytical shortest-path distance matrix on a ring
  nodes = np.arange(num_nodes)
  diff = np.abs(nodes[:, None] - nodes[None, :])
  dist = np.minimum(diff, num_nodes - diff).astype(np.int32)

  return GraphStructure(
      num_nodes=num_nodes,
      edges=edges,
      dist_matrix=dist,
      kind="ring",
  )


def make_grid_2d_graph(num_rows: int, num_cols: int) -> GraphStructure:
  """Constructs an undirected 2D grid lattice graph with 4-connectivity.

  Vertices are indexed by u = r * num_cols + c.
  Hop distance is the Manhattan distance |r1 - r2| + |c1 - c2|.

  Args:
    num_rows: Number of grid rows (>= 1).
    num_cols: Number of grid columns (>= 1).

  Returns:
    GraphStructure containing edges and exact distance matrix.
  """
  if num_rows < 1 or num_cols < 1:
    raise ValueError(
        f"Grid dimensions must be >= 1, got ({num_rows}, {num_cols})."
    )

  num_nodes = num_rows * num_cols
  edge_list = []
  for r in range(num_rows):
    for c in range(num_cols):
      u = r * num_cols + c
      if r + 1 < num_rows:
        edge_list.append((u, (r + 1) * num_cols + c))
      if c + 1 < num_cols:
        edge_list.append((u, r * num_cols + (c + 1)))

  edges = np.array(edge_list, dtype=np.int64) if edge_list else np.zeros((0, 2), dtype=np.int64)

  # Exact Manhattan distance matrix
  coords = np.array([(r, c) for r in range(num_rows) for c in range(num_cols)], dtype=np.int32)
  dist = (
      np.abs(coords[:, 0:1] - coords[:, 0:1].T)
      + np.abs(coords[:, 1:2] - coords[:, 1:2].T)
  ).astype(np.int32)

  return GraphStructure(
      num_nodes=num_nodes,
      edges=edges,
      dist_matrix=dist,
      kind="grid_2d",
  )


def generate_moving_average_noise(
    graph: GraphStructure,
    num_times: int,
    r_s: int,
    r_t: int,
    rng: np.random.Generator,
) -> np.ndarray:
  """Generates pure finite-range noise via a spatial-temporal moving average.

  Innovations epsilon_{v, tau} ~ N(0, 1) are i.i.d. for tau in [-r_t, num_times - 1].
  The moving average field is:
    eta_{u, t} = sum_{v in N_{r_s}(u)} sum_{l=0}^{r_t} epsilon_{v, t - l} / sqrt(|N_{r_s}(u)| * (r_t + 1))

  Because eta_{u, t} depends only on innovations within r_s hops and r_t lags:
  - If d(u, v) > 2 * r_s, N_{r_s}(u) and N_{r_s}(v) are disjoint.
  - If |t - tau| > r_t, the lag sets are disjoint.
  Items separated by > 2*r_s hops OR > r_t steps share zero innovations and are
  strictly independent.

  Args:
    graph: GraphStructure of the spatial domain.
    num_times: Total time steps T >= 1.
    r_s: Spatial moving-average kernel radius in hops (>= 0).
    r_t: Temporal moving-average lag depth in steps (>= 0).
    rng: NumPy random generator.

  Returns:
    Array of shape (num_nodes, num_times) with zero mean and unit variance.
  """
  if r_s < 0 or r_t < 0:
    raise ValueError(f"Radii must be non-negative, got r_s={r_s}, r_t={r_t}.")
  if num_times < 1:
    raise ValueError(f"num_times must be >= 1, got {num_times}.")

  n = graph.num_nodes
  total_t = num_times + r_t
  innovations = rng.normal(0.0, 1.0, size=(n, total_t))

  # 1. Temporal moving average along time axis: sum over lag 0..r_t
  # Shape remains (n, num_times)
  temp_ma = np.zeros((n, num_times), dtype=np.float64)
  for l in range(r_t + 1):
    temp_ma += innovations[:, r_t - l : r_t - l + num_times]

  # 2. Spatial moving average over r_s hop neighborhood
  # Mask of shape (n, n) where mask[u, v] is True if dist(u, v) <= r_s
  spatial_mask = (graph.dist_matrix <= r_s).astype(np.float64)
  deg_s = np.sum(spatial_mask, axis=1, keepdims=True)  # (n, 1)

  # Matrix product: (n, n) @ (n, num_times) -> (n, num_times)
  spatio_temp_sum = spatial_mask @ temp_ma

  # Normalize so each node has unit variance Var(eta_{u,t}) = 1.0
  norm_factor = np.sqrt(deg_s * float(r_t + 1))
  eta = spatio_temp_sum / norm_factor
  return eta


def generate_space_time_field(
    graph: GraphStructure,
    num_times: int,
    r_s: int,
    r_t: int,
    variant: int = 1,
    rho_0: float = 0.05,
    c: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
  """Generates synthetic space-time field under Variant 1, 2, or 3.

  Variants:
    Variant 1 (pure finite-range):
      X_{u, t} = eta_{u, t}
    Variant 2 (finite-range + common shock with correlation rho_0):
      X_{u, t} = sqrt(1 - rho_0) * eta_{u, t} + sqrt(rho_0) * Z_t
      where Z_t ~ N(0, 1) is a common spatial shock across all nodes at time t.
    Variant 3 (finite-range + diffuse global term with per-pair corr c / n):
      X_{u, t} = sqrt(1 - c / n) * eta_{u, t} + sqrt(c / n) * W_t
      where W_t ~ N(0, 1) is a common global shock scaled by 1/sqrt(n).

  Args:
    graph: GraphStructure of the spatial domain.
    num_times: Total time steps T.
    r_s: Spatial moving-average kernel radius.
    r_t: Temporal moving-average lag depth.
    variant: Variant selector (1, 2, or 3).
    rho_0: Common shock correlation parameter for Variant 2 (in [0, 1)).
    c: Diffuse term scaling parameter for Variant 3 (c >= 0, c < n).
    seed: PRNG seed for reproducible generation.

  Returns:
    Array X of shape (num_nodes, num_times).
  """
  rng = np.random.default_rng(seed)
  eta = generate_moving_average_noise(graph, num_times, r_s, r_t, rng)

  if variant == 1:
    return eta
  elif variant == 2:
    if not (0.0 <= rho_0 < 1.0):
      raise ValueError(f"rho_0 must be in [0, 1), got {rho_0}.")
    z_t = rng.normal(0.0, 1.0, size=(1, num_times))
    return math.sqrt(1.0 - rho_0) * eta + math.sqrt(rho_0) * z_t
  elif variant == 3:
    n = graph.num_nodes
    if not (0.0 <= c < float(n)):
      raise ValueError(f"c must be in [0, n) with n={n}, got c={c}.")
    alpha = c / float(n)
    w_t = rng.normal(0.0, 1.0, size=(1, num_times))
    return math.sqrt(1.0 - alpha) * eta + math.sqrt(alpha) * w_t
  else:
    raise ValueError(f"Unknown variant: {variant}. Supported: 1, 2, 3.")


def compute_predictor_and_losses(
    x: np.ndarray,
    graph: GraphStructure,
    k_hops: int,
    l_lookback: int,
    horizon_h: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes rolling neighborhood-average predictions and squared error losses.

  Model:
    At origin time t, predicting target node u at horizon H (target time t + H):
      Y_{u, t+H} = X_{u, t+H}
      Y_hat_{u, t+H} = (1 / (|N_K(u)| * L)) * sum_{v in N_K(u)} sum_{l=0}^{L-1} X_{v, t - l}
      residual e_{u, t} = Y_{u, t+H} - Y_hat_{u, t+H}
      loss s_{u, t} = e_{u, t}^2

  Evaluated items:
    Origin times t in [L - 1, num_times - horizon_h - 1].

  Theoretical Independence Range (Part A Claim, Lemma RE1):
    The residual e_{u, t} depends on data in:
      Spatial footprint: N_{K + r_s}(u)  (radius K + r_s)
      Temporal footprint: [t - L + 1 - r_t,  t + H]
    Therefore, two losses s_{u, t} and s_{v, t + delta_t} have disjoint innovation
    footprints and are strictly independent whenever:
      hop distance d(u, v) > 2*K + 2*r_s
      OR
      temporal lag delta_t >= L + H + r_t.

  Args:
    x: Data field of shape (num_nodes, num_times).
    graph: GraphStructure.
    k_hops: Predictor spatial message-passing radius K (>= 0).
    l_lookback: Predictor temporal lookback window length L (>= 1).
    horizon_h: Forecast horizon H (>= 1).

  Returns:
    Tuple of (losses, eval_times) where losses has shape (num_nodes, num_eval_times)
    and eval_times is a 1-D array of evaluated origin time indices.
  """
  n, total_t = x.shape
  if l_lookback < 1 or horizon_h < 1 or k_hops < 0:
    raise ValueError(
        f"Invalid parameters: K={k_hops}, L={l_lookback}, H={horizon_h}."
    )

  t_start = l_lookback - 1
  t_end = total_t - horizon_h
  if t_start >= t_end:
    raise ValueError(
        f"Insufficient time steps {total_t} for L={l_lookback}, H={horizon_h}."
    )

  eval_times = np.arange(t_start, t_end, dtype=np.int64)
  num_eval = len(eval_times)

  # Spatial aggregation matrix for predictor: N_K(u)
  k_mask = (graph.dist_matrix <= k_hops).astype(np.float64)
  deg_k = np.sum(k_mask, axis=1, keepdims=True)

  # Compute rolling lookback sum over L steps for each node:
  # shape (n, num_eval)
  rolling_temp = np.zeros((n, num_eval), dtype=np.float64)
  for l in range(l_lookback):
    rolling_temp += x[:, eval_times - l]

  # Spatial aggregation of rolling lookback sum
  pred = (k_mask @ rolling_temp) / (deg_k * float(l_lookback))

  # Target label at t + H: shape (n, num_eval)
  targets = x[:, eval_times + horizon_h]

  residuals = targets - pred
  losses = residuals**2
  return losses, eval_times


def compute_empirical_loss_correlations(
    losses: np.ndarray,
    graph: GraphStructure,
    max_hop: int,
    max_lag: int,
) -> np.ndarray:
  """Computes empirical correlation of losses as a function of (hop distance, lag).

  rho_hat(d, delta_t) = mean_{u, v: dist(u, v)=d} corr(s_{u, t}, s_{v, t + delta_t})

  Args:
    losses: Array of shape (num_nodes, num_eval_times).
    graph: GraphStructure.
    max_hop: Maximum hop distance to evaluate.
    max_lag: Maximum temporal lag to evaluate.

  Returns:
    Array rho_hat of shape (max_hop + 1, max_lag + 1).
  """
  n, t_eval = losses.shape
  max_hop = min(max_hop, int(np.max(graph.dist_matrix)))
  max_lag = min(max_lag, t_eval - 2)

  # Pre-center and normalize losses per sensor for fast correlation computation
  means = np.mean(losses, axis=1, keepdims=True)
  stds = np.std(losses, axis=1, keepdims=True)
  stds = np.where(stds > 1e-12, stds, 1.0)
  z = (losses - means) / stds

  rho_matrix = np.full((max_hop + 1, max_lag + 1), np.nan, dtype=np.float64)

  # Group node pairs by hop distance
  pairs_by_hop = {d: [] for d in range(max_hop + 1)}
  for u in range(n):
    for v in range(n):
      d = graph.dist_matrix[u, v]
      if d <= max_hop:
        pairs_by_hop[d].append((u, v))

  for d in range(max_hop + 1):
    pairs = pairs_by_hop[d]
    if not pairs:
      continue
    u_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    v_idx = np.array([p[1] for p in pairs], dtype=np.int64)

    for lag in range(max_lag + 1):
      t_len = t_eval - lag
      if t_len < 10:
        continue
      # Vectorized dot product across all pairs for this (d, lag)
      z_u = z[u_idx, :t_len]
      z_v = z[v_idx, lag : lag + t_len]
      pair_corrs = np.mean(z_u * z_v, axis=1)
      rho_matrix[d, lag] = float(np.mean(pair_corrs))

  return rho_matrix


def compute_out_of_range_row_sum(
    losses: np.ndarray,
    graph: GraphStructure,
    spatial_range: int,
    temporal_range: int,
    max_hop: int | None = None,
    max_lag: int | None = None,
) -> tuple[float, float, float]:
  """Computes empirical row sum of |correlation| beyond the theoretical range.

  For an item (u, t), the out-of-range budget gamma is:
    gamma_emp = mean_u sum_{v, lag : d(u, v) > spatial_range or lag >= temporal_range} |rho_hat(d(u, v), lag)|

  Args:
    losses: Array of shape (num_nodes, num_eval_times).
    graph: GraphStructure.
    spatial_range: Theoretical spatial independence range d_range = 2K + 2r_s.
    temporal_range: Theoretical temporal independence range lag_range = L + H + r_t.
    max_hop: Optional maximum hop distance to include in the sum.
    max_lag: Optional maximum lag to include in the sum.

  Returns:
    Tuple of (mean_out_of_range_row_sum, max_out_of_range_row_sum, in_range_sum)
    where mean and max coincide on vertex-transitive graphs.
  """
  n, t_eval = losses.shape
  if max_hop is None:
    max_hop = int(np.max(graph.dist_matrix))
  if max_lag is None:
    max_lag = min(50, t_eval - 1)

  rho_table = compute_empirical_loss_correlations(losses, graph, max_hop, max_lag)

  # Count multiplicity of each hop distance d for an average node
  hop_counts = np.zeros(max_hop + 1, dtype=np.float64)
  for u in range(n):
    dists = graph.dist_matrix[u, :]
    for d in dists:
      if d <= max_hop:
        hop_counts[d] += 1.0
  hop_counts /= float(n)

  out_of_range_sum = 0.0
  in_range_sum = 0.0

  for d in range(max_hop + 1):
    for lag in range(max_lag + 1):
      val = abs(float(rho_table[d, lag]))
      if not math.isfinite(val):
        continue
      # Temporal lag multiplicity: lag=0 counts once (same time), lag > 0 counts twice (forward/backward)
      t_mult = 1.0 if lag == 0 else 2.0
      term = hop_counts[d] * t_mult * val

      if d > spatial_range or lag >= temporal_range:
        out_of_range_sum += term
      else:
        in_range_sum += term

  return out_of_range_sum, out_of_range_sum, in_range_sum


def run_range_diagnostic(
    num_nodes: int = 200,
    num_times: int = 500,
    k_hops: int = 1,
    l_lookback: int = 3,
    horizon_h: int = 1,
    r_s: int = 1,
    r_t: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
  """Runs the finite-range correlation diagnostic table across variants."""
  graph = make_ring_graph(num_nodes)
  spatial_range = 2 * k_hops + 2 * r_s  # 4
  temporal_range = l_lookback + horizon_h + r_t  # 6

  display_hops = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
  display_lags = [0, 1, 2, 3, 4, 5, 6, 8, 10]

  variants_config = [
      ("Variant 1 (Pure finite-range)", 1, 0.0, 0.0),
      ("Variant 2 (Common shock rho_0=0.05)", 2, 0.05, 0.0),
      ("Variant 3 (Diffuse global term c=1.0)", 3, 0.0, 1.0),
  ]

  results = {}

  print("\n" + "=" * 78)
  print("FINITE-RANGE SPACE-TIME CORRELATION DIAGNOSTIC TABLE")
  print(f"Ring: n={num_nodes}, T={num_times} | Model: K={k_hops}, L={l_lookback}, H={horizon_h} | Noise: r_s={r_s}, r_t={r_t}")
  print(f"Theoretical Ranges: Spatial d_range = 2K + 2r_s = {spatial_range} hops | Temporal lag_range = L + H + r_t = {temporal_range} steps")
  print("=" * 78)

  for var_title, var_num, rho_0_val, c_val in variants_config:
    field = generate_space_time_field(
        graph, num_times=num_times, r_s=r_s, r_t=r_t, variant=var_num,
        rho_0=rho_0_val, c=c_val, seed=seed,
    )
    losses, eval_times = compute_predictor_and_losses(
        field, graph, k_hops=k_hops, l_lookback=l_lookback, horizon_h=horizon_h
    )

    max_h = max(display_hops)
    max_l = max(display_lags)
    rho_mat = compute_empirical_loss_correlations(losses, graph, max_hop=max_h, max_lag=max_l)
    out_sum, _, in_sum = compute_out_of_range_row_sum(
        losses, graph, spatial_range=spatial_range, temporal_range=temporal_range,
        max_hop=50, max_lag=30,
    )

    print(f"\n### {var_title}")
    print(f"Evaluated losses shape: {losses.shape} (T_eval = {len(eval_times)})")
    print(f"Out-of-range row sum gamma_emp (d > {spatial_range} or lag >= {temporal_range}): {out_sum:.4f}")
    print(f"In-range row sum: {in_sum:.4f}")

    header = f"{'Hop d':<8} | " + " | ".join(f"lag={l:<4}" for l in display_lags)
    print("\n" + header)
    print("-" * len(header))

    sub_table = {}
    for d in display_hops:
      row_vals = [rho_mat[d, l] for l in display_lags]
      sub_table[d] = row_vals
      row_str = f"{d:<8d} | " + " | ".join(f"{val:>8.4f}" for val in row_vals)
      print(row_str)

    results[var_title] = {
        "out_of_range_row_sum": out_sum,
        "in_range_row_sum": in_sum,
        "rho_sub_table": sub_table,
    }

  return results


if __name__ == "__main__":
  run_range_diagnostic()

