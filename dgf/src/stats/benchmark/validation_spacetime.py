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

"""Space-time validation benchmark.

Synthetic benchmark evaluating blocks ST1, ST2, ST3a, ST3b, ST4 and ST5 across
32 cells. Records per-cell results to JSONL.
"""

from collections.abc import Sequence
import concurrent.futures
import json
import math
import os
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import scipy.signal
import scipy.stats

from dgf.src.stats import graph as _graph
from dgf.src.stats import partitioners as _partitioners
from dgf.src.stats import refutation as _refutation
from dgf.src.stats import spacetime
from dgf.src.stats import temporal as _temporal
from dgf.src.stats.independent import interval as _ind_interval

Status = _ind_interval.Status

DEFAULT_SEED = 0x59AC371E

_SEED = flags.DEFINE_integer(
    "seed", DEFAULT_SEED, "Base seed for benchmark."
)
_MODE = flags.DEFINE_enum(
    "mode", "quick", ["quick", "full"], "Execution mode: 'quick' or 'full'."
)
_OUTPUT = flags.DEFINE_string(
    "output", None, "Optional output path for validation results JSONL."
)
_WORKERS = flags.DEFINE_integer(
    "workers",
    min(16, os.cpu_count() or 4),
    "Number of parallel worker processes.",
)
_REPS = flags.DEFINE_integer(
    "reps", 0, "Optional override for reps (0 to use per-cell defaults)."
)
_BLOCK = flags.DEFINE_string(
    "block",
    "all",
    "Block to run ('ST1', 'ST2', 'ST3a', 'ST3b', 'ST4', 'ST5', or 'all').",
)
_CELL = flags.DEFINE_integer(
    "cell",
    0,
    "Optional single cell_id to run (0 for all in block).",
)


def clopper_pearson(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
  """Exact Clopper-Pearson binomial confidence interval."""
  if n == 0:
    return 0.0, 1.0
  alpha = 1.0 - confidence
  low = (
      0.0 if k == 0 else float(scipy.stats.beta.ppf(alpha / 2.0, k, n - k + 1))
  )
  high = (
      1.0
      if k == n
      else float(scipy.stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
  )
  return low, high


def one_sided_cp_lower(k: int, n: int, confidence: float = 0.999) -> float:
  """One-sided Clopper-Pearson lower bound on the miss rate at level 0.999."""
  if k == 0:
    return 0.0
  alpha = 1.0 - confidence
  return float(scipy.stats.beta.ppf(alpha, k, n - k + 1))


def q_threshold(reps: int, alpha: float = 0.05) -> int:
  """Binomial upper critical count q(reps, alpha/2) at level 0.999."""
  p = alpha / 2.0
  return int(scipy.stats.binom.ppf(0.999, reps, p))


def build_ring_edges(num_nodes: int) -> np.ndarray:
  """Constructs undirected edges for a 2-regular ring on num_nodes nodes."""
  return np.column_stack([
      np.arange(num_nodes, dtype=np.int64),
      (np.arange(num_nodes, dtype=np.int64) + 1) % num_nodes,
  ])


def nabeya_mae_correlation(rho: float) -> float:
  """Nabeya's exact correlation between |X| and |Y| for standard bivariate normal."""
  if abs(rho) >= 1.0:
    return 1.0 if rho > 0 else -1.0
  if abs(rho) < 1e-12:
    return 0.0
  num = (2.0 / math.pi) * (rho * math.asin(rho) + math.sqrt(1.0 - rho**2) - 1.0)
  den = 1.0 - 2.0 / math.pi
  return float(num / den)


def generate_missing_mask(
    num_nodes: int,
    num_times: int,
    missing_rate: float,
    pattern: str,
    seed: int,
) -> np.ndarray:
  """Generates a boolean mask of shape (num_nodes, num_times) of observed cells."""
  if missing_rate <= 0.0 or pattern == "complete":
    return np.ones((num_nodes, num_times), dtype=bool)

  rng = np.random.default_rng(seed)
  mask = np.ones((num_nodes, num_times), dtype=bool)
  total_cells = num_nodes * num_times

  if pattern == "mcar":
    n_drop = int(round(total_cells * missing_rate))
    flat_indices = rng.choice(total_cells, size=n_drop, replace=False)
    mask.ravel()[flat_indices] = False
  elif pattern == "block_outage":
    block_len = 50
    num_time_blocks = num_times // block_len
    all_blocks = [
        (u, b) for u in range(num_nodes) for b in range(num_time_blocks)
    ]
    n_drop = int(round(total_cells * missing_rate))
    blocks_to_drop = int(round(n_drop / float(block_len)))
    chosen_idx = rng.choice(len(all_blocks), size=blocks_to_drop, replace=False)
    for idx in chosen_idx:
      u, b = all_blocks[idx]
      mask[u, b * block_len : (b + 1) * block_len] = False
  elif pattern == "non_product_drift":
    half_u = num_nodes // 2
    half_t = num_times // 2
    for u in range(num_nodes):
      for t in range(num_times):
        if (u < half_u and t < half_t) or (u >= half_u and t >= half_t):
          p_keep = 0.65
        else:
          p_keep = 0.95
        mask[u, t] = rng.random() < p_keep
  else:
    raise ValueError(f"Unknown missing pattern: {pattern}")

  return mask


# ---------------------------------------------------------------------------
# DGP Generators and Exact VIF / Truth helpers
# ---------------------------------------------------------------------------


def generate_st1_residuals(
    reps: int,
    num_nodes: int,
    num_times: int,
    alpha_s: float,
    rho_t: float,
    rng: np.random.Generator,
) -> np.ndarray:
  """Generates Kronecker separable AR(1) x spatial ring residuals of shape (reps, N, T)."""
  # 1. Stationary AR(1) along time
  z_raw = rng.standard_normal((reps, num_nodes, num_times))
  z_ar1 = np.empty_like(z_raw)
  z_ar1[:, :, 0] = z_raw[:, :, 0]
  for t in range(1, num_times):
    z_ar1[:, :, t] = (
        rho_t * z_ar1[:, :, t - 1] + math.sqrt(1.0 - rho_t**2) * z_raw[:, :, t]
    )

  # 2. Spatial 2-regular ring filter: eta_u = z_u + alpha_s * (z_{u-1} + z_{u+1})
  e = z_ar1 + alpha_s * (
      np.roll(z_ar1, shift=1, axis=1) + np.roll(z_ar1, shift=-1, axis=1)
  )
  return e


def generate_st2_wave(
    reps: int,
    num_nodes: int,
    num_times: int,
    w_s: float,
    w_t: float,
    w_wave: float,
    rng: np.random.Generator,
) -> np.ndarray:
  """Generates non-separable propagating wave field of shape (reps, N, T)."""
  eps = rng.standard_normal((reps, num_nodes, num_times))

  # Spatial edge factors on 40-node ring (40 edges)
  zeta_sp = rng.standard_normal((reps, num_nodes, num_times))
  # Node u touches edges u-1 and u
  spatial_factor = w_s * (zeta_sp + np.roll(zeta_sp, shift=1, axis=1))

  # Temporal same-node factors
  zeta_tm = rng.standard_normal((reps, num_nodes, num_times + 1))
  temporal_factor = w_t * (zeta_tm[:, :, 1:] + zeta_tm[:, :, :-1])

  # Cross-lagged wave factors in both directions
  zeta_wave_fwd = rng.standard_normal((reps, num_nodes, num_times + 1))
  zeta_wave_bwd = rng.standard_normal((reps, num_nodes, num_times + 1))

  wave_term_fwd = zeta_wave_fwd[:, :, 1:] + np.roll(
      zeta_wave_fwd[:, :, :-1], shift=1, axis=1
  )
  wave_term_bwd = zeta_wave_bwd[:, :, 1:] + np.roll(
      zeta_wave_bwd[:, :, :-1], shift=-1, axis=1
  )
  wave_factor = w_wave * (wave_term_fwd + wave_term_bwd)

  return eps + spatial_factor + temporal_factor + wave_factor


def generate_st3a_1hop(
    reps: int,
    num_nodes: int,
    num_times: int,
    rho_s: float,
    rho_t: float,
    rng: np.random.Generator,
) -> np.ndarray:
  """Generates exact 1-hop Kronecker separable residuals with MA(1) time."""
  # 1. Temporal MA(1): correlation at lag 1 is rho_t
  theta_t = (
      (1.0 - math.sqrt(max(0.0, 1.0 - 4.0 * rho_t**2))) / (2.0 * rho_t)
      if rho_t > 0
      else 0.0
  )
  norm_t = math.sqrt(1.0 + theta_t**2)

  z_raw = rng.standard_normal((reps, num_nodes, num_times + 1))
  z_t = (z_raw[:, :, 1:] + theta_t * z_raw[:, :, :-1]) / norm_t

  # 2. Spatial 1-hop moving average: correlation on ring edges is rho_s
  theta_s = (
      (1.0 - math.sqrt(max(0.0, 1.0 - 4.0 * rho_s**2))) / (2.0 * rho_s)
      if rho_s > 0
      else 0.0
  )
  norm_s = math.sqrt(1.0 + 2.0 * theta_s**2)

  e = (
      z_t
      + theta_s
      * (np.roll(z_t, shift=1, axis=1) + np.roll(z_t, shift=-1, axis=1))
  ) / norm_s
  return e


def compute_st1_separable_vifs(
    num_nodes: int,
    num_times: int,
    alpha_s: float,
    rho_t: float,
    target_sizes: Sequence[int],
    window_lengths: Sequence[int],
    metric: str,
) -> tuple[float, float, float, float]:
  """Computes analytical kappa_s and kappa_t for Block ST1."""
  # Spatial ring correlations
  denom_s = 1.0 + 2.0 * alpha_s**2
  rho_s_1 = (2.0 * alpha_s) / denom_s
  rho_s_2 = (alpha_s**2) / denom_s

  if metric == "mse":
    # By Isserlis' theorem, correlation of squared errors is rho^2
    r_s1 = rho_s_1**2
    r_s2 = rho_s_2**2
    r_t1 = rho_t**2
  else:
    # MAE Nabeya correlation
    r_s1 = nabeya_mae_correlation(rho_s_1)
    r_s2 = nabeya_mae_correlation(rho_s_2)
    r_t1 = nabeya_mae_correlation(rho_t)

  # Spatial VIF: max over candidate target sizes
  kappa_s_candidates = []
  for ts in target_sizes:
    # Total ring variance: 1 + 2 r_s1 + 2 r_s2
    v_tot_s = num_nodes * (1.0 + 2.0 * r_s1 + 2.0 * r_s2)
    # Within contiguous community of size ts:
    v_c_s = float(ts) + 2.0 * float(ts - 1) * r_s1
    if ts >= 3:
      v_c_s += 2.0 * float(ts - 2) * r_s2
    g_s = num_nodes // ts
    k_s = v_tot_s / (float(g_s) * v_c_s)
    kappa_s_candidates.append(k_s)
  kappa_s = max(1.0, max(kappa_s_candidates))

  # Temporal VIF: max over candidate window lengths for AR(1) correlation r_t1
  kappa_t_candidates = []
  h_T = np.arange(1, num_times, dtype=np.float64)
  v_tot_t = float(num_times) + 2.0 * float(
      np.sum((num_times - h_T) * (r_t1**h_T))
  )
  for wl in window_lengths:
    h_w = np.arange(1, wl, dtype=np.float64)
    v_w_t = float(wl) + 2.0 * float(np.sum((wl - h_w) * (r_t1**h_w)))
    g_t = num_times // wl
    k_t = v_tot_t / (float(g_t) * v_w_t)
    kappa_t_candidates.append(k_t)
  kappa_t = max(1.0, max(kappa_t_candidates))

  phi_s = r_s1
  phi_t = r_t1
  return float(kappa_s), float(kappa_t), float(phi_s), float(phi_t)


# ---------------------------------------------------------------------------
# Single Cell Simulation Execution
# ---------------------------------------------------------------------------


def run_cell_simulation(
    cell: dict[str, Any],
    reps_override: int = 0,
) -> dict[str, Any]:
  """Runs the Monte Carlo evaluation for a single cell."""
  cell_id = cell["cell_id"]
  block = cell["block"]
  name = cell["name"]
  metric = cell["metric"]
  level = cell.get("level", 0.95)
  m_item = cell.get("m_item", 3.2 if metric == "mse" else 2.0)
  num_nodes = cell.get("N_s", 40)
  num_times = cell.get("T", 300)
  reps = reps_override if reps_override > 0 else cell.get("reps", 5000)

  alpha = 1.0 - level
  seed_base = int(cell.get("seed_base", DEFAULT_SEED))
  design_seed = seed_base + cell_id
  # Offset chosen so that the default seed reproduces truth seed 0x72074000.
  truth_seed = seed_base + 0x185B08E2 + cell_id

  # 1. Build Spatial Graph and Observation Mask
  spatial_edges = build_ring_edges(num_nodes)
  missing_rate = cell.get("missing_rate", 0.0)
  missing_pattern = cell.get("missing_pattern", "complete")
  obs_mask = generate_missing_mask(
      num_nodes, num_times, missing_rate, missing_pattern, seed=design_seed
  )

  observed_coords = np.where(obs_mask)
  obs_nodes = observed_coords[0].astype(np.int64)
  obs_times = observed_coords[1].astype(np.int64)
  n_items = len(obs_nodes)
  is_complete_grid = n_items == num_nodes * num_times

  index = spacetime.spacetime_items(obs_nodes, obs_times)

  # 2. Candidate partitions and design-only selection (Lemma R)
  if block == "ST5" and cell["cell_id"] == 32:
    target_sizes = [1]
  else:
    target_sizes = [2, 5]
  window_lengths = [3, 6] if num_times == 300 else [3]
  cands = spacetime.candidate_partitions(
      index,
      spatial_edges,
      target_sizes=target_sizes,
      window_lengths=window_lengths,
      seed=design_seed,
  )

  # Configure kappa routes per block
  kappa_s1d2 = None
  kappa_s1d3_adv = None

  if block == "ST1":
    alpha_s = cell["alpha_s"]
    rho_t = cell["rho_t"]
    theta_exact = (
        1.0 + 2.0 * alpha_s**2
        if metric == "mse"
        else math.sqrt(1.0 + 2.0 * alpha_s**2) * math.sqrt(2.0 / math.pi)
    )
    k_s, k_t, phi_s, phi_t = compute_st1_separable_vifs(
        num_nodes,
        num_times,
        alpha_s,
        rho_t,
        target_sizes,
        window_lengths,
        metric,
    )
    primary_kappa_fn = spacetime.separable_kappa(k_s, k_t)
    edges, _ = spacetime.spacetime_edges(
        index, spatial_edges, max_lag=1, phi_spatial=phi_s, phi_temporal=[phi_t]
    )
    choice = spacetime.choose_partition(
        cands,
        kappa_fn=primary_kappa_fn,
        m_item=m_item,
        level=level,
        is_complete_grid=is_complete_grid,
        edges=edges,
    )

  elif block == "ST2":
    w_s, w_t, w_wave = cell["weights"]
    sigma_x_sq = 1.0 + 2.0 * w_s**2 + 2.0 * w_t**2 + 4.0 * w_wave**2
    sigma_x = math.sqrt(sigma_x_sq)
    theta_exact = (
        sigma_x_sq if metric == "mse" else sigma_x * math.sqrt(2.0 / math.pi)
    )

    rho_sp = w_s**2 / sigma_x_sq
    rho_tm = w_t**2 / sigma_x_sq
    rho_cross = w_wave**2 / sigma_x_sq

    phi_s = rho_sp**2 if metric == "mse" else nabeya_mae_correlation(rho_sp)
    phi_t = rho_tm**2 if metric == "mse" else nabeya_mae_correlation(rho_tm)
    phi_cr = (
        rho_cross**2 if metric == "mse" else nabeya_mae_correlation(rho_cross)
    )

    edges, _ = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=1,
        phi_spatial=phi_s,
        phi_temporal=[phi_t],
        phi_cross=[phi_cr],
    )
    topo_fn = spacetime.topological_kappa_fn(index.n_items, edges, phi=_)
    choice = spacetime.choose_partition(
        cands,
        kappa_fn=topo_fn,
        m_item=m_item,
        level=level,
        is_complete_grid=is_complete_grid,
        edges=edges,
    )

  elif block == "ST3a":
    rho_s, rho_t = 0.35, 0.40
    theta_exact = 1.0 if metric == "mse" else math.sqrt(2.0 / math.pi)
    r_s = rho_s**2 if metric == "mse" else nabeya_mae_correlation(rho_s)
    r_t = rho_t**2 if metric == "mse" else nabeya_mae_correlation(rho_t)

    # 1-hop exact separable kappa
    k_s = max(1.0, 1.0 + 2.0 * r_s)
    k_t = max(1.0, 1.0 + 2.0 * r_t)
    sep_fn = spacetime.separable_kappa(k_s, k_t)

    edges, _ = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=1,
        phi_spatial=r_s,
        phi_temporal=[r_t],
        phi_cross=[r_s * r_t],
    )
    fallback_fn = spacetime.topological_kappa_fn(index.n_items, edges, phi=_)
    choice = spacetime.choose_partition(
        cands,
        kappa_fn=sep_fn,
        m_item=m_item,
        level=level,
        is_complete_grid=is_complete_grid,
        fallback_kappa_fn=fallback_fn,
        edges=edges,
    )

    # Claim S1D mask bounds
    counts_obs = np.bincount(
        choice.winner.cluster_ids, minlength=choice.winner.G
    )
    m_bar_obs = float(n_items) / float(choice.winner.G)
    kappa_s1d2 = float(k_s * k_t * (choice.winner.n_max / m_bar_obs))
    m_c_full = float(num_nodes * num_times) / float(choice.winner.G)
    kappa_s1d3_adv = float(k_s * k_t * (choice.winner.n_max / m_c_full))

  elif block == "ST3b":
    w_s, w_t, w_wave = (0.25, 0.25, 0.55)
    sigma_x_sq = 2.46
    sigma_x = math.sqrt(sigma_x_sq)
    theta_exact = (
        sigma_x_sq if metric == "mse" else sigma_x * math.sqrt(2.0 / math.pi)
    )

    rho_sp = w_s**2 / sigma_x_sq
    rho_tm = w_t**2 / sigma_x_sq
    rho_cross = w_wave**2 / sigma_x_sq

    phi_s = rho_sp**2 if metric == "mse" else nabeya_mae_correlation(rho_sp)
    phi_t = rho_tm**2 if metric == "mse" else nabeya_mae_correlation(rho_tm)
    phi_cr = (
        rho_cross**2 if metric == "mse" else nabeya_mae_correlation(rho_cross)
    )

    edges, _ = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=1,
        phi_spatial=phi_s,
        phi_temporal=[phi_t],
        phi_cross=[phi_cr],
    )
    topo_fn = spacetime.topological_kappa_fn(index.n_items, edges, phi=_)
    choice = spacetime.choose_partition(
        cands,
        kappa_fn=topo_fn,
        m_item=m_item,
        level=level,
        is_complete_grid=is_complete_grid,
        edges=edges,
    )

  elif block == "ST4":
    # Space-time R^2 with G_s=20, G_t=150
    w_s, w_t, w_wave = (0.25, 0.25, 0.55)
    sigma_x_sq = 2.46
    theta_exact = 0.75  # R^2 = 1 - 2.46 / 9.84 = 0.75

    phi_s = (w_s**2 / sigma_x_sq) ** 2
    phi_t = [(w_t**2 / sigma_x_sq) ** 2]
    phi_cr = [(w_wave**2 / sigma_x_sq) ** 2]

    edges, _ = spacetime.spacetime_edges(
        index,
        spatial_edges,
        max_lag=1,
        phi_spatial=phi_s,
        phi_temporal=phi_t,
        phi_cross=phi_cr,
    )
    topo_fn = spacetime.topological_kappa_fn(index.n_items, edges, phi=_)
    # Fixed partition G_s=20, G_t=150.
    comm_item = (obs_nodes // 2).astype(np.int64)
    win_item = (obs_times // 3).astype(np.int64)
    block_raw = comm_item * 150 + win_item
    _, block_dense = np.unique(block_raw, return_inverse=True)
    g = len(_)
    counts = np.bincount(block_dense, minlength=g)
    cand_st4 = spacetime.Candidate(
        name="fixed_partition_Gs20_Gt150",
        cluster_ids=block_dense.astype(np.int64),
        G=g,
        G_s=20,
        G_t=150,
        n_max=int(np.max(counts)),
        m_bar=float(n_items) / float(g),
        community_ids=comm_item,
        window_ids=win_item,
    )
    choice = spacetime.choose_partition(
        [cand_st4],
        kappa_fn=topo_fn,
        m_item=m_item,
        level=level,
        is_complete_grid=is_complete_grid,
        edges=edges,
    )

  elif block == "ST5":
    if cell["cell_id"] == 31:
      # ST5a: e ~ N(mu_t, 1), mu_t = 2.5 * t / T
      # True pooled MSE = 1 + mean_t mu_t^2, computed exactly over realized t grid
      t_grid = np.arange(num_times, dtype=np.float64)
      mu_t = 2.5 * (t_grid / float(num_times))
      theta_exact = float(1.0 + np.mean(mu_t**2))
    else:
      # ST5b: e_u,t = eps_u,t + 10 * t / T with non-product mask O.
      # Passed to spacetime_interval with metric="mse", which forms summands e_u,t^2.
      # True pooled MSE over O:
      # E[e_u,t^2] = 1 + (10 * t / T)^2
      # theta_exact = (1 / |O|) * sum_{(u,t) in O} [ 1 + (10 * t / T)^2 ]
      t_norm = obs_times.astype(np.float64) / float(num_times)
      theta_exact = float(1.0 + np.mean((10.0 * t_norm) ** 2))
    k_s, k_t = 1.0, 1.0
    sep_fn = spacetime.separable_kappa(k_s, k_t)
    edges, _ = spacetime.spacetime_edges(
        index, spatial_edges, max_lag=1, phi_spatial=0.0, phi_temporal=[0.0]
    )
    fallback_fn = spacetime.topological_kappa_fn(index.n_items, edges, phi=_)
    choice = spacetime.choose_partition(
        cands,
        kappa_fn=sep_fn,
        m_item=m_item,
        level=level,
        is_complete_grid=is_complete_grid,
        fallback_kappa_fn=fallback_fn,
        edges=edges,
    )
  else:
    raise ValueError(f"Unknown block: {block}")

  # 3. Monte Carlo Truth Verification (5,000 draws)
  rng_truth = np.random.default_rng(truth_seed)
  mc_reps = 50000
  if block == "ST1":
    e_mc = generate_st1_residuals(
        min(mc_reps, 5000),
        num_nodes,
        num_times,
        cell["alpha_s"],
        cell["rho_t"],
        rng_truth,
    )
    s_mc = e_mc**2 if metric == "mse" else np.abs(e_mc)
    theta_mc = float(np.mean(s_mc))
  elif block in ("ST2", "ST3b"):
    w_s, w_t, w_wave = cell.get("weights", (0.25, 0.25, 0.55))
    x_mc = generate_st2_wave(
        min(mc_reps, 5000), num_nodes, num_times, w_s, w_t, w_wave, rng_truth
    )
    s_mc = x_mc**2 if metric == "mse" else np.abs(x_mc)
    # Mask to observed
    s_mc_obs = s_mc[:, obs_mask]
    theta_mc = float(np.mean(s_mc_obs))
  elif block == "ST3a":
    e_mc = generate_st3a_1hop(
        min(mc_reps, 5000), num_nodes, num_times, 0.35, 0.40, rng_truth
    )
    s_mc = e_mc**2 if metric == "mse" else np.abs(e_mc)
    s_mc_obs = s_mc[:, obs_mask]
    theta_mc = float(np.mean(s_mc_obs))
  elif block == "ST4":
    theta_mc = 0.75
  else:
    theta_mc = theta_exact

  # 4. Monte Carlo Evaluation over `reps` draws
  eval_rng = np.random.default_rng(design_seed + 1)
  batch_size = 500
  num_batches = (reps + batch_size - 1) // batch_size

  # Misses are counted over every repetition that returned a finite interval,
  # whatever its status (P(miss) <= alpha contract). `unrefuted_miss` counts
  # the joint event: miss, status UNREFUTED and no marginal check REFUTED.
  low_miss = 0
  high_miss = 0
  unrefuted_miss = 0
  issued_count = 0
  num_unrefuted = 0
  num_tail_unresolved = 0
  num_assumption_required = 0
  rel_widths = []

  temp_refuted = 0
  spat_refuted = 0
  drift_warnings = 0

  # For ST5b: grand-mean comparison
  gm_refuted_count = 0

  winner = choice.winner

  for b in range(num_batches):
    cur_b_size = min(batch_size, reps - b * batch_size)
    items_batch = np.zeros((cur_b_size, n_items), dtype=np.float64)
    y_items = np.zeros((cur_b_size, n_items), dtype=np.float64)
    y_pred_b = np.zeros((cur_b_size, num_nodes, num_times), dtype=np.float64)

    # Draw batch of residuals
    if block == "ST1":
      e_batch = generate_st1_residuals(
          cur_b_size,
          num_nodes,
          num_times,
          cell["alpha_s"],
          cell["rho_t"],
          eval_rng,
      )
      items_batch = e_batch[:, obs_mask]
    elif block == "ST2":
      w_s, w_t, w_wave = cell["weights"]
      x_batch = generate_st2_wave(
          cur_b_size, num_nodes, num_times, w_s, w_t, w_wave, eval_rng
      )
      items_batch = x_batch[:, obs_mask]
    elif block == "ST3a":
      e_batch = generate_st3a_1hop(
          cur_b_size, num_nodes, num_times, 0.35, 0.40, eval_rng
      )
      items_batch = e_batch[:, obs_mask]
    elif block == "ST3b":
      x_batch = generate_st2_wave(
          cur_b_size, num_nodes, num_times, 0.25, 0.25, 0.55, eval_rng
      )
      items_batch = x_batch[:, obs_mask]
    elif block == "ST4":
      # R^2: e and f
      e_b = generate_st2_wave(
          cur_b_size, num_nodes, num_times, 0.25, 0.25, 0.55, eval_rng
      )
      f_b = math.sqrt(3.0) * generate_st2_wave(
          cur_b_size, num_nodes, num_times, 0.25, 0.25, 0.55, eval_rng
      )
      y_b = f_b + e_b
      y_pred_b = f_b
      y_items = y_b[:, obs_mask]
      items_batch = e_b[:, obs_mask]
    elif block == "ST5":
      if cell["cell_id"] == 31:
        # ST5a: drift mu_t = 2.5 * t / T
        t_ramp = 2.5 * (
            np.arange(num_times, dtype=np.float64) / float(num_times)
        )
        e_b = (
            eval_rng.standard_normal((cur_b_size, num_nodes, num_times))
            + t_ramp[None, None, :]
        )
        items_batch = e_b[:, obs_mask]
      else:
        # ST5b: drift 10.0 * t / T with non-product missingness
        t_ramp = 10.0 * (
            np.arange(num_times, dtype=np.float64) / float(num_times)
        )
        e_b = (
            eval_rng.standard_normal((cur_b_size, num_nodes, num_times))
            + t_ramp[None, None, :]
        )
        items_batch = e_b[:, obs_mask]

    for i in range(cur_b_size):
      if block == "ST4":
        s_i = (y_items[i], y_pred_b[:, obs_mask][i])
        res = spacetime.spacetime_interval(
            s_i,
            index,
            choice,
            metric="r2",
            level=level,
            m_item=m_item,
            m_labels=3.2,
            kappa_labels=choice.winner_kappa,
        )
      else:
        s_i = items_batch[i]
        res = spacetime.spacetime_interval(
            s_i,
            index,
            choice,
            metric=metric,
            level=level,
            m_item=m_item,
        )

      status_name = res.status.name
      if status_name == "UNREFUTED":
        num_unrefuted += 1
      elif status_name == "TAIL_UNRESOLVED":
        num_tail_unresolved += 1
      elif status_name == "ASSUMPTION_REQUIRED":
        num_assumption_required += 1

      marginal_refuted = False
      if res.marginal_checks is not None:
        if res.marginal_checks.temporal is not None:
          if (
              res.marginal_checks.temporal.outcome
              == _refutation.Refutation.REFUTED
          ):
            temp_refuted += 1
            marginal_refuted = True
        if res.marginal_checks.spatial is not None:
          if (
              res.marginal_checks.spatial.outcome
              == _refutation.Refutation.REFUTED
          ):
            spat_refuted += 1
            marginal_refuted = True

      if (
          status_name != "ASSUMPTION_REQUIRED"
          and math.isfinite(res.low)
          and math.isfinite(res.high)
      ):
        issued_count += 1
        width = res.high - res.low
        rel_widths.append(
            width / abs(theta_exact) if abs(theta_exact) > 0 else width
        )
        missed = False
        if theta_exact < res.low:
          low_miss += 1
          missed = True
        elif theta_exact > res.high:
          high_miss += 1
          missed = True
        if missed and status_name == "UNREFUTED" and not marginal_refuted:
          unrefuted_miss += 1

      if res.drift is not None:
        if res.drift.outcome == _temporal.Drift.WARNING:
          drift_warnings += 1

      # ST5b: per-window vs grand-mean centring in the refutation check.
      if block == "ST5" and cell["cell_id"] == 32:
        s_arr = np.asarray(s_i, dtype=np.float64)
        s_eval = s_arr**2 if metric == "mse" else np.abs(s_arr)
        # 1. Per-window centring
        s_w = np.bincount(
            winner.window_ids, weights=s_eval, minlength=winner.G_t
        )
        n_w = np.bincount(winner.window_ids, minlength=winner.G_t)
        w_means = np.where(n_w > 0, s_w / n_w, 0.0)
        e_pw = s_eval - w_means[winner.window_ids]
        e_comm_pw = np.bincount(
            winner.community_ids, weights=e_pw, minlength=winner.G_s
        )
        n_c = np.bincount(winner.community_ids, minlength=winner.G_s)

        # 2. Grand-mean centring
        gm = float(np.mean(s_eval))
        s_c = np.bincount(
            winner.community_ids, weights=s_eval, minlength=winner.G_s
        )
        e_comm_gm = s_c - n_c * gm

        # Community edges mapped from spatial ring
        node_to_comm = np.zeros(num_nodes, dtype=np.int64)
        node_to_comm[obs_nodes] = winner.community_ids
        c_u = node_to_comm[spatial_edges[:, 0]]
        c_v = node_to_comm[spatial_edges[:, 1]]
        cross = c_u != c_v
        raw_e = np.column_stack([c_u[cross], c_v[cross]])
        comm_edges = np.unique(
            np.column_stack([
                np.minimum(raw_e[:, 0], raw_e[:, 1]),
                np.maximum(raw_e[:, 0], raw_e[:, 1]),
            ]),
            axis=0,
        ).astype(np.int64)

        pw_chk = _refutation.check_uncorrelated_edges(
            e_comm_pw, n_c, comm_edges, level=0.95
        )
        if pw_chk.outcome == _refutation.Refutation.REFUTED:
          spat_refuted += 1

        gm_chk = _refutation.check_uncorrelated_edges(
            e_comm_gm, n_c, comm_edges, level=0.95
        )
        if gm_chk.outcome == _refutation.Refutation.REFUTED:
          gm_refuted_count += 1

  # Compute summary statistics
  issuance_rate = issued_count / float(reps)
  refusal_rate = 1.0 - issuance_rate
  total_miss = low_miss + high_miss
  coverage_rate = 1.0 - (total_miss / float(reps))
  low_miss_rate = low_miss / float(reps)
  high_miss_rate = high_miss / float(reps)

  cp_cov_low, cp_cov_high = clopper_pearson(reps - total_miss, reps)
  low_miss_cp = one_sided_cp_lower(low_miss, reps)
  high_miss_cp = one_sided_cp_lower(high_miss, reps)

  mean_rel_w = float(np.mean(rel_widths)) if rel_widths else 0.0
  median_rel_w = float(np.median(rel_widths)) if rel_widths else 0.0

  temp_ref_rate = temp_refuted / float(reps)
  spat_ref_rate = spat_refuted / float(reps)
  drift_warn_rate = drift_warnings / float(reps)

  q_crit = q_threshold(reps, alpha)

  # Pass criteria evaluation
  if block in ("ST1", "ST2", "ST3a", "ST3b", "ST4"):
    pass_cell = (
        low_miss <= q_crit
        and high_miss <= q_crit
        and issuance_rate >= 0.95
        and (is_complete_grid or refusal_rate == 0.0)
    )
  elif block == "ST5" and cell["cell_id"] == 31:
    pass_cell = (drift_warn_rate >= 0.99) and (spat_ref_rate <= 0.075)
  elif block == "ST5" and cell["cell_id"] == 32:
    gm_ref_rate = gm_refuted_count / float(reps)
    pass_cell = (spat_ref_rate <= 0.075) and (gm_ref_rate >= 0.90)
  else:
    pass_cell = True

  record = {
      "block": block,
      "cell_id": cell_id,
      "cell_name": name,
      "N_s": num_nodes,
      "T": num_times,
      "n_items": n_items,
      "missing_rate": missing_rate,
      "missing_pattern": missing_pattern,
      "is_complete_grid": is_complete_grid,
      "metric": metric,
      "level": level,
      "reps": reps,
      "m_item": m_item,
      "route": choice.route,
      "winner_name": winner.name,
      "G": winner.G,
      "G_s": winner.G_s,
      "G_t": winner.G_t,
      "winner_kappa": choice.winner_kappa,
      "winner_r_eff": choice.winner_r_eff,
      "winner_n_eff": choice.winner_n_eff,
      "kappa_s1d2": kappa_s1d2,
      "kappa_s1d3_adv": kappa_s1d3_adv,
      "theta_exact": float(theta_exact),
      "theta_mc": float(theta_mc),
      "issuance_rate": float(issuance_rate),
      "refusal_rate": float(refusal_rate),
      "low_miss_count": int(low_miss),
      "high_miss_count": int(high_miss),
      "total_miss_count": int(total_miss),
      "miss_rate": float(total_miss / float(reps)),
      "unrefuted_miss_count": int(unrefuted_miss),
      "unrefuted_miss_rate": float(unrefuted_miss / float(reps)),
      "num_unrefuted": int(num_unrefuted),
      "num_tail_unresolved": int(num_tail_unresolved),
      "num_assumption_required": int(num_assumption_required),
      "low_miss_rate": float(low_miss_rate),
      "high_miss_rate": float(high_miss_rate),
      "coverage_rate": float(coverage_rate),
      "coverage_cp_95_low": float(cp_cov_low),
      "coverage_cp_95_high": float(cp_cov_high),
      "low_miss_cp_999_low": float(low_miss_cp),
      "high_miss_cp_999_low": float(high_miss_cp),
      "mean_rel_width": float(mean_rel_w),
      "median_rel_width": float(median_rel_w),
      "temporal_marginal_refuted_rate": float(temp_ref_rate),
      "spatial_marginal_refuted_rate": float(spat_ref_rate),
      "grand_mean_spatial_refuted_rate": float(
          gm_refuted_count / float(reps)
      ),
      "drift_warning_rate": float(drift_warn_rate),
      "pass_cell": bool(pass_cell),
  }
  return record


def get_all_cells() -> list[dict[str, Any]]:
  """Constructs definitions for all 32 validation cells."""
  cells = []
  cell_id = 1

  # Block ST1: Synthetic Separable on Complete Grid (Cells 1-8)
  st1_params = [
      (0.20, 0.30),
      (0.20, 0.60),
      (0.40, 0.30),
      (0.40, 0.60),
  ]
  for alpha_s, rho_t in st1_params:
    for met in ["mse", "mae"]:
      cells.append({
          "cell_id": cell_id,
          "block": "ST1",
          "name": f"ST1_alpha{alpha_s}_rho{rho_t}_{met}",
          "alpha_s": alpha_s,
          "rho_t": rho_t,
          "metric": met,
          "reps": 5000,
          "missing_rate": 0.0,
          "missing_pattern": "complete",
      })
      cell_id += 1

  # Block ST2: Non-Separable Propagating Wave (Cells 9-12)
  st2_regimes = [
      ("regimeA", (0.25, 0.25, 0.55)),
      ("regimeB", (0.50, 0.50, 0.90)),
  ]
  for reg_name, weights in st2_regimes:
    for met in ["mse", "mae"]:
      cells.append({
          "cell_id": cell_id,
          "block": "ST2",
          "name": f"ST2_{reg_name}_{met}",
          "weights": weights,
          "metric": met,
          "reps": 5000,
          "missing_rate": 0.0,
          "missing_pattern": "complete",
      })
      cell_id += 1

  # Block ST3a: Separable 1-hop on Incomplete Grids (Cells 13-20)
  st3_masks = [
      (0.10, "mcar"),
      (0.30, "mcar"),
      (0.10, "block_outage"),
      (0.30, "block_outage"),
  ]
  for mr, pat in st3_masks:
    for met in ["mse", "mae"]:
      cells.append({
          "cell_id": cell_id,
          "block": "ST3a",
          "name": f"ST3a_{pat}_{int(mr*100)}_{met}",
          "missing_rate": mr,
          "missing_pattern": pat,
          "metric": met,
          "reps": 5000,
      })
      cell_id += 1

  # Block ST3b: Non-Separable Wave on Incomplete Grids (Cells 21-28)
  for mr, pat in st3_masks:
    for met in ["mse", "mae"]:
      cells.append({
          "cell_id": cell_id,
          "block": "ST3b",
          "name": f"ST3b_{pat}_{int(mr*100)}_{met}",
          "weights": (0.25, 0.25, 0.55),
          "missing_rate": mr,
          "missing_pattern": pat,
          "metric": met,
          "reps": 5000,
      })
      cell_id += 1

  # Block ST4: Space-Time R^2 (Cells 29-30)
  cells.append({
      "cell_id": cell_id,
      "block": "ST4",
      "name": "ST4_r2_complete",
      "metric": "r2",
      "T": 450,
      "reps": 5000,
      "missing_rate": 0.0,
      "missing_pattern": "complete",
  })
  cell_id += 1
  cells.append({
      "cell_id": cell_id,
      "block": "ST4",
      "name": "ST4_r2_mcar20",
      "metric": "r2",
      "T": 450,
      "reps": 5000,
      "missing_rate": 0.20,
      "missing_pattern": "mcar",
  })
  cell_id += 1

  # Block ST5: Drift Power & Per-Window Invariance (Cells 31-32)
  cells.append({
      "cell_id": cell_id,
      "block": "ST5",
      "name": "ST5a_drift_power",
      "metric": "mse",
      "reps": 2000,
      "m_item": 3.5,
      "missing_rate": 0.0,
      "missing_pattern": "complete",
  })
  cell_id += 1
  cells.append({
      "cell_id": cell_id,
      "block": "ST5",
      "name": "ST5b_non_product_missingness_invariance",
      "metric": "mse",
      "reps": 2000,
      "missing_rate": 0.20,
      "missing_pattern": "non_product_drift",
  })
  cell_id += 1

  return cells


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  mode = str(_MODE.value)
  all_cells = get_all_cells()
  for c in all_cells:
    c["seed_base"] = int(_SEED.value)

  target_block = _BLOCK.value
  target_cell = _CELL.value
  if target_cell > 0:
    cells_to_run = [c for c in all_cells if c["cell_id"] == target_cell]
  elif target_block != "all":
    cells_to_run = [c for c in all_cells if c["block"] == target_block]
  else:
    cells_to_run = all_cells

  if mode == "quick":
    # 1 cell per block: ST1, ST2, ST3a, ST3b, ST4, ST5 (6 cells)
    blocks_seen = set()
    selected = []
    for c in cells_to_run:
      if c["block"] not in blocks_seen:
        blocks_seen.add(c["block"])
        selected.append(c)
    cells_to_run = selected
    reps_override = 20
  else:
    reps_override = _REPS.value

  output_path = _OUTPUT.value
  num_workers = min(_WORKERS.value, len(cells_to_run))

  print(
      f"=== Space-Time Validation Benchmark (mode={mode},"
      f" workers={num_workers}, cells={len(cells_to_run)}) ===",
      flush=True,
  )

  results: list[dict[str, Any]] = []
  t_start = time.perf_counter()

  if num_workers > 1 and len(cells_to_run) > 1:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:
      future_to_cell = {
          executor.submit(
              run_cell_simulation, cell, reps_override
          ): cell
          for cell in cells_to_run
      }
      for future in concurrent.futures.as_completed(future_to_cell):
        cell = future_to_cell[future]
        res = future.result()
        results.append(res)
        cov_str = (
            f"cov={res['coverage_rate']:.4f} [miss L={res['low_miss_count']},"
            f" H={res['high_miss_count']}, unref_miss={res['unrefuted_miss_count']}]"
        )
        pass_str = "PASS" if res["pass_cell"] else "FAIL"
        print(
            f"[{pass_str}] Cell {res['cell_id']:02d} ({res['cell_name']}): "
            f"{cov_str}, rel_w={res['mean_rel_width']:.4f},"
            f" route={res['route']} ({len(results)}/{len(cells_to_run)})",
            flush=True,
        )
  else:
    for cell in cells_to_run:
      c_t0 = time.perf_counter()
      res = run_cell_simulation(
          cell, reps_override=reps_override
      )
      c_time = time.perf_counter() - c_t0
      results.append(res)
      cov_str = (
          f"cov={res['coverage_rate']:.4f} [miss L={res['low_miss_count']},"
          f" H={res['high_miss_count']}, unref_miss={res['unrefuted_miss_count']}]"
      )
      pass_str = "PASS" if res["pass_cell"] else "FAIL"
      print(
          f"[{pass_str}] Cell {res['cell_id']:02d} ({res['cell_name']}): "
          f"{cov_str}, rel_w={res['mean_rel_width']:.4f},"
          f" route={res['route']} ({c_time:.2f}s)",
          flush=True,
      )

  results.sort(key=lambda r: r["cell_id"])

  if output_path:
    out_dir = os.path.dirname(output_path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
      for r in results:
        f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} validation records to {output_path}", flush=True)

  total_time = time.perf_counter() - t_start

  # Summary table
  print("\n" + "=" * 125, flush=True)
  print(
      f"SUMMARY RESULTS TABLE (mode={mode}, total elapsed: {total_time:.1f}s)",
      flush=True,
  )
  print("=" * 125, flush=True)
  print(
      f"{'Cell Name':<42} {'Block':<6} {'Reps':<6} {'Coverage':<12} {'Rel Width':<12} {'Route':<10} {'Passed':<8}",
      flush=True,
  )
  print("-" * 125, flush=True)
  for r in results:
    cov_str = f"{r.get('coverage_rate', 0.0):.2%}"
    w_str = f"{r.get('mean_rel_width', 0.0):.4f}"
    pass_str = "PASS" if r.get("pass_cell", False) else "FAIL"
    reps_cnt = r.get("reps", 0)
    print(
        f"{r['cell_name']:<42} {r.get('block', ''):<6} {reps_cnt:<6d} {cov_str:<12} {w_str:<12} {r.get('route', ''):<10} {pass_str:<8}",
        flush=True,
    )
  print("-" * 125, flush=True)


if __name__ == "__main__":
  app.run(main)
