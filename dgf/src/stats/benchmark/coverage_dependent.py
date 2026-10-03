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

"""Coverage and refutation benchmark for the temporal and graph intervals.

Draws 136 cells in six blocks (T1: AR(1) fields, T2: bucketed counts, T2b:
count-guard blind spot, T3: variance drift, G1: synthetic graphs, F: invalid
inputs) and evaluates each in the temporal (`t_data`, `t_shard`, `t_merged`)
or graph (`g_data`, `g_merged`) modes. Misses are counted over all
repetitions.
"""

from collections.abc import Sequence
import concurrent.futures
import contextlib
import json
import math
import os
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import scipy.signal
import scipy.spatial
import scipy.stats

from dgf.src.stats import cluster_bound
from dgf.src.stats import cluster_sketch
from dgf.src.stats import graph
from dgf.src.stats import partitioners
from dgf.src.stats import refutation
from dgf.src.stats import temporal
from dgf.src.stats.benchmark import coverage_dependent_truth as truth
from dgf.src.stats.independent import interval as _interval

DEFAULT_SEED = 54077361271078  # = 0x312edde8b126

_SEED = (
    flags.FLAGS["seed"]
    if "seed" in flags.FLAGS
    else flags.DEFINE_integer(
        "seed", DEFAULT_SEED, "Seed for benchmark."
    )
)
_MODE = (
    flags.FLAGS["mode"]
    if "mode" in flags.FLAGS
    else flags.DEFINE_enum(
        "mode", "quick", ["quick", "full"], "Execution mode: 'quick' or 'full'."
    )
)
_OUTPUT = (
    flags.FLAGS["output"]
    if "output" in flags.FLAGS
    else flags.DEFINE_string(
        "output", None, "Optional output path for newline-delimited JSON results."
    )
)
_WORKERS = (
    flags.FLAGS["workers"]
    if "workers" in flags.FLAGS
    else flags.DEFINE_integer(
        "workers", min(16, os.cpu_count() or 4), "Number of parallel workers."
    )
)



# ---------------------------------------------------------------------------
# Cell Generation and Specifications
# ---------------------------------------------------------------------------


def _build_geometric_edges(
    g: int, mean_deg: float, seed: int = 42
) -> np.ndarray:
  """Builds 2-D random geometric graph edges on unit square."""
  rng = np.random.default_rng(seed)
  pos = rng.uniform(0.0, 1.0, size=(g, 2))
  r = math.sqrt(mean_deg / (math.pi * float(g)))
  tree = scipy.spatial.KDTree(pos)
  pairs = list(tree.query_pairs(r))
  if not pairs:
    return np.zeros((0, 2), dtype=np.int64)
  edges = np.array(pairs, dtype=np.int64)
  u = np.minimum(edges[:, 0], edges[:, 1])
  v = np.maximum(edges[:, 0], edges[:, 1])
  return np.column_stack([u, v])


def _build_ba_edges(n: int, m: int, seed: int = 42) -> np.ndarray:
  """Builds Barabasi-Albert preferential attachment graph edges."""
  rng = np.random.default_rng(seed)
  ends = np.empty(2 * n * m + 2 * m, dtype=np.int64)
  ends[:m] = np.arange(m)
  cnt = m
  out = []
  for i in range(m, n):
    t = ends[rng.integers(0, cnt, m)]
    out.append(np.column_stack([np.full(m, i), t]))
    ends[cnt : cnt + m] = t
    ends[cnt + m : cnt + 2 * m] = i
    cnt += 2 * m
  all_edges = np.concatenate(out)
  u = np.minimum(all_edges[:, 0], all_edges[:, 1])
  v = np.maximum(all_edges[:, 0], all_edges[:, 1])
  non_self = u != v
  unique_keys = np.unique(u[non_self] * n + v[non_self])
  return np.column_stack([unique_keys // n, unique_keys % n])


def _build_hub_edges(
    n: int, base_edges: np.ndarray, seed: int = 301
) -> np.ndarray:
  """Builds hub graph: base geometric edges plus 20 stars of 300 leaves."""
  hub_edges_list = [base_edges]
  rng_hubs = np.random.default_rng(seed)
  centers = rng_hubs.choice(n, size=20, replace=False)
  for hub in centers:
    others = np.delete(np.arange(n), hub)
    leaves = rng_hubs.choice(others, size=300, replace=False)
    h_min = np.minimum(hub, leaves)
    h_max = np.maximum(hub, leaves)
    hub_edges_list.append(np.column_stack([h_min, h_max]))
  all_hub_edges = np.vstack(hub_edges_list)
  pair_keys = np.unique(all_hub_edges[:, 0] * n + all_hub_edges[:, 1])
  return np.column_stack([pair_keys // n, pair_keys % n]).astype(np.int64)


def generate_graph_structure(
    cell: dict[str, Any], base_seed: int
) -> tuple[np.ndarray, int]:
  """Draws and partitions the graph for a cell, returning (edges, n)."""
  c_id = cell["cell_id"]
  b_id = cell["block"]
  rng = np.random.default_rng([base_seed, c_id, 2**20])

  if b_id == "G1":
    fam = cell["graph_family"]
    n = cell["n"]
    if fam == "rgg":
      mean_deg = float(rng.uniform(6.0, 20.0))
      edges = _build_geometric_edges(
          n, mean_deg=mean_deg, seed=int(rng.integers(1, 1000000))
      )
    elif fam == "ba":
      m = int(cell.get("ba_m") or rng.choice([2, 5]))
      edges = _build_ba_edges(n, m=m, seed=int(rng.integers(1, 1000000)))
    elif fam == "sbm":
      block_size = 500
      edges_list = []
      for b in range(40):
        b_nodes = np.arange(b * block_size, (b + 1) * block_size)
        e_in = _build_geometric_edges(
            block_size, mean_deg=8.0, seed=int(rng.integers(1, 1000000))
        )
        edges_list.append(b_nodes[e_in])
      edges = np.vstack(edges_list)
    elif fam == "isolated":
      n_active = int(round(0.65 * n))
      e_act = _build_geometric_edges(
          n_active, mean_deg=10.0, seed=int(rng.integers(1, 1000000))
      )
      edges = e_act
    elif fam == "hub":
      base_e = _build_geometric_edges(
          n, mean_deg=6.0, seed=int(rng.integers(1, 1000000))
      )
      edges = _build_hub_edges(n, base_e, seed=int(rng.integers(1, 1000000)))
    else:
      edges = _build_geometric_edges(n, mean_deg=6.0, seed=123)

  elif b_id == "F":
    n = cell["n"]
    edges = _build_geometric_edges(
        n, mean_deg=6.0, seed=int(rng.integers(1, 1000000))
    )
  else:
    edges = np.zeros((0, 2), dtype=np.int64)
    n = cell.get("n", 0)

  return edges, n


def check_cell_refusal(
    cell: dict[str, Any],
    gt: dict[str, Any] | None = None,
    graph_data: tuple[Any, ...] | None = None,
) -> bool:
  """Evaluates whether cell declarations cause the library to refuse on constant shard.

  For Block G1, this performs a structural pre-check on an equal-slice
  line-graph surrogate; gt=None.
  """
  b_id = cell["block"]
  metric = cell["metric"]
  alpha = cell["alpha"]
  level = 1.0 - alpha
  m_mode = cell.get("m_mode", "default")
  k_mode = cell.get("kappa_mode", "one")
  scale = cell.get("scale", 1.0)
  p = cell.get("p", 0.5)

  theta = (
      float(gt["theta"])
      if gt is not None and "theta" in gt
      else float(truth.item_mean(metric, scale, p))
  )

  if b_id in ("T1", "T2", "T2b", "T3", "T4"):
    n = int(cell.get("n") or cell.get("n_target") or 20000)
    G = int(cell["G"])
    counts = np.bincount(
        (np.arange(n, dtype=np.int64) * G) // n, minlength=G
    ).astype(np.int64)
    totals = counts.astype(np.float64) * theta
    shard = cluster_sketch.PartialClusterShard(
        cluster_ids=np.arange(G, dtype=np.int64),
        cluster_totals=totals,
        cluster_counts=counts,
    )
    if b_id == "T3":
      m_val = 2.0
    else:
      if m_mode == "default":
        m_val = None
      elif gt is not None and "m_c_true" in gt:
        m_val = float(gt["m_c_true"])
      else:
        m_val = float(G * np.sum(counts.astype(np.float64) ** 2) / (n**2))
    vif = gt.get("vif_true", 1.0) if gt is not None else 1.0
    k_val = (
        1.0
        if k_mode == "one"
        else max(1.0, vif if k_mode == "exact" else vif / 2.0)
    )
    res = temporal.temporal_interval_from_shard(
        shard, metric=metric, level=level, m=m_val, kappa=k_val
    )
  elif b_id in ("G1", "G2"):
    if graph_data is None:
      return False
    edges, n_nodes, c_ids, c_edges = graph_data
    counts = np.bincount(c_ids).astype(np.int64)
    totals = counts.astype(np.float64) * theta
    shard = cluster_sketch.PartialClusterShard(
        cluster_ids=np.arange(len(counts), dtype=np.int64),
        cluster_totals=totals,
        cluster_counts=counts,
    )
    if m_mode == "default":
      m_val = None
    elif gt is not None and "m_c_true" in gt:
      m_val = float(gt["m_c_true"])
    else:
      g_cnt = len(counts)
      m_val = float(
          g_cnt * np.sum(counts.astype(np.float64) ** 2) / (float(n_nodes) ** 2)
      )
    vif = gt.get("vif_true", 1.0) if gt is not None else 1.0
    k_val = (
        1.0
        if k_mode == "one"
        else max(1.0, vif if k_mode == "exact" else vif / 2.0)
    )
    res = graph.graph_interval_from_shard(
        shard, c_edges, metric=metric, level=level, m=m_val, kappa=k_val
    )
  else:
    return False

  int_obj = getattr(res, "interval", res)
  status = getattr(int_obj, "status", None)
  if status is None:
    return True
  status_name = getattr(status, "name", str(status))
  if status_name not in ("UNREFUTED", "TAIL_UNRESOLVED"):
    return True
  low = getattr(int_obj, "low", None)
  high = getattr(int_obj, "high", None)
  if low is None or high is None:
    return True
  try:
    low_f = float(low)
    high_f = float(high)
  except (TypeError, ValueError):
    return True
  if not (math.isfinite(low_f) and math.isfinite(high_f)):
    return True
  return False


def calibrate_t2_a(
    base_seed: int,
    cell_id: int,
    target_h: float,
    n_target: int = 20000,
    G: int = 100,
    draws: int = 2000,
) -> tuple[float, float]:
  """Calibrates parameter 'a' on [0, 50] via bisection using 2000 CRN draws until |E[h] - target| <= 0.002."""
  rng = np.random.default_rng([base_seed, cell_id, 2**21])
  Lambda = 40000.0
  num_pts = rng.poisson(Lambda, size=draws)
  total_pts = int(np.sum(num_pts))
  all_t = rng.uniform(0.0, 1.0, size=total_pts)
  all_u = rng.uniform(0.0, 1.0, size=total_pts)
  draw_idx = np.repeat(np.arange(draws, dtype=np.int64), num_pts)
  bucket_ids = np.clip((all_t * G).astype(np.int64), 0, G - 1)

  def eval_a(a_val: float) -> float:
    prob = (
        (float(n_target) / Lambda) * (1.0 + a_val * all_t) / (1.0 + 0.5 * a_val)
    )
    mask = all_u < prob
    flat_idx = draw_idx[mask] * G + bucket_ids[mask]
    counts_2d = (
        np.bincount(flat_idx, minlength=draws * G)
        .reshape((draws, G))
        .astype(np.float64)
    )
    num1 = np.mean(counts_2d[:, :-1] * counts_2d[:, 1:], axis=1)
    den1 = np.mean(counts_2d, axis=1) ** 2
    h1 = num1 / np.maximum(den1, 1e-12)

    c2 = counts_2d**2
    num2 = np.mean(c2[:, :-1] * c2[:, 1:], axis=1)
    den2 = np.mean(c2, axis=1) ** 2
    h2 = num2 / np.maximum(den2, 1e-12)

    h_per_draw = np.maximum(h1, h2)
    return float(np.mean(h_per_draw))

  if target_h <= 0.0:
    return 0.0, 1.0

  lo = 0.0
  hi = 50.0
  best_a = 0.0
  best_h = 1.0
  for _ in range(20):
    mid = (lo + hi) / 2.0
    h_mid = eval_a(mid)
    best_a = mid
    best_h = h_mid
    if abs(h_mid - target_h) <= 0.002:
      return best_a, best_h
    if h_mid < target_h:
      lo = mid
    else:
      hi = mid

  if abs(best_h - target_h) > 0.005:
    raise RuntimeError(
        f"Could not calibrate T2 'a' to target_h={target_h}: best_a={best_a},"
        f" achieved={best_h}"
    )
  return best_a, best_h


def generate_t1_fixed_cells(start_id: int = 0) -> list[dict[str, Any]]:
  """Generates the 8 deterministic benchmark T1 cells."""
  cells = []
  global_cell_id = start_id
  for k_mode in ("exact", "one"):
    for m_mode in ("default", "tight"):
      cell = {
          "cell_id": global_cell_id,
          "block": "T1",
          "family": "temporal_ar1",
          "name": f"T1_thin_k={k_mode}_m={m_mode}",
          "n": 20000,
          "L": 1000,
          "G": 100,
          "metric": "mse",
          "alpha": 0.05,
          "m_mode": m_mode,
          "kappa_mode": k_mode,
          "scale": 1.0,
          "modes": ["t_data", "t_shard", "t_merged"],
      }
      counts = np.bincount(
          (np.arange(cell["n"], dtype=np.int64) * cell["G"]) // cell["n"],
          minlength=cell["G"],
      ).astype(np.int64)
      phi = 1.0 - 1.0 / float(cell["L"])
      gt = truth.temporal_ground_truth("mse", counts, phi, sigma=1.0)
      if check_cell_refusal(cell, gt):
        cell["refused_at_precheck"] = True
      cells.append(cell)
      global_cell_id += 1

  for p in (0.99, 0.999):
    for L in (1, 4):
      cell = {
          "cell_id": global_cell_id,
          "block": "T1",
          "family": "temporal_ar1",
          "name": f"T1_high_acc_p={p}_L={L}",
          "n": 20000,
          "L": L,
          "G": 200,
          "metric": "accuracy",
          "p": p,
          "alpha": 0.05,
          "m_mode": "default",
          "kappa_mode": "one",
          "scale": 1.0,
          "modes": ["t_data", "t_shard", "t_merged"],
      }
      counts = np.bincount(
          (np.arange(cell["n"], dtype=np.int64) * cell["G"]) // cell["n"],
          minlength=cell["G"],
      ).astype(np.int64)
      phi = 0.0 if L == 1 else 1.0 - 1.0 / float(L)
      gt = truth.temporal_ground_truth("accuracy", counts, phi, sigma=1.0, p=p)
      if check_cell_refusal(cell, gt):
        cell["refused_at_precheck"] = True
      cells.append(cell)
      global_cell_id += 1
  return cells


def generate_all_cells(
    base_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
  """Generates all 136 cells in a fixed block order (T1, T2, T2b, T3, G1, F)."""
  spec_rng = np.random.default_rng(base_seed)
  cells: list[dict[str, Any]] = []
  redraw_counts: dict[str, int] = {}
  global_cell_id = 0

  def draw_declarations():
    alpha = float(spec_rng.choice([0.05, 0.01], p=[0.7, 0.3]))
    m_mode = str(spec_rng.choice(["default", "tight"]))
    kappa_mode = str(
        spec_rng.choice(["exact", "one", "half"], p=[0.5, 0.3, 0.2])
    )
    scale = float(math.exp(spec_rng.uniform(math.log(1e-3), math.log(1e3))))
    return alpha, m_mode, kappa_mode, scale

  # --- Block T1: AR(1) fields (56 cells) ---
  redraw_counts["T1"] = 0
  t1_fixed = generate_t1_fixed_cells(global_cell_id)
  cells.extend(t1_fixed)
  global_cell_id += len(t1_fixed)

  for i in range(48):
    cand_cell = None
    for attempt in range(100):
      n = int(spec_rng.choice([2000, 5000, 20000, 50000]))
      L = int(spec_rng.choice([1, 10, 100, 1000]))
      valid_g = [g for g in [50, 100, 200, 500] if g <= n // 10]
      G = int(spec_rng.choice(valid_g))
      metric = str(spec_rng.choice(["mse", "rmse", "mae", "accuracy"]))
      p = float(spec_rng.uniform(0.6, 0.999)) if metric == "accuracy" else 0.5
      alpha, m_mode, kappa_mode, scale = draw_declarations()

      cand_cell = {
          "cell_id": global_cell_id,
          "block": "T1",
          "family": "temporal_ar1",
          "name": f"T1_rand_{i}",
          "n": n,
          "L": L,
          "G": G,
          "metric": metric,
          "p": p,
          "alpha": alpha,
          "m_mode": m_mode,
          "kappa_mode": kappa_mode,
          "scale": scale,
          "modes": ["t_data", "t_shard", "t_merged"],
      }
      counts = np.bincount(
          (np.arange(n, dtype=np.int64) * G) // n, minlength=G
      ).astype(np.int64)
      phi = 0.0 if L == 1 else 1.0 - 1.0 / float(L)
      gt = truth.temporal_ground_truth(metric, counts, phi, sigma=scale, p=p)
      if not check_cell_refusal(cand_cell, gt):
        break
      redraw_counts["T1"] += 1

    assert cand_cell is not None
    cells.append(cand_cell)
    global_cell_id += 1

  # --- Block T2: Bucketed counts (12 cells) ---
  redraw_counts["T2"] = 0
  t2_configs = [
      (0.0, 1),
      (0.0, 1),
      (0.0, 1),
      (0.0, 1),
      (0.0, 10),
      (0.0, 10),
      (1.05, 1),
      (1.05, 1),
      (1.05, 10),
      (1.20, 1),
      (1.20, 1),
      (1.20, 10),
  ]
  counts_expected = np.bincount(
      (np.arange(20000, dtype=np.int64) * 100) // 20000, minlength=100
  ).astype(np.int64)

  for idx, (target_h, L) in enumerate(t2_configs):
    a_calib, h_achieved = calibrate_t2_a(
        base_seed, global_cell_id, target_h, n_target=20000, G=100
    )
    phi = 0.0 if L == 1 else 1.0 - 1.0 / float(L)
    cand_cell = None
    for attempt in range(100):
      alpha, m_mode, kappa_mode, scale = draw_declarations()
      cand_cell = {
          "cell_id": global_cell_id,
          "block": "T2",
          "family": "temporal_bucketed",
          "name": f"T2_bucketed_{idx}_h={target_h}_L={L}",
          "n_target": 20000,
          "G": 100,
          "target_h": target_h,
          "a": a_calib,
          "achieved_h": h_achieved,
          "L": L,
          "metric": "mse",
          "alpha": alpha,
          "m_mode": m_mode,
          "kappa_mode": kappa_mode,
          "scale": scale,
          "modes": ["t_shard", "t_merged"],
      }
      gt_expected = truth.temporal_ground_truth(
          "mse", counts_expected, phi, sigma=scale
      )
      if not check_cell_refusal(cand_cell, gt_expected):
        break
      redraw_counts["T2"] += 1

    assert cand_cell is not None
    cells.append(cand_cell)
    global_cell_id += 1

  # --- Block T2b: Count-guard blind spot (2 cells) ---
  redraw_counts["T2b"] = 0
  for cell_idx, m_mode in enumerate(("default", "tight")):
    cells.append({
        "cell_id": global_cell_id,
        "block": "T2b",
        "family": "temporal_blind_spot",
        "name": f"T2b_blind_spot_{m_mode}",
        "n": 12000,
        "G": 220,
        "metric": "mse",
        "alpha": 0.05,
        "m_mode": m_mode,
        "kappa_mode": "one",
        "scale": 1.0,
        "modes": ["t_shard", "t_merged"],
    })
    global_cell_id += 1

  # --- Block T3: Drift (6 cells) ---
  redraw_counts["T3"] = 0
  for mult in (1.0, 2.0, 4.0):
    for rep_idx in (1, 2):
      scale = float(math.exp(spec_rng.uniform(math.log(1e-3), math.log(1e3))))
      n = 20000
      G = 200
      counts = np.bincount(
          (np.arange(n, dtype=np.int64) * G) // n, minlength=G
      ).astype(np.int64)
      totals = counts.astype(np.float64) * (scale**2)
      shard = cluster_sketch.PartialClusterShard(
          cluster_ids=np.arange(G, dtype=np.int64),
          cluster_totals=totals,
          cluster_counts=counts,
      )
      res = temporal.temporal_interval_from_shard(
          shard, metric="mse", level=0.95, m=2.0, kappa=1.0
      )
      int_obj = res.interval if hasattr(res, "interval") else res
      w = float((int_obj.high - int_obj.low) / (int_obj.high + int_obj.low))
      delta = mult * w
      cells.append({
          "cell_id": global_cell_id,
          "block": "T3",
          "family": "temporal_drift",
          "name": f"T3_drift_mult={mult}_{rep_idx}",
          "n": n,
          "G": G,
          "L": 1,
          "delta_mult": mult,
          "w": w,
          "delta": delta,
          "metric": "mse",
          "alpha": 0.05,
          "m_mode": "m2",
          "m": 2.0,
          "kappa_mode": "one",
          "scale": scale,
          "modes": ["t_data", "t_shard", "t_merged"],
      })
      global_cell_id += 1


  # --- Block G1: Synthetic graphs (52 cells) ---
  redraw_counts["G1"] = 0
  for fam in ("rgg", "ba"):
    for p in (0.99, 0.999):
      cell = {
          "cell_id": global_cell_id,
          "block": "G1",
          "family": "graph_synthetic",
          "graph_family": fam,
          "name": f"G1_fixed_{fam}_p={p}",
          "n": 20000,
          "t": 20,
          "gamma": 0.0,
          "metric": "accuracy",
          "p": p,
          "alpha": 0.05,
          "m_mode": "default",
          "kappa_mode": "one",
          "scale": 1.0,
          "modes": ["g_data", "g_merged"],
      }
      G_exp = int(round(20000.0 / 20.0))
      counts = np.bincount(
          (np.arange(20000, dtype=np.int64) * G_exp) // 20000, minlength=G_exp
      ).astype(np.int64)
      # structural pre-check on an equal-slice line-graph surrogate; gt=None
      c_edges_dummy = np.column_stack(
          [np.arange(G_exp - 1), np.arange(1, G_exp)]
      )
      c_ids_dummy = np.repeat(np.arange(G_exp, dtype=np.int64), counts)
      if check_cell_refusal(
          cell, None, (c_edges_dummy, 20000, c_ids_dummy, c_edges_dummy)
      ):
        cell["refused_at_precheck"] = True
      cells.append(cell)
      global_cell_id += 1

  for i in range(48):
    cand_cell = None
    for attempt in range(100):
      fam = str(spec_rng.choice(["rgg", "ba", "sbm", "isolated", "hub"]))
      t_target = int(spec_rng.choice([20, 50, 200]))
      gamma = (
          0.0 if spec_rng.random() < 0.3 else float(spec_rng.uniform(0.2, 1.0))
      )
      metric = str(spec_rng.choice(["mse", "rmse", "mae", "accuracy"]))
      p = float(spec_rng.uniform(0.6, 0.999)) if metric == "accuracy" else 0.5
      alpha, m_mode, kappa_mode, scale = draw_declarations()
      cand_cell = {
          "cell_id": global_cell_id,
          "block": "G1",
          "family": "graph_synthetic",
          "graph_family": fam,
          "name": f"G1_rand_{i}_{fam}",
          "n": 20000,
          "t": t_target,
          "gamma": gamma,
          "metric": metric,
          "p": p,
          "alpha": alpha,
          "m_mode": m_mode,
          "kappa_mode": kappa_mode,
          "scale": scale,
          "modes": ["g_data", "g_merged"],
      }
      G_exp = int(round(20000.0 / float(t_target)))
      counts = np.bincount(
          (np.arange(20000, dtype=np.int64) * G_exp) // 20000, minlength=G_exp
      ).astype(np.int64)
      # structural pre-check on an equal-slice line-graph surrogate; gt=None
      c_edges_dummy = np.column_stack(
          [np.arange(G_exp - 1), np.arange(1, G_exp)]
      )
      c_ids_dummy = np.repeat(np.arange(G_exp, dtype=np.int64), counts)
      if not check_cell_refusal(
          cand_cell, None, (c_edges_dummy, 20000, c_ids_dummy, c_edges_dummy)
      ):
        break
      redraw_counts["G1"] += 1

    assert cand_cell is not None
    cells.append(cand_cell)
    global_cell_id += 1


  # --- Block F: Fault injection (8 cells) ---
  redraw_counts["F"] = 0
  f_specs = [
      ("temporal", "mse", "nan_residual"),
      ("temporal", "mae", "inf_residual"),
      ("temporal", "accuracy", "indicator_two"),
      ("temporal", "mse", "negative_total"),
      ("graph", "mse", "nan_residual"),
      ("graph", "mse", "count_zero"),
      ("graph", "mse", "mismatched_ids_len"),
      ("temporal", "mse", "nan_timestamps"),
  ]
  for idx, (domain, metric, f_type) in enumerate(f_specs):
    cells.append({
        "cell_id": global_cell_id,
        "block": "F",
        "family": "fault",
        "name": f"F_{domain}_{f_type}",
        "domain": domain,
        "metric": metric,
        "fault_type": f_type,
        "n": 5000,
        "alpha": 0.05,
        "m_mode": "default",
        "kappa_mode": "one",
        "modes": ["t_data"] if domain == "temporal" else ["g_data"],
    })
    global_cell_id += 1

  return cells, redraw_counts


# ---------------------------------------------------------------------------
# Repetition Execution
# ---------------------------------------------------------------------------


def run_cell_repetition(
    cell: dict[str, Any],
    r: int,
    base_seed: int,
    truth_dict: dict[str, Any] | None,
    graph_data: tuple[Any, ...] | None,
) -> dict[str, Any]:
  """Evaluates one repetition across all applicable modes for a cell."""
  c_id = cell["cell_id"]
  b_id = cell["block"]
  alpha = cell["alpha"]
  metric = cell["metric"]
  modes = cell["modes"]
  k_mode = cell.get("kappa_mode", "one")
  m_mode = cell.get("m_mode", "default")
  scale = cell.get("scale", 1.0)
  p = cell.get("p", 0.5)

  data_rng = np.random.default_rng([base_seed, c_id, r])
  mode_rng = np.random.default_rng([base_seed, c_id, r, 1])

  # Initialise common variables
  n = cell.get("n", 100)
  data = np.zeros(n, dtype=np.float64)
  window_ids = np.zeros(n, dtype=np.int64)
  edges = np.zeros((0, 2), dtype=np.int64)
  c_ids = np.zeros(n, dtype=np.int64)
  c_edges = np.zeros((0, 2), dtype=np.int64)
  f_type = cell.get("fault_type", "")
  gt = truth_dict

  # 1. Generate data & summands
  if b_id == "T1":
    n = cell["n"]
    G = cell["G"]
    L = cell["L"]
    phi = 0.0 if L == 1 else 1.0 - 1.0 / float(L)
    u = data_rng.standard_normal(n)
    if phi != 0.0:
      u[1:] *= math.sqrt(1.0 - phi**2)
      u = scipy.signal.lfilter([1.0], [1.0, -phi], u)
    data = (
        u * scale
        if metric != "accuracy"
        else (u < float(scipy.stats.norm.ppf(p))).astype(np.float64)
    )
    window_ids = (np.arange(n, dtype=np.int64) * G) // n

  elif b_id == "T2":
    G = cell["G"]
    L = cell["L"]
    phi = 0.0 if L == 1 else 1.0 - 1.0 / float(L)
    a_param = cell.get("a", 0.0)
    n_expected = cell["n_target"]
    lam_max = n_expected * (1.0 + a_param) / (1.0 + 0.5 * a_param)
    num_pts = int(data_rng.poisson(lam_max))
    raw_t = data_rng.uniform(0.0, 1.0, size=num_pts)
    acc_prob = (1.0 + a_param * raw_t) / (1.0 + a_param)
    unif_v = data_rng.uniform(0.0, 1.0, size=num_pts)
    t_pts = np.sort(raw_t[unif_v < acc_prob])
    n = len(t_pts)
    window_ids = np.clip((t_pts * G).astype(np.int64), 0, G - 1)
    counts = np.bincount(window_ids, minlength=G).astype(np.int64)
    u = data_rng.standard_normal(n)
    if phi != 0.0:
      u[1:] *= math.sqrt(1.0 - phi**2)
      u = scipy.signal.lfilter([1.0], [1.0, -phi], u)
    data = u * scale
    gt = truth.temporal_ground_truth(metric, counts, phi, sigma=scale, p=p)

  elif b_id == "T2b":
    counts = np.array(([4, 4] + [3, 1] * 10) * 10, dtype=np.int64) * 25
    G = len(counts)
    n = int(np.sum(counts))
    u = data_rng.standard_normal(n)
    rho_t2b = truth.t2b_latent_rho(100, 4.0)
    endpoints = np.cumsum([0] + list(counts))
    for c_idx, nc in enumerate(counts):
      if nc == 100:
        idx_c = range(endpoints[c_idx], endpoints[c_idx + 1])
        common_xi = data_rng.standard_normal()
        u[idx_c] = (
            math.sqrt(1.0 - rho_t2b) * u[idx_c] + math.sqrt(rho_t2b) * common_xi
        )
    data = u * scale
    window_ids = np.repeat(np.arange(G), counts)
    gt = truth.temporal_ground_truth(metric, counts, phi=0.0, sigma=scale)

  elif b_id == "T3":
    n = cell["n"]
    G = cell["G"]
    delta = cell["delta"]
    u = data_rng.standard_normal(n)
    n_half = n // 2
    u[n_half:] *= math.sqrt(1.0 + delta)
    data = u * scale
    window_ids = (np.arange(n, dtype=np.int64) * G) // n
    gt = truth_dict

  elif b_id == "G1":
    if graph_data is not None:
      edges, n, c_ids, c_edges = graph_data
      gamma = cell["gamma"]
      deg = np.bincount(edges[:, 0], minlength=n) + np.bincount(
          edges[:, 1], minlength=n
      )
      eta = data_rng.standard_normal(n)
      num_e = len(edges)
      xi = data_rng.standard_normal(num_e) if num_e > 0 else np.zeros(0)
      sum_xi = np.bincount(edges[:, 0], weights=xi, minlength=n) + np.bincount(
          edges[:, 1], weights=xi, minlength=n
      )
      u = (eta + gamma * sum_xi) / np.sqrt(
          1.0 + (gamma**2) * deg.astype(np.float64)
      )
      data = (
          u * scale
          if metric != "accuracy"
          else (u < float(scipy.stats.norm.ppf(p))).astype(np.float64)
      )

  elif b_id == "F":
    n = cell["n"]
    data = data_rng.standard_normal(n)

  # 2. Evaluate across modes
  mode_results = {}
  with contextlib.nullcontext():
    for m_name in modes:
      try:
        if m_name == "t_data":
          if b_id == "F":
            d_inj = data.copy()
            if f_type == "nan_residual":
              d_inj[0] = np.nan
            elif f_type == "inf_residual":
              d_inj[0] = np.inf
            elif f_type == "indicator_two":
              d_inj[0] = 2.0
            elif f_type == "negative_total":
              shard = cluster_sketch.PartialClusterShard(
                  np.arange(10),
                  np.full(10, -1.0),
                  np.ones(10, dtype=np.int64),
              )
              res = temporal.temporal_interval_from_shard(shard, metric=metric)
              mode_results[m_name] = res
              continue
            elif f_type == "nan_timestamps":
              ts = np.arange(n, dtype=np.float64)
              ts[0] = np.nan
              res = temporal.temporal_interval(
                  d_inj, num_windows=10, timestamps=ts, metric=metric
              )
              mode_results[m_name] = res
              continue
            res = temporal.temporal_interval(
                d_inj, num_windows=cell.get("G", 10), metric=metric
            )
          else:
            k_val = 1.0
            m_val = None
            if gt is not None:
              k_val = (
                  1.0
                  if k_mode == "one"
                  else max(
                      1.0,
                      gt["vif_true"]
                      if k_mode == "exact"
                      else gt["vif_true"] / 2.0,
                  )
              )
              m_val = (
                  2.0
                  if b_id == "T3"
                  else (None if m_mode == "default" else gt["m_c_true"])
              )
            res = temporal.temporal_interval(
                data,
                num_windows=cell["G"],
                metric=metric,
                level=1.0 - alpha,
                m=m_val,
                kappa=k_val,
            )
          mode_results[m_name] = res

        elif m_name == "t_shard":
          if b_id == "F":
            shard = cluster_sketch.PartialClusterShard(
                np.arange(10),
                np.full(10, -1.0 if f_type == "negative_total" else 1.0),
                np.ones(10, dtype=np.int64),
            )
            res = temporal.temporal_interval_from_shard(shard, metric=metric)
          else:
            k_val = 1.0
            m_val = None
            if gt is not None:
              k_val = (
                  1.0
                  if k_mode == "one"
                  else max(
                      1.0,
                      gt["vif_true"]
                      if k_mode == "exact"
                      else gt["vif_true"] / 2.0,
                  )
              )
              m_val = (
                  2.0
                  if b_id == "T3"
                  else (None if m_mode == "default" else gt["m_c_true"])
              )
            shard = cluster_sketch.PartialClusterShard.from_data(
                data, window_ids, metric
            )
            res = temporal.temporal_interval_from_shard(
                shard, metric=metric, level=1.0 - alpha, m=m_val, kappa=k_val
            )
          mode_results[m_name] = res

        elif m_name == "t_merged":
          s_count = int(mode_rng.integers(2, 9))
          k_val = 1.0
          m_val = None
          if gt is not None:
            k_val = (
                1.0
                if k_mode == "one"
                else max(
                    1.0,
                    gt["vif_true"]
                    if k_mode == "exact"
                    else gt["vif_true"] / 2.0,
                )
            )
            m_val = (
                2.0
                if b_id == "T3"
                else (None if m_mode == "default" else gt["m_c_true"])
            )
          splits = np.array_split(np.arange(len(data)), s_count)
          merged_shard = None
          for sp in splits:
            sh = cluster_sketch.PartialClusterShard.from_data(
                data[sp], window_ids[sp], metric
            )
            merged_shard = (
                sh if merged_shard is None else merged_shard.merge(sh)
            )
          assert merged_shard is not None
          res = temporal.temporal_interval_from_shard(
              merged_shard,
              metric=metric,
              level=1.0 - alpha,
              m=m_val,
              kappa=k_val,
          )
          mode_results[m_name] = res

        elif m_name == "g_data":
          if b_id == "F":
            res_fault: Any = None
            if f_type == "nan_residual":
              d_inj = data.copy()
              d_inj[0] = np.nan
              res_fault = graph.graph_interval(
                  d_inj, np.arange(n), edges, metric=metric
              )
            elif f_type == "mismatched_ids_len":
              res_fault = graph.graph_interval(
                  data, np.arange(n - 1), edges, metric=metric
              )
            elif f_type == "count_zero":
              shard = cluster_sketch.PartialClusterShard(
                  np.array([0]), np.array([1.0]), np.array([0], dtype=np.int64)
              )
              res_fault = graph.graph_interval_from_shard(
                  shard, np.zeros((0, 2), dtype=np.int64), metric=metric
              )
            mode_results[m_name] = res_fault
          else:
            k_val = 1.0
            m_val = None
            if gt is not None:
              k_val = (
                  1.0
                  if k_mode == "one"
                  else (
                      gt["vif_true"]
                      if k_mode == "exact"
                      else max(1.0, gt["vif_true"] / 2.0)
                  )
              )
              m_val = None if m_mode == "default" else gt["m_c_true"]
            res = graph.graph_interval(
                data,
                c_ids,
                c_edges,
                metric=metric,
                level=1.0 - alpha,
                m=m_val,
                kappa=k_val,
            )
            mode_results[m_name] = res

        elif m_name == "g_merged":
          if b_id == "F" and f_type == "count_zero":
            shard = cluster_sketch.PartialClusterShard(
                np.array([0]), np.array([1.0]), np.array([0], dtype=np.int64)
            )
            res = graph.graph_interval_from_shard(
                shard, np.zeros((0, 2), dtype=np.int64), metric=metric
            )
            mode_results[m_name] = res
          else:
            s_count = int(mode_rng.integers(2, 9))
            k_val = 1.0
            m_val = None
            if gt is not None:
              k_val = (
                  1.0
                  if k_mode == "one"
                  else (
                      gt["vif_true"]
                      if k_mode == "exact"
                      else max(1.0, gt["vif_true"] / 2.0)
                  )
              )
              m_val = None if m_mode == "default" else gt["m_c_true"]
            splits = np.array_split(np.arange(len(data)), s_count)
            merged_shard = None
            for sp in splits:
              sh = cluster_sketch.PartialClusterShard.from_data(
                  data[sp], c_ids[sp], metric
              )
              merged_shard = (
                  sh if merged_shard is None else merged_shard.merge(sh)
              )
            assert merged_shard is not None
            res = graph.graph_interval_from_shard(
                merged_shard,
                c_edges,
                metric=metric,
                level=1.0 - alpha,
                m=m_val,
                kappa=k_val,
            )
            mode_results[m_name] = res

      except Exception as ex:
        mode_results[m_name] = ex


  return {
      "r": r,
      "gt": gt,
      "mode_results": mode_results,
  }


def is_g5_candidate(
    block: str,
    graph_family: Any,
    kappa_mode: Any,
    vif_true: Any,
) -> tuple[bool, Any]:
  """Determines if a cell_mode is judged by G5 or excluded.

  Returns:
    (is_judged, excluded_reason):
      is_judged is True iff the row is judged by G5.
      excluded_reason is 'hub_edge_test' if excluded due to hub edge test, else
      None.
  """
  if block in ("T2b", "T3", "F"):
    return False, None
  k_mode = kappa_mode or "one"
  is_null_kappa1 = k_mode == "one" and (
      vif_true is not None and vif_true <= 1.0001
  )
  if not (k_mode == "exact" or is_null_kappa1):
    return False, None

  # Hub graphs are excluded from the edge-test size check.
  is_graph = block in ("G1", "G2")
  is_hub = is_graph and graph_family == "hub"
  is_edge_test = (k_mode == "one") or (
      k_mode == "exact" and vif_true is not None and vif_true <= 1.0001
  )
  if is_hub and is_edge_test:
    return False, "hub_edge_test"

  return True, None


def compute_t5_count(cells: list[dict[str, Any]]) -> int:
  """Counts the (cell, mode) pairs judged by the refutation size check."""
  return sum(
      len(c["modes"])
      for c in cells
      if (c.get("kappa_mode") == "exact" or c.get("kappa_mode") == "one")
      and c["block"] not in ("T2b", "T3")
      and c["block"] in ("T1", "T2", "T4", "G1", "G2")
      and not (
          c.get("graph_family") == "hub"
          and (
              c.get("kappa_mode") == "one"
              or (c.get("kappa_mode") == "exact" and c.get("gamma", 0.0) == 0.0)
          )
      )
  )


def evaluate_cell(
    cell: dict[str, Any],
    reps: int,
    base_seed: int,
    bonf_alpha: float = 0.01,
    bonf_alpha_g5: float = 0.01,
    bonf_alpha_g9: float = 0.01,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
  """Evaluates one cell across all repetitions and checks all G criteria."""
  c_id = cell["cell_id"]
  b_id = cell["block"]
  metric = cell["metric"]
  alpha = cell["alpha"]
  modes = cell["modes"]
  scale = cell.get("scale", 1.0)
  p = cell.get("p", 0.5)

  # 1. Precompute static ground truth and topology
  gt = None
  g_data = None
  p_data = None
  edge_sha = None

  if b_id == "T1":
    n = cell["n"]
    G = cell["G"]
    L = cell["L"]
    phi = 0.0 if L == 1 else 1.0 - 1.0 / float(L)
    counts = np.bincount(
        (np.arange(n, dtype=np.int64) * G) // n, minlength=G
    ).astype(np.int64)
    gt = truth.temporal_ground_truth(metric, counts, phi, sigma=scale, p=p)

  elif b_id == "T2":
    gt = None

  elif b_id == "T2b":
    counts = np.array(([4, 4] + [3, 1] * 10) * 10, dtype=np.int64) * 25
    gt = truth.temporal_ground_truth(metric, counts, phi=0.0, sigma=scale)

  elif b_id == "T3":
    n = cell["n"]
    G = cell["G"]
    scale = cell["scale"]
    delta = cell["delta"]
    gt = truth.t3_drift_ground_truth(n, G, scale, delta)

  elif b_id == "G1":
    edges, n_nodes = generate_graph_structure(cell, base_seed)
    t_target = cell["t"]
    c_ids = partitioners.partition_graph(
        n_nodes, edges, target_cluster_size=t_target, seed=0
    )
    c_edges = graph.cluster_adjacency(n_nodes, edges, c_ids)
    counts = np.bincount(c_ids).astype(np.int64)
    deg = np.bincount(edges[:, 0], minlength=n_nodes) + np.bincount(
        edges[:, 1], minlength=n_nodes
    )
    c_u = c_ids[edges[:, 0]]
    c_v = c_ids[edges[:, 1]]
    int_edges = edges[c_u == c_v]
    cross_edges = edges[c_u != c_v]
    gt = truth.graph_ground_truth(
        metric,
        counts,
        c_edges,
        deg,
        int_edges,
        cross_edges,
        cell["gamma"],
        sigma=scale,
        p=p,
    )
    G_actual = len(counts)
    if G_actual < 2:
      raise RuntimeError(
          f"Cell {cell['name']} ({c_id}): partition produced G={G_actual} < 2."
      )
    if np.any(counts < 1):
      raise RuntimeError(
          f"Cell {cell['name']} ({c_id}): partition produced cluster count < 1:"
          f" min_count={int(np.min(counts))}."
      )
    for k, v in gt.items():
      if isinstance(v, (int, float, np.floating, np.integer)):
        if not math.isfinite(float(v)):
          raise RuntimeError(
              f"Cell {cell['name']} ({c_id}): gt[{k}] is non-finite: {v}."
          )
      elif isinstance(v, np.ndarray):
        if not np.all(np.isfinite(v)):
          raise RuntimeError(
              f"Cell {cell['name']} ({c_id}): gt[{k}] contains non-finite"
              " values."
          )
    g_data = (edges, n_nodes, c_ids, c_edges)

  elif b_id == "F":
    edges, n_nodes = generate_graph_structure(cell, base_seed)
    g_data = (edges, n_nodes, None, None)

  # If reps == 0, we only needed precomputation (e.g. edge_sha256)
  if reps <= 0:
    return [], {"c_id": c_id, "block": b_id, "name": cell["name"]}, edge_sha

  # 2. Run repetitions
  t_eval_start = time.time()
  reps_data = []
  for r in range(reps):
    reps_data.append(
        run_cell_repetition(
            cell, r, base_seed, gt, g_data
        )
    )
  elapsed = time.time() - t_eval_start

  # 3. Evaluate criteria
  g1_fails = 0
  g2_fails = 0
  g3_fails = 0
  g5_fails = 0
  g6_fails = 0
  g7_fails = 0
  g8_fails = 0
  g9_fails = 0
  g10_fails = 0
  g11_fails = 0
  edge_n_eff_lt_10_count = 0
  worst_g1_pval = 1.0
  worst_g5_rate = 0.0

  # G3: Fault injection
  if b_id == "F":
    for rep in reps_data:
      for res in rep["mode_results"].values():
        if isinstance(res, Exception):
          continue
        int_obj = res.interval if hasattr(res, "interval") else res
        if (
            int_obj is None
            or int_obj.status.name != "ASSUMPTION_REQUIRED"
            or not math.isnan(int_obj.low)
        ):
          g3_fails += 1

  # G10: Partitioner constraints
  if b_id == "G1" and g_data is not None and g_data[2] is not None:
    c_ids_part = g_data[2]
    counts = np.bincount(c_ids_part).astype(np.int64)
    t_target = cell["t"]
    if np.max(counts) > math.floor(1.5 * t_target):
      g10_fails += 1
    if np.count_nonzero(counts < math.ceil(t_target / 2.0)) > 1:
      g10_fails += 1

  # G2: Modes agreement
  if len(modes) > 1:
    for rep in reps_data:
      m_results = [
          rep["mode_results"][m]
          for m in modes
          if not isinstance(rep["mode_results"][m], Exception)
      ]
      if len(m_results) > 1:
        int_objs = [
            r.interval if hasattr(r, "interval") else r for r in m_results
        ]
        lows = [o.low for o in int_objs if o.low is not None]
        highs = [o.high for o in int_objs if o.high is not None]
        if lows and (max(lows) - min(lows)) > 1e-9 * max(
            1e-9, max(map(abs, lows))
        ):
          g2_fails += 1
        if highs and (max(highs) - min(highs)) > 1e-9 * max(
            1e-9, max(map(abs, highs))
        ):
          g2_fails += 1

  # Per-mode evaluations
  mode_rows = []
  worst_g9_rate = 0.0
  for m_name in modes:
    refuted_count = 0
    warning_count = 0
    untestable_count = 0
    reps_in_contract = 0
    reps_out_contract = 0
    in_contract_miss_lo = 0
    in_contract_miss_hi = 0
    out_contract_miss_lo = 0
    out_contract_miss_hi = 0
    unrefuted_miss = 0
    out_contract_bounds = []
    half_widths = []
    refusal_status = None
    refusal_message = None
    m_decl: float | None = None
    k_decl: float | None = None

    for rep in reps_data:
      res = rep["mode_results"][m_name]
      if isinstance(res, Exception):
        if refusal_status is None:
          refusal_status = type(res).__name__
          refusal_message = str(res)
        continue
      rep_gt = rep["gt"] if rep["gt"] is not None else gt
      int_obj = res.interval if hasattr(res, "interval") else res
      k_check = res.kappa_check if hasattr(res, "kappa_check") else None

      is_returned = (
          int_obj is not None
          and getattr(int_obj, "status", None) is not None
          and int_obj.status.name in ("UNREFUTED", "TAIL_UNRESOLVED")
          and int_obj.low is not None
          and int_obj.high is not None
          and math.isfinite(int_obj.low)
          and math.isfinite(int_obj.high)
      )
      if is_returned:
        half_widths.append((int_obj.high - int_obj.low) / 2.0)
      else:
        if refusal_status is None:
          if (
              int_obj is not None
              and getattr(int_obj, "status", None) is not None
          ):
            refusal_status = int_obj.status.name
            refusal_message = getattr(
                int_obj, "message", getattr(res, "message", "")
            )
          else:
            refusal_status = "NO_INTERVAL"
            refusal_message = getattr(res, "message", "")

      # G7: Default M_c formula
      if (
          cell.get("m_mode") == "default"
          and int_obj is not None
          and getattr(int_obj, "m_declared", None) is not None
          and rep_gt is not None
      ):
        m_form = truth.default_m_c_formula(
            rep_gt["counts"], truth.library_item_default_m(metric)
        )
        if abs(int_obj.m_declared - m_form) / m_form > 1e-12:
          g7_fails += 1

      # G8: Count guard
      if b_id in ("T2", "T2b") and k_check is not None and rep_gt is not None:
        h_realized = truth.compute_count_guard_h(rep_gt["counts"])
        if (
            h_realized > 1.1
            and k_check.outcome != refutation.Refutation.UNTESTABLE
        ):
          g8_fails += 1

      # G11: Edge guard
      if (
          b_id == "G1"
          and cell.get("kappa_mode") == "one"
          and hasattr(res, "edge_n_eff")
          and k_check is not None
      ):
        if g_data and g_data[3] is not None and len(g_data[3]) >= 30:
          if res.edge_n_eff is not None:
            if res.edge_n_eff < 10.0:
              edge_n_eff_lt_10_count += 1
              if k_check.outcome != refutation.Refutation.UNTESTABLE:
                g11_fails += 1
            elif (
                res.edge_n_eff >= 10.0
                and k_check.outcome == refutation.Refutation.UNTESTABLE
                and k_check.reason
                and "effective edges" in k_check.reason
            ):
              g11_fails += 1

      # Theta coverage & Contract per repetition
      if rep_gt is not None:
        theta = rep_gt["theta"]
        if metric == "rmse":
          theta = math.sqrt(theta)

        m_decl = getattr(int_obj, "m_declared", None)
        if m_decl is None:
          m_decl = getattr(res, "m_declared", None)
        if m_decl is None:
          m_decl = cell.get("m_declared", 1.0)

        k_decl = getattr(res, "kappa_declared", None)
        if k_decl is None:
          k_decl = getattr(int_obj, "kappa_declared", None)
        if k_decl is None:
          k_decl = cell.get("kappa_declared", 1.0)

        rep_in_contract = truth.is_in_contract(
            rep_gt["m_c_var"], m_decl, rep_gt["vif_true"], k_decl
        )
        if rep_in_contract:
          reps_in_contract += 1
          if is_returned:
            if theta < int_obj.low:
              in_contract_miss_lo += 1
            if theta > int_obj.high:
              in_contract_miss_hi += 1
        else:
          reps_out_contract += 1
          if is_returned:
            if theta < int_obj.low:
              out_contract_miss_lo += 1
            if theta > int_obj.high:
              out_contract_miss_hi += 1
          g_count = len(rep_gt["counts"]) if "counts" in rep_gt else 10
          rho_t5 = truth.theorem5_rho(
              rep_gt["v_tot"],
              m_decl,
              k_decl,
              rep_gt["n"],
              rep_gt["theta"],
              g_count,
          )
          bnd = truth.theorem5_bound(rho_t5, alpha)
          out_contract_bounds.append(bnd)

        if is_returned:
          is_miss = (theta < int_obj.low or theta > int_obj.high)
          is_refuted = (
              int_obj.status.name != "UNREFUTED"
              or (
                  k_check is not None
                  and k_check.outcome == refutation.Refutation.REFUTED
              )
          )
          if is_miss and not is_refuted:
            unrefuted_miss += 1

        if k_check and k_check.outcome == refutation.Refutation.REFUTED:
          refuted_count += 1
        if k_check and k_check.outcome == refutation.Refutation.UNTESTABLE:
          untestable_count += 1
        if (
            hasattr(res, "drift")
            and res.drift
            and getattr(res.drift, "outcome", None)
            and res.drift.outcome.name == "WARNING"
        ):
          warning_count += 1

    # End reps loop for mode
    # G1: Coverage over in-contract repetitions
    if reps_in_contract > 0 and b_id in ("T1", "T2", "T3", "G1"):
      p_lo = truth.binomial_p_value(
          in_contract_miss_lo, reps_in_contract, alpha / 2.0
      )
      p_hi = truth.binomial_p_value(
          in_contract_miss_hi, reps_in_contract, alpha / 2.0
      )
      worst_g1_pval = min(worst_g1_pval, p_lo, p_hi)
      if p_lo < bonf_alpha or p_hi < bonf_alpha:
        g1_fails += 1

    # G6: Sensitivity over out-of-contract repetitions
    if reps_out_contract > 0 and b_id in ("T1", "T2", "T3", "G1"):
      mean_bnd = float(np.mean(out_contract_bounds))
      p_lo_g6 = truth.binomial_p_value(
          out_contract_miss_lo, reps_out_contract, mean_bnd
      )
      p_hi_g6 = truth.binomial_p_value(
          out_contract_miss_hi, reps_out_contract, mean_bnd
      )
      if p_lo_g6 < bonf_alpha or p_hi_g6 < bonf_alpha:
        g6_fails += 1

    # Extract aggregate truth across repetitions
    rep_gts = [rep["gt"] for rep in reps_data if rep.get("gt") is not None]
    if rep_gts:
      thetas = [float(g["theta"]) for g in rep_gts]
      vifs = [float(g["vif_true"]) for g in rep_gts]
      m_trues = [float(g["m_c_true"]) for g in rep_gts]
      m_vars = [float(g["m_c_var"]) for g in rep_gts]
      n_vals = [
          float(rep.get("n", g.get("n", cell.get("n", 0))))
          for rep, g in zip(reps_data, rep_gts)
      ]
      row_theta = float(np.mean(thetas))
      row_vif = float(np.mean(vifs))
      row_vif_min = float(np.min(vifs))
      row_vif_max = float(np.max(vifs))
      row_m_true = float(np.mean(m_trues))
      row_m_true_min = float(np.min(m_trues))
      row_m_true_max = float(np.max(m_trues))
      row_m_var = float(np.mean(m_vars))
      row_m_var_min = float(np.min(m_vars))
      row_m_var_max = float(np.max(m_vars))
      row_n = float(np.mean(n_vals))
    else:
      row_theta = float(gt["theta"]) if gt else None
      row_vif = float(gt["vif_true"]) if gt else None
      row_vif_min = row_vif
      row_vif_max = row_vif
      row_m_true = float(gt["m_c_true"]) if gt else None
      row_m_true_min = row_m_true
      row_m_true_max = row_m_true
      row_m_var = float(gt["m_c_var"]) if gt else None
      row_m_var_min = row_m_var
      row_m_var_max = row_m_var
      row_n = float(cell.get("n", 0))

    k_mode = cell.get("kappa_mode", "one")
    is_judged_g5, g5_excluded_reason = is_g5_candidate(
        b_id, cell.get("graph_family"), k_mode, row_vif
    )
    if is_judged_g5:
      ref_rate = float(refuted_count) / float(reps)
      worst_g5_rate = max(worst_g5_rate, ref_rate)
      p_ref = truth.binomial_p_value(refuted_count, reps, 0.075)
      if p_ref < bonf_alpha_g5:
        g5_fails += 1

    if b_id == "T1":
      dr_rate = float(warning_count) / float(reps)
      worst_g9_rate = max(worst_g9_rate, dr_rate)
      p_dr = truth.binomial_p_value(warning_count, reps, alpha)
      if p_dr < bonf_alpha_g9:
        g9_fails += 1

    total_miss_lo = in_contract_miss_lo + out_contract_miss_lo
    total_miss_hi = in_contract_miss_hi + out_contract_miss_hi
    lo_cp99, hi_cp99 = truth.clopper_pearson_99(refuted_count, reps)

    # Mode JSONL row
    row = {
        "row_type": "cell_mode",
        "cell_id": c_id,
        "block": b_id,
        "cell_name": cell["name"],
        "mode": m_name,
        "metric": metric,
        "alpha": alpha,
        "n": row_n,
        "m_mode": cell.get("m_mode"),
        "m_declared": float(m_decl) if m_decl is not None else None,
        "kappa_mode": cell.get("kappa_mode"),
        "kappa_declared": float(k_decl) if k_decl is not None else None,
        "reps": reps,
        "theta": row_theta,
        "vif_true": row_vif,
        "vif_true_min": row_vif_min,
        "vif_true_max": row_vif_max,
        "m_c_true": row_m_true,
        "m_c_true_min": row_m_true_min,
        "m_c_true_max": row_m_true_max,
        "m_c_var": row_m_var,
        "m_c_var_min": row_m_var_min,
        "m_c_var_max": row_m_var_max,
        "in_contract": reps_in_contract == reps,
        "in_contract_reps_lo": reps_in_contract,
        "in_contract_reps_hi": reps_in_contract,
        "out_contract_reps_lo": reps_out_contract,
        "out_contract_reps_hi": reps_out_contract,
        "returned_count": len(half_widths),
        "miss_lo": total_miss_lo,
        "miss_hi": total_miss_hi,
        "unrefuted_miss": unrefuted_miss,
        "assumption_required": reps - len(half_widths),
        "coverage_lo": 1.0 - float(total_miss_lo) / float(reps),
        "coverage_hi": 1.0 - float(total_miss_hi) / float(reps),
        "refuted_count": refuted_count,
        "refuted_rate": float(refuted_count) / float(reps),
        "refuted_cp99": [lo_cp99, hi_cp99],
        "g5_excluded_reason": g5_excluded_reason,
        "warning_count": warning_count,
        "warning_rate": float(warning_count) / float(reps),
        "untestable_count": untestable_count,
        "untestable_rate": float(untestable_count) / float(reps),
        "mean_half_width": float(np.mean(half_widths)) if half_widths else None,
        "refusal_status": refusal_status,
        "refusal_message": refusal_message,
    }

    if b_id == "T2":
      row["a"] = cell.get("a")
      row["achieved_h"] = cell.get("achieved_h")
      row["stream_seed"] = [base_seed, c_id, 2**21]
    mode_rows.append(row)

  cell_stats = {
      "c_id": c_id,
      "block": b_id,
      "name": cell["name"],
      "elapsed_s": elapsed,
      "reps": reps,
      "returned_count": mode_rows[0]["returned_count"] if mode_rows else 0,
      "misses": (mode_rows[0]["miss_lo"] + mode_rows[0]["miss_hi"]) if mode_rows else 0,
      "unrefuted_miss": mode_rows[0].get("unrefuted_miss", 0) if mode_rows else 0,
      "assumption_required": mode_rows[0].get("assumption_required", 0) if mode_rows else 0,
      "refuted_count": mode_rows[0].get("refuted_count", 0) if mode_rows else 0,
      "g1_fails": g1_fails,
      "g2_fails": g2_fails,
      "g3_fails": g3_fails,
      "g5_fails": g5_fails,
      "g6_fails": g6_fails,
      "g7_fails": g7_fails,
      "g8_fails": g8_fails,
      "g9_fails": g9_fails,
      "g10_fails": g10_fails,
      "g11_fails": g11_fails,
      "edge_n_eff_lt_10_count": edge_n_eff_lt_10_count,
      "worst_g1_pval": worst_g1_pval,
      "worst_g5_rate": worst_g5_rate,
      "worst_g9_rate": worst_g9_rate,
  }
  return mode_rows, cell_stats, edge_sha


def _evaluate_cell_worker(
    args: tuple[dict[str, Any], int, int, float, float, float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  cell, reps, base_seed, b_alpha, b_alpha_g5, b_alpha_g9 = args
  mode_rows, cell_stats, _ = evaluate_cell(
      cell,
      reps=reps,
      base_seed=base_seed,
      bonf_alpha=b_alpha,
      bonf_alpha_g5=b_alpha_g5,
      bonf_alpha_g9=b_alpha_g9,
  )
  return mode_rows, cell_stats


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  seed = _SEED.value if _SEED.value is not None else DEFAULT_SEED
  mode = str(_MODE.value)
  all_cells, redraw_counts = generate_all_cells(seed)

  if mode == "quick":
    # 1 cell per block: T1, T2, T2b, T3, G1, F (6 cells total)
    blocks_seen = set()
    selected_cells = []
    for c in all_cells:
      if c["block"] not in blocks_seen:
        blocks_seen.add(c["block"])
        selected_cells.append(c)
    cells = selected_cells
    reps_normal = 20
    reps_f = 20
  else:
    cells = all_cells
    reps_normal = 5000
    reps_f = 200

  workers = min(int(_WORKERS.value or 4), len(cells))
  print(
      f"=== Coverage Dependent Benchmark (mode={mode}, seed={seed},"
      f" workers={workers}, cells={len(cells)}) ===",
      flush=True,
  )

  t_count = sum(
      len(c["modes"])
      for c in cells
      if c["block"] in ("T1", "T2", "T3", "G1")
  )
  t5_count = compute_t5_count(cells)
  t9_count = sum(len(c["modes"]) for c in cells if c["block"] == "T1")
  bonf_alpha = 0.01 / float(max(1, t_count))
  bonf_alpha_g5 = 0.01 / float(max(1, t5_count))
  bonf_alpha_g9 = 0.01 / float(max(1, t9_count))

  tasks = [
      (
          c,
          reps_f if c["block"] == "F" else reps_normal,
          seed,
          bonf_alpha,
          bonf_alpha_g5,
          bonf_alpha_g9,
      )
      for c in cells
  ]
  all_rows: list[dict[str, Any]] = []
  all_stats: list[dict[str, Any]] = []

  t_start = time.time()
  if workers > 1:
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
      for rows, stats in executor.map(_evaluate_cell_worker, tasks):
        all_rows.extend(rows)
        all_stats.append(stats)
        print(
            f"  Completed {stats['name']} ({len(all_stats)}/{len(cells)})",
            flush=True,
        )
  else:
    for task in tasks:
      rows, stats = _evaluate_cell_worker(task)
      all_rows.extend(rows)
      all_stats.append(stats)
      print(
          f"  Completed {stats['name']} ({len(all_stats)}/{len(cells)})",
          flush=True,
      )

  total_elapsed = time.time() - t_start




  # Summary table
  print("\n" + "=" * 125, flush=True)
  print(
      f"SUMMARY RESULTS TABLE (mode={mode}, total elapsed: {total_elapsed:.1f}s)",
      flush=True,
  )
  print("=" * 125, flush=True)
  print(
      f"{'Cell Name':<32} {'Block':<6} {'Reps':<6} {'Issued':<8}"
      f" {'Miss':<18} {'Unref. miss':<18} {'AssumpReq':<10} {'Refuted':<8}",
      flush=True,
  )
  print("-" * 125, flush=True)
  for stats in all_stats:
    reps = stats.get("reps", 0)
    issued = stats.get("returned_count", 0)
    misses = stats.get("misses", 0)
    unref_miss = stats.get("unrefuted_miss", 0)
    assump = stats.get("assumption_required", 0)
    refuted = stats.get("refuted_count", 0)
    # Both rates use all repetitions as the denominator.
    miss_rate = (misses / reps) if reps > 0 else 0.0
    joint_rate = (unref_miss / reps) if reps > 0 else 0.0
    miss_str = f"{miss_rate:.2%} ({misses}/{reps})"
    joint_str = f"{joint_rate:.2%} ({unref_miss}/{reps})"
    print(
        f"{stats['name']:<32} {stats['block']:<6} {reps:<6d} {issued:<8d}"
        f" {miss_str:<18} {joint_str:<18} {assump:<10d} {refuted:<8d}",
        flush=True,
    )

  out_path = _OUTPUT.value
  if out_path:
    parent = os.path.dirname(out_path)
    if parent:
      os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
      for r in all_rows:
        f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(all_rows)} result rows to {out_path}", flush=True)


if __name__ == "__main__":
  app.run(main)
