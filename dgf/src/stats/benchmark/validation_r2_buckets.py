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

"""Validation benchmark for correlated R^2, bucketed windows and graph kappa.

Runs blocks V1-V5 (correlated R^2 intervals, midpoint-bucketed temporal
intervals, topological kappa, AR(1) MSE against textbook intervals) and
records per-cell results to JSONL.
"""

from collections.abc import Sequence
import concurrent.futures
import json
import math
import os
import re
import sys
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

DEFAULT_SEED = 0x21DA7A

_SEED = flags.DEFINE_integer(
    "seed", DEFAULT_SEED, "Base random seed."
)
_MODE = flags.DEFINE_enum(
    "mode", "quick", ["quick", "full"], "Execution mode: 'quick' or 'full'."
)
_OUTPUT = flags.DEFINE_string(
    "output", None, "Optional output path for newline-delimited JSON results."
)
_WORKERS = flags.DEFINE_integer(
    "workers", min(16, os.cpu_count() or 4), "Number of parallel workers."
)
_REPS = flags.DEFINE_integer(
    "reps", 0, "Optional override for reps (0 to use defaults)."
)
_CELLS = flags.DEFINE_string(
    "cells",
    None,
    "Optional regular expression; only cells whose name matches (re.search)"
    " are run. Seeds are per cell, so a subset reproduces the full-run rows.",
)


def q_threshold(reps: int, alpha: float) -> int:
  """q(reps, p) = scipy.stats.binom.ppf(0.999, reps, p) where p = alpha / 2."""
  p = alpha / 2.0
  return int(scipy.stats.binom.ppf(0.999, reps, p))


def _ar1_window_vif(rho: float, m_c: int, n: int) -> float:
  """Exact window-total VIF for an AR(1) process with covariance C(h) = rho^h."""
  if rho == 0.0 or m_c <= 1:
    return 1.0
  h_m = np.arange(1, m_c, dtype=np.float64)
  v_c = float(m_c) + 2.0 * float(np.sum((m_c - h_m) * (rho**h_m)))
  h_n = np.arange(1, n, dtype=np.float64)
  v_tot = float(n) + 2.0 * float(np.sum((n - h_n) * (rho**h_n)))
  g = n // m_c
  return float(v_tot / (float(g) * v_c))


def _random_regular_graph(
    num_nodes: int, degree: int = 4, rng: Any = None
) -> np.ndarray:
  """Generates a random regular cluster graph."""
  if rng is None:
    rng = np.random.default_rng(0)
  half_deg = degree // 2
  edges_set = set()
  for i in range(num_nodes):
    for offset in range(1, half_deg + 1):
      j = (i + offset) % num_nodes
      u, v = min(i, j), max(i, j)
      edges_set.add((u, v))
  return np.array(sorted(edges_set), dtype=np.int64)


# -----------------------------------------------------------------------------
# Block V1 Evaluation
# -----------------------------------------------------------------------------


def _eval_v1a_or_v1c_cell(
    cell: dict[str, Any], reps: int = 2000
) -> dict[str, Any]:
  """Evaluates one V1a, V1c, or V5 cell across reps."""
  sub_block = cell["sub_block"]
  cell_seed = cell["seed"]
  alpha = cell["alpha"]
  true_r2 = 0.8
  m_val = cell.get("m")
  m_labels_val = cell.get("m_labels")

  cell_rng = np.random.default_rng(cell_seed)
  g = cell["G"]
  m_bar = cell["m_bar"]
  icc_y, icc_e = cell["icc"]
  law = cell["law"]
  api = cell["api"]

  low_m = int(round(m_bar / 2.0))
  high_m = int(round(3.0 * m_bar / 2.0))
  counts = cell_rng.integers(low_m, high_m + 1, size=g, dtype=np.int64)
  n = int(np.sum(counts))
  c_ids = np.repeat(np.arange(g, dtype=np.int64), counts)

  if api == "graph":
    cluster_edges = _random_regular_graph(g, degree=4, rng=cell_rng)
  else:
    cluster_edges = None

  issued_count = 0
  low_misses = 0
  high_misses = 0
  high_one_count = 0
  rel_widths: list[float] = []
  low_endpoints: list[float] = []
  high_endpoints: list[float] = []

  refuted_e2 = 0
  refuted_q = 0
  refuted_y = 0
  refuted_overall = 0

  for r in range(reps):
    rep_rng = np.random.default_rng([cell_seed, r])
    if law == "gaussian":
      a = rep_rng.normal(0.0, math.sqrt(icc_y * 1.0), size=g)
      b = rep_rng.normal(0.0, math.sqrt((1.0 - icc_y) * 1.0), size=n)
      u = rep_rng.normal(0.0, math.sqrt(icc_e * 0.2), size=g)
      v = rep_rng.normal(0.0, math.sqrt((1.0 - icc_e) * 0.2), size=n)
    else:
      a = rep_rng.laplace(0.0, math.sqrt(icc_y * 1.0 / 2.0), size=g)
      b = rep_rng.laplace(0.0, math.sqrt((1.0 - icc_y) * 1.0 / 2.0), size=n)
      u = rep_rng.laplace(0.0, math.sqrt(icc_e * 0.2 / 2.0), size=g)
      v = rep_rng.laplace(0.0, math.sqrt((1.0 - icc_e) * 0.2 / 2.0), size=n)

    y = a[c_ids] + b
    e = u[c_ids] + v
    y_pred = y - e
    e2 = (y - y_pred) ** 2

    e2_totals = np.bincount(c_ids, weights=e2, minlength=g).astype(np.float64)
    y_totals = np.bincount(c_ids, weights=y, minlength=g).astype(np.float64)
    y2_totals = np.bincount(c_ids, weights=y**2, minlength=g).astype(np.float64)

    shard = cluster_sketch.R2ClusterShard(
        cluster_ids=np.arange(g, dtype=np.int64),
        e2_totals=e2_totals,
        y_totals=y_totals,
        y2_totals=y2_totals,
        cluster_counts=counts,
    )

    if api == "graph":
      assert cluster_edges is not None
      res = graph.graph_interval_from_shard(
          shard,
          cluster_edges,
          metric="r2",
          level=1.0 - alpha,
          m_item=m_val,
          m_labels=m_labels_val,
      )
      low, high = res.interval.low, res.interval.high
      r2_detail = res.r2_detail
      k_check = res.kappa_check
    else:
      res_temp = temporal.temporal_interval_from_shard(
          shard,
          metric="r2",
          level=1.0 - alpha,
          m_item=m_val,
          m_labels=m_labels_val,
      )
      low, high = res_temp.low, res_temp.high
      r2_detail = res_temp.r2_detail
      k_check = res_temp.kappa_check

    is_refused = math.isnan(low) and math.isnan(high)
    if not is_refused:
      issued_count += 1
      if math.isfinite(low):
        low_endpoints.append(low)
      if math.isfinite(high):
        high_endpoints.append(high)
      if math.isfinite(low) and true_r2 < low:
        low_misses += 1
      if math.isfinite(high) and true_r2 > high:
        high_misses += 1
      if high == 1.0:
        high_one_count += 1
      if math.isfinite(low) and math.isfinite(high):
        rel_widths.append(
            (high - low) / true_r2 if true_r2 > 0 else (high - low)
        )
      else:
        rel_widths.append(float("inf"))

    if r2_detail is not None:
      if r2_detail.e2_kappa_check.outcome is refutation.Refutation.REFUTED:
        refuted_e2 += 1
      if r2_detail.q_kappa_check.outcome is refutation.Refutation.REFUTED:
        refuted_q += 1
      if r2_detail.y_kappa_check.outcome is refutation.Refutation.REFUTED:
        refuted_y += 1
    if k_check.outcome is refutation.Refutation.REFUTED:
      refuted_overall += 1

  q_val = q_threshold(reps, alpha)
  pass_coverage = (
      low_misses <= q_val
      and high_misses <= q_val
      and issued_count >= reps * 0.50
  )

  refusal_rate = float(reps - issued_count) / float(reps)
  cond_low_miss = (
      float(low_misses) / float(issued_count) if issued_count > 0 else 0.0
  )
  cond_high_miss = (
      float(high_misses) / float(issued_count) if issued_count > 0 else 0.0
  )

  return {
      "cell_name": cell["name"],
      "block": cell.get("block", "V1"),
      "sub_block": sub_block,
      "reps": reps,
      "alpha": alpha,
      "q_threshold": q_val,
      "low_misses": low_misses,
      "high_misses": high_misses,
      "issued_count": issued_count,
      "refusal_rate": refusal_rate,
      "cond_low_miss_rate": cond_low_miss,
      "cond_high_miss_rate": cond_high_miss,
      "frac_high_one": float(high_one_count) / float(reps),
      "med_width": float(np.median(rel_widths)) if rel_widths else float("nan"),
      "med_low": (
          float(np.median(low_endpoints)) if low_endpoints else float("nan")
      ),
      "med_high": (
          float(np.median(high_endpoints)) if high_endpoints else float("nan")
      ),
      "refuted_rate_e2": float(refuted_e2) / float(reps),
      "refuted_rate_q": float(refuted_q) / float(reps),
      "refuted_rate_y": float(refuted_y) / float(reps),
      "refuted_rate_overall": float(refuted_overall) / float(reps),
      "passed": pass_coverage,
  }


def _eval_v1b_cell(cell: dict[str, Any], reps: int = 2000) -> dict[str, Any]:
  """Evaluates one V1b cell across reps."""
  sub_block = cell["sub_block"]
  cell_seed = cell["seed"]
  alpha = cell["alpha"]
  true_r2 = 0.8

  n = cell["n"]
  g = cell["G"]
  rho_y, rho_e = cell["rho"]
  m_c = n // g
  vif_y = _ar1_window_vif(rho_y, m_c, n)
  vif_e2 = _ar1_window_vif(rho_e**2, m_c, n)
  vif_q = _ar1_window_vif(rho_y**2, m_c, n)
  kappa_declared = max(vif_e2, vif_q)
  kappa_y_declared = vif_y

  issued_count = 0
  low_misses = 0
  high_misses = 0
  high_one_count = 0
  rel_widths: list[float] = []
  low_endpoints: list[float] = []
  high_endpoints: list[float] = []

  refuted_e2 = 0
  refuted_q = 0
  refuted_y = 0
  refuted_overall = 0

  for r in range(reps):
    rep_rng = np.random.default_rng([cell_seed, r])
    u_y = rep_rng.standard_normal(n)
    if rho_y != 0.0:
      u_y[1:] *= math.sqrt(1.0 - rho_y**2)
      y = scipy.signal.lfilter([1.0], [1.0, -rho_y], u_y)
    else:
      y = u_y

    u_e = rep_rng.standard_normal(n)
    if rho_e != 0.0:
      u_e[1:] *= math.sqrt(1.0 - rho_e**2)
      e = scipy.signal.lfilter([1.0], [1.0, -rho_e], u_e) * math.sqrt(0.2)
    else:
      e = u_e * math.sqrt(0.2)

    y_pred = y - e
    res = temporal.temporal_interval(
        (y, y_pred),
        num_windows=g,
        metric="r2",
        level=1.0 - alpha,
        kappa=kappa_declared,
        kappa_labels=kappa_y_declared,
    )
    low, high = res.low, res.high
    r2_detail = res.r2_detail
    k_check = res.kappa_check

    is_refused = math.isnan(low) and math.isnan(high)
    if not is_refused:
      issued_count += 1
      if math.isfinite(low):
        low_endpoints.append(low)
      if math.isfinite(high):
        high_endpoints.append(high)
      if math.isfinite(low) and true_r2 < low:
        low_misses += 1
      if math.isfinite(high) and true_r2 > high:
        high_misses += 1
      if high == 1.0:
        high_one_count += 1
      if math.isfinite(low) and math.isfinite(high):
        rel_widths.append(
            (high - low) / true_r2 if true_r2 > 0 else (high - low)
        )
      else:
        rel_widths.append(float("inf"))

    if r2_detail is not None:
      if r2_detail.e2_kappa_check.outcome is refutation.Refutation.REFUTED:
        refuted_e2 += 1
      if r2_detail.q_kappa_check.outcome is refutation.Refutation.REFUTED:
        refuted_q += 1
      if r2_detail.y_kappa_check.outcome is refutation.Refutation.REFUTED:
        refuted_y += 1
    if k_check.outcome is refutation.Refutation.REFUTED:
      refuted_overall += 1

  q_val = q_threshold(reps, alpha)
  pass_coverage = (
      low_misses <= q_val
      and high_misses <= q_val
      and issued_count >= reps * 0.50
  )

  refusal_rate = float(reps - issued_count) / float(reps)
  cond_low_miss = (
      float(low_misses) / float(issued_count) if issued_count > 0 else 0.0
  )
  cond_high_miss = (
      float(high_misses) / float(issued_count) if issued_count > 0 else 0.0
  )

  return {
      "cell_name": cell["name"],
      "block": cell.get("block", "V1"),
      "sub_block": sub_block,
      "reps": reps,
      "alpha": alpha,
      "q_threshold": q_val,
      "low_misses": low_misses,
      "high_misses": high_misses,
      "issued_count": issued_count,
      "refusal_rate": refusal_rate,
      "cond_low_miss_rate": cond_low_miss,
      "cond_high_miss_rate": cond_high_miss,
      "frac_high_one": float(high_one_count) / float(reps),
      "med_width": float(np.median(rel_widths)) if rel_widths else float("nan"),
      "med_low": (
          float(np.median(low_endpoints)) if low_endpoints else float("nan")
      ),
      "med_high": (
          float(np.median(high_endpoints)) if high_endpoints else float("nan")
      ),
      "refuted_rate_e2": float(refuted_e2) / float(reps),
      "refuted_rate_q": float(refuted_q) / float(reps),
      "refuted_rate_y": float(refuted_y) / float(reps),
      "refuted_rate_overall": float(refuted_overall) / float(reps),
      "passed": pass_coverage,
  }


def eval_v1_cell(cell: dict[str, Any], reps: int = 2000) -> dict[str, Any]:
  """Evaluates one V1 or V5 cell across reps."""
  if cell["sub_block"] in ("V1a", "V1c", "V5"):
    return _eval_v1a_or_v1c_cell(cell, reps)
  elif cell["sub_block"] == "V1b":
    return _eval_v1b_cell(cell, reps)
  else:
    raise ValueError(f"Unknown V1/V5 sub_block: {cell['sub_block']}")


# -----------------------------------------------------------------------------
# Block V2 Evaluation
# -----------------------------------------------------------------------------


def _generate_v2_counts(
    pattern: str, g: int, k: int, rng: Any
) -> tuple[np.ndarray, int, float]:
  """Generates bucket counts ensuring m_bar >= b_max (Lemma W)."""
  if pattern == "bursty":
    raw_c = np.round(
        rng.lognormal(mean=math.log(20.0) - 0.5, sigma=1.0, size=k)
    ).astype(np.int64)
    counts = np.maximum(1, raw_c)
  else:
    counts = np.round(np.linspace(5, 35, k)).astype(np.int64)

  n = int(np.sum(counts))
  b_max = int(np.max(counts))
  m_bar = float(n) / float(g)

  # Lemma W needs m̄ >= b_max. Merging buckets would only raise b_max, so
  # instead any bucket above 0.9 * m̄ is split into two halves (total n is
  # unchanged) until b_max <= m̄.
  while m_bar < b_max or b_max > int(m_bar):
    new_counts: list[int] = []
    threshold = int(m_bar * 0.9)
    for c_val in counts:
      c_int = int(c_val)
      if c_int > threshold:
        half = c_int // 2
        new_counts.append(half)
        new_counts.append(c_int - half)
      else:
        new_counts.append(c_int)
    counts = np.array(new_counts, dtype=np.int64)
    n = int(np.sum(counts))
    b_max = int(np.max(counts))
    m_bar = float(n) / float(g)

  r = float(b_max * g) / float(n) if n > 0 else 1.0
  return counts, b_max, r


def eval_v2_coverage_cell(
    cell: dict[str, Any], reps: int = 4000
) -> dict[str, Any]:
  """Evaluates one V2 coverage cell across 4000 reps."""
  cell_seed = cell["seed"]
  g = cell["G"]
  metric = cell["metric"]
  pattern = cell["pattern"]
  k = 50 * g
  alpha = 0.05
  theta_true = 1.0 if metric == "mse" else 0.99

  cell_rng = np.random.default_rng(cell_seed)
  counts, b_max, r_ratio = _generate_v2_counts(pattern, g, k, cell_rng)

  issued_count = 0
  low_misses = 0
  high_misses = 0
  rel_widths = []

  for r in range(reps):
    rep_rng = np.random.default_rng([cell_seed, r])
    if metric == "mse":
      totals = rep_rng.chisquare(counts).astype(np.float64)
    else:
      totals = rep_rng.binomial(counts, 0.99).astype(np.float64)

    res = temporal.temporal_interval_from_buckets(
        counts, totals, num_windows=g, metric=metric, level=0.95, kappa=1.0
    )
    low, high = res.low, res.high

    is_refused = math.isnan(low) and math.isnan(high)
    if not is_refused:
      issued_count += 1
      if math.isfinite(low) and theta_true < low:
        low_misses += 1
      if math.isfinite(high) and theta_true > high:
        high_misses += 1
      if math.isfinite(low) and math.isfinite(high):
        rel_widths.append(
            (high - low) / theta_true if theta_true > 0 else (high - low)
        )

  q_val = q_threshold(reps, alpha)
  pass_coverage = (
      low_misses <= q_val
      and high_misses <= q_val
      and issued_count >= reps * 0.50
  )

  refusal_rate = float(reps - issued_count) / float(reps)
  cond_low_miss = (
      float(low_misses) / float(issued_count) if issued_count > 0 else 0.0
  )
  cond_high_miss = (
      float(high_misses) / float(issued_count) if issued_count > 0 else 0.0
  )

  return {
      "cell_name": cell["name"],
      "block": "V2",
      "sub_block": "coverage",
      "reps": reps,
      "alpha": alpha,
      "q_threshold": q_val,
      "G": g,
      "b_max": b_max,
      "r_ratio": r_ratio,
      "low_misses": low_misses,
      "high_misses": high_misses,
      "issued_count": issued_count,
      "refusal_rate": refusal_rate,
      "cond_low_miss_rate": cond_low_miss,
      "cond_high_miss_rate": cond_high_miss,
      "med_width": float(np.median(rel_widths)) if rel_widths else float("nan"),
      "passed": pass_coverage,
  }


def eval_v2_size_cell(cell: dict[str, Any], reps: int = 4000) -> dict[str, Any]:
  """Evaluates one V2 size cell measuring path check false-refutation rate."""
  cell_seed = cell["seed"]
  g = cell["G"]
  metric = cell["metric"]
  pattern = cell["pattern"]
  k = 20 * g

  cell_rng = np.random.default_rng(cell_seed)
  counts, b_max, r_ratio = _generate_v2_counts(pattern, g, k, cell_rng)

  n = int(np.sum(counts))
  cum_counts = np.cumsum(counts)
  c_prev = np.empty_like(counts)
  c_prev[0] = 0
  c_prev[1:] = cum_counts[:-1]
  w_k = (g * (2 * c_prev + counts)) // (2 * n)

  w_counts = np.bincount(w_k, weights=counts, minlength=g).astype(np.int64)
  path_edges = np.column_stack(
      [np.arange(g - 1, dtype=np.int64), np.arange(1, g, dtype=np.int64)]
  )

  refuted_count = 0
  for r in range(reps):
    rep_rng = np.random.default_rng([cell_seed, r])
    if metric == "mse":
      totals = rep_rng.chisquare(counts).astype(np.float64)
    else:
      totals = rep_rng.binomial(counts, 0.99).astype(np.float64)

    w_totals = np.bincount(w_k, weights=totals, minlength=g).astype(np.float64)
    n_tot = float(np.sum(w_counts))
    theta_hat = float(np.sum(w_totals)) / n_tot
    e = w_totals - w_counts.astype(np.float64) * theta_hat

    check = refutation.check_uncorrelated_edges(
        e, w_counts, path_edges, level=0.95
    )
    if check.outcome is refutation.Refutation.REFUTED:
      refuted_count += 1

  false_refutation_rate = float(refuted_count) / float(reps)

  return {
      "cell_name": cell["name"],
      "block": "V2",
      "sub_block": "size",
      "reps": reps,
      "G": g,
      "b_max": b_max,
      "r_ratio": r_ratio,
      "false_refutation_rate": false_refutation_rate,
      "refuted_count": refuted_count,
  }


def eval_v2_power_cell(
    cell: dict[str, Any], reps: int = 4000
) -> dict[str, Any]:
  """Evaluates one V2 power cell measuring refutation rate under AR(1) dependence."""
  cell_seed = cell["seed"]
  g = cell["G"]
  k = 20 * g

  cell_rng = np.random.default_rng(cell_seed)
  counts, b_max, r_ratio = _generate_v2_counts("bursty", g, k, cell_rng)
  n = int(np.sum(counts))
  m_bar = float(n) / float(g)
  rho = 1.0 - 1.0 / m_bar

  cum_counts = np.cumsum(counts)
  c_prev = np.empty_like(counts)
  c_prev[0] = 0
  c_prev[1:] = cum_counts[:-1]
  w_k = (g * (2 * c_prev + counts)) // (2 * n)
  w_counts = np.bincount(w_k, weights=counts, minlength=g).astype(np.int64)
  path_edges = np.column_stack(
      [np.arange(g - 1, dtype=np.int64), np.arange(1, g, dtype=np.int64)]
  )

  b_endpoints = np.cumsum(np.concatenate([[0], counts]))

  refuted_count = 0
  for r in range(reps):
    rep_rng = np.random.default_rng([cell_seed, r])
    u = rep_rng.standard_normal(n)
    if rho != 0.0:
      u[1:] *= math.sqrt(1.0 - rho**2)
      e = scipy.signal.lfilter([1.0], [1.0, -rho], u)
    else:
      e = u
    e2 = e**2

    # Bucket totals
    k_buckets = len(counts)
    totals = np.empty(k_buckets, dtype=np.float64)
    for b_idx in range(k_buckets):
      totals[b_idx] = np.sum(e2[b_endpoints[b_idx] : b_endpoints[b_idx + 1]])

    w_totals = np.bincount(w_k, weights=totals, minlength=g).astype(np.float64)

    n_tot = float(np.sum(w_counts))
    theta_hat = float(np.sum(w_totals)) / n_tot
    e_res = w_totals - w_counts.astype(np.float64) * theta_hat

    check = refutation.check_uncorrelated_edges(
        e_res, w_counts, path_edges, level=0.95
    )
    if check.outcome is refutation.Refutation.REFUTED:
      refuted_count += 1

  power_rate = float(refuted_count) / float(reps)

  return {
      "cell_name": cell["name"],
      "block": "V2",
      "sub_block": "power",
      "reps": reps,
      "G": g,
      "power_refutation_rate": power_rate,
      "refuted_count": refuted_count,
  }


# -----------------------------------------------------------------------------
# Block V3 Evaluation
# -----------------------------------------------------------------------------


def _build_geometric_graph(n: int, mean_deg: float, rng: Any) -> np.ndarray:
  """Builds random geometric graph on n nodes with specified mean degree."""
  pts = rng.uniform(0.0, 1.0, size=(n, 2))
  r = math.sqrt(mean_deg / (float(n) * math.pi))
  tree = scipy.spatial.cKDTree(pts)
  pairs = list(tree.query_pairs(r))
  if not pairs:
    return np.zeros((0, 2), dtype=np.int64)
  edges = np.array(pairs, dtype=np.int64)
  return np.unique(np.sort(edges, axis=1), axis=0)


def _build_barabasi_albert(n: int, m: int, rng: Any) -> np.ndarray:
  """Builds Barabasi-Albert preferential attachment graph on n nodes."""
  edges = []
  repeated_nodes = []
  for i in range(m):
    for j in range(i + 1, m):
      edges.append([i, j])
      repeated_nodes.extend([i, j])

  for source in range(m, n):
    targets = set()
    num_repeated = len(repeated_nodes)
    while len(targets) < m:
      pick = int(repeated_nodes[int(rng.integers(num_repeated))])
      if pick != source:
        targets.add(pick)
    for target in targets:
      edges.append([min(source, target), max(source, target)])
    repeated_nodes.extend(list(targets))
    repeated_nodes.extend([source] * m)

  return np.unique(np.array(edges, dtype=np.int64), axis=0)


def eval_v3_cell(cell: dict[str, Any], reps: int = 2000) -> dict[str, Any]:
  """Evaluates one V3 topological kappa cell across 2000 reps."""
  cell_seed = cell["seed"]
  n = cell["n"]
  graph_type = cell["graph_type"]
  w = cell["w"]
  alpha = 0.05

  cell_rng = np.random.default_rng(cell_seed)
  if graph_type == "geometric":
    edges = _build_geometric_graph(n, mean_deg=8.0, rng=cell_rng)
  else:
    edges = _build_barabasi_albert(n, m=3, rng=cell_rng)

  c_ids = partitioners.partition_graph(n, edges, target_cluster_size=20, seed=0)
  c_edges = graph.cluster_adjacency(n, edges, c_ids)
  g = int(np.max(c_ids)) + 1

  deg = np.bincount(edges[:, 0], minlength=n) + np.bincount(
      edges[:, 1], minlength=n
  )
  theta_true = float(np.mean(deg.astype(np.float64) * (w**2) + 1.0))

  # Edge phi = rho_ij^2
  d_u = deg[edges[:, 0]].astype(np.float64)
  d_v = deg[edges[:, 1]].astype(np.float64)
  rho_uv = (w**2) / np.sqrt((d_u * (w**2) + 1.0) * (d_v * (w**2) + 1.0))
  phi = rho_uv**2

  top_res = graph.topological_kappa(n, edges, c_ids, phi)
  kappa_t = top_res.kappa

  # Exact cluster VIF
  c_u = c_ids[edges[:, 0]]
  c_v = c_ids[edges[:, 1]]
  int_edges_count = int(np.count_nonzero(c_u == c_v))
  var_s_nodes = 2.0 * ((deg.astype(np.float64) * (w**2) + 1.0) ** 2)
  v_tot = float(np.sum(var_s_nodes) + 4.0 * (w**4) * len(edges))
  sum_v_c = float(np.sum(var_s_nodes) + 4.0 * (w**4) * int_edges_count)
  vif_exact = float(v_tot / sum_v_c)

  num_e = len(edges)

  issued_count = 0
  low_misses = 0
  high_misses = 0
  rel_widths = []

  # Miss rate if declared at kappa = 1
  k1_low_misses = 0
  k1_high_misses = 0

  for r in range(reps):
    rep_rng = np.random.default_rng([cell_seed, r])
    eta = rep_rng.standard_normal(n)
    xi = rep_rng.standard_normal(num_e) if num_e > 0 else np.zeros(0)
    sum_xi = np.bincount(edges[:, 0], weights=xi, minlength=n) + np.bincount(
        edges[:, 1], weights=xi, minlength=n
    )
    x = w * sum_xi + eta
    data = x

    # Library interval with kappa = kappa_t
    res = graph.graph_interval(
        data, c_ids, c_edges, metric="mse", level=0.95, kappa=kappa_t
    )
    low, high = res.interval.low, res.interval.high

    is_refused = math.isnan(low) and math.isnan(high)
    if not is_refused:
      issued_count += 1
      if math.isfinite(low) and theta_true < low:
        low_misses += 1
      if math.isfinite(high) and theta_true > high:
        high_misses += 1
      if math.isfinite(low) and math.isfinite(high):
        rel_widths.append(
            (high - low) / theta_true if theta_true > 0 else (high - low)
        )

    # Comparison under kappa = 1.0
    res_k1 = graph.graph_interval(
        data, c_ids, c_edges, metric="mse", level=0.95, kappa=1.0
    )
    if math.isfinite(res_k1.interval.low) and theta_true < res_k1.interval.low:
      k1_low_misses += 1
    if (
        math.isfinite(res_k1.interval.high)
        and theta_true > res_k1.interval.high
    ):
      k1_high_misses += 1

  q_val = q_threshold(reps, alpha)
  pass_coverage = (
      low_misses <= q_val
      and high_misses <= q_val
      and issued_count >= reps * 0.50
  )

  refusal_rate = float(reps - issued_count) / float(reps)
  cond_low_miss = (
      float(low_misses) / float(issued_count) if issued_count > 0 else 0.0
  )
  cond_high_miss = (
      float(high_misses) / float(issued_count) if issued_count > 0 else 0.0
  )

  return {
      "cell_name": cell["name"],
      "block": "V3",
      "reps": reps,
      "alpha": alpha,
      "q_threshold": q_val,
      "graph_type": graph_type,
      "w": w,
      "theta_true": theta_true,
      "kappa_T": kappa_t,
      "vif_exact": vif_exact,
      "low_misses": low_misses,
      "high_misses": high_misses,
      "issued_count": issued_count,
      "refusal_rate": refusal_rate,
      "cond_low_miss_rate": cond_low_miss,
      "cond_high_miss_rate": cond_high_miss,
      "med_width": float(np.median(rel_widths)) if rel_widths else float("nan"),
      "k1_low_miss_rate": float(k1_low_misses) / float(reps),
      "k1_high_miss_rate": float(k1_high_misses) / float(reps),
      "k1_total_miss_rate": float(k1_low_misses + k1_high_misses) / float(reps),
      "passed": pass_coverage,
  }


# -----------------------------------------------------------------------------
# Block V4 Evaluation
# -----------------------------------------------------------------------------


def eval_v4_cell(cell: dict[str, Any], reps: int = 2000) -> dict[str, Any]:
  """Evaluates one V4 AR(1) cell across 2000 reps.

  Cells with `library_only=True` skip the textbook intervals and pass the
  cell's `m_item` to the library (M taken from the known generating law).
  """
  cell_seed = cell["seed"]
  n = 100000
  g = 4000
  m_c = n // g  # 25
  rho = cell["rho"]
  m_item = cell.get("m_item")
  library_only = bool(cell.get("library_only", False))
  theta_true = 1.0

  vif_exact = _ar1_window_vif(rho**2, m_c, n)
  t_crit = float(scipy.stats.t.ppf(0.975, df=n - 1))

  t_low_miss = 0
  t_high_miss = 0
  t_widths = []

  b_low_miss = 0
  b_high_miss = 0
  b_widths = []

  l_low_miss = 0
  l_high_miss = 0
  l_widths = []
  l_refused = 0

  for r in range(reps):
    rep_rng = np.random.default_rng([cell_seed, r])
    u = rep_rng.standard_normal(n)
    if rho != 0.0:
      u[1:] *= math.sqrt(1.0 - rho**2)
      e = scipy.signal.lfilter([1.0], [1.0, -rho], u)
    else:
      e = u
    s = e**2

    if not library_only:
      # 1. Student-t
      mean_s = float(np.mean(s))
      se_s = math.sqrt(float(np.var(s, ddof=1)) / n)
      low_t, high_t = mean_s - t_crit * se_s, mean_s + t_crit * se_s
      if theta_true < low_t:
        t_low_miss += 1
      if theta_true > high_t:
        t_high_miss += 1
      t_widths.append(high_t - low_t)

      # 2. Bootstrap (B=500)
      boot_rng = np.random.default_rng([cell_seed, r, 2])
      boot_idx = boot_rng.integers(0, n, size=(500, n))
      boot_means = np.mean(s[boot_idx], axis=1)
      low_boot = float(np.percentile(boot_means, 2.5))
      high_boot = float(np.percentile(boot_means, 97.5))
      if theta_true < low_boot:
        b_low_miss += 1
      if theta_true > high_boot:
        b_high_miss += 1
      b_widths.append(high_boot - low_boot)

    # 3. Library
    res_lib = temporal.temporal_interval(
        e,
        num_windows=g,
        metric="mse",
        level=0.95,
        kappa=vif_exact,
        m_item=m_item,
    )
    low_lib, high_lib = res_lib.low, res_lib.high
    if math.isnan(low_lib) and math.isnan(high_lib):
      l_refused += 1
    else:
      if math.isfinite(low_lib) and theta_true < low_lib:
        l_low_miss += 1
      if math.isfinite(high_lib) and theta_true > high_lib:
        l_high_miss += 1
      if math.isfinite(low_lib) and math.isfinite(high_lib):
        l_widths.append(high_lib - low_lib)

  out = {
      "cell_name": cell["name"],
      "block": "V4",
      "reps": reps,
      "rho": rho,
      "vif_exact": vif_exact,
      "m_item": m_item,
      "library_only": library_only,
  }
  if not library_only:
    out["student_t"] = {
        "lower_miss_rate": float(t_low_miss) / float(reps),
        "upper_miss_rate": float(t_high_miss) / float(reps),
        "total_miss_rate": float(t_low_miss + t_high_miss) / float(reps),
        "med_rel_width": float(np.median(t_widths)),
    }
    out["bootstrap"] = {
        "lower_miss_rate": float(b_low_miss) / float(reps),
        "upper_miss_rate": float(b_high_miss) / float(reps),
        "total_miss_rate": float(b_low_miss + b_high_miss) / float(reps),
        "med_rel_width": float(np.median(b_widths)),
    }
  # Library miss rates are over all reps (refusals count as non-misses).
  out["library"] = {
      "refused_count": l_refused,
      "refusal_rate": float(l_refused) / float(reps),
      "low_miss_count": l_low_miss,
      "high_miss_count": l_high_miss,
      "lower_miss_rate": float(l_low_miss) / float(reps),
      "upper_miss_rate": float(l_high_miss) / float(reps),
      "total_miss_rate": float(l_low_miss + l_high_miss) / float(reps),
      "med_rel_width": (
          float(np.median(l_widths)) if l_widths else float("nan")
      ),
  }
  return out


def run_cell_task(c: dict[str, Any], reps_override: int = 0) -> dict[str, Any]:
  """Executes one validation cell."""
  blk = c["block"]
  if blk in ("V1", "V5"):
    reps = reps_override if reps_override > 0 else 2000
    return eval_v1_cell(c, reps=reps)
  elif blk == "V2":
    sub = c["sub_block"]
    if sub == "coverage":
      reps = reps_override if reps_override > 0 else 4000
      return eval_v2_coverage_cell(c, reps=reps)
    elif sub == "size":
      reps = reps_override if reps_override > 0 else 4000
      return eval_v2_size_cell(c, reps=reps)
    elif sub == "power":
      reps = reps_override if reps_override > 0 else 4000
      return eval_v2_power_cell(c, reps=reps)
    else:
      raise ValueError(f"Unknown V2 sub_block {sub}")
  elif blk == "V3":
    reps = reps_override if reps_override > 0 else 2000
    return eval_v3_cell(c, reps=reps)
  elif blk == "V4":
    reps = reps_override if reps_override > 0 else 2000
    return eval_v4_cell(c, reps=reps)
  else:
    raise ValueError(f"Unknown block {blk}")


# -----------------------------------------------------------------------------
# Main Runner & Analysis
# -----------------------------------------------------------------------------


def get_all_cells(seed_base: int = DEFAULT_SEED) -> list[dict[str, Any]]:
  """Constructs the full list of 69 validation cells across blocks V1-V5."""
  cells = []
  cell_idx = 0

  # Block V1a (16 cells)
  api_toggle = 0
  for g in (2500, 6000):
    for m_bar in (5, 20):
      for icc in ((0.2, 0.2), (0.8, 0.5)):
        for law in ("gaussian", "laplace"):
          api = "graph" if (api_toggle % 2 == 0) else "temporal"
          api_toggle += 1
          cells.append({
              "cell_id": cell_idx,
              "name": (
                  f"V1a_G={g}_m={m_bar}_icc={icc[0]}-{icc[1]}_{law}_{api}"
              ),
              "block": "V1",
              "sub_block": "V1a",
              "seed": seed_base + cell_idx,
              "G": g,
              "m_bar": m_bar,
              "icc": icc,
              "law": law,
              "api": api,
              "alpha": 0.05,
          })
          cell_idx += 1

  # Block V1b (4 cells)
  for g in (2000, 4000):
    for rho in ((0.5, 0.5), (0.9, 0.7)):
      cells.append({
          "cell_id": cell_idx,
          "name": f"V1b_G={g}_rho={rho[0]}-{rho[1]}",
          "block": "V1",
          "sub_block": "V1b",
          "seed": seed_base + cell_idx,
          "n": 120000,
          "G": g,
          "rho": rho,
          "alpha": 0.05,
      })
      cell_idx += 1

  # Block V1c (1 cell): alpha = 0.01 needs G = 12000 to clear the sample-size floor.
  cells.append({
      "cell_id": cell_idx,
      "name": "V1c_G=12000_m=20_icc=0.8-0.5_gaussian_alpha=0.01",
      "block": "V1",
      "sub_block": "V1c",
      "seed": seed_base + cell_idx,
      "G": 12000,
      "m_bar": 20,
      "icc": (0.8, 0.5),
      "law": "gaussian",
      "api": "temporal",
      "alpha": 0.01,
  })
  cell_idx += 1

  # Block V2 coverage (8 cells)
  for g in (2000, 4000):
    for metric in ("mse", "accuracy"):
      for pattern in ("bursty", "trending"):
        cells.append({
            "cell_id": cell_idx,
            "name": f"V2_cov_G={g}_{metric}_{pattern}",
            "block": "V2",
            "sub_block": "coverage",
            "seed": seed_base + cell_idx,
            "G": g,
            "metric": metric,
            "pattern": pattern,
        })
        cell_idx += 1

  # Block V2 size (16 cells)
  for g in (31, 50, 100, 300):
    for metric in ("mse", "accuracy"):
      for pattern in ("bursty", "trending"):
        cells.append({
            "cell_id": cell_idx,
            "name": f"V2_size_G={g}_{metric}_{pattern}",
            "block": "V2",
            "sub_block": "size",
            "seed": seed_base + cell_idx,
            "G": g,
            "metric": metric,
            "pattern": pattern,
        })
        cell_idx += 1

  # Block V2 power (4 cells)
  for g in (31, 50, 100, 300):
    cells.append({
        "cell_id": cell_idx,
        "name": f"V2_power_G={g}_mse_bursty",
        "block": "V2",
        "sub_block": "power",
        "seed": seed_base + cell_idx,
        "G": g,
    })
    cell_idx += 1

  # Block V3 (4 cells)
  for g_type in ("geometric", "barabasi_albert"):
    for w in (0.3, 1.0):
      cells.append({
          "cell_id": cell_idx,
          "name": f"V3_{g_type}_w={w}",
          "block": "V3",
          "seed": seed_base + cell_idx,
          "n": 40000,
          "graph_type": g_type,
          "w": w,
      })
      cell_idx += 1

  # Block V4 (3 cells)
  v4_seeds: dict[float, int] = {}
  for rho in (0.5, 0.8, 0.95):
    v4_seeds[rho] = seed_base + cell_idx
    cells.append({
        "cell_id": cell_idx,
        "name": f"V4_rho={rho}",
        "block": "V4",
        "seed": seed_base + cell_idx,
        "rho": rho,
    })
    cell_idx += 1

  # Block V5 (8 cells)
  v5_idx = 0
  for g in (2500, 6000):
    for m_bar in (5, 20):
      for icc in ((0.2, 0.2), (0.8, 0.5)):
        cells.append({
            "cell_id": 100 + v5_idx,
            "name": (
                f"V5_G={g}_m={m_bar}_icc={icc[0]}-{icc[1]}_gaussian_graph_m=3"
            ),
            "block": "V5",
            "sub_block": "V5",
            "seed": seed_base + 100 + v5_idx,
            "G": g,
            "m_bar": m_bar,
            "icc": icc,
            "law": "gaussian",
            "api": "graph",
            "m": 3.0,
            "m_labels": 3.0,
            "alpha": 0.05,
        })
        v5_idx += 1

  # Larger-G V1c cells (2 cells). Appended with new ids so that the seeds of
  # the cells above are unchanged.
  for extra_idx, g in enumerate((25000, 50000)):
    cells.append({
        "cell_id": 200 + extra_idx,
        "name": f"V1c_G={g}_m=20_icc=0.8-0.5_gaussian_alpha=0.01",
        "block": "V1",
        "sub_block": "V1c",
        "seed": seed_base + 200 + extra_idx,
        "G": g,
        "m_bar": 20,
        "icc": (0.8, 0.5),
        "law": "gaussian",
        "api": "temporal",
        "alpha": 0.01,
    })

  # V4 with M from the known law (3 cells, library only). For Gaussian e,
  # E[e^4] / E[e^2]^2 = 3; rounded up plus one gives m_item = 4. Each cell
  # reuses the seed of the matching default-M V4 cell, so the library sees
  # identical data.
  for extra_idx, rho in enumerate((0.5, 0.8, 0.95)):
    cells.append({
        "cell_id": 210 + extra_idx,
        "name": f"V4_rho={rho}_m_item=4",
        "block": "V4",
        "seed": v4_seeds[rho],
        "rho": rho,
        "m_item": 4.0,
        "library_only": True,
    })

  return cells


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  mode = str(_MODE.value)
  seed_base = int(_SEED.value)
  workers = min(int(_WORKERS.value or 4), os.cpu_count() or 4)
  reps_override = int(_REPS.value) if _REPS.value else 0
  output_path = _OUTPUT.value

  print(
      f"Setting up validation suite (mode={mode},"
      f" seed_base=0x{seed_base:X}, workers={workers})..."
  )

  cells = get_all_cells(seed_base)
  print(f"Constructed {len(cells)} validation cells across blocks V1-V5.")
  if _CELLS.value:
    pattern = re.compile(_CELLS.value)
    cells = [c for c in cells if pattern.search(c["name"])]
    print(f"Selected {len(cells)} cells matching --cells={_CELLS.value!r}.")

  if mode == "quick":
    blocks_seen = set()
    selected_cells = []
    for c in cells:
      if c["block"] not in blocks_seen:
        blocks_seen.add(c["block"])
        selected_cells.append(c)
    cells = selected_cells
    reps_override = 20

  start_t = time.time()
  results: list[dict[str, Any]] = []

  if workers > 1:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
      future_to_cell = {
          executor.submit(run_cell_task, cell, reps_override): cell
          for cell in cells
      }
      for future in concurrent.futures.as_completed(future_to_cell):
        cell = future_to_cell[future]
        try:
          res = future.result()
          results.append(res)
          print(f"Completed {cell['name']} ({len(results)}/{len(cells)})")
        except Exception as e:
          print(f"Error on cell {cell['name']}: {e}", file=sys.stderr)
          raise
  else:
    for idx, cell in enumerate(cells):
      res = run_cell_task(cell, reps_override)
      results.append(res)
      print(f"Completed {cell['name']} ({idx + 1}/{len(cells)})")

  results.sort(key=lambda r: [c["name"] for c in cells].index(r["cell_name"]))

  if output_path:
    parent = os.path.dirname(output_path)
    if parent:
      os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
      for r in results:
        f.write(json.dumps(r) + "\n")
    print(f"\nResults written to {output_path} ({len(results)} rows).")


  # Summary table
  print("\n" + "=" * 125, flush=True)
  print(
      f"SUMMARY RESULTS TABLE (mode={mode}, total elapsed: {time.time() - start_t:.1f}s)",
      flush=True,
  )
  print("=" * 125, flush=True)
  print(
      f"{'Cell Name':<50} {'Block':<6} {'Reps':<6} {'Coverage':<12} {'Miss Rate':<12} {'Passed':<8}",
      flush=True,
  )
  print("-" * 125, flush=True)
  for r in results:
    cov_str = f"{r.get('coverage', 0.0):.2%}" if "coverage" in r else "N/A"
    miss_str = f"{r.get('miss_rate', 0.0):.2%}" if "miss_rate" in r else "N/A"
    if "passed" not in r:
      pass_str = "-"
    else:
      pass_str = "PASS" if r["passed"] else "FAIL"
    reps_cnt = r.get("reps", 0)
    print(
        f"{r['cell_name']:<50} {r.get('block', ''):<6} {reps_cnt:<6d} {cov_str:<12} {miss_str:<12} {pass_str:<8}",
        flush=True,
    )
  print("-" * 125, flush=True)


if __name__ == "__main__":
  app.run(main)
