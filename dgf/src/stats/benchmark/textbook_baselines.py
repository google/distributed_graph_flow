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

"""Compares textbook intervals with the library intervals under dependence.

Evaluates the Student-t interval and the i.i.d. percentile bootstrap against
the library's temporal (T1: AR(1) cells) and graph (G2: public real-graph
topologies) intervals. Library misses are counted over all repetitions,
together with refusals, refuted results and misses that were not refuted.
"""

from collections.abc import Sequence
import concurrent.futures
import json
import math
import os
import sys
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import scipy.signal
import scipy.stats

from dgf.src.stats import graph
from dgf.src.stats import partitioners
from dgf.src.stats import refutation
from dgf.src.stats import temporal
from dgf.src.stats.benchmark import coverage_dependent
from dgf.src.stats.benchmark import coverage_dependent_truth as truth
from dgf.src.stats.benchmark import real_graph_topology as real_graph

DEFAULT_SEED = 54077361271078  # = 0x312edde8b126

_SEED = (
    flags.FLAGS["seed"]
    if "seed" in flags.FLAGS
    else flags.DEFINE_integer(
        "seed", DEFAULT_SEED, "Fixed random seed for evaluation."
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
_BOOT_B = (
    flags.FLAGS["boot_b"]
    if "boot_b" in flags.FLAGS
    else flags.DEFINE_integer(
        "boot_b", 500, "Number of bootstrap resamples (B=500)."
    )
)
_WORKERS = (
    flags.FLAGS["workers"]
    if "workers" in flags.FLAGS
    else flags.DEFINE_integer(
        "workers", min(16, os.cpu_count() or 4), "Number of parallel workers."
    )
)
_DGF_CACHE_DIR = (
    flags.FLAGS["dgf_cache_dir"]
    if "dgf_cache_dir" in flags.FLAGS
    else flags.DEFINE_string(
        "dgf_cache_dir",
        "/tmp/dgf_cache",
        "Directory for DGF dataset caching.",
    )
)
_BLOCK = flags.DEFINE_string(
    "block",
    "all",
    "Block to evaluate ('all', 'T1', or 'G2').",
)


def _eval_cell(
    cell: dict[str, Any],
    base_seed: int,
    reps: int,
    boot_b: int,
    cache_dir: str,
) -> dict[str, Any]:
  """Evaluates one cell across reps for Student-t, bootstrap, and library intervals."""
  c_id = cell["cell_id"]
  b_id = cell["block"]
  metric = cell["metric"]
  scale = cell.get("scale", 1.0)
  p = cell.get("p", 0.5)

  phi = 0.0
  k_val = 1.0
  m_val = None
  theta_true = 0.0
  graph_data = None
  n = 0

  # Precompute structure and ground truth
  if b_id == "T1":
    n = cell["n"]
    g = cell["G"]
    l_param = cell["L"]
    counts = np.bincount(
        (np.arange(n, dtype=np.int64) * g) // n, minlength=g
    ).astype(np.int64)
    phi = 0.0 if l_param == 1 else 1.0 - 1.0 / float(l_param)
    gt = truth.temporal_ground_truth(metric, counts, phi, sigma=scale, p=p)
    k_mode = cell["kappa_mode"]
    m_mode = cell["m_mode"]
    k_val = 1.0 if k_mode == "one" else max(1.0, gt["vif_true"])
    m_val = None if m_mode == "default" else gt["m_c_true"]
    theta_true = float(gt["theta"])
    graph_data = None
  elif b_id == "G2":
    task_short = cell["task_short"]
    graph_obj, schema_obj = real_graph.fetch_g2_graph(
        task_short, cache_dir=cache_dir
    )
    unique_edges = real_graph.extract_task_edges(
        graph_obj, schema_obj, target_nodeset="nodes"
    )
    test_idxs = real_graph.extract_test_node_indices(
        graph_obj, target_nodeset="nodes"
    )
    edges, stats = real_graph.build_test_subgraph(unique_edges, test_idxs)
    n_nodes = int(stats["node_count"])
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
    k_val = 1.0
    m_val = gt["m_c_true"]
    theta_true = float(gt["theta"])
    graph_data = (edges, n_nodes, c_ids, c_edges)
    n = n_nodes
  else:
    raise ValueError(f"Unsupported block: {b_id}")

  # Precompute t critical value
  t_crit = float(scipy.stats.t.ppf(0.975, df=n - 1))

  # Arrays to record per-draw results
  t_low_miss = 0
  t_high_miss = 0
  t_rel_widths = []

  b_low_miss = 0
  b_high_miss = 0
  b_rel_widths = []

  l_low_miss = 0
  l_high_miss = 0
  l_rel_widths = []
  l_refused = 0
  l_issued = 0
  l_unrefuted_miss = 0  # miss with status UNREFUTED and kappa check not REFUTED
  l_tail_unresolved = 0
  l_kappa_refuted = 0

  for r in range(reps):
    data_rng = np.random.default_rng([base_seed, c_id, r])
    data = np.zeros(n, dtype=np.float64)
    summands = np.zeros(n, dtype=np.float64)
    low_lib = float("nan")
    high_lib = float("nan")
    lib_status_name = ""
    lib_kappa_refuted = False

    # Data generation matches coverage_dependent for the same cell.
    if b_id == "T1":
      u = data_rng.standard_normal(n)
      if phi != 0.0:
        u[1:] *= math.sqrt(1.0 - phi**2)
        u = scipy.signal.lfilter([1.0], [1.0, -phi], u)
      if metric != "accuracy":
        data = u * scale
        summands = data**2
      else:
        data = (u < float(scipy.stats.norm.ppf(p))).astype(np.float64)
        summands = data
    elif b_id == "G2":
      assert graph_data is not None
      edges, n_nodes, c_ids, c_edges = graph_data
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
      data = u * scale
      summands = data**2

    # 1. Student-t on item summands
    mean_s = float(np.mean(summands))
    var_s = float(np.var(summands, ddof=1))
    se_s = math.sqrt(var_s / n) if var_s > 0 else 0.0
    low_t = mean_s - t_crit * se_s
    high_t = mean_s + t_crit * se_s

    if theta_true < low_t:
      t_low_miss += 1
    if theta_true > high_t:
      t_high_miss += 1
    t_rel_widths.append(
        (high_t - low_t) / theta_true if theta_true > 0 else (high_t - low_t)
    )

    # 2. i.i.d. Percentile Bootstrap (B resamples)
    boot_rng = np.random.default_rng([base_seed, c_id, r, 2])
    boot_idx = boot_rng.integers(0, n, size=(boot_b, n))
    boot_means = np.mean(summands[boot_idx], axis=1)
    low_boot = float(np.percentile(boot_means, 2.5))
    high_boot = float(np.percentile(boot_means, 97.5))

    if theta_true < low_boot:
      b_low_miss += 1
    if theta_true > high_boot:
      b_high_miss += 1
    b_rel_widths.append(
        (high_boot - low_boot) / theta_true
        if theta_true > 0
        else (high_boot - low_boot)
    )

    # 3. Library interval with cell's own declaration
    if b_id == "T1":
      res_lib = temporal.temporal_interval(
          data,
          num_windows=cell["G"],
          metric=metric,
          level=0.95,
          m=m_val,
          kappa=k_val,
      )
      low_lib, high_lib = res_lib.low, res_lib.high
      lib_status_name = res_lib.status.name
      lib_kappa_refuted = (
          res_lib.kappa_check.outcome is refutation.Refutation.REFUTED
      )
    elif b_id == "G2":
      assert graph_data is not None
      _, _, c_ids, c_edges = graph_data
      res_lib = graph.graph_interval(
          data,
          c_ids,
          c_edges,
          metric="mse",
          level=0.95,
          m=m_val,
          kappa=k_val,
      )
      low_lib, high_lib = res_lib.interval.low, res_lib.interval.high
      lib_status_name = res_lib.interval.status.name
      lib_kappa_refuted = (
          res_lib.kappa_check.outcome is refutation.Refutation.REFUTED
      )

    if lib_status_name == "TAIL_UNRESOLVED":
      l_tail_unresolved += 1
    if lib_kappa_refuted:
      l_kappa_refuted += 1

    # An interval with one finite and one infinite endpoint is issued. Count its finite side normally.
    # It is refused only if both endpoints are NaN.
    is_refused = math.isnan(low_lib) and math.isnan(high_lib)
    if is_refused:
      l_refused += 1
    else:
      l_issued += 1
      is_lib_miss = False
      if math.isfinite(low_lib) and theta_true < low_lib:
        l_low_miss += 1
        is_lib_miss = True
      if math.isfinite(high_lib) and theta_true > high_lib:
        l_high_miss += 1
        is_lib_miss = True
      if (
          is_lib_miss
          and lib_status_name == "UNREFUTED"
          and not lib_kappa_refuted
      ):
        l_unrefuted_miss += 1
      if math.isfinite(low_lib) and math.isfinite(high_lib):
        l_rel_widths.append(
            (high_lib - low_lib) / theta_true
            if theta_true > 0
            else (high_lib - low_lib)
        )
      else:
        l_rel_widths.append(float("inf"))

  refusal_rate = float(l_refused) / float(reps)
  uncond_low_miss = float(l_low_miss) / float(reps)
  uncond_high_miss = float(l_high_miss) / float(reps)
  uncond_total_miss = float(l_low_miss + l_high_miss) / float(reps)

  if l_issued > 0:
    cond_low_miss = float(l_low_miss) / float(l_issued)
    cond_high_miss = float(l_high_miss) / float(l_issued)
    cond_total_miss = float(l_low_miss + l_high_miss) / float(l_issued)
  else:
    cond_low_miss = float("nan")
    cond_high_miss = float("nan")
    cond_total_miss = float("nan")

  med_l_width = float(np.median(l_rel_widths)) if l_rel_widths else float("nan")

  return {
      "cell_id": c_id,
      "cell_name": cell["name"],
      "block": b_id,
      "family": cell["family"],
      "metric": metric,
      "n": n,
      "theta_true": theta_true,
      "student_t": {
          "lower_miss_rate": float(t_low_miss) / float(reps),
          "upper_miss_rate": float(t_high_miss) / float(reps),
          "total_miss_rate": float(t_low_miss + t_high_miss) / float(reps),
          "med_rel_width": float(np.median(t_rel_widths)),
      },
      "bootstrap": {
          "lower_miss_rate": float(b_low_miss) / float(reps),
          "upper_miss_rate": float(b_high_miss) / float(reps),
          "total_miss_rate": float(b_low_miss + b_high_miss) / float(reps),
          "med_rel_width": float(np.median(b_rel_widths)),
      },
      "library": {
          # Plain rates use all repetitions as the denominator, as for the
          # textbook methods above. Refused repetitions (no interval) are
          # not misses; they are counted in refusal_rate.
          "refusal_rate": refusal_rate,
          "lower_miss_rate": uncond_low_miss,
          "upper_miss_rate": uncond_high_miss,
          "total_miss_rate": uncond_total_miss,
          "unrefuted_miss_rate": float(l_unrefuted_miss) / float(reps),
          "num_lower_miss": l_low_miss,
          "num_upper_miss": l_high_miss,
          "num_unrefuted_miss": l_unrefuted_miss,
          "num_refused": l_refused,
          "num_tail_unresolved": l_tail_unresolved,
          "num_kappa_refuted": l_kappa_refuted,
          # Rates among issued intervals only (conditional; not guaranteed).
          "conditional_lower_miss_rate": cond_low_miss,
          "conditional_upper_miss_rate": cond_high_miss,
          "conditional_total_miss_rate": cond_total_miss,
          "med_rel_width": med_l_width,
      },
  }


def get_target_cells(seed: int, block_filter: str = "all") -> list[dict[str, Any]]:
  t1_cells = coverage_dependent.generate_t1_fixed_cells()
  g2_cells = []
  real_graph_tasks = [
      ("hm-prices", "graphland-hm-prices"),
      ("avazu-ctr", "graphland-avazu-ctr"),
      ("ogbn-arxiv", "ogbn-arxiv"),
      ("city-roads-L", "graphland-city-roads-L"),
  ]
  global_id = len(t1_cells)
  for task_short, task_full in real_graph_tasks:
    for gamma in (0.0, 0.5):
      g2_cells.append({
          "cell_id": global_id,
          "block": "G2",
          "family": "graph_real",
          "name": f"G2_{task_short}_gamma={gamma}",
          "task_short": task_short,
          "task_full": task_full,
          "gamma": gamma,
          "metric": "mse",
          "t": 50,
          "alpha": 0.05,
          "m_mode": "tight",
          "kappa_mode": "one",
          "scale": 1.0,
          "modes": ["g_data", "g_merged"],
      })
      global_id += 1

  target = []
  if block_filter in ("all", "T1"):
    target.extend(t1_cells)
  if block_filter in ("all", "G2"):
    target.extend(g2_cells)
  return target


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  base_seed = _SEED.value if _SEED.value is not None else DEFAULT_SEED
  mode = str(_MODE.value)
  workers = _WORKERS.value if _WORKERS.value is not None else 4
  output_path = _OUTPUT.value
  cache_dir = str(_DGF_CACHE_DIR.value)
  block_filter = str(_BLOCK.value or "all")

  all_target_cells = get_target_cells(base_seed, block_filter)

  if mode == "quick":
    # 1 cell per block: 1 from T1, 1 from G2
    t1_subset = [c for c in all_target_cells if c["block"] == "T1"][:1]
    g2_subset = [c for c in all_target_cells if c["block"] == "G2"][:1]
    target_cells = t1_subset + g2_subset
    reps = 20
    boot_b = 50
  else:
    target_cells = all_target_cells
    reps = 2000
    boot_b = _BOOT_B.value if _BOOT_B.value is not None else 500

  print(
      f"=== Textbook Baselines Benchmark (mode={mode}, seed={base_seed},"
      f" reps={reps}, boot_b={boot_b}, workers={workers}, cells={len(target_cells)}) ===",
      flush=True,
  )

  start_time = time.time()
  results: list[dict[str, Any]] = []

  if workers > 1:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
      future_to_cell = {
          executor.submit(
              _eval_cell, cell, base_seed, reps, boot_b, cache_dir
          ): cell
          for cell in target_cells
      }
      for future in concurrent.futures.as_completed(future_to_cell):
        cell = future_to_cell[future]
        try:
          res = future.result()
          results.append(res)
          print(
              f"  Completed {cell['name']} ({len(results)}/{len(target_cells)})",
              flush=True,
          )
        except Exception as e:
          print(f"Error evaluating cell {cell['name']}: {e}", file=sys.stderr)
          raise
  else:
    for idx, cell in enumerate(target_cells):
      res = _eval_cell(cell, base_seed, reps, boot_b, cache_dir)
      results.append(res)
      print(
          f"  Completed {cell['name']} ({idx + 1}/{len(target_cells)})",
          flush=True,
      )

  # Sort results by cell_id to maintain canonical ordering
  results.sort(key=lambda r: r["cell_id"])

  if output_path:
    parent = os.path.dirname(output_path)
    if parent:
      os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
      for row in results:
        f.write(json.dumps(row) + "\n")
    print(f"Results written to {output_path} ({len(results)} rows).")

  # Print formatted stdout table
  header = (
      f"{'Cell Name':<30} {'Block':<6} {'Metric':<8} {'Theta':<7} "
      f"{'Student-t Miss':<16} {'t-W':<7} "
      f"{'Boot Miss':<16} {'B-W':<7} "
      f"{'Lib Miss':<16} {'Lib Unref.':<10} {'Lib Refuse':<10} {'Lib-W':<7}"
  )
  sep = "-" * len(header)
  print("\n" + sep, flush=True)
  print(header, flush=True)
  print(sep, flush=True)
  for r in results:
    c_name = r["cell_name"]
    block = r.get("block", "")
    metric = r["metric"]
    th = f"{r['theta_true']:.3f}"
    t_m = (
        f"{r['student_t']['lower_miss_rate']:.3f} /"
        f" {r['student_t']['upper_miss_rate']:.3f}"
    )
    t_w = f"{r['student_t']['med_rel_width']:.3f}"
    b_m = (
        f"{r['bootstrap']['lower_miss_rate']:.3f} /"
        f" {r['bootstrap']['upper_miss_rate']:.3f}"
    )
    b_w = f"{r['bootstrap']['med_rel_width']:.3f}"
    # All miss rates use all repetitions as the denominator.
    l_m = (
        f"{r['library']['lower_miss_rate']:.3f} /"
        f" {r['library']['upper_miss_rate']:.3f}"
    )
    l_u = f"{r['library']['unrefuted_miss_rate']:.3f}"
    l_ref = f"{r['library']['refusal_rate']:.3f}"
    l_w = f"{r['library']['med_rel_width']:.3f}"
    print(
        f"{c_name:<30} {block:<6} {metric:<8} {th:<7} {t_m:<16} {t_w:<7} {b_m:<16}"
        f" {b_w:<7} {l_m:<16} {l_u:<10} {l_ref:<10} {l_w:<7}",
        flush=True,
    )
  print(sep + "\n", flush=True)
  elapsed = time.time() - start_time
  print(f"Total elapsed time: {elapsed:.2f} s", flush=True)


if __name__ == "__main__":
  app.run(main)
