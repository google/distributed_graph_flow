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

"""Synthetic measurement sweep for the edge-test dominance guard.

Measures size, power, and guard activation rates across 29 synthetic cells
covering heteroscedastic Gaussian, sparse binary, anchored skew counterexamples,
near-the-floor edge counts, and correlated power alternatives.
"""

from collections.abc import Sequence
from concurrent import futures
import dataclasses
import json
import math
import os
from typing import Any

from absl import app
from absl import flags
import numpy as np

from dgf.src.stats import graph
from dgf.src.stats import partitioners
from dgf.src.stats import refutation

DEFAULT_SEED = 20260926

_MODE = flags.DEFINE_enum(
    "mode",
    "quick",
    ["quick", "full"],
    "Execution mode: 'quick' or 'full'.",
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Path to write results JSONL (optional).",
)
_REPS = flags.DEFINE_integer(
    "reps",
    2000,
    "Number of replications per cell (in full mode).",
)
_WORKERS = flags.DEFINE_integer(
    "workers",
    8,
    "Number of worker processes.",
)
_SEED = flags.DEFINE_integer(
    "seed",
    DEFAULT_SEED,
    "Base random seed.",
)


def _build_geometric_edges(g: int, mean_deg: float, seed: int = 42) -> np.ndarray:
  """Builds 2-D random geometric graph edges on unit square."""
  rng = np.random.default_rng(seed)
  pos = rng.uniform(0.0, 1.0, size=(g, 2))
  r = math.sqrt(mean_deg / (math.pi * float(g)))
  diff = pos[:, None, :] - pos[None, :, :]
  dists = np.sqrt(np.sum(diff**2, axis=-1))
  u, v = np.where((dists < r) & (np.arange(g)[:, None] < np.arange(g)[None, :]))
  return np.column_stack([u, v]).astype(np.int64)


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


def _build_hub_edges(g: int, base_edges: np.ndarray, seed: int = 301) -> np.ndarray:
  """Builds hub graph: base geometric edges plus 3 hubs adjacent to 90% of nodes."""
  hub_edges_list = [base_edges]
  rng_hubs = np.random.default_rng(seed)
  for hub in (0, 1, 2):
    others = np.delete(np.arange(g), hub)
    chosen = rng_hubs.choice(others, size=int(0.9 * (g - 1)), replace=False)
    h_min = np.minimum(hub, chosen)
    h_max = np.maximum(hub, chosen)
    hub_edges_list.append(np.column_stack([h_min, h_max]))
  all_hub_edges = np.vstack(hub_edges_list)
  pair_keys = np.unique(all_hub_edges[:, 0] * g + all_hub_edges[:, 1])
  return np.column_stack([pair_keys // g, pair_keys % g]).astype(np.int64)


def _build_matching_edges(g: int) -> np.ndarray:
  """Builds perfect matching graph on G vertices."""
  return np.column_stack([np.arange(0, g, 2), np.arange(1, g, 2)]).astype(np.int64)


@dataclasses.dataclass(frozen=True)
class CellConfig:
  family: str
  cell_name: str
  g: int
  counts: np.ndarray
  edges: np.ndarray
  generator_type: str  # 'gaussian', 'binary', 'anchored_skew', 'power'
  extra_params: dict[str, Any]


def generate_cells() -> list[CellConfig]:
  """Generates all 29 cells across families S-a through S-e."""
  cells: list[CellConfig] = []

  # --- Family S-a: Gaussian heteroscedastic ---
  # Geometric G = 500
  g_a1 = 500
  edges_a1 = _build_geometric_edges(g_a1, mean_deg=6.3, seed=100)
  counts_a1 = np.random.default_rng(101).integers(25, 75, size=g_a1)
  cells.append(
      CellConfig(
          family="S-a",
          cell_name="S-a: Gaussian geometric G=500",
          g=g_a1,
          counts=counts_a1,
          edges=edges_a1,
          generator_type="gaussian",
          extra_params={},
      )
  )

  # BA partition n = 20000, t = 50
  ba_edges_raw = _build_ba_edges(20000, 5, seed=200)
  labels_ba = partitioners.partition_graph(
      20000, ba_edges_raw, target_cluster_size=50, seed=0
  )
  counts_ba = np.bincount(labels_ba).astype(np.int64)
  g_ba = len(counts_ba)
  edges_ba = graph.cluster_adjacency(20000, ba_edges_raw, labels_ba)
  cells.append(
      CellConfig(
          family="S-a",
          cell_name=f"S-a: Gaussian BA partition G={g_ba}",
          g=g_ba,
          counts=counts_ba,
          edges=edges_ba,
          generator_type="gaussian",
          extra_params={},
      )
  )

  # Hub graph G = 500
  base_edges_c = _build_geometric_edges(500, mean_deg=6.0, seed=300)
  edges_hub = _build_hub_edges(500, base_edges_c, seed=301)
  counts_hub = np.random.default_rng(301).integers(25, 75, size=500)
  cells.append(
      CellConfig(
          family="S-a",
          cell_name="S-a: Gaussian hub graph G=500",
          g=500,
          counts=counts_hub,
          edges=edges_hub,
          generator_type="gaussian",
          extra_params={},
      )
  )

  # --- Family S-b: Sparse binary ---
  # theta in {0.99, 0.999} x n_c in {1, 10, 50} x graph in {geometric, BA, matching}
  graphs_sb = [
      ("geometric G=500", 500, edges_a1),
      (f"BA partition G={g_ba}", g_ba, edges_ba),
      ("perfect matching G=1000", 1000, _build_matching_edges(1000)),
  ]
  for theta in (0.99, 0.999):
    for nc in (1, 10, 50):
      for g_label, g_nodes, g_edges in graphs_sb:
        c_arr = np.full(g_nodes, nc, dtype=np.int64)
        cells.append(
            CellConfig(
                family="S-b",
                cell_name=f"S-b: theta={theta} nc={nc} {g_label}",
                g=g_nodes,
                counts=c_arr,
                edges=g_edges,
                generator_type="binary",
                extra_params={"theta": theta, "nc": nc},
            )
        )

  # --- Family S-c: Anchored skew ---
  # perfect matching G = 1000, n_c = 1. k active edges in {3, 5, 10, 50}
  matching_1000 = _build_matching_edges(1000)
  counts_1000_1 = np.ones(1000, dtype=np.int64)
  for k in (3, 5, 10, 50):
    cells.append(
        CellConfig(
            family="S-c",
            cell_name=f"S-c: anchored skew k={k} matching G=1000",
            g=1000,
            counts=counts_1000_1,
            edges=matching_1000,
            generator_type="anchored_skew",
            extra_params={"k": k, "theta": 0.999},
        )
    )

  # --- Family S-d: Near the floor ---
  # geometric with G in {10, 20, 40} and mean degree 6
  for g_d in (10, 20, 40):
    edges_d = _build_geometric_edges(g_d, mean_deg=6.0, seed=400 + g_d)
    counts_d = np.random.default_rng(500 + g_d).integers(25, 75, size=g_d)
    cells.append(
        CellConfig(
            family="S-d",
            cell_name=f"S-d: near floor G={g_d}",
            g=g_d,
            counts=counts_d,
            edges=edges_d,
            generator_type="gaussian",
            extra_params={},
        )
    )

  # --- Family S-e: Power ---
  # geometric G = 500, counts 25-75, eps_c = sqrt(n_c) (u_c + beta sum_{d~c} u_d), beta = 0.1
  beta = 0.1
  u_idx = edges_a1[:, 0]
  v_idx = edges_a1[:, 1]
  deg = np.bincount(u_idx, minlength=g_a1) + np.bincount(v_idx, minlength=g_a1)
  # Exact VIF calculation:
  # Var(sum eps) = sum_c ( sqrt(n_c) + beta sum_{d~c} sqrt(n_d) )^2
  sqrt_n = np.sqrt(counts_a1.astype(np.float64))
  sum_neighbor_sqrt_n = np.bincount(
      u_idx, weights=sqrt_n[v_idx], minlength=g_a1
  ) + np.bincount(v_idx, weights=sqrt_n[u_idx], minlength=g_a1)
  total_var = float(np.sum((sqrt_n + beta * sum_neighbor_sqrt_n) ** 2))
  sum_var = float(
      np.sum(counts_a1.astype(np.float64) * (1.0 + (beta**2) * deg.astype(np.float64)))
  )
  exact_vif = total_var / sum_var

  cells.append(
      CellConfig(
          family="S-e",
          cell_name="S-e: power geometric G=500 beta=0.1",
          g=g_a1,
          counts=counts_a1,
          edges=edges_a1,
          generator_type="power",
          extra_params={"beta": beta, "exact_vif": exact_vif},
      )
  )

  return cells


def run_cell_simulation(
    cell: CellConfig,
    reps: int,
    base_seed: int,
) -> dict[str, Any]:
  """Runs simulation for a single cell across reps."""
  g = cell.g
  counts = cell.counts
  edges = cell.edges
  gen_type = cell.generator_type
  params = cell.extra_params
  n_total = float(np.sum(counts))
  sqrt_counts = np.sqrt(counts.astype(np.float64))

  u_idx = edges[:, 0] if len(edges) > 0 else np.zeros(0, dtype=np.int64)
  v_idx = edges[:, 1] if len(edges) > 0 else np.zeros(0, dtype=np.int64)
  beta = float(params.get("beta", 0.0))

  rej_guarded = 0
  untest_guard = 0
  untest_other = 0
  rej_unguarded = 0
  edge_n_effs: list[float] = []

  rng = np.random.default_rng(base_seed)

  for _ in range(reps):
    if gen_type == "gaussian":
      s_c = rng.lognormal(0.0, 0.5, size=g)
      eps = rng.normal(0.0, sqrt_counts * s_c)
      theta_hat = float(np.sum(eps)) / n_total
      e = eps - counts.astype(np.float64) * theta_hat
    elif gen_type == "binary":
      theta = params["theta"]
      s_c = rng.binomial(counts, theta).astype(np.float64)
      theta_hat = float(np.sum(s_c)) / n_total
      e = s_c - counts.astype(np.float64) * theta_hat
    elif gen_type == "anchored_skew":
      k = params["k"]
      theta = params["theta"]
      s_c = np.full(g, theta, dtype=np.float64)
      s_c[: 2 * k] = rng.binomial(1, theta, size=2 * k).astype(np.float64)
      theta_hat = float(np.sum(s_c)) / n_total
      e = s_c - counts.astype(np.float64) * theta_hat
    elif gen_type == "power":
      u = rng.standard_normal(g)
      sum_neigh_u = np.bincount(
          u_idx, weights=u[v_idx], minlength=g
      ) + np.bincount(v_idx, weights=u[u_idx], minlength=g)
      eps = sqrt_counts * (u + beta * sum_neigh_u)
      theta_hat = float(np.sum(eps)) / n_total
      e = eps - counts.astype(np.float64) * theta_hat
    else:
      raise ValueError(f"Unknown generator type {gen_type}")

    # Run guarded (default min_effective_edges = 10.0)
    res_g = refutation.check_uncorrelated_edges(
        e, counts, edges, level=0.95, min_effective_edges=10.0
    )
    # Run unguarded (min_effective_edges = 0.0)
    res_u = refutation.check_uncorrelated_edges(
        e, counts, edges, level=0.95, min_effective_edges=0.0
    )

    if res_g.outcome is refutation.Refutation.REFUTED:
      rej_guarded += 1
    elif res_g.outcome is refutation.Refutation.UNTESTABLE:
      if res_g.reason is not None and "effective edges" in res_g.reason:
        untest_guard += 1
      else:
        untest_other += 1

    if res_u.outcome is refutation.Refutation.REFUTED:
      rej_unguarded += 1

    if res_g.edge_n_eff is not None:
      edge_n_effs.append(res_g.edge_n_eff)
    elif res_u.edge_n_eff is not None:
      edge_n_effs.append(res_u.edge_n_eff)

  rate_rej_guarded = float(rej_guarded) / float(reps)
  rate_untest_guard = float(untest_guard) / float(reps)
  rate_untest_other = float(untest_other) / float(reps)
  rate_rej_unguarded = float(rej_unguarded) / float(reps)

  testable_reps = reps - (untest_guard + untest_other)
  size_among_testable = (
      float(rej_guarded) / float(testable_reps) if testable_reps > 0 else float("nan")
  )

  if edge_n_effs:
    q01, q10, q50, q90 = np.percentile(edge_n_effs, [1, 10, 50, 90])
  else:
    q01 = q10 = q50 = q90 = float("nan")

  result = {
      "family": cell.family,
      "cell_name": cell.cell_name,
      "g": g,
      "num_edges": len(edges),
      "reps": reps,
      "reject_guarded": rate_rej_guarded,
      "untestable_guard": rate_untest_guard,
      "untestable_other": rate_untest_other,
      "reject_unguarded": rate_rej_unguarded,
      "size_among_testable": size_among_testable,
      "edge_n_eff_q01": float(q01),
      "edge_n_eff_q10": float(q10),
      "edge_n_eff_q50": float(q50),
      "edge_n_eff_q90": float(q90),
  }
  if "exact_vif" in params:
    result["exact_vif"] = float(params["exact_vif"])
  if "theta" in params:
    result["theta"] = params["theta"]
  if "nc" in params:
    result["nc"] = params["nc"]
  if "k" in params:
    result["k"] = params["k"]

  return result


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  base_seed = _SEED.value
  out_path = _OUTPUT.value

  if _MODE.value == "quick":
    reps = 20
    cells = generate_cells()[:1]
  else:
    reps = _REPS.value
    cells = generate_cells()

  print(f"Running edge guard sweep over {len(cells)} cells, {reps} reps each (mode={_MODE.value})...")

  results: list[dict[str, Any]] = []

  # Parallel execution across cells
  workers = min(_WORKERS.value, len(cells))
  with futures.ProcessPoolExecutor(max_workers=workers) as executor:
    future_to_cell = {
        executor.submit(
            run_cell_simulation, cell, reps, base_seed + i * 10007
        ): (i, cell)
        for i, cell in enumerate(cells)
    }
    for fut in futures.as_completed(future_to_cell):
      res = fut.result()
      results.append(res)
      print(
          f"Done cell [{res['family']}] {res['cell_name']}: "
          f"guard_rej={res['reject_guarded']:.4f}, "
          f"untest_guard={res['untestable_guard']:.4f}, "
          f"untest_other={res['untestable_other']:.4f}, "
          f"ung_rej={res['reject_unguarded']:.4f}, "
          f"size_testable={res['size_among_testable']:.4f}, "
          f"n_eff_med={res['edge_n_eff_q50']:.1f}"
      )

  # Sort by family and cell_name
  results.sort(key=lambda r: (r["family"], r["cell_name"]))

  # Write JSONL if requested
  if out_path:
    out_dir = os.path.dirname(out_path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
      for r in results:
        f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows to {out_path}\n")

  # Print Markdown table
  print("| Family | Cell | G | |E| | Reject (Guarded) | Untest (Guard) | Untest (Other) | Reject (Unguarded) | Size (Testable) | n_eff (p01, p10, p50, p90) | Notes |")
  print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
  for r in results:
    q_str = f"({r['edge_n_eff_q01']:.1f}, {r['edge_n_eff_q10']:.1f}, {r['edge_n_eff_q50']:.1f}, {r['edge_n_eff_q90']:.1f})"
    notes = ""
    if "exact_vif" in r:
      notes = f"exact VIF = {r['exact_vif']:.4f}"
    elif "theta" in r and "k" in r:
      notes = f"k={r['k']}, theta={r['theta']}"
    elif "theta" in r:
      notes = f"theta={r['theta']}, nc={r['nc']}"
    print(
        f"| {r['family']} | {r['cell_name']} | {r['g']} | {r['num_edges']} | "
        f"{r['reject_guarded']:.4f} | {r['untestable_guard']:.4f} | {r['untestable_other']:.4f} | "
        f"{r['reject_unguarded']:.4f} | {r['size_among_testable']:.4f} | {q_str} | {notes} |"
    )


if __name__ == "__main__":
  app.run(main)
