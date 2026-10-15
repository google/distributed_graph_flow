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

"""Partitioner cluster-count robustness sweep across seeds and topologies.

Measures the robustness of graph partitioner cluster-count control across:
- Topologies: 100x200 torus, Watts-Strogatz, and SBM.
- Cluster sizes cs in {10, 25, 50}.
- Seeds 0..19 plus 20260923.
Records G, target_g, rel_dev, max_cluster_size, cap, rounds, stop_reason,
num_filled, num_units, num_bins, median_cluster_size, and per-pass merge counts.
"""

from collections.abc import Sequence
import json
import math
import os
from typing import Any

from absl import app
from absl import flags
import numpy as np

from dgf.src.stats import partitioners

_MODE = flags.DEFINE_enum(
    "mode",
    "quick",
    ["quick", "full"],
    "Execution mode: 'quick' or 'full'.",
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Path for output JSONL file (optional).",
)
_CS_LIST = flags.DEFINE_string(
    "cs_list",
    None,
    "Comma-separated list of cluster sizes. If None, uses (10, 25, 50).",
)
_SEEDS_FLAG = flags.DEFINE_string(
    "seeds",
    None,
    "Comma-separated list of seeds (e.g. '0,1,2,3,4'). If None, uses 0..19 +"
    " [20260923].",
)

_N = 20000
DEFAULT_SEED = 0
_DEFAULT_CLUSTER_SIZES = (10, 25, 50)
_DEFAULT_SEEDS = tuple(list(range(20)) + [20260923])


def _build_grid_2d(rows: int, cols: int) -> np.ndarray:
  """Builds rows x cols 2D torus grid."""
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


def _build_sbm_graph(n: int, seed: int) -> np.ndarray:
  """Builds SBM graph: 400 blocks of 50 nodes (p_in=0.3, p_out=0.0005)."""
  rng = np.random.default_rng(seed)
  num_blocks = n // 50
  edges_list: list[np.ndarray] = []

  # Internal edges
  for b in range(num_blocks):
    nodes = np.arange(b * 50, (b + 1) * 50)
    pairs_idx = np.triu_indices(50, k=1)
    mask = rng.random(len(pairs_idx[0])) < 0.3
    u = nodes[pairs_idx[0][mask]]
    v = nodes[pairs_idx[1][mask]]
    if len(u) > 0:
      edges_list.append(np.column_stack([u, v]))

  # Cross-block edges
  num_possible_cross = (num_blocks * (num_blocks - 1) // 2) * 2500
  k_cross = int(rng.poisson(num_possible_cross * 0.0005))
  u_rand = rng.integers(0, n, size=k_cross * 2)
  v_rand = rng.integers(0, n, size=k_cross * 2)
  valid = (u_rand // 50 != v_rand // 50) & (u_rand < v_rand)
  u_sel = u_rand[valid][:k_cross]
  v_sel = v_rand[valid][:k_cross]
  if len(u_sel) > 0:
    edges_list.append(np.column_stack([u_sel, v_sel]))

  all_edges = np.vstack(edges_list)
  pairs = np.sort(all_edges, axis=1)
  return np.unique(pairs, axis=0)


def main(argv: Sequence[str]) -> None:
  del argv
  out_path = _OUTPUT.value

  if _CS_LIST.value:
    cs_list = tuple(
        int(x.strip()) for x in _CS_LIST.value.split(",") if x.strip()
    )
  else:
    cs_list = _DEFAULT_CLUSTER_SIZES

  if _SEEDS_FLAG.value:
    seeds = []
    for item in _SEEDS_FLAG.value.split(","):
      item = item.strip()
      if "-" in item:
        lo, hi = item.split("-")
        seeds.extend(range(int(lo), int(hi) + 1))
      elif item:
        seeds.append(int(item))
    seeds_list = tuple(seeds)
  else:
    seeds_list = _DEFAULT_SEEDS

def run_partitioner_sweep(
    n: int = _N,
    topologies: Sequence[str] = ("torus",),
    cs_list: Sequence[int] = (50,),
    seeds_list: Sequence[int] = (0,),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Runs partitioner robustness sweep over the given topologies, cluster sizes, and seeds."""
  torus_edges = _build_grid_2d(100, 200)
  results: list[dict[str, Any]] = []
  out_of_band: list[dict[str, Any]] = []

  for topo in topologies:
    print(f"\n--- Topology: {topo} ---")
    for cs in cs_list:
      target_g = n // cs
      cap = int(math.ceil(1.5 * float(cs)))

      for s in seeds_list:
        if topo == "torus":
          edges = torus_edges
        elif topo == "watts_strogatz":
          edges = _build_watts_strogatz(n, k=4, p_rewire=0.02, seed=s)
        else:
          edges = _build_sbm_graph(n, seed=s)

        labels, trace = partitioners._partition_traced(
            n, edges, target_cluster_size=cs, max_size=cap, seed=s
        )
        p_m = partitioners.partition_metrics(n, edges, labels)
        g = int(p_m["num_clusters"])
        max_size = int(p_m["max_cluster_size"])
        median_size = float(p_m["median_cluster_size"])
        ratio = float(g) / float(target_g)
        rel_dev = float(g - target_g) / float(target_g)
        rounds = trace["rounds"]
        stop_reason = trace["stop_reason"]

        row = {
            "topology": topo,
            "cs": cs,
            "seed": s,
            "G": g,
            "target_g": target_g,
            "ratio": ratio,
            "rel_dev": rel_dev,
            "max_cluster_size": max_size,
            "median_cluster_size": median_size,
            "cap": cap,
            "rounds": rounds,
            "stop_reason": stop_reason,
            "num_filled": trace["num_filled"],
            "num_units": trace["num_units"],
            "num_bins": trace["num_bins"],
            "merges_per_round": trace["merges_per_round"],
            "g_per_round": trace["g_per_round"],
        }
        results.append(row)

        is_oob = (
            (ratio < 0.80)
            or (ratio > (4.0 / 3.0))
            or (max_size > cap)
            or (g < math.ceil(n / float(cap)))
        )
        flag = " *** OUT-OF-BAND ***" if is_oob else ""
        if is_oob:
          out_of_band.append(row)

        print(
            f"[{topo:<14}] cs={cs:2d} seed={s:<8} G={g:4d}"
            f" G/target={ratio:.4f} max_sz={max_size:2d} med={median_size:4.1f} (cap={cap})"
            f" rounds={rounds:2d} stop={stop_reason}{flag}"
        )

  return results, out_of_band


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  out_path = _OUTPUT.value

  if _CS_LIST.value:
    cs_list = tuple(
        int(x.strip()) for x in _CS_LIST.value.split(",") if x.strip()
    )
  elif _MODE.value == "quick":
    cs_list = (50,)
  else:
    cs_list = _DEFAULT_CLUSTER_SIZES

  if _SEEDS_FLAG.value:
    seeds = []
    for item in _SEEDS_FLAG.value.split(","):
      item = item.strip()
      if "-" in item:
        lo, hi = item.split("-")
        seeds.extend(range(int(lo), int(hi) + 1))
      elif item:
        seeds.append(int(item))
    seeds_list = tuple(seeds)
  elif _MODE.value == "quick":
    seeds_list = (0,)
  else:
    seeds_list = _DEFAULT_SEEDS

  topologies = ("torus",) if _MODE.value == "quick" else ("torus", "watts_strogatz", "sbm")

  print(
      f"=== Partitioner Robustness Sweep (n={_N}, cs_list={cs_list},"
      f" seeds={len(seeds_list)}, mode={_MODE.value}) ==="
  )

  results, out_of_band = run_partitioner_sweep(_N, topologies, cs_list, seeds_list)

  if out_path:
    # If run under blaze run, resolve relative paths against workspace directory
    ws_dir = os.environ.get("BUILD_WORKING_DIRECTORY")
    target_path = (
        os.path.join(ws_dir, out_path)
        if (ws_dir and not os.path.isabs(out_path))
        else out_path
    )
    out_dir = os.path.dirname(target_path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(target_path, "w") as fh:
      for r in results:
        fh.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows to {target_path}.")

  if out_of_band:
    print(
        f"\nWARNING: {len(out_of_band)} rows violated contract band [0.80, 4/3]"
        " or cap or G >= ceil(n/cap)!"
    )


if __name__ == "__main__":
  app.run(main)
