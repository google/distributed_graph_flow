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

"""Cluster-granularity sweep for the cluster-totals (D1) bound.

Measures the miss rate (over all replications) and width of the
cluster-totals bound across clustering granularities and dependence
structures.

The experiment holds the true dependence structure fixed (equicorrelated
blocks of size `block_size` with error correlation `rho`) and varies the clustering
granularity `cluster_size` that the estimator uses, while asserting
`kappa_cluster = 1.0` (uncorrelated clusters). That assertion is:

  - true when clusters contain whole blocks and are aligned to them,
  - false when `cluster_size < block_size`, because each block is split across
    `block_size / cluster_size` clusters that are mutually correlated,
  - false when cluster boundaries are offset from block boundaries.

Reference row: `cluster_size = 1` is route D0 (the item-level bound), run with
the analytically exact `effective_n` as a benchmark.
"""

from collections.abc import Sequence
from concurrent import futures
import dataclasses
import json
import os
from typing import Any

from absl import app
from absl import flags
import numpy as np
import scipy.stats

from dgf.src.stats import cluster_bound
from dgf.src.stats.benchmark import baselines
from dgf.src.stats.benchmark import generators
from dgf.src.stats.independent import interval as ind_interval

DEFAULT_SEED = 20260923

_SUITE = flags.DEFINE_enum(
    "suite",
    "granularity",
    ["granularity", "calibration", "refutation"],
    "Benchmark suite to execute.",
)
_MODE = flags.DEFINE_enum(
    "mode",
    "quick",
    ["quick", "full"],
    "Execution mode: 'quick' or 'full'.",
)
_N = flags.DEFINE_integer("n", 100000, "Items per replication.")
_REPS = flags.DEFINE_integer(
    "reps", 0, "Replications per cell (0 to use mode/suite default)."
)
_SEED = flags.DEFINE_integer("seed", DEFAULT_SEED, "Base random seed.")
_M = flags.DEFINE_float("m", 4.0, "Declared relative variance bound.")
_OUTPUT = flags.DEFINE_string(
    "output", None, "Optional output path for newline-delimited JSON results."
)
_WORKERS = flags.DEFINE_integer("workers", 8, "Worker process count.")

# Granularities to sweep. 1 is item-level (D0).
_CLUSTER_SIZES = (1, 5, 25, 250, 500)
_RHO = (0.3, 0.8)
# 250 is what gives kappa_true = 45 at cluster_size=5, comfortably past the alpha=0.05
# threshold of 21.4. n=100000 keeps 400 independent blocks even at block_size=250, so
# a coverage failure cannot be blamed on too few independent units.
_BLOCK_SIZES = (10, 50, 250)
_ALPHAS = (0.1, 0.05, 0.01)
_ALIGNMENT = ("aligned", "offset")

# The estimand: these errors are standardized, so E[e^2] = 1 exactly.
_THETA_TRUE = 1.0
# True relative second moment of the summand s = e^2 for standard normal e.
_M_TRUE_SUMMAND = 3.0


@dataclasses.dataclass(frozen=True)
class SweepCell:
  block_size: int
  rho: float
  cluster_size: int
  alignment: str
  alpha: float
  m_declared: float = 4.0


def _cluster_ids(
    n: int, cluster_size: int, block_size: int, alignment: str
) -> np.ndarray:
  """Assigns items to clusters of size cluster_size, optionally offset from blocks.

  Args:
    n: Item count.
    cluster_size: Nominal cluster size.
    block_size: True dependence-block size, used only to size the offset.
    alignment: 'aligned' puts cluster boundaries on block boundaries when
      cluster_size is a multiple of block_size; 'offset' shifts them by half a
      block so that every cluster straddles two blocks.

  Returns:
    Integer cluster label per item.
  """
  idx = np.arange(n)
  if alignment == "offset":
    idx = idx + (block_size // 2)
  return idx // cluster_size


def _m_c_minus_1(cluster_size: int, block_size: int, rho: float) -> float:
  """Exact `M_c - 1` for a cluster total, aligned equicorrelated blocks.

  For jointly Gaussian errors with correlation `rho`, the *summands* `s = e^2`
  have `Var(s) = 2`, `E[s] = 1`, and `Corr(s_i, s_j) = rho^2` -- the square is
  easy to forget and halves the apparent dependence at rho=0.8.

  A cluster of size `cluster_size` overlaps `min(cluster_size, block_size)`
  correlated neighbours, whether it sits inside one block or spans several
  whole ones, so a single formula covers both sides.

  Args:
    cluster_size: Cluster size.
    block_size: True block size.
    rho: Error correlation within a block.

  Returns:
    `Var(S_c) / (E S_c)^2`.
  """
  overlap = min(cluster_size, block_size)
  return 2.0 * (1.0 + (overlap - 1) * rho**2) / cluster_size


def _kappa_true(cell: SweepCell) -> tuple[float, bool]:
  """True max absolute row sum of the cluster-total correlation matrix.

  Args:
    cell: The sweep configuration.

  Returns:
    `(kappa, is_exact)`. `is_exact` is False where the grid does not divide
    evenly or the offset makes the geometry ragged, in which case the value is
    an order-of-magnitude heuristic and must be read as such.
  """
  rho2 = cell.rho**2
  if cell.cluster_size == 1:
    return 1.0 + (cell.block_size - 1) * rho2, True

  if cell.cluster_size < cell.block_size:
    # Each block splits into k mutually correlated clusters. Exact only when the
    # split is even; the offset just relabels which items go where, so it does
    # not change the count.
    k = cell.block_size / cell.cluster_size
    corr = cell.cluster_size * rho2 / (1.0 + (cell.cluster_size - 1) * rho2)
    exact = cell.block_size % cell.cluster_size == 0
    return 1.0 + (k - 1.0) * corr, exact

  if cell.alignment == "aligned" and cell.cluster_size % cell.block_size == 0:
    # Whole blocks, aligned: the clusters really are independent.
    return 1.0, True

  # Straddling: each cluster shares part of a block with each neighbour.
  # Heuristic only.
  return 1.0 + cell.block_size / (2.0 * cell.cluster_size), False


def _kappa_star(cell: SweepCell, m_declared: float | None = None) -> float:
  """Multiple by which the asserted kappa may be wrong before under-covering.

  Cantelli gives a half-width `c = sqrt((M_d - 1)(1-a')/(G a'))` on the relative
  scale, while the sampling standard deviation of the cluster mean is
  `sqrt((M_c - 1) kappa / G)`. Coverage is retained while the former exceeds
  `z_{1-alpha/2}` of the latter, i.e. while

      kappa  <  (M_d - 1)(1-a')/a' / ((M_c - 1) z^2)  =:  kappa_star.

  Args:
    cell: The sweep configuration.
    m_declared: The declared relative variance bound (defaults to cell.m_declared).

  Returns:
    The robustness margin.

  Note:
    This is a **normal approximation**, not a finite-sample claim. It holds when the cluster
    totals are close to Gaussian, which needs a large G and light-tailed
    summands. It must not be used to justify weakening any certificate branch;
    it is a diagnostic for reading this table.
  """
  if m_declared is None:
    m_declared = cell.m_declared
  a_p = cell.alpha / 2.0
  z = float(scipy.stats.norm.ppf(1.0 - a_p))
  return (
      (m_declared - 1.0)
      * (1.0 - a_p)
      / a_p
      / (_m_c_minus_1(cell.cluster_size, cell.block_size, cell.rho) * z * z)
  )


def _cell_seed(cell: SweepCell, base_seed: int) -> int:
  """Deterministic per-cell seed.

  Deliberately not `hash()`: Python salts `hash` per process, and this runs
  under `ProcessPoolExecutor`, so a `hash`-derived seed would make the sweep
  irreproducible and would silently change between runs.

  Args:
    cell: The sweep configuration.
    base_seed: The run-level seed.

  Returns:
    A seed in [base_seed, base_seed + 2**32).
  """
  align = 1 if cell.alignment == "offset" else 0
  m_decl_term = int(round((cell.m_declared - 4.0) * 1000)) * 13
  mixed = (
      cell.block_size * 1_000_003
      + int(round(cell.rho * 1000)) * 10_007
      + cell.cluster_size * 101
      + int(round(cell.alpha * 10000)) * 7
      + align
      + m_decl_term
  )
  return base_seed + (mixed % (2**32))


def evaluate_cell(
    cell: SweepCell,
    n: int,
    reps: int,
    base_seed: int,
    include_m_observed: bool = False,
) -> dict[str, Any]:
  """Runs one configuration."""
  rng = np.random.default_rng(_cell_seed(cell, base_seed))
  level = 1.0 - cell.alpha
  ids = _cluster_ids(n, cell.cluster_size, cell.block_size, cell.alignment)
  num_clusters = int(np.unique(ids).size)
  m_declared = cell.m_declared

  # D0 reference at cluster_size == 1: supply the analytically exact effective_n
  # rather than asserting independence, so the item-level row is correct by
  # construction and the cluster rows are compared against something sound.
  analytic_neff = generators.analytic_neff_equicorr(
      n, cell.rho, cell.block_size, "mse"
  )

  certified = 0  # status UNREFUTED
  covered_certified = 0
  num_missed = 0  # any issued interval (UNREFUTED or TAIL_UNRESOLVED) missing
  unrefuted_miss = 0
  num_refuted = 0  # TAIL_UNRESOLVED
  num_assumption_required = 0
  widths: list[float] = []
  m_observed_values: list[float] = []
  for _ in range(reps):
    errors = generators.sample_equicorr(n, cell.rho, cell.block_size, rng)
    # Both entry points take *residuals* and derive the summand themselves. Do
    # not pre-square here: `metric="mse"` would then square a second time.
    if cell.cluster_size == 1:
      res = ind_interval.confidence_interval(
          errors, metric="mse", level=level, m=m_declared,
          effective_n=analytic_neff,
      )
      low, high = res.low, res.high
      status = res.status
      m_obs = res.m_observed
    else:
      cres = cluster_bound.cluster_interval(
          errors, ids, metric="mse", level=level, m=m_declared,
          kappa_cluster=1.0,
      )
      low, high = cres.low, cres.high
      status = cres.status
      m_obs = cres.m_observed

    m_observed_values.append(m_obs)

    if status is ind_interval.Status.ASSUMPTION_REQUIRED:
      num_assumption_required += 1
      continue
    if status is ind_interval.Status.TAIL_UNRESOLVED:
      num_refuted += 1
    is_miss = not low <= _THETA_TRUE <= high
    if is_miss:
      num_missed += 1
    if status is ind_interval.Status.UNREFUTED:
      certified += 1
      if np.isfinite(low) and np.isfinite(high):
        widths.append(high - low)
      if is_miss:
        unrefuted_miss += 1
      else:
        covered_certified += 1

  kappa, kappa_exact = _kappa_true(cell)
  miss_lo, miss_hi = baselines.clopper_pearson(num_missed, reps, 0.99)
  out: dict[str, Any] = {
      # m_* is reserved strictly for relative second moments E[s^2]/(E[s])^2.
      # Block and cluster sizes are explicitly named block_size and cluster_size.
      "block_size": cell.block_size,
      "cluster_size": cell.cluster_size,
      "rho": cell.rho,
      "alignment": cell.alignment,
      "alpha": cell.alpha,
      "num_clusters": num_clusters,
      "route": "D0" if cell.cluster_size == 1 else "D1",
      "kappa_true": kappa,
      "kappa_true_exact": kappa_exact,
      "kappa_star": _kappa_star(cell, m_declared),
      "m_c_minus_1": _m_c_minus_1(cell.cluster_size, cell.block_size, cell.rho),
      # Rates below use all `reps` replications as the denominator.
      "miss_rate": num_missed / reps,
      "miss_lo": miss_lo,  # 99% Clopper-Pearson interval for miss_rate.
      "miss_hi": miss_hi,
      "unrefuted_miss_rate": unrefuted_miss / reps,
      "certified_rate": certified / reps,
      "num_missed": num_missed,
      "num_unrefuted_miss": unrefuted_miss,
      "num_refuted": num_refuted,
      "num_assumption_required": num_assumption_required,
      # Coverage among UNREFUTED results only (conditional; not guaranteed).
      "conditional_coverage_rate": (
          (covered_certified / certified) if certified else float("nan")
      ),
      "median_width": float(np.median(widths)) if widths else float("nan"),
      "reps": reps,
      "n": n,
      "m_declared": m_declared,
      "m_true_summand": _M_TRUE_SUMMAND,
  }
  if include_m_observed:
    m_obs_arr = np.array(m_observed_values, dtype=np.float64)
    m_obs_arr = m_obs_arr[np.isfinite(m_obs_arr)]
    out["m_c"] = 1.0 + _m_c_minus_1(
        cell.cluster_size, cell.block_size, cell.rho
    )
    out["m_observed_mean"] = (
        float(np.mean(m_obs_arr)) if m_obs_arr.size else float("nan")
    )
    out["m_observed_std"] = (
        float(np.std(m_obs_arr, ddof=1)) if m_obs_arr.size > 1 else float("nan")
    )
  return out


def _build_granularity_cells() -> list[SweepCell]:
  """Enumerates the granularity grid."""
  cells = []
  for bs in _BLOCK_SIZES:
    for r in _RHO:
      for cs in _CLUSTER_SIZES:
        for a in _ALIGNMENT:
          if a == "offset" and cs < bs:
            continue
          if a == "offset" and cs == 1:
            continue
          for alpha in _ALPHAS:
            cells.append(
                SweepCell(
                    block_size=bs,
                    rho=r,
                    cluster_size=cs,
                    alignment=a,
                    alpha=alpha,
                    m_declared=4.0,
                )
            )
  return cells


def _build_calibration_cells() -> list[SweepCell]:
  """Calibration grid: straddles the kappa_true / kappa_star crossing."""
  block_size_vals = (80, 100, 125, 160, 200, 250)
  alphas = (0.1, 0.05, 0.01)
  cells = []
  for bs in block_size_vals:
    for alpha in alphas:
      cells.append(
          SweepCell(
              block_size=bs,
              rho=0.8,
              cluster_size=5,
              alignment="aligned",
              alpha=alpha,
              m_declared=4.0,
          )
      )
  return cells


def _build_refutation_cells() -> list[SweepCell]:
  """Refutation grid: straddles M_c at granularities 1, 5, 25 across two rhos."""
  cluster_size_vals = (1, 5, 25)
  m_declared_vals = (1.2, 1.5, 2.0, 2.3, 2.4, 2.5, 3.0, 4.0)
  rhos = (0.3, 0.8)
  cells = []
  for cs in cluster_size_vals:
    for r in rhos:
      for md in m_declared_vals:
        cells.append(
            SweepCell(
                block_size=50,
                rho=r,
                cluster_size=cs,
                alignment="aligned",
                alpha=0.05,
                m_declared=md,
            )
        )
  return cells


SUITES: dict[str, list[SweepCell]] = {
    "granularity": _build_granularity_cells(),
    "calibration": _build_calibration_cells(),
    "refutation": _build_refutation_cells(),
}

SUITE_DEFAULT_REPS: dict[str, int] = {
    "granularity": 2000,
    "calibration": 5000,
    "refutation": 2000,
}


def main(argv: Sequence[str]) -> None:
  del argv
  suite_name = _SUITE.value
  if suite_name not in SUITES:
    raise ValueError(f"Unknown suite: {suite_name!r}")
  full_cells = SUITES[suite_name]

  is_quick = _MODE.value == "quick"
  if is_quick:
    # Quick mode: 2 representative cells, smaller n, 20 reps
    cells = full_cells[:2]
    n_items = 10000
    reps = _REPS.value if _REPS.value > 0 else 20
  else:
    cells = full_cells
    n_items = _N.value
    reps = _REPS.value if _REPS.value > 0 else SUITE_DEFAULT_REPS[suite_name]

  include_m_observed = suite_name in ("refutation", "calibration")

  print(
      f"Cluster sweep [{suite_name}] mode={_MODE.value}: {len(cells)} cells, reps={reps},"
      f" n={n_items}"
  )
  print(
      "kappa_star is the multiple by which the asserted kappa_cluster=1 may be "
      "wrong before coverage breaks (normal approximation, diagnostic only)."
  )

  out_path = _OUTPUT.value
  if out_path:
    parent = os.path.dirname(out_path)
    if parent:
      os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as fh:
      fh.truncate(0)

  with futures.ProcessPoolExecutor(max_workers=_WORKERS.value) as pool:
    tasks = [
        pool.submit(
            evaluate_cell,
            c,
            n_items,
            reps,
            _SEED.value,
            include_m_observed,
        )
        for c in cells
    ]
    for fut in futures.as_completed(tasks):
      res = fut.result()
      # `kappa_star` compares against an *asserted* kappa_cluster = 1. The D0
      # rows never make that assertion -- they are handed the exact
      # `analytic_neff_equicorr`, which is precisely `n / kappa_true` -- so the
      # comparison is meaningless for them and flagging them would brand the
      # correct-by-construction reference route as the most fragile in the table.
      marks: list[str] = []
      if res["route"] != "D0" and res["kappa_true"] > res["kappa_star"]:
        marks.append("EXPECT BREAK")
      # Covering because the interval is wider than the estimand is not a win,
      # and coverage alone cannot distinguish it from a real one.
      if res["median_width"] > _THETA_TRUE:
        marks.append("VACUOUS")

      m_c = 1.0 + res["m_c_minus_1"]
      m_obs_str = ""
      if "m_observed_mean" in res:
        m_obs_str = (
            f" m_obs={res['m_observed_mean']:.3f}±{res['m_observed_std']:.3f}"
        )
      m_decl_str = ""
      if suite_name == "refutation":
        m_decl_str = f" Md={res['m_declared']:.2f} Mc={m_c:.3f}"
        if res["m_declared"] < m_c * 0.9 and res["certified_rate"] >= 0.99999:
          marks.append("FAIL_REFUTATION_CHANNEL")

      flag = ("  <-- " + ", ".join(marks)) if marks else ""

      print(
          f"bs={res['block_size']:4d} rho={res['rho']:.1f}"
          f" cs={res['cluster_size']:4d} {res['alignment']:<8}"
          f" a={res['alpha']:.3f} G={res['num_clusters']:7d}"
          f" {m_decl_str}k_true={res['kappa_true']:7.2f}{'' if res['kappa_true_exact'] else '~'}"
          f" k_star={res['kappa_star']:7.1f}"
          f" cert={res['certified_rate']*100:5.1f}%"
          f" miss={res['miss_rate']*100:6.2f}%"
          f" unref_miss={res['unrefuted_miss_rate']*100:6.2f}%"
          f" med_w={res['median_width']:.4f}{m_obs_str}{flag}"
      )
      if out_path:
        with open(out_path, "a") as fh:
          fh.write(json.dumps(res) + "\n")

  print("Cluster sweep completed.")


if __name__ == "__main__":
  app.run(main)
