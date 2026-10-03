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

"""Textbook intervals for RMSE on dependent and heavy-tailed data.

Evaluates empirical coverage of standard textbook confidence intervals for RMSE
(percentile bootstrap, BCa bootstrap, Bayesian bootstrap, Student-t) alongside
dependence-aware variants (moving-block bootstrap, cluster bootstrap) and the
library's certified bounds, across:
  Part A: Spatio-temporally correlated fields (ring benchmark).
  Part B: Independent, heavy-tailed error distributions.
"""

from collections.abc import Sequence
import dataclasses
import json
import math
import multiprocessing
import os
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

from absl import app
from absl import flags
from absl import logging
import numpy as np

from dgf.src.stats import cluster_bound as _cluster_bound
from dgf.src.stats import cluster_sketch as _cluster_sketch
from dgf.src.stats import range_envelope as _re
from dgf.src.stats import spacetime
from dgf.src.stats.benchmark import baselines
from dgf.src.stats.benchmark import distributions
from dgf.src.stats.benchmark import range_generators
from dgf.src.stats.independent import interval as _ind_interval

# Re-export key range constructs
RangeDeclaration = _re.RangeDeclaration
range_candidates = _re.range_candidates
range_kappa_fn = _re.range_kappa_fn

DEFAULT_SEED_PART_A = 0x820000
DEFAULT_SEED_PART_B = 0x880000


def compute_exact_theta(
    graph: range_generators.GraphStructure,
    k_hops: int,
    l_lookback: int,
    horizon_h: int,
    r_s: int,
    r_t: int,
    variant: int = 1,
    rho_0: float = 0.0,
    c: float = 0.0,
) -> float:
  """Computes exact analytical truth theta(D) = E[s_{u,t}] for the moving-average model."""
  n = graph.num_nodes
  u_target = 0
  max_dist = k_hops + r_s
  active_nodes = [v for v in range(n) if graph.dist_matrix[u_target, v] <= max_dist]
  tau_min = -(l_lookback - 1) - r_t
  tau_max = horizon_h
  tau_steps = list(range(tau_min, tau_max + 1))

  s_mask = (graph.dist_matrix <= r_s).astype(np.float64)
  deg_s = np.sum(s_mask, axis=1)
  norm_s = np.sqrt(deg_s * float(r_t + 1))

  k_mask = (graph.dist_matrix <= k_hops).astype(np.float64)
  deg_k_target = float(np.sum(k_mask[u_target, :]))

  a_sq_sum = 0.0
  for v0 in active_nodes:
    for tau0 in tau_steps:
      if graph.dist_matrix[u_target, v0] <= r_s and (horizon_h - r_t <= tau0 <= horizon_h):
        w_y = 1.0 / norm_s[u_target]
      else:
        w_y = 0.0

      w_pred = 0.0
      for v in range(n):
        if graph.dist_matrix[u_target, v] <= k_hops:
          if graph.dist_matrix[v, v0] <= r_s:
            l_start = max(0, -tau0 - r_t)
            l_end = min(l_lookback - 1, -tau0)
            if l_start <= l_end:
              num_matching_l = l_end - l_start + 1
              w_pred += float(num_matching_l) / norm_s[v]

      w_pred /= deg_k_target * float(l_lookback)
      coeff = w_y - w_pred
      a_sq_sum += coeff**2

  var_eta = a_sq_sum
  var_common = 1.0 + (1.0 / float(l_lookback))

  if variant == 1:
    return float(var_eta)
  elif variant == 2:
    return float((1.0 - rho_0) * var_eta + rho_0 * var_common)
  elif variant == 3:
    alpha_param = c / float(n)
    return float((1.0 - alpha_param) * var_eta + alpha_param * var_common)
  else:
    raise ValueError(f"Unknown variant: {variant}")



# ==============================================================================
# Baseline Interval Estimators for RMSE
# ==============================================================================


def percentile_bootstrap_rmse(
    summands_e2: np.ndarray,
    confidence: float = 0.95,
    num_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
  """Efron nonparametric percentile bootstrap for RMSE.

  Resamples items i.i.d. with replacement. Since RMSE = sqrt(mean(e^2)) is a
  monotone transformation of the sample mean of e^2, the quantiles of RMSE
  equal the square roots of the quantiles of the resample means of e^2.

  Args:
    summands_e2: 1D array of squared errors e^2.
    confidence: Confidence level (e.g. 0.95).
    num_resamples: Bootstrap sample count B (default 1000).
    rng: Optional numpy Generator.

  Returns:
    (low, high) confidence interval for RMSE.
  """
  n = len(summands_e2)
  if n < 2:
    return float("-inf"), float("inf")
  if rng is None:
    rng = np.random.default_rng()

  alpha = 1.0 - confidence
  batch_size = 100
  means = np.empty(num_resamples, dtype=np.float64)
  for start in range(0, num_resamples, batch_size):
    k = min(batch_size, num_resamples - start)
    idx = rng.integers(0, n, size=(k, n), dtype=np.int32)
    means[start : start + k] = np.mean(summands_e2[idx], axis=1)

  low_mse = float(np.quantile(means, alpha / 2.0))
  high_mse = float(np.quantile(means, 1.0 - alpha / 2.0))
  return math.sqrt(max(0.0, low_mse)), math.sqrt(max(0.0, high_mse))


def bca_bootstrap_rmse(
    summands_e2: np.ndarray,
    confidence: float = 0.95,
    num_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
  """Efron BCa bootstrap for RMSE.

  Applies evidence/baselines.py::bca_bootstrap to squared residuals e^2,
  then takes square roots of the endpoints (monotone transform).

  Args:
    summands_e2: 1D array of squared errors e^2.
    confidence: Confidence level.
    num_resamples: Bootstrap sample count B.
    rng: Optional numpy Generator.

  Returns:
    (low, high) confidence interval for RMSE.
  """
  low_mse, high_mse = baselines.bca_bootstrap(
      summands_e2,
      confidence=confidence,
      num_resamples=num_resamples,
      rng=rng,
  )
  return math.sqrt(max(0.0, low_mse)), math.sqrt(max(0.0, high_mse))


def bayesian_bootstrap_rmse(
    summands_e2: np.ndarray,
    confidence: float = 0.95,
    num_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
  """Rubin's Bayesian bootstrap for RMSE.

  Draws Dirichlet(1, ..., 1) weights over items, computes weighted mean of e^2,
  and takes square roots of posterior quantiles.

  Args:
    summands_e2: 1D array of squared errors e^2.
    confidence: Confidence level.
    num_resamples: Bootstrap sample count B.
    rng: Optional numpy Generator.

  Returns:
    (low, high) confidence interval for RMSE.
  """
  n = len(summands_e2)
  if n < 2:
    return float("-inf"), float("inf")
  if rng is None:
    rng = np.random.default_rng()

  alpha = 1.0 - confidence
  batch_size = 100
  means = np.empty(num_resamples, dtype=np.float64)
  for start in range(0, num_resamples, batch_size):
    k = min(batch_size, num_resamples - start)
    g = rng.exponential(scale=1.0, size=(k, n)).astype(np.float64)
    w = g / np.sum(g, axis=1, keepdims=True)
    means[start : start + k] = w @ summands_e2

  low_mse = float(np.quantile(means, alpha / 2.0))
  high_mse = float(np.quantile(means, 1.0 - alpha / 2.0))
  return math.sqrt(max(0.0, low_mse)), math.sqrt(max(0.0, high_mse))


def student_t_rmse(
    summands_e2: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
  """Student-t on squared residuals, square-rooted endpoints (lower clipped at 0)."""
  low_mse, high_mse = baselines.student_t(summands_e2, confidence=confidence)
  return math.sqrt(max(0.0, low_mse)), math.sqrt(max(0.0, high_mse))


def moving_block_bootstrap_rmse(
    residuals_grid: np.ndarray,
    block_length: int,
    confidence: float = 0.95,
    num_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
  """Moving-block bootstrap over time, resampling whole time slices.

  Args:
    residuals_grid: 2D array of residuals of shape (num_nodes, num_eval_times).
    block_length: Temporal block length (e.g. L + H + R_t + 1).
    confidence: Confidence level.
    num_resamples: Bootstrap sample count B.
    rng: Optional numpy Generator.

  Returns:
    (low, high) confidence interval for RMSE.
  """
  num_nodes, num_times = residuals_grid.shape
  total_items = num_nodes * num_times
  if num_times < block_length or total_items < 2:
    return float("-inf"), float("inf")
  if rng is None:
    rng = np.random.default_rng()

  # Precompute time-slice sum of squared errors: shape (num_times,)
  slice_sums = np.sum(residuals_grid**2, axis=0)

  # Overlapping blocks of length block_length
  num_blocks = num_times - block_length + 1
  # Sum of each block
  block_sums = np.empty(num_blocks, dtype=np.float64)
  curr = float(np.sum(slice_sums[:block_length]))
  block_sums[0] = curr
  for j in range(1, num_blocks):
    curr += slice_sums[j + block_length - 1] - slice_sums[j - 1]
    block_sums[j] = curr

  # Number of blocks needed to cover num_times
  k_blocks = int(math.ceil(num_times / float(block_length)))
  idx = rng.integers(0, num_blocks, size=(num_resamples, k_blocks))
  boot_sse = np.sum(block_sums[idx], axis=1)
  # Correct for truncation if k_blocks * block_length > num_times
  if k_blocks * block_length != num_times:
    excess = k_blocks * block_length - num_times
    boot_sse *= float(num_times) / float(k_blocks * block_length)

  boot_mse = boot_sse / float(total_items)
  boot_rmse = np.sqrt(np.maximum(0.0, boot_mse))

  alpha = 1.0 - confidence
  low = float(np.quantile(boot_rmse, alpha / 2.0))
  high = float(np.quantile(boot_rmse, 1.0 - alpha / 2.0))
  return low, high


def cluster_bootstrap_rmse(
    residuals_grid: np.ndarray,
    confidence: float = 0.95,
    num_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
  """Cluster bootstrap resampling whole nodes (all times of a node).

  Args:
    residuals_grid: 2D array of residuals of shape (num_nodes, num_eval_times).
    confidence: Confidence level.
    num_resamples: Bootstrap sample count B.
    rng: Optional numpy Generator.

  Returns:
    (low, high) confidence interval for RMSE.
  """
  num_nodes, num_times = residuals_grid.shape
  total_items = num_nodes * num_times
  if num_nodes < 2 or total_items < 2:
    return float("-inf"), float("inf")
  if rng is None:
    rng = np.random.default_rng()

  # Node sum of squared errors across all time steps: shape (num_nodes,)
  node_sums = np.sum(residuals_grid**2, axis=1)

  idx = rng.integers(0, num_nodes, size=(num_resamples, num_nodes))
  boot_mse = np.sum(node_sums[idx], axis=1) / float(total_items)
  boot_rmse = np.sqrt(np.maximum(0.0, boot_mse))

  alpha = 1.0 - confidence
  low = float(np.quantile(boot_rmse, alpha / 2.0))
  high = float(np.quantile(boot_rmse, 1.0 - alpha / 2.0))
  return low, high


# ==============================================================================
# Metric Results Dataclasses
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class MethodCoverageResult:
  """Per-(config, method) results.

  `miss_rate`, `lower_miss_rate`, `upper_miss_rate` and the Clopper-Pearson
  interval (`cp_low`, `cp_high`, for the miss rate) use all replications as
  the denominator. A replication without an interval (library status
  ASSUMPTION_REQUIRED) is not a miss; it lowers `issuance_rate`.
  `conditional_coverage_rate` is coverage among issued intervals only.
  """

  config_name: str
  method: str
  theta_true: float
  num_reps: int
  num_issued: int
  issuance_rate: float
  num_covered: int
  miss_rate: float
  cp_low: float
  cp_high: float
  lower_misses: int
  upper_misses: int
  lower_miss_rate: float
  upper_miss_rate: float
  conditional_coverage_rate: float
  median_rel_half_width: float
  mean_rel_half_width: float
  runtime_sec: float
  extra: dict[str, Any] = dataclasses.field(default_factory=dict)


# ==============================================================================
# Part A: Dependent Spatio-Temporal Data
# ==============================================================================


def _eval_part_a_worker(args: tuple[Any, ...]) -> dict[str, Any]:
  """Worker function evaluating a chunk of Part A replications."""
  (
      chunk_seeds,
      graph,
      num_nodes,
      num_times,
      k_hops,
      l_lookback,
      horizon_h,
      r_s_noise,
      r_t_noise,
      variant,
      eval_times,
      k_mask,
      deg_k,
      block_length,
      winner_cluster_ids,
      resolved_kappa,
      choice_m_item,
      confidence,
      num_resamples,
      theta_true_rmse,
      only_library_range,
  ) = args

  methods = (
      ["library_range"]
      if only_library_range
      else [
          "percentile_bootstrap",
          "bca_bootstrap",
          "bayesian_bootstrap",
          "student_t",
          "moving_block_bootstrap",
          "cluster_bootstrap",
          "library_range",
      ]
  )

  issued = {m: 0 for m in methods}
  covered = {m: 0 for m in methods}
  lower_misses = {m: 0 for m in methods}
  upper_misses = {m: 0 for m in methods}
  half_widths = {m: [] for m in methods}
  method_times = {m: 0.0 for m in methods}
  lib_status_counts = {
      "unrefuted": 0,
      "tail_unresolved": 0,
      "assumption_required": 0,
  }
  unrefuted_covered = 0
  miss_and_unrefuted_count = 0
  num_eval = len(eval_times)

  for rep_seed in chunk_seeds:
    field = range_generators.generate_space_time_field(
        graph,
        num_times=num_times,
        r_s=r_s_noise,
        r_t=r_t_noise,
        variant=variant,
        rho_0=0.0,
        c=0.0,
        seed=rep_seed,
    )

    # Compute predictor and residuals
    rolling_temp = np.zeros((num_nodes, num_eval), dtype=np.float64)
    for l in range(l_lookback):
      rolling_temp += field[:, eval_times - l]
    pred = (k_mask @ rolling_temp) / (deg_k * float(l_lookback))
    targets = field[:, eval_times + horizon_h]
    residuals = targets - pred
    r_flat = residuals.reshape(-1)
    e2 = r_flat**2

    if not only_library_range:
      rep_boot_rng = np.random.default_rng(rep_seed)

      # 1. Percentile bootstrap
      t0 = time.perf_counter()
      p_low, p_high = percentile_bootstrap_rmse(
          e2, confidence=confidence, num_resamples=num_resamples, rng=rep_boot_rng
      )
      method_times["percentile_bootstrap"] += time.perf_counter() - t0
      _record_result("percentile_bootstrap", p_low, p_high, theta_true_rmse, issued, covered, lower_misses, upper_misses, half_widths)

      # 2. BCa bootstrap
      t0 = time.perf_counter()
      bca_low, bca_high = bca_bootstrap_rmse(
          e2, confidence=confidence, num_resamples=num_resamples, rng=rep_boot_rng
      )
      method_times["bca_bootstrap"] += time.perf_counter() - t0
      _record_result("bca_bootstrap", bca_low, bca_high, theta_true_rmse, issued, covered, lower_misses, upper_misses, half_widths)

      # 3. Bayesian bootstrap
      t0 = time.perf_counter()
      bb_low, bb_high = bayesian_bootstrap_rmse(
          e2, confidence=confidence, num_resamples=num_resamples, rng=rep_boot_rng
      )
      method_times["bayesian_bootstrap"] += time.perf_counter() - t0
      _record_result("bayesian_bootstrap", bb_low, bb_high, theta_true_rmse, issued, covered, lower_misses, upper_misses, half_widths)

      # 4. Student-t
      t0 = time.perf_counter()
      st_low, st_high = student_t_rmse(e2, confidence=confidence)
      method_times["student_t"] += time.perf_counter() - t0
      _record_result("student_t", st_low, st_high, theta_true_rmse, issued, covered, lower_misses, upper_misses, half_widths)

      # 5. Moving-block bootstrap
      t0 = time.perf_counter()
      mb_low, mb_high = moving_block_bootstrap_rmse(
          residuals, block_length=block_length, confidence=confidence, num_resamples=num_resamples, rng=rep_boot_rng
      )
      method_times["moving_block_bootstrap"] += time.perf_counter() - t0
      _record_result("moving_block_bootstrap", mb_low, mb_high, theta_true_rmse, issued, covered, lower_misses, upper_misses, half_widths)

      # 6. Cluster bootstrap
      t0 = time.perf_counter()
      cb_low, cb_high = cluster_bootstrap_rmse(
          residuals, confidence=confidence, num_resamples=num_resamples, rng=rep_boot_rng
      )
      method_times["cluster_bootstrap"] += time.perf_counter() - t0
      _record_result("cluster_bootstrap", cb_low, cb_high, theta_true_rmse, issued, covered, lower_misses, upper_misses, half_widths)

    # 7. Library range route
    t0 = time.perf_counter()
    shard = _cluster_sketch.PartialClusterShard.from_summands(e2, winner_cluster_ids)
    sk = shard.to_cluster_sketch(kappa_cluster=resolved_kappa)
    lib_res = _cluster_bound.cluster_interval(
        sk, metric="rmse", m=choice_m_item, level=confidence
    )
    method_times["library_range"] += time.perf_counter() - t0
    status_key = lib_res.status.value
    lib_status_counts[status_key] = lib_status_counts.get(status_key, 0) + 1
    if lib_res.status != _ind_interval.Status.ASSUMPTION_REQUIRED:
      _record_result("library_range", lib_res.low, lib_res.high, theta_true_rmse, issued, covered, lower_misses, upper_misses, half_widths)
      if lib_res.status == _ind_interval.Status.UNREFUTED:
        if lib_res.low <= theta_true_rmse <= lib_res.high:
          unrefuted_covered += 1
        else:
          miss_and_unrefuted_count += 1

  return {
      "issued": issued,
      "covered": covered,
      "lower_misses": lower_misses,
      "upper_misses": upper_misses,
      "half_widths": half_widths,
      "method_times": method_times,
      "lib_status_counts": lib_status_counts,
      "unrefuted_covered": unrefuted_covered,
      "miss_and_unrefuted_count": miss_and_unrefuted_count,
  }


def run_part_a(
    num_reps: int = 5000,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed_base: int = DEFAULT_SEED_PART_A,
    skip_t3000: bool = True,
    num_workers: int = 1,
    only_library_range: bool = False,
) -> list[MethodCoverageResult]:
  """Executes Part A: Dependent Spatio-Temporal RMSE Benchmark.

  Evaluates the 100k ring configuration (200 nodes x 500 time steps).
  Reports T=3000 as skipped if skip_t3000 is True.
  """
  logging.info("Starting Part A (Dependent Data Benchmark, reps=%d, workers=%d)...", num_reps, num_workers)
  num_nodes = 200
  num_times = 500
  k_hops = 1
  l_lookback = 3
  horizon_h = 1
  r_s_noise = 1
  r_t_noise = 2
  variant = 1
  decl_r_s = 2
  decl_r_t = 2
  decl_gamma = 0.0
  m_declared = 16.0

  # Temporal block length: L + H + R_t + 1 = 3 + 1 + 2 + 1 = 7
  block_length = l_lookback + horizon_h + decl_r_t + 1

  graph = range_generators.make_ring_graph(num_nodes)

  # Exact analytical truth for MSE
  theta_mse = compute_exact_theta(
      graph,
      k_hops=k_hops,
      l_lookback=l_lookback,
      horizon_h=horizon_h,
      r_s=r_s_noise,
      r_t=r_t_noise,
      variant=variant,
  )
  theta_true_rmse = math.sqrt(theta_mse)
  logging.info("Exact analytical truth: MSE=%.6f, RMSE=%.6f", theta_mse, theta_true_rmse)

  # Precompute design-only space-time indexing and partition choice
  declaration = RangeDeclaration(
      k_hops=k_hops,
      lookback=l_lookback,
      horizon=horizon_h,
      data_range_s=decl_r_s,
      data_range_t=decl_r_t,
      gamma=decl_gamma,
  )
  t_start = l_lookback - 1
  t_end = num_times - horizon_h
  eval_times = np.arange(t_start, t_end, dtype=np.int64)
  num_eval = len(eval_times)

  obs_u = np.repeat(np.arange(num_nodes, dtype=np.int64), num_eval)
  obs_t = np.tile(eval_times, num_nodes)
  index = spacetime.spacetime_items(obs_u, obs_t)

  cands = range_candidates(index, graph.edges, declaration)
  kappa_fn = range_kappa_fn(index, graph.edges, declaration, method="kronecker")
  choice = spacetime.choose_partition(
      cands,
      kappa_fn=kappa_fn,
      m_item=m_declared,
      level=confidence,
      is_complete_grid=True,
  )
  winner = choice.winner
  resolved_kappa = choice.winner_kappa

  k_mask = (graph.dist_matrix <= k_hops).astype(np.float64)
  deg_k = np.sum(k_mask, axis=1, keepdims=True)

  methods = (
      ["library_range"]
      if only_library_range
      else [
          "percentile_bootstrap",
          "bca_bootstrap",
          "bayesian_bootstrap",
          "student_t",
          "moving_block_bootstrap",
          "cluster_bootstrap",
          "library_range",
      ]
  )

  issued = {m: 0 for m in methods}
  covered = {m: 0 for m in methods}
  lower_misses = {m: 0 for m in methods}
  upper_misses = {m: 0 for m in methods}
  half_widths = {m: [] for m in methods}
  method_times = {m: 0.0 for m in methods}
  lib_status_counts = {
      "unrefuted": 0,
      "tail_unresolved": 0,
      "assumption_required": 0,
  }
  unrefuted_covered = 0
  miss_and_unrefuted_count = 0

  rng = np.random.default_rng(seed_base + 1)
  rep_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(num_reps)]

  if num_workers > 1 and num_reps >= num_workers:
    chunk_size = int(math.ceil(num_reps / float(num_workers)))
    chunks = [rep_seeds[i : i + chunk_size] for i in range(0, num_reps, chunk_size)]
    tasks = [
        (
            chunk,
            graph,
            num_nodes,
            num_times,
            k_hops,
            l_lookback,
            horizon_h,
            r_s_noise,
            r_t_noise,
            variant,
            eval_times,
            k_mask,
            deg_k,
            block_length,
            winner.cluster_ids,
            resolved_kappa,
            choice.m_item,
            confidence,
            num_resamples,
            theta_true_rmse,
            only_library_range,
        )
        for chunk in chunks
    ]
    with multiprocessing.Pool(processes=len(tasks)) as pool:
      chunk_results = pool.map(_eval_part_a_worker, tasks)

    for cr in chunk_results:
      for m in methods:
        issued[m] += cr["issued"][m]
        covered[m] += cr["covered"][m]
        lower_misses[m] += cr["lower_misses"][m]
        upper_misses[m] += cr["upper_misses"][m]
        half_widths[m].extend(cr["half_widths"][m])
        method_times[m] += cr["method_times"][m]
      for k, v in cr["lib_status_counts"].items():
        lib_status_counts[k] = lib_status_counts.get(k, 0) + v
      unrefuted_covered += cr["unrefuted_covered"]
      miss_and_unrefuted_count += cr["miss_and_unrefuted_count"]
  else:
    task_arg = (
        rep_seeds,
        graph,
        num_nodes,
        num_times,
        k_hops,
        l_lookback,
        horizon_h,
        r_s_noise,
        r_t_noise,
        variant,
        eval_times,
        k_mask,
        deg_k,
        block_length,
        winner.cluster_ids,
        resolved_kappa,
        choice.m_item,
        confidence,
        num_resamples,
        theta_true_rmse,
        only_library_range,
    )
    cr = _eval_part_a_worker(task_arg)
    for m in methods:
      issued[m] = cr["issued"][m]
      covered[m] = cr["covered"][m]
      lower_misses[m] = cr["lower_misses"][m]
      upper_misses[m] = cr["upper_misses"][m]
      half_widths[m] = cr["half_widths"][m]
      method_times[m] = cr["method_times"][m]
    lib_status_counts = dict(cr["lib_status_counts"])
    unrefuted_covered = cr["unrefuted_covered"]
    miss_and_unrefuted_count = cr["miss_and_unrefuted_count"]

  results: list[MethodCoverageResult] = []
  for m in methods:
    extra_info = {}
    if m == "library_range":
      extra_info["status_counts"] = dict(lib_status_counts)
      extra_info["unrefuted_rate"] = lib_status_counts["unrefuted"] / float(num_reps)
      extra_info["tail_unresolved_rate"] = lib_status_counts["tail_unresolved"] / float(num_reps)
      extra_info["assumption_required_rate"] = lib_status_counts["assumption_required"] / float(num_reps)
      extra_info["unrefuted_count"] = lib_status_counts["unrefuted"]
      extra_info["unrefuted_covered"] = unrefuted_covered
      extra_info["unrefuted_coverage"] = (
          unrefuted_covered / float(lib_status_counts["unrefuted"])
          if lib_status_counts["unrefuted"] > 0
          else None
      )
      extra_info["miss_and_unrefuted_count"] = miss_and_unrefuted_count
      extra_info["miss_and_unrefuted_rate"] = miss_and_unrefuted_count / float(num_reps)
    res = _build_summary(
        config_name="ring_100k",
        method=m,
        theta_true=theta_true_rmse,
        num_reps=num_reps,
        issued_count=issued[m],
        covered_count=covered[m],
        lower_miss_count=lower_misses[m],
        upper_miss_count=upper_misses[m],
        half_widths=half_widths[m],
        runtime_sec=method_times[m],
        extra=extra_info,
    )
    results.append(res)

  return results


# ==============================================================================
# Part B: Independent, Heavy-Tailed Error Distributions
# ==============================================================================


def get_part_b_distributions() -> list[tuple[str, distributions.ErrorDistribution]]:
  """Returns the 6 distributions specified for Part B."""
  return [
      ("gaussian", distributions.GAUSSIAN),
      ("laplace", distributions.LAPLACE),
      ("lognormal_1.0", distributions.make_lognormal(1.0)),
      ("student_t_5.0", distributions.make_student_t(5.0)),
      ("contaminated_gaussian", distributions.make_contaminated_gaussian(0.01, 10.0)),
      ("truncated_pareto_2_1e4", distributions.make_truncated_pareto(2.0, 1e4)),
  ]


def _eval_part_b_cell(task_args: tuple[Any, ...]) -> list[MethodCoverageResult]:
  """Evaluates one cell (dist x sample size) in Part B."""
  (
      dist_idx,
      dist_name,
      n_idx,
      n,
      num_reps,
      num_resamples,
      confidence,
      seed_base,
  ) = task_args
  dist = get_part_b_distributions()[dist_idx][1]
  config_name = f"{dist_name}_n{n}"
  theta_true = 1.0
  methods = [
      "percentile_bootstrap",
      "bca_bootstrap",
      "bayesian_bootstrap",
      "student_t",
      "library_independent",
  ]
  issued = {m: 0 for m in methods}
  covered = {m: 0 for m in methods}
  lower_misses = {m: 0 for m in methods}
  upper_misses = {m: 0 for m in methods}
  half_widths = {m: [] for m in methods}
  method_times = {m: 0.0 for m in methods}

  status_counts = {
      "unrefuted": 0,
      "tail_unresolved": 0,
      "assumption_required": 0,
  }
  unrefuted_covered = 0
  miss_and_unrefuted_count = 0

  cell_rng = np.random.default_rng(seed_base + dist_idx * 100 + n_idx)

  for rep in range(num_reps):
    errors = dist.sample(n, cell_rng)
    e2 = errors**2

    rep_boot_rng = np.random.default_rng(int(cell_rng.integers(0, 2**31 - 1)))

    # 1. Percentile bootstrap
    t0 = time.perf_counter()
    p_low, p_high = percentile_bootstrap_rmse(
        e2, confidence=confidence, num_resamples=num_resamples, rng=rep_boot_rng
    )
    method_times["percentile_bootstrap"] += time.perf_counter() - t0
    _record_result("percentile_bootstrap", p_low, p_high, theta_true, issued, covered, lower_misses, upper_misses, half_widths)

    # 2. BCa bootstrap
    t0 = time.perf_counter()
    bca_low, bca_high = bca_bootstrap_rmse(
        e2, confidence=confidence, num_resamples=num_resamples, rng=rep_boot_rng
    )
    method_times["bca_bootstrap"] += time.perf_counter() - t0
    _record_result("bca_bootstrap", bca_low, bca_high, theta_true, issued, covered, lower_misses, upper_misses, half_widths)

    # 3. Bayesian bootstrap
    t0 = time.perf_counter()
    bb_low, bb_high = bayesian_bootstrap_rmse(
        e2, confidence=confidence, num_resamples=num_resamples, rng=rep_boot_rng
    )
    method_times["bayesian_bootstrap"] += time.perf_counter() - t0
    _record_result("bayesian_bootstrap", bb_low, bb_high, theta_true, issued, covered, lower_misses, upper_misses, half_widths)

    # 4. Student-t
    t0 = time.perf_counter()
    st_low, st_high = student_t_rmse(e2, confidence=confidence)
    method_times["student_t"] += time.perf_counter() - t0
    _record_result("student_t", st_low, st_high, theta_true, issued, covered, lower_misses, upper_misses, half_widths)

    # 5. Library interval (stats.independent)
    t0 = time.perf_counter()
    ind_res = _ind_interval.confidence_interval(
        errors, metric="rmse", level=confidence
    )
    method_times["library_independent"] += time.perf_counter() - t0
    status_key = ind_res.status.value
    status_counts[status_key] = status_counts.get(status_key, 0) + 1

    if ind_res.status != _ind_interval.Status.ASSUMPTION_REQUIRED:
      _record_result("library_independent", ind_res.low, ind_res.high, theta_true, issued, covered, lower_misses, upper_misses, half_widths)
      if ind_res.status == _ind_interval.Status.UNREFUTED:
        if ind_res.low <= theta_true <= ind_res.high:
          unrefuted_covered += 1
        else:
          miss_and_unrefuted_count += 1

  cell_res = []
  for m in methods:
    extra_info = {}
    if m == "library_independent":
      extra_info["status_counts"] = dict(status_counts)
      extra_info["unrefuted_rate"] = status_counts["unrefuted"] / float(num_reps)
      extra_info["tail_unresolved_rate"] = status_counts["tail_unresolved"] / float(num_reps)
      extra_info["assumption_required_rate"] = status_counts["assumption_required"] / float(num_reps)
      extra_info["unrefuted_count"] = status_counts["unrefuted"]
      extra_info["unrefuted_covered"] = unrefuted_covered
      extra_info["unrefuted_coverage"] = (
          unrefuted_covered / float(status_counts["unrefuted"])
          if status_counts["unrefuted"] > 0
          else None
      )
      extra_info["miss_and_unrefuted_count"] = miss_and_unrefuted_count
      extra_info["miss_and_unrefuted_rate"] = miss_and_unrefuted_count / float(num_reps)

    res = _build_summary(
        config_name=config_name,
        method=m,
        theta_true=theta_true,
        num_reps=num_reps,
        issued_count=issued[m],
        covered_count=covered[m],
        lower_miss_count=lower_misses[m],
        upper_miss_count=upper_misses[m],
        half_widths=half_widths[m],
        runtime_sec=method_times[m],
        extra=extra_info,
    )
    cell_res.append(res)
  return cell_res


def run_part_b(
    num_reps: int = 5000,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    sample_sizes: Sequence[int] = (1000, 10000),
    seed_base: int = DEFAULT_SEED_PART_B,
    num_workers: int = 1,
    distributions: Sequence[tuple[str, Any]] | None = None,
) -> list[MethodCoverageResult]:
  """Executes Part B: Independent Heavy-Tailed RMSE Benchmark."""
  logging.info(
      "Starting Part B (Independent Data Benchmark, reps=%d, workers=%d)...",
      num_reps,
      num_workers,
  )
  dists = distributions if distributions is not None else get_part_b_distributions()

  tasks = []
  for dist_idx, (dist_name, dist) in enumerate(dists):
    if not dist.admits("rmse"):
      logging.warning("Skipping %s: does not admit rmse", dist_name)
      continue

    for n_idx, n in enumerate(sample_sizes):
      tasks.append((
          dist_idx,
          dist_name,
          n_idx,
          n,
          num_reps,
          num_resamples,
          confidence,
          seed_base,
      ))

  all_results: list[MethodCoverageResult] = []
  if num_workers > 1 and len(tasks) > 1:
    with multiprocessing.Pool(processes=min(num_workers, len(tasks))) as pool:
      results_nested = pool.map(_eval_part_b_cell, tasks)
    for cell_res in results_nested:
      all_results.extend(cell_res)
  else:
    for task in tasks:
      all_results.extend(_eval_part_b_cell(task))

  return all_results



# ==============================================================================
# Helper Functions
# ==============================================================================


def _record_result(
    method: str,
    low: float,
    high: float,
    theta_true: float,
    issued: dict[str, int],
    covered: dict[str, int],
    lower_misses: dict[str, int],
    upper_misses: dict[str, int],
    half_widths: dict[str, list[float]],
) -> None:
  """Records one replication's interval coverage."""
  issued[method] += 1
  if low <= theta_true <= high:
    covered[method] += 1
  elif theta_true < low:
    lower_misses[method] += 1
  else:
    upper_misses[method] += 1

  if math.isfinite(low) and math.isfinite(high):
    half_widths[method].append((high - low) / (2.0 * theta_true))


def _build_summary(
    config_name: str,
    method: str,
    theta_true: float,
    num_reps: int,
    issued_count: int,
    covered_count: int,
    lower_miss_count: int,
    upper_miss_count: int,
    half_widths: list[float],
    runtime_sec: float,
    extra: dict[str, Any] | None = None,
) -> MethodCoverageResult:
  """Constructs a MethodCoverageResult summary record."""
  issuance_rate = issued_count / float(num_reps) if num_reps > 0 else 0.0
  num_missed = lower_miss_count + upper_miss_count
  miss_rate = num_missed / float(num_reps) if num_reps > 0 else 0.0
  cp_low, cp_high = baselines.clopper_pearson(num_missed, num_reps, confidence=0.95)
  lower_miss_rate = lower_miss_count / float(num_reps) if num_reps > 0 else 0.0
  upper_miss_rate = upper_miss_count / float(num_reps) if num_reps > 0 else 0.0
  conditional_coverage_rate = (
      covered_count / float(issued_count) if issued_count > 0 else float("nan")
  )

  if half_widths:
    med_hw = float(np.median(half_widths))
    mean_hw = float(np.mean(half_widths))
  else:
    med_hw = float("nan")
    mean_hw = float("nan")

  return MethodCoverageResult(
      config_name=config_name,
      method=method,
      theta_true=theta_true,
      num_reps=num_reps,
      num_issued=issued_count,
      issuance_rate=issuance_rate,
      num_covered=covered_count,
      miss_rate=miss_rate,
      cp_low=cp_low,
      cp_high=cp_high,
      lower_misses=lower_miss_count,
      upper_misses=upper_miss_count,
      lower_miss_rate=lower_miss_rate,
      upper_miss_rate=upper_miss_rate,
      conditional_coverage_rate=conditional_coverage_rate,
      median_rel_half_width=med_hw,
      mean_rel_half_width=mean_hw,
      runtime_sec=runtime_sec,
      extra=extra or {},
  )


def print_results_table(title: str, results: Sequence[MethodCoverageResult]) -> None:
  """Prints formatted summary table to stdout."""
  print("\n" + "=" * 135)
  print(f"{title}")
  print("=" * 135)
  header = (
      f"{'Config':<20} {'Method':<24} {'Miss (95% CI)':<26} "
      f"{'Low/High':<11} {'Unref. miss':<12} {'Med RelHW':<11} {'Issuance':<10} {'Diagnostic Counts':<20}"
  )
  print(header)
  print("-" * 135)
  for r in results:
    # All rates below use every replication as the denominator.
    miss_str = f"{r.miss_rate:.2%} [{r.cp_low:.4f}, {r.cp_high:.4f}]"
    lh_str = f"{r.lower_misses}/{r.upper_misses}"

    extra = r.extra
    if "miss_and_unrefuted_count" in extra:
      joint_miss = extra["miss_and_unrefuted_count"] / max(1, r.num_reps)
      joint_miss_str = f"{joint_miss:.2%}"
    else:
      joint_miss_str = "-"

    diag_str = ""
    if "lib_status_counts" in extra:
      counts = extra["lib_status_counts"]
      diag_str = f"U:{counts.get('unrefuted', 0)} A:{counts.get('assumption_required', 0)} T:{counts.get('tail_unresolved', 0)}"

    hw_str = f"{r.median_rel_half_width:.4f}" if math.isfinite(r.median_rel_half_width) else "nan"
    print(
        f"{r.config_name:<20} {r.method:<24} {miss_str:<26} "
        f"{lh_str:<11} {joint_miss_str:<12} {hw_str:<11} {r.issuance_rate:<10.2%} {diag_str:<20}"
    )


# ==============================================================================
# Main and Flags
# ==============================================================================

_MODE = flags.DEFINE_enum(
    "mode",
    "quick",
    ["quick", "full"],
    "Execution mode: 'quick' or 'full'.",
)
_SEED = flags.DEFINE_integer(
    "seed",
    None,
    "Optional seed override for all parts.",
)
_SEED_PART_A = flags.DEFINE_integer(
    "seed_part_a",
    DEFAULT_SEED_PART_A,
    "Random seed for Part A.",
)
_SEED_PART_B = flags.DEFINE_integer(
    "seed_part_b",
    DEFAULT_SEED_PART_B,
    "Random seed for Part B.",
)
_PART = flags.DEFINE_enum(
    "part",
    "all",
    ["all", "dependent", "iid"],
    "Which part to execute: 'dependent' (Part A), 'iid' (Part B), or 'all'.",
)
_REPS = flags.DEFINE_integer(
    "num_reps",
    5000,
    "Number of replications per configuration (in full mode).",
)
_NUM_RESAMPLES = flags.DEFINE_integer(
    "num_resamples",
    1000,
    "Number of bootstrap resamples (B) (in full mode).",
)
_OUTPUT_DEP = flags.DEFINE_string(
    "output_dependent",
    None,
    "Optional path to output JSONL results file for Part A.",
)
_OUTPUT_IID = flags.DEFINE_string(
    "output_iid",
    None,
    "Optional path to output JSONL results file for Part B.",
)
_WORKERS = flags.DEFINE_integer(
    "workers",
    6,
    "Number of worker processes for parallel evaluation (at most 6 = quarter of cores).",
)
_ONLY_LIBRARY_RANGE = flags.DEFINE_boolean(
    "only_library_range",
    False,
    "If True, Part A evaluates only the library_range method on the same seeds.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  seed_a = _SEED.value if _SEED.value is not None else _SEED_PART_A.value
  seed_b = _SEED.value if _SEED.value is not None else _SEED_PART_B.value

  if _MODE.value == "quick":
    reps = 20
    num_boot = 50
    part_b_dists = [get_part_b_distributions()[0]]  # gaussian only
    part_b_sizes = (1000,)
  else:
    reps = _REPS.value
    num_boot = _NUM_RESAMPLES.value
    part_b_dists = None  # all admitted distributions
    part_b_sizes = (1000, 10000)

  if _PART.value in ("all", "dependent"):
    res_a = run_part_a(
        num_reps=reps,
        num_resamples=num_boot,
        seed_base=seed_a,
        num_workers=_WORKERS.value,
        only_library_range=_ONLY_LIBRARY_RANGE.value,
    )
    print_results_table(f"PART A: DEPENDENT SPATIO-TEMPORAL RMSE ({reps} REPS)", res_a)

    dep_path = _OUTPUT_DEP.value
    if dep_path:
      os.makedirs(os.path.dirname(dep_path), exist_ok=True)
      with open(dep_path, "w", encoding="utf-8") as f:
        for r in res_a:
          f.write(json.dumps(dataclasses.asdict(r)) + "\n")
      logging.info("Wrote Part A results to %s", dep_path)

  if _PART.value in ("all", "iid"):
    res_b = run_part_b(
        num_reps=reps,
        num_resamples=num_boot,
        sample_sizes=part_b_sizes,
        seed_base=seed_b,
        num_workers=_WORKERS.value,
        distributions=part_b_dists,
    )
    print_results_table(f"PART B: I.I.D. HEAVY-TAILED RMSE ({reps} REPS)", res_b)

    iid_path = _OUTPUT_IID.value
    if iid_path:
      os.makedirs(os.path.dirname(iid_path), exist_ok=True)
      with open(iid_path, "w", encoding="utf-8") as f:
        for r in res_b:
          f.write(json.dumps(dataclasses.asdict(r)) + "\n")
      logging.info("Wrote Part B results to %s", iid_path)


if __name__ == "__main__":
  app.run(main)
