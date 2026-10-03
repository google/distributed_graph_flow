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

"""Coverage benchmark for range-envelope dependence declarations.

Evaluates the coverage of space-time MSE intervals built from a
`RangeDeclaration` against the analytical truth theta(D), over 9
configurations on ring and 2D lattice graphs (5000 replications each in full
mode).
"""

from collections.abc import Sequence
import concurrent.futures
import dataclasses
import json
import math
import os
import time
from absl import app
from absl import flags
import numpy as np
from scipy import stats

from dgf.src.stats import cluster_bound as _cluster_bound
from dgf.src.stats import cluster_sketch as _cluster_sketch
from dgf.src.stats import range_envelope as _re
from dgf.src.stats import spacetime
from dgf.src.stats.benchmark import range_generators
from dgf.src.stats.independent import interval as _ind_interval

RangeDeclaration = _re.RangeDeclaration
RangeKappa = _re.RangeKappa
range_kappa = _re.range_kappa
range_kappa_fn = _re.range_kappa_fn
range_candidates = _re.range_candidates

# ==============================================================================
# Exact Analytical Truth theta(D) and Verification
# ==============================================================================


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
  """Computes exact analytical truth theta(D) = E[s_{u,t}] for the moving-average model.

  By linearity and Gaussianity:
    e_{u,t} = Y_{u, t+H} - Y_hat_{u, t+H}
  is a deterministic linear combination of i.i.d. standard Gaussian innovations:
    e_{u,t} = sum_{v, tau} a_{v, tau} * epsilon_{v, tau} (+ common/diffuse shock).
  Since E[e] = 0, E[s] = Var(e) = sum_{v, tau} a_{v, tau}^2.

  For Variant 2 (common shock rho_0):
    Var(e) = (1 - rho_0) * Var(e_eta) + rho_0 * (1 + 1 / L).

  For Variant 3 (diffuse term c/n):
    Var(e) = (1 - c / n) * Var(e_eta) + (c / n) * (1 + 1 / L).

  Args:
    graph: GraphStructure.
    k_hops: Predictor spatial neighborhood radius K.
    l_lookback: Predictor temporal lookback window length L.
    horizon_h: Prediction horizon H.
    r_s: Moving average spatial kernel radius.
    r_t: Moving average temporal lag depth.
    variant: 1 (pure), 2 (common shock), 3 (diffuse).
    rho_0: Correlation of common shock in Variant 2.
    c: Parameter for diffuse term in Variant 3.

  Returns:
    Exact analytical value of theta(D).
  """
  n = graph.num_nodes
  u_target = 0
  t_origin = 0

  # Support of innovations for e_{0, 0}:
  # Spatial support: nodes v with dist(0, v) <= K + r_s
  max_dist = k_hops + r_s
  active_nodes = [v for v in range(n) if graph.dist_matrix[u_target, v] <= max_dist]

  # Temporal support: tau in [-(L - 1) - r_t,  H]
  tau_min = -(l_lookback - 1) - r_t
  tau_max = horizon_h
  tau_steps = list(range(tau_min, tau_max + 1))

  # Precompute degrees
  s_mask = (graph.dist_matrix <= r_s).astype(np.float64)
  deg_s = np.sum(s_mask, axis=1)  # shape (n,)
  norm_s = np.sqrt(deg_s * float(r_t + 1))  # normalization for eta

  k_mask = (graph.dist_matrix <= k_hops).astype(np.float64)
  deg_k_target = float(np.sum(k_mask[u_target, :]))

  # Evaluate impulse response for each unit innovation epsilon_{v0, tau0} = 1
  a_sq_sum = 0.0

  for v0 in active_nodes:
    for tau0 in tau_steps:
      # 1. Contribution to target Y_{u_target, H} = eta_{u_target, H}
      # eta_{u, H} has innovation at (v0, tau0) if dist(u_target, v0) <= r_s
      # and tau0 in [H - r_t, H].
      if graph.dist_matrix[u_target, v0] <= r_s and (horizon_h - r_t <= tau0 <= horizon_h):
        w_y = 1.0 / norm_s[u_target]
      else:
        w_y = 0.0

      # 2. Contribution to prediction Y_hat_{u_target, H}:
      # Y_hat is (1 / (deg_k * L)) * sum_{v in N_K(u)} sum_{l=0}^{L-1} eta_{v, -l}
      w_pred = 0.0
      for v in range(n):
        if graph.dist_matrix[u_target, v] <= k_hops:
          if graph.dist_matrix[v, v0] <= r_s:
            # eta_{v, -l} has innovation at (v0, tau0) if tau0 in [-l - r_t, -l]
            # <=> -l - r_t <= tau0 <= -l <=> -tau0 - r_t <= l <= -tau0
            l_start = max(0, -tau0 - r_t)
            l_end = min(l_lookback - 1, -tau0)
            if l_start <= l_end:
              num_matching_l = (l_end - l_start + 1)
              w_pred += (float(num_matching_l) / norm_s[v])

      w_pred /= (deg_k_target * float(l_lookback))

      coeff = w_y - w_pred
      a_sq_sum += coeff**2

  var_eta = a_sq_sum
  var_common = 1.0 + (1.0 / float(l_lookback))

  if variant == 1:
    return float(var_eta)
  elif variant == 2:
    return float((1.0 - rho_0) * var_eta + rho_0 * var_common)
  elif variant == 3:
    alpha = c / float(n)
    return float((1.0 - alpha) * var_eta + alpha * var_common)
  else:
    raise ValueError(f"Unknown variant: {variant}")


def verify_theta_with_mc(
    graph: range_generators.GraphStructure,
    k_hops: int,
    l_lookback: int,
    horizon_h: int,
    r_s: int,
    r_t: int,
    variant: int = 1,
    rho_0: float = 0.0,
    c: float = 0.0,
    num_reps: int = 200_000,
    seed: int = 12345,
) -> tuple[float, float, float]:
  """Empirically estimates theta(D) via 200,000 independent single-item replications."""
  rng = np.random.default_rng(seed)
  # Sample 200k replications at node 0, origin time 0
  losses_mc = []
  batch_size = 20_000
  rem = num_reps

  t_needed = l_lookback + horizon_h + r_t + 2
  while rem > 0:
    curr_b = min(rem, batch_size)
    rem -= curr_b

    # Generate small field of length t_needed for curr_b replications
    field_batch = []
    for _ in range(curr_b):
      f = range_generators.generate_space_time_field(
          graph, num_times=t_needed, r_s=r_s, r_t=r_t, variant=variant,
          rho_0=rho_0, c=c, seed=int(rng.integers(0, 2**31 - 1)),
      )
      l, _ = range_generators.compute_predictor_and_losses(
          f, graph, k_hops=k_hops, l_lookback=l_lookback, horizon_h=horizon_h
      )
      field_batch.append(l[0, 0])
    losses_mc.extend(field_batch)

  mean_mc = float(np.mean(losses_mc))
  std_mc = float(np.std(losses_mc))
  se_mc = std_mc / math.sqrt(num_reps)
  return mean_mc, std_mc, se_mc


# ==============================================================================
# Clopper-Pearson Exact Binomial Confidence Interval
# ==============================================================================


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
  """Exact Clopper-Pearson confidence interval for a binomial proportion."""
  if n == 0:
    return (float("nan"), float("nan"))
  alpha = 1.0 - confidence
  low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
  high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
  return low, high


# ==============================================================================
# Configuration Data Structures
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
  config_id: int
  name: str
  graph_kind: str
  num_nodes: int
  num_times: int
  k_hops: int
  l_lookback: int
  horizon_h: int
  r_s_noise: int
  r_t_noise: int
  variant: int
  rho_0: float
  c_param: float
  decl_r_s: int
  decl_r_t: int
  decl_gamma: float
  description: str


@dataclasses.dataclass(frozen=True)
class ConfigResult:
  """Per-configuration results. Rates use all replications as denominator."""

  config_id: int
  name: str
  theta_true: float
  num_reps: int
  num_missed_low: int
  num_missed_high: int
  miss_rate: float
  cp_low: float  # Clopper-Pearson 95% interval for the miss rate.
  cp_high: float
  num_unrefuted_miss: int
  num_refuted: int
  num_assumption_required: int
  median_width: float
  mean_width: float
  median_kappa: float
  winner_name: str
  method_used: str
  relative_half_width: float = 0.0


# ==============================================================================
# Benchmark Orchestrator
# ==============================================================================


def get_standard_configs() -> list[BenchmarkConfig]:
  """Returns the 7 standard range-envelope benchmark configurations."""
  configs = [
      BenchmarkConfig(
          config_id=1,
          name="Config 1: Variant 1 (Pure, True Decl, gamma=0)",
          graph_kind="ring",
          num_nodes=200,
          num_times=500,
          k_hops=1,
          l_lookback=3,
          horizon_h=1,
          r_s_noise=1,
          r_t_noise=2,
          variant=1,
          rho_0=0.0,
          c_param=0.0,
          decl_r_s=2,
          decl_r_t=2,
          decl_gamma=0.0,
          description="Pure finite-range field, exact declaration (R_s=2, R_t=2, gamma=0).",
      ),
      BenchmarkConfig(
          config_id=2,
          name="Config 2: Variant 3 (Diffuse c=2, gamma=c)",
          graph_kind="ring",
          num_nodes=200,
          num_times=500,
          k_hops=1,
          l_lookback=3,
          horizon_h=1,
          r_s_noise=1,
          r_t_noise=2,
          variant=3,
          rho_0=0.0,
          c_param=2.0,
          decl_r_s=2,
          decl_r_t=2,
          decl_gamma=2.0,
          description="Diffuse global term with c=2, budget declared as gamma=c=2.0.",
      ),
      BenchmarkConfig(
          config_id=3,
          name="Config 3: Variant 2 (Common shock rho_0=0.05, gamma=0, under-declared)",
          graph_kind="ring",
          num_nodes=200,
          num_times=500,
          k_hops=1,
          l_lookback=3,
          horizon_h=1,
          r_s_noise=1,
          r_t_noise=2,
          variant=2,
          rho_0=0.05,
          c_param=0.0,
          decl_r_s=2,
          decl_r_t=2,
          decl_gamma=0.0,
          description="Common shock rho_0=0.05 with under-declared gamma=0.",
      ),
      BenchmarkConfig(
          config_id=4,
          name="Config 4: Variant 2 (Common shock rho_0=0.05, gamma=(n-1)*rho_0)",
          graph_kind="ring",
          num_nodes=200,
          num_times=500,
          k_hops=1,
          l_lookback=3,
          horizon_h=1,
          r_s_noise=1,
          r_t_noise=2,
          variant=2,
          rho_0=0.05,
          c_param=0.0,
          decl_r_s=2,
          decl_r_t=2,
          decl_gamma=(200 - 1) * 0.05,  # 9.95
          description="Common shock rho_0=0.05 with correct budget gamma=(n-1)*rho_0=9.95.",
      ),
      BenchmarkConfig(
          config_id=5,
          name="Config 5: Variant 1 (Under-declared R_t=0)",
          graph_kind="ring",
          num_nodes=200,
          num_times=500,
          k_hops=1,
          l_lookback=3,
          horizon_h=1,
          r_s_noise=1,
          r_t_noise=2,
          variant=1,
          rho_0=0.0,
          c_param=0.0,
          decl_r_s=2,
          decl_r_t=0,
          decl_gamma=0.0,
          description="Pure field with under-declared temporal range R_t=0 (true R_t=2).",
      ),
      BenchmarkConfig(
          config_id=6,
          name="Config 6: Variant 1 (Over-declared R_s=4, R_t=6, gamma=1)",
          graph_kind="ring",
          num_nodes=200,
          num_times=500,
          k_hops=1,
          l_lookback=3,
          horizon_h=1,
          r_s_noise=1,
          r_t_noise=2,
          variant=1,
          rho_0=0.0,
          c_param=0.0,
          decl_r_s=4,
          decl_r_t=6,
          decl_gamma=1.0,
          description="Pure field with conservative over-declared ranges and gamma=1.0.",
      ),
      BenchmarkConfig(
          config_id=7,
          name="Config 7: 20x20 Lattice (Pure, True Decl, gamma=0)",
          graph_kind="grid_2d",
          num_nodes=400,
          num_times=500,
          k_hops=1,
          l_lookback=3,
          horizon_h=1,
          r_s_noise=1,
          r_t_noise=2,
          variant=1,
          rho_0=0.0,
          c_param=0.0,
          decl_r_s=2,
          decl_r_t=2,
          decl_gamma=0.0,
          description="2D Lattice (20x20=400 nodes), exact declaration (R_s=2, R_t=2, gamma=0).",
      ),
  ]
  return configs


def run_single_config_benchmark(
    cfg: BenchmarkConfig,
    num_reps: int = 5000,
    seed_base: int = 0x820000,
    method: str = "kronecker",
) -> ConfigResult:
  """Runs `num_reps` coverage replications for a single configuration."""
  print(f"\nRunning {cfg.name} ({num_reps} replications, method={method})...", flush=True)

  if cfg.graph_kind == "ring":
    graph = range_generators.make_ring_graph(cfg.num_nodes)
  elif cfg.graph_kind == "grid_2d":
    side = int(round(math.sqrt(cfg.num_nodes)))
    graph = range_generators.make_grid_2d_graph(side, side)
  else:
    raise ValueError(f"Unknown graph kind: {cfg.graph_kind}")

  # 1. Compute exact analytical truth
  theta_true = compute_exact_theta(
      graph,
      k_hops=cfg.k_hops,
      l_lookback=cfg.l_lookback,
      horizon_h=cfg.horizon_h,
      r_s=cfg.r_s_noise,
      r_t=cfg.r_t_noise,
      variant=cfg.variant,
      rho_0=cfg.rho_0,
      c=cfg.c_param,
  )
  print(f"  Exact analytical truth theta(D) = {theta_true:.6f}", flush=True)

  # 2. Design-only setup (precomputed once per configuration)
  declaration = RangeDeclaration(
      k_hops=cfg.k_hops,
      lookback=cfg.l_lookback,
      horizon=cfg.horizon_h,
      data_range_s=cfg.decl_r_s,
      data_range_t=cfg.decl_r_t,
      gamma=cfg.decl_gamma,
  )

  t_start = cfg.l_lookback - 1
  t_end = cfg.num_times - cfg.horizon_h
  eval_times = np.arange(t_start, t_end, dtype=np.int64)
  num_eval = len(eval_times)

  obs_u = np.repeat(np.arange(cfg.num_nodes, dtype=np.int64), num_eval)
  obs_t = np.tile(eval_times, cfg.num_nodes)
  index = spacetime.spacetime_items(obs_u, obs_t)

  # Candidates and partition choice depend only on the design (index, graph,
  # declaration), never on the residuals.
  cands = range_candidates(index, graph.edges, declaration)
  kappa_fn = range_kappa_fn(index, graph.edges, declaration, method=method)

  choice = spacetime.choose_partition(
      cands,
      kappa_fn=kappa_fn,
      m_item=16.0,
      level=0.95,
      is_complete_grid=True,
  )
  print(
      f"  Partition: winner={choice.winner.name} (G={choice.winner.G},"
      f" kappa={choice.winner_kappa:.4f})",
      flush=True,
  )


  winner = choice.winner
  resolved_kappa = choice.winner_kappa

  # 3. Replication loop. The design is fixed, so only the field is redrawn.
  # Every replication counts in the denominator; a replication with status
  # ASSUMPTION_REQUIRED has no interval and is counted separately, not as a
  # miss.
  status_enum = _ind_interval.Status
  k_mask = (graph.dist_matrix <= cfg.k_hops).astype(np.float64)
  deg_k = np.sum(k_mask, axis=1, keepdims=True)
  missed_low = 0  # theta < low
  missed_high = 0  # theta > high
  unrefuted_miss = 0
  num_refuted = 0
  num_assumption_required = 0
  widths = []
  rng = np.random.default_rng(seed_base + cfg.config_id)

  t0_reps = time.perf_counter()
  last_progress_log = t0_reps
  for rep_idx in range(num_reps):
    rep_seed = int(rng.integers(0, 2**31 - 1))
    field = range_generators.generate_space_time_field(
        graph,
        num_times=cfg.num_times,
        r_s=cfg.r_s_noise,
        r_t=cfg.r_t_noise,
        variant=cfg.variant,
        rho_0=cfg.rho_0,
        c=cfg.c_param,
        seed=rep_seed,
    )

    # Raw residuals e = target - prediction; the squared residuals are the
    # MSE summands passed to PartialClusterShard.from_summands below.
    rolling_temp = np.zeros((cfg.num_nodes, num_eval), dtype=np.float64)
    for l in range(cfg.l_lookback):
      rolling_temp += field[:, eval_times - l]
    pred = (k_mask @ rolling_temp) / (deg_k * float(cfg.l_lookback))
    targets = field[:, eval_times + cfg.horizon_h]
    residuals = targets - pred

    # Flatten residuals matching index order
    r_flat = residuals.reshape(-1)

    # Shard and interval
    shard = _cluster_sketch.PartialClusterShard.from_summands(r_flat**2, winner.cluster_ids)
    sk = shard.to_cluster_sketch(kappa_cluster=resolved_kappa)
    res = _cluster_bound.cluster_interval(
        sk, metric="mse", level=0.95, m_item=16.0, kappa_cluster=resolved_kappa
    )

    if res.status is status_enum.ASSUMPTION_REQUIRED or not math.isfinite(
        res.low
    ):
      num_assumption_required += 1
      continue
    if res.status is status_enum.TAIL_UNRESOLVED:
      num_refuted += 1
    widths.append(float(res.high - res.low))
    is_miss = False
    if theta_true < res.low:
      missed_low += 1
      is_miss = True
    elif theta_true > res.high:
      missed_high += 1
      is_miss = True
    if is_miss and res.status is status_enum.UNREFUTED:
      unrefuted_miss += 1

    now = time.perf_counter()
    if now - last_progress_log >= 60.0:
      print(
          f"  [{cfg.name}] progress: {rep_idx + 1}/{num_reps} reps"
          f" ({((rep_idx + 1) / num_reps):.1%}) in {now - t0_reps:.1f}s",
          flush=True,
      )
      last_progress_log = now

  time_total = time.perf_counter() - t0_reps
  num_missed = missed_low + missed_high
  miss_rate = float(num_missed) / float(num_reps)
  cp_low, cp_high = clopper_pearson(num_missed, num_reps, confidence=0.95)
  med_w = float(np.median(widths)) if widths else float("nan")

  mean_w = float(np.mean(widths)) if widths else float("nan")
  rel_hw = (
      float(med_w / (2.0 * theta_true))
      if (widths and theta_true > 0 and math.isfinite(med_w))
      else float("nan")
  )

  print(
      f"  Done {num_reps} reps in {time_total:.2f}s: "
      f"misses = {num_missed} ({missed_low} low / {missed_high} high), "
      f"miss rate = {miss_rate:.2%} [{cp_low:.4f}, {cp_high:.4f}], "
      f"UNREFUTED misses = {unrefuted_miss}, refuted = {num_refuted}, "
      f"ASSUMPTION_REQUIRED = {num_assumption_required}, "
      f"median width = {med_w:.4f}, relative half-width = {rel_hw:.4f}",
      flush=True,
  )

  return ConfigResult(
      config_id=cfg.config_id,
      name=cfg.name,
      theta_true=theta_true,
      num_reps=num_reps,
      num_missed_low=missed_low,
      num_missed_high=missed_high,
      miss_rate=miss_rate,
      cp_low=cp_low,
      cp_high=cp_high,
      num_unrefuted_miss=unrefuted_miss,
      num_refuted=num_refuted,
      num_assumption_required=num_assumption_required,
      median_width=med_w,
      mean_width=mean_w,
      median_kappa=float(resolved_kappa),
      winner_name=winner.name,
      method_used=method,
      relative_half_width=rel_hw,
  )


def get_long_panel_configs() -> list[BenchmarkConfig]:
  """Returns configurations 1 and 2 repeated on a longer panel (T=3000)."""
  return [
      BenchmarkConfig(
          config_id=101,
          name="Config 101: Config 1 at T=3000",
          graph_kind="ring",
          num_nodes=200,
          num_times=3000,
          k_hops=1,
          l_lookback=3,
          horizon_h=1,
          r_s_noise=1,
          r_t_noise=2,
          variant=1,
          rho_0=0.0,
          c_param=0.0,
          decl_r_s=2,
          decl_r_t=2,
          decl_gamma=0.0,
          description="Config 1 on a T=3000 panel.",
      ),
      BenchmarkConfig(
          config_id=102,
          name="Config 102: Config 2 at T=3000",
          graph_kind="ring",
          num_nodes=200,
          num_times=3000,
          k_hops=1,
          l_lookback=3,
          horizon_h=1,
          r_s_noise=1,
          r_t_noise=2,
          variant=3,
          rho_0=0.0,
          c_param=2.0,
          decl_r_s=2,
          decl_r_t=2,
          decl_gamma=2.0,
          description="Config 2 on a T=3000 panel.",
      ),
  ]


def run_all_benchmarks(
    num_reps: int = 5000,
    include_long_panels: bool = True,
    results_path: str | None = None,
    seed: int = 0x820000,
    workers: int = 1,
) -> list[ConfigResult]:
  """Executes all benchmark configurations and records results."""
  configs = get_standard_configs()
  if include_long_panels:
    configs.extend(get_long_panel_configs())
  results = []

  print("=" * 88)
  print(f"RANGE ENVELOPE COVERAGE HARNESS ({num_reps} REPLICATIONS)")
  print("=" * 88)

  if workers > 1:
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
      future_to_cfg = {
          executor.submit(
              run_single_config_benchmark,
              cfg,
              num_reps=num_reps,
              seed_base=seed,
              method="kronecker",
          ): cfg
          for cfg in configs
      }
      results_by_id = {}
      for future in concurrent.futures.as_completed(future_to_cfg):
        cfg = future_to_cfg[future]
        res = future.result()
        results_by_id[cfg.config_id] = res
      results = [results_by_id[cfg.config_id] for cfg in configs]
  else:
    for cfg in configs:
      res = run_single_config_benchmark(
          cfg, num_reps=num_reps, seed_base=seed, method="kronecker"
      )
      results.append(res)

  # Print Summary Table. All rates use every replication as the denominator.
  print("\n" + "=" * 110)
  print(f"SUMMARY RESULTS TABLE ({num_reps} REPLICATIONS PER CONFIGURATION)")
  print("=" * 110)
  print(
      f"{'Config':<8} {'Miss (95% CI)':<26} {'Miss low/high':<14}"
      f" {'Unref. miss':<12} {'Refuted':<8} {'Assump.':<8}"
      f" {'Med Width':<11} {'Kappa':<10} {'Winner':<18}"
  )
  print("-" * 110)
  for r in results:
    miss_str = f"{r.miss_rate:.2%} [{r.cp_low:.4f}, {r.cp_high:.4f}]"
    print(
        f"{r.config_id:<8d} {miss_str:<26}"
        f" {f'{r.num_missed_low}/{r.num_missed_high}':<14}"
        f" {r.num_unrefuted_miss:<12d} {r.num_refuted:<8d}"
        f" {r.num_assumption_required:<8d} {r.median_width:<11.4f}"
        f" {r.median_kappa:<10.4f} {r.winner_name:<18}"
    )

  if results_path:
    out_dir = os.path.dirname(results_path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
      for r in results:
        f.write(json.dumps(dataclasses.asdict(r)) + "\n")
    print(f"\nWrote {len(results)} records to {results_path}")

  return results


DEFAULT_SEED = 0x820000

_SEED = flags.DEFINE_integer("seed", DEFAULT_SEED, "Base random seed.")
_MODE = flags.DEFINE_enum("mode", "quick", ["quick", "full"], "Execution mode: 'quick' or 'full'.")
_OUTPUT = flags.DEFINE_string("output", None, "Optional path to output JSONL results file.")
_REPS = flags.DEFINE_integer("reps", 0, "Optional override for replications.")
_WORKERS = flags.DEFINE_integer(
    "workers", 1, "Number of worker processes (parallelize across configs)."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  mode = str(_MODE.value)
  if mode == "quick":
    num_reps = int(_REPS.value or 20)
    include_long_panels = False
  else:
    num_reps = int(_REPS.value or 5000)
    include_long_panels = True
  run_all_benchmarks(
      num_reps=num_reps,
      include_long_panels=include_long_panels,
      results_path=_OUTPUT.value,
      seed=int(_SEED.value),
      workers=int(_WORKERS.value),
  )


if __name__ == "__main__":
  app.run(main)

