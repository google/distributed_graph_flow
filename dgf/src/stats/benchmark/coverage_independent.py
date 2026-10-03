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

"""Coverage benchmark for the independent-sample intervals.

Draws 106 random cells in six blocks (A: continuous losses, B: accuracy,
C: paired R^2, D: equicorrelated clusters with analytic effective_n,
E: finite-pool sampling without replacement, F: invalid inputs) and evaluates
each cell in the `data`, `sketch_np`, `merged` and `sketch_jax` modes. Misses
are counted over all repetitions.
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
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from dgf.src.stats.benchmark import generators
from dgf.src.stats.independent import bound
from dgf.src.stats.independent import interval
from dgf.src.stats.independent import jax_accumulator
from dgf.src.stats.independent import metrics
from dgf.src.stats.independent import sketch

DEFAULT_SEED = 80017972348887  # = 0x48c6a2d147d7

_SEED = flags.DEFINE_integer(
    "seed", DEFAULT_SEED, "Base seed for benchmark."
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


def clopper_pearson_99(k: int, n: int) -> tuple[float, float]:
  """Computes exact two-sided Clopper-Pearson 99% confidence interval."""
  if n <= 0:
    return 0.0, 1.0
  lo = 0.0 if k == 0 else float(scipy.stats.beta.ppf(0.005, k, n - k + 1))
  hi = 1.0 if k == n else float(scipy.stats.beta.ppf(0.995, k + 1, n - k))
  return lo, hi


def binomial_p_value(misses: int, reps: int, alpha: float) -> float:
  """Exact one-sided binomial test p-value: P(Binomial(reps, alpha) >= misses)."""
  if misses <= 0:
    return 1.0
  return float(scipy.stats.binom.sf(misses - 1, reps, alpha))


# ---------------------------------------------------------------------------
# Ground Truth Moments and Distributions
# ---------------------------------------------------------------------------


def student_t_moments(nu: float) -> tuple[float, float, float]:
  """Returns (E|T|, E[T^2], E[T^4]) for Student-t(nu)."""
  e_abs = float(
      2.0
      * math.sqrt(nu)
      * math.gamma((nu + 1.0) / 2.0)
      / (math.sqrt(math.pi) * (nu - 1.0) * math.gamma(nu / 2.0))
  )
  e_sq = float(nu / (nu - 2.0))
  e_four = float(3.0 * (nu**2) / ((nu - 2.0) * (nu - 4.0)))
  return e_abs, e_sq, e_four


def lognormal_moments(sig: float) -> tuple[float, float, float]:
  """Returns (E|Y|, E[Y^2], E[Y^4]) for centred lognormal Y = exp(sig*Z) - exp(sig^2 / 2)."""
  shift = math.exp(sig**2 / 2.0)
  omega = math.exp(sig**2)
  e_abs = float(2.0 * shift * (2.0 * scipy.stats.norm.cdf(sig / 2.0) - 1.0))
  e_sq = float(omega * (omega - 1.0))
  m_mse = float(omega**4 + 2.0 * omega**3 + 3.0 * omega**2 - 3.0)
  e_four = float(m_mse * (e_sq**2))
  return e_abs, e_sq, e_four


def normal_moments(sigma: float) -> tuple[float, float, float]:
  """Returns (E|e|, E[e^2], E[e^4]) for N(0, sigma^2)."""
  return (
      float(sigma * math.sqrt(2.0 / math.pi)),
      float(sigma**2),
      float(3.0 * (sigma**4)),
  )


def laplace_moments(b: float) -> tuple[float, float, float]:
  """Returns (E|e|, E[e^2], E[e^4]) for Laplace(b)."""
  return float(b), float(2.0 * (b**2)), float(24.0 * (b**4))


def uniform_moments(a: float) -> tuple[float, float, float]:
  """Returns (E|e|, E[e^2], E[e^4]) for Uniform(-a, a)."""
  return float(a / 2.0), float((a**2) / 3.0), float((a**4) / 5.0)


def sample_scale_mixture(
    k_comps: int,
    kinds: Sequence[str],
    scales: Sequence[float],
    weights: Sequence[float],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
  """Samples from scale mixture of standard families."""
  comp_choices = rng.choice(k_comps, p=weights, size=size)
  samples = np.empty(size, dtype=np.float64)
  for k in range(k_comps):
    idx = comp_choices == k
    count = int(np.sum(idx))
    if count == 0:
      continue
    kind = kinds[k]
    s = scales[k]
    if kind == "normal":
      samples[idx] = rng.normal(scale=s, size=count)
    elif kind == "laplace":
      samples[idx] = rng.laplace(scale=s, size=count)
    elif kind == "uniform":
      samples[idx] = rng.uniform(-s, s, size=count)
  return samples


def contaminated_gaussian_moments(
    eps: float, s: float
) -> tuple[float, float, float]:
  """Returns (E|e|, E[e^2], E[e^4]) for (1-eps) N(0, 1) + eps N(0, s^2)."""
  root = math.sqrt(2.0 / math.pi)
  return (
      float((1.0 - eps) * root + eps * s * root),
      float((1.0 - eps) + eps * s**2),
      float(3.0 * ((1.0 - eps) + eps * s**4)),
  )


def _component_moments(kind: str, scale: float) -> tuple[float, float, float]:
  if kind == "normal":
    return normal_moments(scale)
  if kind == "laplace":
    return laplace_moments(scale)
  return uniform_moments(scale)


_REGRESSION_FAMILIES = (
    "scale_mixture",
    "student_t",
    "lognormal",
    "contaminated_gaussian",
)
_REGRESSION_FAMILY_WEIGHTS = (0.4, 0.2, 0.2, 0.2)


def draw_regression_family(
    spec_rng: np.random.Generator, cell: dict[str, Any]
) -> tuple[float, float, float]:
  """Draws a block A error family into `cell`; returns (E|e|, E[e^2], E[e^4])."""
  family = str(
      spec_rng.choice(_REGRESSION_FAMILIES, p=_REGRESSION_FAMILY_WEIGHTS)
  )
  cell["family"] = family
  if family == "scale_mixture":
    k_comps = int(spec_rng.choice([1, 2, 3]))
    kinds = [
        str(k)
        for k in spec_rng.choice(["normal", "laplace", "uniform"], size=k_comps)
    ]
    scales = [
        float(np.exp(spec_rng.uniform(np.log(0.1), np.log(10.0))))
        for _ in range(k_comps)
    ]
    weights = [float(w) for w in spec_rng.dirichlet(np.ones(k_comps))]
    cell["kinds"] = kinds
    cell["scales"] = scales
    cell["weights"] = weights
    comp = [_component_moments(k, s) for k, s in zip(kinds, scales)]
    return (
        sum(w * c[0] for w, c in zip(weights, comp)),
        sum(w * c[1] for w, c in zip(weights, comp)),
        sum(w * c[2] for w, c in zip(weights, comp)),
    )
  if family == "student_t":
    nu = float(spec_rng.uniform(4.3, 30.0))
    cell["nu"] = nu
    return student_t_moments(nu)
  if family == "lognormal":
    sig = float(spec_rng.uniform(0.1, 0.9))
    cell["sig"] = sig
    return lognormal_moments(sig)
  eps = float(np.exp(spec_rng.uniform(np.log(1e-3), np.log(0.1))))
  s = float(spec_rng.uniform(2.0, 40.0))
  cell["eps"] = eps
  cell["s"] = s
  return contaminated_gaussian_moments(eps, s)


def sample_regression_family(
    cell: dict[str, Any], size: int, rng: np.random.Generator
) -> np.ndarray:
  """Samples `size` unscaled errors from the family stored in `cell`."""
  fam = cell["family"]
  if fam == "scale_mixture":
    return sample_scale_mixture(
        len(cell["kinds"]),
        cell["kinds"],
        cell["scales"],
        cell["weights"],
        size,
        rng,
    )
  if fam == "student_t":
    return rng.standard_t(df=cell["nu"], size=size)
  if fam == "lognormal":
    z = rng.normal(size=size)
    return np.exp(cell["sig"] * z) - math.exp(cell["sig"] ** 2 / 2.0)
  is_contam = rng.random(size=size) < cell["eps"]
  return np.where(
      is_contam,
      rng.normal(scale=cell["s"], size=size),
      rng.normal(scale=1.0, size=size),
  )


# ---------------------------------------------------------------------------
# Grid and Redraw Rule
# ---------------------------------------------------------------------------

_N_GRID = (100, 300, 1000, 3000, 10000, 30000)
_N_WEIGHTS = (0.1, 0.1, 0.2, 0.2, 0.2, 0.2)

_ALPHA_GRID = (0.1, 0.05, 0.01, 0.001)
_ALPHA_WEIGHTS = (0.2, 0.4, 0.25, 0.15)


def check_refusal(
    n: int,
    m_declared: float,
    alpha: float,
    metric: str,
    effective_n: float | None = None,
) -> bool:
  """Returns True if the configuration refuses deterministically in data mode."""
  if metric == "r2":
    _, st_a = bound.bound(n, 1.0, 1.0, [], alpha / 2.0, m_declared, "mse")
    _, st_b = bound.bound(n // 2, 1.0, 1.0, [], alpha / 2.0, m_declared, "mse")
    return st_a == "ASSUMPTION_REQUIRED" or st_b == "ASSUMPTION_REQUIRED"
  _, st = bound.bound(
      n, 1.0, 1.0, [], alpha, m_declared, metric, effective_n=effective_n
  )
  return st == "ASSUMPTION_REQUIRED"


def draw_cell_spec(
    block_id: str,
    cell_idx: int,
    spec_rng: np.random.Generator,
) -> tuple[dict[str, Any], int]:
  """Draws a cell, redrawing (up to 100 times) cells that always refuse."""
  redraw_count = 0
  for attempt in range(100):
    if block_id == "F":
      n_val = 1000
      alpha_val = 0.05
    else:
      n_val = int(spec_rng.choice(_N_GRID, p=_N_WEIGHTS))
      alpha_val = float(spec_rng.choice(_ALPHA_GRID, p=_ALPHA_WEIGHTS))

    m_mode = str(spec_rng.choice(["default", "tight"]))

    # Draw block-specific parameters
    cell = _draw_block_params(
        block_id, cell_idx, n_val, alpha_val, m_mode, spec_rng
    )

    # Check redraw rule
    if block_id == "F":
      return cell, redraw_count
    refuses = check_refusal(
        cell["n"],
        cell["M_declared"],
        cell["alpha"],
        cell["metric"],
        cell.get("effective_n"),
    )
    if not refuses:
      return cell, redraw_count
    redraw_count += 1

  # If reached 100 attempts, keep last drawn
  return cell, redraw_count


def _draw_block_params(
    block_id: str,
    cell_idx: int,
    n: int,
    alpha: float,
    m_mode: str,
    spec_rng: np.random.Generator,
) -> dict[str, Any]:
  """Draws block-specific distribution parameters and ground-truth values."""
  cell: dict[str, Any] = {
      "block": block_id,
      "cell_idx": cell_idx,
      "n": n,
      "alpha": alpha,
      "m_mode": m_mode,
  }

  if block_id == "A":
    metric = str(spec_rng.choice(["mae", "mse", "rmse"]))
    cell["metric"] = metric
    e1, e2, e4 = draw_regression_family(spec_rng, cell)
    scale = float(np.exp(spec_rng.uniform(np.log(1e-3), np.log(1e3))))
    cell["scale"] = scale

    # Compute M_true and theta
    if metric == "mae":
      m_true = e2 / (e1**2)
      theta = scale * e1
    elif metric == "mse":
      m_true = e4 / (e2**2)
      theta = (scale**2) * e2
    else:  # rmse
      m_true = e4 / (e2**2)
      theta = scale * math.sqrt(e2)

    cell["M_true"] = m_true
    cell["theta"] = theta
    cell["effective_n"] = None
    default_m = metrics.get_default_m(metric)
    cell["M_declared"] = m_true if m_mode == "tight" else default_m
    cell["in_contract"] = bool(m_true <= cell["M_declared"])

  elif block_id == "B":
    # Accuracy
    metric = "accuracy"
    if spec_rng.random() < 0.7:
      p = float(spec_rng.uniform(0.3, 0.995))
    else:
      p = float(np.exp(spec_rng.uniform(np.log(0.005), np.log(0.3))))
    cell["metric"] = metric
    cell["p"] = p
    cell["theta"] = p
    m_true = 1.0 / p
    cell["M_true"] = m_true
    cell["effective_n"] = None
    default_m = 4.0
    cell["M_declared"] = m_true if m_mode == "tight" else default_m
    cell["in_contract"] = bool(m_true <= cell["M_declared"])

  elif block_id == "C":
    # R^2
    metric = "r2"
    y_marginal = str(spec_rng.choice(["gaussian", "laplace"]))
    mu = float(spec_rng.uniform(-5.0, 5.0))
    tau = float(np.exp(spec_rng.uniform(np.log(1e-2), np.log(1e2))))
    c = float(spec_rng.uniform(0.1, 0.9))

    # Scale mixture for residual e_tilde
    k_comps = int(spec_rng.choice([1, 2, 3]))
    kinds = list(
        spec_rng.choice(["normal", "laplace", "uniform"], size=k_comps)
    )
    scales = [
        float(np.exp(spec_rng.uniform(np.log(0.1), np.log(10.0))))
        for _ in range(k_comps)
    ]
    weights = list(spec_rng.dirichlet(np.ones(k_comps)))
    v_raw = sum(
        weights[k]
        * (
            scales[k] ** 2
            if kinds[k] == "normal"
            else (
                2.0 * (scales[k] ** 2)
                if kinds[k] == "laplace"
                else (scales[k] ** 2) / 3.0
            )
        )
        for k in range(k_comps)
    )
    e4_raw = sum(
        weights[k]
        * (
            3.0 * (scales[k] ** 4)
            if kinds[k] == "normal"
            else (
                24.0 * (scales[k] ** 4)
                if kinds[k] == "laplace"
                else (scales[k] ** 4) / 5.0
            )
        )
        for k in range(k_comps)
    )
    m_a = e4_raw / (v_raw**2)
    kappa_y = 3.0 if y_marginal == "gaussian" else 6.0
    m_b = (kappa_y + 3.0) / 2.0
    m_true = max(m_a, m_b)

    cell["metric"] = metric
    cell["y_marginal"] = y_marginal
    cell["mu"] = mu
    cell["tau"] = tau
    cell["c"] = c
    cell["kinds"] = kinds
    cell["scales"] = scales
    cell["weights"] = weights
    cell["v_raw"] = v_raw
    cell["theta"] = 1.0 - (c**2)
    cell["M_true"] = m_true
    cell["effective_n"] = None
    default_m = 16.0
    cell["M_declared"] = m_true if m_mode == "tight" else default_m
    cell["in_contract"] = bool(m_true <= cell["M_declared"])

  elif block_id == "D":
    # Equicorrelated clusters
    metric = str(spec_rng.choice(["mae", "mse", "rmse"]))
    rho = float(spec_rng.uniform(0.05, 0.8))
    m_clust = int(np.exp(spec_rng.uniform(np.log(2.0), np.log(64.0))))
    eff_n = float(generators.analytic_neff_equicorr(n, rho, m_clust, metric))

    if metric == "mae":
      m_true = math.pi / 2.0
      theta = math.sqrt(2.0 / math.pi)
    else:
      m_true = 3.0
      theta = 1.0

    cell["metric"] = metric
    cell["rho"] = rho
    cell["m_clust"] = m_clust
    cell["effective_n"] = eff_n
    cell["theta"] = theta
    cell["M_true"] = m_true
    default_m = metrics.get_default_m(metric)
    cell["M_declared"] = m_true if m_mode == "tight" else default_m
    cell["in_contract"] = bool(m_true <= cell["M_declared"])

  elif block_id == "E":
    # Finite pool
    metric = str(spec_rng.choice(["mae", "mse", "rmse"]))
    n_pool = int(spec_rng.integers(2 * n, 20 * n + 1))
    cell["metric"] = metric
    # The pool is drawn from a block A error family.
    draw_regression_family(spec_rng, cell)
    scale = float(np.exp(spec_rng.uniform(np.log(1e-3), np.log(1e3))))
    cell["scale"] = scale
    pool_errors = sample_regression_family(cell, n_pool, spec_rng) * scale
    if metric == "mae":
      s_pool = np.abs(pool_errors)
    else:
      s_pool = pool_errors**2

    theta_pool = float(np.mean(s_pool))
    m_pool = float(n_pool * np.sum(s_pool**2) / ((np.sum(s_pool)) ** 2))
    theta = math.sqrt(theta_pool) if metric == "rmse" else theta_pool

    cell["metric"] = metric
    cell["n_pool"] = n_pool
    cell["pool_errors"] = pool_errors
    cell["s_pool"] = s_pool
    cell["theta"] = theta
    cell["M_true"] = m_pool
    cell["effective_n"] = float(n)  # Path 1
    default_m = metrics.get_default_m(metric)
    cell["M_declared"] = m_pool if m_mode == "tight" else default_m
    cell["in_contract"] = bool(m_pool <= cell["M_declared"])

  elif block_id == "F":
    # Invalid input cells (indices 0..5)
    f_configs = [
        ("mae", "nan_residual"),
        ("mse", "inf_residual"),
        ("accuracy", "nan_prediction"),
        ("r2", "nan_label"),
        ("raw_summands", "negative_summand"),
        ("accuracy_1d", "indicator_two"),
    ]
    m_name, f_type = f_configs[cell_idx]
    cell["metric"] = m_name
    cell["fault_type"] = f_type
    cell["theta"] = float("nan")
    cell["M_true"] = 16.0
    cell["M_declared"] = 16.0
    cell["effective_n"] = None
    cell["in_contract"] = False

  # Per-cell mode parameters (shards, JAX batching).
  cell["shard_count"] = int(spec_rng.integers(2, 17))
  cell["shard_weights"] = list(spec_rng.dirichlet(np.ones(cell["shard_count"])))
  cell["jax_batch_size"] = int(
      round(np.exp(spec_rng.uniform(np.log(7.0), np.log(4096.0))))
  )
  cell["jax_pad_frac"] = float(spec_rng.uniform(0.0, 0.3))
  cell["jax_d"] = int(spec_rng.choice([1, 2, 4, 8]))

  return cell


# ---------------------------------------------------------------------------
# Data Generation and Repetition Evaluation
# ---------------------------------------------------------------------------


def generate_cell_data(
    cell: dict[str, Any],
    r: int,
    base_seed: int,
) -> tuple[Any, Any]:
  """Generates repetition data using default_rng([base_seed, cell_id, r])."""
  data_seed = [base_seed, cell["cell_id"], r]
  rng = np.random.default_rng(data_seed)
  n = cell["n"]
  b_id = cell["block"]

  if b_id == "A":
    e = sample_regression_family(cell, n, rng) * cell["scale"]
    return e, None

  elif b_id == "B":
    p = cell["p"]
    labels = rng.integers(0, 10, size=n)
    is_correct = rng.random(size=n) < p
    wrong_offset = rng.integers(1, 10, size=n)
    preds = np.where(is_correct, labels, (labels + wrong_offset) % 10)
    return (labels, preds), None

  elif b_id == "C":
    mu = cell["mu"]
    tau = cell["tau"]
    c = cell["c"]
    if cell["y_marginal"] == "gaussian":
      y_tilde = rng.normal(size=n)
    else:
      y_tilde = rng.laplace(scale=1.0 / math.sqrt(2.0), size=n)
    y = mu + tau * y_tilde
    e_raw = sample_scale_mixture(
        len(cell["kinds"]),
        cell["kinds"],
        cell["scales"],
        cell["weights"],
        n,
        rng,
    )
    e_tilde = e_raw / math.sqrt(cell["v_raw"])
    y_pred = y + tau * c * e_tilde
    return (y, y_pred), None

  elif b_id == "D":
    rho = cell["rho"]
    m_c = cell["m_clust"]
    e = generators.sample_equicorr(n, rho, m_c, rng)
    return e, None

  elif b_id == "E":
    pool = cell["pool_errors"]
    idx = rng.choice(len(pool), size=n, replace=False)
    return pool[idx], None

  elif b_id == "F":
    # Fault injection at random position
    fault_pos = int(rng.integers(0, n))
    f_type = cell["fault_type"]
    if f_type == "nan_residual":
      e = rng.normal(size=n)
      e[fault_pos] = np.nan
      return e, None
    elif f_type == "inf_residual":
      e = rng.normal(size=n)
      e[fault_pos] = np.inf
      return e, None
    elif f_type == "nan_prediction":
      labels = rng.integers(0, 10, size=n).astype(np.float64)
      preds = labels.copy()
      preds[fault_pos] = np.nan
      return (labels, preds), None
    elif f_type == "nan_label":
      y = np.linspace(1.0, 10.0, n)
      y[fault_pos] = np.nan
      y_pred = y + 0.1
      return (y, y_pred), None
    elif f_type == "negative_summand":
      s = rng.uniform(0.1, 5.0, size=n)
      s[fault_pos] = -1.0
      return s, None
    elif f_type == "indicator_two":
      s = (rng.random(size=n) < 0.8).astype(np.float64)
      s[fault_pos] = 2.0
      return s, None

  return None, None


# ---------------------------------------------------------------------------
# Execution of Single Draw across Modes
# ---------------------------------------------------------------------------


def _normalize_eval_metric(m: str) -> str:
  if m == "raw_summands":
    return "mse"
  if m == "accuracy_1d":
    return "accuracy"
  return m


def run_draw_numpy(
    data: Any,
    cell: dict[str, Any],
    mode: str,
) -> tuple[float, float, str]:
  """Runs confidence_interval for numpy modes ('data', 'sketch_np', 'merged')."""
  metric = cell["metric"]
  level = 1.0 - cell["alpha"]
  m = cell["M_declared"]
  eff_n = cell.get("effective_n")
  norm_m = _normalize_eval_metric(metric)

  try:
    if mode == "data":
      res = interval.confidence_interval(
          data, metric=norm_m, level=level, m=m, effective_n=eff_n
      )
      return res.low, res.high, res.status.name

    elif mode == "sketch_np":
      if metric == "raw_summands":
        sk = sketch.from_array(data, effective_n=eff_n)
        res = interval.confidence_interval(sk, metric="mse", level=level, m=m)
      else:
        sk = sketch.from_data(data, metric=norm_m, effective_n=eff_n)
        res = interval.confidence_interval(sk, metric=norm_m, level=level, m=m)
      return res.low, res.high, res.status.name

    elif mode == "merged":
      s_count = cell["shard_count"]
      n = cell["n"]
      b_id = cell["block"]

      # Shard boundaries
      if b_id == "D":
        # Shard boundaries on cluster boundaries, partial cluster in last shard
        m_c = cell["m_clust"]
        num_clusters = n // m_c
        s_eff = min(s_count, max(1, num_clusters))
        c_per_shard = [
            num_clusters // s_eff + (1 if j < num_clusters % s_eff else 0)
            for j in range(s_eff)
        ]
        cuts = [0]
        cur = 0
        for j in range(s_eff - 1):
          cur += c_per_shard[j] * m_c
          cuts.append(cur)
        cuts.append(n)
      else:
        weights = np.array(cell["shard_weights"], dtype=np.float64)
        rem = n - 2 * s_count
        add = np.floor(weights * rem).astype(int)
        sizes = 2 + add
        diff = n - int(np.sum(sizes))
        frac = (weights * rem) - add
        order = np.argsort(-frac)
        for idx in range(diff):
          sizes[order[idx]] += 1
        cuts = [0] + list(np.cumsum(sizes))

      # Build and merge shards
      merged_sk = None
      num_shards = len(cuts) - 1
      for i in range(num_shards):
        start, end = cuts[i], cuts[i + 1]
        if end <= start:
          continue
        if isinstance(data, tuple):
          shard_data = (data[0][start:end], data[1][start:end])
        else:
          shard_data = data[start:end]

        shard_eff_n = None
        if eff_n is not None:
          if b_id == "D":
            shard_eff_n = generators.analytic_neff_equicorr(
                end - start, cell["rho"], cell["m_clust"], metric
            )
          elif b_id == "E":
            shard_eff_n = float(end - start)

        if metric == "raw_summands":
          sk_part = sketch.from_array(shard_data, effective_n=shard_eff_n)
        else:
          sk_part = sketch.from_data(
              shard_data, metric=norm_m, effective_n=shard_eff_n
          )
        merged_sk = (
            sk_part if merged_sk is None else sketch.merge(merged_sk, sk_part)
        )

      res = interval.confidence_interval(
          merged_sk, metric=norm_m, level=level, m=m
      )
      return res.low, res.high, res.status.name
  except ValueError:
    return float("nan"), float("nan"), "RAISED_VALUE_ERROR"

  return float("nan"), float("nan"), "ASSUMPTION_REQUIRED"


def _scan_step(carry: Any, elem: tuple[jax.Array, jax.Array]):
  s_batch, mask_batch = elem
  return jax_accumulator.update(carry, s_batch, mask=mask_batch), None


def _device_scan(
    s_batches: jax.Array, mask_batches: jax.Array
) -> Any:
  init_st = jax_accumulator.init(dtype=jnp.float32)
  final_st, _ = lax.scan(_scan_step, init_st, (s_batches, mask_batches))
  return final_st


@jax.jit
def _run_single_sequence_accumulate(
    s_d: jax.Array, mask_d: jax.Array
) -> Any:
  dev_states = jax.vmap(_device_scan)(s_d, mask_d)
  all_reduced_states = jax.vmap(
      lambda s: jax_accumulator.all_reduce(s, axis_name="d"),
      axis_name="d",
  )(dev_states)
  return jax.tree.map(lambda x: x[0], all_reduced_states)


def _scan_step_r2(
    carry: Any, elem: tuple[jax.Array, jax.Array, jax.Array]
):
  y_batch, pred_batch, mask_batch = elem
  return (
      jax_accumulator.update_r2(carry, y_batch, pred_batch, mask=mask_batch),
      None,
  )


def _device_scan_r2(
    y_batches: jax.Array, pred_batches: jax.Array, mask_batches: jax.Array
) -> Any:
  init_st = jax_accumulator.init_r2(dtype=jnp.float32)
  final_st, _ = lax.scan(
      _scan_step_r2, init_st, (y_batches, pred_batches, mask_batches)
  )
  return final_st


@jax.jit
def _run_r2_accumulate(
    y_d: jax.Array, pred_d: jax.Array, mask_d: jax.Array
) -> Any:
  dev_states = jax.vmap(_device_scan_r2)(y_d, pred_d, mask_d)
  all_reduced_states = jax.vmap(
      lambda s: jax_accumulator.all_reduce_r2(s, axis_name="d"),
      axis_name="d",
  )(dev_states)
  return jax.tree.map(lambda x: x[0], all_reduced_states)


def run_draw_jax(
    data: Any,
    cell: dict[str, Any],
    r: int,
    base_seed: int,
) -> tuple[float, float, str]:
  """Runs sketch_jax mode in the main process with mask and padding."""
  metric = cell["metric"]
  level = 1.0 - cell["alpha"]
  m = cell["M_declared"]
  eff_n = cell.get("effective_n")
  n = cell["n"]

  batch = cell["jax_batch_size"]
  d_count = cell["jax_d"]
  f_pad = cell["jax_pad_frac"]

  # Slot count = n / (1 - f), rounded up to multiple of batch * D
  raw_slots = int(math.ceil(n / max(1e-6, 1.0 - f_pad)))
  block_size = batch * d_count
  num_blocks = int(math.ceil(raw_slots / float(block_size)))
  total_slots = num_blocks * block_size

  # JAX padding layout seed: [base_seed, cell_id, r, 1]
  pad_rng = np.random.default_rng([base_seed, cell["cell_id"], r, 1])
  chosen_slots = np.sort(pad_rng.choice(total_slots, size=n, replace=False))

  # Garbage cycle: {NaN, +inf, -inf, -1.0, 1e38}
  garbage_cycle = [np.nan, np.inf, -np.inf, -1.0, 1e38]
  garbage_full = np.array(
      [garbage_cycle[i % len(garbage_cycle)] for i in range(total_slots)],
      dtype=np.float32,
  )

  mask = np.zeros(total_slots, dtype=bool)
  mask[chosen_slots] = True

  if metric == "r2":
    y_full = garbage_full.copy()
    pred_full = garbage_full.copy()
    y_full[chosen_slots] = np.asarray(data[0], dtype=np.float32)
    pred_full[chosen_slots] = np.asarray(data[1], dtype=np.float32)

    # Reshape to (D, num_batches, batch)
    num_batches_per_dev = num_blocks
    y_d = y_full.reshape(d_count, num_batches_per_dev, batch)
    pred_d = pred_full.reshape(d_count, num_batches_per_dev, batch)
    mask_d = mask.reshape(d_count, num_batches_per_dev, batch)

    st_r2_merged = _run_r2_accumulate(
        jnp.asarray(y_d), jnp.asarray(pred_d), jnp.asarray(mask_d)
    )
    r2_sk = jax_accumulator.to_r2_sketch(st_r2_merged, effective_n=eff_n)
    res = interval.confidence_interval(r2_sk, metric="r2", level=level, m=m)
    return res.low, res.high, res.status.name

  else:
    # Single-sequence metrics
    if metric == "raw_summands":
      s_valid = np.asarray(data, dtype=np.float32)
    elif metric == "accuracy":
      if isinstance(data, tuple):
        labels_valid = np.asarray(data[0], dtype=np.float32)
        preds_valid = np.asarray(data[1], dtype=np.float32)
        s_valid = np.asarray(
            jax_accumulator.summands(
                "accuracy", jnp.asarray(labels_valid), jnp.asarray(preds_valid)
            )
        )
      else:
        s_valid = np.asarray(data, dtype=np.float32)
    else:
      e_valid = np.asarray(data, dtype=np.float32)
      s_valid = np.asarray(
          jax_accumulator.summands(
              metric, jnp.zeros_like(e_valid), jnp.asarray(e_valid)
          )
      )

    s_full = garbage_full.copy()
    s_full[chosen_slots] = s_valid

    num_batches_per_dev = num_blocks
    s_d = s_full.reshape(d_count, num_batches_per_dev, batch)
    mask_d = mask.reshape(d_count, num_batches_per_dev, batch)

    st_merged = _run_single_sequence_accumulate(
        jnp.asarray(s_d), jnp.asarray(mask_d)
    )
    sk = jax_accumulator.to_sketch(st_merged, effective_n=eff_n)
    ci_metric = _normalize_eval_metric(metric)
    try:
      res = interval.confidence_interval(sk, metric=ci_metric, level=level, m=m)
      return res.low, res.high, res.status.name
    except ValueError:
      return float("nan"), float("nan"), "RAISED_VALUE_ERROR"


# ---------------------------------------------------------------------------
# Cell Evaluation Driver
# ---------------------------------------------------------------------------


def evaluate_cell_reps(
    cell: dict[str, Any],
    reps: int,
    base_seed: int,
) -> list[dict[str, Any]]:
  """Evaluates all repetitions and modes for a single cell."""
  b_id = cell["block"]
  metric = cell["metric"]
  theta = cell["theta"]
  alpha = cell["alpha"]

  # Determine applicable modes
  modes: list[str] = []
  if b_id in ("A", "B", "C", "D"):
    modes = ["data", "sketch_np", "merged", "sketch_jax"]
  elif b_id == "E":
    # Finite pool: pass/fail on effective_n = n, report-only on effective_n = None
    modes = [
        "data",
        "sketch_np",
        "merged",
        "sketch_jax",
        "data_none",
        "sketch_np_none",
        "merged_none",
        "sketch_jax_none",
    ]
  elif b_id == "F":
    f_type = cell["fault_type"]
    if f_type in (
        "nan_residual",
        "inf_residual",
        "nan_prediction",
        "nan_label",
    ):
      modes = ["data", "sketch_np", "merged", "sketch_jax"]
    elif f_type == "negative_summand":
      modes = ["sketch_np", "sketch_jax"]
    elif f_type == "indicator_two":
      modes = ["data"]
    else:
      modes = ["data"]

  # Storage per mode
  mode_stats = {
      m: {
          "returned": 0,
          "misses": 0,
          "statuses": [],
          "lows": [],
          "highs": [],
      }
      for m in modes
  }

  for r in range(reps):
    data, _ = generate_cell_data(cell, r, base_seed)

    # 1. Run numpy modes
    for m in modes:
      if m.startswith("sketch_jax"):
        continue

      # Handle Block E report-only variants
      cur_cell = cell
      if m.endswith("_none"):
        base_m = m.replace("_none", "")
        cur_cell = dict(cell)
        cur_cell["effective_n"] = None
      else:
        base_m = m

      low, high, st_name = run_draw_numpy(data, cur_cell, base_m)
      is_ret = st_name in ("UNREFUTED", "TAIL_UNRESOLVED")
      is_miss = is_ret and not (low <= theta <= high)

      mode_stats[m]["statuses"].append(st_name)
      mode_stats[m]["lows"].append(low)
      mode_stats[m]["highs"].append(high)
      if is_ret:
        mode_stats[m]["returned"] += 1
      if is_miss:
        mode_stats[m]["misses"] += 1

    # 2. Run JAX modes
    for m in modes:
      if not m.startswith("sketch_jax"):
        continue
      cur_cell = cell
      if m.endswith("_none"):
        cur_cell = dict(cell)
        cur_cell["effective_n"] = None

      low_j, high_j, st_j = run_draw_jax(data, cur_cell, r, base_seed)
      is_ret = st_j in ("UNREFUTED", "TAIL_UNRESOLVED")
      is_miss = is_ret and not (low_j <= theta <= high_j)

      mode_stats[m]["statuses"].append(st_j)
      mode_stats[m]["lows"].append(low_j)
      mode_stats[m]["highs"].append(high_j)
      if is_ret:
        mode_stats[m]["returned"] += 1
      if is_miss:
        mode_stats[m]["misses"] += 1

  # Compile rows
  rows = []
  for m in modes:
    ret_cnt = mode_stats[m]["returned"]
    miss_cnt = mode_stats[m]["misses"]
    miss_rate = float(miss_cnt / reps)
    p_val = binomial_p_value(miss_cnt, reps, alpha)
    lo, hi = clopper_pearson_99(miss_cnt, reps)

    # In-contract check
    in_contract = bool(cell["in_contract"])
    if b_id == "E" and m.endswith("_none"):
      in_contract = False  # Report only

    # Status counts and joint unrefuted misses
    statuses = mode_stats[m]["statuses"]
    lows = mode_stats[m]["lows"]
    highs = mode_stats[m]["highs"]
    unref_miss = sum(
        1
        for st, l, h in zip(statuses, lows, highs)
        if st == "UNREFUTED" and not (l <= theta <= h)
    )
    assump_cnt = sum(1 for st in statuses if st == "ASSUMPTION_REQUIRED")
    tail_cnt = sum(1 for st in statuses if st == "TAIL_UNRESOLVED")
    unref_cnt = sum(1 for st in statuses if st == "UNREFUTED")

    # Agreement between sketch_jax (float32) and sketch_np (float64).
    g2_agreement = None
    g2_max_delta_ratio = None
    if m == "sketch_jax" and b_id in ("A", "B", "D", "E"):
      st_j = mode_stats["sketch_jax"]["statuses"]
      st_np = mode_stats["sketch_np"]["statuses"]
      same_st = sum(1 for a, b in zip(st_j, st_np) if a == b)
      g2_agreement = float(same_st / reps)

      lo_j = mode_stats["sketch_jax"]["lows"]
      hi_j = mode_stats["sketch_jax"]["highs"]
      lo_np = mode_stats["sketch_np"]["lows"]
      hi_np = mode_stats["sketch_np"]["highs"]

      delta_ratios = []
      for lj, hj, lnp, hnp, sj, snp in zip(
          lo_j, hi_j, lo_np, hi_np, st_j, st_np
      ):
        if sj in ("UNREFUTED", "TAIL_UNRESOLVED") and snp in (
            "UNREFUTED",
            "TAIL_UNRESOLVED",
        ):
          w = hnp - lnp
          if w > 0 and math.isfinite(w):
            max_d = max(abs(lj - lnp), abs(hj - hnp))
            delta_ratios.append(max_d / w)
      g2_max_delta_ratio = float(max(delta_ratios)) if delta_ratios else 0.0

    row = {
        "block": b_id,
        "cell_idx": cell["cell_idx"],
        "cell_id": cell["cell_id"],
        "metric": metric,
        "n": cell["n"],
        "alpha": alpha,
        "m_mode": cell["m_mode"],
        "M_declared": cell["M_declared"],
        "M_true": cell["M_true"],
        "theta": theta,
        "in_contract": in_contract,
        "effective_n": cell.get("effective_n"),
        "mode": m,
        "reps": reps,
        "returned": ret_cnt,
        "misses": miss_cnt,
        "unrefuted_miss": unref_miss,
        "assumption_required": assump_cnt,
        "tail_unresolved": tail_cnt,
        "unrefuted_count": unref_cnt,
        "miss_rate": miss_rate,
        "p_value": p_val,
        "cp99_low": lo,
        "cp99_high": hi,
        "g2_status_agreement": g2_agreement,
        "g2_max_delta_ratio": g2_max_delta_ratio,
        # Full cell specification: family and mode parameters. The pool
        # arrays of block E are omitted; they are reproducible from the seed.
        "spec": {
            k: v for k, v in cell.items() if k not in ("pool_errors", "s_pool")
        },
    }
    rows.append(row)

  return rows


def _evaluate_independent_cell_worker(
    args: tuple[dict[str, Any], int, int]
) -> list[dict[str, Any]]:
  cell, reps, base_seed = args
  return evaluate_cell_reps(cell, reps, base_seed)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  seed = int(_SEED.value)
  mode = str(_MODE.value)
  t_start = time.time()

  # Draw cell specifications in fixed order: A (60), B (12), C (8), D (12), E (8), F (6)
  spec_rng = np.random.default_rng(seed)
  blocks = [
      ("A", 60),
      ("B", 12),
      ("C", 8),
      ("D", 12),
      ("E", 8),
      ("F", 6),
  ]

  all_cells: list[dict[str, Any]] = []
  redraw_stats: dict[str, int] = {}
  for b_id, count in blocks:
    redraw_total = 0
    for idx in range(count):
      cell, redraws = draw_cell_spec(b_id, idx, spec_rng)
      cell["cell_id"] = len(all_cells)  # Global 0-based index.
      all_cells.append(cell)
      redraw_total += redraws
    redraw_stats[b_id] = redraw_total

  if mode == "quick":
    # 1 cell per block: A, B, C, D, E, F (6 cells total)
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
      f"=== Coverage Independent Benchmark (mode={mode}, seed={seed},"
      f" workers={workers}, cells={len(cells)}) ===",
      flush=True,
  )

  tasks = [
      (c, reps_f if c["block"] == "F" else reps_normal, seed)
      for c in cells
  ]
  all_rows: list[dict[str, Any]] = []

  if workers > 1:
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
      for res_rows in executor.map(_evaluate_independent_cell_worker, tasks):
        all_rows.extend(res_rows)
        c_info = res_rows[0]
        print(
            f"  Completed Block {c_info['block']} cell {c_info['cell_idx']} ({c_info['metric']})",
            flush=True,
        )
  else:
    for task in tasks:
      res_rows = _evaluate_independent_cell_worker(task)
      all_rows.extend(res_rows)
      c_info = res_rows[0]
      print(
          f"  Completed Block {c_info['block']} cell {c_info['cell_idx']} ({c_info['metric']})",
          flush=True,
      )

  total_elapsed = time.time() - t_start

  # Summary table: pick representative mode ('data' or 'sketch_np') per cell
  seen_cell_keys = set()
  display_rows = []
  for r in all_rows:
    key = (r["block"], r["cell_idx"])
    if key not in seen_cell_keys and r["mode"] in ("data", "sketch_np"):
      seen_cell_keys.add(key)
      display_rows.append(r)
  if not display_rows:
    display_rows = all_rows

  print("\n" + "=" * 125, flush=True)
  print(
      f"SUMMARY RESULTS TABLE (mode={mode}, total elapsed: {total_elapsed:.1f}s)",
      flush=True,
  )
  print("=" * 125, flush=True)
  print(
      f"{'Cell Idx':<10} {'Block':<6} {'Metric':<8} {'Reps':<6} {'Issued':<8}"
      f" {'Miss':<18} {'Unref. miss':<18} {'AssumpReq':<10} {'TailUnres':<10}",
      flush=True,
  )
  print("-" * 125, flush=True)
  for r in display_rows:
    reps = r.get("reps", 0)
    issued = r.get("returned", 0)
    misses = r.get("misses", 0)
    unref_miss = r.get("unrefuted_miss", 0)
    assump = r.get("assumption_required", 0)
    tail = r.get("tail_unresolved", 0)
    # Both rates use all repetitions as the denominator.
    miss_rate = (misses / reps) if reps > 0 else 0.0
    joint_rate = (unref_miss / reps) if reps > 0 else 0.0
    miss_str = f"{miss_rate:.2%} ({misses}/{reps})"
    joint_str = f"{joint_rate:.2%} ({unref_miss}/{reps})"
    cell_label = f"[{r['block']}_{r['cell_idx']}]"
    print(
        f"{cell_label:<10} {r['block']:<6} {r['metric']:<8} {reps:<6d} {issued:<8d}"
        f" {miss_str:<18} {joint_str:<18} {assump:<10d} {tail:<10d}",
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
