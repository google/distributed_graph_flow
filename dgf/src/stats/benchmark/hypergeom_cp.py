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

"""Exact numerical check of Clopper-Pearson coverage under sampling without replacement.

Computes exact hypergeometric miss probabilities for Clopper-Pearson accuracy
confidence intervals across a grid of pool sizes N, sample sizes n, defect
counts D, and significance levels alpha.
"""

from collections.abc import Sequence
import json
import math
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
import scipy.stats

from dgf.src.stats.independent import interval

_MODE = flags.DEFINE_enum(
    "mode",
    "quick",
    ["quick", "full"],
    "Execution mode: 'quick' or 'full'.",
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Output path for JSONL results (optional).",
)


def get_sample_sizes_for_pool(pool_n: int) -> list[int]:
  """Returns the list of valid sample sizes n for a given pool size N."""
  standard_n = [1, 2, 5, 10, 20, 50, 100, 200, 500]
  candidates = set([n for n in standard_n if 1 <= n <= pool_n])
  if pool_n - 1 >= 1:
    candidates.add(pool_n - 1)
  candidates.add(pool_n)
  return sorted(candidates)


def compute_interval_cache(
    n: int, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
  """Caches Clopper-Pearson intervals for all k in 0..n at level 1 - alpha."""
  lows = np.zeros(n + 1, dtype=np.float64)
  highs = np.zeros(n + 1, dtype=np.float64)
  level = 1.0 - alpha

  for k in range(n + 1):
    arr = np.zeros(n, dtype=np.float64)
    if k > 0:
      arr[:k] = 1.0
    res = interval.confidence_interval(arr, metric="accuracy", level=level)
    if res.status is not interval.Status.UNREFUTED:
      raise RuntimeError(
          f"confidence_interval returned {res.status} for n={n}, k={k},"
          f" alpha={alpha}; expected UNREFUTED."
      )
    lows[k] = res.low
    highs[k] = res.high

  return lows, highs


def evaluate_grid_cell(
    pool_n: int,
    sample_n: int,
    alpha: float,
    lows: np.ndarray,
    highs: np.ndarray,
) -> dict[str, Any]:
  """Evaluates exact hypergeometric and binomial miss probabilities over all D in 0..N."""
  alpha_half = alpha / 2.0
  k_arr = np.arange(sample_n + 1)

  max_hyper_lower_ratio = -1.0
  max_hyper_upper_ratio = -1.0
  d_max_lower = 0
  d_max_upper = 0

  max_binom_lower_ratio = -1.0
  max_binom_upper_ratio = -1.0
  d_binom_max_lower = 0
  d_binom_max_upper = 0

  num_d_exceeding = 0
  exceeding_details = []

  for d in range(pool_n + 1):
    theta = float(d) / float(pool_n)

    # Indicator of misses
    # Lower miss: theta < low
    lower_miss_mask = (theta < lows).astype(np.float64)
    # Upper miss: theta > high
    upper_miss_mask = (theta > highs).astype(np.float64)

    # Hypergeometric exact PMF
    p_k_hyper = scipy.stats.hypergeom.pmf(k_arr, pool_n, d, sample_n)

    # For n == pool_n, sample is the entire pool so k == d exactly; miss is 0
    if sample_n == pool_n:
      lower_miss_hyper = 0.0
      upper_miss_hyper = 0.0
    else:
      lower_miss_hyper = float(np.sum(p_k_hyper * lower_miss_mask))
      upper_miss_hyper = float(np.sum(p_k_hyper * upper_miss_mask))

    # Binomial reference PMF
    if theta == 0.0:
      p_k_binom = np.zeros(sample_n + 1, dtype=np.float64)
      p_k_binom[0] = 1.0
    elif theta == 1.0:
      p_k_binom = np.zeros(sample_n + 1, dtype=np.float64)
      p_k_binom[-1] = 1.0
    else:
      p_k_binom = scipy.stats.binom.pmf(k_arr, sample_n, theta)

    lower_miss_binom = float(np.sum(p_k_binom * lower_miss_mask))
    upper_miss_binom = float(np.sum(p_k_binom * upper_miss_mask))

    hyper_lower_ratio = lower_miss_hyper / alpha_half
    hyper_upper_ratio = upper_miss_hyper / alpha_half

    binom_lower_ratio = lower_miss_binom / alpha_half
    binom_upper_ratio = upper_miss_binom / alpha_half

    if hyper_lower_ratio > max_hyper_lower_ratio:
      max_hyper_lower_ratio = hyper_lower_ratio
      d_max_lower = d
    if hyper_upper_ratio > max_hyper_upper_ratio:
      max_hyper_upper_ratio = hyper_upper_ratio
      d_max_upper = d

    if binom_lower_ratio > max_binom_lower_ratio:
      max_binom_lower_ratio = binom_lower_ratio
      d_binom_max_lower = d
    if binom_upper_ratio > max_binom_upper_ratio:
      max_binom_upper_ratio = binom_upper_ratio
      d_binom_max_upper = d

    # Check if hypergeometric miss exceeds alpha/2 * (1 + 1e-9)
    threshold = alpha_half * (1.0 + 1e-9)
    if lower_miss_hyper > threshold or upper_miss_hyper > threshold:
      num_d_exceeding += 1
      exceeding_details.append({
          "D": d,
          "theta": theta,
          "lower_miss": lower_miss_hyper,
          "upper_miss": upper_miss_hyper,
          "max_ratio": max(hyper_lower_ratio, hyper_upper_ratio),
      })

  # Invariants: binomial ratios never exceed 1; no misses when n == N.
  if max_binom_lower_ratio > 1.0 + 1e-9:
    raise AssertionError(
        f"Binomial lower miss ratio {max_binom_lower_ratio} > 1 + 1e-9 at"
        f" N={pool_n}, n={sample_n}, alpha={alpha}, D={d_binom_max_lower}"
    )
  if max_binom_upper_ratio > 1.0 + 1e-9:
    raise AssertionError(
        f"Binomial upper miss ratio {max_binom_upper_ratio} > 1 + 1e-9 at"
        f" N={pool_n}, n={sample_n}, alpha={alpha}, D={d_binom_max_upper}"
    )
  if sample_n == pool_n:
    if max_hyper_lower_ratio > 0.0 or max_hyper_upper_ratio > 0.0:
      raise AssertionError(
          f"Hypergeometric miss non-zero at n == N={pool_n}:"
          f" lower={max_hyper_lower_ratio}, upper={max_hyper_upper_ratio}"
      )

  max_hyper_ratio = max(max_hyper_lower_ratio, max_hyper_upper_ratio)

  record = {
      "N": pool_n,
      "n": sample_n,
      "alpha": alpha,
      "max_hyper_ratio": max_hyper_ratio,
      "max_hyper_lower_ratio": max_hyper_lower_ratio,
      "d_max_lower": d_max_lower,
      "max_hyper_upper_ratio": max_hyper_upper_ratio,
      "d_max_upper": d_max_upper,
      "binom_max_lower_ratio": max_binom_lower_ratio,
      "d_binom_max_lower": d_binom_max_lower,
      "binom_max_upper_ratio": max_binom_upper_ratio,
      "d_binom_max_upper": d_binom_max_upper,
      "num_d_exceeding": num_d_exceeding,
      "exceeding_details": exceeding_details,
  }
  return record


def run_full_grid(
    pools: Sequence[int] | None = None,
    alphas: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
  """Runs the evaluation grid over pool sizes, sample sizes, and alpha levels."""
  if pools is None:
    pools = [10, 20, 50, 100, 200, 500, 1000]
  if alphas is None:
    alphas = [0.1, 0.05, 0.01, 0.001]

  records = []
  for pool_n in pools:
    sample_sizes = get_sample_sizes_for_pool(pool_n)
    for sample_n in sample_sizes:
      for alpha in alphas:
        lows, highs = compute_interval_cache(sample_n, alpha)
        rec = evaluate_grid_cell(pool_n, sample_n, alpha, lows, highs)
        records.append(rec)

  return records


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  output_path = _OUTPUT.value

  logging.info("Starting hypergeometric Clopper-Pearson evaluation (mode=%s)...", _MODE.value)
  if _MODE.value == "quick":
    records = run_full_grid(pools=[50], alphas=[0.05])
  else:
    records = run_full_grid()

  # Write JSONL if requested
  if output_path:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
      for r in records:
        f.write(json.dumps(r) + "\n")
    logging.info("Wrote %d records to %s.", len(records), output_path)

  # Sort by larger of the two hypergeometric ratios, descending
  sorted_records = sorted(records, key=lambda x: x["max_hyper_ratio"], reverse=True)

  header = (
      f"{'N':<6} {'n':<6} {'alpha':<8} {'Max Hyper Ratio':<18} "
      f"{'Lower Ratio (D)':<18} {'Upper Ratio (D)':<18} "
      f"{'Binom Max (L, U)':<20} {'Num D > alpha/2':<15}"
  )
  print("\n" + "=" * len(header))
  print("Hypergeometric Sampling Without Replacement vs Binomial Reference")
  print("=" * len(header))
  print(header)
  print("-" * len(header))
  for r in sorted_records[:25]:
    l_str = f"{r['max_hyper_lower_ratio']:.4f} ({r['d_max_lower']})"
    u_str = f"{r['max_hyper_upper_ratio']:.4f} ({r['d_max_upper']})"
    b_str = f"{r['binom_max_lower_ratio']:.4f}, {r['binom_max_upper_ratio']:.4f}"
    print(
        f"{r['N']:<6} {r['n']:<6} {r['alpha']:<8} {r['max_hyper_ratio']:<18.4f} "
        f"{l_str:<18} {u_str:<18} {b_str:<20} {r['num_d_exceeding']:<15}"
    )
  print("=" * len(header) + "\n")


if __name__ == "__main__":
  app.run(main)
