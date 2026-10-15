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

"""Baseline confidence interval estimators for benchmarking comparison."""

import math
import numpy as np
import scipy.special
import scipy.stats


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
  """Exact Clopper-Pearson interval for binomial success probability."""
  if n <= 0:
    return 0.0, 1.0
  alpha = 1.0 - confidence
  low = 0.0 if k == 0 else float(scipy.stats.beta.ppf(alpha / 2.0, k, n - k + 1))
  high = 1.0 if k == n else float(scipy.stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
  return low, high


def student_t(summands: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
  """Standard Student-t confidence interval for the mean."""
  n = len(summands)
  if n < 2:
    return float("-inf"), float("inf")
  mean = float(np.mean(summands))
  se = float(scipy.stats.sem(summands))
  if se == 0.0 or not math.isfinite(se):
    return mean, mean
  alpha = 1.0 - confidence
  crit = float(scipy.stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
  return mean - crit * se, mean + crit * se


def bca_bootstrap(
    summands: np.ndarray,
    confidence: float = 0.95,
    num_resamples: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
  """Bias-corrected and accelerated (BCa) percentile bootstrap for the mean."""
  n = len(summands)
  if n < 2:
    return float("-inf"), float("inf")
  if rng is None:
    rng = np.random.default_rng(2026)

  center = float(np.mean(summands))
  # Draw bootstrap resamples
  indices = rng.integers(0, n, size=(num_resamples, n))
  resamples = summands[indices]
  boot_means = np.mean(resamples, axis=1)

  alpha = 1.0 - confidence
  # Bias correction z0
  below = float(np.mean(boot_means < center))
  if below <= 0.0 or below >= 1.0:
    low, high = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)
  z0 = float(scipy.special.ndtri(below))

  # Acceleration a from jackknife influence values
  # For the sample mean, the leave-one-out mean is (n*center - x_i)/(n-1)
  # Influence u_i = (n-1) * (center - mean_{-i}) = x_i - center
  influence = summands - center
  sum_sq = float(np.sum(influence ** 2))
  if sum_sq <= 0.0:
    acc = 0.0
  else:
    acc = float(np.sum(influence ** 3) / (6.0 * (sum_sq ** 1.5)))

  def get_adjusted_prob(prob: float) -> float:
    z = scipy.special.ndtri(prob)
    denom = 1.0 - acc * (z0 + z)
    if denom <= 0.0:
      return 1.0 if (z0 + z) > 0 else 0.0
    return float(scipy.special.ndtr(z0 + (z0 + z) / denom))

  a1 = get_adjusted_prob(alpha / 2.0)
  a2 = get_adjusted_prob(1.0 - alpha / 2.0)
  a1 = np.clip(a1, 0.0, 1.0)
  a2 = np.clip(a2, 0.0, 1.0)
  low = float(np.quantile(boot_means, a1))
  high = float(np.quantile(boot_means, a2))
  return low, high
