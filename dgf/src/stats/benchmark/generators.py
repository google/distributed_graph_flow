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

"""Synthetic and distribution-based data generators for benchmarking."""

import math
from typing import Any
import numpy as np

from dgf.src.stats.benchmark import distributions


def r_mae_factor(rho: float) -> float:
  """Correlation of |X|, |Y| for standard bivariate normal with correlation rho."""
  if abs(rho) < 1e-12:
    return 0.0
  if abs(rho - 1.0) < 1e-12:
    return 1.0
  num = (2.0 / math.pi) * (math.sqrt(max(0.0, 1.0 - rho ** 2)) + rho * math.asin(np.clip(rho, -1.0, 1.0)) - 1.0)
  den = 1.0 - 2.0 / math.pi
  return float(num / den)


def analytic_neff_equicorr(n: int, rho: float, m_cluster: int, metric: str) -> float:
  """Computes exact analytical effective sample size for equicorrelated Gaussian clusters."""
  if rho <= 0.0 or m_cluster <= 1:
    return float(n)
  m = metric.lower().strip()
  if m in ("mse", "rmse"):
    return float(n / (1.0 + (m_cluster - 1) * (rho ** 2)))
  elif m == "mae":
    r_mae = r_mae_factor(rho)
    return float(n / (1.0 + (m_cluster - 1) * r_mae))
  raise ValueError(f"Analytic n_eff not defined for metric {metric!r}")


def sample_equicorr(
    n: int, rho: float, m_cluster: int, rng: np.random.Generator
) -> np.ndarray:
  """Generates equicorrelated Gaussian sample of size n in clusters of size m_cluster."""
  if rho <= 0.0 or m_cluster <= 1:
    return rng.normal(size=n)
  g = n // m_cluster
  remainder = n % m_cluster
  z = rng.normal(size=(g, 1))
  u = rng.normal(size=(g, m_cluster))
  e = math.sqrt(rho) * z + math.sqrt(1.0 - rho) * u
  e_flat = e.reshape(-1)
  if remainder > 0:
    z_rem = rng.normal()
    u_rem = rng.normal(size=remainder)
    e_rem = math.sqrt(rho) * z_rem + math.sqrt(1.0 - rho) * u_rem
    e_flat = np.concatenate([e_flat, e_rem])
  return e_flat


def get_distribution(name: str) -> Any:
  """Retrieves named distribution from tuning or held-out sets."""
  for d in distributions.TUNING_SET:
    if d.name == name:
      return d
  for d in distributions.HELD_OUT_SET:
    if d.name == name:
      return d
  raise KeyError(f"Unknown distribution {name!r}")


def generate_errors(
    family: str,
    n: int,
    rng: np.random.Generator,
    rho: float = 0.0,
    m_cluster: int = 1,
    p_bern: float = 0.5,
) -> tuple[np.ndarray, float | None]:
  """Generates error sample and returns (errors, analytic_neff).

  Args:
    family: Name of error family (e.g. 'gaussian', 'bernoulli_p05', 'equicorr_r03_m10').
    n: Number of samples.
    rng: NumPy random generator.
    rho: Cluster correlation (if equicorrelated).
    m_cluster: Cluster size (if equicorrelated).
    p_bern: Bernoulli parameter (if Bernoulli).

  Returns:
    (errors, None): errors array and None (caller computes metric-specific effective_n if needed).
  """
  fam = family.lower().strip()
  if fam.startswith("bernoulli"):
    if p_bern is None or not (0.0 < p_bern < 1.0):
      raise ValueError(f"p_bern must be in (0, 1), got {p_bern}")
    draws = rng.binomial(n=1, p=p_bern, size=n).astype(np.float64)
    return draws, None
  elif fam.startswith("equicorr"):
    errors = sample_equicorr(n, rho, m_cluster, rng)
    # Return errors; caller computes metric-specific n_eff via analytic_neff_equicorr
    return errors, None
  else:
    dist = get_distribution(fam)
    return dist.sample(n, rng), None
