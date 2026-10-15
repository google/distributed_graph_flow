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

"""Reference error distributions with analytic ground truth for CI calibration.

Every distribution here is standardized to zero mean and unit variance, so the
population RMSE target is exactly 1.0 and interval widths are directly
comparable across distributions. The population MAE target varies and is
supplied analytically (verified against Monte Carlo in the accompanying test).

The central concept is `max_finite_moment`: the supremum of p such that
E|e|^p < inf. It determines, per metric, whether a distribution is inside the
library's contract:

  metric      summand   needs                admissible iff
  ----------  --------  -------------------  ----------------------
  MSE/RMSE/R2 e^2       Var(e^2) < inf       max_finite_moment > 4
  MAE         |e|       Var(|e|) < inf       max_finite_moment > 2
  accuracy    indicator bounded              always

This replaces the single global `expect_nominal_coverage` flag used by the
previous harness, which could not express that Student-t(3) is a legitimate
in-contract case for MAE while being out of contract for RMSE.
"""

import dataclasses
import math
from typing import Callable, Sequence

import numpy as np
import scipy.special


# Sampler contract: (size, rng) -> standardized errors, shape (size,).
Sampler = Callable[[int, np.random.Generator], np.ndarray]

# CDF contract: elementwise P(e <= x) on the standardized scale.
Cdf = Callable[[np.ndarray], np.ndarray]

# Summand moment order required for each metric's summand to have finite
# variance. Derived from the metric definition, not tuned.
_METRIC_REQUIRED_MOMENT: dict[str, float] = {
    "mse": 4.0,
    "rmse": 4.0,
    "r2_ind": 4.0,
    "r2_trans": 4.0,
    "mae": 2.0,
    "accuracy": 0.0,
}

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_SQRT_3 = math.sqrt(3.0)

# Smallest probability the copula's uniform coordinate is allowed to take.
# Phi^-1(1e-15) is about -7.94, so clamping here is invisible to any realistic
# correlation while keeping the inverse normal transform finite.
_PIT_EPSILON = 1e-15


@dataclasses.dataclass(frozen=True)
class ErrorDistribution:
  """A standardized error distribution with analytically known moments.

  Errors are standardized so that E[e] = 0 and E[e^2] = 1. Consequently the
  population RMSE is 1.0 for every distribution and only `mean_abs` (the
  population MAE) differs.

  Usage example:

  ```python
    dist = GAUSSIAN
    rng = np.random.default_rng(0)
    errors = dist.sample(1000, rng)
    assert dist.admits("rmse")
  ```

  Attributes:
    name: Short identifier used in result tables.
    mean_abs: E|e|, the population MAE.
    max_finite_moment: sup{p : E|e|^p < inf}; math.inf for light tails.
    support_max: The exact supremum of |e| on the standardized scale, or
      math.inf when the support is unbounded. A numeric bound rather than a
      boolean because it is the quantity the declared-support knob consumes,
      and because it can be checked exactly against draws.
    skewness: Population skewness of e, or None if not finite/known.
    sampler: Draws standardized errors.
    cdf: Right-continuous P(e <= x), elementwise, on the standardized scale.
      Used to couple errors to labels through a Gaussian copula without
      disturbing this marginal.
    cdf_left: P(e < x). Equal to `cdf` for continuous distributions, which is
      why it defaults to None and is only supplied by distributions with
      atoms.
  """

  name: str
  mean_abs: float
  max_finite_moment: float
  support_max: float
  sampler: Sampler
  cdf: Cdf
  skewness: float | None = None
  cdf_left: Cdf | None = None

  @property
  def bounded(self) -> bool:
    """Whether the support is bounded, enabling the declared-support knob."""
    return math.isfinite(self.support_max)

  @property
  def mean_sq(self) -> float:
    """Population E[e^2]. Unity by standardization."""
    return 1.0

  def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draws `size` standardized errors."""
    return self.sampler(size, rng)

  def probability_integral_transform(self, errors: np.ndarray) -> np.ndarray:
    """Maps errors to (0, 1) for use as the copula's uniform coordinate.

    For a continuous distribution this is exactly F(e) ~ Uniform(0, 1). For a
    distribution with atoms the PIT cannot be uniform, so the mid-rank
    (F(e-) + F(e)) / 2 is used instead: it is the standard convention, it is
    symmetric, and above all it stays strictly inside (0, 1) so that the
    subsequent inverse normal transform is finite. The consequence is that the
    realised correlation for discrete distributions falls short of the
    requested one, which is why the harness measures it rather than assuming
    it.

    Args:
      errors: Standardized errors.

    Returns:
      Values in (0, 1) with the same shape as `errors`.
    """
    upper = np.asarray(self.cdf(errors), dtype=np.float64)
    if self.cdf_left is None:
      raw = upper
    else:
      lower = np.asarray(self.cdf_left(errors), dtype=np.float64)
      raw = 0.5 * (lower + upper)
    # A far-out draw can round F(e) to exactly 0.0 or 1.0 in float64, and the
    # copula's inverse normal transform of that is infinite. Clamping keeps
    # the transform finite; the clamp is 8 standard normal deviations out, so
    # it cannot affect any realistic correlation.
    return np.clip(raw, _PIT_EPSILON, 1.0 - _PIT_EPSILON)

  def admits(self, metric: str) -> bool:
    """Whether this distribution is inside the contract for `metric`.

    Args:
      metric: One of the keys of `_METRIC_REQUIRED_MOMENT`.

    Returns:
      True if the metric's summand has finite variance under this distribution.
    """
    if metric not in _METRIC_REQUIRED_MOMENT:
      raise ValueError(
          f"Unknown metric {metric!r}; expected one of"
          f" {sorted(_METRIC_REQUIRED_MOMENT)}."
      )
    return self.max_finite_moment > _METRIC_REQUIRED_MOMENT[metric]


def admissible_metrics(dist: ErrorDistribution, metrics: Sequence[str]) -> list[str]:
  """Filters `metrics` down to those inside the contract for `dist`."""
  return [m for m in metrics if dist.admits(m)]


# ---------------------------------------------------------------------------
# Light-tailed reference distributions (all moments finite)
# ---------------------------------------------------------------------------


def _gaussian(size: int, rng: np.random.Generator) -> np.ndarray:
  return rng.standard_normal(size)


def _gaussian_cdf(x: np.ndarray) -> np.ndarray:
  return scipy.special.ndtr(x)


GAUSSIAN = ErrorDistribution(
    name="gaussian",
    # E|Z| = sqrt(2/pi) for Z ~ N(0, 1).
    mean_abs=_SQRT_2_OVER_PI,
    max_finite_moment=math.inf,
    support_max=math.inf,
    sampler=_gaussian,
    cdf=_gaussian_cdf,
    skewness=0.0,
)


def _uniform(size: int, rng: np.random.Generator) -> np.ndarray:
  return rng.uniform(-_SQRT_3, _SQRT_3, size=size)


def _uniform_cdf(x: np.ndarray) -> np.ndarray:
  return np.clip((np.asarray(x) + _SQRT_3) / (2.0 * _SQRT_3), 0.0, 1.0)


UNIFORM = ErrorDistribution(
    name="uniform",
    # E|U| = half_width / 2 = sqrt(3) / 2 for U ~ Uniform(-sqrt3, sqrt3).
    mean_abs=_SQRT_3 / 2.0,
    max_finite_moment=math.inf,
    support_max=_SQRT_3,
    sampler=_uniform,
    cdf=_uniform_cdf,
    skewness=0.0,
)


_LAPLACE_SCALE = 1.0 / math.sqrt(2.0)


def _laplace(size: int, rng: np.random.Generator) -> np.ndarray:
  # Var = 2 b^2 = 1  =>  b = 1/sqrt(2).
  return rng.laplace(loc=0.0, scale=_LAPLACE_SCALE, size=size)


def _laplace_cdf_scaled(x: np.ndarray, scale: float) -> np.ndarray:
  """CDF of Laplace(0, scale), written to avoid overflow in either tail."""
  z = np.asarray(x, dtype=np.float64) / scale
  # np.where evaluates both branches, so exp(z) alone would overflow for large
  # positive z even though that branch is discarded.
  half_tail = 0.5 * np.exp(-np.abs(z))
  return np.where(z < 0.0, half_tail, 1.0 - half_tail)


LAPLACE = ErrorDistribution(
    name="laplace",
    # E|X| = b for X ~ Laplace(0, b).
    mean_abs=_LAPLACE_SCALE,
    max_finite_moment=math.inf,
    support_max=math.inf,
    sampler=_laplace,
    cdf=lambda x: _laplace_cdf_scaled(x, _LAPLACE_SCALE),
    skewness=0.0,
)


def make_lognormal(sigma: float) -> ErrorDistribution:
  """Standardized lognormal: all moments finite, but very slow CLT convergence.

  A deliberately awkward in-contract case. Every moment exists, so the metric
  contract is satisfied for all metrics, yet the summand skewness is large
  enough that naive normal-approximation intervals under-cover badly. This is
  the distribution that separates "assumption satisfied" from "easy".

  Args:
    sigma: Log-scale parameter of the underlying lognormal.

  Returns:
    The standardized error distribution.
  """
  if sigma <= 0.0:
    raise ValueError(f"sigma must be positive, got {sigma}.")
  omega = math.exp(sigma * sigma)
  raw_mean = math.exp(sigma * sigma / 2.0)
  raw_std = math.sqrt((omega - 1.0) * omega)
  # For Y ~ LogNormal(0, sigma) with m = E[Y], the mean absolute deviation is
  # E|Y - m| = 2 E[(Y - m)^+] = 2 m (2 Phi(sigma/2) - 1).
  phi = 0.5 * (1.0 + math.erf((sigma / 2.0) / math.sqrt(2.0)))
  mean_abs = 2.0 * raw_mean * (2.0 * phi - 1.0) / raw_std
  skewness = (omega + 2.0) * math.sqrt(omega - 1.0)

  def sampler(size: int, rng: np.random.Generator) -> np.ndarray:
    y = rng.lognormal(mean=0.0, sigma=sigma, size=size)
    return (y - raw_mean) / raw_std

  def cdf(x: np.ndarray) -> np.ndarray:
    # e = (y - raw_mean) / raw_std with y > 0, so P(e <= x) = P(y <= raw)
    # and the standardized support starts at -raw_mean / raw_std.
    raw = np.asarray(x, dtype=np.float64) * raw_std + raw_mean
    positive = raw > 0.0
    out = np.zeros_like(raw)
    out[positive] = scipy.special.ndtr(np.log(raw[positive]) / sigma)
    return out

  return ErrorDistribution(
      name=f"lognormal_s{sigma:g}".replace(".", ""),
      mean_abs=mean_abs,
      max_finite_moment=math.inf,
      support_max=math.inf,
      sampler=sampler,
      cdf=cdf,
      skewness=skewness,
  )


def make_exponential() -> ErrorDistribution:
  """Standardized centred exponential: skewed, one-sided, all moments finite."""
  # X ~ Exp(1) has mean 1, sd 1, so the standardized error is X - 1.
  # E|X - 1| = 2 E[(X-1)^+] = 2 exp(-1).
  def cdf(x: np.ndarray) -> np.ndarray:
    # Support of the standardized error starts at -1.
    raw = np.asarray(x, dtype=np.float64) + 1.0
    return -np.expm1(-np.maximum(raw, 0.0))

  return ErrorDistribution(
      name="exponential",
      mean_abs=2.0 * math.exp(-1.0),
      max_finite_moment=math.inf,
      support_max=math.inf,
      sampler=lambda size, rng: rng.standard_exponential(size) - 1.0,
      cdf=cdf,
      skewness=2.0,
  )


def make_discrete_tied() -> ErrorDistribution:
  """Five-point discrete errors: maximal ties, bounded, platykurtic.

  Reproduces the tie-heavy case that broke the previous implementation's
  threshold-based tail fitting (+75% width inflation from zero excesses at the
  threshold). A correct tail model must be exactly inert here.
  """
  values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
  probs = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
  variance = float(np.sum(probs * values**2))
  scale = math.sqrt(variance)
  mean_abs = float(np.sum(probs * np.abs(values))) / scale
  atoms = values / scale
  cumulative = np.cumsum(probs)

  def sampler(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(values, size=size, p=probs) / scale

  def _cumulative_at(index: np.ndarray) -> np.ndarray:
    # index == -1 means "below the smallest atom"; plain negative indexing
    # would silently wrap around to the top of `cumulative`.
    return np.where(index < 0, 0.0, cumulative[np.maximum(index, 0)])

  def cdf(x: np.ndarray) -> np.ndarray:
    # 'right' counts atoms <= x, giving the right-continuous CDF.
    return _cumulative_at(np.searchsorted(atoms, x, side="right") - 1)

  def cdf_left(x: np.ndarray) -> np.ndarray:
    # 'left' counts atoms < x, giving P(e < x).
    return _cumulative_at(np.searchsorted(atoms, x, side="left") - 1)

  return ErrorDistribution(
      name="discrete_tied",
      mean_abs=mean_abs,
      max_finite_moment=math.inf,
      support_max=float(np.max(np.abs(values))) / scale,
      sampler=sampler,
      cdf=cdf,
      skewness=0.0,
      cdf_left=cdf_left,
  )


def make_contaminated_gaussian(
    epsilon: float, outlier_scale: float
) -> ErrorDistribution:
  """Gaussian body with a rare wide-Gaussian contaminant.

  This is the case the tail knob must get right per the light-tails-by-default
  requirement: all moments are finite and the bulk is exactly Gaussian, but a
  small fraction of draws look like outliers. The tail term should react to
  observed outliers without assuming a power law.

  Args:
    epsilon: Contamination fraction in (0, 1).
    outlier_scale: Standard deviation of the contaminating component.

  Returns:
    The standardized error distribution.
  """
  if not 0.0 < epsilon < 1.0:
    raise ValueError(f"epsilon must be in (0, 1), got {epsilon}.")
  if outlier_scale <= 0.0:
    raise ValueError(f"outlier_scale must be positive, got {outlier_scale}.")
  raw_var = (1.0 - epsilon) + epsilon * outlier_scale**2
  scale = math.sqrt(raw_var)
  # Mixture of two zero-mean Gaussians: E|X| = sum_i w_i sigma_i sqrt(2/pi).
  mean_abs = (
      (1.0 - epsilon) * 1.0 + epsilon * outlier_scale
  ) * _SQRT_2_OVER_PI / scale

  def sampler(size: int, rng: np.random.Generator) -> np.ndarray:
    is_outlier = rng.random(size) < epsilon
    sigma = np.where(is_outlier, outlier_scale, 1.0)
    return rng.standard_normal(size) * sigma / scale

  def cdf(x: np.ndarray) -> np.ndarray:
    # Mixture CDF is the weighted sum of the component CDFs; closed form,
    # unlike its inverse.
    raw = np.asarray(x, dtype=np.float64) * scale
    return (1.0 - epsilon) * scipy.special.ndtr(raw) + epsilon * (
        scipy.special.ndtr(raw / outlier_scale)
    )

  name = f"contam_e{epsilon:g}_s{outlier_scale:g}".replace(".", "")
  return ErrorDistribution(
      name=name,
      mean_abs=mean_abs,
      max_finite_moment=math.inf,
      support_max=math.inf,
      sampler=sampler,
      cdf=cdf,
      skewness=0.0,
  )


# ---------------------------------------------------------------------------
# Heavy-tailed distributions with an explicit finite-moment boundary
# ---------------------------------------------------------------------------


def make_student_t(df: float) -> ErrorDistribution:
  """Standardized Student-t with `df` degrees of freedom.

  `max_finite_moment` is exactly `df`, so df=3 is admissible for MAE (needs
  > 2) but not for RMSE (needs > 4), and df=5 is admissible for both. This is
  the cleanest dial for probing the contract boundary.

  Args:
    df: Degrees of freedom; must exceed 2 for the variance to exist.

  Returns:
    The standardized error distribution.
  """
  if df <= 2.0:
    raise ValueError(f"df must exceed 2 for finite variance, got {df}.")
  scale = math.sqrt((df - 2.0) / df)
  # E|T_df| = 2 sqrt(df) Gamma((df+1)/2) / ((df-1) sqrt(pi) Gamma(df/2)).
  log_ratio = scipy.special.gammaln((df + 1.0) / 2.0) - scipy.special.gammaln(
      df / 2.0
  )
  mean_abs_raw = (
      2.0 * math.sqrt(df) * math.exp(log_ratio) / ((df - 1.0) * math.sqrt(math.pi))
  )
  # Skewness exists only for df > 3.
  skewness = 0.0 if df > 3.0 else None

  def sampler(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_t(df, size=size) * scale

  def cdf(x: np.ndarray) -> np.ndarray:
    return scipy.special.stdtr(df, np.asarray(x, dtype=np.float64) / scale)

  return ErrorDistribution(
      name=f"student_t{df:g}".replace(".", "_"),
      mean_abs=mean_abs_raw * scale,
      max_finite_moment=df,
      support_max=math.inf,
      sampler=sampler,
      cdf=cdf,
      skewness=skewness,
  )


def make_pareto(alpha: float) -> ErrorDistribution:
  """Standardized Pareto Type I with shape `alpha` on [1, inf).

  `max_finite_moment` is exactly `alpha`. Note that alpha slightly above 4 is
  in contract for RMSE yet still has infinite kurtosis variance, which is the
  regime where plug-in kurtosis estimates are worst behaved.

  Args:
    alpha: Tail shape; must exceed 2 for the variance to exist.

  Returns:
    The standardized error distribution.
  """
  if alpha <= 2.0:
    raise ValueError(f"alpha must exceed 2 for finite variance, got {alpha}.")
  raw_mean = alpha / (alpha - 1.0)
  raw_var = alpha / (((alpha - 1.0) ** 2) * (alpha - 2.0))
  raw_std = math.sqrt(raw_var)
  # E|X - mu| = 2 ((alpha-1)/alpha)^(alpha-1) / (alpha - 1) for Pareto(alpha).
  mean_abs = (
      2.0 * ((alpha - 1.0) / alpha) ** (alpha - 1.0) / (alpha - 1.0)
  ) / raw_std
  skewness = (
      (2.0 * (1.0 + alpha) / (alpha - 3.0)) * math.sqrt((alpha - 2.0) / alpha)
      if alpha > 3.0
      else None
  )

  def sampler(size: int, rng: np.random.Generator) -> np.ndarray:
    # numpy's pareto is Lomax; add 1 for Pareto Type I on [1, inf).
    x = rng.pareto(a=alpha, size=size) + 1.0
    return (x - raw_mean) / raw_std

  def cdf(x: np.ndarray) -> np.ndarray:
    # F(raw) = 1 - raw^-alpha on [1, inf), zero below the support edge.
    raw = np.asarray(x, dtype=np.float64) * raw_std + raw_mean
    return np.where(raw <= 1.0, 0.0, 1.0 - np.maximum(raw, 1.0) ** (-alpha))

  return ErrorDistribution(
      name=f"pareto_a{alpha:g}".replace(".", "_"),
      mean_abs=mean_abs,
      max_finite_moment=alpha,
      support_max=math.inf,
      sampler=sampler,
      cdf=cdf,
      skewness=skewness,
  )


def make_truncated_pareto(alpha: float, upper: float) -> ErrorDistribution:
  """Pareto truncated at `upper`: bounded support that *looks* heavy-tailed.

  The adversarial case for any tail model. Every moment is finite because the
  support is bounded, so the metric contract holds for all metrics, but any
  finite sample looks power-law. A model that extrapolates a Pareto tail here
  will be over-wide; one that ignores the evidence will be over-narrow at
  moderate n. Reported, not tuned against.

  Args:
    alpha: Tail shape of the underlying Pareto.
    upper: Truncation point on the raw scale; must exceed 1.

  Returns:
    The standardized error distribution.
  """
  if alpha <= 0.0:
    raise ValueError(f"alpha must be positive, got {alpha}.")
  if upper <= 1.0:
    raise ValueError(f"upper must exceed 1, got {upper}.")

  # Moments of Pareto(alpha) on [1, upper] via the truncated density
  # f(x) = alpha x^{-alpha-1} / (1 - upper^{-alpha}).
  norm = 1.0 - upper ** (-alpha)

  def raw_moment(order: int) -> float:
    if abs(alpha - order) < 1e-12:
      return alpha * math.log(upper) / norm
    return (
        alpha
        * (upper ** (order - alpha) - 1.0)
        / ((order - alpha) * norm)
    )

  raw_mean = raw_moment(1)
  raw_var = raw_moment(2) - raw_mean**2
  raw_std = math.sqrt(raw_var)
  # E|X - mu| = 2 E[(X - mu)^+] = 2 (integral_mu^upper (x - mu) f(x) dx).
  tail_mass = (raw_mean ** (-alpha) - upper ** (-alpha)) / norm
  if abs(alpha - 1.0) < 1e-12:
    partial_mean = alpha * math.log(upper / raw_mean) / norm
  else:
    partial_mean = (
        alpha
        * (upper ** (1.0 - alpha) - raw_mean ** (1.0 - alpha))
        / ((1.0 - alpha) * norm)
    )
  mean_abs = 2.0 * (partial_mean - raw_mean * tail_mass) / raw_std

  def sampler(size: int, rng: np.random.Generator) -> np.ndarray:
    # Inverse-CDF sampling of the truncated Pareto.
    u = rng.random(size)
    x = (1.0 - u * norm) ** (-1.0 / alpha)
    return (x - raw_mean) / raw_std

  def cdf(x: np.ndarray) -> np.ndarray:
    # Exact inverse of the sampler's transform: F(raw) = (1 - raw^-alpha)/norm
    # on [1, upper].
    raw = np.clip(
        np.asarray(x, dtype=np.float64) * raw_std + raw_mean, 1.0, upper
    )
    return (1.0 - raw ** (-alpha)) / norm

  name = f"trunc_pareto_a{alpha:g}_u{upper:g}".replace(".", "_")
  return ErrorDistribution(
      name=name,
      mean_abs=mean_abs,
      max_finite_moment=math.inf,  # bounded support
      # Raw support is [1, upper], so standardized |e| peaks at whichever
      # endpoint is further from the mean.
      support_max=max(raw_mean - 1.0, upper - raw_mean) / raw_std,
      sampler=sampler,
      cdf=cdf,
      skewness=None,
  )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Design decisions are made against this set only.
TUNING_SET: tuple[ErrorDistribution, ...] = (
    GAUSSIAN,
    UNIFORM,
    LAPLACE,
    make_discrete_tied(),
    make_exponential(),
    make_lognormal(0.25),
    make_lognormal(0.5),
    make_student_t(5.0),
    make_student_t(8.0),
    make_pareto(5.0),
    make_pareto(9.0),
    make_contaminated_gaussian(0.01, 10.0),
    make_contaminated_gaussian(0.05, 5.0),
    # Out of contract for RMSE, in contract for MAE. Included so the
    # admissibility filter itself is exercised.
    make_student_t(3.0),
)

# Held-out set, disjoint from the tuning set above. Contains shapes absent
# from the tuning set: near-boundary tail index, and heavy-tail-lookalikes.
HELD_OUT_SET: tuple[ErrorDistribution, ...] = (
    make_student_t(4.5),
    make_pareto(4.5),
    make_truncated_pareto(2.0, 1e4),
    make_truncated_pareto(3.0, 1e2),
    make_lognormal(1.0),
    make_contaminated_gaussian(0.002, 30.0),
)


def _mixture_component_moments(
    kind: str, scale: float
) -> tuple[float, float, float]:
  """Returns (variance, mean_abs, max_finite_moment) of a zero-centred component."""
  match kind:
    case "normal":
      return scale**2, scale * _SQRT_2_OVER_PI, math.inf
    case "laplace":
      # Laplace(0, b) with b = scale: Var = 2 b^2, E|X| = b.
      return 2.0 * scale**2, scale, math.inf
    case "uniform":
      # Uniform(-scale, scale): Var = scale^2 / 3, E|X| = scale / 2.
      return scale**2 / 3.0, scale / 2.0, math.inf
    case _:
      raise ValueError(f"Unknown mixture component kind {kind!r}.")


def sample_random_distributions(
    count: int, rng: np.random.Generator
) -> list[ErrorDistribution]:
  """Draws a pool of random symmetric scale-mixture error distributions.

  Zero-centred symmetric mixtures keep every population moment analytic (the
  mixture mean is exactly zero, so E|e| and E[e^2] are weight-weighted sums of
  component values) while spanning a far wider range of shapes than any fixed
  list. This is the primary defence against fitting the estimator to a handful
  of named distributions.

  Usage example:

  ```python
    pool = sample_random_distributions(200, np.random.default_rng(0))
  ```

  Args:
    count: Number of distributions to draw.
    rng: Source of randomness for the distribution *specifications*, not for
      the samples they later generate.

  Returns:
    A list of standardized error distributions with exact analytic moments.
  """
  if count <= 0:
    raise ValueError(f"count must be positive, got {count}.")
  kinds = ("normal", "laplace", "uniform")
  pool: list[ErrorDistribution] = []
  for index in range(count):
    num_components = int(rng.integers(1, 4))
    component_kinds = [str(rng.choice(kinds)) for _ in range(num_components)]
    # Log-uniform scales spanning two orders of magnitude produce both mild
    # heteroscedasticity and severe outlier contamination.
    scales = np.exp(rng.uniform(math.log(0.1), math.log(10.0), num_components))
    weights = rng.dirichlet(np.ones(num_components))

    variance = 0.0
    mean_abs_raw = 0.0
    for kind, scale, weight in zip(component_kinds, scales, weights):
      comp_var, comp_mean_abs, _ = _mixture_component_moments(kind, float(scale))
      variance += float(weight) * comp_var
      mean_abs_raw += float(weight) * comp_mean_abs
    std = math.sqrt(variance)

    frozen_kinds = tuple(component_kinds)
    frozen_scales = tuple(float(s) for s in scales)
    frozen_weights = tuple(float(w) for w in weights)

    def sampler(
        size: int,
        sample_rng: np.random.Generator,
        _kinds: tuple[str, ...] = frozen_kinds,
        _scales: tuple[float, ...] = frozen_scales,
        _weights: tuple[float, ...] = frozen_weights,
        _std: float = std,
    ) -> np.ndarray:
      which = sample_rng.choice(len(_kinds), size=size, p=np.array(_weights))
      out = np.empty(size, dtype=np.float64)
      for component_index, (kind, scale) in enumerate(zip(_kinds, _scales)):
        mask = which == component_index
        num = int(np.count_nonzero(mask))
        if num == 0:
          continue
        match kind:
          case "normal":
            out[mask] = sample_rng.standard_normal(num) * scale
          case "laplace":
            out[mask] = sample_rng.laplace(0.0, scale, num)
          case "uniform":
            out[mask] = sample_rng.uniform(-scale, scale, num)
      return out / _std

    def cdf(
        x: np.ndarray,
        _kinds: tuple[str, ...] = frozen_kinds,
        _scales: tuple[float, ...] = frozen_scales,
        _weights: tuple[float, ...] = frozen_weights,
        _std: float = std,
    ) -> np.ndarray:
      # Mixture CDF is the weighted sum of the component CDFs, each of which
      # is closed form. Default args freeze the loop variables, as in
      # `sampler` above.
      raw = np.asarray(x, dtype=np.float64) * _std
      total = np.zeros_like(raw)
      for kind, scale, weight in zip(_kinds, _scales, _weights):
        match kind:
          case "normal":
            component = scipy.special.ndtr(raw / scale)
          case "laplace":
            component = _laplace_cdf_scaled(raw, scale)
          case "uniform":
            component = np.clip((raw + scale) / (2.0 * scale), 0.0, 1.0)
          case _:
            raise ValueError(f"Unknown mixture component kind {kind!r}.")
        total += weight * component
      return total

    # Normal and Laplace components have unbounded support, so the mixture is
    # bounded only if every component is uniform; then |e| peaks at the widest
    # component half-width.
    if all(k == "uniform" for k in frozen_kinds):
      support_max = max(frozen_scales) / std
    else:
      support_max = math.inf

    pool.append(
        ErrorDistribution(
            name=f"random_{index:04d}",
            mean_abs=mean_abs_raw / std,
            max_finite_moment=math.inf,
            support_max=support_max,
            sampler=sampler,
            cdf=cdf,
            skewness=0.0,
        )
    )
  return pool
