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

"""Finite-sample confidence intervals for evaluation metrics.

Public API:
- Status: Enum describing the diagnostic status of the interval.
- Result: Immutable dataclass containing the interval and diagnostics.
- confidence_interval: The entry point.

The finite-sample coverage bound and its single assumption are stated in
THEORY_INDEPENDENT.md (Claim 1 for i.i.d. data, Claim 1′ for correlated data
with a declared effective sample size).
"""

from collections.abc import Sequence as _Sequence
import dataclasses as _dataclasses
import enum as _enum
import math as _math
from typing import Any as _Any

import numpy as _np
import scipy.special as _special

from dgf.src.stats.independent import bound as _bound
from dgf.src.stats.independent import metrics as _metrics
from dgf.src.stats.independent import sketch as _sketch

__all__ = [
    "Status",
    "Result",
    "confidence_interval",
]

# Minimum confidence level required for accuracy under without-replacement
# sampling (Claim 3′, alpha <= 2 - sqrt(2)).
_MIN_ACCURACY_LEVEL: float = _math.sqrt(2.0) - 1.0  # sqrt(2) - 1 ≈ 0.41421356


class Status(_enum.Enum):
  UNREFUTED = "unrefuted"  # M_hat <= M; nothing contradicts the declaration
  TAIL_UNRESOLVED = "tail_unresolved"  # M_hat > M; interval still returned
  ASSUMPTION_REQUIRED = (  # no interval; n too small for M, or invalid input
      "assumption_required"
  )


@_dataclasses.dataclass(frozen=True)
class Result:
  """Immutable result of a confidence interval estimation.

  Attributes:
    low: Lower confidence bound.
    high: Upper confidence bound.
    level: Confidence level (e.g. 0.95). None exactly when status is
      ASSUMPTION_REQUIRED.
    status: Diagnostic status of the interval.
    m_declared: Relative variance bound M assumed or declared by the caller.
    m_observed: Empirical relative variance M_hat = n * sum(s^2) / (sum s)^2.
  """

  low: float
  high: float
  level: float | None
  status: Status
  m_declared: float
  m_observed: float


def confidence_interval(
    data_or_sketch: _Any,
    metric: str,
    level: float = 0.95,
    m: float | None = None,
    effective_n: float | None = None,
) -> Result:
  """Computes a finite-sample confidence interval for the mean of a metric.

  Whenever an interval is returned, Pr(theta in [low, high]) >= level under the
  single assumption P2*(M): E[s^2] <= M * (E[s])^2, where s >= 0 is the per-item
  summand of the metric. Coverage is not conditional on the returned
  status. M is declared, not estimated; the status only reports whether the
  data contradict it. Apart from invalid (non-finite) input, whether an
  interval is returned at all depends only on (n, M, level, effective_n), not
  on the data values.

  Exception: accuracy on the independent path (effective_n is None) uses the
  exact binomial (Clopper-Pearson) interval, which needs no M. It is returned
  for every n >= 1 with status UNREFUTED; m is accepted and reported in
  m_declared but not used.

  Args:
    data_or_sketch: Raw data (array of residuals or pair of (labels,
      predictions)) or an IndependentSketch / R2Sketch instance.
    metric: The metric name ('mae', 'mse', 'rmse', 'accuracy', 'r2').
      'average_precision' is not supported and raises ValueError.
    level: Confidence level in (0, 1), default 0.95.
    m: Declared relative variance bound M >= 1 (finite). If None, uses metric
      defaults:
      MAE: M = 4 (Gaussian ~1.57) MSE / RMSE: M = 16 (Gaussian kurtosis 3)
      Accuracy: M = 4 (used only with effective_n)
      R2: M = 16
    effective_n: Effective sample size for correlated data in raw-data mode.
      Must not be passed if data_or_sketch is an IndependentSketch or R2Sketch. Required for
      temporally correlated evaluation windows (e.g. temporal splits); see
      README.md, "Temporal splits", for what to expect there.

  Important Notice on Accuracy:
    On the independent path, 0/1 indicators of any event (e.g. errors instead
    of correct predictions) passed as 'accuracy' get the exact interval, rare
    or not. With effective_n the generic bound applies, where M = 1/p: a
    classifier with accuracy 0.99 has M ~ 1.01, but its error rate 0.01 has
    M = 100 and needs far more data.

  Returns:
    A Result containing [low, high], the level, the diagnostic status,
    m_declared,
    and m_observed.
  """
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")

  alpha = 1.0 - level
  norm_metric = _metrics.normalize_metric(metric)

  if m is not None:
    m_declared = float(m)
    # Written so that NaN fails: `m_declared < 1.0` is False for NaN.
    if not (_math.isfinite(m_declared) and m_declared >= 1.0):
      raise ValueError(f"m must be finite and >= 1.0, got {m_declared}.")
  else:
    m_declared = _metrics.get_default_m(norm_metric)

  # Branch: R^2 metric requires disjoint pairing of y for variance Var(y)
  if norm_metric == "r2":
    return _confidence_interval_r2(
        data_or_sketch, alpha, level, m_declared, effective_n=effective_n
    )

  # Resolution rule for effective_n:
  # sketch field | call argument | behaviour
  # None         | None          | i.i.d. path
  # set          | None          | use the sketch's value
  # None         | set           | ValueError - sketch asserts i.i.d.
  # set          | set           | ValueError - even if equal
  if isinstance(data_or_sketch, _sketch.IndependentSketch):
    if data_or_sketch.effective_n is None and effective_n is not None:
      raise ValueError(
          "Cannot pass effective_n when IndependentSketch has effective_n=None (asserts"
          " i.i.d.)."
      )
    if data_or_sketch.effective_n is not None and effective_n is not None:
      raise ValueError(
          "Cannot pass effective_n when IndependentSketch already has effective_n set."
      )
    eff_n = data_or_sketch.effective_n
    n = data_or_sketch.n
    sum_s = data_or_sketch.sum_s
    sum_s2 = data_or_sketch.sum_s2
    top_k = data_or_sketch.top_32

    if norm_metric == "accuracy":
      if len(top_k) > 0 and top_k[0] > 1.0:
        raise ValueError(
            f"Accuracy sketch contains values exceeding 1.0 (max value: {top_k[0]});"
            " accuracy expects 0/1 correctness values."
        )
      if _math.isfinite(sum_s) and _math.isfinite(sum_s2):
        k = int(round(sum_s))
        if float(k) != sum_s or sum_s2 != sum_s or not 0 <= k <= n:
          raise ValueError(
              f"Accuracy sketch sums must be exact integers with sum_s2 == sum_s and"
              f" 0 <= sum_s <= n={n}, got sum_s={sum_s}, sum_s2={sum_s2}."
          )
  else:
    s = _metrics.extract_summands(data_or_sketch, norm_metric)
    n = len(s)
    if n == 0:
      return Result(
          low=float("nan"),
          high=float("nan"),
          level=None,
          status=Status.ASSUMPTION_REQUIRED,
          m_declared=m_declared,
          m_observed=float("nan"),
      )
    if effective_n is not None:
      if (
          not _math.isfinite(effective_n)
          or effective_n < 1.0
          or effective_n > float(n)
      ):
        raise ValueError(
            f"effective_n must be finite in [1.0, n={n}], got {effective_n}"
        )
    eff_n = effective_n
    sum_s = float(_np.sum(s))
    sum_s2 = float(_np.sum(s**2))
    top_k = _np.sort(s)[::-1]  # Data mode: all n elements

  # m_observed uses the raw n: M_hat = n * sum_s2 / (sum_s)^2 is a ratio of two
  # sample means, each unbiased regardless of dependence, so it estimates
  # E[s^2]/theta^2 in the correlated case too. Using n_eff here would be wrong.
  if sum_s == 0.0:
    m_observed = 1.0
  elif n < 2 or not _math.isfinite(sum_s) or not _math.isfinite(sum_s2):
    m_observed = float("nan")
  else:
    m_observed = float((n * sum_s2) / (sum_s**2))

  # Accuracy on the independent path: the summands are Bernoulli(theta), so the
  # exact binomial interval applies and M is not needed (THEORY_INDEPENDENT.md, §4.2, Claim 3).
  # The correlated path keeps the generic bound: under dependence the count of
  # correct items is not binomial.
  if norm_metric == "accuracy" and eff_n is None:
    return _accuracy_exact(
        n, sum_s, sum_s2, alpha, level, m_declared, m_observed, tuple(top_k)
    )

  (low, high), st_str = _bound.bound(
      n=n,
      sum_s=sum_s,
      sum_s2=sum_s2,
      top_k=top_k,
      alpha=alpha,
      M=m_declared,
      metric=norm_metric,
      effective_n=eff_n,
  )

  if st_str == "ASSUMPTION_REQUIRED":
    return Result(
        low=float("nan"),
        high=float("nan"),
        level=None,
        status=Status.ASSUMPTION_REQUIRED,
        m_declared=m_declared,
        m_observed=m_observed,
    )

  status = (
      Status.TAIL_UNRESOLVED
      if st_str == "TAIL_UNRESOLVED"
      else Status.UNREFUTED
  )
  return Result(
      low=float(low),
      high=float(high),
      level=level,
      status=status,
      m_declared=m_declared,
      m_observed=m_observed,
  )


def _accuracy_exact(
    n: int,
    sum_s: float,
    sum_s2: float,
    alpha: float,
    level: float,
    m_declared: float,
    m_observed: float,
    top_32: tuple[float, ...] = (),
) -> Result:
  """Exact (Clopper-Pearson) interval for accuracy on the independent path.

  Endpoints are evaluated at level alpha_tilde = alpha - alpha**2 / 4, which
  makes the interval valid for uniform samples without replacement too (Claim
  3′, alpha <= 2 - sqrt(2)).

  For 0/1 summands sum_s is the count k of correct items and sum_s2 == sum_s.
  Integer counts are exact in float64 up to 2**53; a sketch from the JAX
  accumulator adds a float32 sum and its compensation term, both integers for
  0/1 input. A sketch whose sums are not exact integers with sum_s2 == sum_s
  and 0 <= sum_s <= n was not built from 0/1 indicators and raises ValueError.

  Args:
    n: Number of items.
    sum_s: Sum of the correctness indicators.
    sum_s2: Sum of their squares.
    alpha: 1 - level.
    level: Confidence level.
    m_declared: Reported unchanged; the exact interval does not use it.
    m_observed: Reported unchanged.
    top_32: Retained top order statistics from sketch, if available.

  Returns:
    The Result. The status is UNREFUTED whenever an interval is returned: there
    is no declaration for the data to contradict.

  Raises:
    ValueError: If level < sqrt(2) - 1 ≈ 0.4142, or if accuracy sketch sums are
      not exact integers with sum_s2 == sum_s and 0 <= sum_s <= n, or if
      top_32 contains values exceeding 1.0.
  """
  if level < _MIN_ACCURACY_LEVEL:
    raise ValueError(
        f"Accuracy confidence interval requires level >= sqrt(2) - 1 ≈ 0.4142"
        f" (Claim 3′), got level={level}."
    )

  if len(top_32) > 0 and top_32[0] > 1.0:
    raise ValueError(
        f"Accuracy sketch contains values exceeding 1.0 (max value: {top_32[0]});"
        " accuracy expects 0/1 correctness values."
    )

  refused = Result(
      low=float("nan"),
      high=float("nan"),
      level=None,
      status=Status.ASSUMPTION_REQUIRED,
      m_declared=m_declared,
      m_observed=m_observed,
  )
  if n < 1 or not (_math.isfinite(sum_s) and _math.isfinite(sum_s2)):
    return refused

  k = int(round(sum_s))
  if (
      float(k) != sum_s
      or sum_s2 != sum_s
      or not 0 <= k <= n
  ):
    raise ValueError(
        f"Accuracy sketch sums must be exact integers with sum_s2 == sum_s and"
        f" 0 <= sum_s <= n={n}, got sum_s={sum_s}, sum_s2={sum_s2}."
    )

  # Clopper-Pearson: the endpoints are Beta quantiles, alpha_tilde/2 on each side.
  alpha_tilde = alpha - (alpha**2) / 4.0
  if k == 0:
    low = 0.0
  else:
    low = float(_special.betaincinv(k, n - k + 1, alpha_tilde / 2.0))
  if k == n:
    high = 1.0
  else:
    high = float(_special.betaincinv(k + 1, n - k, 1.0 - alpha_tilde / 2.0))
  return Result(
      low=low,
      high=high,
      level=level,
      status=Status.UNREFUTED,
      m_declared=m_declared,
      m_observed=m_observed,
  )


def _confidence_interval_r2(
    data_or_sketch: _Any,
    alpha: float,
    level: float,
    m_declared: float,
    effective_n: float | None = None,
) -> Result:
  """Subroutine for R^2 confidence interval using disjoint pairing."""
  is_correlated = effective_n is not None
  if isinstance(data_or_sketch, _sketch.R2Sketch):
    sk_a = data_or_sketch.sketch_a
    sk_b = data_or_sketch.sketch_b
    if sk_a.effective_n is not None or sk_b.effective_n is not None:
      is_correlated = True
    n_a, sum_a, sum2_a, top_a = sk_a.n, sk_a.sum_s, sk_a.sum_s2, sk_a.top_32
    n_b, sum_b, sum2_b, top_b = sk_b.n, sk_b.sum_s, sk_b.sum_s2, sk_b.top_32
  elif (
      isinstance(data_or_sketch, tuple)
      and len(data_or_sketch) == 2
      and isinstance(data_or_sketch[0], _sketch.IndependentSketch)
      and isinstance(data_or_sketch[1], _sketch.IndependentSketch)
  ):
    sk_a = data_or_sketch[0]
    sk_b = data_or_sketch[1]
    if sk_a.effective_n is not None or sk_b.effective_n is not None:
      is_correlated = True
    n_a, sum_a, sum2_a, top_a = sk_a.n, sk_a.sum_s, sk_a.sum_s2, sk_a.top_32
    n_b, sum_b, sum2_b, top_b = sk_b.n, sk_b.sum_s, sk_b.sum_s2, sk_b.top_32
  else:
    s_a, s_b = _metrics.extract_r2_summands(data_or_sketch)
    n_a = len(s_a)
    n_b = len(s_b)
    if n_a == 0 or n_b == 0:
      return Result(
          low=float("nan"),
          high=float("nan"),
          level=None,
          status=Status.ASSUMPTION_REQUIRED,
          m_declared=m_declared,
          m_observed=float("nan"),
      )
    sum_a = float(_np.sum(s_a))
    sum2_a = float(_np.sum(s_a**2))
    top_a = _np.sort(s_a)[::-1]
    sum_b = float(_np.sum(s_b))
    sum2_b = float(_np.sum(s_b**2))
    top_b = _np.sort(s_b)[::-1]

  # M_observed for each sequence (using raw n)
  if sum_a == 0.0:
    m_obs_a = 1.0
  elif n_a < 2 or not _math.isfinite(sum_a) or not _math.isfinite(sum2_a):
    m_obs_a = float("nan")
  else:
    m_obs_a = float((n_a * sum2_a) / (sum_a**2))

  if sum_b == 0.0:
    m_obs_b = 1.0
  elif n_b < 2 or not _math.isfinite(sum_b) or not _math.isfinite(sum2_b):
    m_obs_b = float("nan")
  else:
    m_obs_b = float((n_b * sum2_b) / (sum_b**2))

  m_observed = (
      max(m_obs_a, m_obs_b)
      if _math.isfinite(m_obs_a) and _math.isfinite(m_obs_b)
      else float("nan")
  )

  # R2 with correlated data refuses with ASSUMPTION_REQUIRED (not an exception):
  # the two R2 sequences (e^2 and the paired b) have different and unknown effective sizes,
  # and the pairing itself changes the dependence structure.
  if is_correlated:
    return Result(
        low=float("nan"),
        high=float("nan"),
        level=None,
        status=Status.ASSUMPTION_REQUIRED,
        m_declared=m_declared,
        m_observed=m_observed,
    )

  # Each R^2 endpoint combines one endpoint from each sequence, so each side
  # gets alpha/2 split evenly over the two sequences: every call runs at total
  # alpha/2, i.e. alpha/4 per side.
  alpha_call = alpha / 2.0
  (l_a, u_a), st_a = _bound.bound(
      n=n_a,
      sum_s=sum_a,
      sum_s2=sum2_a,
      top_k=top_a,
      alpha=alpha_call,
      M=m_declared,
      metric="mse",
  )
  (l_b, u_b), st_b = _bound.bound(
      n=n_b,
      sum_s=sum_b,
      sum_s2=sum2_b,
      top_k=top_b,
      alpha=alpha_call,
      M=m_declared,
      metric="mse",
  )

  if st_a == "ASSUMPTION_REQUIRED" or st_b == "ASSUMPTION_REQUIRED":
    return Result(
        low=float("nan"),
        high=float("nan"),
        level=None,
        status=Status.ASSUMPTION_REQUIRED,
        m_declared=m_declared,
        m_observed=m_observed,
    )

  status = (
      Status.TAIL_UNRESOLVED
      if (st_a == "TAIL_UNRESOLVED" or st_b == "TAIL_UNRESOLVED")
      else Status.UNREFUTED
  )

  if l_b > 0.0 and _math.isfinite(u_a) and _math.isfinite(u_b):
    low = 1.0 - u_a / l_b
    high = 1.0 - l_a / u_b
  else:
    low = float("-inf")
    high = 1.0

  return Result(
      low=float(low),
      high=float(high),
      level=level,
      status=status,
      m_declared=m_declared,
      m_observed=m_observed,
  )
