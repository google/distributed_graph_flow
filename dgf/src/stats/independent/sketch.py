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

"""Mergeable fixed-size sketches of evaluation summands."""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np

from dgf.src.stats.independent import metrics as _metrics


@dataclasses.dataclass(frozen=True)
class IndependentSketch:
  """Fixed-size evaluation summary sketch for single-sequence metrics.

  Fields:
    n: Total count of observations.
    sum_s: Sum of per-item non-negative summands s_i.
    sum_s2: Sum of squared summands s_i^2.
    top_32: Tuple of up to 32 largest summand values in descending order.
    effective_n: Declared effective sample size for correlated data, in
      [1, n], or None for i.i.d. data. See README.md, "Correlated data".
    schema_version: Wire-format version; sketches of different versions
      refuse to merge.
  """

  n: int
  sum_s: float
  sum_s2: float
  top_32: tuple[float, ...]
  effective_n: float | None = None
  schema_version: int = 1

  def __post_init__(self) -> None:
    if self.effective_n is not None:
      if not math.isfinite(self.effective_n):
        raise ValueError(f"effective_n must be finite, got {self.effective_n}")
      if self.effective_n < 1.0 or self.effective_n > float(self.n):
        raise ValueError(
            f"effective_n must be in [1.0, n={self.n}], got {self.effective_n}"
        )

  def merge(self, other: "IndependentSketch") -> "IndependentSketch":
    return merge(self, other)

  def __add__(self, other: "IndependentSketch") -> "IndependentSketch":
    return merge(self, other)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, IndependentSketch):
      return False
    if self.n != other.n or self.schema_version != other.schema_version:
      return False
    if not math.isclose(self.sum_s, other.sum_s, rel_tol=1e-9, abs_tol=1e-12):
      return False
    if not math.isclose(self.sum_s2, other.sum_s2, rel_tol=1e-9, abs_tol=1e-12):
      return False
    if self.top_32 != other.top_32:
      return False
    if (self.effective_n is None) != (other.effective_n is None):
      return False
    if self.effective_n is not None and other.effective_n is not None:
      if not math.isclose(
          self.effective_n, other.effective_n, rel_tol=1e-9, abs_tol=1e-12
      ):
        return False
    return True


@dataclasses.dataclass(frozen=True)
class R2Sketch:
  """Sketch container for R^2, holding two single-statistic sketches.

  Attributes:
    sketch_a: Sketch of squared errors e_i^2 (sample size n).
    sketch_b: Sketch of disjoint pair differences b_i = 0.5 * (y_{2i} -
      y_{2i-1})^2 (sample size floor(n/2)).

  NOT SHARD-INVARIANT. Pairing is index-local, so a shard of odd size drops its
  last element. Merging S shards therefore yields sum_k floor(n_k / 2) pairs,
  which is up to S/2 fewer than the floor(sum_k n_k / 2) that sketching the
  concatenated data would give. `merge` stays associative and commutative --
  which is the property distributed reduction actually needs, and the reason
  this is not "fixed" by holding an orphan value and pairing orphans across a
  merge (that would make the result depend on the shape of the merge tree).

  The dropped elements are chosen by shard boundaries, never by value, so the
  surviving pairs are still i.i.d. with mean Var(y) and the interval stays
  valid. The cost is efficiency: slightly fewer pairs, hence a slightly wider
  interval for theta_B.

  The one layout that breaks outright is many tiny shards. With every n_k = 1,
  sketch_b is empty and the R^2 path can only return ASSUMPTION_REQUIRED. Use
  even shard sizes, or at least shards much larger than 1.
  """

  sketch_a: IndependentSketch
  sketch_b: IndependentSketch

  def merge(self, other: "R2Sketch") -> "R2Sketch":
    return merge(self, other)

  def __add__(self, other: "R2Sketch") -> "R2Sketch":
    return merge(self, other)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, R2Sketch):
      return False
    return self.sketch_a == other.sketch_a and self.sketch_b == other.sketch_b


def merge(a: Any, b: Any) -> Any:
  """Associative and commutative merge of two sketches.

  n, sum_s, sum_s2 and top_32 merge exactly. For correlated sketches, the
  effective sample sizes merge via the conservative kappa-maximum rule:
  kappa_k = n_k / effective_n_k, merged effective_n = (n_a + n_b) / max(kappa_a, kappa_b).
  This rule is conservative because the worst shard's variance inflation applies
  to all items. It is valid with unequal shard means (unlike W-space addition),
  provided shards are mutually uncorrelated and each shard's effective_n
  certifies the variance-inflation bound (V_k) via Lemma H or a variance ratio.
  Merging remains optimistic, hence unsafe, when a correlated group is split
  across two shards.
  """
  if isinstance(a, R2Sketch) and isinstance(b, R2Sketch):
    return R2Sketch(
        sketch_a=merge(a.sketch_a, b.sketch_a),
        sketch_b=merge(a.sketch_b, b.sketch_b),
    )
  if isinstance(a, IndependentSketch) and isinstance(b, IndependentSketch):
    if a.schema_version != b.schema_version:
      raise ValueError(
          f"Schema version mismatch: {a.schema_version} != {b.schema_version}"
      )
    if a.effective_n is None and b.effective_n is None:
      merged_eff_n = None
    elif a.effective_n is not None and b.effective_n is not None:
      # Conservative kappa-max merge: kappa_k = n_k / effective_n_k,
      # merged effective_n = (n_a + n_b) / max(kappa_a, kappa_b).
      if a.n == 0 and b.n == 0:
        merged_eff_n = None
      elif a.n == 0:
        merged_eff_n = b.effective_n
      elif b.n == 0:
        merged_eff_n = a.effective_n
      else:
        kappa_a = float(a.n) / float(a.effective_n)
        kappa_b = float(b.n) / float(b.effective_n)
        merged_eff_n = float(a.n + b.n) / max(kappa_a, kappa_b)
    else:
      raise ValueError(
          "Cannot merge i.i.d. shard"
          f" (n={a.n if a.effective_n is None else b.n}) with correlated shard"
          f" (n={b.n if a.effective_n is None else a.n})."
      )

    combined = list(a.top_32) + list(b.top_32)
    # Sort descending, keep top 32
    if combined:
      sorted_top = tuple(sorted(combined, reverse=True)[:32])
    else:
      sorted_top = ()
    return IndependentSketch(
        n=a.n + b.n,
        sum_s=float(a.sum_s + b.sum_s),
        sum_s2=float(a.sum_s2 + b.sum_s2),
        top_32=sorted_top,
        effective_n=merged_eff_n,
        schema_version=a.schema_version,
    )
  raise TypeError(f"Cannot merge {type(a)} and {type(b)}.")


def from_array(
    arr: Sequence[float] | np.ndarray,
    effective_n: float | None = None,
) -> IndependentSketch:
  """Constructs an IndependentSketch from a 1D sequence of non-negative summands.

  Non-finite or negative summands poison the sketch rather than being dropped:
  sum_s = sum_s2 = nan. n and top_32 are still computed from the input array.
  """
  s = np.asarray(arr, dtype=np.float64)
  n = len(s)
  if n == 0:
    return IndependentSketch(
        n=0,
        sum_s=0.0,
        sum_s2=0.0,
        top_32=(),
        effective_n=effective_n,
    )
  is_invalid = np.any(~np.isfinite(s)) or np.any(s < 0.0)
  if is_invalid:
    sum_s = float("nan")
    sum_s2 = float("nan")
  else:
    sum_s = float(np.sum(s))
    sum_s2 = float(np.sum(s**2))
  top_32 = tuple(float(x) for x in np.sort(s)[::-1][:32])
  return IndependentSketch(
      n=n,
      sum_s=sum_s,
      sum_s2=sum_s2,
      top_32=top_32,
      effective_n=effective_n,
  )


def from_data(
    data: Any,
    metric: str = "mse",
    effective_n: float | None = None,
) -> IndependentSketch | R2Sketch:
  """Constructs an IndependentSketch or R2Sketch from raw evaluation data.

  Note on r2 and effective_n asymmetry:
    Asking for an interval whose assumptions cannot be met is a statistical
    outcome and gets a Status (confidence_interval refuses with
    ASSUMPTION_REQUIRED); building a sketch that could never be valid is a
    programming error and gets an exception (ValueError).
  """
  m = _metrics.normalize_metric(metric)
  if m == "r2":
    if effective_n is not None:
      raise ValueError(
          "effective_n is not supported for r2: the two R^2 sequences (e^2 and"
          " the paired differences b) have different and unknown effective"
          " sample sizes, and the pairing itself changes the dependence"
          " structure."
      )
    s_a, s_b = _metrics.extract_r2_summands(data)
    return R2Sketch(
        sketch_a=from_array(s_a, effective_n=effective_n),
        sketch_b=from_array(s_b, effective_n=effective_n),
    )
  s = _metrics.extract_summands(data, m)
  return from_array(s, effective_n=effective_n)
