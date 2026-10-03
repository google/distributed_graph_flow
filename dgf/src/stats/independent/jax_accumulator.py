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

"""Accelerator-side accumulator for evaluation sketches, in JAX.

Builds the same summary as `sketch.from_data` inside a jitted training or
evaluation loop: `init`, then `update` per batch (with an optional padding
mask), `merge` or `all_reduce` across shards and devices, and finally
`to_sketch` on the host. The result is an ordinary `sketch.IndependentSketch` and is
passed to `interval.confidence_interval`.

Numerical and contract notes:

- The sketch is of the float32-rounded summands. Compensated summation keeps
  the relative error of the sums far below the interval width, but that
  rounding is outside Claim 1.
- `n` is limited to 2^31 - 1 per state; beyond that `to_sketch` poisons. Very
  large evaluations should convert per-host states with `to_sketch` and merge
  the resulting sketches on the host.
- Float32 overflow of the sum of squared summands poisons the sketch.
- Non-finite or negative values poison the sketch rather than being dropped:
  a poisoned sketch makes `confidence_interval` return ASSUMPTION_REQUIRED.
- Shards joined by `merge` or `all_reduce` inherit the merge contract of
  `sketch.merge`: they must be mutually uncorrelated and each shard's
  `effective_n` must come from Lemma H or a variance ratio (premise V_k).
"""

import math
from typing import Any, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from dgf.src.stats.independent import metrics as _metrics
from dgf.src.stats.independent import sketch as _sketch


class State(NamedTuple):
  """State of the JAX single-sequence accumulator.

  Fields:
    n: Count of valid entries (int32). Limited to 2^31 - 1.
    sum_s: Running sum of summands (dtype).
    sum_s_comp: Neumaier compensation for sum_s (dtype).
    sum_s2: Running sum of squared summands (dtype).
    sum_s2_comp: Neumaier compensation for sum_s2 (dtype).
    top: The 32 largest valid summands in descending order, padded with -inf;
      shape (32,).
    n_invalid: Count of invalid masked-in items encountered (int32).
  """

  n: jax.Array
  sum_s: jax.Array
  sum_s_comp: jax.Array
  sum_s2: jax.Array
  sum_s2_comp: jax.Array
  top: jax.Array
  n_invalid: jax.Array


class R2State(NamedTuple):
  """State of the JAX R^2 accumulator holding two single-sequence states."""

  a: State
  b: State


def init(dtype: Any = jnp.float32) -> State:
  """Initializes an empty accumulator State.

  Args:
    dtype: Accumulation floating point dtype (default jnp.float32).

  Returns:
    An initialized State.
  """
  return State(
      n=jnp.int32(0),
      sum_s=jnp.array(0.0, dtype=dtype),
      sum_s_comp=jnp.array(0.0, dtype=dtype),
      sum_s2=jnp.array(0.0, dtype=dtype),
      sum_s2_comp=jnp.array(0.0, dtype=dtype),
      top=jnp.full((32,), -jnp.inf, dtype=dtype),
      n_invalid=jnp.int32(0),
  )


def init_r2(dtype: Any = jnp.float32) -> R2State:
  """Initializes an empty R2State."""
  return R2State(a=init(dtype=dtype), b=init(dtype=dtype))


def summands(
    metric: str,
    labels: jax.Array,
    predictions: jax.Array,
) -> jax.Array:
  """Computes per-item non-negative summands s_i for standard metrics in JAX.

  Same semantics as `metrics.extract_summands` for mae, mse, rmse and accuracy
  given as (labels, predictions). Invalid items become NaN: a non-finite
  residual, or for accuracy a non-finite float label or prediction. `update`
  counts them as invalid and `to_sketch` poisons.

  Args:
    metric: 'mae', 'mse', 'rmse' or 'accuracy'.
    labels: Array of labels.
    predictions: Array of predictions, shaped like `labels`.

  Returns:
    Array of summands, shaped like `labels`.

  Raises:
    ValueError: For 'r2' (use `update_r2`) and for unknown metrics.
  """
  m = _metrics.normalize_metric(metric)
  if m == "r2":
    raise ValueError("For metric 'r2', use update_r2 instead of summands.")

  if m == "accuracy":
    res = (labels == predictions).astype(jnp.float32)
    if jnp.issubdtype(labels.dtype, jnp.floating):
      res = jnp.where(~jnp.isfinite(labels), jnp.nan, res)
    if jnp.issubdtype(predictions.dtype, jnp.floating):
      res = jnp.where(~jnp.isfinite(predictions), jnp.nan, res)
    return res

  e = predictions - labels
  e_valid = jnp.where(~jnp.isfinite(e), jnp.nan, e)
  if m == "mae":
    return jnp.abs(e_valid)
  elif m in ("mse", "rmse"):
    return e_valid**2
  else:
    raise ValueError(f"Unhandled metric {m}")


def _neumaier_add(
    sum_val: jax.Array,
    comp_val: jax.Array,
    x: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Neumaier compensated addition of scalar x to running sum and compensation."""
  t = sum_val + x
  c = jnp.where(
      jnp.abs(sum_val) >= jnp.abs(x),
      (sum_val - t) + x,
      (x - t) + sum_val,
  )
  return t, comp_val + c


def update(
    state: State,
    s: jax.Array,
    mask: jax.Array | None = None,
) -> State:
  """Adds a batch of summands to the state. Pure and jittable.

  Masked-out entries do not influence the result, whatever they contain
  (NaN, inf and negative padding are all safe). A masked-in entry that is
  non-finite or negative is counted in `n_invalid` and excluded from `n`, the
  sums and `top`; `to_sketch` then poisons.

  Args:
    state: The current state.
    s: Batch of summands, any shape.
    mask: Optional boolean array shaped like `s`. None means all entries are
      valid.

  Returns:
    The updated state.
  """
  s_arr = jnp.asarray(s, dtype=state.sum_s.dtype)
  s_flat = s_arr.reshape(-1)

  if mask is None:
    mask_flat = jnp.ones_like(s_flat, dtype=bool)
  else:
    mask_flat = jnp.asarray(mask, dtype=bool).reshape(-1)

  # Apply jnp.where before any arithmetic so NaNs/infs behind mask do not leak.
  s_masked = jnp.where(mask_flat, s_flat, 0.0)

  # Masked-in items that are non-finite or negative are invalid
  is_invalid = mask_flat & (~jnp.isfinite(s_masked) | (s_masked < 0.0))
  batch_n_invalid = jnp.sum(is_invalid.astype(jnp.int32))

  # Valid masked-in items
  is_valid = mask_flat & ~is_invalid
  batch_n = jnp.sum(is_valid.astype(jnp.int32))

  # Cleaned summands for summation
  s_clean = jnp.where(is_valid, s_masked, 0.0)
  batch_sum_s = jnp.sum(s_clean)
  batch_sum_s2 = jnp.sum(s_clean**2)

  # Neumaier summation with batch totals
  new_sum_s, new_comp_s = _neumaier_add(
      state.sum_s, state.sum_s_comp, batch_sum_s
  )
  new_sum_s2, new_comp_s2 = _neumaier_add(
      state.sum_s2, state.sum_s2_comp, batch_sum_s2
  )

  # Top-32 update: invalid replaced by -inf
  s_top_candidates = jnp.where(is_valid, s_masked, -jnp.inf)
  all_top = jnp.concatenate([state.top, s_top_candidates])
  new_top, _ = lax.top_k(all_top, 32)

  return State(
      n=state.n + batch_n,
      sum_s=new_sum_s,
      sum_s_comp=new_comp_s,
      sum_s2=new_sum_s2,
      sum_s2_comp=new_comp_s2,
      top=new_top,
      n_invalid=state.n_invalid + batch_n_invalid,
  )


def merge(a: State, b: State) -> State:
  """Associative and commutative merge of two States with compensated addition."""
  n_total = a.n + b.n
  n_invalid_total = a.n_invalid + b.n_invalid

  # Commutative Neumaier sum for sum_s
  s_max = jnp.maximum(a.sum_s, b.sum_s)
  s_min = jnp.minimum(a.sum_s, b.sum_s)
  t_s = s_max + s_min
  c_s = (s_max - t_s) + s_min
  new_sum_s = t_s
  new_comp_s = a.sum_s_comp + b.sum_s_comp + c_s

  # Commutative Neumaier sum for sum_s2
  s2_max = jnp.maximum(a.sum_s2, b.sum_s2)
  s2_min = jnp.minimum(a.sum_s2, b.sum_s2)
  t_s2 = s2_max + s2_min
  c_s2 = (s2_max - t_s2) + s2_min
  new_sum_s2 = t_s2
  new_comp_s2 = a.sum_s2_comp + b.sum_s2_comp + c_s2

  # Top-32 merge
  combined_top = jnp.concatenate([a.top, b.top])
  new_top, _ = lax.top_k(combined_top, 32)

  return State(
      n=n_total,
      sum_s=new_sum_s,
      sum_s_comp=new_comp_s,
      sum_s2=new_sum_s2,
      sum_s2_comp=new_comp_s2,
      top=new_top,
      n_invalid=n_invalid_total,
  )


def all_reduce(state: State, axis_name: str) -> State:
  """All-reduce across pseudo-devices under pmap/vmap."""
  n = lax.psum(state.n, axis_name)
  n_invalid = lax.psum(state.n_invalid, axis_name)
  sum_s = lax.psum(state.sum_s, axis_name)
  sum_s_comp = lax.psum(state.sum_s_comp, axis_name)
  sum_s2 = lax.psum(state.sum_s2, axis_name)
  sum_s2_comp = lax.psum(state.sum_s2_comp, axis_name)

  # Gather tops from all devices and take top 32
  all_tops = lax.all_gather(state.top, axis_name)
  top_flat = all_tops.reshape(-1)
  top, _ = lax.top_k(top_flat, 32)

  return State(
      n=n,
      sum_s=sum_s,
      sum_s_comp=sum_s_comp,
      sum_s2=sum_s2,
      sum_s2_comp=sum_s2_comp,
      top=top,
      n_invalid=n_invalid,
  )


def to_sketch(
    state: State,
    effective_n: float | None = None,
) -> _sketch.IndependentSketch:
  """Converts a State to an `IndependentSketch` on the host, in float64.

  The sketch is poisoned (sum_s = sum_s2 = nan) if any invalid entry was seen,
  if a sum is non-finite, or if `n` is negative (int32 wraparound).

  Args:
    state: The accumulated state, after any `merge` or `all_reduce`.
    effective_n: Declared effective sample size for correlated data, or None for
      i.i.d. data; see `IndependentSketch`.

  Returns:
    The equivalent `IndependentSketch`.
  """
  n = int(state.n)
  n_invalid = int(state.n_invalid)
  sum_s = float(state.sum_s) + float(state.sum_s_comp)
  sum_s2 = float(state.sum_s2) + float(state.sum_s2_comp)

  top_arr = np.asarray(state.top, dtype=np.float64)
  valid_top_count = min(max(0, n), 32)
  top_32 = tuple(float(x) for x in top_arr[:valid_top_count])

  # Poison condition
  is_poisoned = (
      n_invalid > 0
      or not math.isfinite(sum_s)
      or not math.isfinite(sum_s2)
      or n < 0
  )
  if is_poisoned:
    sum_s = float("nan")
    sum_s2 = float("nan")

  return _sketch.IndependentSketch(
      n=n,
      sum_s=sum_s,
      sum_s2=sum_s2,
      top_32=top_32,
      effective_n=effective_n,
  )


def update_r2(
    state: R2State,
    labels: jax.Array,
    predictions: jax.Array,
    mask: jax.Array | None = None,
) -> R2State:
  """Updates an R2State with a batch of (labels, predictions).

  `a` accumulates e^2 over the valid entries. `b` pairs the valid entries
  within the batch: they are compacted in order, positions (0, 1), (2, 3), ...
  are paired among the first 2 * floor(k / 2) of the k valid entries, and
  0.5 * (y_2 - y_1)^2 is accumulated. Each batch therefore behaves as one shard
  of an `sketch.R2Sketch`: an odd valid count drops its last entry, chosen by
  position, never by value.

  A non-finite label or prediction on a valid entry is invalid in `a`, and in
  `b` wherever it enters a pair.

  Args:
    state: The current state.
    labels: Batch of labels.
    predictions: Batch of predictions, shaped like `labels`.
    mask: Optional boolean array shaped like `labels`; False entries are ignored
      whatever they contain. None means all entries are valid.

  Returns:
    The updated state.
  """
  y_flat = jnp.asarray(labels, dtype=state.a.sum_s.dtype).reshape(-1)
  y_pred_flat = jnp.asarray(predictions, dtype=state.a.sum_s.dtype).reshape(-1)

  if mask is None:
    mask_flat = jnp.ones_like(y_flat, dtype=bool)
  else:
    mask_flat = jnp.asarray(mask, dtype=bool).reshape(-1)

  # Masked-out entries must not influence anything
  y_m = jnp.where(mask_flat, y_flat, 0.0)
  y_pred_m = jnp.where(mask_flat, y_pred_flat, 0.0)

  # Sequence A: squared errors
  e = y_pred_m - y_m
  s_a = e**2
  # If y or y_pred is non-finite on a masked-in entry, poison s_a
  non_finite_entry = mask_flat & (~jnp.isfinite(y_m) | ~jnp.isfinite(y_pred_m))
  s_a = jnp.where(non_finite_entry, jnp.nan, s_a)
  new_a = update(state.a, s_a, mask=mask_flat)

  # Sequence B: within-batch disjoint pairs
  # Compact masked-in entries in order with stable argsort of ~mask
  b_len = y_flat.shape[0]
  sort_keys = (~mask_flat).astype(jnp.int32)
  perm = jnp.argsort(sort_keys, stable=True)

  y_compact = y_m[perm]
  mask_compact = mask_flat[perm]

  # Number of valid entries in batch
  k_valid = jnp.sum(mask_flat.astype(jnp.int32))
  num_pairs = k_valid // 2

  p_max = b_len // 2
  first_idx = 2 * jnp.arange(p_max)
  second_idx = first_idx + 1

  y_first = y_compact[first_idx]
  y_second = y_compact[second_idx]
  mask_first = mask_compact[first_idx]
  mask_second = mask_compact[second_idx]

  pair_mask = (jnp.arange(p_max) < num_pairs) & mask_first & mask_second

  # Difference and squared half-difference
  diff = y_second - y_first
  s_b = 0.5 * (diff**2)
  pair_non_finite = pair_mask & (
      ~jnp.isfinite(y_first) | ~jnp.isfinite(y_second)
  )
  s_b = jnp.where(pair_non_finite, jnp.nan, s_b)

  new_b = update(state.b, s_b, mask=pair_mask)

  return R2State(a=new_a, b=new_b)


def merge_r2(a: R2State, b: R2State) -> R2State:
  """Merges two R2States."""
  return R2State(a=merge(a.a, b.a), b=merge(a.b, b.b))


def all_reduce_r2(state: R2State, axis_name: str) -> R2State:
  """All-reduce of R2State across devices."""
  return R2State(
      a=all_reduce(state.a, axis_name),
      b=all_reduce(state.b, axis_name),
  )


def to_r2_sketch(
    state: R2State,
    effective_n: float | None = None,
) -> _sketch.R2Sketch:
  """Converts R2State on host to a sketch.R2Sketch."""
  sk_a = to_sketch(state.a, effective_n=effective_n)
  sk_b = to_sketch(state.b, effective_n=effective_n)
  return _sketch.R2Sketch(sketch_a=sk_a, sketch_b=sk_b)
