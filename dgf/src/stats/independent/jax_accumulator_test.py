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

"""Tests for the JAX accumulator."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from dgf.src.stats.independent import interval
from dgf.src.stats.independent import jax_accumulator
from dgf.src.stats.independent import sketch


class JaxAccumulatorTest(parameterized.TestCase):

  def test_empty_accumulator(self):
    """Test 6: to_sketch(init()) equals sketch.from_array([])."""
    sk_jax = jax_accumulator.to_sketch(jax_accumulator.init())
    sk_np = sketch.from_array([])
    self.assertEqual(sk_jax.n, sk_np.n)
    self.assertEqual(sk_jax.sum_s, sk_np.sum_s)
    self.assertEqual(sk_jax.sum_s2, sk_np.sum_s2)
    self.assertEqual(sk_jax.top_32, sk_np.top_32)

  def test_masking_garbage_ignored(self):
    """Test 4: Padding filled with NaN, inf, -inf, -1.0, 1e38 behind a false mask gives exactly clean state."""
    clean_s = jnp.array([1.5, 2.5, 3.5, 4.5], dtype=jnp.float32)
    st_clean = jax_accumulator.update(jax_accumulator.init(), clean_s)

    garbage = [jnp.nan, jnp.inf, -jnp.inf, -1.0, 1e38]
    for g_val in garbage:
      padded_s = jnp.array(
          [1.5, 2.5, g_val, 3.5, g_val, 4.5], dtype=jnp.float32
      )
      mask = jnp.array([True, True, False, True, False, True], dtype=bool)
      st_padded = jax_accumulator.update(
          jax_accumulator.init(), padded_s, mask=mask
      )

      self.assertEqual(int(st_clean.n), int(st_padded.n))
      self.assertEqual(int(st_padded.n_invalid), 0)
      self.assertAlmostEqual(
          float(st_clean.sum_s), float(st_padded.sum_s), places=6
      )
      self.assertAlmostEqual(
          float(st_clean.sum_s2), float(st_padded.sum_s2), places=6
      )
      np.testing.assert_allclose(
          np.asarray(st_clean.top), np.asarray(st_padded.top)
      )

  def test_invalid_values_poison(self):
    """Test 5: One unmasked NaN, inf, or -1.0 gives n_invalid = 1 and ASSUMPTION_REQUIRED."""
    for bad_val in [jnp.nan, jnp.inf, -1.0]:
      st = jax_accumulator.init()
      s = jnp.array([1.0, 2.0, bad_val, 4.0], dtype=jnp.float32)
      st = jax_accumulator.update(st, s)
      self.assertEqual(int(st.n_invalid), 1)
      self.assertEqual(int(st.n), 3)

      sk = jax_accumulator.to_sketch(st)
      self.assertTrue(math.isnan(sk.sum_s))
      self.assertTrue(math.isnan(sk.sum_s2))
      res = interval.confidence_interval(sk, metric="mse")
      self.assertEqual(res.status, interval.Status.ASSUMPTION_REQUIRED)

    # Accuracy with a NaN float prediction via summands
    labels = jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    preds = jnp.array([1.0, 0.0, jnp.nan, 0.0], dtype=jnp.float32)
    s_acc = jax_accumulator.summands("accuracy", labels, preds)
    st_acc = jax_accumulator.update(jax_accumulator.init(), s_acc)
    self.assertEqual(int(st_acc.n_invalid), 1)
    sk_acc = jax_accumulator.to_sketch(st_acc)
    self.assertTrue(math.isnan(sk_acc.sum_s))
    res_acc = interval.confidence_interval(sk_acc, metric="accuracy")
    self.assertEqual(res_acc.status, interval.Status.ASSUMPTION_REQUIRED)

  def test_compensation_works(self):
    """Test 8: In float32, add 1.0 to initial sum_s = 1e8 ten thousand times via lax.scan.

    A naive float32 sum stays at 1e8 because 1e8 + 1.0 rounds down to 1e8 in
    IEEE 754 float32,
    which is why the Neumaier compensation exists.
    """
    initial_st = jax_accumulator.State(
        n=jnp.int32(0),
        sum_s=jnp.array(1e8, dtype=jnp.float32),
        sum_s_comp=jnp.array(0.0, dtype=jnp.float32),
        sum_s2=jnp.array(0.0, dtype=jnp.float32),
        sum_s2_comp=jnp.array(0.0, dtype=jnp.float32),
        top=jnp.full((32,), -jnp.inf, dtype=jnp.float32),
        n_invalid=jnp.int32(0),
    )

    one_batch = jnp.array([1.0], dtype=jnp.float32)

    def scan_fn(carry_st, _):
      next_st = jax_accumulator.update(carry_st, one_batch)
      return next_st, None

    final_st, _ = lax.scan(scan_fn, initial_st, None, length=10000)
    sk = jax_accumulator.to_sketch(final_st)

    # Naive float32 addition stays at 1e8.
    naive_sum = float(jnp.float32(1e8))
    for _ in range(10000):
      naive_sum = float(jnp.float32(naive_sum + 1.0))
    self.assertEqual(naive_sum, 1e8)

    # Neumaier compensated addition recovers 1e8 + 1e4 within 1.0
    expected = 1e8 + 1e4
    self.assertLessEqual(abs(sk.sum_s - expected), 1.0)

  def test_merge_associativity_and_commutativity(self):
    """Test 7: Associativity and commutativity on random states within rounding."""
    rng = np.random.default_rng(42)
    s1 = jnp.array(rng.uniform(0.1, 10.0, size=50), dtype=jnp.float32)
    s2 = jnp.array(rng.uniform(0.1, 10.0, size=75), dtype=jnp.float32)
    s3 = jnp.array(rng.uniform(0.1, 10.0, size=60), dtype=jnp.float32)

    st1 = jax_accumulator.update(jax_accumulator.init(), s1)
    st2 = jax_accumulator.update(jax_accumulator.init(), s2)
    st3 = jax_accumulator.update(jax_accumulator.init(), s3)

    # Commutativity: merge(st1, st2) == merge(st2, st1)
    m12 = jax_accumulator.merge(st1, st2)
    m21 = jax_accumulator.merge(st2, st1)
    self.assertEqual(int(m12.n), int(m21.n))
    self.assertAlmostEqual(
        float(m12.sum_s + m12.sum_s_comp),
        float(m21.sum_s + m21.sum_s_comp),
        places=5,
    )
    np.testing.assert_allclose(np.asarray(m12.top), np.asarray(m21.top))

    # Associativity: merge(merge(st1, st2), st3) == merge(st1, merge(st2, st3))
    m_12_3 = jax_accumulator.merge(m12, st3)
    m_1_23 = jax_accumulator.merge(st1, jax_accumulator.merge(st2, st3))
    self.assertEqual(int(m_12_3.n), int(m_1_23.n))
    self.assertAlmostEqual(
        float(m_12_3.sum_s + m_12_3.sum_s_comp),
        float(m_1_23.sum_s + m_1_23.sum_s_comp),
        delta=1e-4 * float(m_12_3.sum_s),
    )
    np.testing.assert_allclose(np.asarray(m_12_3.top), np.asarray(m_1_23.top))

  def test_transforms_jit_vmap_scan(self):
    """Test 2: jit, vmap over independent states, and lax.scan match Python loop."""
    rng = np.random.default_rng(101)
    batches = [
        jnp.array(rng.uniform(0.1, 5.0, size=20), dtype=jnp.float32)
        for _ in range(5)
    ]

    # Python loop
    st_loop = jax_accumulator.init()
    for b in batches:
      st_loop = jax_accumulator.update(st_loop, b)

    # JIT-compiled loop
    @jax.jit
    def run_jit(b_list):
      st = jax_accumulator.init()
      for b in b_list:
        st = jax_accumulator.update(st, b)
      return st

    st_jit = run_jit(batches)
    self.assertEqual(int(st_loop.n), int(st_jit.n))
    np.testing.assert_allclose(np.asarray(st_loop.top), np.asarray(st_jit.top))

    # lax.scan
    batches_stacked = jnp.stack(batches)

    @jax.jit
    def scan_update(carry, b):
      return jax_accumulator.update(carry, b), None

    st_scan, _ = lax.scan(scan_update, jax_accumulator.init(), batches_stacked)
    self.assertEqual(int(st_loop.n), int(st_scan.n))
    np.testing.assert_allclose(np.asarray(st_loop.top), np.asarray(st_scan.top))

    # vmap over batch of states
    init_states = jax.vmap(lambda _: jax_accumulator.init())(jnp.arange(5))
    vmapped_update = jax.vmap(jax_accumulator.update)
    st_vmapped = vmapped_update(init_states, batches_stacked)
    self.assertEqual(list(np.asarray(st_vmapped.n)), [20] * 5)

  def test_all_reduce_matches_merge(self):
    """Test 3: all_reduce under jax.vmap with 4 pseudo-devices equals merge folded over 4 states."""
    rng = np.random.default_rng(202)
    d4_data = [
        jnp.array(rng.uniform(0.1, 5.0, size=30), dtype=jnp.float32)
        for _ in range(4)
    ]
    st_list = [
        jax_accumulator.update(jax_accumulator.init(), d) for d in d4_data
    ]

    # Folded merge
    st_merged = st_list[0]
    for s in st_list[1:]:
      st_merged = jax_accumulator.merge(st_merged, s)

    # all_reduce via vmap(axis_name="d")
    st_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *st_list)

    @jax.jit
    def run_all_reduce(s_batch):
      return jax.vmap(
          lambda s: jax_accumulator.all_reduce(s, axis_name="d"),
          axis_name="d",
      )(s_batch)

    st_reduced_batch = run_all_reduce(st_stacked)
    # Check that each device has the reduced state equal to st_merged
    for dev in range(4):
      self.assertEqual(int(st_reduced_batch.n[dev]), int(st_merged.n))
      self.assertAlmostEqual(
          float(st_reduced_batch.sum_s[dev] + st_reduced_batch.sum_s_comp[dev]),
          float(st_merged.sum_s + st_merged.sum_s_comp),
          delta=1e-4 * float(st_merged.sum_s),
      )
      np.testing.assert_allclose(
          np.asarray(st_reduced_batch.top[dev]),
          np.asarray(st_merged.top),
      )

  def test_r2_accumulator(self):
    """Test 9: R2State pairs valid entries; batch of 7 drops odd element; compare with from_data."""
    rng = np.random.default_rng(303)
    y_batch1 = jnp.array(rng.normal(size=7), dtype=jnp.float32)
    pred_batch1 = y_batch1 + jnp.array(
        rng.normal(scale=0.5, size=7), dtype=jnp.float32
    )

    st = jax_accumulator.init_r2()
    st = jax_accumulator.update_r2(st, y_batch1, pred_batch1)
    # In batch of 7, 7 // 2 = 3 pairs, so b.n must be 3
    self.assertEqual(int(st.b.n), 3)
    self.assertEqual(int(st.a.n), 7)

    # 10 batches of 7 -> sum floor(7/2) = 30 pairs
    st_10 = jax_accumulator.init_r2()
    for _ in range(10):
      yb = jnp.array(rng.normal(size=7), dtype=jnp.float32)
      pb = yb + jnp.array(rng.normal(scale=0.5, size=7), dtype=jnp.float32)
      st_10 = jax_accumulator.update_r2(st_10, yb, pb)
    self.assertEqual(int(st_10.b.n), 30)

    # Even batch sizes (no dropped pairs between batches) without padding: compare endpoints to from_data
    n_even = 3000
    y_even = rng.normal(size=n_even).astype(np.float32)
    pred_even = y_even + rng.normal(scale=0.3, size=n_even).astype(np.float32)

    st_even = jax_accumulator.init_r2()
    st_even = jax_accumulator.update_r2(
        st_even, jnp.asarray(y_even), jnp.asarray(pred_even)
    )
    r2_sk = jax_accumulator.to_r2_sketch(st_even)

    # numpy from_data on same float32 upcast to float64
    r2_np = sketch.from_data(
        (y_even.astype(np.float64), pred_even.astype(np.float64)), metric="r2"
    )

    res_jax = interval.confidence_interval(r2_sk, metric="r2", level=0.95)
    res_np = interval.confidence_interval(r2_np, metric="r2", level=0.95)

    self.assertEqual(res_jax.status, res_np.status)
    w_np = res_np.high - res_np.low
    self.assertLessEqual(abs(res_jax.low - res_np.low), 1e-5 * w_np)
    self.assertLessEqual(abs(res_jax.high - res_np.high), 1e-5 * w_np)

  def test_overflow_and_wraparound_poisons(self):
    """Test 10: n wraparound (< 0) or float32 inf in sum_s2 poisons."""
    # Negative n (wraparound)
    st_wrap = jax_accumulator.State(
        n=jnp.int32(-5),
        sum_s=jnp.array(10.0, dtype=jnp.float32),
        sum_s_comp=jnp.array(0.0, dtype=jnp.float32),
        sum_s2=jnp.array(100.0, dtype=jnp.float32),
        sum_s2_comp=jnp.array(0.0, dtype=jnp.float32),
        top=jnp.full((32,), 1.0, dtype=jnp.float32),
        n_invalid=jnp.int32(0),
    )
    sk_wrap = jax_accumulator.to_sketch(st_wrap)
    self.assertTrue(math.isnan(sk_wrap.sum_s))
    self.assertTrue(math.isnan(sk_wrap.sum_s2))

    # Float32 overflow to inf in sum_s2
    st_inf = jax_accumulator.State(
        n=jnp.int32(100),
        sum_s=jnp.array(10.0, dtype=jnp.float32),
        sum_s_comp=jnp.array(0.0, dtype=jnp.float32),
        sum_s2=jnp.array(jnp.inf, dtype=jnp.float32),
        sum_s2_comp=jnp.array(0.0, dtype=jnp.float32),
        top=jnp.full((32,), 1.0, dtype=jnp.float32),
        n_invalid=jnp.int32(0),
    )
    sk_inf = jax_accumulator.to_sketch(st_inf)
    self.assertTrue(math.isnan(sk_inf.sum_s))
    self.assertTrue(math.isnan(sk_inf.sum_s2))

  def test_equivalence_across_metrics_and_sizes(self):
    """Test 1: Equivalence against from_data on same float32 data across 20 seeds and sizes."""
    # Test over 20 seeds with n in {1, 31, 32, 33, 1000, 100000}
    test_ns = [1, 31, 32, 33, 1000, 100000]
    metrics_to_test = ["mae", "mse", "rmse", "accuracy"]

    for seed in range(20):
      rng = np.random.default_rng(seed + 1000)
      n = test_ns[seed % len(test_ns)]
      m = metrics_to_test[seed % len(metrics_to_test)]
      batch_size = int(rng.integers(7, 4096))

      if m == "accuracy":
        labels = rng.integers(0, 10, size=n, dtype=np.int32)
        preds = rng.integers(0, 10, size=n, dtype=np.int32)
        # JAX summands
        s_jax = jax_accumulator.summands(
            "accuracy", jnp.asarray(labels), jnp.asarray(preds)
        )
        # numpy raw data tuple
        data_np = (labels, preds)
      else:
        labels = rng.normal(size=n).astype(np.float32)
        preds = labels + rng.normal(scale=0.5, size=n).astype(np.float32)
        s_jax = jax_accumulator.summands(
            m, jnp.asarray(labels), jnp.asarray(preds)
        )
        data_np = (labels.astype(np.float64), preds.astype(np.float64))

      # Accumulate in batches on JAX
      st = jax_accumulator.init()
      num_batches = int(math.ceil(n / float(batch_size)))
      for b_idx in range(num_batches):
        start = b_idx * batch_size
        end = min(n, (b_idx + 1) * batch_size)
        st = jax_accumulator.update(st, s_jax[start:end])

      sk_jax = jax_accumulator.to_sketch(st)
      sk_np = sketch.from_data(data_np, metric=m)
      assert isinstance(sk_np, sketch.IndependentSketch)

      # 1. n is equal
      self.assertEqual(sk_jax.n, sk_np.n)

      # 2. sums agree within 1e-5 relative
      if sk_np.sum_s > 0:
        self.assertLessEqual(
            abs(sk_jax.sum_s - sk_np.sum_s) / sk_np.sum_s,
            1e-5,
            f"sum_s failed on seed={seed} m={m} n={n}",
        )
      if sk_np.sum_s2 > 0:
        self.assertLessEqual(
            abs(sk_jax.sum_s2 - sk_np.sum_s2) / sk_np.sum_s2,
            1e-5,
            f"sum_s2 failed on seed={seed} m={m} n={n}",
        )

      # 3. top_32 agrees within 1e-6 relative, elementwise
      self.assertEqual(len(sk_jax.top_32), len(sk_np.top_32))
      for x_jax, x_np in zip(sk_jax.top_32, sk_np.top_32):
        if abs(x_np) > 1e-12:
          self.assertLessEqual(
              abs(x_jax - x_np) / abs(x_np),
              1e-6,
              f"top_32 element failed on seed={seed} m={m} n={n}",
          )
        else:
          self.assertLessEqual(abs(x_jax - x_np), 1e-6)

      # 4. confidence_interval status is same, endpoints agree within 1e-4 of numpy width
      res_jax = interval.confidence_interval(sk_jax, metric=m, level=0.95)
      res_np = interval.confidence_interval(sk_np, metric=m, level=0.95)
      self.assertEqual(res_jax.status, res_np.status)

      if res_np.status != interval.Status.ASSUMPTION_REQUIRED:
        w_np = res_np.high - res_np.low
        if math.isfinite(w_np) and w_np > 0:
          self.assertLessEqual(
              abs(res_jax.low - res_np.low),
              1e-4 * w_np,
              f"low endpoint failed on seed={seed} m={m} n={n}",
          )
          self.assertLessEqual(
              abs(res_jax.high - res_np.high),
              1e-4 * w_np,
              f"high endpoint failed on seed={seed} m={m} n={n}",
          )


if __name__ == "__main__":
  absltest.main()
