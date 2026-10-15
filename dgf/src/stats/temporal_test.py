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

"""Unit tests for temporal confidence intervals and refutation checks."""

import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import scipy.signal

from dgf.src.stats import cluster_sketch
from dgf.src.stats import refutation
from dgf.src.stats import temporal
from dgf.src.stats.independent import interval

# Set DGF_STATS_LONG_TESTS=1 (blaze test --test_env=...) for the full-size runs.
_LONG = os.environ.get("DGF_STATS_LONG_TESTS") == "1"


def _generate_ar1_series(
    total_len: int,
    phi: float,
    rng: np.random.Generator,
) -> np.ndarray:
  """Generates stationary AR(1) Gaussian sequence of length total_len."""
  if abs(phi) < 1e-12:
    return rng.standard_normal(total_len)
  innovations = rng.standard_normal(total_len)
  innovations[1:] *= math.sqrt(max(0.0, 1.0 - phi**2))
  u = scipy.signal.lfilter([1.0], [1.0, -phi], innovations)
  return u.astype(np.float64)


class TemporalTest(parameterized.TestCase):

  def test_ordering_and_stable_ties(self):
    """Tests sorting by timestamps and stable ordering on tied timestamps."""
    n = 2000
    g = 100
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 1.0, size=n)
    timestamps = rng.uniform(0.0, 100.0, size=n)

    # Shuffled input with timestamps matches sorted input without timestamps
    perm = rng.permutation(n)
    res_perm = temporal.temporal_interval(
        residuals[perm], timestamps=timestamps[perm], num_windows=g, m=2.0
    )
    sorted_order = np.argsort(timestamps, kind="stable")
    res_sorted = temporal.temporal_interval(
        residuals[sorted_order], num_windows=g, m=2.0
    )
    self.assertEqual(res_perm.status, interval.Status.UNREFUTED)
    self.assertEqual(res_perm.low, res_sorted.low)
    self.assertEqual(res_perm.high, res_sorted.high)
    self.assertEqual(res_perm.status, res_sorted.status)

    # Ties: two items with equal timestamps keep input order, affecting window totals
    # n=4, G=2: window 0 has items 0, 1; window 1 has items 2, 3.
    # Tied timestamp 2.0 at indices 1 and 2.
    ts = np.array([1.0, 2.0, 2.0, 3.0])
    data_a = np.array(
        [10.0, 0.0, 5.0, 0.0]
    )  # window 0: [10, 0], window 1: [5, 0]
    data_b = np.array(
        [10.0, 5.0, 0.0, 0.0]
    )  # window 0: [10, 5], window 1: [0, 0]

    res_a = temporal.temporal_interval(data_a, timestamps=ts, num_windows=2)
    res_b = temporal.temporal_interval(data_b, timestamps=ts, num_windows=2)
    self.assertNotEqual(res_a.m_observed, res_b.m_observed)

  def test_uneven_windows(self):
    """Tests window partitioning with n = 1003 and G = 10."""
    n = 1003
    g = 10
    e = np.ones(n, dtype=np.float64)
    res = temporal.temporal_interval(e, num_windows=g)
    self.assertEqual(res.num_windows, g)
    self.assertAlmostEqual(res.mean_window_size, 100.3)

    # Check window size distribution
    indices = np.arange(n, dtype=np.int64)
    w_ids = (indices * g) // n
    counts = np.bincount(w_ids)
    self.assertEqual(len(counts), 10)
    for c in counts:
      self.assertIn(c, (100, 101))

  def test_size_and_power(self):
    """Tests empirical size and power of the kappa = 1 check through the API."""
    n = 20000 if _LONG else 4000
    g = 500 if _LONG else 100

    # Size test: independent Gaussian residuals, seeds -> rejection in nominal bounds
    rejections_size = 0
    seeds_size = 200 if _LONG else 30
    for seed in range(seeds_size):
      rng = np.random.default_rng(100000 + seed)
      e = rng.normal(0.0, 1.0, size=n)
      res = temporal.temporal_interval(e, num_windows=g, kappa=1.0)
      if res.kappa_check.outcome is refutation.Refutation.REFUTED:
        rejections_size += 1
    size_rate = rejections_size / float(seeds_size)
    min_size = (
        0.02
        if _LONG
        else max(0.0, 0.05 - 3.0 * math.sqrt(0.05 * 0.95 / float(seeds_size)))
    )
    max_size = (
        0.09
        if _LONG
        else min(1.0, 0.05 + 3.0 * math.sqrt(0.05 * 0.95 / float(seeds_size)))
    )
    self.assertGreaterEqual(size_rate, min_size)
    self.assertLessEqual(size_rate, max_size)

    # Power test: AR(1) with L = 100, seeds -> rejection >= 0.95 (or 0.90 in short mode)
    phi = float(np.exp(-1.0 / 100.0))
    rejections_power = 0
    seeds_power = 100 if _LONG else 20
    for seed in range(seeds_power):
      rng = np.random.default_rng(200000 + seed)
      u = _generate_ar1_series(n, phi, rng)
      res = temporal.temporal_interval(u, num_windows=g, kappa=1.0)
      if res.kappa_check.outcome is refutation.Refutation.REFUTED:
        rejections_power += 1
    power_rate = rejections_power / float(seeds_power)
    min_power = 0.95 if _LONG else 0.90
    self.assertGreaterEqual(power_rate, min_power)

  def test_invalid_values_refusal(self):
    """Tests that non-finite values in data or timestamps refuse without raising."""
    n = 100
    g = 5

    # NaN in data
    e_nan = np.ones(n)
    e_nan[10] = np.nan
    res = temporal.temporal_interval(e_nan, num_windows=g)
    self.assertEqual(res.status, interval.Status.ASSUMPTION_REQUIRED)
    self.assertTrue(math.isnan(res.low))
    self.assertTrue(math.isnan(res.high))
    self.assertIsNone(res.level)
    self.assertEqual(res.kappa_check.outcome, refutation.Refutation.UNTESTABLE)
    self.assertTrue(math.isnan(res.m_observed))
    self.assertIsNone(res.diagnostic)
    self.assertIn("1 invalid value", res.message)

    # +inf in data
    e_inf = np.ones(n)
    e_inf[20] = np.inf
    res = temporal.temporal_interval(e_inf, num_windows=g)
    self.assertEqual(res.status, interval.Status.ASSUMPTION_REQUIRED)
    self.assertTrue(math.isnan(res.low))
    self.assertIsNone(res.diagnostic)

    # -inf in data
    e_ninf = np.ones(n)
    e_ninf[30] = -np.inf
    res = temporal.temporal_interval(e_ninf, num_windows=g)
    self.assertEqual(res.status, interval.Status.ASSUMPTION_REQUIRED)
    self.assertTrue(math.isnan(res.low))
    self.assertIsNone(res.diagnostic)

    # NaN in timestamps
    ts_nan = np.arange(n, dtype=np.float64)
    ts_nan[5] = np.nan
    res = temporal.temporal_interval(
        np.ones(n), timestamps=ts_nan, num_windows=g
    )
    self.assertEqual(res.status, interval.Status.ASSUMPTION_REQUIRED)
    self.assertTrue(math.isnan(res.low))
    self.assertEqual(res.kappa_check.outcome, refutation.Refutation.UNTESTABLE)
    self.assertIsNone(res.diagnostic)
    self.assertIn("1 invalid non-finite value", res.message)

  def test_untestable_condition(self):
    """Tests that kappa = 2 with G = 20 returns UNTESTABLE and not refuted."""
    n = 1000
    e = np.ones(n)
    res = temporal.temporal_interval(e, num_windows=20, kappa=2.0)
    self.assertEqual(res.kappa_check.outcome, refutation.Refutation.UNTESTABLE)
    self.assertFalse(res.refuted)

  def test_value_error_validation(self):
    """Tests ValueError checks in temporal_interval."""
    # Tuple data
    with self.assertRaisesRegex(ValueError, "Tuples of .* are not accepted"):
      temporal.temporal_interval((np.ones(10), np.ones(10)), num_windows=2)

    # Data not 1-D
    with self.assertRaisesRegex(ValueError, "data must be 1-D"):
      temporal.temporal_interval(np.ones((10, 2)), num_windows=2)

    # Timestamps not 1-D
    with self.assertRaisesRegex(ValueError, "timestamps must be 1-D"):
      temporal.temporal_interval(
          np.ones(10), timestamps=np.ones((10, 2)), num_windows=2
      )

    # Length mismatch
    with self.assertRaisesRegex(ValueError, "Length mismatch"):
      temporal.temporal_interval(
          np.ones(10), timestamps=np.ones(5), num_windows=2
      )

    # num_windows not int
    with self.assertRaisesRegex(ValueError, "num_windows must be an integer"):
      temporal.temporal_interval(np.ones(10), num_windows=2.5)  # pytype: disable=wrong-arg-types
    with self.assertRaisesRegex(ValueError, "num_windows must be an integer"):
      temporal.temporal_interval(np.ones(10), num_windows=True)  # pytype: disable=wrong-arg-types

    # num_windows < 2
    with self.assertRaisesRegex(ValueError, "num_windows must be in"):
      temporal.temporal_interval(np.ones(10), num_windows=1)

    # num_windows > n
    with self.assertRaisesRegex(ValueError, "num_windows must be in"):
      temporal.temporal_interval(np.ones(10), num_windows=11)

    # kappa non-finite or < 1
    with self.assertRaisesRegex(ValueError, "kappa must be finite and >= 1.0"):
      temporal.temporal_interval(np.ones(10), num_windows=2, kappa=float("nan"))
    with self.assertRaisesRegex(ValueError, "kappa must be finite and >= 1.0"):
      temporal.temporal_interval(np.ones(10), num_windows=2, kappa=0.5)

    # level, c_target and m raise even when the data contain invalid values
    # (which would otherwise produce a refusal rather than an exception).
    e_nan = np.ones(10)
    e_nan[3] = np.nan
    for data in (np.ones(10), e_nan):
      with self.assertRaisesRegex(ValueError, "level must be in"):
        temporal.temporal_interval(data, num_windows=2, level=1.0)
      with self.assertRaisesRegex(ValueError, "level must be in"):
        temporal.temporal_interval(data, num_windows=2, level=float("nan"))
      with self.assertRaisesRegex(ValueError, "c_target must be in"):
        temporal.temporal_interval(data, num_windows=2, c_target=0.0)
      with self.assertRaisesRegex(ValueError, "m must be finite and >= 1.0"):
        temporal.temporal_interval(data, num_windows=2, m=0.5)
      with self.assertRaisesRegex(ValueError, "m must be finite and >= 1.0"):
        temporal.temporal_interval(data, num_windows=2, m=float("nan"))
      with self.assertRaisesRegex(ValueError, "m must be finite and >= 1.0"):
        temporal.temporal_interval(data, num_windows=2, m=float("inf"))

  def test_refuted_property(self):
    """Tests that refuted is True if either M check or kappa check refutes."""
    n = 2000
    g = 100
    rng = np.random.default_rng(42)

    # 1. TAIL_UNRESOLVED with NOT_REFUTED kappa check (declare tiny m = 1.0001)
    e = rng.normal(0.0, 1.0, size=n)
    res_m = temporal.temporal_interval(e, num_windows=g, kappa=10.0, m=1.0001)
    self.assertEqual(res_m.status, interval.Status.TAIL_UNRESOLVED)
    self.assertEqual(
        res_m.kappa_check.outcome, refutation.Refutation.NOT_REFUTED
    )
    self.assertTrue(res_m.refuted)

    # 2. UNREFUTED M check with REFUTED kappa check (AR(1), L = 100, kappa = 1)
    phi = float(np.exp(-1.0 / 100.0))
    u = _generate_ar1_series(20000, phi, rng)
    res_k = temporal.temporal_interval(u, num_windows=1000, kappa=1.0)
    self.assertEqual(res_k.status, interval.Status.UNREFUTED)
    self.assertEqual(res_k.kappa_check.outcome, refutation.Refutation.REFUTED)
    self.assertTrue(res_k.refuted)

  def test_exact_equality_raw_and_shard(self):
    """Verifies exact equality between temporal_interval and temporal_interval_from_shard."""
    for n in (10007, 20000):
      for g in (7, 100):
        for metric in ("mse", "mae", "accuracy"):
          for kappa in (1.0, 2.0):
            for m in (None, 1.05):
              rng = np.random.default_rng(n + g)
              if metric == "accuracy":
                data = (rng.random(n) < 0.85).astype(np.float64)
              else:
                data = rng.standard_normal(n)
              indices = np.arange(n, dtype=np.int64)
              window_ids = (indices * g) // n
              t_raw = temporal.temporal_interval(
                  data,
                  num_windows=g,
                  metric=metric,
                  level=0.95,
                  m=m,
                  kappa=kappa,
              )
              shard = cluster_sketch.PartialClusterShard.from_data(
                  data, window_ids, metric
              )
              t_shard = temporal.temporal_interval_from_shard(
                  shard, metric=metric, level=0.95, m=m, kappa=kappa
              )
              if math.isnan(t_raw.low):
                self.assertTrue(math.isnan(t_shard.low))
                self.assertTrue(math.isnan(t_shard.high))
              else:
                self.assertEqual(t_raw.low, t_shard.low)
                self.assertEqual(t_raw.high, t_shard.high)
              self.assertEqual(t_raw.status, t_shard.status)
              self.assertEqual(
                  t_raw.kappa_check.outcome, t_shard.kappa_check.outcome
              )
              self.assertEqual(
                  t_raw.kappa_check.statistic, t_shard.kappa_check.statistic
              )
              self.assertEqual(
                  t_raw.kappa_check.threshold, t_shard.kappa_check.threshold
              )
              self.assertEqual(t_raw.m_declared, t_shard.m_declared)
              self.assertEqual(t_raw.count_energy, t_shard.count_energy)
              self.assertEqual(t_raw.count_h, t_shard.count_h)
              self.assertEqual(t_raw.drift.outcome, t_shard.drift.outcome)
              if math.isnan(t_raw.drift.first_half[0]):
                self.assertTrue(math.isnan(t_shard.drift.first_half[0]))
              else:
                self.assertEqual(
                    t_raw.drift.first_half, t_shard.drift.first_half
                )
              if math.isnan(t_raw.drift.second_half[0]):
                self.assertTrue(math.isnan(t_shard.drift.second_half[0]))
              else:
                self.assertEqual(
                    t_raw.drift.second_half, t_shard.drift.second_half
                )

  def test_count_guard(self):
    """Verifies count heterogeneity guard under equal, near-equal, and moderate window counts."""
    g = 100
    ids = np.arange(g, dtype=np.int64)

    # Case A: MSE on N(0, 1) residuals, m_bar = 1000, alternating m_bar + 3, m_bar - 3
    counts_a = np.array(
        [1003 if i % 2 == 0 else 997 for i in range(g)], dtype=np.int64
    )
    rng_a = np.random.default_rng(42)
    totals_a = np.array(
        [float(np.sum(rng_a.standard_normal(c) ** 2)) for c in counts_a]
    )
    shard_a = cluster_sketch.PartialClusterShard(ids, totals_a, counts_a)
    res_a = temporal.temporal_interval_from_shard(
        shard_a, metric="mse", level=0.95, m=1.05
    )
    self.assertIsNotNone(res_a.count_energy)
    assert res_a.count_energy is not None
    self.assertAlmostEqual(res_a.count_energy, 0.005, delta=0.004)
    self.assertIsNotNone(res_a.count_h)
    assert res_a.count_h is not None
    self.assertLessEqual(res_a.count_h, 1.1)
    if _LONG:
      print(f"Case A count_h = {res_a.count_h}")
    self.assertNotEqual(
        res_a.kappa_check.outcome, refutation.Refutation.UNTESTABLE
    )

    # Case B: Accuracy at theta = 0.9, m_bar = 100, counts uniform from 80..120
    rng_b = np.random.default_rng(123)
    counts_b = rng_b.integers(80, 121, size=g, dtype=np.int64)
    totals_b = rng_b.binomial(counts_b, 0.9).astype(np.float64)
    shard_b = cluster_sketch.PartialClusterShard(ids, totals_b, counts_b)
    res_b = temporal.temporal_interval_from_shard(
        shard_b, metric="accuracy", level=0.95, m=1.05
    )
    self.assertNotEqual(
        res_b.kappa_check.outcome, refutation.Refutation.UNTESTABLE
    )
    self.assertIsNotNone(res_b.count_h)
    assert res_b.count_h is not None
    self.assertLessEqual(res_b.count_h, 1.1)
    if _LONG:
      print(f"Case B count_h = {res_b.count_h}")
    self.assertIsNotNone(res_b.count_energy)
    assert res_b.count_energy is not None
    self.assertAlmostEqual(res_b.count_energy, 10.0, delta=5.0)

    # Case C: Counts within +-1: count_energy is None
    counts_c = np.array(
        [100 if i % 2 == 0 else 101 for i in range(g)], dtype=np.int64
    )
    totals_c = np.ones(g, dtype=np.float64) * 100.0
    shard_c = cluster_sketch.PartialClusterShard(ids, totals_c, counts_c)
    res_c = temporal.temporal_interval_from_shard(
        shard_c, metric="mse", level=0.95, m=1.05
    )
    self.assertIsNone(res_c.count_energy)
    self.assertIsNotNone(res_c.count_h)
    assert res_c.count_h is not None
    self.assertLessEqual(res_c.count_h, 1.1)
    if _LONG:
      print(f"Case C count_h = {res_c.count_h}")

  def test_shard_ids_validation(self):
    """Verifies that non-contiguous or non-integer cluster_ids raise ValueError."""
    totals = np.ones(5, dtype=np.float64)
    counts = np.ones(5, dtype=np.int64) * 10

    ids_gap = np.array([0, 1, 3, 4, 5], dtype=np.int64)
    shard_gap = cluster_sketch.PartialClusterShard(ids_gap, totals, counts)
    with self.assertRaisesRegex(ValueError, "cluster_ids must be contiguous"):
      temporal.temporal_interval_from_shard(shard_gap)

    ids_str = np.array(["0", "1", "2", "3", "4"])
    shard_str = cluster_sketch.PartialClusterShard(ids_str, totals, counts)
    with self.assertRaisesRegex(ValueError, "cluster_ids must be integers"):
      temporal.temporal_interval_from_shard(shard_str)

  def test_drift_size(self):
    """Verifies drift warning false-positive rate is <= alpha under stationary i.i.d. data."""
    n = 20000 if _LONG else 4000
    g = 100 if _LONG else 40
    reps = 2000 if _LONG else 40
    rng = np.random.default_rng(2026)
    warnings = 0
    untestable_count = 0

    indices = np.arange(n, dtype=np.int64)
    window_ids = (indices * g) // n

    for _ in range(reps):
      e = np.abs(rng.standard_normal(n))
      shard = cluster_sketch.PartialClusterShard.from_data(e, window_ids, "mae")
      res = temporal.temporal_interval_from_shard(
          shard, metric="mae", level=0.95, m=1.05
      )
      if res.drift.outcome is temporal.Drift.WARNING:
        warnings += 1
      elif res.drift.outcome is temporal.Drift.UNTESTABLE:
        untestable_count += 1

    warning_rate = warnings / float(reps)
    bound = min(1.0, 0.05 + 3.0 * math.sqrt(0.05 * 0.95 / float(reps)))
    if _LONG:
      print(
          f"Drift size warning rate = {warning_rate:.4f}"
          f" ({warnings}/{reps}), untestable rate ="
          f" {untestable_count / float(reps):.4f}"
      )
    self.assertLessEqual(warning_rate, bound)
    # Untestable rate under stationarity is near 0. At reps=40, allow up to 2 (0.05)
    self.assertLessEqual(untestable_count / float(reps), 0.05)

  def test_drift_power(self):
    """Verifies drift warning power >= 0.9 when scale doubles in second half."""
    n = 20000
    g = 100
    reps = 200 if _LONG else 30
    rng = np.random.default_rng(2027)
    warnings = 0

    indices = np.arange(n, dtype=np.int64)
    window_ids = (indices * g) // n

    for _ in range(reps):
      e = np.abs(rng.standard_normal(n))
      e[n // 2 :] *= 2.0
      shard = cluster_sketch.PartialClusterShard.from_data(e, window_ids, "mae")
      res = temporal.temporal_interval_from_shard(
          shard, metric="mae", level=0.95, m=1.05
      )
      if res.drift.outcome is temporal.Drift.WARNING:
        warnings += 1

    power = warnings / float(reps)
    if _LONG:
      print(
          f"Drift power warning rate = {power:.4f}"
          f" ({warnings}/{reps})"
      )
    # Doubling scale gives huge power (1.0). In short mode (reps=30) assert >= 0.85
    min_power = 0.9 if _LONG else 0.85
    self.assertGreaterEqual(power, min_power)

  def test_drift_untestable_cases(self):
    """Verifies drift returns UNTESTABLE when G < 4 or a half refuses."""
    rng = np.random.default_rng(42)
    e_small = np.abs(rng.standard_normal(300))
    res_g3 = temporal.temporal_interval(
        e_small, num_windows=3, metric="mae", m=1.05
    )
    self.assertEqual(res_g3.drift.outcome, temporal.Drift.UNTESTABLE)

    e_long = np.abs(rng.standard_normal(20000))
    res_ref = temporal.temporal_interval(
        e_long, num_windows=100, metric="mae", m=None
    )
    self.assertEqual(res_ref.drift.outcome, temporal.Drift.UNTESTABLE)

  def test_guard_refusal_linear_counts(self):
    """Verifies that linearly rising counts trip the heterogeneity guard."""
    g = 100
    ids = np.arange(g, dtype=np.int64)
    counts = np.round(np.linspace(20, 380, g)).astype(np.int64)
    rng = np.random.default_rng(42)
    totals = np.array(
        [float(np.sum(rng.standard_normal(c) ** 2)) for c in counts]
    )
    shard = cluster_sketch.PartialClusterShard(ids, totals, counts)
    res = temporal.temporal_interval_from_shard(
        shard, metric="mse", level=0.95, m=1.05
    )
    self.assertEqual(res.kappa_check.outcome, refutation.Refutation.UNTESTABLE)
    self.assertIn(
        "window counts too heterogeneous for the κ check", res.message
    )
    self.assertIsNotNone(res.count_h)
    assert res.count_h is not None
    if _LONG:
      print(f"Linear counts count_h = {res.count_h}")
    self.assertGreater(res.count_h, 1.1)

  def test_high_accuracy_null(self):
    """Verifies refutation rate under the null at high accuracy (theta in {0.99, 0.999})."""
    n = 20070 if _LONG else 4000
    g = 200 if _LONG else 40
    reps = 2000 if _LONG else 40
    rng = np.random.default_rng(2026)
    indices = np.arange(n, dtype=np.int64)
    window_ids = (indices * g) // n
    _, counts = np.unique(window_ids, return_counts=True)
    ids = np.arange(g, dtype=np.int64)

    alpha_nominal = 0.05
    bound = min(1.0, alpha_nominal + 3.0 * math.sqrt(alpha_nominal * 0.95 / float(reps)))
    min_rate = 0.02 if _LONG else max(0.0, alpha_nominal - 3.0 * math.sqrt(alpha_nominal * 0.95 / float(reps)))

    for theta in (0.99, 0.999):
      refutations = 0
      for _ in range(reps):
        totals = rng.binomial(counts, theta).astype(np.float64)
        shard = cluster_sketch.PartialClusterShard(ids, totals, counts)
        res = temporal.temporal_interval_from_shard(
            shard, metric="accuracy", m=1.05, kappa=1.0
        )
        if res.kappa_check.outcome is refutation.Refutation.REFUTED:
          refutations += 1

      rate = refutations / float(reps)
      if _LONG:
        print(
            f"Null theta={theta} refutation rate = {rate:.4f}"
            f" ({refutations}/{reps})"
        )
      self.assertLessEqual(rate, bound)
      if theta == 0.99:
        self.assertGreaterEqual(rate, min_rate)

  def test_high_accuracy_power(self):
    """Verifies refutation power >= 0.40 under stationary AR(1) error rates at theta = 0.99."""
    n = 20070
    g = 200
    reps = 500 if _LONG else 40
    rng = np.random.default_rng(2027)
    indices = np.arange(n, dtype=np.int64)
    window_ids = (indices * g) // n
    _, counts = np.unique(window_ids, return_counts=True)
    ids = np.arange(g, dtype=np.int64)

    refutations = 0
    for _ in range(reps):
      eta = _generate_ar1_series(g, 0.5, rng)
      p_c = np.minimum(1.0, 0.01 * np.exp(0.6 * eta - 0.18))
      totals = rng.binomial(counts, 1.0 - p_c).astype(np.float64)
      shard = cluster_sketch.PartialClusterShard(ids, totals, counts)
      res = temporal.temporal_interval_from_shard(
          shard, metric="accuracy", m=1.05, kappa=1.0
      )
      if res.kappa_check.outcome is refutation.Refutation.REFUTED:
        refutations += 1

    power = refutations / float(reps)
    if _LONG:
      print(
          f"Power theta=0.99 refutation rate = {power:.4f}"
          f" ({refutations}/{reps})"
      )
    # At theta=0.99 AR(1), power is ~0.45. In short mode (reps=40), assert >= 0.30
    min_power = 0.40 if _LONG else 0.30
    self.assertGreaterEqual(power, min_power)

  def test_high_accuracy_raw_totals_failure(self):
    """Documents the failure of raw totals check at high accuracy under the null."""
    n = 20070 if _LONG else 4000
    g = 200 if _LONG else 40
    reps = 2000 if _LONG else 40
    rng = np.random.default_rng(2026)
    indices = np.arange(n, dtype=np.int64)
    window_ids = (indices * g) // n
    _, counts = np.unique(window_ids, return_counts=True)

    theta = 0.99
    raw_refutations = 0
    for _ in range(reps):
      totals = rng.binomial(counts, theta).astype(np.float64)
      # This is the raw-totals behaviour that the residualised check replaces.
      res = refutation.check_uncorrelated(totals, level=0.99)
      if res.outcome is refutation.Refutation.REFUTED:
        raw_refutations += 1

    raw_rate = raw_refutations / float(reps)
    if _LONG:
      print(
          f"Raw totals theta=0.99 refutation rate = {raw_rate:.6f}"
          f" ({raw_refutations}/{reps})"
      )
    # In short mode (reps=40), 0 or at most 1 rejection (0.025)
    max_raw_rate = 0.01 if _LONG else (1.0 / float(reps) + 1e-6)
    self.assertLessEqual(raw_rate, max_raw_rate)

  def test_bucket_quantile_window_properties(self):
    """Bucket-quantile window property test: random bucket counts with bursts."""
    rng = np.random.default_rng(20260928)
    trials = 200 if _LONG else 30
    for trial in range(trials):
      k = int(rng.integers(20, 150))
      if trial % 2 == 0:
        counts = rng.integers(1, 20, size=k)
      else:
        counts = rng.integers(1, 10, size=k)
        burst_idx = rng.choice(k, size=min(3, k), replace=False)
        counts[burst_idx] = rng.integers(15, 30, size=len(burst_idx))

      b_max = int(np.max(counts))
      # Ensure total items n allows at least g >= 2 with m_bar > b_max
      if np.sum(counts) <= 3 * b_max:
        counts = np.append(counts, np.full(10, b_max))
      n = int(np.sum(counts))
      b_max = int(np.max(counts))

      max_g = max(2, int(n // b_max))
      if n / float(max_g) > b_max:
        g = int(rng.integers(2, max_g + 1))
      else:
        g = int(rng.integers(2, max(3, max_g)))
        while float(n) / float(g) <= b_max and g > 2:
          g -= 1

      m_bar = float(n) / float(g)
      self.assertGreater(m_bar, b_max)

      cum_counts = np.cumsum(counts)
      c_prev = np.empty_like(counts)
      c_prev[0] = 0
      c_prev[1:] = cum_counts[:-1]
      w_k = (g * (2 * c_prev + counts)) // (2 * n)

      # 1. windows are contiguous and nondecreasing
      self.assertTrue(
          np.all(np.diff(w_k) >= 0), f"Trial {trial}: w_k not nondecreasing"
      )
      # 2. every window is nonempty
      unique_w = np.unique(w_k)
      np.testing.assert_array_equal(
          unique_w,
          np.arange(g),
          err_msg=f"Trial {trial}: not all windows present",
      )
      # 3. |n_c - m_bar| <= b_max for every window
      w_counts = np.bincount(w_k, weights=counts, minlength=g)
      diffs = np.abs(w_counts - m_bar)
      max_diff = np.max(diffs)
      self.assertLessEqual(
          max_diff,
          b_max + 1e-9,
          f"Trial {trial}: max diff {max_diff} > b_max {b_max}",
      )

  def test_buckets_equivalence_with_temporal_interval(self):
    g = 2000
    b = 10
    k = g
    n = k * b
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(n)
    counts = np.full(k, b, dtype=np.int64)
    totals = np.bincount(np.arange(n) // b, weights=residuals**2, minlength=k)

    res_items = temporal.temporal_interval(
        residuals,
        num_windows=g,
        metric="mse",
        m=2.0,
        kappa=1.0,
    )
    res_buckets = temporal.temporal_interval_from_buckets(
        counts,
        totals,
        num_windows=g,
        metric="mse",
        m=2.0,
        kappa=1.0,
    )
    self.assertTrue(math.isfinite(res_items.low))
    self.assertTrue(math.isfinite(res_items.high))
    self.assertAlmostEqual(res_buckets.low, res_items.low, places=12)
    self.assertAlmostEqual(res_buckets.high, res_items.high, places=12)

  def test_buckets_value_error_when_m_bar_leq_b_max(self):
    counts = [100, 1]
    totals = [1.0, 1.0]
    with self.assertRaises(ValueError) as ctx:
      temporal.temporal_interval_from_buckets(
          counts, totals, num_windows=3, metric="mse"
      )
    self.assertIn("some window could be empty", str(ctx.exception))
    self.assertIn("some window could be empty", str(ctx.exception))

  def test_buckets_r2_equals_items_when_windows_coincide(self):
    g = 2000
    b = 5
    k = g
    n = k * b
    rng = np.random.default_rng(123)
    y_true = rng.standard_normal(n)
    y_pred = y_true + 0.2 * rng.standard_normal(n)
    e2 = (y_true - y_pred) ** 2
    w_ids = np.arange(n) // b

    counts = np.full(k, b, dtype=np.int64)
    totals = np.column_stack([
        np.bincount(w_ids, weights=e2, minlength=k),
        np.bincount(w_ids, weights=y_true, minlength=k),
        np.bincount(w_ids, weights=y_true**2, minlength=k),
    ])

    res_items = temporal.temporal_interval(
        (y_true, y_pred),
        num_windows=g,
        metric="r2",
        kappa=1.0,
    )
    res_buckets = temporal.temporal_interval_from_buckets(
        counts,
        totals,
        num_windows=g,
        metric="r2",
        kappa=1.0,
    )
    self.assertTrue(math.isfinite(res_items.low))
    self.assertTrue(math.isfinite(res_items.high))
    self.assertAlmostEqual(res_buckets.low, res_items.low, places=12)
    self.assertAlmostEqual(res_buckets.high, res_items.high, places=12)

  def test_tuple_input_handling(self):
    rng = np.random.default_rng(1)
    y1 = rng.standard_normal(50)
    y2 = rng.standard_normal(50)
    with self.assertRaises(ValueError):
      temporal.temporal_interval((y1, y2), num_windows=5, metric="mse")
    with self.assertRaises(ValueError):
      temporal.temporal_interval(y1, num_windows=5, metric="r2")
    with self.assertRaises(ValueError):
      temporal.temporal_interval((y1, y2, y1), num_windows=5, metric="r2")

  def test_r2_shift_invariance(self):
    """y versus y + 1e6, with the same shift applied to predictions, gives the same R2 interval (rtol 1e-9)."""
    rng = np.random.default_rng(20260936)
    n = 4000
    g = 1000
    y_true = rng.normal(50.0, 10.0, size=n)
    y_pred = y_true + rng.normal(0.0, 2.0, size=n)

    shift = 1e6
    y_true_shifted = y_true + shift
    y_pred_shifted = y_pred + shift

    res_base = temporal.temporal_interval(
        (y_true, y_pred),
        num_windows=g,
        metric="r2",
        level=0.95,
        m_item=4.0,
        m_labels=4.0,
        kappa=1.0,
    )
    res_shifted = temporal.temporal_interval(
        (y_true_shifted, y_pred_shifted),
        num_windows=g,
        metric="r2",
        level=0.95,
        m_item=4.0,
        m_labels=4.0,
        kappa=1.0,
    )

    self.assertEqual(res_base.status, interval.Status.UNREFUTED)
    self.assertEqual(res_shifted.status, interval.Status.UNREFUTED)
    np.testing.assert_allclose(res_base.low, res_shifted.low, rtol=1e-9)
    np.testing.assert_allclose(res_base.high, res_shifted.high, rtol=1e-9)

  def test_item_order_invariance(self):
    """Permuting item order (with timestamps) leaves MSE and MAE intervals unchanged (rtol 1e-12)."""
    rng = np.random.default_rng(20260937)
    n = 2000
    g = 200
    timestamps = np.arange(n, dtype=np.int64)
    residuals = rng.normal(0.0, 1.0, size=n)
    perm = rng.permutation(n)

    for metric in ("mse", "mae"):
      res_orig = temporal.temporal_interval(
          residuals,
          num_windows=g,
          metric=metric,
          level=0.95,
          m_item=4.0 if metric == "mse" else 2.0,
          kappa=1.0,
          timestamps=timestamps,
      )
      res_perm = temporal.temporal_interval(
          residuals[perm],
          num_windows=g,
          metric=metric,
          level=0.95,
          m_item=4.0 if metric == "mse" else 2.0,
          kappa=1.0,
          timestamps=timestamps[perm],
      )
      self.assertEqual(res_orig.status, interval.Status.UNREFUTED)
      self.assertEqual(res_perm.status, interval.Status.UNREFUTED)
      np.testing.assert_allclose(
          res_orig.low, res_perm.low, rtol=1e-12, atol=1e-12
      )
      np.testing.assert_allclose(
          res_orig.high, res_perm.high, rtol=1e-12, atol=1e-12
      )


if __name__ == "__main__":
  absltest.main()
