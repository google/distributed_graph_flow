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

"""Tests for the D1 cluster-totals bound.

The contract under test is not "the interval is correct" -- correctness of the
underlying Cantelli bound is already covered by
`independent/interval_test.py`. What is tested here is the *reduction*: that
`cluster_interval` is exactly the independent estimator applied to the cluster totals, rescaled, and
that every place where the reduction could silently lose coverage
is pinned down.

`cluster_interval` takes **residuals**, like the independent estimator's array API.
Passing pre-transformed summands would cause double-squaring in `mse` mode;
`InputContractTest` verifies this contract.
"""

import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dgf.src.stats import cluster_bound
from dgf.src.stats import cluster_sketch
from dgf.src.stats import refutation
from dgf.src.stats.independent import interval as ind_interval
from dgf.src.stats.independent import metrics as ind_metrics
from dgf.src.stats.independent import sketch as ind_sketch

# Set DGF_STATS_LONG_TESTS=1 (blaze test --test_env=...) for the full-size runs.
_LONG = os.environ.get("DGF_STATS_LONG_TESTS") == "1"


def _residuals(n: int, seed: int = 7) -> np.ndarray:
  """Standard normal residuals, so the mse estimand is exactly 1."""
  rng = np.random.default_rng(seed)
  return rng.standard_normal(n)


def _independent_on_totals(
    residuals: np.ndarray,
    ids: np.ndarray,
    metric: str,
    effective_n: float,
    m: float,
    level: float = 0.95,
) -> ind_interval.Result:
  """The reduction, spelled out by hand, on the cluster-total scale.

  Goes through the Sketch branch deliberately: the totals are already summands,
  and the array branch would transform them again.
  """
  summands = ind_metrics.extract_summands(residuals, metric)
  totals, _ = cluster_bound.aggregate(summands, ids)
  sk = ind_sketch.from_array(totals, effective_n=effective_n)
  return ind_interval.confidence_interval(sk, metric="mse", level=level, m=m)


class AggregateTest(parameterized.TestCase):
  """`aggregate` is the low-level helper and takes summands, not residuals."""

  def test_contiguous_ids(self):
    totals, n = cluster_bound.aggregate([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
    np.testing.assert_allclose(totals, [3.0, 7.0])
    self.assertEqual(n, 4)

  def test_ids_need_not_be_contiguous_or_sorted(self):
    # Same partition as above, expressed with sparse out-of-order labels. Only
    # the induced partition may matter; if this ever depended on the label
    # values, a real clustering algorithm's arbitrary ids would break it.
    totals, n = cluster_bound.aggregate(
        [1.0, 3.0, 2.0, 4.0], [900, 17, 900, 17]
    )
    # `np.unique` orders by label value, so 17 comes first.
    np.testing.assert_allclose(totals, [7.0, 3.0])
    self.assertEqual(n, 4)

  def test_unequal_cluster_sizes(self):
    totals, n = cluster_bound.aggregate(
        [1.0, 2.0, 3.0, 4.0, 5.0], [0, 0, 0, 1, 2]
    )
    np.testing.assert_allclose(totals, [6.0, 4.0, 5.0])
    self.assertEqual(n, 5)

  def test_string_ids(self):
    totals, _ = cluster_bound.aggregate([1.0, 2.0, 3.0], ["b", "a", "b"])
    np.testing.assert_allclose(totals, [2.0, 4.0])

  def test_length_mismatch_raises(self):
    with self.assertRaises(ValueError):
      cluster_bound.aggregate([1.0, 2.0], [0])

  def test_empty_raises(self):
    with self.assertRaises(ValueError):
      cluster_bound.aggregate([], [])


class InputContractTest(parameterized.TestCase):
  """`cluster_interval` takes residuals, exactly like the independent estimator's array API.

  Whether a caller who ignores that is *caught* depends on the metric, and the
  split is worth stating explicitly rather than discovering:

    - `mse`/`rmse` transform by `e**2`, which is **not** idempotent, so passing
      summands silently bounds `E[e^4]` instead of `E[e^2]`. This is a real
      wrong answer and it is detectable.
    - `mae` transforms by `np.abs` and `accuracy` by the identity, both
      **idempotent**, so passing summands produces bit-identical output. The
      contract violation is undetectable -- and harmless, because the answer is
      right anyway.
  """

  @parameterized.parameters("mse", "rmse")
  def test_squaring_metrics_reject_pre_transformed_input(self, metric: str):
    # These are the metrics where the residuals-in contract has teeth. If these
    # two ever agree, the transform has been dropped and every caller who
    # followed the independent convention is silently getting `E[e^4]`.
    e = _residuals(4000)
    ids = np.arange(4000) // 5
    from_residuals = cluster_bound.cluster_interval(
        e, ids, metric=metric, m=4.0
    )
    from_summands = cluster_bound.cluster_interval(
        ind_metrics.extract_summands(e, metric), ids, metric=metric, m=4.0
    )
    self.assertNotAlmostEqual(
        from_residuals.high, from_summands.high, places=6
    )

  @parameterized.named_parameters(
      ("mae", "mae", False),
      ("accuracy", "accuracy", True),
  )
  def test_idempotent_metrics_silently_accept_pre_transformed_input(
      self, metric: str, indicator: bool
  ):
    # Documented forgiveness, not a guarantee to build on: callers should still
    # pass residuals, because the same mistake against `mse` is a wrong answer.
    # Pinned so that nobody "completes" the parameter list above with these two
    # and reintroduces an assertion that cannot hold.
    rng = np.random.default_rng(23)
    data = (
        (rng.random(4000) < 0.9).astype(np.float64)
        if indicator
        else _residuals(4000)
    )
    ids = np.arange(4000) // 5
    from_data = cluster_bound.cluster_interval(data, ids, metric=metric, m=4.0)
    from_summands = cluster_bound.cluster_interval(
        ind_metrics.extract_summands(data, metric), ids, metric=metric, m=4.0
    )
    self.assertEqual(from_data.high, from_summands.high)
    self.assertEqual(from_data.low, from_summands.low)


  def test_mse_estimand_brackets_one(self):
    # Standard normal residuals, so E[e^2] = 1. A double-squaring bug bounds
    # E[e^4] = 3 instead, which this catches directly.
    e = _residuals(20000)
    ids = np.arange(20000) // 10
    res = cluster_bound.cluster_interval(e, ids, metric="mse", m=4.0)
    self.assertEqual(res.status, ind_interval.Status.UNREFUTED)
    self.assertLessEqual(res.low, 1.0)
    self.assertGreaterEqual(res.high, 1.0)

  def test_accuracy_array_is_taken_as_the_indicator(self):
    # The independent estimator treats an `accuracy` array as the 0/1 indicator itself, with no
    # transform. This must match.
    ind = np.array([1.0, 0.0, 1.0, 1.0])
    summands = ind_metrics.extract_summands(ind, "accuracy")
    np.testing.assert_allclose(summands, ind)


class ReductionTest(parameterized.TestCase):
  """Pins the reduction to independent."""

  def test_singleton_clusters_reproduce_independent_with_effective_n(self):
    # With one item per cluster the totals *are* the summands and G == n, so the
    # result must be bit-identical to independent told `effective_n = n`. If this drifts,
    # the cluster path has grown an extra approximation that independent does not have.
    e = _residuals(500)
    ids = np.arange(500)
    got = cluster_bound.cluster_interval(
        e, ids, metric="mse", level=0.95, m=4.0, kappa_cluster=1.0
    )
    want = ind_interval.confidence_interval(
        e, metric="mse", level=0.95, m=4.0, effective_n=500.0
    )
    self.assertEqual(got.status, want.status)
    self.assertAlmostEqual(got.low, want.low, places=12)
    self.assertAlmostEqual(got.high, want.high, places=12)
    self.assertAlmostEqual(got.mean_cluster_size, 1.0, places=12)
    self.assertEqual(got.num_clusters, 500)

  def test_equals_independent_on_totals_then_rescaled(self):
    # The general statement of the same thing, at a real granularity.
    e = _residuals(600)
    ids = np.arange(600) // 4
    got = cluster_bound.cluster_interval(e, ids, metric="mse", m=4.0)
    want = _independent_on_totals(e, ids, "mse", effective_n=150.0, m=4.0)
    self.assertAlmostEqual(got.low, want.low / 4.0, places=12)
    self.assertAlmostEqual(got.high, want.high / 4.0, places=12)

  def test_rescaling_uses_mean_size_for_unequal_clusters(self):
    # The pooled cluster mean is (n/G)*theta even when the clusters differ in
    # size, so n/G -- not any per-cluster size -- is the right divisor. This is
    # the step most likely to be "fixed" into a wrong per-cluster normalisation.
    #
    # G = 300 is deliberately well above the Cantelli existence threshold
    # n_A = (4-1)(1-0.025)/0.025 = 117; with too few clusters both sides refuse
    # and return NaN, and the comparison would be vacuous.
    sizes = [2, 3, 5] * 100  # 1000 items, 300 ragged clusters
    ids = np.concatenate(
        [np.full(k, i) for i, k in enumerate(sizes)]
    )
    e = _residuals(int(ids.size))
    got = cluster_bound.cluster_interval(e, ids, metric="mse", m=4.0)
    self.assertEqual(got.num_clusters, 300)
    self.assertAlmostEqual(got.mean_cluster_size, 1000 / 300, places=12)
    self.assertEqual(got.status, ind_interval.Status.UNREFUTED)

    want = _independent_on_totals(e, ids, "mse", effective_n=300.0, m=4.0)
    self.assertAlmostEqual(got.low, want.low / (1000 / 300), places=12)
    self.assertAlmostEqual(got.high, want.high / (1000 / 300), places=12)

  def test_kappa_divides_the_cluster_count(self):
    e = _residuals(400)
    ids = np.arange(400) // 2  # G = 200
    got = cluster_bound.cluster_interval(
        e, ids, metric="mse", m=4.0, kappa_cluster=2.0
    )
    self.assertAlmostEqual(got.effective_n, 100.0, places=12)

  def test_larger_kappa_never_narrows(self):
    e = _residuals(4000)
    ids = np.arange(4000) // 4
    tight = cluster_bound.cluster_interval(
        e, ids, metric="mse", m=4.0, kappa_cluster=1.0
    )
    loose = cluster_bound.cluster_interval(
        e, ids, metric="mse", m=4.0, kappa_cluster=3.0
    )
    self.assertGreaterEqual(loose.high, tight.high)
    self.assertLessEqual(loose.low, tight.low)


class MetricTransformTest(parameterized.TestCase):

  def test_rmse_is_sqrt_of_mse(self):
    # rmse must be the square root of the *rescaled per-item* mse endpoints, not
    # the square root of a cluster total. Getting this wrong inflates the bound
    # by sqrt(mean_cluster_size) and would still look plausible.
    e = _residuals(2000)
    ids = np.arange(2000) // 5
    mse = cluster_bound.cluster_interval(e, ids, metric="mse", m=4.0)
    rmse = cluster_bound.cluster_interval(e, ids, metric="rmse", m=4.0)
    self.assertAlmostEqual(rmse.low, math.sqrt(mse.low), places=12)
    self.assertAlmostEqual(rmse.high, math.sqrt(mse.high), places=12)

  def test_accuracy_is_clamped_to_one(self):
    rng = np.random.default_rng(3)
    ind = (rng.random(1000) < 0.9).astype(np.float64)
    ids = np.arange(1000) // 2
    res = cluster_bound.cluster_interval(ind, ids, metric="accuracy", m=4.0)
    self.assertLessEqual(res.high, 1.0)
    # Pin that the clamp is actually doing something here, so this does not
    # quietly become a tautology if the bound tightens.
    unclamped = _independent_on_totals(ind, ids, "accuracy", effective_n=500.0, m=4.0)
    self.assertGreater(unclamped.high / 2.0, 1.0)

  def test_mae_applies_no_final_transform(self):
    e = _residuals(2000)
    ids = np.arange(2000) // 5
    mae = cluster_bound.cluster_interval(e, ids, metric="mae", m=4.0)
    want = _independent_on_totals(e, ids, "mae", effective_n=400.0, m=4.0)
    self.assertAlmostEqual(mae.low, want.low / 5.0, places=12)
    self.assertAlmostEqual(mae.high, want.high / 5.0, places=12)

  def test_r2_is_refused(self):
    e = _residuals(100)
    ids = np.arange(100) // 2
    with self.assertRaisesRegex(ValueError, "r2"):
      cluster_bound.cluster_interval(e, ids, metric="r2", m_item=4.0)

  def test_unknown_metric_raises(self):
    e = _residuals(100)
    ids = np.arange(100) // 2
    with self.assertRaises(ValueError):
      cluster_bound.cluster_interval(e, ids, metric="average_precision", m=4.0)


class GuardTest(parameterized.TestCase):

  @parameterized.parameters(0.0, 0.5, 0.999, -1.0)
  def test_kappa_below_one_raises(self, kappa: float):
    e = _residuals(100)
    ids = np.arange(100) // 2
    with self.assertRaisesRegex(ValueError, "kappa_cluster"):
      cluster_bound.cluster_interval(
          e, ids, metric="mse", m=4.0, kappa_cluster=kappa
      )

  def test_nan_kappa_raises(self):
    # `not (nan >= 1)` is True, so the guard catches NaN; this pins that, since
    # rewriting the guard as `if kappa < 1` would silently let NaN through.
    e = _residuals(100)
    ids = np.arange(100) // 2
    with self.assertRaisesRegex(ValueError, "kappa_cluster"):
      cluster_bound.cluster_interval(
          e, ids, metric="mse", m=4.0, kappa_cluster=float("nan")
      )

  def test_too_few_clusters_refuses_rather_than_returning_garbage(self):
    # G = 2 is far below the Cantelli existence threshold, so this must come
    # back refused, not as a nonsense interval.
    e = _residuals(1000)
    ids = np.arange(1000) // 500
    res = cluster_bound.cluster_interval(e, ids, metric="mse", m=4.0)
    self.assertIsNot(res.status, ind_interval.Status.UNREFUTED)


class CoverageSanityTest(parameterized.TestCase):
  """A cheap end-to-end check that the reduction is actually conservative."""

  def test_independent_blocks_aligned_clusters_cover(self):
    # Truly independent blocks of size 10, clusters aligned to them, so
    # kappa_cluster = 1 is *true*. Coverage must not fall below nominal.
    n, block = 5000, 10
    reps = 300 if _LONG else 40
    rng = np.random.default_rng(101)
    ids = np.arange(n) // block
    covered = 0
    certified = 0
    for _ in range(reps):
      # Equicorrelated within block via a shared component; blocks independent.
      shared = np.repeat(rng.standard_normal(n // block), block)
      idio = rng.standard_normal(n)
      e = (shared + idio) / math.sqrt(2.0)  # unit variance, so E[e^2] = 1
      res = cluster_bound.cluster_interval(
          e, ids, metric="mse", level=0.95, m=4.0, kappa_cluster=1.0
      )
      if res.status is ind_interval.Status.UNREFUTED:
        certified += 1
        if res.low <= 1.0 <= res.high:
          covered += 1
    self.assertGreater(certified, 0.5 * reps)
    # Cantelli is very conservative here; anything below nominal is a red flag.
    # In short mode (reps=40), nominal coverage p=0.95 allows 0.95 - 3*sqrt(p*(1-p)/certified).
    min_cov = (
        0.95
        if _LONG
        else max(0.0, 0.95 - 3.0 * math.sqrt(0.95 * 0.05 / float(certified)))
    )
    self.assertGreaterEqual(covered / certified, min_cov)


class ClusterSketchContractTest(parameterized.TestCase):
  """Pins ClusterSketch and PartialClusterShard equivalence and merge invariants."""

  @parameterized.parameters("mae", "mse", "rmse", "accuracy")
  def test_sketch_array_equivalence(self, metric: str):
    rng = np.random.default_rng(42)
    n = 1000
    ids = np.arange(n) // 5
    if metric == "accuracy":
      data = (rng.random(n) < 0.85).astype(np.float64)
    else:
      data = rng.standard_normal(n)

    res_arr = cluster_bound.cluster_interval(data, ids, metric=metric, m=4.0)
    sk = cluster_sketch.ClusterSketch.from_data(data, ids, metric=metric)
    res_sk = cluster_bound.cluster_interval(sk, metric=metric, m=4.0)

    self.assertEqual(res_arr.status, res_sk.status)
    self.assertAlmostEqual(res_arr.low, res_sk.low, places=12)
    self.assertAlmostEqual(res_arr.high, res_sk.high, places=12)
    self.assertEqual(res_arr.m_declared, res_sk.m_declared)
    self.assertAlmostEqual(res_arr.m_observed, res_sk.m_observed, places=12)
    self.assertEqual(res_arr.diagnostic.diagnosis, res_sk.diagnostic.diagnosis)
    self.assertAlmostEqual(
        res_arr.diagnostic.c_up, res_sk.diagnostic.c_up, places=12
    )

  def test_disjoint_sketch_merge_associative_and_exact(self):
    rng = np.random.default_rng(101)
    n = 1000
    ids = np.arange(n) // 5  # 200 clusters of 5
    data = rng.standard_normal(n)

    # Shard A has clusters 0..99, Shard B has clusters 100..199
    sk_a = cluster_sketch.ClusterSketch.from_data(
        data[:500], ids[:500], metric="mse"
    )
    sk_b = cluster_sketch.ClusterSketch.from_data(
        data[500:], ids[500:], metric="mse"
    )

    sk_merged = sk_a + sk_b
    res_merged = cluster_bound.cluster_interval(sk_merged, metric="mse", m=4.0)
    res_full = cluster_bound.cluster_interval(data, ids, metric="mse", m=4.0)

    self.assertEqual(res_merged.status, res_full.status)
    self.assertAlmostEqual(res_merged.low, res_full.low, places=12)
    self.assertAlmostEqual(res_merged.high, res_full.high, places=12)
    self.assertEqual(res_merged.num_clusters, 200)
    self.assertAlmostEqual(res_merged.mean_cluster_size, 5.0, places=12)

  def test_overlapping_sketch_merge_raises(self):
    rng = np.random.default_rng(202)
    n = 500
    data = rng.standard_normal(n)
    # Both shards share cluster IDs 50..99
    ids_a = np.arange(n) // 5  # clusters 0..99
    ids_b = (np.arange(n) // 5) + 50  # clusters 50..149

    sk_a = cluster_sketch.ClusterSketch.from_data(data, ids_a, metric="mse")
    sk_b = cluster_sketch.ClusterSketch.from_data(data, ids_b, metric="mse")

    with self.assertRaisesRegex(ValueError, "overlapping cluster IDs"):
      sk_a.merge(sk_b)

  def test_partial_cluster_shard_merge_exact_on_split_clusters(self):
    rng = np.random.default_rng(303)
    n = 1000
    ids = np.arange(n) // 5  # 200 clusters of 5
    data = rng.standard_normal(n)

    # Shard A gets even indices, Shard B gets odd indices (every cluster is split!)
    p_a = cluster_sketch.PartialClusterShard.from_data(
        data[0::2], ids[0::2], metric="mse"
    )
    p_b = cluster_sketch.PartialClusterShard.from_data(
        data[1::2], ids[1::2], metric="mse"
    )

    p_merged = p_a + p_b
    sk_from_shards = p_merged.to_cluster_sketch()

    res_from_shards = cluster_bound.cluster_interval(
        sk_from_shards, metric="mse", m=4.0
    )
    res_full = cluster_bound.cluster_interval(data, ids, metric="mse", m=4.0)

    self.assertEqual(res_from_shards.status, res_full.status)
    self.assertAlmostEqual(res_from_shards.low, res_full.low, places=12)
    self.assertAlmostEqual(res_from_shards.high, res_full.high, places=12)
    self.assertAlmostEqual(
        res_from_shards.m_observed, res_full.m_observed, places=12
    )


class DiagnosisPayloadTest(parameterized.TestCase):
  """Pins the 5 Diagnosis states and DiagnosticPayload attributes."""

  def test_certified_useful(self):
    # G = 4000, M = 4, alpha = 0.05, c_target = 0.2 -> n_A = 117.0, c_up = sqrt(117/4000) = 0.1710 <= 0.2
    e = _residuals(20000)
    ids = np.arange(20000) // 5  # G = 4000
    res = cluster_bound.cluster_interval(
        e, ids, metric="mse", m=4.0, level=0.95, c_target=0.2
    )
    self.assertEqual(res.status, ind_interval.Status.UNREFUTED)
    self.assertEqual(
        res.diagnostic.diagnosis, cluster_bound.Diagnosis.CERTIFIED_USEFUL
    )
    self.assertLessEqual(res.diagnostic.c_up, 0.2)
    self.assertAlmostEqual(
        res.diagnostic.min_effective_n_exist, 117.0, places=6
    )
    self.assertIsNotNone(res.diagnostic.required_clusters_target)

  def test_certified_wide(self):
    # G = 200, M = 4, alpha = 0.05, c_target = 0.2 -> n_A = 117 < 200 (certifies), but c_up = 0.7649 > 0.2
    e = _residuals(1000)
    ids = np.arange(1000) // 5  # G = 200
    res = cluster_bound.cluster_interval(
        e, ids, metric="mse", m=4.0, level=0.95, c_target=0.2
    )
    self.assertEqual(res.status, ind_interval.Status.UNREFUTED)
    self.assertEqual(
        res.diagnostic.diagnosis, cluster_bound.Diagnosis.CERTIFIED_WIDE
    )
    self.assertGreater(res.diagnostic.c_up, 0.2)
    # required_clusters_target = ceil(1.0 * 117.0 / (0.2**2)) + 1 = 2926
    self.assertEqual(res.diagnostic.required_clusters_target, 2926)

  def test_resolvable_sample_size(self):
    # G = 50, M = 4, alpha = 0.05, kappa = 1.0 -> n_A = 117 > 50 (refuses)
    e = _residuals(250)
    ids = np.arange(250) // 5  # G = 50
    res = cluster_bound.cluster_interval(
        e, ids, metric="mse", m=4.0, level=0.95, kappa_cluster=1.0
    )
    self.assertEqual(res.status, ind_interval.Status.ASSUMPTION_REQUIRED)
    self.assertEqual(
        res.diagnostic.diagnosis, cluster_bound.Diagnosis.RESOLVABLE_SAMPLE_SIZE
    )
    self.assertEqual(res.diagnostic.required_clusters_exist, 118)
    self.assertEqual(res.diagnostic.required_items_exist, 118 * 5)
    # max_supported_alpha = 2*(3) / (50 + 3) = 6/53 ≈ 0.1132
    self.assertAlmostEqual(
        res.diagnostic.max_supported_alpha, 6.0 / 53.0, places=4
    )

  def test_saturated_graph_mixing_via_max_achievable(self):
    e = _residuals(250)
    ids = np.arange(250) // 5  # G = 50
    res = cluster_bound.cluster_interval(
        e,
        ids,
        metric="mse",
        m=4.0,
        level=0.95,
        max_achievable_effective_n=80.0,
    )
    self.assertEqual(res.status, ind_interval.Status.ASSUMPTION_REQUIRED)
    self.assertEqual(
        res.diagnostic.diagnosis, cluster_bound.Diagnosis.SATURATED_GRAPH_MIXING
    )
    self.assertIsNone(res.diagnostic.required_clusters_exist)
    self.assertIsNone(res.diagnostic.required_items_exist)
    self.assertIsNone(res.diagnostic.required_clusters_target)
    self.assertIsNone(res.diagnostic.required_items_target)

  def test_saturated_graph_mixing_via_kappa_span(self):
    # G = 400, kappa = 300.0 >= 0.5 * 400 -> saturated
    e = _residuals(2000)
    ids = np.arange(2000) // 5  # G = 400
    res = cluster_bound.cluster_interval(
        e, ids, metric="mse", m=4.0, level=0.95, kappa_cluster=300.0
    )
    self.assertEqual(res.status, ind_interval.Status.ASSUMPTION_REQUIRED)
    self.assertEqual(
        res.diagnostic.diagnosis, cluster_bound.Diagnosis.SATURATED_GRAPH_MIXING
    )
    self.assertIsNone(res.diagnostic.required_clusters_exist)
    self.assertIsNone(res.diagnostic.required_items_exist)

  def test_saturated_graph_mixing_via_nu_sub_one_graceful(self):
    # G = 400, kappa = 600.0 -> nu = 400 / 600 = 0.667 < 1.0 (must not crash)
    e = _residuals(2000)
    ids = np.arange(2000) // 5  # G = 400
    res = cluster_bound.cluster_interval(
        e, ids, metric="mse", m=4.0, level=0.95, kappa_cluster=600.0
    )
    self.assertEqual(res.status, ind_interval.Status.ASSUMPTION_REQUIRED)
    self.assertTrue(math.isnan(res.low))
    self.assertTrue(math.isnan(res.high))
    self.assertIsNone(res.level)
    self.assertEqual(
        res.diagnostic.diagnosis, cluster_bound.Diagnosis.SATURATED_GRAPH_MIXING
    )
    self.assertIsNone(res.diagnostic.required_clusters_exist)

  def test_tail_unresolved_refuted(self):
    # G = 400 > n_A, but with contaminated draw where m_observed > m_declared
    rng = np.random.default_rng(999)
    e = rng.standard_normal(2000)
    # Introduce large outlier
    e[0] = 50.0
    ids = np.arange(2000) // 5  # G = 400
    res = cluster_bound.cluster_interval(
        e, ids, metric="mse", m=1.2, level=0.95
    )
    self.assertEqual(res.status, ind_interval.Status.TAIL_UNRESOLVED)
    self.assertEqual(
        res.diagnostic.diagnosis,
        cluster_bound.Diagnosis.TAIL_UNRESOLVED_REFUTED,
    )
    self.assertIsNotNone(res.diagnostic.required_clusters_exist)
    assert res.diagnostic.required_clusters_exist is not None
    self.assertGreater(res.diagnostic.required_clusters_exist, 118)


class R2ClusterBoundTest(parameterized.TestCase):
  """Tests for Claim 7 R2 cluster confidence intervals."""

  def test_r2_hand_computed(self):
    g = 10000
    counts = np.ones(g, dtype=np.int64)
    y_totals = np.concatenate([np.full(5000, -1.0), np.full(5000, 1.0)])
    y2_totals = np.ones(g, dtype=np.float64)
    e2_totals = np.full(g, 0.25, dtype=np.float64)

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low, high, level_out, status, detail, _, _ = (
        cluster_bound.compute_r2_interval(
            e2_totals,
            y_totals,
            y2_totals,
            counts,
            level=0.95,
            m_item=2.0,
            m_labels=2.0,
            kappa=1.0,
            kappa_labels=1.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )

    self.assertEqual(status, ind_interval.Status.UNREFUTED)
    self.assertEqual(level_out, 0.95)
    self.assertIsNotNone(detail)
    assert detail is not None

    # Analytical values from cantelli_c formula: sqrt((M_c - 1)*kappa*(1 - a') / (a' * G))
    c_a_expected = math.sqrt((1.0 * (1.0 - 0.0125)) / (0.0125 * 10000.0))
    c_b_lower_expected = c_a_expected
    c_b_upper_expected = math.sqrt(
        (1.0 * (1.0 - 0.00625)) / (0.00625 * 10000.0)
    )
    chebyshev_expected = 1.0 / (10000.0 * 0.00625)  # 0.016

    d_expected = 1.0 - c_b_upper_expected - chebyshev_expected
    l_b_expected = 1.0 / (1.0 + c_b_lower_expected)
    u_b_expected = 1.0 / d_expected

    l_a_expected = 0.25 / (1.0 + c_a_expected)
    u_a_expected = 0.25 / (1.0 - c_a_expected)

    low_expected = 1.0 - (u_a_expected / l_b_expected)
    high_expected = 1.0 - (l_a_expected / u_b_expected)

    self.assertAlmostEqual(detail.L_B, l_b_expected, places=12)
    self.assertAlmostEqual(detail.U_B, u_b_expected, places=12)
    self.assertAlmostEqual(detail.D, d_expected, places=12)
    self.assertAlmostEqual(low, low_expected, places=12)
    self.assertAlmostEqual(high, high_expected, places=12)

  def test_r2_v_hat_zero(self):
    g = 1000
    counts = np.ones(g, dtype=np.int64)
    y_totals = np.full(g, 5.0)
    y2_totals = np.full(g, 25.0)
    e2_totals = np.full(g, 0.5)

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low, high, level_out, status, detail, _, _ = (
        cluster_bound.compute_r2_interval(
            e2_totals,
            y_totals,
            y2_totals,
            counts,
            level=0.95,
            m_item=2.0,
            m_labels=2.0,
            kappa=1.0,
            kappa_labels=1.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )
    self.assertEqual(low, float("-inf"))
    self.assertEqual(high, 1.0)
    self.assertEqual(level_out, 0.95)
    self.assertEqual(status, ind_interval.Status.UNREFUTED)
    self.assertIsNotNone(detail)
    assert detail is not None
    self.assertEqual(detail.L_B, 0.0)
    self.assertEqual(detail.U_B, float("inf"))

  def test_r2_numerator_matches_cluster_interval_mse(self):
    rng = np.random.default_rng(2026)
    for g, m, kappa in ((1000, 2.0, 1.0), (2000, 4.0, 1.5), (5000, 3.0, 2.0)):
      counts = rng.integers(1, 10, size=g, dtype=np.int64)
      n = int(np.sum(counts))
      e2_totals = rng.uniform(0.1, 2.0, size=g) * counts
      y_totals = rng.standard_normal(g) * counts
      y2_totals = (y_totals**2 / counts) + rng.uniform(
          0.1, 1.0, size=g
      ) * counts

      def dummy_kappa(res, counts, k):
        return refutation.RefutationResult(
            refutation.Refutation.NOT_REFUTED, 0.0, 1.96
        )

      _, _, _, _, detail, _, _ = cluster_bound.compute_r2_interval(
          e2_totals,
          y_totals,
          y2_totals,
          counts,
          level=0.95,
          m_item=m,
          m_labels=m,
          kappa=kappa,
          kappa_labels=kappa,
          run_kappa_check_fn=dummy_kappa,
      )
      assert detail is not None

      s_counts_sq = int(np.sum(counts**2))
      m_c = cluster_sketch.default_m_c_from_counts(
          m, n, g, int(np.max(counts)), s_counts_sq
      )
      sk_a = cluster_sketch.ClusterSketch(
          n_items=n,
          num_clusters=g,
          sum_s=float(np.sum(e2_totals)),
          sum_s2=float(np.sum(e2_totals**2)),
          top_32=tuple(
              sorted([float(x) for x in e2_totals], reverse=True)[:32]
          ),
          max_cluster_size=int(np.max(counts)),
          sum_count_sq=s_counts_sq,
          kappa_cluster=kappa,
      )
      res_a = cluster_bound.cluster_interval(
          sk_a, metric="mse", level=1.0 - 0.05 / 2.0, m=m_c, kappa_cluster=kappa
      )

      self.assertAlmostEqual(detail.L_A, res_a.low, places=12)
      self.assertAlmostEqual(detail.U_A, res_a.high, places=12)

  def test_r2_d_leq_zero_gives_high_one(self):
    # G = 200, n_c = 1, m = 2.0 -> c_A = sqrt(0.395) < 1, but chebyshev_term = 0.8 -> D < 0
    g = 200
    counts = np.ones(g, dtype=np.int64)
    y_totals = np.concatenate([np.full(100, -1.0), np.full(100, 1.0)])
    y2_totals = np.ones(g)
    e2_totals = np.full(g, 0.25)

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low, high, level_out, status, detail, _, _ = (
        cluster_bound.compute_r2_interval(
            e2_totals,
            y_totals,
            y2_totals,
            counts,
            level=0.95,
            m_item=2.0,
            m_labels=2.0,
            kappa=1.0,
            kappa_labels=1.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )
    self.assertIsNotNone(detail)
    assert detail is not None
    self.assertLessEqual(detail.D, 0.0)
    self.assertEqual(high, 1.0)
    self.assertEqual(status, ind_interval.Status.UNREFUTED)
    self.assertTrue(math.isfinite(low))

  def test_r2_refusal_at_small_g(self):
    # G = 10, m = 4.0 -> c_A >= 1.0 -> Refused with ASSUMPTION_REQUIRED
    g = 10
    counts = np.ones(g, dtype=np.int64)
    y_totals = np.zeros(g)
    y2_totals = np.ones(g)
    e2_totals = np.ones(g)

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low, high, level_out, status, _, msg, _ = cluster_bound.compute_r2_interval(
        e2_totals,
        y_totals,
        y2_totals,
        counts,
        level=0.95,
        m_item=4.0,
        run_kappa_check_fn=dummy_kappa,
    )
    self.assertTrue(math.isnan(low))
    self.assertTrue(math.isnan(high))
    self.assertIsNone(level_out)
    self.assertEqual(status, ind_interval.Status.ASSUMPTION_REQUIRED)
    self.assertIn("insufficient sample size", msg)

    # G < 2 raises ValueError
    with self.assertRaises(ValueError):
      cluster_bound.compute_r2_interval(
          np.ones(1),
          np.zeros(1),
          np.ones(1),
          np.ones(1, dtype=np.int64),
          level=0.95,
          run_kappa_check_fn=dummy_kappa,
      )

  def test_r2_shard_merge(self):
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.standard_normal(n)
    y_pred = y_true + 0.3 * rng.standard_normal(n)
    cluster_ids = rng.integers(0, 10, size=n)

    shard1 = cluster_sketch.R2ClusterShard.from_data(
        y_true[:50], y_pred[:50], cluster_ids[:50]
    )
    shard2 = cluster_sketch.R2ClusterShard.from_data(
        y_true[50:], y_pred[50:], cluster_ids[50:]
    )
    merged = shard1.merge(shard2)
    full = cluster_sketch.R2ClusterShard.from_data(y_true, y_pred, cluster_ids)

    self.assertEqual(merged.n_items, full.n_items)
    np.testing.assert_array_equal(merged.cluster_ids, full.cluster_ids)
    np.testing.assert_array_equal(merged.cluster_counts, full.cluster_counts)
    np.testing.assert_allclose(
        merged.e2_totals, full.e2_totals, rtol=1e-12, atol=1e-12
    )

    # merged is centred at shard1.ref; re-centre to full.ref for exact comparison
    merged_aligned = merged.recenter(full.ref)
    np.testing.assert_allclose(
        merged_aligned.y_totals, full.y_totals, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        merged_aligned.y2_totals, full.y2_totals, rtol=1e-12, atol=1e-12
    )

  def test_r2_shard_merge_different_means(self):
    """Verifies that merging shards with very different means gives the same R2 interval."""
    rng = np.random.default_rng(20260929)
    g = 1000
    m_c = 5
    n = g * m_c
    c_ids = np.repeat(np.arange(g, dtype=np.int64), m_c)

    # Shard A has y ~ 1e6 + N(0, 1), Shard B has y ~ N(0, 1)
    n_half = n // 2
    y_true_A = 1e6 + rng.normal(0.0, 1.0, size=n_half)
    y_pred_A = y_true_A + rng.normal(0.0, 0.4, size=n_half)
    y_true_B = rng.normal(0.0, 1.0, size=n - n_half)
    y_pred_B = y_true_B + rng.normal(0.0, 0.4, size=n - n_half)

    y_true = np.concatenate([y_true_A, y_true_B])
    y_pred = np.concatenate([y_pred_A, y_pred_B])

    shard_A = cluster_sketch.R2ClusterShard.from_data(
        y_true_A, y_pred_A, c_ids[:n_half]
    )
    shard_B = cluster_sketch.R2ClusterShard.from_data(
        y_true_B, y_pred_B, c_ids[n_half:]
    )
    merged = shard_A.merge(shard_B)
    full = cluster_sketch.R2ClusterShard.from_data(y_true, y_pred, c_ids)

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low_merged, high_merged, _, status_m, detail_m, _, _ = (
        cluster_bound.compute_r2_interval(
            merged.e2_totals,
            merged.y_totals,
            merged.y2_totals,
            merged.cluster_counts,
            level=0.95,
            m_item=4.0,
            m_labels=4.0,
            kappa=1.0,
            kappa_labels=1.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )

    low_full, high_full, _, status_f, detail_f, _, _ = (
        cluster_bound.compute_r2_interval(
            full.e2_totals,
            full.y_totals,
            full.y2_totals,
            full.cluster_counts,
            level=0.95,
            m_item=4.0,
            m_labels=4.0,
            kappa=1.0,
            kappa_labels=1.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )

    self.assertEqual(status_m, ind_interval.Status.UNREFUTED)
    self.assertEqual(status_f, ind_interval.Status.UNREFUTED)
    assert detail_m is not None and detail_f is not None

    np.testing.assert_allclose(low_merged, low_full, rtol=1e-9)
    np.testing.assert_allclose(high_merged, high_full, rtol=1e-9)
    np.testing.assert_allclose(detail_m.v_hat, detail_f.v_hat, rtol=1e-9)

  def test_r2_shard_merge_commutative_and_associative_after_recenter(self):
    """R2ClusterShard.merge is commutative and associative after recenter to common ref (rtol 1e-12)."""
    rng = np.random.default_rng(20260930)
    c_ids_a = np.array([0, 1, 2, 3], dtype=np.int64)
    c_ids_b = np.array([2, 3, 4, 5], dtype=np.int64)
    c_ids_c = np.array([1, 3, 5, 6], dtype=np.int64)

    y_t_a = rng.normal(10.0, 2.0, size=len(c_ids_a))
    y_p_a = y_t_a + rng.normal(0.0, 0.5, size=len(c_ids_a))
    shard_a = cluster_sketch.R2ClusterShard.from_data(
        y_t_a, y_p_a, c_ids_a, ref=10.0
    )

    y_t_b = rng.normal(100.0, 5.0, size=len(c_ids_b))
    y_p_b = y_t_b + rng.normal(0.0, 0.5, size=len(c_ids_b))
    shard_b = cluster_sketch.R2ClusterShard.from_data(
        y_t_b, y_p_b, c_ids_b, ref=100.0
    )

    y_t_c = rng.normal(-50.0, 3.0, size=len(c_ids_c))
    y_p_c = y_t_c + rng.normal(0.0, 0.5, size=len(c_ids_c))
    shard_c = cluster_sketch.R2ClusterShard.from_data(
        y_t_c, y_p_c, c_ids_c, ref=-50.0
    )

    common_ref = 0.0

    # Commutativity: a.merge(b) vs b.merge(a) after recenter to common ref
    ab = shard_a.merge(shard_b).recenter(common_ref)
    ba = shard_b.merge(shard_a).recenter(common_ref)

    np.testing.assert_array_equal(ab.cluster_ids, ba.cluster_ids)
    np.testing.assert_array_equal(ab.cluster_counts, ba.cluster_counts)
    np.testing.assert_allclose(
        ab.e2_totals, ba.e2_totals, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(ab.y_totals, ba.y_totals, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        ab.y2_totals, ba.y2_totals, rtol=1e-12, atol=1e-12
    )

    # Associativity: (a.merge(b)).merge(c) vs a.merge(b.merge(c)) after recenter
    abc = shard_a.merge(shard_b).merge(shard_c).recenter(common_ref)
    a_bc = shard_a.merge(shard_b.merge(shard_c)).recenter(common_ref)

    np.testing.assert_array_equal(abc.cluster_ids, a_bc.cluster_ids)
    np.testing.assert_array_equal(abc.cluster_counts, a_bc.cluster_counts)
    np.testing.assert_allclose(
        abc.e2_totals, a_bc.e2_totals, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        abc.y_totals, a_bc.y_totals, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        abc.y2_totals, a_bc.y2_totals, rtol=1e-12, atol=1e-12
    )

  def test_r2_shard_recenter_chaining(self):
    """recenter(a) followed by recenter(b) equals recenter(b) (rtol 1e-12)."""
    rng = np.random.default_rng(20260931)
    c_ids = np.repeat(np.arange(10, dtype=np.int64), 3)
    y_true = rng.normal(50.0, 10.0, size=len(c_ids))
    y_pred = y_true + rng.normal(0.0, 1.0, size=len(c_ids))

    shard = cluster_sketch.R2ClusterShard.from_data(
        y_true, y_pred, c_ids, ref=25.0
    )

    ref_a = 1e4
    ref_b = -3.5e5

    chained = shard.recenter(ref_a).recenter(ref_b)
    direct = shard.recenter(ref_b)

    self.assertEqual(chained.ref, direct.ref)
    np.testing.assert_array_equal(chained.cluster_ids, direct.cluster_ids)
    np.testing.assert_array_equal(chained.cluster_counts, direct.cluster_counts)
    np.testing.assert_allclose(
        chained.e2_totals, direct.e2_totals, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        chained.y_totals, direct.y_totals, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        chained.y2_totals, direct.y2_totals, rtol=1e-12, atol=1e-12
    )

  def test_r2_shard_merge_refs_0_and_1e6_matches_full(self):
    """Merging shards with explicit refs 0 and 1e6 gives same interval as full data (rtol 1e-9)."""
    rng = np.random.default_rng(20260932)
    g = 1000
    m_c = 5
    n = g * m_c
    c_ids = np.repeat(np.arange(g, dtype=np.int64), m_c)

    y_true = rng.normal(500.0, 20.0, size=n)
    y_pred = y_true + rng.normal(0.0, 5.0, size=n)

    n_half = n // 2
    shard_0 = cluster_sketch.R2ClusterShard.from_data(
        y_true[:n_half], y_pred[:n_half], c_ids[:n_half], ref=0.0
    )
    shard_1e6 = cluster_sketch.R2ClusterShard.from_data(
        y_true[n_half:], y_pred[n_half:], c_ids[n_half:], ref=1e6
    )

    merged = shard_0.merge(shard_1e6)
    full = cluster_sketch.R2ClusterShard.from_data(y_true, y_pred, c_ids)

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low_m, high_m, _, _, _, _, _ = cluster_bound.compute_r2_interval(
        merged.e2_totals,
        merged.y_totals,
        merged.y2_totals,
        merged.cluster_counts,
        level=0.95,
        m_item=4.0,
        m_labels=4.0,
        kappa=1.0,
        kappa_labels=1.0,
        run_kappa_check_fn=dummy_kappa,
    )
    low_f, high_f, _, _, _, _, _ = cluster_bound.compute_r2_interval(
        full.e2_totals,
        full.y_totals,
        full.y2_totals,
        full.cluster_counts,
        level=0.95,
        m_item=4.0,
        m_labels=4.0,
        kappa=1.0,
        kappa_labels=1.0,
        run_kappa_check_fn=dummy_kappa,
    )

    self.assertTrue(math.isfinite(low_m))
    self.assertTrue(math.isfinite(high_m))
    np.testing.assert_allclose(low_m, low_f, rtol=1e-9)
    np.testing.assert_allclose(high_m, high_f, rtol=1e-9)

  def test_r2_monte_carlo_sanity(self):
    rng = np.random.default_rng(20260928)
    g = 2000
    m_c = 5
    n = g * m_c
    c_ids = np.repeat(np.arange(g, dtype=np.int64), m_c)
    true_r2 = 0.8

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low_misses = 0
    high_misses = 0
    counts = np.full(g, m_c, dtype=np.int64)

    reps = 200 if _LONG else 40
    for _ in range(reps):
      gamma = rng.normal(0.0, math.sqrt(0.5), size=g)
      eps = rng.normal(0.0, math.sqrt(0.5), size=n)
      y_true = gamma[c_ids] + eps
      eta = rng.normal(0.0, math.sqrt(0.2), size=n)
      y_pred = y_true + eta
      e = y_true - y_pred

      e2 = e**2
      y = y_true
      y2 = y_true**2

      e2_totals = np.bincount(c_ids, weights=e2, minlength=g).astype(np.float64)
      y_totals = np.bincount(c_ids, weights=y, minlength=g).astype(np.float64)
      y2_totals = np.bincount(c_ids, weights=y2, minlength=g).astype(np.float64)

      low, high, _, status, _, _, _ = cluster_bound.compute_r2_interval(
          e2_totals,
          y_totals,
          y2_totals,
          counts,
          level=0.95,
          m_item=16.0,
          m_labels=16.0,
          kappa=1.0,
          kappa_labels=1.0,
          run_kappa_check_fn=dummy_kappa,
      )
      self.assertEqual(status, ind_interval.Status.UNREFUTED)
      if true_r2 < low:
        low_misses += 1
      if true_r2 > high:
        high_misses += 1

    # In long mode (reps=200), max 2 misses allowed.
    # In short mode (reps=40), nominal one-sided miss rate is at most alpha/2 = 0.025.
    # We allow ceil(reps * (p + 3*sqrt(p*(1-p)/reps))) = ceil(40 * (0.025 + 3*sqrt(0.025*0.975/40))) = 4.
    alpha_half = 0.025
    max_misses = (
        2
        if _LONG
        else int(
            math.ceil(
                reps
                * (
                    alpha_half
                    + 3.0 * math.sqrt(alpha_half * (1.0 - alpha_half) / float(reps))
                )
            )
        )
    )
    self.assertLessEqual(
        low_misses, max_misses, f"Too many lower misses: {low_misses}"
    )
    self.assertLessEqual(
        high_misses, max_misses, f"Too many upper misses: {high_misses}"
    )

  def test_r2_m_vs_m_item_equal_sizes(self):
    g = 2000
    counts = np.full(g, 5, dtype=np.int64)
    rng = np.random.default_rng(42)
    e2_totals = rng.uniform(0.1, 1.0, size=g) * counts
    y_totals = rng.standard_normal(g) * counts
    y2_totals = (y_totals**2 / counts) + rng.uniform(0.1, 1.0, size=g) * counts

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low_m, high_m, level_m, status_m, detail_m, _, _ = (
        cluster_bound.compute_r2_interval(
            e2_totals,
            y_totals,
            y2_totals,
            counts,
            level=0.95,
            m=4.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )
    low_item, high_item, level_item, status_item, detail_item, _, _ = (
        cluster_bound.compute_r2_interval(
            e2_totals,
            y_totals,
            y2_totals,
            counts,
            level=0.95,
            m_item=4.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )

    self.assertEqual(status_m, ind_interval.Status.UNREFUTED)
    self.assertEqual(status_item, ind_interval.Status.UNREFUTED)
    self.assertEqual(level_m, level_item)
    self.assertAlmostEqual(low_m, low_item, places=12)
    self.assertAlmostEqual(high_m, high_item, places=12)
    assert detail_m is not None and detail_item is not None
    self.assertAlmostEqual(detail_m.M_c_A, detail_item.M_c_A, places=12)
    self.assertAlmostEqual(detail_m.M_c_q, detail_item.M_c_q, places=12)

  def test_r2_m_vs_m_item_unequal_sizes(self):
    g = 5000
    rng = np.random.default_rng(123)
    counts = rng.integers(2, 6, size=g, dtype=np.int64)
    e2_totals = rng.uniform(0.1, 1.0, size=g) * counts
    y_totals = rng.standard_normal(g) * counts
    y2_totals = (y_totals**2 / counts) + rng.uniform(0.1, 1.0, size=g) * counts

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low_m, high_m, _, status_m, detail_m, _, _ = (
        cluster_bound.compute_r2_interval(
            e2_totals,
            y_totals,
            y2_totals,
            counts,
            level=0.95,
            m=4.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )
    low_item, high_item, _, status_item, detail_item, _, _ = (
        cluster_bound.compute_r2_interval(
            e2_totals,
            y_totals,
            y2_totals,
            counts,
            level=0.95,
            m_item=4.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )

    self.assertEqual(status_m, ind_interval.Status.UNREFUTED)
    self.assertEqual(status_item, ind_interval.Status.UNREFUTED)
    self.assertTrue(math.isfinite(low_m) and math.isfinite(high_m))
    self.assertTrue(math.isfinite(low_item) and math.isfinite(high_item))
    assert detail_m is not None and detail_item is not None
    self.assertGreater(detail_item.M_c_A, detail_m.M_c_A)
    width_m = high_m - low_m
    width_item = high_item - low_item
    self.assertGreaterEqual(width_item, width_m)

  def test_r2_m_with_m_labels_none_uses_unconverted_m_for_labels(self):
    g = 100
    rng = np.random.default_rng(456)
    counts = rng.integers(1, 10, size=g, dtype=np.int64)
    e2_totals = rng.uniform(0.1, 1.0, size=g) * counts
    y_totals = rng.standard_normal(g) * counts
    y2_totals = (y_totals**2 / counts) + rng.uniform(0.1, 1.0, size=g) * counts

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    _, _, _, _, detail, _, _ = cluster_bound.compute_r2_interval(
        e2_totals,
        y_totals,
        y2_totals,
        counts,
        level=0.95,
        m=5.0,
        m_labels=None,
        run_kappa_check_fn=dummy_kappa,
    )
    assert detail is not None
    self.assertEqual(detail.M_c_A, 5.0)
    self.assertEqual(detail.M_c_q, 5.0)

  def test_r2_large_mean_relative_accuracy(self):
    """Verifies that raw-data R2 entry point matches shifted-by-1e6 data to rtol 1e-9."""
    rng = np.random.default_rng(20260929)
    g = 1000
    m_c = 10
    n = g * m_c
    c_ids = np.repeat(np.arange(g, dtype=np.int64), m_c)

    eps = rng.normal(0.0, 1.0, size=n)
    eta = rng.normal(0.0, 0.4, size=n)

    # 1. Unshifted data
    y_true_un = eps
    y_pred_un = eps + eta
    shard_un = cluster_sketch.R2ClusterShard.from_data(
        y_true_un, y_pred_un, c_ids
    )

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low_un, high_un, _, status_un, detail_un, _, _ = (
        cluster_bound.compute_r2_interval(
            shard_un.e2_totals,
            shard_un.y_totals,
            shard_un.y2_totals,
            shard_un.cluster_counts,
            level=0.95,
            m_item=4.0,
            m_labels=4.0,
            kappa=1.0,
            kappa_labels=1.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )

    # 2. Shifted data: y = 1e6 + eps, y_pred = 1e6 + eps + eta
    shift = 1e6
    y_true_sh = shift + eps
    y_pred_sh = shift + eps + eta
    shard_sh = cluster_sketch.R2ClusterShard.from_data(
        y_true_sh, y_pred_sh, c_ids
    )

    low_sh, high_sh, _, status_sh, detail_sh, _, _ = (
        cluster_bound.compute_r2_interval(
            shard_sh.e2_totals,
            shard_sh.y_totals,
            shard_sh.y2_totals,
            shard_sh.cluster_counts,
            level=0.95,
            m_item=4.0,
            m_labels=4.0,
            kappa=1.0,
            kappa_labels=1.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )

    self.assertEqual(status_un, ind_interval.Status.UNREFUTED)
    self.assertEqual(status_sh, ind_interval.Status.UNREFUTED)
    assert detail_un is not None and detail_sh is not None

    # Assert endpoints match to rtol 1e-9
    np.testing.assert_allclose(low_sh, low_un, rtol=1e-9)
    np.testing.assert_allclose(high_sh, high_un, rtol=1e-9)
    np.testing.assert_allclose(detail_sh.v_hat, detail_un.v_hat, rtol=1e-9)

  def test_r2_unshifted_precomputed_totals_allowed(self):
    """Verifies that the unshifted precomputed-totals path is still allowed without refusal."""
    g = 1000
    counts = np.full(g, 5, dtype=np.int64)
    # Precomputed totals with non-zero mean without pre-centering
    y_totals = np.full(g, 50.0)  # mean per item is 10.0
    y2_totals = np.full(g, 550.0)  # variance per item is 10.0
    e2_totals = np.full(g, 5.0)

    def dummy_kappa(res, counts, k):
      return refutation.RefutationResult(
          refutation.Refutation.NOT_REFUTED, 0.0, 1.96
      )

    low, high, level_out, status, detail, _, _ = (
        cluster_bound.compute_r2_interval(
            e2_totals,
            y_totals,
            y2_totals,
            counts,
            level=0.95,
            m_item=4.0,
            m_labels=4.0,
            kappa=1.0,
            kappa_labels=1.0,
            run_kappa_check_fn=dummy_kappa,
        )
    )

    self.assertEqual(status, ind_interval.Status.UNREFUTED)
    self.assertEqual(level_out, 0.95)
    self.assertTrue(math.isfinite(low) and math.isfinite(high))
    assert detail is not None
    self.assertGreater(detail.v_hat, 0.0)


if __name__ == "__main__":
  absltest.main()
