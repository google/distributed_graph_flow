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

"""Tests for cluster_sketch.py."""

import dataclasses
import math

from absl.testing import absltest
import numpy as np

from dgf.src.stats import cluster_bound
from dgf.src.stats import cluster_sketch
from dgf.src.stats.independent import metrics as ind_metrics


class ClusterSketchTest(absltest.TestCase):

  def test_default_m_c_sizes_1_3(self):
    """Sizes (1, 3), m_item = M for M in {4, 16}: default_m_c equals 1.5*M to 1e-12 (branch a)."""
    counts = np.array([1, 3], dtype=np.int64)
    totals = np.array([1.0, 3.0], dtype=np.float64)
    ids = np.array([0, 1], dtype=np.int64)
    shard = cluster_sketch.PartialClusterShard(
        cluster_ids=ids,
        cluster_totals=totals,
        cluster_counts=counts,
    )
    sk = shard.to_cluster_sketch()
    self.assertEqual(sk.schema_version, 3)
    self.assertEqual(sk.max_cluster_size, 3)
    self.assertEqual(sk.sum_count_sq, 10)  # 1^2 + 3^2 = 10

    for m in (4.0, 16.0):
      expected = 1.5 * m
      self.assertAlmostEqual(sk.default_m_c(m), expected, places=12)

  def test_default_m_c_sizes_1_3_m_1(self):
    """Sizes (1, 3), M = 1: default_m_c equals 1.25 to 1e-12 (branch b wins: r = 1.5, cv2 = 0.25)."""
    counts = np.array([1, 3], dtype=np.int64)
    totals = np.array([1.0, 3.0], dtype=np.float64)
    ids = np.array([0, 1], dtype=np.int64)
    shard = cluster_sketch.PartialClusterShard(
        cluster_ids=ids,
        cluster_totals=totals,
        cluster_counts=counts,
    )
    sk = shard.to_cluster_sketch()
    self.assertAlmostEqual(sk.default_m_c(1.0), 1.25, places=12)

  def test_default_m_c_false_tail_example(self):
    """The pooled item premise holds with M = 1.5 and the old formula gave 1.778 < 2."""
    g = 200
    counts = np.array(
        [1 if i % 2 == 0 else 2 for i in range(g)], dtype=np.int64
    )
    totals = np.array(
        [0.0 if i % 2 == 0 else 2.0 for i in range(g)], dtype=np.float64
    )
    ids = np.arange(g, dtype=np.int64)
    shard = cluster_sketch.PartialClusterShard(
        cluster_ids=ids,
        cluster_totals=totals,
        cluster_counts=counts,
    )
    sk = shard.to_cluster_sketch()
    self.assertAlmostEqual(sk.default_m_c(1.5), 2.0, places=12)
    m_hat = sk.num_clusters * sk.sum_s2 / (sk.sum_s**2)
    self.assertAlmostEqual(m_hat, 2.0, places=12)
    self.assertLessEqual(m_hat, sk.default_m_c(1.5) + 1e-12)

  def test_default_m_c_equal_sizes_exact(self):
    """Equal sizes (10 clusters of 7): default_m_c(M) == M exactly."""
    counts = np.full(10, 7, dtype=np.int64)
    totals = np.ones(10, dtype=np.float64)
    ids = np.arange(10, dtype=np.int64)
    shard = cluster_sketch.PartialClusterShard(
        cluster_ids=ids,
        cluster_totals=totals,
        cluster_counts=counts,
    )
    sk = shard.to_cluster_sketch()
    for m in (1.05, 4.0, 16.0):
      self.assertEqual(sk.default_m_c(m), m)

  def test_merge_invariants(self):
    """Merge: n_max, sum_count_sq, default_m_c match union; merged kappa is max."""
    counts_a = np.array([3, 5], dtype=np.int64)
    totals_a = np.array([1.0, 2.0], dtype=np.float64)
    ids_a = np.array([0, 1], dtype=np.int64)
    shard_a = cluster_sketch.PartialClusterShard(ids_a, totals_a, counts_a)
    sk_a = shard_a.to_cluster_sketch(kappa_cluster=1.5)

    counts_b = np.array([2, 8], dtype=np.int64)
    totals_b = np.array([3.0, 4.0], dtype=np.float64)
    ids_b = np.array([2, 3], dtype=np.int64)
    shard_b = cluster_sketch.PartialClusterShard(ids_b, totals_b, counts_b)
    sk_b = shard_b.to_cluster_sketch(kappa_cluster=2.5)

    # Merged sketch
    sk_merged = sk_a.merge(sk_b)

    # Union sketch built in one go
    all_counts = np.concatenate([counts_a, counts_b])
    all_totals = np.concatenate([totals_a, totals_b])
    all_ids = np.concatenate([ids_a, ids_b])
    shard_union = cluster_sketch.PartialClusterShard(
        all_ids, all_totals, all_counts
    )
    sk_union = shard_union.to_cluster_sketch(kappa_cluster=2.5)

    self.assertEqual(sk_merged.max_cluster_size, sk_union.max_cluster_size)
    self.assertEqual(sk_merged.max_cluster_size, 8)
    self.assertEqual(sk_merged.sum_count_sq, sk_union.sum_count_sq)
    self.assertEqual(sk_merged.sum_count_sq, 3**2 + 5**2 + 2**2 + 8**2)
    self.assertEqual(sk_merged.kappa_cluster, 2.5)
    self.assertEqual(
        sk_merged.kappa_cluster, max(sk_a.kappa_cluster, sk_b.kappa_cluster)
    )
    self.assertEqual(sk_merged.default_m_c(4.0), sk_union.default_m_c(4.0))

  def test_schema_version_and_validation_failures(self):
    """Schema: version 2 and each validation failure raise ValueError."""
    # Version 2 is rejected
    with self.assertRaisesRegex(ValueError, "schema_version must be 3"):
      cluster_sketch.ClusterSketch(
          n_items=10,
          num_clusters=2,
          sum_s=5.0,
          sum_s2=15.0,
          top_32=(3.0, 2.0),
          max_cluster_size=6,
          sum_count_sq=52,
          schema_version=2,
      )

    # max_cluster_size not int
    with self.assertRaises(ValueError):
      cluster_sketch.ClusterSketch(
          n_items=10,
          num_clusters=2,
          sum_s=5.0,
          sum_s2=15.0,
          top_32=(3.0, 2.0),
          max_cluster_size=6.0,  # pyrefly: ignore[bad-argument-type]
          sum_count_sq=52,
          schema_version=3,
      )

    # sum_count_sq not int
    with self.assertRaises(ValueError):
      cluster_sketch.ClusterSketch(
          n_items=10,
          num_clusters=2,
          sum_s=5.0,
          sum_s2=15.0,
          top_32=(3.0, 2.0),
          max_cluster_size=6,
          sum_count_sq=52.0,  # pyrefly: ignore[bad-argument-type]
          schema_version=3,
      )

    # max_cluster_size < 1
    with self.assertRaises(ValueError):
      cluster_sketch.ClusterSketch(
          n_items=10,
          num_clusters=2,
          sum_s=5.0,
          sum_s2=15.0,
          top_32=(3.0, 2.0),
          max_cluster_size=0,
          sum_count_sq=52,
          schema_version=3,
      )

    # max_cluster_size > n_items
    with self.assertRaises(ValueError):
      cluster_sketch.ClusterSketch(
          n_items=10,
          num_clusters=2,
          sum_s=5.0,
          sum_s2=15.0,
          top_32=(3.0, 2.0),
          max_cluster_size=11,
          sum_count_sq=52,
          schema_version=3,
      )

    # max_cluster_size * num_clusters < n_items
    with self.assertRaises(ValueError):
      cluster_sketch.ClusterSketch(
          n_items=10,
          num_clusters=2,
          sum_s=5.0,
          sum_s2=15.0,
          top_32=(3.0, 2.0),
          max_cluster_size=4,  # 4 * 2 = 8 < 10
          sum_count_sq=52,
          schema_version=3,
      )

    # sum_count_sq < ceil(n_items^2 / num_clusters) (Cauchy-Schwarz)
    # n_items = 10, num_clusters = 2 -> ceil(100 / 2) = 50
    with self.assertRaises(ValueError):
      cluster_sketch.ClusterSketch(
          n_items=10,
          num_clusters=2,
          sum_s=5.0,
          sum_s2=15.0,
          top_32=(3.0, 2.0),
          max_cluster_size=6,
          sum_count_sq=48,  # < 50
          schema_version=3,
      )

    # sum_count_sq > max_cluster_size * n_items
    # 6 * 10 = 60
    with self.assertRaises(ValueError):
      cluster_sketch.ClusterSketch(
          n_items=10,
          num_clusters=2,
          sum_s=5.0,
          sum_s2=15.0,
          top_32=(3.0, 2.0),
          max_cluster_size=6,
          sum_count_sq=62,  # > 60
          schema_version=3,
      )

  def test_cluster_interval_m_none_uses_default_m_c(self):
    """cluster_interval with m=None reports m_declared == default_m_c(...)."""
    # Unequal cluster sizes: 10 of size 2, 10 of size 6 (G = 20, n = 80)
    data = np.ones(80, dtype=np.float64)
    cluster_ids = []
    for c in range(10):
      cluster_ids.extend([c] * 2)
    for c in range(10, 20):
      cluster_ids.extend([c] * 6)
    cluster_ids = np.array(cluster_ids, dtype=np.int64)

    sk = cluster_sketch.ClusterSketch.from_data(data, cluster_ids, metric="mse")
    res = cluster_bound.cluster_interval(sk, metric="mse", m=None)

    expected_m = sk.default_m_c(ind_metrics.get_default_m("mse"))
    self.assertEqual(res.m_declared, expected_m)
    # Also verify from raw data
    res_raw = cluster_bound.cluster_interval(
        data, cluster_ids, metric="mse", m=None
    )
    self.assertEqual(res_raw.m_declared, expected_m)

  def test_to_independent_sketch(self):
    """Verifies that to_independent_sketch produces an IndependentSketch."""
    from dgf.src.stats.independent import sketch as ind_sketch
    counts = np.array([2, 3], dtype=np.int64)
    totals = np.array([1.0, 4.0], dtype=np.float64)
    ids = np.array([0, 1], dtype=np.int64)
    shard = cluster_sketch.PartialClusterShard(ids, totals, counts)
    sk = shard.to_cluster_sketch(kappa_cluster=1.5)
    ind_sk = sk.to_independent_sketch()
    self.assertIsInstance(ind_sk, ind_sketch.IndependentSketch)
    self.assertEqual(ind_sk.n, 2)
    self.assertEqual(ind_sk.sum_s, 5.0)
    self.assertIsNotNone(ind_sk.effective_n)
    assert ind_sk.effective_n is not None
    self.assertAlmostEqual(ind_sk.effective_n, 2.0 / 1.5, places=12)


if __name__ == "__main__":
  absltest.main()

