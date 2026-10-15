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

"""Tests for declare.py."""

import json
import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dgf.src.stats import cluster_bound
from dgf.src.stats import cluster_sketch
from dgf.src.stats import declare
from dgf.src.stats import graph
from dgf.src.stats import temporal
from dgf.src.stats.independent import interval as ind_interval

# Set DGF_STATS_LONG_TESTS=1 (blaze test --test_env=...) for the full-size runs.
_LONG = os.environ.get("DGF_STATS_LONG_TESTS") == "1"


class DeclareTest(parameterized.TestCase):

  def test_hand_example(self):
    summands = [1.0, 1.0, 1.0, 5.0]
    res = declare.estimate_m(summands, num_resamples=500, seed=42)
    # n=4, sum(s)=8, sum(s^2)=1+1+1+25=28, m_point = 4*28/64 = 1.75
    self.assertAlmostEqual(res.m_point, 1.75, places=12)
    self.assertEqual(res.num_units, 4)
    self.assertEqual(res.num_items, 4)
    self.assertGreaterEqual(res.m_upper, 1.75)

  def test_cluster_resampling_singleton_equals_item(self):
    rng = np.random.default_rng(123)
    summands = rng.uniform(0.5, 2.5, size=50)
    singletons = np.arange(50)

    res_item = declare.estimate_m(
        summands, cluster_ids=None, level=0.95, num_resamples=500, seed=999
    )
    res_clust = declare.estimate_m(
        summands,
        cluster_ids=singletons,
        level=0.95,
        num_resamples=500,
        seed=999,
    )

    self.assertAlmostEqual(res_item.m_point, res_clust.m_point, places=12)
    self.assertAlmostEqual(res_item.m_upper, res_clust.m_upper, places=12)
    self.assertAlmostEqual(
        res_item.max_unit_share, res_clust.max_unit_share, places=12
    )
    self.assertAlmostEqual(
        res_item.top1pct_share, res_clust.top1pct_share, places=12
    )
    self.assertAlmostEqual(
        res_item.m_without_top1pct, res_clust.m_without_top1pct, places=12
    )

  def test_gaussian_calibration(self):
    # e ~ N(0, 1), s = e^2 has true M = 3.
    # Over 200 seeds (or 20 in short mode), check that fraction with m_upper >= 3 is at least nominal.
    n = 20000 if _LONG else 5000
    covered = 0
    num_seeds = 200 if _LONG else 20
    num_resamples = 500 if _LONG else 100

    for s_idx in range(num_seeds):
      rng = np.random.default_rng(10000 + s_idx)
      e = rng.standard_normal(n)
      s = e**2
      res = declare.estimate_m(
          s, level=0.95, num_resamples=num_resamples, seed=s_idx
      )
      if res.m_upper >= 3.0:
        covered += 1

    coverage_rate = covered / num_seeds
    # Nominal coverage rate is 0.95. In long mode, 0.90 is asserted.
    # In short mode (num_seeds=20), p=0.95 gives 0.95 - 3*sqrt(0.95*0.05/20) ~ 0.803.
    min_cov = (
        0.90
        if _LONG
        else max(0.0, 0.95 - 3.0 * math.sqrt(0.95 * 0.05 / float(num_seeds)))
    )
    self.assertGreaterEqual(
        coverage_rate,
        min_cov,
        f"Coverage rate {coverage_rate:.3f} below expected {min_cov:.3f}",
    )

  def test_correlated_archive(self):
    # Build clusters of 20 items sharing a common factor, e = a_c + b_i with ICC 0.8
    rng = np.random.default_rng(42)
    num_clusters = 50
    items_per_cluster = 20
    n = num_clusters * items_per_cluster
    cluster_ids = np.repeat(np.arange(num_clusters), items_per_cluster)

    a_c = rng.normal(0.0, math.sqrt(0.8), size=num_clusters)
    b_i = rng.normal(0.0, math.sqrt(0.2), size=n)
    e = a_c[cluster_ids] + b_i
    s = e**2

    # Resample with cluster vs item units and check spread of M*
    # using bootstrap distributions
    num_resamples = 1000
    res_clust = declare.estimate_m(
        s,
        cluster_ids=cluster_ids,
        level=0.95,
        num_resamples=num_resamples,
        seed=123,
    )
    res_item = declare.estimate_m(
        s, cluster_ids=None, level=0.95, num_resamples=num_resamples, seed=123
    )

    # Re-draw bootstrap distribution quantiles (95% - 5%) to compare spread
    rng_c = np.random.default_rng(123)
    _, inv = np.unique(cluster_ids, return_inverse=True)
    c_n = np.bincount(inv).astype(np.int64)
    c_s = np.bincount(inv, weights=s).astype(np.float64)
    c_s2 = np.bincount(inv, weights=s**2).astype(np.float64)
    c_idx = rng_c.integers(0, num_clusters, size=(num_resamples, num_clusters))
    boot_m_c = (np.sum(c_n[c_idx], axis=1) * np.sum(c_s2[c_idx], axis=1)) / (
        np.sum(c_s[c_idx], axis=1) ** 2
    )
    spread_c = float(np.quantile(boot_m_c, 0.95) - np.quantile(boot_m_c, 0.05))

    rng_i = np.random.default_rng(123)
    i_idx = rng_i.integers(0, n, size=(num_resamples, n))
    boot_m_i = (n * np.sum((s**2)[i_idx], axis=1)) / (
        np.sum(s[i_idx], axis=1) ** 2
    )
    spread_i = float(np.quantile(boot_m_i, 0.95) - np.quantile(boot_m_i, 0.05))

    self.assertGreater(
        spread_c,
        spread_i,
        f"Cluster spread {spread_c:.4f} should exceed item spread"
        f" {spread_i:.4f}",
    )

  def test_heavy_tail(self):
    # One summand holds 50% of sum(s^2).
    # Check max_unit_share ~ 0.5, and m_without_top1pct < m_point.
    n = 1000
    s = np.ones(n, dtype=np.float64)
    # Total sum(s^2) for 999 ones is 999.
    # To have one element hold 50% of total sum(s^2):
    # s_0^2 = 999 => s_0 = sqrt(999) approx 31.60696
    s[0] = math.sqrt(n - 1)
    res = declare.estimate_m(s, num_resamples=500, seed=0)

    self.assertAlmostEqual(res.max_unit_share, 0.5, delta=0.01)
    self.assertLess(res.m_without_top1pct, res.m_point)
    self.assertAlmostEqual(res.m_without_top1pct, 1.0, delta=0.05)

  def test_estimate_m_r2_gaussian(self):
    # y and e Gaussian with n = 50000 -> m_point of both parts within 0.3 of 3
    rng = np.random.default_rng(2026)
    n = 50000
    y_true = rng.normal(0.0, 1.0, size=n)
    e = rng.normal(0.0, 0.5, size=n)
    y_pred = y_true - e

    res = declare.estimate_m_r2(y_true, y_pred, num_resamples=500, seed=42)
    self.assertAlmostEqual(res.m_errors.m_point, 3.0, delta=0.3)
    self.assertAlmostEqual(res.m_labels.m_point, 3.0, delta=0.3)

  def test_validation_errors(self):
    # Negative summands
    with self.assertRaises(ValueError):
      declare.estimate_m([1.0, -0.5, 2.0])
    # NaN
    with self.assertRaises(ValueError):
      declare.estimate_m([1.0, float("nan"), 2.0])
    # Inf
    with self.assertRaises(ValueError):
      declare.estimate_m([1.0, float("inf"), 2.0])
    # Length mismatch
    with self.assertRaises(ValueError):
      declare.estimate_m([1.0, 2.0], cluster_ids=[0, 1, 2])
    # sum(s) == 0
    with self.assertRaises(ValueError):
      declare.estimate_m([0.0, 0.0, 0.0])
    # level out of bounds
    with self.assertRaises(ValueError):
      declare.estimate_m([1.0, 2.0], level=1.0)
    with self.assertRaises(ValueError):
      declare.estimate_m([1.0, 2.0], level=0.0)
    # num_resamples < 100
    with self.assertRaises(ValueError):
      declare.estimate_m([1.0, 2.0], num_resamples=50)

    # estimate_m_r2 validation errors
    with self.assertRaises(ValueError):
      declare.estimate_m_r2([1.0, 2.0], [1.0])
    with self.assertRaises(ValueError):
      declare.estimate_m_r2([], [])
    with self.assertRaises(ValueError):
      declare.estimate_m_r2([float("nan")], [1.0])

  # --- Declaration Serialization Tests ---

  def test_declaration_json_round_trip(self):
    d = declare.Declaration(
        metric="mse",
        m=4.5,
        kappa=1.2,
        source="archive_2025_a",
        archive_fingerprint="abc123hash",
        created="2026-09-28",
        method="estimate_m units=1000",
        diagnostics={"m_point": 3.2, "top1pct_share": 0.08},
    )
    json_text = d.to_json()
    d_loaded = declare.Declaration.from_json(json_text)
    self.assertEqual(d, d_loaded)

  def test_declaration_kwargs(self):
    d_mse = declare.Declaration(metric="mse", m=4.0, kappa=1.5)
    kw_mse = d_mse.kwargs()
    self.assertEqual(kw_mse, {"m_item": 4.0, "kappa": 1.5})
    self.assertNotIn("m_labels", kw_mse)
    self.assertNotIn("kappa_labels", kw_mse)

    d_r2 = declare.Declaration(
        metric="r2", m=4.0, m_labels=3.5, kappa=1.2, kappa_labels=1.1
    )
    kw_r2 = d_r2.kwargs()
    self.assertEqual(
        kw_r2,
        {"m_item": 4.0, "m_labels": 3.5, "kappa": 1.2, "kappa_labels": 1.1},
    )

  def test_from_estimate_picks_m_upper(self):
    s = [1.0, 2.0, 3.0, 4.0] * 10
    est = declare.estimate_m(s, num_resamples=500, seed=0)
    d = declare.Declaration.from_estimate("mse", est, source="test")
    self.assertEqual(d.metric, "mse")
    self.assertEqual(d.m, est.m_upper)
    self.assertIsNone(d.kappa)
    self.assertEqual(d.source, "test")

    # R2
    y = np.array([1.0, 2.0, 3.0, 4.0] * 10)
    yp = np.array([0.9, 2.1, 2.8, 4.2] * 10)
    est_r2 = declare.estimate_m_r2(y, yp, num_resamples=500, seed=0)
    d_r2 = declare.Declaration.from_estimate("r2", est_r2, kappa=1.3)
    self.assertEqual(d_r2.metric, "r2")
    self.assertEqual(d_r2.m, est_r2.m_errors.m_upper)
    self.assertEqual(d_r2.m_labels, est_r2.m_labels.m_upper)
    self.assertEqual(d_r2.kappa, 1.3)
    self.assertEqual(d_r2.kappa_labels, 1.3)

  def test_declaration_validation_errors(self):
    # Unknown format_version
    with self.assertRaises(ValueError):
      declare.Declaration.from_json(
          json.dumps({"format_version": 2, "metric": "mse"})
      )
    # Unknown metric
    with self.assertRaises(ValueError):
      declare.Declaration.from_json(json.dumps({"metric": "unknown_metric"}))
    # m < 1
    with self.assertRaises(ValueError):
      declare.Declaration.from_json(json.dumps({"metric": "mse", "m": 0.5}))
    # kappa < 1
    with self.assertRaises(ValueError):
      declare.Declaration.from_json(json.dumps({"metric": "mse", "kappa": 0.8}))
    # m_labels on non-r2
    with self.assertRaises(ValueError):
      declare.Declaration.from_json(
          json.dumps({"metric": "mse", "m_labels": 3.0})
      )
    # kappa_labels on non-r2
    with self.assertRaises(ValueError):
      declare.Declaration.from_json(
          json.dumps({"metric": "mse", "kappa_labels": 1.5})
      )

  def test_save_and_load_declarations(self):
    temp_dir = self.create_tempdir()
    path = os.path.join(temp_dir, "declarations.json")

    decls = {
        "model_a_mse": declare.Declaration(metric="mse", m=3.5, kappa=1.1),
        "model_a_r2": declare.Declaration(
            metric="r2", m=3.5, m_labels=3.0, kappa=1.1, kappa_labels=1.0
        ),
    }

    declare.save_declarations(path, decls)
    loaded = declare.load_declarations(path)

    self.assertEqual(decls, loaded)

  def test_fingerprint(self):
    arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    arr2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # fingerprint converts to little-endian float64
    self.assertEqual(declare.fingerprint(arr1), declare.fingerprint(arr2))

  def test_kwargs_splat_matches_direct_call(self):
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 1.0, size=10000)
    d = declare.Declaration(metric="mse", m=4.0, kappa=1.2)

    res_splat = temporal.temporal_interval(
        residuals, num_windows=500, metric=d.metric, **d.kwargs()
    )
    res_direct = temporal.temporal_interval(
        residuals, num_windows=500, metric="mse", m_item=4.0, kappa=1.2
    )

    self.assertTrue(math.isfinite(res_splat.low))
    self.assertTrue(math.isfinite(res_splat.high))
    self.assertAlmostEqual(res_splat.low, res_direct.low, places=12)
    self.assertAlmostEqual(res_splat.high, res_direct.high, places=12)
    self.assertEqual(res_splat.status, res_direct.status)

  def test_bounded_memory_large_archive_determinism(self):
    # estimate_m on 300,000 items with num_resamples=200 runs,
    # and two calls with the same seed return exactly equal m_upper.
    rng = np.random.default_rng(2026)
    summands = rng.uniform(0.1, 2.0, size=300000)

    res1 = declare.estimate_m(summands, num_resamples=200, seed=42)
    res2 = declare.estimate_m(summands, num_resamples=200, seed=42)

    self.assertEqual(res1.m_upper, res2.m_upper)
    self.assertEqual(res1.m_point, res2.m_point)
    self.assertEqual(res1.num_items, 300000)
    self.assertEqual(res1.num_units, 300000)

  def test_m_item_unequal_sizes(self):
    # 1 cluster of 20 items, and 2000 clusters of 1 item -> G = 2001, n = 2020
    # n_max/m_bar ~ 19.8, M_c ~ 20.8 > M = 2.0, with G=2001 > n_A so interval is certified
    rng = np.random.default_rng(42)
    c_ids = np.concatenate(
        [np.zeros(20, dtype=np.int64), np.arange(1, 2001, dtype=np.int64)]
    )
    data = rng.uniform(0.1, 1.0, size=2020)
    m_val = 2.0
    res_m_item = cluster_bound.cluster_interval(
        data, c_ids, metric="mse", m_item=m_val
    )
    res_m = cluster_bound.cluster_interval(data, c_ids, metric="mse", m=m_val)

    sk = cluster_sketch.ClusterSketch.from_data(data, c_ids, metric="mse")
    expected_m_c = sk.default_m_c(m_val)
    self.assertAlmostEqual(res_m_item.m_declared, expected_m_c, places=10)
    self.assertGreater(expected_m_c, m_val)
    self.assertTrue(math.isfinite(res_m_item.low))
    self.assertTrue(math.isfinite(res_m_item.high))
    width_m_item = res_m_item.high - res_m_item.low
    width_m = res_m.high - res_m.low
    self.assertGreater(width_m_item, width_m)

  def test_m_item_equal_sizes(self):
    # 100 clusters of 10 items -> G = 100, n = 1000, n_max = 10, m_bar = 10
    rng = np.random.default_rng(42)
    c_ids = np.repeat(np.arange(100, dtype=np.int64), 10)
    data = rng.uniform(0.1, 1.0, size=1000)
    m_val = 4.0
    res_m_item = cluster_bound.cluster_interval(
        data, c_ids, metric="mse", m_item=m_val
    )
    sk = cluster_sketch.ClusterSketch.from_data(data, c_ids, metric="mse")
    expected_m_c = sk.default_m_c(m_val)
    self.assertAlmostEqual(expected_m_c, m_val, places=10)
    self.assertAlmostEqual(res_m_item.m_declared, m_val, places=10)

  @parameterized.named_parameters(
      ("cluster_interval", "cluster_interval"),
      ("compute_r2_interval", "compute_r2_interval"),
      ("temporal_interval", "temporal_interval"),
      ("temporal_interval_from_shard", "temporal_interval_from_shard"),
      ("temporal_interval_from_buckets", "temporal_interval_from_buckets"),
      ("graph_interval_from_shard", "graph_interval_from_shard"),
      ("graph_interval", "graph_interval"),
  )
  def test_mutual_exclusivity_raises_value_error(self, entry_point: str):
    rng = np.random.default_rng(42)
    n = 20
    g = 4
    c_ids = np.repeat(np.arange(g, dtype=np.int64), 5)
    data = rng.normal(0.0, 1.0, size=n)
    edges = np.column_stack([np.arange(g - 1), np.arange(1, g)])

    with self.assertRaises(ValueError):
      if entry_point == "cluster_interval":
        cluster_bound.cluster_interval(data, c_ids, m=4.0, m_item=4.0)
      elif entry_point == "compute_r2_interval":
        cluster_bound.compute_r2_interval(
            np.ones(g),
            np.ones(g),
            np.ones(g),
            np.full(g, 5),
            m=4.0,
            m_item=4.0,
            run_kappa_check_fn=lambda *a: None,
        )
      elif entry_point == "temporal_interval":
        temporal.temporal_interval(data, num_windows=g, m=4.0, m_item=4.0)
      elif entry_point == "temporal_interval_from_shard":
        shard = cluster_sketch.PartialClusterShard.from_data(data, c_ids, "mse")
        temporal.temporal_interval_from_shard(shard, m=4.0, m_item=4.0)
      elif entry_point == "temporal_interval_from_buckets":
        temporal.temporal_interval_from_buckets(
            np.full(g, 5), np.ones(g), num_windows=2, m=4.0, m_item=4.0
        )
      elif entry_point == "graph_interval_from_shard":
        shard = cluster_sketch.PartialClusterShard.from_data(data, c_ids, "mse")
        graph.graph_interval_from_shard(shard, edges, m=4.0, m_item=4.0)
      elif entry_point == "graph_interval":
        graph.graph_interval(data, c_ids, edges, m=4.0, m_item=4.0)

  def test_kwargs_splat_all_entry_points(self):
    rng = np.random.default_rng(42)
    archive = rng.uniform(0.1, 1.0, size=500)
    est = declare.estimate_m(archive, num_resamples=100, seed=42)
    decl = declare.Declaration.from_estimate("mse", est, source="test")

    # Evaluation data
    eval_data = rng.uniform(0.1, 1.0, size=400)
    g = 20
    cluster_ids = np.repeat(np.arange(g, dtype=np.int64), 20)
    cluster_edges = np.column_stack([np.arange(g - 1), np.arange(1, g)])

    # 1. cluster_interval
    res_cl_splat = cluster_bound.cluster_interval(
        eval_data, cluster_ids, metric="mse", **decl.kwargs()
    )
    res_cl_direct = cluster_bound.cluster_interval(
        eval_data, cluster_ids, metric="mse", m_item=decl.m
    )
    self.assertAlmostEqual(res_cl_splat.low, res_cl_direct.low, places=12)
    self.assertAlmostEqual(res_cl_splat.high, res_cl_direct.high, places=12)
    self.assertEqual(res_cl_splat.m_declared, res_cl_direct.m_declared)

    # 2. temporal_interval
    res_temp_splat = temporal.temporal_interval(
        eval_data, num_windows=g, metric="mse", **decl.kwargs()
    )
    res_temp_direct = temporal.temporal_interval(
        eval_data, num_windows=g, metric="mse", m_item=decl.m
    )
    self.assertAlmostEqual(res_temp_splat.low, res_temp_direct.low, places=12)
    self.assertAlmostEqual(res_temp_splat.high, res_temp_direct.high, places=12)
    self.assertEqual(res_temp_splat.m_declared, res_temp_direct.m_declared)

    # 3. graph_interval
    res_gr_splat = graph.graph_interval(
        eval_data, cluster_ids, cluster_edges, metric="mse", **decl.kwargs()
    )
    res_gr_direct = graph.graph_interval(
        eval_data,
        cluster_ids,
        cluster_edges,
        metric="mse",
        m_item=decl.m,
    )
    self.assertAlmostEqual(
        res_gr_splat.interval.low, res_gr_direct.interval.low, places=12
    )
    self.assertAlmostEqual(
        res_gr_splat.interval.high, res_gr_direct.interval.high, places=12
    )
    self.assertEqual(
        res_gr_splat.interval.m_declared, res_gr_direct.interval.m_declared
    )

  def test_r2_m_item_equals_m(self):
    g = 1000
    b = 4
    n = g * b
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(n)
    y_pred = y_true + 0.1 * rng.standard_normal(n)
    c_ids = np.repeat(np.arange(g, dtype=np.int64), b)
    c_edges = np.column_stack([np.arange(g - 1), np.arange(1, g)])

    # temporal_interval
    res_t_m = temporal.temporal_interval(
        (y_true, y_pred), num_windows=g, metric="r2", m=3.0
    )
    res_t_item = temporal.temporal_interval(
        (y_true, y_pred), num_windows=g, metric="r2", m_item=3.0
    )
    self.assertAlmostEqual(res_t_m.low, res_t_item.low, places=12)
    self.assertAlmostEqual(res_t_m.high, res_t_item.high, places=12)
    self.assertEqual(res_t_m.m_declared, res_t_item.m_declared)
    self.assertEqual(res_t_m.m_observed, res_t_item.m_observed)
    self.assertEqual(res_t_m.status, res_t_item.status)

    # graph_interval
    res_g_m = graph.graph_interval(
        (y_true, y_pred), c_ids, c_edges, metric="r2", m=3.0
    )
    res_g_item = graph.graph_interval(
        (y_true, y_pred), c_ids, c_edges, metric="r2", m_item=3.0
    )
    self.assertAlmostEqual(
        res_g_m.interval.low, res_g_item.interval.low, places=12
    )
    self.assertAlmostEqual(
        res_g_m.interval.high, res_g_item.interval.high, places=12
    )
    self.assertEqual(
        res_g_m.interval.m_declared, res_g_item.interval.m_declared
    )
    self.assertEqual(
        res_g_m.interval.m_observed, res_g_item.interval.m_observed
    )
    self.assertEqual(
        res_g_m.interval.diagnostic.diagnosis,
        res_g_item.interval.diagnostic.diagnosis,
    )

  def test_kappa_exclusivity_raises_value_error(self):
    data = np.ones(20)
    c_ids = np.repeat(np.arange(4), 5)
    with self.assertRaises(ValueError):
      cluster_bound.cluster_interval(data, c_ids, kappa_cluster=1.0, kappa=1.0)

  def test_graph_r2_diagnosis_all_four_states(self):
    rng = np.random.default_rng(42)

    # 1. Saturated: G >= 4 and kappa >= 0.5 * G
    g = 10
    b = 2
    n = g * b
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n)
    c_ids = np.repeat(np.arange(g, dtype=np.int64), b)
    c_edges = np.column_stack([np.arange(g - 1), np.arange(1, g)])

    res_sat = graph.graph_interval(
        (y_true, y_pred), c_ids, c_edges, metric="r2", kappa=6.0
    )
    self.assertEqual(
        res_sat.interval.status, ind_interval.Status.ASSUMPTION_REQUIRED
    )
    self.assertEqual(
        res_sat.interval.diagnostic.diagnosis,
        cluster_bound.Diagnosis.SATURATED_GRAPH_MIXING,
    )
    self.assertTrue(
        math.isnan(res_sat.interval.diagnostic.min_effective_n_exist)
    )
    self.assertTrue(math.isnan(res_sat.interval.diagnostic.target_effective_n))
    self.assertTrue(math.isnan(res_sat.interval.diagnostic.max_supported_alpha))

    # 2. Resolvable: kappa < 0.5 * G
    res_res = graph.graph_interval(
        (y_true, y_pred), c_ids, c_edges, metric="r2", kappa=1.0
    )
    self.assertEqual(
        res_res.interval.status, ind_interval.Status.ASSUMPTION_REQUIRED
    )
    self.assertEqual(
        res_res.interval.diagnostic.diagnosis,
        cluster_bound.Diagnosis.RESOLVABLE_SAMPLE_SIZE,
    )

    # 3. Certified wide: G=2000, c_up > c_target=0.2
    g_w = 2000
    n_w = g_w * 4
    y_t_w = rng.standard_normal(n_w)
    y_p_w = y_t_w + 0.1 * rng.standard_normal(n_w)
    c_ids_w = np.repeat(np.arange(g_w, dtype=np.int64), 4)
    c_edges_w = np.column_stack([np.arange(g_w - 1), np.arange(1, g_w)])

    res_wide = graph.graph_interval(
        (y_t_w, y_p_w),
        c_ids_w,
        c_edges_w,
        metric="r2",
        m_item=3.0,
        c_target=0.2,
    )
    self.assertEqual(res_wide.interval.status, ind_interval.Status.UNREFUTED)
    self.assertEqual(
        res_wide.interval.diagnostic.diagnosis,
        cluster_bound.Diagnosis.CERTIFIED_WIDE,
    )
    self.assertGreater(res_wide.interval.diagnostic.c_up, 0.2)

    # 4. Certified useful: G=8000, c_up <= c_target=0.3 and U_B < inf
    g_u = 8000
    n_u = g_u * 4
    y_t_u = rng.standard_normal(n_u)
    y_p_u = y_t_u + 0.1 * rng.standard_normal(n_u)
    c_ids_u = np.repeat(np.arange(g_u, dtype=np.int64), 4)
    c_edges_u = np.column_stack([np.arange(g_u - 1), np.arange(1, g_u)])

    res_useful = graph.graph_interval(
        (y_t_u, y_p_u),
        c_ids_u,
        c_edges_u,
        metric="r2",
        m_item=2.0,
        c_target=0.3,
    )
    self.assertEqual(res_useful.interval.status, ind_interval.Status.UNREFUTED)
    self.assertEqual(
        res_useful.interval.diagnostic.diagnosis,
        cluster_bound.Diagnosis.CERTIFIED_USEFUL,
    )
    self.assertLessEqual(res_useful.interval.diagnostic.c_up, 0.3)
    assert res_useful.r2_detail is not None
    self.assertTrue(math.isfinite(res_useful.r2_detail.U_B))

  def test_r2_label_side_tail_refutation_message(self):
    g = 2000
    b = 4
    n = g * b
    rng = np.random.default_rng(123)
    y_true = rng.standard_normal(n)
    y_pred = y_true + 0.05 * rng.standard_normal(n)
    c_ids = np.repeat(np.arange(g, dtype=np.int64), b)
    c_edges = np.column_stack([np.arange(g - 1), np.arange(1, g)])

    res = graph.graph_interval(
        (y_true, y_pred),
        c_ids,
        c_edges,
        metric="r2",
        m_item=16.0,
        m_labels=1.01,
    )
    self.assertEqual(res.interval.status, ind_interval.Status.TAIL_UNRESOLVED)
    self.assertIn("label totals Q_c m_hat=", res.message)
    self.assertNotIn("squared-error totals", res.message)
    self.assertLessEqual(res.interval.m_observed, res.interval.m_declared)
    assert res.r2_detail is not None
    self.assertGreater(res.r2_detail.m_hat_q, res.r2_detail.M_c_q)

  def test_temporal_drift_half_intervals_with_m_item(self):
    rng = np.random.default_rng(42)
    g = 1000
    n = 10000
    data = rng.uniform(0.1, 1.0, size=n)
    m_val = 3.0

    res = temporal.temporal_interval(
        data, num_windows=g, metric="mse", m_item=m_val
    )
    self.assertTrue(math.isfinite(res.drift.first_half[0]))
    self.assertTrue(math.isfinite(res.drift.first_half[1]))
    self.assertTrue(math.isfinite(res.drift.second_half[0]))
    self.assertTrue(math.isfinite(res.drift.second_half[1]))

    g1 = g // 2
    n1 = (n * g1) // g
    data_1 = data[:n1]
    data_2 = data[n1:]
    c_ids_1 = (np.arange(n1) * g1) // n1
    c_ids_2 = (np.arange(n - n1) * (g - g1)) // (n - n1)

    sk_1 = cluster_sketch.ClusterSketch.from_data(data_1, c_ids_1, "mse")
    sk_2 = cluster_sketch.ClusterSketch.from_data(data_2, c_ids_2, "mse")
    half_level = 1.0 - (1.0 - 0.95) / 2.0

    res_1 = cluster_bound.cluster_interval(
        sk_1, metric="mse", level=half_level, m_item=m_val
    )
    res_2 = cluster_bound.cluster_interval(
        sk_2, metric="mse", level=half_level, m_item=m_val
    )

    self.assertAlmostEqual(res.drift.first_half[0], res_1.low, places=12)
    self.assertAlmostEqual(res.drift.first_half[1], res_1.high, places=12)
    self.assertAlmostEqual(res.drift.second_half[0], res_2.low, places=12)
    self.assertAlmostEqual(res.drift.second_half[1], res_2.high, places=12)


if __name__ == "__main__":
  absltest.main()
