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

"""Unit tests for range_coverage benchmark harness."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dgf.src.stats.benchmark import range_coverage
from dgf.src.stats.benchmark import range_generators

_cluster_bound = range_coverage._cluster_bound
_cluster_sketch = range_coverage._cluster_sketch
_re = range_coverage._re
spacetime = range_coverage.spacetime


class RangeCoverageTest(parameterized.TestCase):

  def test_clopper_pearson_bounds(self):
    # 950 successes out of 1000 trials
    low, high = range_coverage.clopper_pearson(950, 1000, confidence=0.95)
    self.assertAlmostEqual(low, 0.9345, places=3)
    self.assertAlmostEqual(high, 0.9628, places=3)

    # Edge cases
    low_0, high_0 = range_coverage.clopper_pearson(0, 100)
    self.assertEqual(low_0, 0.0)
    self.assertLess(high_0, 0.05)

    low_n, high_n = range_coverage.clopper_pearson(100, 100)
    self.assertGreater(low_n, 0.95)
    self.assertEqual(high_n, 1.0)

  def test_exact_theta_against_monte_carlo(self):
    graph = range_generators.make_ring_graph(20)
    # K=1, L=2, H=1, r_s=1, r_t=1
    theta_analytical = range_coverage.compute_exact_theta(
        graph, k_hops=1, l_lookback=2, horizon_h=1, r_s=1, r_t=1, variant=1
    )
    self.assertGreater(theta_analytical, 0.0)

    # Verify against 10,000 MC samples
    mean_mc, _, se_mc = range_coverage.verify_theta_with_mc(
        graph, k_hops=1, l_lookback=2, horizon_h=1, r_s=1, r_t=1, variant=1,
        num_reps=10_000, seed=42,
    )
    # Should agree within 3.5 standard errors
    self.assertAlmostEqual(theta_analytical, mean_mc, delta=3.5 * se_mc)

  def test_run_single_config_smoke(self):
    cfg = range_coverage.BenchmarkConfig(
        config_id=1,
        name="Smoke Test",
        graph_kind="ring",
        num_nodes=20,
        num_times=50,
        k_hops=1,
        l_lookback=2,
        horizon_h=1,
        r_s_noise=1,
        r_t_noise=1,
        variant=1,
        rho_0=0.0,
        c_param=0.0,
        decl_r_s=2,
        decl_r_t=2,
        decl_gamma=0.0,
        description="Smoke test config",
    )
    res = range_coverage.run_single_config_benchmark(cfg, num_reps=5, method="kronecker")
    self.assertEqual(res.num_reps, 5)
    self.assertGreater(res.theta_true, 0.0)
    self.assertTrue(0.0 <= res.miss_rate <= 1.0)
    self.assertLessEqual(
        res.num_missed_low + res.num_missed_high + res.num_assumption_required,
        res.num_reps,
    )
    self.assertLessEqual(
        res.num_unrefuted_miss, res.num_missed_low + res.num_missed_high
    )


  def test_range_coverage_public_interval_equivalence(self):
    """Verifies that the harness cluster_interval and public spacetime_interval agree to 1e-12."""
    # Smallest ring configuration with K=1, L=3, H=1, R_s=2, R_t=2, gamma=0, Variant 1
    # that clears the Cantelli issuance threshold nu = G / kappa > n_A = 585 at level=0.95, m_item=16.0.
    # On a ring with w_base = L + H + R_t + 1 = 7, kappa ~= 110.74, so G >= 64784 is required.
    # num_nodes=200, num_times=2300 gives G = 200 * (2297 // 7) = 65600 clusters, nu = 592.37 > 585.
    num_nodes = 200
    num_times = 2300
    k_hops = 1
    l_lookback = 3
    horizon_h = 1
    r_s_noise = 1
    r_t_noise = 2
    decl_r_s = 2
    decl_r_t = 2
    decl_gamma = 0.0

    graph = range_generators.make_ring_graph(num_nodes)
    declaration = _re.RangeDeclaration(
        k_hops=k_hops,
        lookback=l_lookback,
        horizon=horizon_h,
        data_range_s=decl_r_s,
        data_range_t=decl_r_t,
        gamma=decl_gamma,
    )

    t_start = l_lookback - 1
    t_end = num_times - horizon_h
    eval_times = np.arange(t_start, t_end, dtype=np.int64)
    num_eval = len(eval_times)

    obs_u = np.repeat(np.arange(num_nodes, dtype=np.int64), num_eval)
    obs_t = np.tile(eval_times, num_nodes)
    index = spacetime.spacetime_items(obs_u, obs_t)

    cands = _re.range_candidates(
        index,
        graph.edges,
        declaration,
        target_sizes=[1],
        window_multipliers=[1],
    )
    cands = [c for c in cands if not c.name.endswith("_wl1")]
    kappa_fn = _re.range_kappa_fn(
        index, graph.edges, declaration, method="kronecker"
    )
    choice = spacetime.choose_partition(
        cands,
        kappa_fn=kappa_fn,
        m_item=16.0,
        level=0.95,
        is_complete_grid=True,
    )
    self.assertGreater(choice.winner.G / choice.winner_kappa, 585.0)

    winner = choice.winner
    resolved_kappa = choice.winner_kappa

    k_mask = (graph.dist_matrix <= k_hops).astype(np.float64)
    deg_k = np.sum(k_mask, axis=1, keepdims=True)

    rng = np.random.default_rng(0x890001)
    num_reps = 20

    for rep in range(num_reps):
      rep_seed = int(rng.integers(0, 2**31 - 1))
      field = range_generators.generate_space_time_field(
          graph,
          num_times=num_times,
          r_s=r_s_noise,
          r_t=r_t_noise,
          variant=1,
          rho_0=0.0,
          c=0.0,
          seed=rep_seed,
      )

      rolling_temp = np.zeros((num_nodes, num_eval), dtype=np.float64)
      for l in range(l_lookback):
        rolling_temp += field[:, eval_times - l]
      pred = (k_mask @ rolling_temp) / (deg_k * float(l_lookback))
      targets = field[:, eval_times + horizon_h]
      residuals = targets - pred
      r_flat = residuals.reshape(-1)

      # 1. Harness interval (exactly as run_single_config_benchmark does)
      shard = _cluster_sketch.PartialClusterShard.from_summands(
          r_flat**2, winner.cluster_ids
      )
      sk = shard.to_cluster_sketch(kappa_cluster=resolved_kappa)
      res_harness = _cluster_bound.cluster_interval(
          sk,
          metric="mse",
          level=0.95,
          m_item=16.0,
          kappa_cluster=resolved_kappa,
      )

      # 2. Public spacetime.spacetime_interval call (uses choice.m_item)
      res_public = spacetime.spacetime_interval(
          r_flat,
          index,
          choice,
          metric="mse",
          level=0.95,
      )

      # 3. Assertions
      self.assertEqual(res_harness.status, res_public.status)
      self.assertEqual(res_public.status, spacetime._interval.Status.UNREFUTED)
      np.testing.assert_allclose(
          res_harness.low, res_public.low, rtol=1e-12, atol=1e-12
      )
      np.testing.assert_allclose(
          res_harness.high, res_public.high, rtol=1e-12, atol=1e-12
      )


if __name__ == "__main__":
  absltest.main()

