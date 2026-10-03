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

"""Unit tests for synthetic space-time range generators."""

import math
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dgf.src.stats.benchmark import range_generators


class RangeGeneratorsTest(parameterized.TestCase):

  def test_ring_graph_shapes_and_distances(self):
    graph = range_generators.make_ring_graph(10)
    self.assertEqual(graph.num_nodes, 10)
    self.assertEqual(graph.edges.shape, (10, 2))
    self.assertEqual(graph.dist_matrix.shape, (10, 10))
    # Distance between 0 and 1 is 1; distance between 0 and 9 is 1; distance between 0 and 5 is 5
    self.assertEqual(graph.dist_matrix[0, 1], 1)
    self.assertEqual(graph.dist_matrix[0, 9], 1)
    self.assertEqual(graph.dist_matrix[0, 5], 5)

  def test_grid_2d_graph_shapes_and_distances(self):
    graph = range_generators.make_grid_2d_graph(4, 5)
    self.assertEqual(graph.num_nodes, 20)
    self.assertEqual(graph.dist_matrix.shape, (20, 20))
    # Distance between (0, 0) and (3, 4) is 3 + 4 = 7
    u = 0
    v = 3 * 5 + 4
    self.assertEqual(graph.dist_matrix[u, v], 7)

  def test_determinism_under_fixed_seed(self):
    graph = range_generators.make_ring_graph(50)
    field_a = range_generators.generate_space_time_field(
        graph, num_times=100, r_s=1, r_t=2, variant=1, seed=42
    )
    field_b = range_generators.generate_space_time_field(
        graph, num_times=100, r_s=1, r_t=2, variant=1, seed=42
    )
    np.testing.assert_array_equal(field_a, field_b)

  def test_losses_shape_and_range(self):
    graph = range_generators.make_ring_graph(30)
    field = range_generators.generate_space_time_field(
        graph, num_times=100, r_s=1, r_t=2, variant=1, seed=0
    )
    losses, eval_times = range_generators.compute_predictor_and_losses(
        field, graph, k_hops=1, l_lookback=3, horizon_h=1
    )
    expected_times = 100 - (3 - 1) - 1  # 100 - 2 - 1 = 97
    self.assertEqual(losses.shape, (30, expected_times))
    self.assertEqual(len(eval_times), expected_times)
    self.assertTrue(np.all(losses >= 0.0))

  def test_independence_beyond_range_variant1(self):
    """Verifies that for Variant 1, losses outside the theoretical range have zero correlation within Monte Carlo error."""
    # Ring with 100 nodes, 1000 time steps
    # K=1, L=2, H=1, r_s=1, r_t=1
    # Theoretical spatial range: 2*K + 2*r_s = 2(1) + 2(1) = 4 hops
    # Theoretical temporal range: L + H + r_t = 2 + 1 + 1 = 4 steps
    graph = range_generators.make_ring_graph(100)
    num_times = 1000
    field = range_generators.generate_space_time_field(
        graph, num_times=num_times, r_s=1, r_t=1, variant=1, seed=123
    )
    losses, _ = range_generators.compute_predictor_and_losses(
        field, graph, k_hops=1, l_lookback=2, horizon_h=1
    )

    rho = range_generators.compute_empirical_loss_correlations(
        losses, graph, max_hop=15, max_lag=15
    )

    # 1. Inside range: strong positive correlation
    self.assertGreater(rho[0, 0], 0.99)  # self-correlation
    self.assertGreater(rho[1, 0], 0.05)  # 1 hop, same time
    self.assertGreater(rho[0, 1], 0.05)  # same node, 1 lag

    # 2. Outside range: for d > 4 and lag >= 4, correlation should be within Monte Carlo error
    # With T ~ 1000, 4 / sqrt(T) ~ 0.126
    t_eval = losses.shape[1]
    mc_threshold = 4.0 / math.sqrt(t_eval)

    # Far spatial pairs (d = 6..15) at same time
    for d in range(6, 16):
      self.assertLess(
          abs(rho[d, 0]),
          mc_threshold,
          f"Loss correlation at spatial distance {d} exceeds MC threshold: {rho[d, 0]}",
      )

    # Far temporal pairs (lag = 6..15) at same node
    for lag in range(6, 16):
      self.assertLess(
          abs(rho[0, lag]),
          mc_threshold,
          f"Loss correlation at lag {lag} exceeds MC threshold: {rho[0, lag]}",
      )

  def test_common_shock_variant2_has_positive_long_range_correlation(self):
    """Verifies that Variant 2 retains positive correlation beyond the spatial range."""
    graph = range_generators.make_ring_graph(100)
    num_times = 1000
    rho_0 = 0.15
    field = range_generators.generate_space_time_field(
        graph, num_times=num_times, r_s=1, r_t=1, variant=2, rho_0=rho_0, seed=77
    )
    losses, _ = range_generators.compute_predictor_and_losses(
        field, graph, k_hops=1, l_lookback=2, horizon_h=1
    )
    rho = range_generators.compute_empirical_loss_correlations(
        losses, graph, max_hop=20, max_lag=5
    )

    # Even at distance d = 10..20 (far outside spatial range 4), same-time loss correlation is strictly positive
    far_corrs = [rho[d, 0] for d in range(10, 21)]
    self.assertGreater(
        float(np.mean(far_corrs)),
        0.01,
        f"Variant 2 failed to produce persistent positive long-range correlation: {far_corrs}",
    )

  def test_diffuse_global_variant3_decays_with_n(self):
    """Verifies that Variant 3 per-pair correlation decays as O(1/n)."""
    t_steps = 1000
    c_param = 2.0

    # Small ring n = 40
    graph_small = range_generators.make_ring_graph(40)
    field_small = range_generators.generate_space_time_field(
        graph_small, num_times=t_steps, r_s=1, r_t=1, variant=3, c=c_param, seed=10
    )
    losses_small, _ = range_generators.compute_predictor_and_losses(
        field_small, graph_small, k_hops=1, l_lookback=2, horizon_h=1
    )
    rho_small = range_generators.compute_empirical_loss_correlations(
        losses_small, graph_small, max_hop=15, max_lag=0
    )

    # Large ring n = 200
    graph_large = range_generators.make_ring_graph(200)
    field_large = range_generators.generate_space_time_field(
        graph_large, num_times=t_steps, r_s=1, r_t=1, variant=3, c=c_param, seed=10
    )
    losses_large, _ = range_generators.compute_predictor_and_losses(
        field_large, graph_large, k_hops=1, l_lookback=2, horizon_h=1
    )
    rho_large = range_generators.compute_empirical_loss_correlations(
        losses_large, graph_large, max_hop=30, max_lag=0
    )

    far_small = float(np.mean([rho_small[d, 0] for d in range(8, 15)]))
    far_large = float(np.mean([rho_large[d, 0] for d in range(15, 30)]))

    # Correlation in n=200 should be substantially smaller than in n=40
    self.assertGreater(far_small, far_large)


if __name__ == "__main__":
  absltest.main()
