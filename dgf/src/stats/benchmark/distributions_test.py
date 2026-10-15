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

"""Tests that the analytic ground truth in `distributions` is actually correct.

A wrong `mean_abs` would silently bias every MAE coverage number in the whole
study without any test failing, so each analytic constant is cross-checked
against a large Monte Carlo estimate with a tolerance derived from that
estimate's own standard error.
"""

import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dgf.src.stats.benchmark import distributions


# Large enough that the Monte Carlo standard error on E|e| is ~7e-4.
_MC_SIZE = 2_000_000
_MC_SEED = 20260911

# Number of standard errors allowed before a mismatch is called a failure.
_SIGMA_TOLERANCE = 6.0

# The CDF check is a KS test, whose resolution scales as 1/sqrt(n); 200k
# detects a discrepancy of about 0.7% and costs far less than the moment test.
_CDF_SIZE = 200_000
_CDF_SEED = 20260912


def _all_named_distributions() -> list[distributions.ErrorDistribution]:
  return list(distributions.TUNING_SET) + list(distributions.HELD_OUT_SET)


def _named_cases() -> list[tuple[str, distributions.ErrorDistribution]]:
  return [(d.name, d) for d in _all_named_distributions()]


class DistributionMomentTest(parameterized.TestCase):
  """Verifies analytic moments against Monte Carlo."""

  @parameterized.named_parameters(*_named_cases())
  def test_analytic_moments_match_monte_carlo(
      self, dist: distributions.ErrorDistribution
  ):
    rng = np.random.default_rng(_MC_SEED)
    errors = dist.sample(_MC_SIZE, rng)

    self.assertTrue(np.all(np.isfinite(errors)), f"{dist.name} produced non-finite draws")

    abs_errors = np.abs(errors)
    sample_mean_abs = float(np.mean(abs_errors))
    # SE of the mean of |e|; Var(|e|) <= E[e^2] so this is always finite here.
    se_mean_abs = float(np.std(abs_errors, ddof=1)) / math.sqrt(_MC_SIZE)
    self.assertAlmostEqual(
        sample_mean_abs,
        dist.mean_abs,
        delta=_SIGMA_TOLERANCE * se_mean_abs,
        msg=(
            f"{dist.name}: analytic mean_abs={dist.mean_abs:.6f} but Monte"
            f" Carlo gives {sample_mean_abs:.6f} (SE {se_mean_abs:.2e})"
        ),
    )

    # The mean is zero and the variance is one by standardization.
    sample_mean = float(np.mean(errors))
    se_mean = float(np.std(errors, ddof=1)) / math.sqrt(_MC_SIZE)
    self.assertAlmostEqual(
        sample_mean, 0.0, delta=_SIGMA_TOLERANCE * se_mean,
        msg=f"{dist.name} is not centred",
    )

    sample_mean_sq = float(np.mean(errors**2))
    if dist.max_finite_moment > 4.0:
      # Var(e^2) is finite, so the sample second moment has a usable SE.
      se_mean_sq = float(np.std(errors**2, ddof=1)) / math.sqrt(_MC_SIZE)
      tolerance = max(_SIGMA_TOLERANCE * se_mean_sq, 1e-3)
    else:
      # Var(e^2) is infinite; the estimate converges slowly and erratically.
      tolerance = 0.15
    self.assertAlmostEqual(
        sample_mean_sq,
        dist.mean_sq,
        delta=tolerance,
        msg=f"{dist.name} is not standardized to unit variance",
    )

  @parameterized.named_parameters(*_named_cases())
  def test_sampling_is_deterministic_given_seed(
      self, dist: distributions.ErrorDistribution
  ):
    first = dist.sample(1000, np.random.default_rng(7))
    second = dist.sample(1000, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)

  @parameterized.named_parameters(*_named_cases())
  def test_draws_never_exceed_declared_support(
      self, dist: distributions.ErrorDistribution
  ):
    rng = np.random.default_rng(11)
    observed_max = float(np.max(np.abs(dist.sample(1_000_000, rng))))
    # Floating point round-trips through the standardizing division, so allow
    # a relative slack of 1e-9 rather than an absolute tolerance.
    self.assertLessEqual(
        observed_max,
        dist.support_max * (1.0 + 1e-9),
        f"{dist.name} produced a draw beyond its declared support_max",
    )

  @parameterized.named_parameters(*_named_cases())
  def test_cdf_agrees_with_sampler(
      self, dist: distributions.ErrorDistribution
  ):
    # If F is the true CDF of the sampler then F(e) is Uniform(0, 1), so the
    # Kolmogorov-Smirnov distance between the sorted PIT values and the
    # uniform grid must be small. A CDF that disagrees with its sampler would
    # silently distort the error-label coupling, and hence every R^2 number,
    # without any other test noticing.
    rng = np.random.default_rng(_CDF_SEED)
    errors = dist.sample(_CDF_SIZE, rng)
    pit = np.sort(dist.probability_integral_transform(errors))

    self.assertTrue(
        np.all((pit > 0.0) & (pit < 1.0)),
        f"{dist.name} PIT left the open interval (0, 1)",
    )
    if dist.cdf_left is not None:
      # With atoms the PIT is a step function and cannot be uniform, so the
      # KS test below does not apply. The range check above still does.
      return

    grid = (np.arange(_CDF_SIZE) + 0.5) / _CDF_SIZE
    ks_distance = float(np.max(np.abs(pit - grid)))
    # Conservative asymptotic KS critical threshold at 3.0 / sqrt(n) (p ~ 3e-8).
    self.assertLess(
        ks_distance,
        3.0 / math.sqrt(_CDF_SIZE),
        f"{dist.name} CDF is inconsistent with its sampler",
    )


class AdmissibilityTest(absltest.TestCase):
  """The per-metric contract from D1."""

  def test_student_t3_is_admissible_for_mae_but_not_rmse(self):
    t3 = distributions.make_student_t(3.0)
    self.assertTrue(t3.admits("mae"))
    self.assertFalse(t3.admits("rmse"))
    self.assertFalse(t3.admits("mse"))
    self.assertFalse(t3.admits("r2_ind"))

  def test_student_t5_is_admissible_for_everything(self):
    t5 = distributions.make_student_t(5.0)
    self.assertTrue(t5.admits("mae"))
    self.assertTrue(t5.admits("rmse"))
    self.assertTrue(t5.admits("r2_ind"))

  def test_pareto_boundary_is_exactly_alpha(self):
    self.assertFalse(distributions.make_pareto(4.0).admits("rmse"))
    self.assertTrue(distributions.make_pareto(4.01).admits("rmse"))

  def test_light_tails_admit_all_metrics(self):
    for dist in (distributions.GAUSSIAN, distributions.UNIFORM, distributions.LAPLACE):
      for metric in ("mse", "rmse", "mae", "r2_ind", "r2_trans"):
        self.assertTrue(dist.admits(metric), f"{dist.name} / {metric}")

  def test_truncated_pareto_is_in_contract_despite_looking_heavy(self):
    # Bounded support means every moment exists, even though finite samples
    # look power-law. This is the adversarial case for any tail model.
    dist = distributions.make_truncated_pareto(2.0, 1e4)
    self.assertTrue(dist.admits("rmse"))
    self.assertTrue(dist.bounded)

  def test_unknown_metric_raises(self):
    with self.assertRaises(ValueError):
      distributions.GAUSSIAN.admits("f1_score")

  def test_admissible_metrics_filters(self):
    t3 = distributions.make_student_t(3.0)
    self.assertEqual(
        distributions.admissible_metrics(t3, ["rmse", "mae", "mse"]), ["mae"]
    )

  def test_invalid_parameters_raise(self):
    with self.assertRaises(ValueError):
      distributions.make_student_t(2.0)
    with self.assertRaises(ValueError):
      distributions.make_pareto(1.5)
    with self.assertRaises(ValueError):
      distributions.make_lognormal(0.0)
    with self.assertRaises(ValueError):
      distributions.make_contaminated_gaussian(1.5, 10.0)
    with self.assertRaises(ValueError):
      distributions.make_truncated_pareto(2.0, 0.5)


class RandomPoolTest(absltest.TestCase):
  """The randomised generator that guards against fitting a fixed list."""

  def test_pool_moments_are_analytic_and_correct(self):
    pool = distributions.sample_random_distributions(
        25, np.random.default_rng(3)
    )
    self.assertLen(pool, 25)
    rng = np.random.default_rng(5)
    for dist in pool:
      errors = dist.sample(400_000, rng)
      abs_errors = np.abs(errors)
      se_abs = float(np.std(abs_errors, ddof=1)) / math.sqrt(400_000)
      self.assertAlmostEqual(
          float(np.mean(abs_errors)),
          dist.mean_abs,
          delta=max(6.0 * se_abs, 1e-3),
          msg=f"{dist.name} analytic mean_abs is wrong",
      )
      se_sq = float(np.std(errors**2, ddof=1)) / math.sqrt(400_000)
      self.assertAlmostEqual(
          float(np.mean(errors**2)),
          1.0,
          delta=max(6.0 * se_sq, 2e-3),
          msg=f"{dist.name} is not standardized",
      )

  def test_pool_is_reproducible(self):
    first = distributions.sample_random_distributions(5, np.random.default_rng(1))
    second = distributions.sample_random_distributions(5, np.random.default_rng(1))
    for a, b in zip(first, second):
      self.assertEqual(a.name, b.name)
      self.assertAlmostEqual(a.mean_abs, b.mean_abs, places=12)
      np.testing.assert_array_equal(
          a.sample(100, np.random.default_rng(2)),
          b.sample(100, np.random.default_rng(2)),
      )

  def test_pool_spans_a_range_of_shapes(self):
    pool = distributions.sample_random_distributions(
        200, np.random.default_rng(9)
    )
    mean_abs_values = np.array([d.mean_abs for d in pool])
    # A degenerate generator would collapse to a single shape. Gaussian is
    # 0.798 and uniform is 0.866; heavy scale mixtures go well below both.
    self.assertLess(float(np.min(mean_abs_values)), 0.6)
    self.assertGreater(float(np.max(mean_abs_values)), 0.75)

  def test_invalid_count_raises(self):
    with self.assertRaises(ValueError):
      distributions.sample_random_distributions(0, np.random.default_rng(0))


if __name__ == "__main__":
  absltest.main()
