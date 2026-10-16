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

"""Tests for the training utilities shared by the ten-lines models."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import training as training_lib
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

_Batch = tuple[jax.Array, jax.Array]


class _ToyModel(nn.Module):
  """Linear regression, optionally followed by BatchNorm and dropout."""

  batch_norm: bool = False
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, batch: _Batch, training: bool) -> jax.Array:
    features, _ = batch
    predictions = nn.Dense(1)(features)
    if self.batch_norm:
      predictions = nn.BatchNorm(use_running_average=not training)(predictions)
    if self.dropout_rate > 0.0:
      predictions = nn.Dropout(self.dropout_rate, deterministic=not training)(
          predictions
      )
    return predictions[:, 0]


class _ParameterFreeModel(nn.Module):
  """A model without any variable."""

  @nn.compact
  def __call__(self, batch: _Batch, training: bool) -> jax.Array:
    del training  # Unused.
    features, _ = batch
    return jnp.sum(features, axis=-1)


def _compute_loss(
    predictions: jax.Array, batch: _Batch
) -> tuple[jax.Array, dict[str, jax.Array]]:
  _, labels = batch
  loss = jnp.mean(jnp.square(predictions - labels))
  return loss, {"mean_prediction": jnp.mean(predictions)}


def _make_batch() -> _Batch:
  rng = np.random.RandomState(0)
  features = rng.randn(16, 3).astype(np.float32)
  labels = 2.0 * features[:, 0] - features[:, 1] + 3.0
  return jnp.asarray(features), jnp.asarray(labels)


def _init_variables(model: nn.Module, batch: _Batch) -> Mapping[str, Any]:
  return model.init(
      {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
      batch,
      training=True,
  )


def _init_opt_state(
    opt: optax.GradientTransformation, variables: Mapping[str, Any]
) -> optax.OptState:
  # Like `flax_train.train`, the optimizer only covers the trainable variables.
  return opt.init({k: v for k, v in variables.items() if k != "batch_stats"})


class LearningRateScheduleTest(absltest.TestCase):

  def test_warmup_followed_by_cosine_decay(self):
    schedule = training_lib.learning_rate_schedule(0.1, num_train_steps=1000)
    # 200 warmup steps, followed by 800 decay steps.
    np.testing.assert_allclose(schedule(0), 0.0)
    np.testing.assert_allclose(schedule(100), 0.05, rtol=1e-5)
    np.testing.assert_allclose(schedule(200), 0.1, rtol=1e-5)
    np.testing.assert_allclose(schedule(600), 0.05, rtol=1e-5)
    np.testing.assert_allclose(schedule(1000), 0.0, atol=1e-7)

  def test_short_training_has_short_warmup(self):
    schedule = training_lib.learning_rate_schedule(0.1, num_train_steps=10)
    # 1 + 10 // 5 = 3 warmup steps, followed by 7 decay steps.
    np.testing.assert_allclose(schedule(3), 0.1, rtol=1e-5)
    np.testing.assert_allclose(schedule(10), 0.0, atol=1e-7)

  def test_custom_warmup(self):
    schedule = training_lib.learning_rate_schedule(
        0.1, num_train_steps=1000, warmup_steps=10
    )
    np.testing.assert_allclose(schedule(5), 0.05, rtol=1e-5)
    np.testing.assert_allclose(schedule(10), 0.1, rtol=1e-5)
    np.testing.assert_allclose(schedule(505), 0.05, rtol=1e-5)

  def test_no_warmup(self):
    schedule = training_lib.learning_rate_schedule(
        0.1, num_train_steps=1000, warmup_steps=0
    )
    np.testing.assert_allclose(schedule(0), 0.1, rtol=1e-5)
    np.testing.assert_allclose(schedule(500), 0.05, rtol=1e-5)
    np.testing.assert_allclose(schedule(1000), 0.0, atol=1e-7)


class BuildOptimizerTest(parameterized.TestCase):

  def test_follows_learning_rate_warmup(self):
    hparams = common.HParam(learning_rate=0.1, opt_weight_decay=0.0)
    opt = training_lib.build_optimizer(hparams, num_train_steps=1000)
    params = jnp.ones((3,))
    grads = jnp.ones((3,))
    opt_state = opt.init(params)

    # The learning rate is zero at the first step of the warmup.
    updates, opt_state = opt.update(grads, opt_state, params)
    assert isinstance(updates, jax.Array)
    np.testing.assert_allclose(updates, 0.0)

    # It then increases, and the update follows the opposite of the gradient.
    updates, _ = opt.update(grads, opt_state, params)
    assert isinstance(updates, jax.Array)
    self.assertTrue(np.all(updates < 0.0))

  @parameterized.parameters((None, 0.1), (1e-12, 0.0))
  def test_gradient_clipping(
      self, gradient_clip_norm: float | None, expected_update_magnitude: float
  ):
    hparams = common.HParam(
        learning_rate=0.1,
        opt_weight_decay=0.0,
        warmup_steps=0,
        gradient_clip_norm=gradient_clip_norm,
    )
    opt = training_lib.build_optimizer(hparams, num_train_steps=1000)
    params = jnp.ones((3,))
    # Adam normalizes the gradients, except when they are smaller than its
    # epsilon (1e-8), which is the case after clipping to a 1e-12 norm.
    updates, _ = opt.update(jnp.ones((3,)), opt.init(params), params)
    assert isinstance(updates, jax.Array)
    np.testing.assert_allclose(
        -updates, expected_update_magnitude, atol=1e-4, rtol=1e-4
    )

  def test_weight_decay(self):
    hparams = common.HParam(
        learning_rate=0.1,
        opt_weight_decay=0.5,
        warmup_steps=0,
        gradient_clip_norm=None,
    )
    opt = training_lib.build_optimizer(hparams, num_train_steps=1000)
    params = jnp.full((3,), 2.0)
    updates, _ = opt.update(jnp.zeros((3,)), opt.init(params), params)
    assert isinstance(updates, jax.Array)
    # update = -lr * weight_decay * params.
    np.testing.assert_allclose(updates, -0.1, rtol=1e-5)


class BuildTrainStepTest(parameterized.TestCase):

  def _train(
      self, model: nn.Module, num_steps: int
  ) -> tuple[Any, list[dict[str, Any]]]:
    """Trains `model` and returns the final variables and the step metrics."""
    batch = _make_batch()
    variables = _init_variables(model, batch)
    opt = optax.sgd(0.05)
    opt_state = _init_opt_state(opt, variables)
    train_step = training_lib.build_train_step(model, opt, _compute_loss)
    step_metrics = []
    for step in range(num_steps):
      variables, opt_state, metrics = train_step(
          variables, opt_state, batch, jax.random.PRNGKey(step)
      )
      step_metrics.append(metrics)
    return variables, step_metrics

  @parameterized.parameters(False, True)
  def test_minimizes_loss(self, batch_norm: bool):
    _, step_metrics = self._train(_ToyModel(batch_norm=batch_norm), 20)
    self.assertSameElements(step_metrics[0], ["loss", "mean_prediction"])
    self.assertLess(step_metrics[-1]["loss"], step_metrics[0]["loss"])

  def test_updates_batch_statistics(self):
    model = _ToyModel(batch_norm=True)
    initial_variables = _init_variables(model, _make_batch())
    variables, _ = self._train(model, num_steps=1)

    self.assertSameElements(variables, ["params", "batch_stats"])
    initial_stats = initial_variables["batch_stats"]["BatchNorm_0"]
    stats = variables["batch_stats"]["BatchNorm_0"]
    self.assertFalse(np.allclose(stats["mean"], initial_stats["mean"]))
    self.assertFalse(np.allclose(stats["var"], initial_stats["var"]))

  def test_model_without_batch_statistics(self):
    variables, _ = self._train(_ToyModel(), num_steps=1)
    self.assertSameElements(variables, ["params"])

  def test_model_without_variables(self):
    variables, step_metrics = self._train(_ParameterFreeModel(), num_steps=2)
    self.assertEmpty(variables)
    np.testing.assert_allclose(step_metrics[0]["loss"], step_metrics[1]["loss"])

  def test_uses_dropout_random_stream(self):
    model = _ToyModel(dropout_rate=0.5)
    batch = _make_batch()
    variables = _init_variables(model, batch)
    opt = optax.sgd(0.05)
    opt_state = _init_opt_state(opt, variables)
    train_step = training_lib.build_train_step(model, opt, _compute_loss)

    losses = [
        train_step(variables, opt_state, batch, jax.random.PRNGKey(seed))[2][
            "loss"
        ]
        for seed in [0, 0, 1]
    ]
    # The dropout masks only depend on the random key of the step.
    np.testing.assert_allclose(losses[0], losses[1])
    self.assertNotAlmostEqual(float(losses[0]), float(losses[2]))

  def test_splits_key_across_multiple_rng_streams(self):
    class _MultiRngModel(nn.Module):

      @nn.compact
      def __call__(self, batch: _Batch, training: bool) -> jax.Array:
        del training
        features, _ = batch
        preds = nn.Dense(1)(features)[:, 0]
        dropout_noise = jax.random.normal(
            self.make_rng("dropout"), preds.shape
        )
        masking_noise = jax.random.normal(
            self.make_rng("masking"), preds.shape
        )
        return preds + 0.1 * dropout_noise + 0.1 * masking_noise

    model = _MultiRngModel()
    batch = _make_batch()
    variables: Mapping[str, Any] = model.init(
        {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
            "masking": jax.random.PRNGKey(2),
        },
        batch,
        training=True,
    )
    opt = optax.sgd(0.05)
    opt_state = _init_opt_state(opt, variables)
    train_step = training_lib.build_train_step(
        model, opt, _compute_loss, rng_streams=("dropout", "masking")
    )

    new_vars_0, _, metrics_0 = train_step(
        variables, opt_state, batch, jax.random.PRNGKey(7)
    )
    new_vars_0_dict: Any = new_vars_0
    _, _, metrics_0_repeat = train_step(
        variables, opt_state, batch, jax.random.PRNGKey(7)
    )
    _, _, metrics_1 = train_step(
        variables, opt_state, batch, jax.random.PRNGKey(8)
    )
    np.testing.assert_allclose(metrics_0["loss"], metrics_0_repeat["loss"])
    self.assertNotAlmostEqual(
        float(metrics_0["loss"]), float(metrics_1["loss"])
    )
    self.assertFalse(
        np.allclose(
            new_vars_0_dict["params"]["Dense_0"]["kernel"],
            variables["params"]["Dense_0"]["kernel"],
        )
    )


class BuildValidStepTest(absltest.TestCase):

  def test_runs_model_in_inference_mode(self):
    model = _ToyModel(batch_norm=True, dropout_rate=0.5)
    batch = _make_batch()
    variables = {
        **_init_variables(model, batch),
        # Non-trivial BatchNorm statistics.
        "batch_stats": {
            "BatchNorm_0": {"mean": jnp.array([1.0]), "var": jnp.array([4.0])}
        },
    }

    valid_step = training_lib.build_valid_step(model, _compute_loss)
    metrics = valid_step(params=variables, opt_state={}, batch=batch)

    predictions = model.apply(variables, batch, training=False)
    assert isinstance(predictions, jax.Array)
    expected_loss, expected_metrics = _compute_loss(predictions, batch)
    self.assertSameElements(metrics, ["loss", "mean_prediction"])
    np.testing.assert_allclose(metrics["loss"], expected_loss, rtol=1e-6)
    np.testing.assert_allclose(
        metrics["mean_prediction"],
        expected_metrics["mean_prediction"],
        rtol=1e-6,
    )

  def test_passes_fixed_rngs_in_valid_step(self):
    class _MaskingEvalModel(nn.Module):

      @nn.compact
      def __call__(self, batch: _Batch, training: bool) -> jax.Array:
        del training
        features, _ = batch
        preds = nn.Dense(1)(features)[:, 0]
        mask_noise = jax.random.uniform(self.make_rng("masking"), preds.shape)
        return preds + mask_noise

    model = _MaskingEvalModel()
    batch = _make_batch()
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "masking": jax.random.PRNGKey(1)},
        batch,
        training=False,
    )
    fixed_rngs = {"masking": jax.random.PRNGKey(42)}
    valid_step = training_lib.build_valid_step(
        model, _compute_loss, rngs=fixed_rngs
    )
    metrics_a = valid_step(params=variables, opt_state={}, batch=batch)
    metrics_b = valid_step(params=variables, opt_state={}, batch=batch)
    expected_preds = model.apply(
        variables, batch, training=False, rngs=fixed_rngs
    )
    assert isinstance(expected_preds, jax.Array)
    expected_loss, _ = _compute_loss(expected_preds, batch)
    np.testing.assert_allclose(metrics_a["loss"], expected_loss, rtol=1e-6)
    np.testing.assert_allclose(metrics_a["loss"], metrics_b["loss"], rtol=1e-6)


class _CountingIteratorFn:
  """Returns an iterator over `batches`, and counts the calls.

  Attributes:
    batches: The batches to yield.
    error: If set, the first iteration raises this error after the first batch.
    num_calls: The number of iterations started.
  """

  def __init__(
      self, batches: Sequence[int], error: Exception | None = None
  ) -> None:
    self.batches = batches
    self.error = error
    self.num_calls = 0

  def __call__(self) -> Iterator[int]:
    self.num_calls += 1
    for batch_idx, batch in enumerate(self.batches):
      if self.error is not None and self.num_calls == 1 and batch_idx == 1:
        raise self.error
      yield batch


class CacheValidDatasetTest(parameterized.TestCase):

  def _cache(
      self, iterator_fn: _CountingIteratorFn, verbose: int = 0
  ) -> Callable[[], Iterator[int]]:
    return training_lib.cache_valid_dataset(
        iterator_fn,
        num_examples=len(iterator_fn.batches),
        batch_size=1,
        drop_remainder=False,
        num_valid_steps=None,
        verbose=verbose,
    )

  @parameterized.parameters(0, 2)
  def test_generates_batches_once(self, verbose: int):
    iterator_fn = _CountingIteratorFn([1, 2, 3])
    cached_iterator_fn = self._cache(iterator_fn, verbose=verbose)
    self.assertEqual(list(cached_iterator_fn()), [1, 2, 3])
    self.assertEqual(list(cached_iterator_fn()), [1, 2, 3])
    self.assertEqual(iterator_fn.num_calls, 1)

  def test_out_of_memory_falls_back_to_uncached_batches(self):
    iterator_fn = _CountingIteratorFn(
        [1, 2, 3], error=RuntimeError("RESOURCE_EXHAUSTED: Out of memory")
    )
    cached_iterator_fn = self._cache(iterator_fn)
    self.assertEqual(list(cached_iterator_fn()), [1, 2, 3])
    self.assertEqual(list(cached_iterator_fn()), [1, 2, 3])
    # The failed caching attempt, and then one iteration per call.
    self.assertEqual(iterator_fn.num_calls, 3)

  def test_other_errors_are_raised(self):
    iterator_fn = _CountingIteratorFn([1, 2, 3], error=ValueError("Unexpected"))
    cached_iterator_fn = self._cache(iterator_fn)
    with self.assertRaisesRegex(ValueError, "Unexpected"):
      list(cached_iterator_fn())


if __name__ == "__main__":
  absltest.main()
