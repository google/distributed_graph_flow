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


class BuildOptimizerTest(absltest.TestCase):

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
