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

"""Training utilities shared by the ten-lines-of-code models.

A model is trained with `flax_train.train` and the utilities of this module as
follows:

  ```python
  def compute_loss(output, batch):
    loss = ...  # Computed from the model output and the batch labels.
    return loss, {"accuracy": ...}

  opt = training.build_optimizer(hparams, num_train_steps)
  flax_train.train(
      model=model,
      opt=opt,
      train_step=training.build_train_step(model, opt, compute_loss),
      valid_step=training.build_valid_step(model, compute_loss),
      ...
  )
  ```
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, TypeVar

from dgf.src.learning.jax import flax_train
from dgf.src.learning.ten_lines import common
from dgf.src.util import log
from dgf.src.util import util
import flax.linen as nn
import jax
import optax
import tqdm

_Batch = TypeVar("_Batch")

# Computes the loss to minimize and a dictionary of scalar metrics, from the
# output of the model on a batch and from the batch itself.
ComputeLossFn = Callable[[Any, Any], tuple[jax.Array, dict[str, Any]]]


def learning_rate_schedule(
    learning_rate: float, num_train_steps: int | None
) -> optax.Schedule:
  """Returns a linear warmup followed by a cosine decay of the learning rate.

  The learning rate increases linearly from 0 to `learning_rate` during the
  first `min(200, 1 + num_train_steps // 5)` steps, and then decays to 0 at step
  `num_train_steps` following a cosine.

  Args:
    learning_rate: The maximum learning rate, reached at the end of the warmup.
    num_train_steps: The total number of training steps.

  Returns:
    The learning rate schedule.
  """
  warmup_steps = min(200, 1 + num_train_steps // 5)  # pyrefly: ignore[unsupported-operation]
  return optax.join_schedules(
      schedules=[
          optax.linear_schedule(
              init_value=0.0,
              end_value=learning_rate,
              transition_steps=warmup_steps,
          ),
          optax.cosine_decay_schedule(
              init_value=learning_rate,
              decay_steps=int((num_train_steps or 100_000) - warmup_steps),
          ),
      ],
      boundaries=[warmup_steps],
  )


def build_optimizer(
    hparams: common.HParam, num_train_steps: int | None
) -> optax.GradientTransformation:
  """Returns the optimizer of the ten-lines models.

  The optimizer is AdamW with the learning rate schedule of
  `learning_rate_schedule`, applied on gradients clipped to a global norm of 1.

  Args:
    hparams: The hyper-parameters of the model. Defines the learning rate and
      the weight decay.
    num_train_steps: The total number of training steps.

  Returns:
    The optimizer.
  """
  return optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(
          learning_rate=learning_rate_schedule(
              hparams.learning_rate, num_train_steps
          ),
          weight_decay=hparams.opt_weight_decay,
      ),
  )


def build_train_step(
    model: nn.Module,
    opt: optax.GradientTransformation,
    compute_loss: ComputeLossFn,
    *,
    rng_streams: Sequence[str] = ("dropout",),
) -> flax_train.TrainStep:
  """Returns a jitted training step for `flax_train.train`.

  The training step runs the model in training mode with the random streams in
  `rng_streams`, and updates the trainable parameters (i.e., the "params"
  variable collection) with `opt` to minimize the loss computed by
  `compute_loss`. The batch statistics (i.e., the "batch_stats" variable
  collection, e.g., the BatchNorm running averages) are not trained by the
  optimizer, but updated by the forward pass.

  Args:
    model: The model to train, called as `model.apply(variables, batch,
      training=True)`.
    opt: The optimizer. Its state only covers the "params" variable collection.
    compute_loss: Computes the loss to minimize and the metrics of a batch, from
      the output of the model on the batch.
    rng_streams: Names of the Flax RNG streams passed to `model.apply`. When a
      single stream is given, `rng_key` is passed directly to that stream; when
      multiple streams are given, `rng_key` is split across them.

  Returns:
    The training step. The returned metrics contain the "loss", and the metrics
    returned by `compute_loss`.
  """
  rng_streams = tuple(rng_streams)

  def loss_fn(core_params, batch_stats, batch, rng_key):
    if not rng_streams:
      rngs = {}
    elif len(rng_streams) == 1:
      rngs = {rng_streams[0]: rng_key}
    else:
      keys = jax.random.split(rng_key, len(rng_streams))
      rngs = dict(zip(rng_streams, keys))
    if batch_stats:
      output, mutated_variables = model.apply(
          {**core_params, "batch_stats": batch_stats},
          batch,
          training=True,
          rngs=rngs,
          mutable=["batch_stats"],
      )
      batch_stats = mutated_variables["batch_stats"]
    else:
      output = model.apply(core_params, batch, training=True, rngs=rngs)
    loss, metrics = compute_loss(output, batch)
    return loss, (metrics, batch_stats)

  def train_step(params, opt_state, batch, rng_key):
    core_params = {"params": params["params"]} if "params" in params else {}
    batch_stats = params.get("batch_stats", {})
    (loss, (metrics, batch_stats)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(core_params, batch_stats, batch, rng_key)
    updates, opt_state = opt.update(grads, opt_state, core_params)
    new_params = optax.apply_updates(core_params, updates)
    if "batch_stats" in params:
      new_params = {**new_params, "batch_stats": batch_stats}  # pyrefly: ignore[invalid-argument]
    return new_params, opt_state, {"loss": loss, **metrics}

  return jax.jit(train_step)


def build_valid_step(
    model: nn.Module,
    compute_loss: ComputeLossFn,
    *,
    rngs: Mapping[str, jax.Array] | None = None,
) -> flax_train.ValidStep:
  """Returns a jitted validation step for `flax_train.train`.

  The validation step runs the model in inference mode, e.g., without dropout
  and with the BatchNorm running averages.

  Args:
    model: The model to evaluate, called as `model.apply(variables, batch,
      training=False)`.
    compute_loss: Computes the loss and the metrics of a batch, from the output
      of the model on the batch.
    rngs: Optional fixed RNG streams passed to `model.apply` during evaluation
      (e.g., `{"masking": jax.random.PRNGKey(0)}` for deterministic evaluation
      masking in GraphMAE).

  Returns:
    The validation step. The returned metrics contain the "loss", and the
    metrics returned by `compute_loss`.
  """

  def valid_step(params, opt_state, batch):
    del opt_state  # Unused.
    if rngs is None:
      output = model.apply(params, batch, training=False)
    else:
      output = model.apply(params, batch, training=False, rngs=dict(rngs))
    loss, metrics = compute_loss(output, batch)
    return {"loss": loss, **metrics}

  return jax.jit(valid_step)


def _num_valid_batches(
    num_examples: int | None,
    batch_size: int,
    drop_remainder: bool,
    num_valid_steps: int | None,
) -> int | None:
  """Returns the expected number of validation batches, or None if unknown."""
  if num_examples is None:
    num_batches = None
  else:
    num_batches = util.num_batches(
        num_examples, batch_size=batch_size, drop_remainder=drop_remainder
    )
  if num_valid_steps is not None:
    if num_batches is None:
      num_batches = num_valid_steps
    else:
      num_batches = min(num_batches, num_valid_steps)
  return num_batches


def cache_valid_dataset(
    valid_dataset_iterator_fn: Callable[[], Iterator[_Batch]],
    *,
    num_examples: int | None,
    batch_size: int,
    drop_remainder: bool,
    num_valid_steps: int | None,
    verbose: int,
) -> Callable[[], Iterator[_Batch]]:
  """Caches the batches of a validation dataset in memory.

  The validation batches are generated and cached the first time the returned
  function is called. The following calls yield the cached batches. If the
  device runs out of memory while caching, the batches are generated on the fly
  for this call and all the following ones instead.

  Args:
    valid_dataset_iterator_fn: Returns a new iterator over the validation
      batches each time it is called.
    num_examples: The number of validation examples, or None if unknown. Only
      used to display the caching progress.
    batch_size: The number of examples in a batch. Only used to display the
      caching progress.
    drop_remainder: Whether the last incomplete batch is dropped. Only used to
      display the caching progress.
    num_valid_steps: The maximum number of validation batches, or None if
      unlimited. Only used to display the caching progress.
    verbose: The verbosity level.

  Returns:
    A function returning a new iterator over the (cached) validation batches
    each time it is called.
  """
  num_batches = _num_valid_batches(
      num_examples, batch_size, drop_remainder, num_valid_steps
  )
  cached_batches: list[_Batch] | None = None
  caching_failed = False

  def cached_valid_dataset_iterator_fn() -> Iterator[_Batch]:
    nonlocal cached_batches, caching_failed
    if caching_failed:
      yield from valid_dataset_iterator_fn()
      return
    if cached_batches is None:
      batches = []
      try:
        with util.print_timer("Caching validation dataset", verbose >= 1):
          batch_iterator = valid_dataset_iterator_fn()
          if verbose >= 2:
            batch_iterator = tqdm.tqdm(
                batch_iterator,
                total=num_batches,
                desc="Caching validation dataset",
            )
          for batch in batch_iterator:
            batches.append(batch)
      except (jax.errors.JaxRuntimeError, RuntimeError, ValueError) as e:
        batches.clear()
        if "RESOURCE_EXHAUSTED" not in str(e):
          raise
        log.warning(
            "Out of device memory while caching validation dataset (%s);"
            " falling back to uncached validation dataset.",
            e,
        )
        caching_failed = True
        yield from valid_dataset_iterator_fn()
        return
      if verbose >= 1:
        log.info("Number of cache validation batches: %d", len(batches))
      cached_batches = batches
    yield from cached_batches

  return cached_valid_dataset_iterator_fn
