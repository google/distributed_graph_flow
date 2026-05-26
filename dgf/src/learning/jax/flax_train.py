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

"""Generic training loop for FLAX based models (focus on GNNs).

This module provides a flexible training loop for JAX/Flax models,
handling common tasks such as:
  - Checkpointing: Saving and restoring model parameters and optimizer state,
    enabling pre-emption tolerance.
  - Logging: Writing scalar metrics to various backends using
  `clu.metric_writers`.
  - Metrics Accumulation: Efficiently accumulating metrics over steps.
  - Validation: Running evaluation on a validation dataset at regular intervals.
  - Progress Bar: Displaying training progress and metrics using `tqdm`.
"""

import dataclasses
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple
from clu import metric_writers
from dgf.src.learning.jax import common
from dgf.src.learning.ten_lines import common as ten_lines_common
from dgf.src.util import log
from dgf.src.util import util
import flax.linen as nn
import jax
import jaxtyping as jt
import optax
import orbax.checkpoint as ocp
import tqdm


# TODO(gbm): Should the input of the train-step be a dataclass instead of
# individual fields e.g. params? This way, it is easier to add new fields
# (e.g., batch norm stats) without breaking existing code.
class TrainStep(Protocol):
  """Function signature for a TrainStep."""

  def __call__(
      self,
      params: optax.Params,
      opt_state: optax.OptState,
      batch: Any,
      rng_key: Any,
  ) -> Tuple[optax.Params, optax.OptState, Dict[str, Any]]:
    """The call function for a generic training step.

    Args:
      params: The model parameters.
      opt_state: The optimizer state.
      batch: The training batch.
      rng_key: The random key for the training step.

    Returns:
      A tuple containing the updated model params, the updated optimizer state
      and a dictionary of keyed (by string) scalar metrics.
    """
    ...


class ValidStep(Protocol):
  """Function signature for a ValidStep."""

  def __call__(
      self,
      params: optax.Params,
      opt_state: optax.OptState,
      batch: Any,
  ) -> Dict[str, Any]:
    """The call function for a generic validation step.

    Args:
      params: The model parameters.
      opt_state: The optimizer state.
      batch: The training batch.

    Returns:
      A dictionary of keyed (by string) scalar metrics.
    """
    ...


class DummyDataFn(Protocol):
  """Function signature for user-defined dummy data preprocessor."""

  def __call__(self, batch: jt.PyTree[jt.ArrayLike]) -> jt.PyTree[jt.ArrayLike]:
    """Call signature for DummyDataFn interface.

    Args:
      batch: A batch of PyTree like data.

    Returns:
      A PyTree structure of jt.Arrays.
    """
    ...


@dataclasses.dataclass
class MetricAccumulator:
  """Basic structure to accumulate jax metric values without host sync."""

  def __init__(self):
    self._reset()

  def _reset(self):
    self.acc = {}
    self.count = 0

  def add(self, values: Dict[str, jax.Array]):
    """Add a dictionary of jax metric values to the accumulator."""
    if self.count == 0:
      self.acc = values
    else:
      for k, v in values.items():
        self.acc[k] = self.acc[k] + v
    self.count += 1

  def get_and_reset(self) -> Dict[str, float]:
    """Returns accumulated metrics as Python floats and resets."""
    if self.count == 0:
      return {}
    ret = {k: v.item() / self.count for k, v in self.acc.items()}
    self._reset()
    return ret


def _append_to_display_dict(
    metrics: Dict[str, float], prefix: str, output: Dict[str, str]
):
  """Formats float metrics and appends them to a string dictionary."""
  for k, v in metrics.items():
    output[f"{prefix}{k}"] = f"{v:.4f}"


LogItem = ten_lines_common.LogItem


@dataclasses.dataclass
class TrainResult:
  """Output of the "train" method.

  Attributes:
    model_params: The trained model parameters.
    opt_state: The final optimizer state.
    train_logs: A list of LogItem objects containing the training metrics.
    valid_logs: A list of LogItem objects containing the validation metrics.
  """

  model_params: optax.Params
  opt_state: optax.OptState
  train_logs: List[LogItem]
  valid_logs: List[LogItem]


# TODO(bmayer): Support user-defined best_step fns?
# TODO(gbm): Support early stopping.
def train(
    model: nn.Module,
    opt: optax.GradientTransformation,
    train_step: TrainStep,
    dataset_iterator: Iterator[Any],
    num_train_steps: int,
    rng_key: Any,
    *,
    model_params: Optional[optax.Params] = None,
    opt_state: Optional[optax.OptState] = None,
    dummy_data: Optional[jt.PyTree[jt.ArrayLike]] = None,
    dummy_data_fn: Optional[DummyDataFn] = None,
    working_path: Optional[str] = None,
    metric_writer: Optional[metric_writers.MetricWriter] = None,
    checkpoint_manager: Optional[ocp.CheckpointManager] = None,
    train_log_every_n_steps: int = 100,
    checkpoint_every_n_steps: Optional[int] = 1000,
    disable_progress_bar: bool = False,
    display_model_structure: bool = False,
    valid_every_n_steps: int = 1000,
    valid_step: Optional[ValidStep] = None,
    valid_dataset_iterator_fn: Optional[Callable[[], Iterator[Any]]] = None,
    print_logs: bool = False,
    print_initial_model_params: bool = False,
    max_training_time_seconds: Optional[int] = None,
    export_metrics_to_xm: bool = False,
) -> TrainResult:
  """Trains a Flax module with a flexible and feature-rich training loop.

  This function orchestrates the training process, including:
    - Initialization of model parameters and optimizer state.
    - Iterating over the training dataset.
    - Executing custom training steps.
    - Periodic validation on a separate dataset.
    - Checkpointing model state for pre-emption tolerance.
    - Logging metrics to various backends and the console.
    - Displaying progress with a tqdm progress bar.

  Args:
    model: A `flax.linen.Module` instance to be trained.
    opt: An `optax.GradientTransformation` optimizer instance.
    train_step: A callable matching the `TrainStep` protocol, responsible for a
      single training step, including forward pass, loss calculation, backward
      pass, and optimizer update.
    dataset_iterator: An iterator yielding batches of training data.
    num_train_steps: The total number of training steps to perform.
    rng_key: A JAX PRNG key for random operations.
    model_params: Optional initial model parameters. If `None`, parameters are
      initialized using `model.init` with `dummy_data` or the first batch from
      `dataset_iterator`. This is typically a PyTree containing tensors.
    opt_state: Optional initial optimizer state. If `None`, it's initialized
      using `opt.init` with the `model_params`.
    dummy_data: Optional sample data matching the model's input structure, used
      for model initialization if `model_params` is not provided. Mutually
      exclusive with `dummy_data_fn`.
    dummy_data_fn: An optional callable that takes a batch from
      `dataset_iterator` and returns the model input. Used for model
      initialization if `model_params` and `dummy_data` are not provided. Useful
      when the iterator yields more than just the model input (e.g., labels).
    working_path: Optional directory path for saving checkpoints and metrics.
      Required if `checkpoint_manager` or `metric_writer` are not provided.
    metric_writer: An optional `clu.metric_writers.MetricWriter` instance for
      logging metrics. If `None` and `working_path` is provided, a default
      writer will be created.
    checkpoint_manager: An optional `orbax.checkpoint.CheckpointManager` for
      handling model checkpointing. If `None` and `working_path` is provided, a
      default manager will be created.
    train_log_every_n_steps: The frequency (in steps) at which training metrics
      are accumulated, logged, and displayed.
    checkpoint_every_n_steps: The frequency (in steps) at which model
      checkpoints are saved. If `None`, checkpointing is disabled.
    disable_progress_bar: If `True`, the tqdm progress bar will be suppressed.
    display_model_structure: If `True`, the model structure will be printed
      using `model.tabulate` before training starts.
    valid_every_n_steps: The frequency (in steps) at which validation is
      performed.
    valid_step: A callable matching the `ValidStep` protocol, responsible for a
      single validation step. Required if `valid_dataset_iterator_fn` is
      provided.
    valid_dataset_iterator_fn: A callable that returns a new iterator over the
      validation dataset each time validation is run. Required if `valid_step`
      is provided.
    print_logs: If `True`, metrics will be printed to the console at each
      logging interval.
    print_initial_model_params: If `True`, the shapes of the initial model
      parameters and optimizer state will be logged.

  Returns:
    A `TrainResult` dataclass containing:
      - `model_params`: The trained model parameters.
      - `opt_state`: The final optimizer state.
      - `train_logs`: A list of `LogItem` for training metrics.
      - `valid_logs`: A list of `LogItem` for validation metrics.
  """

  if dummy_data is not None and dummy_data_fn is not None:
    raise ValueError(
        "Do not simulataneously provide dummy data and a dummy data callable"
        " function."
    )

  if dummy_data_fn is not None and not callable(dummy_data_fn):
    raise ValueError("`dummy_data_fn` must be a callable function.")

  if valid_step is None != valid_dataset_iterator_fn is None:
    raise ValueError(
        "`valid_step` and `valid_dataset_iterator_fn` must either both be"
        " provided or both be None."
    )

  # Check for existing checkpoints (pre-emption tolerance).
  if checkpoint_manager is None:
    if working_path is not None:
      options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
      checkpoint_manager = ocp.CheckpointManager(
          os.path.join(working_path, "checkpoints"),
          ocp.PyTreeCheckpointer(),
          options=options,
      )
    else:
      checkpoint_manager = None

  if checkpoint_manager is not None:
    log.info(
        f"Checking for existing checkpoints {checkpoint_manager.directory}"
    )
    # TODO(bmayer): Figure out if we want to support iterator catchup.
    latest_step: int | None = checkpoint_manager.latest_step()
    if latest_step is not None:
      log.info(
          "Restoring state from step %D found in %s.",
          latest_step,
          checkpoint_manager.directory,
      )
      restored_state = checkpoint_manager.restore(latest_step)
      if "params" not in restored_state:
        raise ValueError("Restored state does not contain 'params'.")
      if "opt_state" not in restored_state:
        raise ValueError("Restored state does not contain 'opt_state'.")
      model_params = restored_state["params"]
      opt_state = restored_state["opt_state"]
      start_step = latest_step + 1
    else:
      log.info(
          "No existing checkpoint found in %s.", checkpoint_manager.directory
      )
      start_step = 0
  else:
    start_step = 0

  # If we didn"t find an existing checkpoint and the user didn"t supply
  # `model_params` or `opt_state`, we instantiate using either supplied
  # `dummy_data` or first datum on iterator.
  if metric_writer is None:
    writers = [metric_writers.LoggingWriter()]
    if working_path is not None:
      writers.append(
          metric_writers.SummaryWriter(os.path.join(working_path, "summary"))
      )
    if export_metrics_to_xm:
      try:
        from clu.metric_writers import XmMeasurementsWriter

        writers.append(XmMeasurementsWriter(asynchronous=True))
      except ImportError:
        pass
    metric_writer = metric_writers.MultiWriter(writers)
  if model_params is None:
    if dummy_data is None:
      log.info("Generate first batch to initialize model")
      if dummy_data_fn is not None:
        dummy_data = dummy_data_fn(next(dataset_iterator))
      else:
        dummy_data = next(dataset_iterator)

    with util.print_timer("Create model variables", True):
      rng_key, model_key = jax.random.split(rng_key, 2)
      model_params = model.init(model_key, dummy_data, training=True)
      # WARNING: "model_params" is a dictionary containing the model params in the
      # "params" key. It is not the model param directly.

    if display_model_structure:
      model_structure = model.tabulate(
          model_key,
          dummy_data,
          training=True,
          console_kwargs={"width": 800, "force_terminal": False},
      )
      log.info(f"Model structure:\n{model_structure}")

    if print_initial_model_params:
      common.log_info_shape("Initial model parameters", model_params)

  if opt_state is None:
    model_params_without_batch_stats = {
        k: v for k, v in model_params.items() if k != "batch_stats"
    }
    opt_state = opt.init(model_params_without_batch_stats)

    if print_initial_model_params:
      common.log_info_shape("Initial opt parameters", opt_state)

  # Accumulate training metrics until the next log.
  train_metric_accumulator = MetricAccumulator()

  # Dictionary of the last computed validation metrics.
  last_valid_metrics = {}

  # Dictionary of the last computed metrics (including train and validation)
  # that are displayed in the progress bar.
  list_display_dict = {}
  train_metrics = {}

  validation_duration_was_printed = False

  train_logs = []
  valid_logs = []

  def run_check_point():
    """Create a new checkpoint."""
    if checkpoint_manager is None:
      return
    checkpoint_manager.save(
        step, {"params": model_params, "opt_state": opt_state}
    )

  def run_valid_logs():
    """Compute and prints the logs on the validation dataset."""
    # TODO(gbm): Reword code so we don't need nonlocal.
    nonlocal last_valid_metrics
    nonlocal validation_duration_was_printed

    if valid_step is None:
      return
    valid_metric_accumulator = MetricAccumulator()
    with jax.profiler.TraceAnnotation("validation"):

      start_time = time.time()
      for batch in valid_dataset_iterator_fn():
        with jax.profiler.TraceAnnotation("valid step"):
          step_valid_metrics = valid_step(
              params=model_params,
              opt_state=opt_state,
              batch=batch,
          )
          valid_metric_accumulator.add(step_valid_metrics)
      end_time = time.time()
      str_duration = util.format_duration(end_time - start_time)
      if not validation_duration_was_printed:
        validation_duration_was_printed = True
        pbar.write(f"Validation loop took {str_duration} (only printed once)")

      if valid_metric_accumulator.count == 0:
        raise ValueError(
            "The validation dataset iterator didn't yield any batches. Make"
            " sure the validation dataset contains at least one batch of data."
        )
      last_valid_metrics = valid_metric_accumulator.get_and_reset()
      valid_logs.append(LogItem(metrics=last_valid_metrics, step=step))

      if metric_writer is not None:
        metric_writer.write_scalars(
            step, {f"valid-{k}": v for k, v in last_valid_metrics.items()}
        )

      if print_logs:
        display_dict = {"step": str(step)}
        _append_to_display_dict(train_metrics, "train-", display_dict)
        _append_to_display_dict(last_valid_metrics, "valid-", display_dict)
        log_str = " ".join([f"{k}:{v}" for k, v in display_dict.items()])
        pbar.write(log_str)

  if valid_every_n_steps is not None:
    log.info("Will validate model every %s step(s)", valid_every_n_steps)
  if checkpoint_every_n_steps is not None:
    log.info("Will checkpoint model every %s step(s)", checkpoint_every_n_steps)

  log.info("Start training. The first two steps are generally slow.")
  start_time = time.time()
  pbar = tqdm.tqdm(
      range(start_step, num_train_steps),
      disable=disable_progress_bar,
      desc="Training",
  )
  for step in pbar:
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      if max_training_time_seconds is not None:
        elapsed_time = time.time() - start_time
        if elapsed_time > max_training_time_seconds:
          log.info(
              "Max training time of %s seconds exceeded. Stopping training.",
              max_training_time_seconds,
          )
          break

      rng_key, step_key = jax.random.split(rng_key, 2)
      with jax.profiler.TraceAnnotation("gen batch"):
        batch = next(dataset_iterator, None)
      if batch is None:
        log.info("Dataset iterator depleted (returned None). Stopping training")
        break

      with jax.profiler.TraceAnnotation("train step"):
        model_params, opt_state, metrics = train_step(
            model_params,
            opt_state,
            batch,
            step_key,
        )
        train_metric_accumulator.add(metrics)

      # Training metrics + logging
      if train_log_every_n_steps > 0 and step % train_log_every_n_steps == 0:
        train_metrics = train_metric_accumulator.get_and_reset()
        train_logs.append(LogItem(metrics=train_metrics, step=step))

        if metric_writer is not None:
          metric_writer.write_scalars(
              step, {f"train-{k}": v for k, v in train_metrics.items()}
          )

        display_dict = {"step": str(step)}
        _append_to_display_dict(train_metrics, "train-", display_dict)
        _append_to_display_dict(last_valid_metrics, "valid-", display_dict)

        pbar.set_postfix(display_dict)
        list_display_dict = display_dict

      # Checkpointing
      if (
          checkpoint_every_n_steps is not None
          and step > 0
          and step % checkpoint_every_n_steps == 0
      ):
        run_check_point()

      # Validation metrics + logging
      # Note: We skip the validation at step 0.
      if step > 0 and step % valid_every_n_steps == 0:
        run_valid_logs()

  # Final logging
  step = num_train_steps
  run_valid_logs()

  # Final checkpoing
  if checkpoint_manager is not None:
    pbar.write("Saving final checkpoint...")
    run_check_point()
    checkpoint_manager.wait_until_finished()
    pbar.write(
        f"Training complete. Final model saved at step {num_train_steps}.",
    )

  if metric_writer is not None:
    metric_writer.flush()

  log.info(f"Final metrics: {list_display_dict}")
  return TrainResult(
      model_params=model_params,
      opt_state=opt_state,
      train_logs=train_logs,
      valid_logs=valid_logs,
  )
