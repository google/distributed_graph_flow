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

"""Utility to handle early stopping of model training."""

from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class EarlyStoppingMonitorConfig:
  """Configuration for EarlyStoppingMonitor.

  Attributes:
    patience: Number of checks with no improvement after which training stops.
    min_improvement: Minimum change in loss to qualify as an improvement.
  """

  patience: int = 5
  min_improvement: float = 1e-6


class EarlyStoppingMonitor:
  """Tracks training loss and decides when to stop early."""

  def __init__(self, config: EarlyStoppingMonitorConfig) -> None:
    self._config = config
    self._patience_counter = 0
    self._should_stop = False
    self.best_loss = None
    self.best_params = None
    self.best_step = None

  def add_loss(self, step: int, value: float, params: Any = None) -> None:
    """Registers a loss value and optionally the associated parameters and step."""

    if self._should_stop:
      # We should already stop.
      return

    if (
        self.best_loss is None
        or self.best_loss - value > self._config.min_improvement
    ):
      # A better model
      self.best_loss = value
      self._patience_counter = 0
      self.best_params = params
      self.best_step = step
    else:
      # Not a better model
      self._patience_counter += 1

    if self._patience_counter >= self._config.patience:
      self._should_stop = True

  def should_stop(self) -> bool:
    """Returns True if training should stop."""
    return self._should_stop

  def get_state(self) -> dict[str, Any]:
    """Returns the serializable state of the monitor."""
    return {
        # We don't save the config, as it is provided at initialization.
        "patience_counter": self._patience_counter,
        "should_stop": self._should_stop,
        "best_loss": self.best_loss,
        "best_params": self.best_params,
        "best_step": self.best_step,
    }

  def get_template_state(self, model_params: Any = None) -> dict[str, Any]:
    """Returns a template state for checkpoint restoration."""
    return {
        "patience_counter": 0,
        "should_stop": False,
        "best_loss": 0.0,
        "best_params": model_params,
        "best_step": 0,
    }

  def set_state(self, state: dict[str, Any]) -> None:
    """Restores the monitor state from a serialized dictionary."""
    # TODO(dtl): Possibly emit a warning if keys are not found?
    if (patience_counter := state.get("patience_counter", None)) is not None:
      self._patience_counter = int(patience_counter)
    else:
      self._patience_counter = 0

    if (should_stop := state.get("should_stop", None)) is not None:
      self._should_stop = bool(should_stop)
    else:
      self._should_stop = False

    self.best_loss = state.get("best_loss", None)
    self.best_params = state.get("best_params", None)
    self.best_step = state.get("best_step", None)


def normalize_early_stopping_config(
    value: Union[bool, int],
) -> Optional[EarlyStoppingMonitorConfig]:
  if isinstance(value, bool):
    if value:
      return EarlyStoppingMonitorConfig()
    else:
      return None
  else:
    return EarlyStoppingMonitorConfig(patience=value)
