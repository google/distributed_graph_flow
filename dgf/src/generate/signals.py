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

import abc
import dataclasses
from typing import List, Literal
import numpy as np


@dataclasses.dataclass
class SignalConfig(abc.ABC):
  """Abstract base class for all time series configurations."""

  @abc.abstractmethod
  def evaluate(self, t: np.ndarray) -> np.ndarray:
    """Evaluates the signal for the given time steps t."""
    pass


@dataclasses.dataclass
class CompositeSignal(SignalConfig):
  """Weighted linear combination of signals."""

  components: List[SignalConfig]
  weights: List[float]

  def __post_init__(self):
    assert len(self.components) == len(
        self.weights
    ), 'Components and weights must have the same length.'

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    # Sum the result of all sub-components
    total_signal = np.zeros_like(t)
    for i, component in enumerate(self.components):
      total_signal += self.weights[i] * component.evaluate(t)
    return total_signal


@dataclasses.dataclass
class Sinusoidal(SignalConfig):
  """A sinusoidal signal."""

  amplitude: float = 1.0
  frequency: float = 1.0  # Hz
  phase: float = 0.0  # Radians
  offset: float = 0.0

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    return self.offset + self.amplitude * np.sin(
        2 * np.pi * self.frequency * t + self.phase
    )


@dataclasses.dataclass
class Linear(SignalConfig):
  """A linear signal."""

  slope: float = 1.0
  intercept: float = 0.0

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    return self.slope * t + self.intercept


@dataclasses.dataclass
class LinearCentered(SignalConfig):
  """A linear signal centered around a target value."""

  target_at_center: float
  slope: float = 1.0

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    assert t.ndim == 1
    center = (t[-1] - t[0]) / 2
    intercept = self.target_at_center - (self.slope * center)
    return self.slope * t + intercept


@dataclasses.dataclass
class Rectangle(SignalConfig):
  """A rectangle pulse signal."""

  start_time: float
  end_time: float
  low_value: float = 0.0
  high_value: float = 1.0

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    # Create a mask where time is within the rectangle window
    mask = (t >= self.start_time) & (t < self.end_time)
    result = np.full_like(t, self.low_value)
    result[mask] = self.high_value
    return result


@dataclasses.dataclass
class PoissonSeries(SignalConfig):
  """Sample an indepenent poission RV at each timestep.

  E.g., a decent model for packet drops.
  """

  lamb: float = 2

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    return np.random.poisson(self.lamb, size=len(t))


@dataclasses.dataclass
class Impulse(SignalConfig):
  """A single impulse signal."""

  location: float
  height: float = 1.0

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    y = np.zeros_like(t)
    # Find the index closest to the specified location
    idx = (np.abs(t - self.location)).argmin()
    y[idx] = self.height
    return y


@dataclasses.dataclass
class WhiteNoise(SignalConfig):
  """Sample a series of independent normal RVs."""

  mean: float = 0.0
  std_dev: float = 1.0

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    return np.random.normal(self.mean, self.std_dev, size=t.shape)


@dataclasses.dataclass
class BetaSeries(SignalConfig):
  """Sample a process as independent beta RVs."""

  alpha: float = 2
  beta: float = 10
  # Series will be on [0,1], can use scale to map to [0, scale].
  # Useful for modeling percentages.
  scale: float = 1.0

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    return np.random.beta(a=self.alpha, b=self.beta, size=len(t)) * self.scale


@dataclasses.dataclass
class BernoulliSeries(SignalConfig):
  """Sample independent bernoulli process with prob. p."""

  p: float
  scale: float = 1.0

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    mask = np.random.random(size=len(t)) < self.p
    return mask.astype(float) * self.scale


@dataclasses.dataclass
class TemperatureBetaSeries(SignalConfig):
  """A temperature series modeled using a Beta distribution."""

  min_val: float
  max_val: float

  variance_type: Literal['LOW', 'HIGH'] = 'LOW'

  # Just for tracking intent, not used to generate the series.
  units: Literal['F', 'C'] = 'F'

  def __post_init__(self):
    if self.variance_type not in ['LOW', 'HIGH']:
      raise ValueError(f'Invalid variance type: {self.variance_type}')

    if self.units not in ['F', 'C']:
      raise ValueError(f'Invalid units: {self.units}')

    if self.min_val >= self.max_val:
      raise ValueError(
          f'Min ({self.min_val}) must be less than max ({self.max_val})'
      )

  def evaluate(self, t: np.ndarray) -> np.ndarray:
    alpha = beta = 20  # LOW
    if self.variance_type == 'HIGH':
      alpha = beta = 2
    s = self.max_val - self.min_val
    y = np.random.beta(a=alpha, b=beta, size=len(t))

    return self.min_val + (y * s)


class SignalRegistry:
  """Registry for convenient signal configurations.

  Allows registering factory functions that create specific signal
  configurations (either single or composite) and creating them by name.

  Example:
    # Register a custom signal
    @SignalRegistry.register('MY_SIGNAL')
    def create_my_signal(freq=2.0):
      return Sinusoidal(frequency=freq)

    # Use the registry to create and evaluate signals
    config = SignalRegistry.create('MY_SIGNAL', freq=5.0)
    t = np.linspace(0, 10, 1000)
    signal = config.evaluate(t)
  """

  _registry = {}

  @classmethod
  def register(cls, name: str):
    """Decorator to register a factory function."""

    def decorator(func):
      cls._registry[name] = func
      return func

    return decorator

  @classmethod
  def create(cls, name: str, **kwargs) -> SignalConfig:
    """Create a signal configuration from the registry."""
    if name not in cls._registry:
      raise ValueError(f"Unknown signal configuration: '{name}'")
    return cls._registry[name](**kwargs)


# Register convenient single/composite functions
@SignalRegistry.register('CPU_IDLE')
def create_cpu_idle() -> SignalConfig:
  """Returns a configuration for idle CPU usage (low average)."""
  return BetaSeries(alpha=2, beta=10, scale=100)


@SignalRegistry.register('CPU_HEAVY')
def create_cpu_heavy() -> SignalConfig:
  """Returns a configuration for heavy CPU usage (high average)."""
  return BetaSeries(alpha=10, beta=2, scale=100)


@SignalRegistry.register('CPU_NORMAL')
def create_cpu_normal() -> SignalConfig:
  """Returns a configuration for normal CPU usage (centered average)."""
  return BetaSeries(alpha=5, beta=5, scale=100)


@SignalRegistry.register('NOISY_SINE')
def create_noisy_sine(
    amplitude: float = 1.0, frequency: float = 1.0, noise_std: float = 0.1
) -> SignalConfig:
  """Creates a sine wave with added white noise."""
  return CompositeSignal(
      components=[
          Sinusoidal(amplitude=amplitude, frequency=frequency),
          WhiteNoise(std_dev=noise_std),
      ],
      weights=[1.0, 1.0],
  )
