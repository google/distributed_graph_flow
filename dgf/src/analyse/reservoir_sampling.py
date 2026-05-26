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

"""Reservoir sampling."""

import dataclasses
from typing import List, Tuple
from apache_beam import coders
import numpy as np


@dataclasses.dataclass
class BatchReservoirSampling:
  """Computes the quantiles of a stream of batches.

  Based on <yggdrasil_decision_forests>/port/python/ydf/dataset/dataset.py
  """

  cache_size: int = 1_000_000

  _samples: np.ndarray = dataclasses.field(init=False)
  _num_in_cache: int = dataclasses.field(init=False, default=0)
  _num_seen: int = dataclasses.field(init=False, default=0)

  def __post_init__(self):
    # TODO(gbm): Avoid allocating the entire `cache_size` upfront. Implement a
    # dynamic resizing strategy for the reservoir.
    self._samples = np.empty(self.cache_size, dtype=np.float32)

  def add(self, data: np.ndarray) -> None:
    """Adds a batch of observations."""
    if data.ndim > 1:
      data = data.flatten()
    elif data.ndim == 0:
      data = np.expand_dims(data, 0)

    batch_size = data.shape[0]
    new_num_in_cache = self._num_in_cache + batch_size

    if new_num_in_cache <= self.cache_size:
      # Add the entire batch to the cache
      self._samples[self._num_in_cache : new_num_in_cache] = data
      self._num_in_cache = new_num_in_cache
      self._num_seen += batch_size
      return

    if self._num_in_cache + 1 <= self.cache_size:
      # Add the first elements to the cache until it's full
      num_consumed = self.cache_size - self._num_in_cache
      self._samples[self._num_in_cache : self.cache_size] = data[:num_consumed]
      data = data[num_consumed:]
      batch_size -= num_consumed
      self._num_in_cache += num_consumed
      self._num_seen += num_consumed
      assert self._num_in_cache == self.cache_size

    # Reservoir sampling
    rnd01 = np.random.uniform(0.0, 1.0, batch_size)
    sample_idx = np.arange(self._num_seen + 1, self._num_seen + batch_size + 1)
    indexes = (rnd01 * sample_idx).astype(np.int32)
    is_selected = (indexes < self.cache_size).nonzero()
    self._samples[indexes[is_selected]] = data[is_selected]
    self._num_seen += batch_size

  def add_reservoir(self, other: "BatchReservoirSampling") -> None:
    """Adds another reservoir to the current reservoir."""
    self.add(other._samples[: other._num_in_cache])

  def get_quantiles(self, num_quantiles: int) -> Tuple[
      List[float],
      List[float],
  ]:
    """Computes the quantiles and corresponding quantile arguments."""
    # TODO(b/381397901): This is not optimal when a few values have a lot of
    # weights: The functions might return less than "num_quantiles" quantiles
    # when returning "num_quantiles" quantiles is possible.
    thresholds = np.linspace(0, 1, num_quantiles)
    quantiles = np.nanquantile(self._samples[: self._num_in_cache], thresholds)
    return quantiles.tolist(), thresholds.tolist()


coders.registry.register_coder(BatchReservoirSampling, coders.RowCoder)
