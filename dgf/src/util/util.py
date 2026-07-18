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

"""Utility functions for the DGF project."""

import contextlib
import math
import time
from typing import Iterator, Optional, Union
from dgf.src.util import log
import numpy as np


def format_duration(seconds: float) -> str:
  """Formats a duration in seconds into a human-readable string.

  For example:
    1h 30m 0s
    34m 5s
    56.32s (print the second decimal if less than 1 minute)

  Args:
    seconds: The duration in seconds.

  Returns:
    A human-readable string representing the duration.
  """
  hours = int(seconds // 3600)
  seconds %= 3600
  minutes = int(seconds // 60)
  seconds %= 60

  if hours > 0:
    return f"{hours}h {minutes}m {int(seconds)}s"
  elif minutes > 0:
    return f"{minutes}m {int(seconds)}s"
  else:
    return f"{seconds:.2f}s"


@contextlib.contextmanager
def print_timer(message: str, enabled: bool):
  """A context manager to print a message and the execution time."""
  if not enabled:
    yield
    return

  log.info(message)
  start_time = time.time()
  try:
    yield
  finally:
    elapsed_time = time.time() - start_time
    log.info(f"{message} finished in {elapsed_time:.2f} seconds")


def num_batches(
    items: Union[int, np.ndarray], *, batch_size: int, drop_remainder: bool
):
  """Returns the number of batches returned by "batch_indices_generator"."""
  if isinstance(items, int):
    num_items = items
  elif isinstance(items, np.ndarray):
    num_items = len(items)
  else:
    raise TypeError(
        f"Unsupported type for 'items': {type(items)}. Should be int or a numpy"
        " array."
    )

  if batch_size <= 0:
    raise ValueError("batch_size must be positive")

  if drop_remainder:
    return num_items // batch_size
  else:
    return math.ceil(num_items / batch_size)


def batch_indices_generator(
    items: Union[int, np.ndarray],
    *,
    batch_size: int,
    drop_remainder: bool,
    shuffle: bool,
) -> Iterator[np.ndarray]:
  """Generates batches of indices.

  If `items` is an integer, generates batches of indices from `0` to `num_items
  - 1`. If `items` is a numpy array, generates batches from `items`.

  These batches are typically used to feed node indices to a graph sampler.

  Usage example:

  ```python
  # Typical batch generation for training
  for batch in batch_indices_generator(
        num_nodes=10,
        batch_size=3,
        drop_remainder=True,
        shuffle=True):
    print(batch)
  # Example output (order varies due to shuffle):
  # [2 8 0]
  # [9 1 5]
  # [4 6 7]

  # Typical batch generation for evaluation
  for batch in batch_indices_generator(
        num_nodes=10,
        batch_size=3,
        drop_remainder=False,
        shuffle=False):
    print(batch)
  # [0 1 2]
  # [3 4 5]
  # [6 7 8]
  # [9]
  ```

  Args:
    items: If an integer, the number of items to sample batches from. The
      returned indices will be in the range `[0, items)`. If a numpy array, the
      array to sample batches from.
    batch_size: The desired size of each batch. If `drop_remainder` is `False`,
      the last batch may contain fewer than `batch_size` elements.
    drop_remainder: If `True`, the last batch will be dropped if it contains
      fewer than `batch_size` elements. If `False`, all indices are returned,
      potentially with a smaller last batch.
    shuffle: If `True`, the indices are shuffled before batching. If `False`,
      indices are returned in ascending order (e.g., `[0, ..., j]`, `[j+1,
      ...]`).

  Yields:
    A numpy array containing a batch of indices.
  """

  if isinstance(items, int):
    indices = np.arange(items)
  elif isinstance(items, np.ndarray):
    indices = items
  else:
    raise TypeError(
        f"Unsupported type for 'items': {type(items)}. Should be int or a numpy"
        " array."
    )

  if shuffle:
    np.random.shuffle(indices)

  num_items = len(indices)
  for i in range(0, num_items, batch_size):
    if drop_remainder and i + batch_size > num_items:
      break
    yield indices[i : i + batch_size]


class RichDisplay:
  """An object printed either as text (terminal) or html (notebook)."""

  def __init__(
      self,
      html: str,
      text: str = "<This object does not have a text representation. Use a Colab or Jupyter notebook to see the rich HTML output.>",
  ):
    self._text = text
    self._html = html

  def _repr_html_(self) -> str:
    return self._html

  def _repr_(self) -> str:
    return self._text


def split_train_valid(
    num_values: int,
    validation_ratio: float,
    random_seed: int,
    batch_size: int = 1,
    max_num_valid_examples: Optional[int] = None,
):
  """Splits indices into training and validation sets.

  Args:
    num_values: The total number of values to split.
    validation_ratio: The ratio of values to use for validation.
    random_seed: The random seed for reproducibility.
    batch_size: Batch size.
    max_num_valid_examples: Optional maximum number of valid examples.

  Returns:
    A tuple of (train_idxs, valid_idxs) containing the indices for each set.

  Raises:
    ValueError: If validation_ratio is between 0 and 1 and num_values < 2.
  """

  if num_values < batch_size:
    raise ValueError(
        f"`num_values` ({num_values}) must be at least `batch_size`"
        f" ({batch_size})."
    )
  if validation_ratio > 0 and num_values < 2 * batch_size:
    raise ValueError(
        f"Cannot split {num_values} values with validation ratio"
        f" {validation_ratio} in two batches of size {batch_size}. At least"
        f" {2*batch_size} values are required."
    )

  rng = np.random.default_rng(random_seed)
  all_idxs = np.arange(num_values)
  rng.shuffle(all_idxs)

  num_valid = int(num_values * validation_ratio)
  if validation_ratio > 0 and num_values >= 2:
    num_valid = max(batch_size, num_valid)

  if max_num_valid_examples is not None:
    num_valid = min(num_valid, max_num_valid_examples)

  valid_idxs = all_idxs[:num_valid]
  train_idxs = all_idxs[num_valid:]
  return train_idxs, valid_idxs


def indent_string(s: str, num_spaces: int = 8) -> str:
  """Indents a multi-line string by num_spaces."""
  indent = " " * num_spaces
  return s.strip().replace("\n", f"\n{indent}")
