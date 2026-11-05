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

"""Heuristic detection of text features."""

from collections.abc import Sequence
from typing import Any
import numpy as np

_MIN_NUM_VALUES = 10
_MAX_COLLISION_RATIO = 0.1
_MIN_MULTI_WORD_RATIO = 0.5
_MAX_SAMPLED_VALUES = 10_000


def looks_like_text(values: np.ndarray | Sequence[Any]) -> bool:
  """Tries to determine if a set of values is a set of TEXT features.

  Current logic: Free-form text values are rarely identical and contain spaces
  (unlike unique IDs). Detected as True if there are at least 10 non-empty
  values, less than 10% of them are collisions (i.e., equal to another value),
  and more than 50% of them contain multiple whitespace-separated words.

  Usage example:

  ```python
  text_detection.looks_like_text([
      "The cat sleeps.", "A dog barks.", "Birds sing loudly.", "It rains.",
      "I like tea.", "She reads books.", "We play chess.", "They run fast.",
      "He cooks pasta.", "You write code.",
  ])
  # True
  text_detection.looks_like_text([
      "user_1", "user_2", "user_3", "user_4", "user_5",
      "user_6", "user_7", "user_8", "user_9", "user_10",
  ])
  # False
  ```

  Args:
    values: 1D array (or 2D array with a single column) of bytes or str values.
      Empty values are ignored.

  Returns:
    True if the values look like free-form text.
  """
  arr = np.asarray(values)
  if arr.ndim == 2 and arr.shape[1] == 1:
    arr = arr[:, 0]
  elif arr.ndim != 1:
    return False

  if arr.dtype.kind not in ("S", "U", "O"):
    return False

  non_empty = [
      value
      for value in arr[:_MAX_SAMPLED_VALUES]
      if isinstance(value, (bytes, str)) and value.strip()
  ]
  if len(non_empty) < _MIN_NUM_VALUES:
    return False
  num_collisions = len(non_empty) - len(set(non_empty))
  num_multi_words = sum(len(value.split()) > 1 for value in non_empty)
  return (
      num_collisions < _MAX_COLLISION_RATIO * len(non_empty)
      and num_multi_words > _MIN_MULTI_WORD_RATIO * len(non_empty)
  )
