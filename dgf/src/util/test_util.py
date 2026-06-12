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

"""Utilities for unit tests."""

import dataclasses
import logging
import math
import os
import pprint
from typing import Any, List, Optional
from absl import flags
from absl.testing import absltest
from dgf.src.data import distributed_graph
from immutabledict import immutabledict
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


def data_root_path() -> str:
  """Root directory of the repo."""
  return ""

def assertProto2Equal(self: absltest.TestCase, a, b):
  """Checks that protos "a" and "b" are equal."""
  self.assertEqual(a, b)

TEST_DATA_PATH_IN_REPO ="test_data"


def dgf_test_data_path() -> str:
  return os.path.join(data_root_path(), TEST_DATA_PATH_IN_REPO)


@dataclasses.dataclass
class TwoDiffObjects:
  """Two different objects."""

  obj1: Any
  obj2: Any


# The last two different objects computed by are_equal.
_last_diff: Optional[TwoDiffObjects] = None


def disable_diff_truncation():
  """Disable diff truncation in unittest.

  This function ensures that the full diff is shown when using self.assertEqual.
  """
  if "unittest.util" in __import__("sys").modules:
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999  # pylint: disable=protected-access


# TODO(gbm): Improve error message.
def assert_are_equal(test, obj1: Any, obj2: Any, abs_tol: float | None = None):
  test.assertTrue(
      are_equal(obj1, obj2, abs_tol=abs_tol),
      "Objects are not"
      f" equal:\nobj1={pprint.pformat(obj1)}\nobj2={pprint.pformat(obj2)}\n\nDiff"
      f" part:\n{pprint.pformat(_last_diff)}",
  )


def _ragged_arrays_equal(
    a: np.ndarray, b: np.ndarray, abs_tol: float | None = None
) -> bool:
  """Tests if two ragged arrays are equal."""
  if a.shape != b.shape:
    return False
  if a.dtype != object:
    return False
  if b.dtype != object:
    return False
  return all(are_equal(a, b, abs_tol=abs_tol) for a, b in zip(a.flat, b.flat))


def are_equal(obj1: Any, obj2: Any, abs_tol: float | None = None) -> bool:
  """Tests if two objects are equal.

  Set _last_diff with the first two different objects.
  """

  try:

    def ret(equal_result: bool) -> bool:
      global _last_diff
      if not equal_result and _last_diff is None:
        _last_diff = TwoDiffObjects(obj1, obj2)
      return equal_result

    # NumPy arrays
    if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
      if obj1.dtype == object and obj2.dtype == object:
        return ret(_ragged_arrays_equal(obj1, obj2, abs_tol=abs_tol))
      if (
          abs_tol is not None
          and obj1.dtype != np.bytes_
          and obj2.dtype != np.bytes_
      ):
        return ret(np.allclose(obj1, obj2, atol=abs_tol))
      else:
        return ret(np.array_equal(obj1, obj2))

    # JAX arrays
    if isinstance(obj1, jnp.ndarray) and isinstance(obj2, jnp.ndarray):
      if abs_tol is not None:
        return ret(jnp.allclose(obj1, obj2, atol=abs_tol))
      else:
        return ret(jnp.array_equal(obj1, obj2))

    # TensorFlow arrays
    if isinstance(obj1, tf.Tensor) and isinstance(obj2, tf.Tensor):
      if abs_tol is not None:
        return ret(np.allclose(obj1.numpy(), obj2.numpy(), atol=abs_tol))
      else:
        return ret(np.array_equal(obj1.numpy(), obj2.numpy()))

    # TensorFlow vs non-Tensor
    if isinstance(obj1, tf.Tensor) and not isinstance(obj2, tf.Tensor):
      return ret(are_equal(obj1.numpy(), obj2, abs_tol=abs_tol))
    if not isinstance(obj1, tf.Tensor) and isinstance(obj2, tf.Tensor):
      return ret(are_equal(obj1, obj2.numpy(), abs_tol=abs_tol))

    # TensorFlow Ragged vs Numpy/List
    if isinstance(obj1, tf.RaggedTensor) and isinstance(obj2, np.ndarray):
      list1 = obj1.to_list()
      list2 = obj2.tolist()
      return ret(are_equal(list1, list2, abs_tol=abs_tol))
    if isinstance(obj1, np.ndarray) and isinstance(obj2, tf.RaggedTensor):
      list1 = obj1.tolist()
      list2 = obj2.to_list()
      return ret(are_equal(list1, list2, abs_tol=abs_tol))
    if isinstance(obj1, tf.RaggedTensor) and isinstance(obj2, tf.RaggedTensor):
      list1 = obj1.to_list()
      list2 = obj2.to_list()
      return ret(are_equal(list1, list2, abs_tol=abs_tol))

    # Dictionaries
    if isinstance(obj1, (dict, immutabledict)) and isinstance(
        obj2, (dict, immutabledict)
    ):
      if obj1.keys() != obj2.keys():
        return ret(False)
      return ret(
          all(are_equal(obj1[k], obj2[k], abs_tol=abs_tol) for k in obj1)
      )

    # Sets (unordered and unique)
    if isinstance(obj1, set) and isinstance(obj2, set):
      return ret(
          len(obj1) == len(obj2)
          and all(
              any(are_equal(x, y, abs_tol=abs_tol) for y in obj2) for x in obj1
          )
      )

    # Lists or Tuples (order-sensitive)
    if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
      return ret(
          len(obj1) == len(obj2)
          and all(are_equal(x, y, abs_tol=abs_tol) for x, y in zip(obj1, obj2))
      )

    # Floats with tolerance
    if isinstance(obj1, float) and isinstance(obj2, float):
      if math.isnan(obj1) and math.isnan(obj2):
        return ret(True)
      if math.isinf(obj1) and math.isinf(obj2):
        return ret(True)
      if abs_tol is not None:
        return ret(abs(obj1 - obj2) <= abs_tol)

    # Dataclasses
    if dataclasses.is_dataclass(obj1) and dataclasses.is_dataclass(obj2):
      if dataclasses.fields(obj1) != dataclasses.fields(obj2):
        return ret(False)
      return ret(
          all(
              are_equal(
                  getattr(obj1, field.name),
                  getattr(obj2, field.name),
                  abs_tol=abs_tol,
              )
              for field in dataclasses.fields(obj1)
          )
      )

    # TensorFlow ExtensionType
    if isinstance(obj1, tf.experimental.ExtensionType):
      if not isinstance(obj2, tf.experimental.ExtensionType):
        return ret(False)

      # Compare fields using __dict__ attributes.
      a_attrs = sorted(obj1.__dict__.keys())
      b_attrs = sorted(obj2.__dict__.keys())

      if a_attrs != b_attrs:
        return ret(False)

      for attr in a_attrs:
        if not are_equal(
            getattr(obj1, attr), getattr(obj2, attr), abs_tol=abs_tol
        ):

          return ret(False)
      return ret(True)

    # Needed some help un-nesting the features dictionary, the default __eq__
    # on the dataclass wasn't working.
    if isinstance(obj1, distributed_graph.Node) and isinstance(
        obj2, distributed_graph.Node
    ):
      return ret(obj1.id == obj2.id and are_equal(obj1.features, obj2.features))

    # Fallback: direct comparison
    return ret(obj1 == obj2)
  except Exception as e:
    raise RuntimeError(f"Error comparing:\nobj1={obj1!r}\nobj2={obj2!r}") from e


def unique_subset_of_length(values: List, allowed: List, length: int) -> bool:
  """Tests if "values" are unique and only contain values in "allowed".

  Args:
    values: The list of values to check.
    allowed: The list of allowed values.
    length: The expected length of `values`.

  Returns:
    True if `values` contains unique elements, all of which are in `allowed`,
    and has the specified `length`. Otherwise, False.
  """
  if len(values) != length:
    return False
  if len(values) != len(set(values)):
    # Not all values are unique.
    return False
  allowed_set = set(allowed)
  for value in values:
    if value not in allowed_set:
      # Contains a value not in allowed.
      return False
  return True


def assert_unique_subset_of_length(
    test, values: List, allowed: List, length: int
):
  """Test asserts that "unique_subset_of_length" is true."""
  test.assertTrue(
      unique_subset_of_length(values, allowed, length),
      f"{values!r} is not a subset of {allowed!r} of len {length}",
  )


def assert_golden_string(
    test, value: str, golden_path: str, postfix: str = "", strip: bool = False
):
  """Ensures that "value" is equal to the content of the file "golden_path".

  Args:
    test: A test.
    value: Value to test.
    golden_path: Path to golden file expressed from the root of the repo.
    postfix: Optional postfix to the path of the file containing the actual
      value.
    strip: Whether to strip whitespace from both value and golden data before
      comparison.
  """
  # Add the test data location.
  golden_path = os.path.join(TEST_DATA_PATH_IN_REPO, golden_path)
  full_golden_path = os.path.join(data_root_path(), golden_path)

  with open(full_golden_path) as f:
    golden_data = f.read()

  compare_value = value.strip() if strip else value
  compare_golden = golden_data.strip() if strip else golden_data

  if compare_value != compare_golden:
    value_path = os.path.join(
        "/tmp/gf_unit_test", os.path.basename(golden_path) + postfix
    )
    os.makedirs(os.path.dirname(value_path), exist_ok=True)
    logging.info(
        "Golden test failed for %s. Save the effective value to %s",
        golden_path,
        value_path,
    )
    with open(value_path, "w") as f:
      f.write(value)
    copy_command = f"cp {value_path} {golden_path}"
    diff_command = f"diff {full_golden_path} {value_path}"
    error_msg = (
        f"Value does not match golden data in: {golden_path}\n"
        f"The actual value has been saved to: {value_path}\n"
        f"To see the diff, run:\n{diff_command}\n"
        f"To update the golden file, run:\n{copy_command}"
    )
  else:
    error_msg = None  # No custom message needed if values are equal

  test.assertEqual(
      compare_value,
      compare_golden,
      msg=error_msg,
  )
