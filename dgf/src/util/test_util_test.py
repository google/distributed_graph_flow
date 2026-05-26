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

import dataclasses
from absl.testing import absltest
from dgf.src.util import test_util
import numpy as np


class TestUtilTest(absltest.TestCase):

  def test_are_equal_with_numbers(self):
    self.assertTrue(test_util.are_equal(1, 1))
    self.assertFalse(test_util.are_equal(1, 2))

  def test_are_equal_with_numpy_arrays(self):
    self.assertTrue(test_util.are_equal(np.array([1, 2]), np.array([1, 2])))
    self.assertFalse(test_util.are_equal(np.array([1, 2]), np.array([1, 3])))

  def test_are_equal_with_numpy_arrays_with_ragged_arrays(self):
    v1 = np.array([np.array([11, 12]), np.array([12, 13, 14])], dtype=object)
    v2 = v1.copy()
    v3 = np.array(
        [np.array([11, 12]), np.array([12, 13, 14, 15])], dtype=object
    )
    self.assertTrue(test_util.are_equal(v1, v2))
    self.assertFalse(test_util.are_equal(v1, v3))

  def test_are_equal_with_dictionaries(self):
    self.assertTrue(test_util.are_equal({"a": 1, "b": 2}, {"a": 1, "b": 2}))
    self.assertFalse(test_util.are_equal({"a": 1, "b": 2}, {"a": 1, "c": 2}))
    self.assertFalse(test_util.are_equal({"a": 1, "b": 2}, {"a": 1, "b": 3}))

  def test_are_equal_with_sets(self):
    self.assertTrue(test_util.are_equal({1, 2}, {1, 2}))
    self.assertFalse(test_util.are_equal({1, 2}, {1, 3}))

  def test_are_equal_with_lists(self):
    self.assertTrue(test_util.are_equal([1, 2], [1, 2]))
    self.assertFalse(test_util.are_equal([1, 2], [1, 3]))
    self.assertFalse(test_util.are_equal([1, 2], [1, 2, 3]))

  def test_are_equal_with_tuples(self):
    self.assertTrue(test_util.are_equal((1, 2), (1, 2)))
    self.assertFalse(test_util.are_equal((1, 2), (1, 3)))

  def test_are_equal_with_list_and_tuple(self):
    self.assertTrue(test_util.are_equal([1, 2], (1, 2)))

  def test_are_equal_with_nested_structures_same(self):
    obj1 = {"a": [1, 2], "b": {3, 4}}
    obj2 = {"a": [1, 2], "b": {4, 3}}
    self.assertTrue(test_util.are_equal(obj1, obj2))

  def test_are_equal_with_nested_structures_different(self):
    obj1 = {"a": [1, 2], "b": {3, 4}}
    obj2 = {"a": [1, 3], "b": {4, 3}}
    self.assertFalse(test_util.are_equal(obj1, obj2))

  def test_are_equal_with_bytes(self):
    self.assertTrue(test_util.are_equal(b"abc", b"abc"))
    self.assertFalse(test_util.are_equal(b"abc", b"abd"))

  def test_are_equal_with_dataclasses(self):
    @dataclasses.dataclass
    class Point:
      x: int
      y: int

    self.assertTrue(test_util.are_equal(Point(1, 2), Point(1, 2)))
    self.assertFalse(test_util.are_equal(Point(1, 2), Point(1, 3)))

    @dataclasses.dataclass
    class Point3D:
      x: int
      y: int
      z: int

    self.assertFalse(test_util.are_equal(Point(1, 2), Point3D(1, 2, 3)))

  def test_unique_subset_of_length(self):
    allowed = [1, 2, 3, 4, 5]

    # Valid cases
    self.assertTrue(test_util.unique_subset_of_length([1, 2], allowed, 2))
    self.assertTrue(test_util.unique_subset_of_length([5, 1], allowed, 2))
    self.assertTrue(test_util.unique_subset_of_length([], allowed, 0))
    self.assertTrue(
        test_util.unique_subset_of_length([1, 2, 3, 4, 5], allowed, 5)
    )

    # Incorrect length
    self.assertFalse(test_util.unique_subset_of_length([1, 2], allowed, 3))
    self.assertFalse(test_util.unique_subset_of_length([1, 2, 3], allowed, 2))
    self.assertFalse(test_util.unique_subset_of_length([], allowed, 1))

    # Contains duplicates
    self.assertFalse(test_util.unique_subset_of_length([1, 1], allowed, 2))
    self.assertFalse(test_util.unique_subset_of_length([1, 2, 1], allowed, 3))

    # Contains values not in allowed
    self.assertFalse(test_util.unique_subset_of_length([1, 6], allowed, 2))
    self.assertFalse(test_util.unique_subset_of_length([0, 1], allowed, 2))
    self.assertFalse(test_util.unique_subset_of_length([1, 2, 6], allowed, 3))

    # Mixed failures
    self.assertFalse(test_util.unique_subset_of_length([1, 6, 1], allowed, 3))

  def test_assert_golden_string_success(self):
    expected_content = (
        "This is a dummy golden file for testing assert_golden_string.\n"
    )
    test_util.assert_golden_string(self, expected_content, "dummy_golden.txt")


if __name__ == "__main__":
  absltest.main()
