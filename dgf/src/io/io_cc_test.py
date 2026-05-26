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

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.io import io_ext as lib
from dgf.src.util import test_util
import numpy as np


class IOCCTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "with_all_matches",
          np.array([b"X", b"Y"]),
          np.array([b"Y", b"X", b"Y"]),
          np.array([1, 0, 1], dtype=np.int64),
          -1,
      ),
      (
          "with_missing_match",
          np.array([b"X", b"Y"]),
          np.array([b"Y", b"X", b"Z"]),
          np.array([1, 0, -1], dtype=np.int64),
          2,
      ),
      (
          "empty_index",
          np.array([], dtype=np.bytes_),
          np.array([], dtype=np.bytes_),
          np.array([], dtype=np.int64),
          -1,
      ),
      (
          "empty_value",
          np.array([b"X", b""]),
          np.array([b"", b"X", b"Z"]),
          np.array([1, 0, -1]),
          2,
      ),
  )
  def test_NumpyBytesArray(
      self,
      index: np.ndarray,
      input_array: np.ndarray,
      expected_result: np.ndarray,
      expected_missmatch: int,
  ):
    mapper = lib.ByteIdToIdxMapper(index)
    result, missmatch = mapper(input_array)
    test_util.assert_are_equal(self, result, expected_result)
    self.assertEqual(missmatch, expected_missmatch)

  def test_two_NumpyBytesArray(self):
    mapper1 = lib.ByteIdToIdxMapper(np.array([b"X", b"Y"]))
    mapper2 = lib.ByteIdToIdxMapper(np.array([b"Z", b"X"]))

    result, missmatch1, missmatch2 = lib.PairMapping(
        mapper1,
        mapper2,
        np.array([b"Y", b"X", b"_"]),
        np.array([b"Z", b"Z", b"X"]),
        3,
    )
    test_util.assert_are_equal(
        self, result, np.array([[1, 0, -1], [0, 0, 1]], dtype=np.int64)
    )
    self.assertEqual(missmatch1, 2)
    self.assertEqual(missmatch2, -1)


if __name__ == "__main__":
  absltest.main()
