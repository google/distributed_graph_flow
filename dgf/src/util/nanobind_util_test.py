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
from dgf.src.util import nanobind_util_test_ext as lib
from dgf.src.util import test_util
import numpy as np


# Note: We keep the C++ / uppercase function name in the test names.
class NanoBindUtilTest(absltest.TestCase):

  def test_NumpyBytesArray(self):
    a = np.array([b"a", b"bcd", b"", b"f"], dtype=np.bytes_)
    lib.TestNumpyBytesArray(a)

  def test_CCVectorToNumpyArray(self):
    test_util.assert_are_equal(
        self,
        lib.CCVectorToNumpyArrayInt32ToInt32([1, 2, 3]),
        np.array([1, 2, 3], dtype=np.int32),
    )
    test_util.assert_are_equal(
        self,
        lib.CCVectorToNumpyArrayInt64ToInt32([1, 2, 3]),
        np.array([1, 2, 3], dtype=np.int32),
    )

  def test_ListOfBytesToVectorOfStrings(self):
    self.assertEqual(
        lib.ListOfBytesToVectorOfStrings([b"a", b"b", b"c"]), ["a", "b", "c"]
    )

    # Test with invalid inputs.
    with self.assertRaises(ValueError):
      lib.ListOfBytesToVectorOfStrings("not a list")
    with self.assertRaises(ValueError):
      lib.ListOfBytesToVectorOfStrings([b"a", 1, b"c"])
    with self.assertRaises(ValueError):
      lib.ListOfBytesToVectorOfStrings([1, b"b"])

  def test_CCArrayToNumpyArray(self):
    test_util.assert_are_equal(
        self,
        lib.TestCCArrayToNumpyArray(["hello", "abc"]),
        np.array([b"hello", b"abc"], dtype=np.bytes_),
    )
    test_util.assert_are_equal(
        self,
        lib.TestCCArrayToNumpyArray([]),
        np.array([], dtype=np.bytes_),
    )


if __name__ == "__main__":
  absltest.main()
