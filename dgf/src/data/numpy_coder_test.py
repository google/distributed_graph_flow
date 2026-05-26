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

import logging
from absl.testing import absltest
from absl.testing import parameterized
from apache_beam.coders import typecoders
from apache_beam.typehints import trivial_inference
from dgf.src.data import numpy_coder
from dgf.src.util import test_util
import numpy as np


def _multi_dim_variable_shape_array() -> np.ndarray:
  all_features = [
      np.array([[1, 2], [3, 4]], dtype=np.int64),
      np.array([[4], [5]], dtype=np.int64),
  ]
  ret = np.empty(len(all_features), dtype=np.object_)
  ret[:] = all_features
  return ret


class NumpyCoderTest(parameterized.TestCase):

  def test_registration(self):
    value = np.array([1, 2, 3])
    value_type = trivial_inference.instance_to_type(value)
    value_coder = typecoders.registry.get_coder(value_type)
    self.assertIsInstance(value_coder, numpy_coder.NDArrayCoder)

  @parameterized.parameters(
      (np.array([], np.int32),),
      (np.array([], np.float32),),
      (np.array([[]], np.int64),),
      (np.array([[], []], np.float32),),
      (np.array([1, 2, 3]),),
      (np.array([[1], [2], [3]]),),
      (np.array(["1", "2", "3"]),),
      (np.array([["a", "b"], ["c", "d"]]),),
      (np.array([b"a", b"bbb", b"cccc", b"ddddd"]),),
      (np.array([1.0, 2.0, 3.0]),),
      (np.array([[1.0, 2.0], [3.0, 4.0]]),),
      (np.array([[[True], [False]], [[False], [True]]]),),
      (
          np.array(
              [
                  np.array([1, 2, 3]),
                  np.array([4, 5]),
              ],
              dtype=np.object_,
          ),
      ),
      (_multi_dim_variable_shape_array(),),
  )
  def test_encoding_and_decoding(self, value):
    coder = numpy_coder.NDArrayCoder()
    encoded = coder.encode(value)
    decoded = coder.decode(encoded)
    test_util.assert_are_equal(self, value, decoded)

    logging.info(
        "value:%s coder:%s serialized_len:%d",
        value,
        coder,
        len(encoded),
    )


if __name__ == "__main__":
  absltest.main()
