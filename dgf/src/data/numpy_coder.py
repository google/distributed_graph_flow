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

"""Efficient serialization of numpy arrays in beam."""

import apache_beam as beam
from apache_beam.coders import typecoders
from apache_beam.typehints import typehints
import numpy as np


class NDArrayCoder(beam.coders.Coder):
  """Beam coder for Numpy N-dimensional array of TF-compatible data types.

  Supports all numeric data types and bytes (represented as `np.object_`).
  The numpy array is serialized as a tuple of `(dtype, shape, flat values)`.
  For numeric values serialization we rely on `tobytes()` and `frombuffer` from
  the numpy library. It, seems, has the best speed/space tradeoffs. Tensorflow
  represents `tf.string` as `np.object_` (as `np.string_` is for arrays
  containing fixed-width byte strings, which can lead to lots of wasted
  memory). Because `np.object_` is an array of references to arbitrary
  objects, we could not rely on numpy native serialization and using
  `IterableCoder` from the Beam library instead.

  NOTE: for some simple stages the execution time may be dominated by data
  serialization/deserialization, so any imporvement here translates directly to
  the total execution costs.

  From: tensorflow_gnn/experimental/sampler/beam/executor_lib.py
  """

  def __init__(self):
    Tuple = typehints.Tuple

    encoded_struct = Tuple[str, Tuple[int, ...], bytes]
    self._coder = typecoders.registry.get_coder(encoded_struct)

    # Store: { sub-dtype, {shape, raw bytes}*} for the sub arrays.
    self._sub_array_coder = typecoders.registry.get_coder(
        Tuple[str, typehints.Iterable[Tuple[Tuple[int, ...], bytes]]]
    )

  def encode(self, value: np.ndarray) -> bytes:
    if value.dtype == np.object_:
      if len(value) > 0:
        sub_dtype = value[0].dtype.str
      else:
        sub_dtype = np.int32.str
      flat_values = self._sub_array_coder.encode(
          (sub_dtype, ((x.shape, x.tobytes()) for x in value.flat))
      )
    else:
      flat_values = value.tobytes()
    return self._coder.encode((value.dtype.str, value.shape, flat_values))

  def decode(self, encoded: bytes) -> np.ndarray:
    dtype_str, shape, serialized_values = self._coder.decode(encoded)
    dtype = np.dtype(dtype_str)
    if dtype == np.object_:
      subdtype_str, sub_serialized_values = self._sub_array_coder.decode(
          serialized_values
      )
      sub_dtype = np.dtype(subdtype_str)
      sub_arrays = [
          np.frombuffer(sub_raw_value, dtype=sub_dtype).reshape(sub_shape)
          for sub_shape, sub_raw_value in sub_serialized_values
      ]
      # Note: We cannot build the array with np.array(sub_arrays) as numpy will
      # try to merge the arrays.
      flat_values = np.empty(len(sub_arrays), dtype=np.object_)
      flat_values[:] = sub_arrays
    else:
      flat_values = np.frombuffer(serialized_values, dtype=dtype)
    return np.reshape(flat_values, shape)

  def is_deterministic(self):
    return True

  def to_type_hint(self):
    return np.ndarray


beam.coders.registry.register_coder(np.ndarray, NDArrayCoder)
