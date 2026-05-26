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

import os
import tempfile
from absl.testing import absltest
from dgf.src.io import tfrecord
import numpy as np
import tensorflow as tf


class TfRecordTest(absltest.TestCase):

  def _write_tfrecord(self, path, examples, compressed=False):
    options = tf.io.TFRecordOptions(
        compression_type="GZIP" if compressed else ""
    )
    with tf.io.TFRecordWriter(path, options=options) as writer:
      for example in examples:
        writer.write(example.SerializeToString())

  def _create_example(self, f1_val, f2_val, f3_val, f4_val):
    feature = {
        "f1": tf.train.Feature(float_list=tf.train.FloatList(value=[f1_val])),
        "f2": tf.train.Feature(int64_list=tf.train.Int64List(value=f2_val)),
        "f3": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[f3_val.encode("utf-8")])
        ),
        "f4": tf.train.Feature(int64_list=tf.train.Int64List(value=[f4_val])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

  def test_read_tf_record_sharded(self):
    test_dir = tempfile.mkdtemp()
    file_path1 = os.path.join(test_dir, "shard-00000-of-00002.tfrecord")
    file_path2 = os.path.join(test_dir, "shard-00001-of-00002.tfrecord")

    examples1 = [
        self._create_example(1.0, [10, 11], "A", 100),
        self._create_example(2.0, [12, 13], "B", 200),
    ]
    examples2 = [
        self._create_example(3.0, [14, 15], "C", 300),
    ]

    self._write_tfrecord(file_path1, examples1)
    self._write_tfrecord(file_path2, examples2)

    data, num_examples = tfrecord.read_tf_record(
        [file_path1, file_path2],
        {
            "f1": (tf.float32, ()),
            "f2": (tf.int64, (2, 1)),
            "f3": (tf.string, (1,)),
        },
        compressed=False,
        preserve_order=True,
    )

    np.testing.assert_array_equal(data["f1"], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(
        data["f2"], np.array([[[10], [11]], [[12], [13]], [[14], [15]]])
    )
    np.testing.assert_array_equal(
        data["f3"], np.array([[b"A"], [b"B"], [b"C"]])
    )
    self.assertEqual(num_examples, 3)


if __name__ == "__main__":
  absltest.main()
