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

r"""Test the parquet io utilities.

You can change the WORKING_DIR variable to also run the test on CNS and GCS.
However, you cannot run such test with blaze test. Instead, run:

blaze build -c opt //third_party/py/dgf/src/io:parquet_test && \
blaze-bin/third_party/py/dgf/src/io/parquet_test

"""

import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.io import parquet as parquet_lib
from dgf.src.util import filesystem
from dgf.src.util import shard as shard_lib
from dgf.src.util import test_util
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Working directory used by the test. If None, use a local tmp directory.
WORKING_DIR = None
# WORKING_DIR = "/cns/is-d/home/gbm/tmp/ttl=15d"
# WORKING_DIR = "gs://graph-flow/test"

# For the cloud version, you probably need to login before:
# gcloud auth application-default login
# gcloud config set project biggraphs-poc

test_util.disable_diff_truncation()


class ParquetTest(parameterized.TestCase):

  def tmp_dir(self):
    if WORKING_DIR:
      tmpdir = WORKING_DIR
      filesystem.makedirs(tmpdir)
    else:
      tmpdir = tempfile.mkdtemp()
    return tmpdir

  def test_base(self):
    tmpdir = self.tmp_dir()
    file1_path = os.path.join(tmpdir, "file1.parquet")
    file2_path = os.path.join(tmpdir, "file2.parquet")
    with filesystem.open_write(file1_path, True) as f:
      pq.write_table(
          pa.Table.from_pydict({
              "f1": pa.array([1, 2]),
              "f2": pa.array([b"a", b"b"]),
              "f3": pa.array([[1, 2], [3, 4]], type=pa.list_(pa.float64(), 2)),
              "f4": pa.array([[1], [2, 3]], type=pa.list_(pa.float64())),
              "f5": pa.array(
                  [["a", "b"], ["c", "d"]], type=pa.list_(pa.binary(), 2)
              ),
          }),
          f,
      )
    with filesystem.open_write(file2_path, True) as f:
      pq.write_table(
          pa.Table.from_pydict({
              "f1": pa.array([3, 4]),
              "f2": pa.array([b"c", b"d"]),
              "f3": pa.array([[5, 6], [7, 8]], type=pa.list_(pa.float64(), 2)),
              "f4": pa.array([[], [4, 5, 6]], type=pa.list_(pa.float64())),
              "f5": pa.array(
                  [["e", "f"], ["g", "h"]], type=pa.list_(pa.binary(), 2)
              ),
          }),
          f,
      )

    data, num_rows = parquet_lib.read_parquet_to_numpy_dict(
        [file1_path, file2_path]
    )
    self.assertEqual(num_rows, 4)
    test_util.assert_are_equal(
        self,
        data,
        {
            "f1": np.array([1, 2, 3, 4]),
            "f2": np.array([b"a", b"b", b"c", b"d"], dtype="|S1"),
            "f3": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            "f4": np.array(
                [
                    np.array([1]),
                    np.array([2, 3]),
                    np.array([], dtype=np.float64),
                    np.array([4, 5, 6]),
                ],
                dtype=np.object_,
            ),
            "f5": np.array(
                [[b"a", b"b"], [b"c", b"d"], [b"e", b"f"], [b"g", b"h"]],
                dtype="|S1",
            ),
        },
    )

  def test_read_empty(self):
    tmpdir = self.tmp_dir()
    file1_path = os.path.join(tmpdir, "file1.parquet")
    file2_path = os.path.join(tmpdir, "file2.parquet")
    with filesystem.open_write(file1_path, True) as f:
      pq.write_table(
          pa.Table.from_pydict({
              "f1": pa.array([], type=pa.int32()),
              "f2": pa.array([], type=pa.list_(pa.binary(), 2)),
          }),
          f,
      )
    with filesystem.open_write(file2_path, True) as f:
      pq.write_table(
          pa.Table.from_pydict({}),
          f,
      )
    data, num_rows = parquet_lib.read_parquet_to_numpy_dict(
        [file1_path, file2_path]
    )
    self.assertEqual(num_rows, 0)
    test_util.assert_are_equal(
        self,
        data,
        {"f1": np.array([], dtype=np.int32), "f2": np.array([], dtype="|S2")},
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="multiple_chunks",
          chunked_arr=pa.chunked_array([
              pa.array([b"a", b"bb"], type=pa.binary()),
              pa.array([b"ccc", b"d"], type=pa.binary()),
              pa.array([b"eeee"], type=pa.binary()),
          ]),
          expected=np.array([b"a", b"bb", b"ccc", b"d", b"eeee"], dtype="|S4"),
      ),
      dict(
          testcase_name="empty_strings",
          chunked_arr=pa.chunked_array([
              pa.array([b"", b"a"], type=pa.binary()),
              pa.array([b"bb", b""], type=pa.binary()),
          ]),
          expected=np.array([b"", b"a", b"bb", b""], dtype="|S2"),
      ),
      dict(
          testcase_name="only_empty_chunks",
          chunked_arr=pa.chunked_array([
              pa.array([], type=pa.binary()),
              pa.array([], type=pa.binary()),
          ]),
          expected=np.array([], dtype="|S0"),
      ),
      dict(
          testcase_name="single_chunk",
          chunked_arr=pa.chunked_array([
              pa.array([b"long", b"short"], type=pa.binary()),
          ]),
          expected=np.array([b"long", b"short"], dtype="|S5"),
      ),
      dict(
          testcase_name="large_binary_type",
          chunked_arr=pa.chunked_array([
              pa.array([b"large_a", b"large_bb"], type=pa.large_binary()),
              pa.array([b"large_ccc"], type=pa.large_binary()),
          ]),
          expected=np.array(
              [b"large_a", b"large_bb", b"large_ccc"], dtype="|S9"
          ),
      ),
  )
  def test_py_arrow_chunked_binary_to_np_bytes(self, chunked_arr, expected):
    result = parquet_lib.py_arrow_chunked_binary_to_np_bytes(chunked_arr)
    test_util.assert_are_equal(self, result, expected)

  def test_write_numpy_dict_to_parquet(self):
    tmpdir = self.tmp_dir()
    base_path = os.path.join(tmpdir, "write_test")
    filesystem.makedirs(base_path)

    # TODO(gbm): Enable when reading code support this kind of shape.
    # Avoid Numpy broadcast.
    # f6 = np.empty(2, dtype=np.object_)
    # f6[:] = [
    #     np.array([[8, 9], [10, 11]], dtype=np.int64),
    #     np.array([[12], [13]], dtype=np.int64),
    # ]

    data = {
        "f1": np.array([1, 2]),
        "f2": np.array([b"a", b"b"], dtype=np.bytes_),
        "f3": np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
        "f4": np.array(
            [
                np.array([1], dtype=np.float64),
                np.array([2, 3], dtype=np.float64),
            ],
            dtype=np.object_,
        ),
        "f5": np.array(
            [[b"a", b"b"], [b"c", b"d"]],
            dtype=np.bytes_,
        ),
        # Feature with shape (2, None)
        # "f6": f6,
        # Feature with shape (None, 2)
        # TODO(gbm): Enable when reading code support this kind of shape.
        # "f7": np.array(
        #     [
        #         np.array([[12, 13], [14, 15]], dtype=np.int64),
        #         np.array([[16, 17]], dtype=np.int64),
        #     ],
        #     dtype=np.object_,
        # ),
    }

    # Test writing and reading with a single shard.
    num_shards = 2
    parquet_lib.write_numpy_dict_to_parquet(
        data,
        "data",
        base_path,
        schema={
            "f1": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.INTEGER_64
            ),
            "f2": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.BYTES
            ),
            "f3": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_64, shape=(2,)
            ),
            "f4": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_64, shape=(None,)
            ),
            "f5": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.BYTES, shape=(2,)
            ),
            # "f6": schema_lib.FeatureSchema(
            #     format=schema_lib.FeatureFormat.INTEGER_64, shape=(2, None)
            # ),
            # "f7": schema_lib.FeatureSchema(
            #     format=schema_lib.FeatureFormat.INTEGER_64, shape=(None, 2)
            # ),
        },
        num_shards=num_shards,
    )

    # Construct paths for reading the sharded files using shard_lib.
    file_paths = shard_lib.expand_input_paths(
        os.path.join(base_path, "data@*.parquet")
    )

    read_data, num_rows = parquet_lib.read_parquet_to_numpy_dict(file_paths)
    self.assertEqual(num_rows, 2)
    test_util.assert_are_equal(self, read_data, data)

  def disabled_test_write_numpy_dict_to_parquet_empty(self):
    tmpdir = self.tmp_dir()

    # Test writing an empty dictionary.
    base_path_empty_dict = os.path.join(tmpdir, "write_test_empty_dict")
    filesystem.makedirs(base_path_empty_dict)
    parquet_lib.write_numpy_dict_to_parquet(
        {}, "data", base_path_empty_dict, schema={}, num_shards=1
    )
    self.assertEmpty(filesystem.glob(os.path.join(base_path_empty_dict, "*")))

    # Test writing arrays with 0 rows.
    base_path_zero_rows = os.path.join(tmpdir, "write_test_zero_rows")
    filesystem.makedirs(base_path_zero_rows)
    zero_data = {
        "f1": np.array([], dtype=np.int32),
        "f2": np.array([], dtype=np.float32),
    }
    parquet_lib.write_numpy_dict_to_parquet(
        zero_data,
        "data",
        base_path_zero_rows,
        schema={
            "f1": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.INTEGER_32
            ),
            "f2": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32
            ),
        },
        num_shards=1,
    )
    self.assertEmpty(filesystem.glob(os.path.join(base_path_zero_rows, "*")))

    # Test reading after writing zero rows (should yield empty dict).
    read_data, num_rows = parquet_lib.read_parquet_to_numpy_dict(
        [os.path.join(base_path_zero_rows, "data-00000-of-00001.parquet")]
    )
    self.assertEqual(num_rows, 0)
    test_util.assert_are_equal(self, read_data, zero_data)

  @parameterized.named_parameters(
      dict(
          testcase_name="int64_scalar",
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.INTEGER_64
          ),
          expected_type=pa.int64(),
      ),
      dict(
          testcase_name="bytes_scalar",
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.BYTES
          ),
          expected_type=pa.binary(),
      ),
      dict(
          testcase_name="float32_fixed_list",
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.FLOAT_32, shape=(4,)
          ),
          expected_type=pa.list_(pa.float32(), 4),
      ),
      dict(
          testcase_name="float64_variable_list",
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.FLOAT_64, shape=(None,)
          ),
          expected_type=pa.list_(pa.float64()),
      ),
      dict(
          testcase_name="bool_fixed_variable_list",
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.BOOL, shape=(2, None)
          ),
          expected_type=pa.list_(pa.list_(pa.bool_()), 2),
      ),
      dict(
          testcase_name="int32_empty_shape",
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.INTEGER_32, shape=()
          ),
          expected_type=pa.int32(),
      ),
  )
  def test_feature_schema_to_py_arrow_full_data_type(
      self, schema, expected_type
  ):
    result = parquet_lib.feature_schema_to_py_arrow_full_data_type(schema)
    self.assertEqual(result, expected_type)

  def test_feature_schema_to_py_arrow_full_data_type_invalid_shape(self):
    with self.assertRaisesRegex(ValueError, "Invalid dimension size in shape"):
      parquet_lib.feature_schema_to_py_arrow_full_data_type(
          schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.INTEGER_32, shape=(-1,)
          )
      )


if __name__ == "__main__":
  absltest.main()
