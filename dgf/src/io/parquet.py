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

"""Utilities to read parquet files."""

from concurrent import futures
import dataclasses
import os
import time
from typing import Dict, Optional, Sequence, Tuple
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.util import filesystem
from dgf.src.util import log
from dgf.src.util import shard as shard_lib
import numba
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@numba.njit(parallel=False)
def _numba_copy_kernel(offset, offsets: np.ndarray, data, out, max_len):
  """Copies data from a flat array to a padded output array using Numba."""
  for i in numba.prange(len(offsets) - 1):
    start = offsets[i]
    end = offsets[i + 1]
    length = end - start
    local_offset = i * max_len + offset
    for j in range(length):
      out[local_offset + j] = data[start + j]


def arrow_to_numpy_numba(
    arr: pa.Array, offset: int, max_len: int, out: np.ndarray
):
  """Copies a pyarrow binary chunk to a numpy array of bytes.

  Args:
    arr: The pyarrow array chunk to copy.
    offset: The starting offset in the output numpy array.
    max_len: The maximum length of any binary/string in the chunk.
    out: The pre-allocated numpy array to write the bytes to.
  """
  offset_dtype = np.int64 if pa.types.is_large_binary(arr.type) else np.int32
  offsets = np.frombuffer(arr.buffers()[1], dtype=offset_dtype)
  data = np.frombuffer(arr.buffers()[2], dtype=np.uint8)
  _numba_copy_kernel(offset, offsets, data, out, max_len)


def py_arrow_chunked_binary_to_np_bytes(column: pa.ChunkedArray) -> np.ndarray:
  """Copies a pyarrow binary chunked array to a numpy array of bytes.

  Args:
    column: A pyarrow ChunkedArray containing binary or string data.

  Returns:
    A numpy array of type `bytes`.
  """
  # TODO(gbm): Implement the full function in c++.
  max_len = pc.max(pa.compute.binary_length(column)).as_py()  # pytype: disable=module-attr
  if max_len is None:
    max_len = 1
  num_values = len(column)
  out = np.zeros(num_values * max_len, dtype=np.uint8)
  offset = 0
  for arr in column.chunks:
    arrow_to_numpy_numba(arr, offset, max_len, out)
    offset += len(arr) * max_len
  return out.view(f"S{max_len}")


def _table_to_numpy_dict(table: pa.Table) -> Dict[str, np.ndarray]:
  """Converts a pyarrow Table to a dictionary of numpy arrays."""

  def process_column(value):

    pq_type = value.type
    if pa.types.is_binary(pq_type) or pa.types.is_large_binary(pq_type):
      # A numpy array of np.bytes
      return py_arrow_chunked_binary_to_np_bytes(value)
    else:
      # Can be a simple numpy array, or a numpy array of numpy array.
      feature = value.to_numpy()

      if pa.types.is_fixed_size_list(pq_type):
        # TODO(gbm): Directly create the final feature (skip the vstack and
        # np.object steps).
        if len(feature) > 0:  # pylint: disable=g-explicit-length-test
          feature = np.vstack(feature)
        pq_type = pq_type.value_type

      if pq_type == pa.binary() or pq_type == pa.string():
        feature = feature.astype(np.bytes_)

    return feature

  return {key: process_column(table[key]) for key in table.column_names}


def _read_single_parquet_file(
    path: str, columns: Optional[Sequence[str]] = None
) -> Dict[str, np.ndarray]:
  """Reads a single Parquet file into a pyarrow Table."""
  try:
    with filesystem.open_read(path, binary=True) as f:
      table = pq.ParquetFile(f).read(columns=columns)
      # Note: We immediately convert the table into numpy array.
      return _table_to_numpy_dict(table)
  except Exception as e:
    raise ValueError(f"Failed to read Parquet file {path!r}") from e


def read_parquet_to_numpy_dict(
    paths: Sequence[str],
    columns: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, np.ndarray], int]:
  """Reads multiple Parquet files into a single pyarrow Table.

  Args:
    paths: A sequence of paths to the Parquet files to read.
    columns: An optional sequence of column names to read.
    verbose: Show informations.

  Returns:
    A tuple containing: a dictionary mapping column names to concatenated numpy
    arrays, and the total number of rows.
  """

  with futures.ThreadPoolExecutor(
      max_workers=max(1, min(len(paths), 20))
  ) as executor:
    future_list = [
        executor.submit(_read_single_parquet_file, path, columns)
        for path in paths
    ]
    num_paths = len(paths)
    chunks = []
    last_time = time.monotonic()
    for path_idx, future in enumerate(future_list):
      result = future.result()
      if result:
        chunks.append(result)

      if verbose:
        cur_time = time.monotonic()
        if cur_time - last_time > 5:
          log.info(f".reading parquet file {path_idx+1}/{num_paths}")
          last_time = cur_time

  if verbose:
    log.info(".concatenating values")
  final_data: Dict[str, np.ndarray] = {}
  if not chunks:
    return final_data, 0
  columns = list(chunks[0].keys())
  for key in columns:
    column_chunks = []
    for chunk in chunks:
      column_chunk = chunk[key]
      if len(column_chunk) > 0:  # pylint: disable=g-explicit-length-test
        column_chunks.append(column_chunk)
    if column_chunks:
      final_data[key] = np.concatenate(column_chunks, axis=0)
    else:
      final_data[key] = chunks[0][key]
  num_rows = final_data[columns[0]].shape[0] if columns else 0
  return final_data, num_rows

  # TODO: b/454335246 - Add option to read the files without ordering.
  # TODO: b/454335246 - Avoid the multiple memory copies.
  # TODO: b/454335246 - Better tune ThreadPoolExecutor.
  # TODO: b/454335246 - Add option to return chunks of numpy arrays.


def feature_schema_to_py_arrow_full_data_type(
    schema: schema_lib.FeatureSchema,
) -> pa.DataType:
  """Converts a FeatureSchema to a pyarrow DataType."""
  current_type = feature_format_lib.FEATURE_FORMAT_TO_PYARROW_DATA_TYPE[
      schema.format
  ]
  if schema.shape is None or schema.shape == ():
    return current_type  # Scalar feature

  for size in reversed(schema.shape):
    if size is None:
      # Variable-length list
      current_type = pa.list_(current_type)
    elif isinstance(size, int) and size > 0:
      # Fixed-size list
      current_type = pa.list_(current_type, size)
    else:
      raise ValueError(f"Invalid dimension size in shape: {size}")
  return current_type


@dataclasses.dataclass
class WriteParquetSpec:
  pa_type: pa.DataType
  schema: schema_lib.FeatureSchema
  is_scalar: bool
  is_static_shape: bool


def _write_single_shard(
    data: Dict[str, np.ndarray],
    shard_path: str,
    start_row: int,
    end_row: int,
    write_specs: Dict[str, WriteParquetSpec],
    compression: str = "snappy",
):

  def to_pa_array(key, np_values, spec):
    try:

      if spec.is_scalar:
        return pa.array(np_values, type=spec.pa_type)
      elif spec.is_static_shape:
        return pa.FixedSizeListArray.from_arrays(
            np.ravel(np_values, order="C"),
            type=spec.pa_type,
        )
      else:
        # Slow unrolling of all the feature values.
        return pa.array([x.tolist() for x in np_values], type=spec.pa_type)
    except Exception as e:
      raise ValueError(
          f"Failed to write feature {key!r} with values="
          f" {np_values!r} and spec={spec!r}"
      ) from e

  shard_data = {
      key: to_pa_array(key, data[key][start_row:end_row], spec)
      for key, spec in write_specs.items()
  }
  table = pa.Table.from_pydict(shard_data)
  with filesystem.open_write(shard_path, binary=True) as f:
    pq.write_table(table, f, compression=compression)


def write_numpy_dict_to_parquet(
    data: Dict[str, np.ndarray],
    filename: str,
    base_path: str,
    schema: schema_lib.FeatureSetSchema,
    num_shards: int = 1,
    verbose: bool = False,
    compression: str = "snappy",
):
  """Writes a dictionary of numpy arrays to sharded Parquet files.

  Args:
    data: A dictionary mapping column names to numpy arrays.
    filename: The base filename for the sharded Parquet files.
    base_path: The base path for the sharded Parquet files.
    schema: The schema defining the features to write.
    num_shards: The number of shards to write.
    verbose: If True, print progress information.
    compression: The parquet compression codec to use.
  """
  if not data:
    raise ValueError("Input data dictionary is empty.")

  # Determine the number of rows from the first column.
  first_key = next(iter(data))
  num_rows = data[first_key].shape[0]

  rows_per_shard = (num_rows + num_shards - 1) // num_shards

  write_specs = {
      feature_name: WriteParquetSpec(
          pa_type=feature_schema_to_py_arrow_full_data_type(feature_schema),
          schema=feature_schema,
          is_scalar=feature_schema.shape is None or feature_schema.shape == (),
          is_static_shape=feature_schema.is_static_shape(),
      )
      for feature_name, feature_schema in schema.items()
  }

  def _write_shard_task(shard_index: int):
    start_row = shard_index * rows_per_shard
    end_row = min((shard_index + 1) * rows_per_shard, num_rows)

    if start_row >= end_row:
      return

    shard_filename = shard_lib.sharded_filename(
        filename=filename,
        shard=shard_index,
        num_shards=num_shards,
        extension=".parquet",
    )
    shard_path = os.path.join(base_path, shard_filename)

    if verbose:
      print(f"Writing shard {shard_index + 1}/{num_shards} to {shard_path}")

    _write_single_shard(
        data,
        shard_path,
        start_row,
        end_row,
        write_specs,
        compression=compression,
    )

  with futures.ThreadPoolExecutor(
      max_workers=max(1, min(num_shards, 20))
  ) as executor:
    future_to_shard = {
        executor.submit(_write_shard_task, i): i for i in range(num_shards)
    }
    for future in futures.as_completed(future_to_shard):
      future.result()
