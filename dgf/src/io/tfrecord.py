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

"""Utilities to read/write TFRecord files efficiently in memory."""

import datetime
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf


# Note: With minimal changes, can also support sstable and recordios.
def read_tf_record(
    paths: list[str],
    columns: Dict[str, Tuple[tf.DType, Tuple[int, ...]]],
    preserve_order: bool,
    compressed: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, np.ndarray], int]:
  """Reads a TensorFlow Records data and return a dict of numpy arrays.

  This implementation is optimized for speed by using tf.data and vectorized
  parsing.

  Args:
    paths: List of paths to TFRecord files. Supports sharded paths.
    columns: Columns to read, as a dictionary of column name to TensorFlow dtype
      + shape. For example: `{'feature1': (tf.string, ()), 'feature2':
      (tf.float32, (2))}`. All features are assumed to be of variable length.
      The original `columns` as a list of strings is not supported as
      `tf.io.parse_example` requires type information beforehand.
    preserve_order: If True, the order of records is preserved as they appear in
      the input files, reading shards sequentially. If False, shards are read in
      parallel using `tf.data.Dataset.interleave`, which is faster but does not
      guarantee order.
    compressed: Whether the TFRecord is compressed.
    verbose: If True, print status of the dataset reading.

  Returns:
    A dict of numpy arrays and the number of rows.
  """
  if len(columns) != len(set(columns.keys())):
    raise ValueError(f"The 'columns' dict contains duplicate keys: {columns}")

  time_begin = datetime.datetime.now()
  if verbose:
    print(f"Reading {len(paths)} shard(s)")

  # Create a feature specification for parsing.
  # We assume all features are VarLenFeatures, which matches the flexibility of
  # the previous implementation.
  feature_spec = {
      col: tf.io.VarLenFeature(dtype) for col, (dtype, _) in columns.items()
  }

  # Create a dataset of file paths.
  path_dataset = tf.data.Dataset.from_tensor_slices(paths)

  def _read_tfrecord_dataset(path):
    return tf.data.TFRecordDataset(
        path, compression_type="GZIP" if compressed else ""
    )
  if not paths:
    raise ValueError("paths should not be empty")
  if preserve_order or len(paths) <= 1:
    # Use flat_map to read records from each file sequentially,
    dataset = path_dataset.flat_map(_read_tfrecord_dataset)
  else:
    # Use interleave to read records from multiple files in parallel.
    dataset = path_dataset.interleave(
        _read_tfrecord_dataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

  # Batch records and parse them in parallel.
  dataset = dataset.batch(
      1024, num_parallel_calls=None if preserve_order else tf.data.AUTOTUNE
  )
  dataset = dataset.map(
      lambda x: tf.io.parse_example(x, feature_spec),
      num_parallel_calls=tf.data.AUTOTUNE,
  )
  dataset = dataset.prefetch(tf.data.AUTOTUNE)

  # Collect the data from the dataset.
  data: Dict[str, List[np.ndarray]] = {column: [] for column in columns}
  num_examples = 0
  for batch in dataset:
    batch_size = 0
    for column, values in batch.items():
      # Note: VarLenFeature produces a SparseTensor.
      dense_tensor = tf.sparse.to_dense(values).numpy()
      if dense_tensor.dtype == object:
        dense_tensor = dense_tensor.astype(np.bytes_)
      batch_size = dense_tensor.shape[0]
      data[column].append(dense_tensor)
    num_examples += batch_size

  if verbose:
    print(
        f"{num_examples} examples read in"
        f" {datetime.datetime.now() - time_begin}",
    )

  # Concatenate batches and finalize data.
  final_data: Dict[str, np.ndarray] = {}
  for key, (_, shape) in columns.items():
    value = np.concatenate(data[key], axis=0)
    if shape == ():
      if value.shape[1] != 1:
        raise ValueError(
            f"Expected scalar value for feature '{key}' but got value with"
            f" shape {value.shape[1:]}. If the feature is multi-dimentionnal,"
            " its `shape` should be specified in the Graph Schema. Note: If"
            " you cannot fix the schema file, use the `override_schema` or"
            " `schema_transformer` argument."
        )
      value = np.squeeze(value, axis=1)
    else:
      value = np.reshape(value, (-1,) + shape)
    final_data[key] = value

  return final_data, num_examples


def write_tf_record(
    path: str, examples: list[tf.train.Example], compression: str = "GZIP"
):
  """Writes a list of tf.train.Example to a TFRecord file."""
  with tf.io.TFRecordWriter(path, options=compression) as writer:
    for example in examples:
      writer.write(example.SerializeToString())
