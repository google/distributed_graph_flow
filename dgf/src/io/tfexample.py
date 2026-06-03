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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tqdm


def _read_dataset_generic(
    paths: list[str],
    columns: Dict[str, Tuple[tf.DType, Tuple[int, ...]]],
    preserve_order: bool,
    dataset_creator: Callable[[str], tf.data.Dataset],
    key_column: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, np.ndarray], int]:
  """Generic helper to read a dataset of TF Examples."""
  if len(columns) != len(set(columns.keys())):
    raise ValueError(f"The 'columns' dict contains duplicate keys: {columns}")

  time_begin = datetime.datetime.now()
  if verbose:
    print(f"Reading {len(paths)} shard(s)")

  feature_spec = {
      col: tf.io.VarLenFeature(dtype) for col, (dtype, _) in columns.items()
  }

  path_dataset = tf.data.Dataset.from_tensor_slices(paths)

  if not paths:
    raise ValueError("paths should not be empty")
  if preserve_order or len(paths) <= 1:
    dataset = path_dataset.flat_map(dataset_creator)
  else:
    dataset = path_dataset.interleave(
        dataset_creator,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

  has_keys = (
      isinstance(dataset.element_spec, tuple) and len(dataset.element_spec) == 2
  )

  dataset = dataset.batch(
      1024, num_parallel_calls=None if preserve_order else tf.data.AUTOTUNE
  )
  if has_keys:
    dataset = dataset.map(
        lambda k, v: (k, tf.io.parse_example(v, feature_spec)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
  else:
    dataset = dataset.map(
        lambda x: tf.io.parse_example(x, feature_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
  dataset = dataset.prefetch(tf.data.AUTOTUNE)

  data: Dict[str, List[np.ndarray]] = {column: [] for column in columns}
  num_examples = 0
  pbar = tqdm.tqdm(desc="Reading examples", unit="row") if verbose else None

  inject_keys = has_keys and key_column is not None and key_column in columns

  if has_keys:
    for keys_batch, batch in dataset:
      keys_noop = keys_batch.numpy()
      if keys_noop.dtype == object:
        keys_noop = keys_noop.astype(np.bytes_)
      batch_size = keys_noop.shape[0]

      for column in columns:
        if inject_keys and column == key_column:
          expected_shape = columns[column][1]
          if expected_shape != ():
            raise ValueError(
                f"Key column '{column}' must have scalar shape () but got"
                f" {expected_shape}"
            )
          data[column].append(np.expand_dims(keys_noop, axis=-1))
        else:
          values = batch[column]
          dense_tensor = tf.sparse.to_dense(values).numpy()
          if dense_tensor.dtype == object:
            dense_tensor = dense_tensor.astype(np.bytes_)
          expected_shape = columns[column][1]
          if expected_shape != ():
            expected_flat_size = int(np.prod(expected_shape))
            if dense_tensor.shape[1] != expected_flat_size:
              if dense_tensor.shape[1] == 0:
                dense_tensor = np.zeros(
                    shape=(dense_tensor.shape[0], expected_flat_size),
                    dtype=dense_tensor.dtype,
                )
              else:
                raise ValueError(
                    f"Feature '{column}' has unexpected shape in a batch."
                    f" Expected flat size: {expected_flat_size} (shape:"
                    f" {expected_shape}), but got flat size:"
                    f" {dense_tensor.shape[1]} (shape: {dense_tensor.shape}). "
                )
          data[column].append(dense_tensor)
      num_examples += batch_size
      if pbar is not None:
        pbar.update(batch_size)
  else:
    for batch in dataset:
      batch_size = 0
      for column, values in batch.items():
        dense_tensor = tf.sparse.to_dense(values).numpy()
        if dense_tensor.dtype == object:
          dense_tensor = dense_tensor.astype(np.bytes_)
        expected_shape = columns[column][1]
        if expected_shape != ():
          expected_flat_size = int(np.prod(expected_shape))
          if dense_tensor.shape[1] != expected_flat_size:
            if dense_tensor.shape[1] == 0:
              dense_tensor = np.zeros(
                  shape=(dense_tensor.shape[0], expected_flat_size),
                  dtype=dense_tensor.dtype,
              )
            else:
              raise ValueError(
                  f"Feature '{column}' has unexpected shape in a batch."
                  f" Expected flat size: {expected_flat_size} (shape:"
                  f" {expected_shape}), but got flat size:"
                  f" {dense_tensor.shape[1]} (shape: {dense_tensor.shape}). "
              )
        batch_size = dense_tensor.shape[0]
        data[column].append(dense_tensor)
      num_examples += batch_size
      if pbar is not None:
        pbar.update(batch_size)

  if pbar is not None:
    pbar.close()

  if verbose:
    print(
        f"{num_examples} examples read in"
        f" {datetime.datetime.now() - time_begin}",
    )

  final_data: Dict[str, np.ndarray] = {}
  for key, (_, shape) in columns.items():
    try:
      value = np.concatenate(data[key], axis=0)
    except ValueError as e:
      raise ValueError(
          f"Failed to concatenate data for feature '{key}'. Original error: {e}"
      ) from e
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

  def _read_tfrecord_dataset(path):
    return tf.data.TFRecordDataset(
        path, compression_type="GZIP" if compressed else ""
    )

  return _read_dataset_generic(
      paths=paths,
      columns=columns,
      preserve_order=preserve_order,
      dataset_creator=_read_tfrecord_dataset,
      verbose=verbose,
  )


def write_tf_record(
    path: str, examples: list[tf.train.Example], compression: str = "GZIP"
):
  """Writes a list of tf.train.Example to a TFRecord file."""
  with tf.io.TFRecordWriter(path, options=compression) as writer:
    for example in examples:
      writer.write(example.SerializeToString())
