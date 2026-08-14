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

"""Common utilities for TF-based graph serialization and processing."""

import enum
from typing import Optional

from dgf.src.data import schema as schema_lib
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
import numpy as np


class TfExampleContainer(enum.Enum):
  TF_RECORD = "TF_RECORD"
  SSTABLE = "SSTABLE"
  RECORDIO = "RECORDIO"


DEFAULT_KEY_ID = "#id"
KEY_SOURCE = "#source"
KEY_TARGET = "#target"


def populate_features(
    example: tf.train.Example,
    index: int,
    feature_schema: schema_lib.FeatureSetSchema,
    features: dict[str, np.ndarray],
    ignore_keys: tuple[str, ...],
):
  """Populates a tf.train.Example with features from a target schema definition.

  Args:
    example: Output tf.train.Example to populate.
    index: Row index corresponding to the node/edge in the structural batched
      arrays.
    feature_schema: Schema dictionary mapping feature layouts and types.
    features: Dictionary mapping keys to layout arrays.
    ignore_keys: Tuple of features to explicitly skip mapping.
  """
  for feature_name, feature_desc in feature_schema.items():
    if feature_name in ignore_keys:
      continue
    feature_values = features[feature_name]
    if not feature_desc.shape:
      feature_value = [feature_values[index]]
    else:
      feature_value = feature_values[index]

    val_array = np.asarray(feature_value)
    feature_format = feature_desc.format
    val_flat = val_array.ravel()

    if feature_format.is_integer():
      example.features.feature[feature_name].int64_list.value.extend(val_flat)
    elif feature_format.is_float():
      example.features.feature[feature_name].float_list.value.extend(val_flat)
    elif feature_format == schema_lib.FeatureFormat.BYTES:
      if val_array.dtype.kind == "S":
        encoded_value = val_flat
      else:
        encoded_value = [
            v if isinstance(v, bytes) else str(v).encode("utf-8")
            for v in val_flat
        ]
      example.features.feature[feature_name].bytes_list.value.extend(
          encoded_value
      )
    else:
      raise ValueError(
          f"Unsupported format {feature_format} for feature {feature_name} "
          f"with value {feature_value}"
      )


def set_tf_scalar(
    example: tf.train.Example,
    feature_name: str,
    value: int | bytes,
    format: schema_lib.FeatureFormat,
):
  """Sets a scalar value in a tf example."""
  if format.is_integer():
    example.features.feature[feature_name].int64_list.value.append(value)
  elif format == schema_lib.FeatureFormat.BYTES:
    if not isinstance(value, bytes):
      value = str(value).encode("utf-8")
    example.features.feature[feature_name].bytes_list.value.append(value)
  else:
    raise ValueError(f"Unsupported format: {format}")


def maybe_set_id_column(
    example: tf.train.Example,
    index: int,
    feature_schema: schema_lib.FeatureSetSchema,
    features: dict[str, np.ndarray],
    id_column: Optional[str],
):
  """Sets the id column if present in features and schema."""
  if (
      id_column is not None
      and id_column in feature_schema
      and DEFAULT_KEY_ID in features
      and id_column not in example.features.feature
  ):
    set_tf_scalar(
        example,
        id_column,
        features[DEFAULT_KEY_ID][index],
        feature_schema[id_column].format,
    )


def in_memory_node_to_tf_example(
    node_index: int,
    feature_schema: schema_lib.FeatureSetSchema,
    features: dict[str, np.ndarray] | None,
    node_id_column: Optional[str],
    ignore_keys: tuple[str, ...] = (),
) -> tf.train.Example:
  """Builds a tf.train.Example for a single node inside an InMemoryNodeSet.

  Args:
    node_index: The layout index of the node mapped inside the features.
    feature_schema: Schema describing features and shapes bound to this nodeset.
    features: A set of layout arrays for the features containing `node_index`.
    node_id_column: Explicit string identifier column if the dataset uses #id.
    ignore_keys: Pre-calculated tuple of keys to skip copying.

  Returns:
    A populated tf.train.Example representing precisely this node index.
  """
  example = tf.train.Example()
  if features is not None:
    populate_features(
        example, node_index, feature_schema, features, ignore_keys
    )

    maybe_set_id_column(
        example, node_index, feature_schema, features, node_id_column
    )

  return example


def in_memory_edge_to_tf_example(
    edge_index: int,
    feature_schema: schema_lib.FeatureSetSchema,
    source: int | bytes,
    source_format: schema_lib.FeatureFormat,
    target: int | bytes,
    target_format: schema_lib.FeatureFormat,
    features: dict[str, np.ndarray] | None,
    edge_id_column: Optional[str],
    ignore_keys: tuple[str, ...] = (KEY_SOURCE, KEY_TARGET),
) -> tf.train.Example:
  """Builds a tf.train.Example for a single edge inside an InMemoryEdgeSet.

  Args:
    edge_index: The layout index of the edge mapped inside the structural
      features.
    feature_schema: Schema describing features and shapes bound to this edgeset.
    source: Single scalar/bytes representing the source node id.
    source_format: FeatureFormat configuring storage dimensions for the source
      node id.
    target: Single scalar/bytes representing the target node id.
    target_format: FeatureFormat configuring storage dimensions for the target
      node id.
    features: A set of layout arrays for the edge features containing
      `edge_index`.
    edge_id_column: Explicit string identifier column if the edgeset utilizes
      edge IDs.
    ignore_keys: Pre-calculated tuple of keys to skip copying.

  Returns:
    A populated tf.train.Example representing precisely this edge index.
  """
  example = tf.train.Example()
  set_tf_scalar(example, KEY_SOURCE, source, source_format)
  set_tf_scalar(example, KEY_TARGET, target, target_format)

  if features is not None:
    populate_features(
        example, edge_index, feature_schema, features, ignore_keys
    )

    maybe_set_id_column(
        example, edge_index, feature_schema, features, edge_id_column
    )

  return example


def tf_feature_to_feature(
    example: tf.train.Example,
    key: str,
    feature_schema: schema_lib.FeatureSchema,
) -> np.ndarray:
  """Extracts features from a tf.train.Example feature."""

  feature = example.features.feature.get(key)
  if feature is None:
    raise ValueError(f"Missing feature {key}")
  if feature.HasField("int64_list"):
    value = np.array(feature.int64_list.value, dtype=np.int64)
  elif feature.HasField("float_list"):
    value = np.array(feature.float_list.value, dtype=np.float32)
  elif feature.HasField("bytes_list"):
    value = np.array(feature.bytes_list.value, dtype=np.bytes_)
  else:
    raise ValueError("Non supported type")

  if feature_schema.shape is None or feature_schema.shape == ():
    if value.shape[0] != 1:
      raise ValueError(
          f"Expected scalar value for feature '{key}' but got value with shape"
          f" {value.shape}. If the feature is multi-dimensional, its `shape`"
          " should be specified in the Graph Schema. Note: If you cannot fix"
          " the schema file, use the `override_schema` or `schema_transformer`"
          " argument of the `read_graphai_hgraph` function."
      )
    value = np.squeeze(value, axis=0)
  else:
    # Shape of the expected array. Replace None values with -1 in the shape:
    # DGF uses None for unknown shape while NP uses -1.
    # (1,None,4) => (1,-1,4)
    expected_shape = [s if s is not None else -1 for s in feature_schema.shape]
    value = np.reshape(value, expected_shape)
  return value


def tf_feature_to_bytes(example: tf.train.Example, key: str) -> bytes | int:
  """Extracts a byte value from a tf.train.Example feature."""
  feature = example.features.feature.get(key)
  if feature is None:
    raise ValueError(f"Missing feature {key}")
  if feature.HasField("bytes_list"):
    if len(feature.bytes_list.value) != 1:
      raise ValueError(
          f"Expected a single bytes value for {key}. Instead got"
          f" {len(feature.bytes_list.value)} values."
      )
    return feature.bytes_list.value[0]
  elif feature.HasField("int64_list"):
    if len(feature.int64_list.value) != 1:
      raise ValueError(
          f"Expected a single int value for {key}. Instead got"
          f" {len(feature.int64_list.value)} values."
      )
    return feature.int64_list.value[0]
  else:
    raise ValueError("Non supported type")


def extract_features(
    example: tf.train.Example,
    schema_features: schema_lib.FeatureSetSchema,
    ignore_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
  """Extracts multiple features from a tf example."""
  extracted = {}
  for feature_name, feature_schema in schema_features.items():
    if feature_name in ignore_keys:
      continue
    extracted[feature_name] = tf_feature_to_feature(
        example, feature_name, feature_schema
    )
  return extracted
