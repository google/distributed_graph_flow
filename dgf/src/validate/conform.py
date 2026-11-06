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

"""Conform an in-memory graph's features to match its schema."""

from typing import Any

from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
import numpy as np
import pandas as pd


def _is_missing_scalar(val: Any) -> bool:
  """Returns True if a scalar element represents a missing value."""
  if val is None or val is pd.NA or val is pd.NaT:
    return True
  if isinstance(val, (float, np.floating)):
    return bool(np.isnan(val))
  return False


def _conform_array(
    values: Any,
    fmt: schema_lib.FeatureFormat,
    expected_dtype: type[np.generic],
    *,
    source: str,
    feature_name: str,
    missing_bytes: bytes,
) -> np.ndarray:
  """Conforms a static feature array or a single variable-length row."""
  if isinstance(values, pd.Series):
    values = values.array
  if isinstance(values, np.ndarray) and feature_format.np_dtype_matches(
      values.dtype, expected_dtype
  ):
    return values
  if (
      fmt == schema_lib.FeatureFormat.FLOAT_32
      or fmt == schema_lib.FeatureFormat.FLOAT_64
  ):
    if isinstance(values, np.ndarray) and values.dtype.kind in ("f", "i", "u"):
      return values.astype(expected_dtype)
    if (
        isinstance(values, pd.api.extensions.ExtensionArray)
        and pd.api.types.is_numeric_dtype(values.dtype)
        and not pd.api.types.is_bool_dtype(values.dtype)
    ):
      return values.to_numpy(dtype=expected_dtype, na_value=np.nan)
    arr = np.asarray(values, dtype=object)
    inferred = pd.api.types.infer_dtype(arr, skipna=True)
    if inferred not in (
        "floating",
        "integer",
        "integer-na",
        "mixed-integer-float",
        "empty",
    ):
      raise ValueError(
          f"Feature {feature_name!r} in {source} has non-numeric values"
          f" (inferred type {inferred!r}) for format {fmt}."
      )
    return (
        pd.array(arr.ravel(), dtype="Float64")
        .to_numpy(dtype=expected_dtype, na_value=np.nan)
        .reshape(arr.shape)
    )

  elif (
      fmt == schema_lib.FeatureFormat.INTEGER_32
      or fmt == schema_lib.FeatureFormat.INTEGER_64
  ):
    if isinstance(values, np.ndarray) and values.dtype.kind in ("i", "u"):
      return values.astype(expected_dtype)
    missing_int_msg = (
        f"Feature {feature_name!r} in {source} has format {fmt} and contains"
        " missing values; use FLOAT_32 or FLOAT_64 for nullable numeric"
        " features."
    )
    if isinstance(values, np.ndarray) and values.dtype.kind == "f":
      if np.isnan(values).any():
        raise ValueError(missing_int_msg)
      if np.isinf(values).any():
        raise ValueError(
            f"Feature {feature_name!r} in {source} has infinite values for"
            f" format {fmt}."
        )
    if isinstance(values, pd.api.extensions.ExtensionArray):
      if values.isna().any():
        raise ValueError(missing_int_msg)
      if pd.api.types.is_integer_dtype(values.dtype):
        return values.to_numpy(dtype=expected_dtype)
    arr = np.asarray(values, dtype=object)
    inferred = pd.api.types.infer_dtype(arr, skipna=True)
    if inferred not in (
        "integer",
        "floating",
        "integer-na",
        "mixed-integer-float",
        "empty",
    ):
      raise ValueError(
          f"Feature {feature_name!r} in {source} has non-numeric values"
          f" (inferred type {inferred!r}) for format {fmt}."
      )
    if np.any(pd.isna(arr)):
      raise ValueError(missing_int_msg)
    if arr.size == 0:
      return np.empty(arr.shape, dtype=expected_dtype)
    pd_int_dtype = (
        "Int32" if fmt == schema_lib.FeatureFormat.INTEGER_32 else "Int64"
    )
    try:
      return (
          pd.array(arr.ravel(), dtype=pd_int_dtype)
          .to_numpy(dtype=expected_dtype)
          .reshape(arr.shape)
      )
    except (ValueError, TypeError, OverflowError) as exc:
      raise ValueError(
          f"Feature {feature_name!r} in {source} cannot convert values to"
          f" {fmt}: {exc}"
      ) from exc
  elif fmt == schema_lib.FeatureFormat.BOOL:
    if isinstance(
        values, pd.api.extensions.ExtensionArray
    ) and pd.api.types.is_bool_dtype(values.dtype):
      if values.isna().any():
        raise ValueError(
            f"Feature {feature_name!r} in {source} has missing values."
        )
      return values.to_numpy(dtype=np.bool_)
    if isinstance(values, np.ndarray) and values.dtype.kind in ("i", "u"):
      if not ((values == 0) | (values == 1)).all():
        raise ValueError(
            f"Feature {feature_name!r} in {source} has non-boolean integer"
            f" values for format {fmt}."
        )
      return values.astype(np.bool_)
    arr = np.asarray(values, dtype=object)

    def to_bool(x: Any) -> bool:
      if _is_missing_scalar(x):
        raise ValueError(
            f"Feature {feature_name!r} in {source} has missing values."
        )
      if isinstance(x, (bool, np.bool_)):
        return bool(x)
      if isinstance(x, (int, np.integer)) and x in (0, 1):
        return bool(x)
      raise ValueError(
          f"Feature {feature_name!r} in {source} has non-boolean value"
          f" {x!r} for format {fmt}."
      )

    return np.array([to_bool(x) for x in arr.ravel()], dtype=np.bool_).reshape(
        arr.shape
    )
  elif fmt == schema_lib.FeatureFormat.BYTES:
    if isinstance(values, np.ndarray) and values.dtype.type == np.str_:
      return np.char.encode(values, "utf-8")
    if isinstance(values, pd.api.extensions.ExtensionArray):
      values = values.to_numpy(dtype=object, na_value=missing_bytes)
    arr = np.asarray(values, dtype=object)
    if arr.size == 0:
      return np.empty(arr.shape, dtype=np.bytes_)

    def to_bytes(x: Any) -> bytes:
      if _is_missing_scalar(x):
        return missing_bytes
      if isinstance(x, (bytes, np.bytes_)):
        return bytes(x)
      if isinstance(x, (str, np.str_)):
        return x.encode("utf-8")
      raise ValueError(
          f"Feature {feature_name!r} in {source} has non-string/bytes value"
          f" {x!r} for format {fmt}."
      )

    return np.array(
        [to_bytes(x) for x in arr.ravel()], dtype=np.bytes_
    ).reshape(arr.shape)
  else:
    raise ValueError(
        f"Unsupported feature format {fmt} for feature {feature_name!r} in"
        f" {source}."
    )


def _conform_feature(
    values: Any,
    feature_schema: schema_lib.FeatureSchema,
    *,
    source: str,
    feature_name: str,
    missing_bytes: bytes,
) -> np.ndarray:
  """Conforms a single static or variable-length feature to `feature_schema`."""
  expected_dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[
      feature_schema.format
  ]
  expected_shape = feature_schema.shape or ()
  is_varlen = len(expected_shape) >= 1 and expected_shape[0] is None
  if not is_varlen:
    return _conform_array(
        values,
        feature_schema.format,
        expected_dtype,
        source=source,
        feature_name=feature_name,
        missing_bytes=missing_bytes,
    )
  if (
      isinstance(values, np.ndarray)
      and values.dtype == np.object_
      and values.ndim == 1
      and all(
          isinstance(row, np.ndarray)
          and feature_format.np_dtype_matches(row.dtype, expected_dtype)
          for row in values
      )
  ):
    return values
  rows = values.tolist() if isinstance(values, pd.Series) else list(values)
  # We use np.empty(..., dtype=object) instead of np.array(list) to avoid
  # numpy collapsing equal-length rows into a 2-D object array.
  out = np.empty(len(rows), dtype=object)
  for idx, row in enumerate(rows):
    if _is_missing_scalar(row):
      raise ValueError(
          f"Feature {feature_name!r} in {source} has missing row at index"
          f" {idx}."
      )
    out[idx] = _conform_array(
        row,
        feature_schema.format,
        expected_dtype,
        source=source,
        feature_name=feature_name,
        missing_bytes=missing_bytes,
    )
  return out


def conform_graph(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    *,
    missing_bytes: bytes = b"",
) -> in_memory_graph.InMemoryGraph:
  """Returns a copy of `graph` with features conformed to `schema`.

  Converts each feature defined in `schema` to its target NumPy dtype while
  leaving already well-formed arrays unchanged without copying:

  * Converts missing float values (`None`, `pd.NA`, `NaN`) to `np.nan`.
  * Encodes strings as UTF-8 bytes and replaces missing string/bytes values
    with `missing_bytes`.
  * Converts variable-length features into 1-D `object` arrays of typed row
    `np.ndarray`s.
  * Rejects missing or infinite integer values, missing boolean values, and
    values incompatible with the target format.

  Usage example:

  ```python
  graph = dgf.validate.conform_graph(graph, schema)
  dgf.validate.validate_graph(graph, schema)
  ```

  Args:
    graph: The in-memory graph to conform.
    schema: The schema defining expected feature formats and shapes.
    missing_bytes: Replacement bytes for missing values in `BYTES` features.

  Returns:
    A new `InMemoryGraph` whose schema-declared features match `schema`.

  Raises:
    ValueError: If a feature contains disallowed missing, infinite, or
      incompatible values for its schema format.
  """
  new_node_sets = {}
  new_edge_sets = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    nodeset_data = graph.node_sets[nodeset_name]
    new_features = dict(nodeset_data.features)
    for feature_name, feature_schema in nodeset_schema.features.items():
      new_features[feature_name] = _conform_feature(
          new_features[feature_name],
          feature_schema,
          source=f"nodeset {nodeset_name!r}",
          feature_name=feature_name,
          missing_bytes=missing_bytes,
      )
    new_node_sets[nodeset_name] = in_memory_graph.InMemoryNodeSet(
        num_nodes=nodeset_data.num_nodes, features=new_features
    )
  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    edgeset_data = graph.edge_sets[edgeset_name]
    new_features = dict(edgeset_data.features)
    for feature_name, feature_schema in edgeset_schema.features.items():
      new_features[feature_name] = _conform_feature(
          new_features[feature_name],
          feature_schema,
          source=f"edgeset {edgeset_name!r}",
          feature_name=feature_name,
          missing_bytes=missing_bytes,
      )
    new_edge_sets[edgeset_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=edgeset_data.adjacency, features=new_features
    )
  return in_memory_graph.InMemoryGraph(
      node_sets=new_node_sets,
      edge_sets=new_edge_sets,
      timestamp=graph.timestamp,
  )
