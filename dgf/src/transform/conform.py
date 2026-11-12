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

import dataclasses
from typing import Any

from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
from dgf.src.util import log
import numpy as np
import pandas as pd

# `pd.api.types.infer_dtype` results (with missing values skipped) that we
# accept as numeric input for numeric formats.
_NUMERIC_INFERRED_TYPES = (
    "floating",
    "integer",
    "integer-na",
    "mixed-integer-float",
    "empty",
)


@dataclasses.dataclass(frozen=True)
class _Options:
  """Options shared by the conform helpers."""

  replace_none_bytes: bytes | None
  verbose: bool


def _is_missing_scalar(val: Any) -> bool:
  """Returns True if a scalar element represents a missing value."""
  if val is None or val is pd.NA or val is pd.NaT:
    return True
  if isinstance(val, (float, np.floating)):
    return bool(np.isnan(val))
  return False


def _to_numpy(values: Any) -> Any:
  """Converts pandas inputs to numpy; other inputs are returned unchanged."""
  if isinstance(values, pd.Series):
    # The user provided a pandas Series instead of a numpy array.
    values = values.array
  if isinstance(values, pd.api.extensions.ExtensionArray):
    # Nullable pandas arrays (e.g. Int64, boolean, string) hold pd.NA, which
    # numpy cannot store natively; use an object array with None instead.
    return values.to_numpy(dtype=object, na_value=None)
  return values


def _numeric_object_array(
    values: Any, fmt: schema_lib.FeatureFormat, name: str
) -> np.ndarray:
  """Returns `values` as an object array, if all non-missing are numbers."""
  arr = np.asarray(values, dtype=object)
  inferred = pd.api.types.infer_dtype(arr, skipna=True)
  if inferred not in _NUMERIC_INFERRED_TYPES:
    raise ValueError(
        f"{name} has non-numeric values (inferred type {inferred!r}) for"
        f" format {fmt}."
    )
  return arr


def _conform_float(
    values: Any, fmt: schema_lib.FeatureFormat, name: str
) -> np.ndarray:
  """Conforms to FLOAT_32/FLOAT_64; missing values become NaN."""
  expected_dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[fmt]
  if isinstance(values, np.ndarray) and values.dtype.kind in ("f", "i", "u"):
    return values.astype(expected_dtype)
  arr = _numeric_object_array(values, fmt, name)
  # Missing values (None, pd.NA, NaN) become NaN; ±inf is kept.
  return np.where(pd.isna(arr), np.nan, arr).astype(expected_dtype)


def _conform_integer(
    values: Any, fmt: schema_lib.FeatureFormat, name: str
) -> np.ndarray:
  """Conforms to an integer format; missing values raise."""
  expected_dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[fmt]
  info = np.iinfo(expected_dtype)
  out_of_range_msg = (
      f"{name} has values outside the range [{info.min}, {info.max}] of"
      f" format {fmt}."
  )
  if isinstance(values, np.ndarray) and values.dtype.kind in ("i", "u"):
    # `astype` would silently wrap around out-of-range values.
    if values.size and (values.min() < info.min or values.max() > info.max):
      raise ValueError(out_of_range_msg)
    return values.astype(expected_dtype)
  arr = _numeric_object_array(values, fmt, name)
  if np.any(pd.isna(arr)):
    raise ValueError(
        f"{name} has format {fmt} and contains missing values; use FLOAT_32"
        " or FLOAT_64 for nullable numeric features."
    )
  as_float = arr.astype(np.float64)
  if not np.isfinite(as_float).all() or (as_float != np.round(as_float)).any():
    raise ValueError(f"{name} has non-integer values for format {fmt}.")
  # Exact comparison on Python numbers (no float64 rounding).
  if any(x < info.min or x > info.max for x in arr.flat):
    raise ValueError(out_of_range_msg)
  # Convert via the target dtype, so any integer width is supported.
  return arr.astype(expected_dtype)


def _conform_bool(
    values: Any, fmt: schema_lib.FeatureFormat, name: str
) -> np.ndarray:
  """Conforms to BOOL; accepts bools and 0/1 integers, missing values raise."""
  if isinstance(values, np.ndarray) and values.dtype.kind in ("i", "u"):
    if not ((values == 0) | (values == 1)).all():
      raise ValueError(
          f"{name} has non-boolean integer values for format {fmt}."
      )
    return values.astype(np.bool_)

  def to_bool(x: Any) -> bool:
    if _is_missing_scalar(x):
      raise ValueError(f"{name} has missing values for format {fmt}.")
    if isinstance(x, (bool, np.bool_)) or (
        isinstance(x, (int, np.integer)) and x in (0, 1)
    ):
      return bool(x)
    raise ValueError(f"{name} has non-boolean value {x!r} for format {fmt}.")

  arr = np.asarray(values, dtype=object)
  return np.array([to_bool(x) for x in arr.ravel()], dtype=np.bool_).reshape(
      arr.shape
  )


def _conform_bytes(
    values: Any, fmt: schema_lib.FeatureFormat, name: str, options: _Options
) -> np.ndarray:
  """Conforms to BYTES; strings are UTF-8 encoded."""
  if isinstance(values, np.ndarray) and values.dtype.kind == "U":
    return np.char.encode(values, "utf-8")

  def to_bytes(x: Any) -> bytes:
    if _is_missing_scalar(x):
      if options.replace_none_bytes is None:
        raise ValueError(
            f"{name} has missing values for format {fmt}. Set"
            " `replace_none_bytes` to replace them."
        )
      return options.replace_none_bytes
    if isinstance(x, (bytes, np.bytes_)):
      return bytes(x)
    if isinstance(x, (str, np.str_)):
      return x.encode("utf-8")
    raise ValueError(
        f"{name} has non-string/bytes value {x!r} for format {fmt}."
    )

  arr = np.asarray(values, dtype=object)
  # `dtype=np.bytes_` keeps empty inputs typed as bytes.
  return np.array([to_bytes(x) for x in arr.ravel()], dtype=np.bytes_).reshape(
      arr.shape
  )


def _conform_array(
    values: Any,
    fmt: schema_lib.FeatureFormat,
    name: str,
    options: _Options,
) -> np.ndarray:
  """Conforms a static feature array or a single variable-length row."""
  values = _to_numpy(values)
  expected_dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[fmt]
  if isinstance(values, np.ndarray) and feature_format.np_dtype_matches(
      values.dtype, expected_dtype
  ):
    # Already well-formed: return without copying.
    return values
  if fmt in (
      schema_lib.FeatureFormat.FLOAT_32,
      schema_lib.FeatureFormat.FLOAT_64,
  ):
    return _conform_float(values, fmt, name)
  if np.issubdtype(expected_dtype, np.integer):
    return _conform_integer(values, fmt, name)
  if fmt == schema_lib.FeatureFormat.BOOL:
    return _conform_bool(values, fmt, name)
  if fmt == schema_lib.FeatureFormat.BYTES:
    return _conform_bytes(values, fmt, name, options)
  raise ValueError(f"Unsupported feature format {fmt} for {name}.")


def _conform_feature(
    values: Any,
    feature_schema: schema_lib.FeatureSchema,
    name: str,
    options: _Options,
) -> np.ndarray:
  """Conforms a single static or variable-length feature to `feature_schema`."""
  fmt = feature_schema.format
  # Same rule as `validate_graph`: only fully static shapes are dense arrays.
  if feature_schema.is_static_shape():
    out = _conform_array(values, fmt, name, options)
  else:
    expected_dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[fmt]
    if (
        isinstance(values, np.ndarray)
        and values.dtype == np.object_
        and all(
            isinstance(row, np.ndarray)
            and row.ndim >= 1
            and feature_format.np_dtype_matches(row.dtype, expected_dtype)
            for row in values.flat
        )
    ):
      # Already well-formed: return without copying.
      return values
    if (feature_schema.shape or ())[0] is not None:
      raise ValueError(
          f"{name} has a dynamic dimension that is not the first one; it must"
          " already be an object array of typed 1-D rows."
      )
    rows = values.tolist() if isinstance(values, pd.Series) else list(values)
    # np.empty(..., dtype=object) instead of np.array(rows), so that numpy
    # does not collapse equal-length rows into a 2-D array.
    out = np.empty(len(rows), dtype=object)
    for idx, row in enumerate(rows):
      if _is_missing_scalar(row):
        raise ValueError(f"{name} has a missing row at index {idx}.")
      # A scalar (incl. str/bytes) would become a 0-D row without a length.
      if isinstance(row, (str, bytes)) or np.ndim(row) == 0:
        raise ValueError(
            f"{name} has a scalar row {row!r} at index {idx}; each row of a"
            " variable-length feature must be a sequence."
        )
      out[idx] = _conform_array(row, fmt, name, options)
  if options.verbose and out is not values:
    log.info(f"conform_graph: converted {name} to format {fmt}.")
  return out


def _conform_features(
    features: in_memory_graph.Features,
    features_schema: dict[str, schema_lib.FeatureSchema],
    source: str,
    options: _Options,
) -> in_memory_graph.Features:
  """Conforms the schema features; other features are kept as they are."""
  return features | {
      feature_name: _conform_feature(
          features[feature_name],
          feature_schema,
          f"Feature {feature_name!r} in {source}",
          options,
      )
      for feature_name, feature_schema in features_schema.items()
  }


def conform_graph(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    *,
    replace_none_bytes: bytes | None = None,
    verbose: bool = True,
) -> in_memory_graph.InMemoryGraph:
  """Tries to fix invalid data in a graph. Returns the fixed graph.

  This function is designed for development only and should not be used in
  production. Its behavior is not guaranteed to be stable or backward
  compatible: the heuristics used to fix the graph might change.

  Already well-formed arrays are returned without copying. Other features are
  converted to their schema format:

  * Missing float values (`None`, `pd.NA`, `NaN`) become `np.nan`; ±inf is
    kept.
  * Strings are encoded as UTF-8 bytes.
  * Variable-length features become 1-D `object` arrays of typed rows.
  * Missing integer or boolean values, and values incompatible with the
    format, raise an error.

  Usage example:

  ```python
  graph = dgf.transform.conform_graph(graph, schema)
  dgf.validate.validate_graph(graph, schema)
  ```

  Args:
    graph: The in-memory graph to fix.
    schema: The schema defining expected feature formats and shapes.
    replace_none_bytes: Missing values (`None`, `pd.NA`, `NaN`) in BYTES
      features are replaced with this value. If None, a BYTES feature with a
      missing value raises an error.
    verbose: If True, logs each converted feature.

  Returns:
    A new `InMemoryGraph` whose schema-declared features match `schema`.
  """
  options = _Options(replace_none_bytes=replace_none_bytes, verbose=verbose)
  # Node/edge sets and features that are not in the schema are kept as is.
  return in_memory_graph.InMemoryGraph(
      node_sets=graph.node_sets | {
          name: in_memory_graph.InMemoryNodeSet(
              num_nodes=graph.node_sets[name].num_nodes,
              features=_conform_features(
                  graph.node_sets[name].features,
                  nodeset_schema.features,
                  f"nodeset {name!r}",
                  options,
              ),
          )
          for name, nodeset_schema in schema.node_sets.items()
      },
      edge_sets=graph.edge_sets | {
          name: in_memory_graph.InMemoryEdgeSet(
              adjacency=graph.edge_sets[name].adjacency,
              features=_conform_features(
                  graph.edge_sets[name].features,
                  edgeset_schema.features,
                  f"edgeset {name!r}",
                  options,
              ),
          )
          for name, edgeset_schema in schema.edge_sets.items()
      },
      timestamp=graph.timestamp,
  )
