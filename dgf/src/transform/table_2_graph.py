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

"""Utility to convert tables to In-Memory Graphs and their schemas."""

from typing import Any

from dgf.src.analyse import text_detection
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
from dgf.src.transform import conform as conform_lib
import numpy as np
import pandas as pd


# `pd.api.types.infer_dtype` results (missing values skipped) of object
# columns that hold only numbers and missing values.
_NULLABLE_NUMERIC_INFERRED_TYPES = (
    "integer",
    "floating",
    "mixed-integer-float",
)


def _infer_feature_schema(
    name: str, arr: np.ndarray, detect_semantic: bool = True
) -> schema_lib.FeatureSchema:
  """Infers the FeatureSchema from a numpy array."""
  if (
      arr.dtype == np.object_
      and pd.api.types.infer_dtype(arr, skipna=True)
      in _NULLABLE_NUMERIC_INFERRED_TYPES
  ):
    # Numbers with missing values: integers cannot hold NaN, so use a float
    # format. FLOAT_64 keeps integers exact up to 2**53 (FLOAT_32 only up to
    # 2**24, which would corrupt e.g. IDs, counts or timestamps).
    fmt = schema_lib.FeatureFormat.FLOAT_64
  else:
    fmt = feature_format.NP_DTYPE_TO_FEATURE_FORMAT.get(
        arr.dtype.type, schema_lib.FeatureFormat.BYTES
    )
  shape = arr.shape[1:] if arr.ndim > 1 else ()

  semantic = schema_lib.FeatureSemantic.UNKNOWN
  is_utf8_string = False
  if detect_semantic:
    if name.lower() in ("id", "#id"):
      semantic = schema_lib.FeatureSemantic.PRIMARY_ID
    elif fmt.is_numerical():
      size = int(np.prod(shape)) if shape else 1
      if size > 1:
        semantic = schema_lib.FeatureSemantic.EMBEDDING
      else:
        semantic = schema_lib.FeatureSemantic.NUMERICAL
    elif fmt == schema_lib.FeatureFormat.BYTES:
      if text_detection.looks_like_text(arr):
        semantic = schema_lib.FeatureSemantic.TEXT
        is_utf8_string = True
      else:
        semantic = schema_lib.FeatureSemantic.CATEGORICAL

  return schema_lib.FeatureSchema(
      format=fmt,
      shape=shape,
      semantic=semantic,
      is_utf8_string=is_utf8_string,
  )


def _column_to_numpy(values: Any) -> np.ndarray:
  """Converts a table column to a numpy array, without choosing a format."""
  if isinstance(values, pd.Series):
    # Nullable pandas columns (e.g. Int64, Float64, string) hold pd.NA, which
    # numpy cannot store; they become object arrays with None.
    if values.hasnans and isinstance(
        values.dtype, pd.api.extensions.ExtensionDtype
    ):
      return values.to_numpy(dtype=object, na_value=None)
    return values.to_numpy()
  return np.asarray(values)


def table2graph(
    table: dict[str, np.ndarray] | pd.DataFrame,
    nodeset_name: str = "nodes",
    detect_semantic: bool = True,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Converts a table (dict of arrays or DataFrame) into an InMemoryGraph and Schema.

  The resulting graph will have no edgeset and a single nodeset.

  Usage example:

  ```python
    import numpy as np
    table = {
        "feature_a": np.array([1, 2, 3]),
        "feature_b": np.array([4, 5, 6]),
    }
    graph, schema = dgf.transform.table2graph(table, nodeset_name="my_nodes")
  ```

  Args:
    table: A dictionary of numpy arrays or a pandas DataFrame.
    nodeset_name: The name of the single nodeset in the output graph.
    detect_semantic: Whether to automatically infer the semantic of the
      features.

  Missing values: numeric columns with missing values (None, pd.NA, NaN) get
  the FLOAT_64 format and NaN for missing. Missing values in string/bytes
  columns are replaced with `b""`.

  Returns:
    A tuple of (InMemoryGraph, GraphSchema).

  Raises:
    TypeError: If the input table is not a dict or pandas DataFrame.
    ValueError: If the input table is empty, or if the arrays/columns
      have different lengths.
  """
  if not isinstance(table, (dict, pd.DataFrame)):
    raise TypeError(
        "table must be a dict of numpy arrays or a pandas DataFrame"
    )

  if isinstance(table, pd.DataFrame):
    features = {col: _column_to_numpy(table[col]) for col in table.columns}
  else:
    features = {k: _column_to_numpy(v) for k, v in table.items()}

  if not features:
    raise ValueError("Input table is empty")

  # Validate lengths
  lengths = {k: len(v) for k, v in features.items()}
  unique_lengths = set(lengths.values())

  if len(unique_lengths) > 1:
    raise ValueError(
        f"All columns must have the same length. Found lengths: {lengths}"
    )

  num_nodes = next(iter(unique_lengths))

  # Create Graph
  nodeset = in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=num_nodes, features=features
  )
  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={nodeset_name: nodeset},
      edge_sets={},
  )

  # Create Schema
  node_features_schema = {
      k: _infer_feature_schema(k, v, detect_semantic=detect_semantic)
      for k, v in features.items()
  }
  nodeset_schema = schema_lib.NodeSchema(features=node_features_schema)
  schema = schema_lib.GraphSchema(
      node_sets={nodeset_name: nodeset_schema},
      edge_sets={},
  )

  # Convert the columns to the inferred formats (e.g. missing -> NaN).
  graph = conform_lib.conform_graph(
      graph, schema, replace_none_bytes=b"", verbose=False
  )
  return graph, schema
