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

"""Client-independent core for reading Spanner Property Graphs into memory."""

import base64
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import datetime
import itertools
import os
import re
import time
from typing import Any, TypeVar

from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.analyse import schema as schema_analyse_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import io_ext
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.util import log
from dgf.src.util import util as util_lib
import numpy as np
import tqdm

# Matches the "(vector_length=>N)" suffix of the type of a fixed-size array.
_VECTOR_LENGTH_REGEX = re.compile(
    r"\(\s*vector_length\s*=>\s*(\d+)\s*\)\s*$", re.IGNORECASE
)
# Origin of the TIMESTAMP values, read as microseconds since the epoch.
_UTC_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
# Primary key feature created for the node tables with a composite key.
_COMPOSITE_PRIMARY_KEY_NAME = "#id"
# Numpy dtype of the BYTES arrays without any non-empty value.
_EMPTY_BYTES_DTYPE = np.dtype("|S1")

# A node table or an edge table of the definition of a property graph.
_ElementTableT = TypeVar(
    "_ElementTableT",
    spanner_graph_metadata_lib.NodeTable,
    spanner_graph_metadata_lib.EdgeTable,
)

# Spanner scalar types, as returned by parse_spanner_type.
TYPE_BOOL = "BOOL"
TYPE_BYTES = "BYTES"
TYPE_DATE = "DATE"
TYPE_FLOAT32 = "FLOAT32"
TYPE_FLOAT64 = "FLOAT64"
TYPE_INT32 = "INT32"
TYPE_INT64 = "INT64"
TYPE_JSON = "JSON"
TYPE_NUMERIC = "NUMERIC"
TYPE_PROTO = "PROTO"
TYPE_STRING = "STRING"
TYPE_TIMESTAMP = "TIMESTAMP"

# Other names of Spanner scalar types (used by Borg Spanner).
_SCALAR_TYPE_ALIASES: dict[str, str] = {
    "FLOAT": TYPE_FLOAT32,
    "DOUBLE": TYPE_FLOAT64,
}

# Spanner types read as raw bytes (in base64 by some clients).
_RAW_BYTES_TYPES = frozenset({TYPE_BYTES, TYPE_PROTO})
# Spanner types of the columns that can be filtered on.
_SUPPORTED_FILTER_TYPES = frozenset(
    {TYPE_BOOL, TYPE_DATE, TYPE_INT64, TYPE_STRING, TYPE_TIMESTAMP}
)
# Spanner types of the key columns that can be encoded in a node id.
_SUPPORTED_KEY_TYPES = frozenset({
    TYPE_BOOL,
    TYPE_BYTES,
    TYPE_DATE,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_STRING,
    TYPE_TIMESTAMP,
})

# Spanner types that can be read into a feature of each format.
_COMPATIBLE_TYPES: dict[schema_lib.FeatureFormat, frozenset[str]] = {
    schema_lib.FeatureFormat.BOOL: frozenset({TYPE_BOOL}),
    schema_lib.FeatureFormat.BYTES: frozenset({
        TYPE_BYTES,
        TYPE_DATE,
        TYPE_JSON,
        TYPE_PROTO,
        TYPE_STRING,
    }),
    schema_lib.FeatureFormat.FLOAT_32: frozenset({TYPE_FLOAT32, TYPE_FLOAT64}),
    schema_lib.FeatureFormat.FLOAT_64: frozenset(
        {TYPE_FLOAT32, TYPE_FLOAT64, TYPE_NUMERIC}
    ),
    schema_lib.FeatureFormat.INTEGER_32: frozenset({TYPE_INT32, TYPE_INT64}),
    schema_lib.FeatureFormat.INTEGER_64: frozenset(
        {TYPE_INT32, TYPE_INT64, TYPE_TIMESTAMP}
    ),
}

# Query reading the Spanner type of all the columns of the database.
COLUMNS_TYPE_QUERY = (
    "SELECT c.TABLE_NAME, c.COLUMN_NAME, c.SPANNER_TYPE FROM"
    " INFORMATION_SCHEMA.COLUMNS AS c WHERE c.TABLE_SCHEMA = ''"
)


def property_graph_metadata_query(metadata_column: str) -> str:
  """Gets the query reading the definition of the property graph @graph_name."""
  gcp_common_lib.validate_identifier(metadata_column)
  return (
      f"SELECT g.`{metadata_column}` FROM INFORMATION_SCHEMA.PROPERTY_GRAPHS AS"
      " g WHERE g.PROPERTY_GRAPH_NAME = @graph_name"
  )


def single_property_graph_row(rows: Iterable[Sequence[Any]], graph: str) -> Any:
  """Gets the value of the only row of a property_graph_metadata_query."""
  rows = list(rows)
  if len(rows) != 1:
    raise ValueError(
        f"Expected one property graph named {graph!r}, found {len(rows)}."
    )
  return rows[0][0]


def column_types_from_rows(
    rows: Iterable[Sequence[Any]],
) -> dict[tuple[str, str], str]:
  """Gets the Spanner type of each (table, column) from COLUMNS_TYPE_QUERY."""
  return {
      (str(table_name), str(column_name)): str(spanner_type)
      for table_name, column_name, spanner_type in rows
  }


def _strip_scalar_type_params(raw_scalar: str) -> str:
  """Normalizes a scalar Spanner type string by stripping parameters."""
  cleaned = raw_scalar.strip().upper()
  if cleaned.startswith("PROTO"):
    return TYPE_PROTO
  paren_idx = cleaned.find("(")
  if paren_idx != -1:
    cleaned = cleaned[:paren_idx].strip()
  return _SCALAR_TYPE_ALIASES.get(cleaned, cleaned)


def parse_spanner_type(raw_type: str) -> tuple[str, bool, int | None]:
  """Parses a Spanner type into (scalar_type, is_array, vector_length)."""
  cleaned = raw_type.strip()
  vector_length: int | None = None
  vector_match = _VECTOR_LENGTH_REGEX.search(cleaned)
  if vector_match is not None:
    vector_length = int(vector_match.group(1))
    cleaned = cleaned[: vector_match.start()].strip()

  if cleaned.upper().startswith("ARRAY"):
    inner = cleaned[5:].strip()
    if len(inner) < 2 or not inner.startswith("<") or not inner.endswith(">"):
      raise ValueError(f"Invalid Spanner ARRAY type: {raw_type!r}.")
    element_raw = inner[1:-1].strip()
    if element_raw.upper().startswith("ARRAY"):
      raise ValueError(
          f"Nested Spanner ARRAY types are not supported: {raw_type!r}."
      )
    return _strip_scalar_type_params(element_raw), True, vector_length

  return _strip_scalar_type_params(cleaned), False, None


def _escape_composite_part(text: str) -> str:
  """Escapes backslashes and slashes in a single composite-key component."""
  return text.replace("\\", "\\\\").replace("/", "\\/")


def timestamp_to_micros(value: datetime.datetime | int | np.integer) -> int:
  """Converts a UTC datetime or integer microsecond value to int64 micros."""
  if isinstance(value, datetime.datetime):
    if value.tzinfo is None:
      raise ValueError(f"Datetime {value!r} must be timezone-aware.")
    return (value.astimezone(datetime.timezone.utc) - _UTC_EPOCH) // (
        datetime.timedelta(microseconds=1)
    )
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise ValueError(f"Expected datetime or int micros, got {value!r}.")
  return int(value)


def encode_composite_key(
    values: Sequence[Any], column_types: Sequence[str]
) -> bytes:
  """Encodes composite key values into escaped slash-delimited UTF-8 bytes."""
  if len(values) != len(column_types) or len(values) < 2:
    raise ValueError(
        f"Expected {len(column_types)} composite key values (>= 2), got"
        f" {len(values)}."
    )
  encoded_parts: list[str] = []
  for value, column_type in zip(values, column_types):
    if value is None:
      raise ValueError("Composite key component cannot be None.")
    if column_type in (TYPE_STRING, TYPE_DATE):
      text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
      encoded_parts.append(_escape_composite_part(text))
    elif column_type == TYPE_BYTES:
      if not isinstance(value, (bytes, bytearray, np.bytes_)):
        raise ValueError(
            f"Expected bytes for {column_type} key column, got {value!r}."
        )
      encoded_parts.append(bytes(value).hex())
    elif column_type == TYPE_TIMESTAMP:
      encoded_parts.append(str(timestamp_to_micros(value)))
    elif column_type in (TYPE_INT32, TYPE_INT64):
      if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"Expected int for {column_type} key column, got {value!r}."
        )
      encoded_parts.append(str(int(value)))
    elif column_type == TYPE_BOOL:
      if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Expected bool for BOOL key column, got {value!r}.")
      encoded_parts.append("true" if value else "false")
    else:
      raise ValueError(f"Unsupported composite key type: {column_type!r}.")
  return "/".join(encoded_parts).encode("utf-8")


def decode_composite_key(
    encoded: bytes | str, column_types: Sequence[str]
) -> tuple[Any, ...]:
  """Decodes escaped slash-delimited composite key bytes into typed values."""
  if len(column_types) < 2:
    raise ValueError(
        f"Expected at least 2 column types, got {len(column_types)}."
    )
  text = encoded.decode("utf-8") if isinstance(encoded, bytes) else encoded
  parts: list[str] = []
  current: list[str] = []
  idx = 0
  while idx < len(text):
    char = text[idx]
    if char == "\\":
      if idx + 1 >= len(text):
        raise ValueError(f"Trailing backslash in composite key {encoded!r}.")
      escaped = text[idx + 1]
      if escaped not in ("\\", "/"):
        raise ValueError(
            f"Invalid escape sequence '\\{escaped}' in composite key"
            f" {encoded!r}."
        )
      current.append(escaped)
      idx += 2
    elif char == "/":
      parts.append("".join(current))
      current = []
      idx += 1
    else:
      current.append(char)
      idx += 1
  parts.append("".join(current))

  if len(parts) != len(column_types):
    raise ValueError(
        f"Composite key {encoded!r} has {len(parts)} components, expected"
        f" {len(column_types)}."
    )

  return tuple(
      _parse_key_component(part, column_type, encoded)
      for part, column_type in zip(parts, column_types)
  )


def _parse_key_component(
    part: str, column_type: str, encoded: bytes | str
) -> Any:
  """Parses a component of an encoded key into a query parameter value."""
  if column_type in (TYPE_STRING, TYPE_DATE):
    return part
  if column_type == TYPE_BYTES:
    return bytes.fromhex(part)
  if column_type in (TYPE_INT32, TYPE_INT64, TYPE_TIMESTAMP):
    return int(part)
  if column_type == TYPE_BOOL:
    if part == "true":
      return True
    if part == "false":
      return False
    raise ValueError(f"Invalid BOOL component {part!r} in key {encoded!r}.")
  raise ValueError(f"Unsupported key type: {column_type!r}.")


def decode_key(encoded: bytes, column_types: Sequence[str]) -> tuple[Any, ...]:
  """Decodes a key encoded by key_bytes_sql into query parameter values."""
  if len(column_types) > 1:
    return decode_composite_key(encoded, column_types)
  if column_types[0] == TYPE_BYTES:
    return (encoded,)
  return (
      _parse_key_component(encoded.decode("utf-8"), column_types[0], encoded),
  )


@dataclasses.dataclass(frozen=True)
class Column:
  """Spanner base-table column backing a DGF feature or key."""

  name: str
  type_kind: str
  is_array: bool = False

  def __post_init__(self) -> None:
    gcp_common_lib.validate_identifier(self.name)

  def sql(self, table_alias: str) -> str:
    """Gets the backtick-quoted SQL reference to this column."""
    return f"{table_alias}.`{self.name}`"

  def to_json_sql(self, expression: str) -> str:
    """Gets the SQL expression converting expression to a JSON-friendly type."""
    if self.type_kind == TYPE_TIMESTAMP:
      scalar_sql = "UNIX_MICROS({0})"
    elif self.type_kind == TYPE_DATE:
      scalar_sql = "CAST({0} AS STRING)"
    elif self.type_kind == TYPE_JSON:
      # TO_JSON_STRING(NULL) is the string "null".
      scalar_sql = "IF({0} IS NULL, NULL, TO_JSON_STRING({0}))"
    elif self.type_kind == TYPE_NUMERIC:
      scalar_sql = "CAST({0} AS FLOAT64)"
    elif self.type_kind == TYPE_PROTO:
      scalar_sql = "CAST({0} AS BYTES)"
    else:
      return expression
    if not self.is_array:
      return scalar_sql.format(expression)
    return (
        f"IF({expression} IS NULL, NULL, ARRAY(SELECT {scalar_sql.format('t')}"
        f" FROM UNNEST({expression}) AS t WITH OFFSET off ORDER BY off))"
    )

  def composite_part_sql(self, expression: str) -> str:
    """Gets the SQL STRING expression for this column in a composite key."""
    if self.type_kind == TYPE_STRING:
      return f"REPLACE(REPLACE({expression}, '\\\\', '\\\\\\\\'), '/', '\\\\/')"
    if self.type_kind == TYPE_DATE:
      return (
          f"REPLACE(REPLACE(CAST({expression} AS STRING), '\\\\', '\\\\\\\\'),"
          " '/', '\\\\/')"
      )
    if self.type_kind == TYPE_BYTES:
      return f"TO_HEX({expression})"
    if self.type_kind == TYPE_TIMESTAMP:
      return f"CAST(UNIX_MICROS({expression}) AS STRING)"
    if self.type_kind in (TYPE_INT32, TYPE_INT64):
      return f"CAST({expression} AS STRING)"
    if self.type_kind == TYPE_BOOL:
      return f"IF({expression}, 'true', 'false')"
    raise ValueError(f"Unsupported key column type: {self.type_kind!r}.")

  def single_key_bytes_sql(self, expression: str) -> str:
    """Gets the SQL BYTES expression when this column is the sole key."""
    if self.type_kind == TYPE_BYTES:
      return expression
    if self.type_kind == TYPE_STRING:
      return f"CAST({expression} AS BYTES)"
    if self.type_kind == TYPE_TIMESTAMP:
      return f"CAST(CAST(UNIX_MICROS({expression}) AS STRING) AS BYTES)"
    if self.type_kind in (TYPE_INT32, TYPE_INT64, TYPE_DATE):
      return f"CAST(CAST({expression} AS STRING) AS BYTES)"
    if self.type_kind == TYPE_BOOL:
      return f"CAST(IF({expression}, 'true', 'false') AS BYTES)"
    raise ValueError(f"Unsupported key column type: {self.type_kind!r}.")


def key_bytes_sql(columns: Sequence[Column], expressions: Sequence[str]) -> str:
  """Builds the SQL BYTES expression for a single or composite key."""
  if len(columns) != len(expressions) or not columns:
    raise ValueError(
        f"Mismatched key columns ({len(columns)}) and expressions"
        f" ({len(expressions)})."
    )
  if len(columns) == 1:
    return columns[0].single_key_bytes_sql(expressions[0])
  parts = [
      col.composite_part_sql(expr) for col, expr in zip(columns, expressions)
  ]
  concat_args = ", '/', ".join(parts)
  return f"CAST(CONCAT({concat_args}) AS BYTES)"


def _property_column_name(
    element_name: str, property_name: str, expression: str
) -> str:
  """Gets the base-table column of a property defined by a plain column."""
  column_name = expression.strip()
  if not gcp_common_lib.IDENTIFIER_REGEX.match(column_name):
    raise NotImplementedError(
        f"Feature {property_name!r} of {element_name!r} is defined by the"
        f" expression {expression!r} in the Spanner Graph. Only properties"
        " defined by a plain column are supported."
    )
  return column_name


def _make_value_parser(
    element_name: str,
    feature_name: str,
    column: Column,
    dtype: Any,
    scalar_default: Any,
    is_ragged: bool,
    expected_len: int | None,
    *,
    bytes_are_base64: bool = False,
) -> Callable[[Any], Any]:
  """Builds the precomputed per-row parser for a feature column."""
  if column.type_kind in _RAW_BYTES_TYPES:
    if bytes_are_base64:
      parse_scalar = lambda v: b"" if v is None else base64.b64decode(v)
    else:
      parse_scalar = lambda v: b"" if v is None else bytes(v)
  elif column.type_kind in (TYPE_STRING, TYPE_DATE, TYPE_JSON):
    parse_scalar = lambda v: b"" if v is None else v.encode("utf-8")
  else:
    parse_scalar = None

  if not column.is_array:
    if parse_scalar is not None:
      return parse_scalar
    return lambda v: scalar_default if v is None else v

  if is_ragged:
    empty_dtype = _EMPTY_BYTES_DTYPE if dtype is np.bytes_ else dtype
    empty_element = np.empty(0, dtype=empty_dtype)

    def parse_ragged(value: Any) -> np.ndarray:
      if value is None:
        return empty_element
      if parse_scalar is not None:
        value = [parse_scalar(item) for item in value]
      elif None in value:
        value = [scalar_default if item is None else item for item in value]
      if not value:
        return empty_element
      packed = np.asarray(value, dtype=dtype)
      if dtype is np.bytes_ and packed.dtype.itemsize == 0:
        return packed.astype(_EMPTY_BYTES_DTYPE)
      return packed

    return parse_ragged

  assert expected_len is not None
  default_list = [scalar_default] * expected_len

  def parse_static_array(value: Any) -> list[Any]:
    if value is None:
      return default_list
    if len(value) != expected_len:
      raise ValueError(
          f"Feature {feature_name!r} of {element_name!r} has length"
          f" {len(value)}, expected {expected_len}."
      )
    if parse_scalar is not None:
      return [parse_scalar(item) for item in value]
    if None in value:
      return [scalar_default if item is None else item for item in value]
    return value

  return parse_static_array


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
  """Resolved Spanner column and precomputed array packer for a DGF feature."""

  column: Column
  dtype: Any
  is_ragged: bool
  empty_array: np.ndarray
  parse_value: Callable[[Any], Any]


def _pack_feature(feature: FeatureSpec, values: list[Any]) -> np.ndarray:
  """Packs parsed per-element values into a single numpy array."""
  if not values:
    return feature.empty_array
  if feature.is_ragged:
    stacked = np.empty(len(values), dtype=np.object_)
    for index, item in enumerate(values):
      stacked[index] = item
    return stacked
  packed = np.array(values, dtype=feature.dtype)
  if feature.dtype is np.bytes_ and packed.dtype.itemsize == 0:
    return packed.astype(_EMPTY_BYTES_DTYPE)
  return packed


def pack_bytes_ids(values: Sequence[bytes]) -> np.ndarray:
  """Packs byte IDs into a fixed-width numpy bytes array with itemsize >= 1."""
  if not values:
    return np.empty(0, dtype=_EMPTY_BYTES_DTYPE)
  packed = np.array(values, dtype=np.bytes_)
  if packed.dtype.itemsize == 0:
    return packed.astype(_EMPTY_BYTES_DTYPE)
  return packed


class FeatureColumns:
  """Accumulates the feature values of the elements of a nodeset or edgeset.

  The readers and the samplers both convert the values returned by Spanner with
  this class, so that they return the same feature values.
  """

  def __init__(self, features: Mapping[str, FeatureSpec]):
    self._features = features
    self._values: dict[str, list[Any]] = {name: [] for name in features}
    self._parsers = [
        (feature.parse_value, self._values[name].append)
        for name, feature in features.items()
    ]

  def append(self, values: Iterable[Any]) -> None:
    """Parses and adds the Spanner values of an element, one per feature."""
    for (parse_value, append), value in zip(self._parsers, values, strict=True):
      append(parse_value(value))

  def pack(self) -> dict[str, np.ndarray]:
    """Packs the values of each feature into a numpy array."""
    return {
        name: _pack_feature(feature, self._values[name])
        for name, feature in self._features.items()
    }


def _resolve_feature_spec(
    element_name: str,
    feature_name: str,
    feature_schema: schema_lib.FeatureSchema,
    expressions: Mapping[str, str],
    property_types: Mapping[str, str],
    *,
    bytes_are_base64: bool = False,
) -> FeatureSpec:
  """Resolves and validates the Spanner column and parser of a DGF feature."""
  gcp_common_lib.validate_identifier(feature_name)
  if feature_name not in expressions:
    raise ValueError(
        f"Feature {feature_name!r} of {element_name!r} is not a property of"
        f" the Spanner Graph. Available properties: {sorted(expressions)}"
    )
  column_name = _property_column_name(
      element_name, feature_name, expressions[feature_name]
  )
  type_kind, is_array, _ = parse_spanner_type(property_types[feature_name])

  shape = feature_schema.shape or ()
  if len(shape) > 1:
    raise NotImplementedError(
        f"Feature {feature_name!r} of {element_name!r} has shape {shape}."
        " Only scalar and 1D features are supported."
    )
  compatible_types = _COMPATIBLE_TYPES.get(feature_schema.format)
  if compatible_types is None:
    raise NotImplementedError(
        f"Unsupported format {feature_schema.format} on feature"
        f" {feature_name!r} of {element_name!r}."
    )
  if type_kind not in compatible_types or is_array != bool(shape):
    raise ValueError(
        f"Feature {feature_name!r} of {element_name!r} (format"
        f" {feature_schema.format}, shape {shape}) is not compatible with the"
        f" Spanner type {property_types[feature_name]}."
    )

  column = Column(name=column_name, type_kind=type_kind, is_array=is_array)
  dtype = feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[feature_schema.format]
  expected_len = shape[0] if shape else None
  is_ragged = bool(shape) and expected_len is None
  empty_dtype = _EMPTY_BYTES_DTYPE if dtype is np.bytes_ else dtype
  if is_ragged:
    empty_array = np.empty(0, dtype=np.object_)
  elif expected_len is not None:
    empty_array = np.empty((0, expected_len), dtype=empty_dtype)
  else:
    empty_array = np.empty(0, dtype=empty_dtype)
  scalar_default = feature_format_lib.default_scalar_value(
      feature_schema.format
  )
  return FeatureSpec(
      column=column,
      dtype=dtype,
      is_ragged=is_ragged,
      empty_array=empty_array,
      parse_value=_make_value_parser(
          element_name,
          feature_name,
          column,
          dtype,
          scalar_default,
          is_ragged,
          expected_len,
          bytes_are_base64=bytes_are_base64,
      ),
  )


def _property_expressions(
    element_table: (
        spanner_graph_metadata_lib.NodeTable
        | spanner_graph_metadata_lib.EdgeTable
    ),
) -> dict[str, str]:
  """Gets the {property_name: sql_expression} map of an element table."""
  return {
      definition.property_declaration_name: (
          definition.value_expression_sql.strip()
      )
      for definition in element_table.property_definitions or ()
  }


def find_element_table(
    element_name: str, element_tables: Sequence[_ElementTableT]
) -> tuple[_ElementTableT, str | None, bool]:
  """Finds the element table of a nodeset or edgeset.

  A nodeset or edgeset is named after a label defined on a single element
  table, or after an element table.

  Args:
    element_name: Name of the nodeset or edgeset.
    element_tables: The node tables or the edge tables of the property graph.

  Returns:
    The element table, its label (None if element_name is an element table
    with several labels), and whether other element tables have this label.
  """
  labeled_tables = [
      table for table in element_tables if element_name in table.label_names
  ]
  if len(labeled_tables) == 1:
    return labeled_tables[0], element_name, False
  named_tables = [
      table for table in element_tables if table.name == element_name
  ]
  if len(named_tables) == 1:
    table = named_tables[0]
    if len(table.label_names) != 1:
      return table, None, False
    label = table.label_names[0]
    num_tables_with_label = sum(
        label in other_table.label_names for other_table in element_tables
    )
    return table, label, num_tables_with_label > 1
  if labeled_tables:
    raise ValueError(
        f"The label {element_name!r} is defined on {len(labeled_tables)}"
        " element tables of the Spanner Graph:"
        f" {sorted(table.name for table in labeled_tables)}. Name the nodeset"
        " or edgeset after one of these element tables instead."
    )
  labels = set()
  for table in element_tables:
    labels.update(table.label_names)
  raise ValueError(
      f"{element_name!r} is neither a label nor an element table of the"
      f" Spanner Graph. Available labels: {sorted(labels)}. Available"
      f" element tables: {sorted(table.name for table in element_tables)}."
  )


def _resolve_key_columns(
    nodeset_name: str,
    node_table: spanner_graph_metadata_lib.NodeTable,
    property_types: Mapping[str, str],
    column_spanner_types: Mapping[tuple[str, str], str] | None = None,
) -> tuple[tuple[Column, ...], tuple[str | None, ...]]:
  """Resolves the key columns of a node table and their properties, if any."""
  if not node_table.key_columns:
    raise ValueError(f"Node table {nodeset_name!r} has no key columns.")

  column_to_property: dict[str, str] = {}
  for definition in node_table.property_definitions or ():
    expr = definition.value_expression_sql.strip()
    if (
        gcp_common_lib.IDENTIFIER_REGEX.match(expr)
        and expr not in column_to_property
    ):
      column_to_property[expr] = definition.property_declaration_name

  columns: list[Column] = []
  key_properties: list[str | None] = []
  for key_col in node_table.key_columns:
    gcp_common_lib.validate_identifier(key_col)
    prop_name = column_to_property.get(key_col)
    raw_type: str | None = None
    if prop_name is not None and prop_name in property_types:
      raw_type = property_types[prop_name]
    elif (
        column_spanner_types is not None
        and (node_table.base_table_name, key_col) in column_spanner_types
    ):
      raw_type = column_spanner_types[(node_table.base_table_name, key_col)]
    if raw_type is None:
      raise ValueError(
          f"Cannot determine Spanner type of key column {key_col!r} on table"
          f" {node_table.base_table_name!r} for nodeset {nodeset_name!r}."
      )
    type_kind, is_array, _ = parse_spanner_type(raw_type)
    if is_array or type_kind not in _SUPPORTED_KEY_TYPES:
      raise ValueError(
          f"Key column {key_col!r} of {nodeset_name!r} has unsupported key"
          f" type {raw_type!r}."
      )
    if prop_name is not None:
      gcp_common_lib.validate_identifier(prop_name)
    columns.append(Column(name=key_col, type_kind=type_kind))
    key_properties.append(prop_name)

  return tuple(columns), tuple(key_properties)


def _resolve_endpoint_columns(
    edgeset_name: str,
    expected_nodeset: str,
    reference: spanner_graph_metadata_lib.NodeTableRef,
    expected_node_table_name: str,
    target_keys: Sequence[Column],
) -> tuple[Column, ...]:
  """Resolves edge foreign-key columns aligned to target_keys order."""
  if reference.node_table_name != expected_node_table_name:
    raise ValueError(
        f"Edgeset {edgeset_name!r} references node table"
        f" {reference.node_table_name!r} in the Spanner Graph, expected"
        f" {expected_node_table_name!r} for nodeset {expected_nodeset!r}."
    )
  if len(reference.edge_table_columns) != len(reference.node_table_columns):
    raise ValueError(
        f"Edgeset {edgeset_name!r} has mismatched endpoint column counts:"
        f" {reference.edge_table_columns} vs {reference.node_table_columns}."
    )
  expected_names = {col.name for col in target_keys}
  if set(reference.node_table_columns) != expected_names or len(
      reference.node_table_columns
  ) != len(target_keys):
    raise ValueError(
        f"Edgeset {edgeset_name!r} references columns"
        f" {reference.node_table_columns} of {expected_nodeset!r} instead of"
        f" its key {[col.name for col in target_keys]}."
    )
  edge_col_by_node_col = dict(
      zip(reference.node_table_columns, reference.edge_table_columns)
  )
  return tuple(
      Column(
          name=edge_col_by_node_col[key_col.name], type_kind=key_col.type_kind
      )
      for key_col in target_keys
  )


@dataclasses.dataclass(frozen=True)
class NodeTable:
  """Spanner table, key, and features of a nodeset."""

  name: str
  table: str
  # Label of the nodeset in GQL. None if its element table has several labels.
  label: str | None
  # Whether other node tables have this label.
  label_is_shared: bool
  keys: tuple[Column, ...]
  # Property exposing each key column, if any.
  key_properties: tuple[str | None, ...]
  # Feature filled with the encoded key (not read from a property), if any.
  primary_key: str | None
  features: dict[str, FeatureSpec]

  def __post_init__(self) -> None:
    gcp_common_lib.validate_identifier(self.table)
    if self.label is not None:
      gcp_common_lib.validate_identifier(self.label)
    for key_property in self.key_properties:
      if key_property is not None:
        gcp_common_lib.validate_identifier(key_property)
    if self.primary_key is not None:
      gcp_common_lib.validate_identifier(self.primary_key.removeprefix("#"))
    for feature_name in self.features:
      gcp_common_lib.validate_identifier(feature_name)


@dataclasses.dataclass(frozen=True)
class EdgeTable:
  """Spanner table, endpoint columns, and features of an edgeset."""

  table: str
  # Label of the edgeset in GQL. None if its element table has several labels.
  label: str | None
  # Whether other edge tables have this label.
  label_is_shared: bool
  source: tuple[Column, ...]
  target: tuple[Column, ...]
  features: dict[str, FeatureSpec]

  def __post_init__(self) -> None:
    gcp_common_lib.validate_identifier(self.table)
    if self.label is not None:
      gcp_common_lib.validate_identifier(self.label)
    for feature_name in self.features:
      gcp_common_lib.validate_identifier(feature_name)


@dataclasses.dataclass(frozen=True)
class GraphTables:
  """Spanner tables of all the nodesets and edgesets of a DGF schema."""

  node_tables: dict[str, NodeTable]
  edge_tables: dict[str, EdgeTable]


def _encoded_primary_key(
    nodeset_name: str,
    nodeset_schema: schema_lib.NodeSchema,
    keys: Sequence[Column],
    expressions: Mapping[str, str],
) -> str | None:
  """Gets the primary key feature to fill with the encoded key, if any."""
  primary_key = schema_analyse_lib.primary_feature_or_none(
      nodeset_name, nodeset_schema
  )
  if primary_key is None:
    return None
  if len(keys) > 1 and primary_key in expressions:
    raise ValueError(
        f"Nodeset {nodeset_name!r} has a composite key"
        f" {[key.name for key in keys]}, so its primary key feature"
        f" {primary_key!r} cannot be a property of the Spanner Graph."
    )
  if (
      nodeset_schema.features[primary_key].format
      != schema_lib.FeatureFormat.BYTES
  ):
    return None
  if primary_key in expressions and expressions[primary_key] != keys[0].name:
    return None
  return primary_key


def resolve_graph_tables(
    metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
    schema: schema_lib.GraphSchema,
    column_spanner_types: Mapping[tuple[str, str], str] | None = None,
    *,
    bytes_are_base64: bool = False,
) -> GraphTables:
  """Resolves the Spanner tables and columns of all the schema elements.

  Args:
    metadata: The definition of the property graph.
    schema: The DGF schema.
    column_spanner_types: Spanner type of each (table_name, column_name), used
      for the key columns that are not properties.
    bytes_are_base64: Whether the BYTES feature values are read in base64.

  Returns:
    The Spanner tables of the nodesets and edgesets of the schema.
  """
  property_types = metadata.property_types or {}
  node_tables: dict[str, NodeTable] = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    gcp_common_lib.validate_identifier(nodeset_name)
    node_table, label, label_is_shared = find_element_table(
        nodeset_name, metadata.node_tables
    )
    keys, key_properties = _resolve_key_columns(
        nodeset_name, node_table, property_types, column_spanner_types
    )
    expressions = _property_expressions(node_table)
    primary_key = _encoded_primary_key(
        nodeset_name, nodeset_schema, keys, expressions
    )
    features = {
        feature_name: _resolve_feature_spec(
            nodeset_name,
            feature_name,
            feature_schema,
            expressions,
            property_types,
            bytes_are_base64=bytes_are_base64,
        )
        for feature_name, feature_schema in nodeset_schema.features.items()
        if feature_name != primary_key
    }
    node_tables[nodeset_name] = NodeTable(
        name=node_table.name,
        table=node_table.base_table_name,
        label=label,
        label_is_shared=label_is_shared,
        keys=keys,
        key_properties=key_properties,
        primary_key=primary_key,
        features=features,
    )

  edge_tables: dict[str, EdgeTable] = {}
  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    gcp_common_lib.validate_identifier(edgeset_name)
    edge_table, label, label_is_shared = find_element_table(
        edgeset_name, metadata.edge_tables
    )
    for nodeset_name in (edgeset_schema.source, edgeset_schema.target):
      if nodeset_name not in node_tables:
        raise ValueError(
            f"Edgeset {edgeset_name!r} references unknown nodeset"
            f" {nodeset_name!r}."
        )
    expressions = _property_expressions(edge_table)
    features = {
        feature_name: _resolve_feature_spec(
            edgeset_name,
            feature_name,
            feature_schema,
            expressions,
            property_types,
            bytes_are_base64=bytes_are_base64,
        )
        for feature_name, feature_schema in edgeset_schema.features.items()
    }
    edge_tables[edgeset_name] = EdgeTable(
        table=edge_table.base_table_name,
        label=label,
        label_is_shared=label_is_shared,
        source=_resolve_endpoint_columns(
            edgeset_name,
            edgeset_schema.source,
            edge_table.source_node_table,
            node_tables[edgeset_schema.source].name,
            node_tables[edgeset_schema.source].keys,
        ),
        target=_resolve_endpoint_columns(
            edgeset_name,
            edgeset_schema.target,
            edge_table.destination_node_table,
            node_tables[edgeset_schema.target].name,
            node_tables[edgeset_schema.target].keys,
        ),
        features=features,
    )
  return GraphTables(node_tables=node_tables, edge_tables=edge_tables)


def graph_schema_from_metadata(
    metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
    column_spanner_types: Mapping[tuple[str, str], str] | None = None,
) -> schema_lib.GraphSchema:
  """Infers a DGF GraphSchema from Spanner property graph metadata."""
  property_types = metadata.property_types or {}
  node_sets: dict[str, schema_lib.NodeSchema] = {}
  for node_table in metadata.node_tables:
    node_sets[node_table.name] = schema_lib.NodeSchema(
        features=_infer_element_features(
            element_table=node_table,
            property_types=property_types,
            is_node=True,
            column_spanner_types=column_spanner_types,
        )
    )

  edge_sets: dict[str, schema_lib.EdgeSchema] = {}
  for edge_table in metadata.edge_tables:
    edge_sets[edge_table.name] = schema_lib.EdgeSchema(
        source=edge_table.source_node_table.node_table_name,
        target=edge_table.destination_node_table.node_table_name,
        features=_infer_element_features(
            element_table=edge_table,
            property_types=property_types,
            is_node=False,
            column_spanner_types=column_spanner_types,
        ),
    )

  return schema_lib.GraphSchema(node_sets=node_sets, edge_sets=edge_sets)


def _infer_element_features(
    element_table: (
        spanner_graph_metadata_lib.NodeTable
        | spanner_graph_metadata_lib.EdgeTable
    ),
    property_types: Mapping[str, str],
    is_node: bool,
    column_spanner_types: Mapping[tuple[str, str], str] | None,
) -> schema_lib.FeatureSetSchema:
  """Infers the DGF FeatureSetSchema of a single node or edge table."""
  features: schema_lib.FeatureSetSchema = {}
  definitions = element_table.property_definitions or []
  is_composite_key = len(element_table.key_columns) > 1
  single_key_col = (
      element_table.key_columns[0]
      if len(element_table.key_columns) == 1
      else None
  )
  if is_node and is_composite_key:
    features[_COMPOSITE_PRIMARY_KEY_NAME] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
        is_utf8_string=True,
    )

  primary_assigned = _COMPOSITE_PRIMARY_KEY_NAME in features
  if not is_node:
    # The key columns of an edge table are not edge features.
    definitions = [
        definition
        for definition in definitions
        if definition.value_expression_sql.strip()
        not in element_table.key_columns
    ]
  for definition in definitions:
    feature_name = definition.property_declaration_name
    expr_col = definition.value_expression_sql.strip()
    raw_type = property_types[feature_name]
    if (
        column_spanner_types is not None
        and (element_table.base_table_name, expr_col) in column_spanner_types
    ):
      col_raw_type = column_spanner_types[
          (element_table.base_table_name, expr_col)
      ]
      _, _, vector_length = parse_spanner_type(col_raw_type)
    else:
      _, _, vector_length = parse_spanner_type(raw_type)

    scalar_type, is_array, _ = parse_spanner_type(raw_type)
    feature_format, is_utf8_string = gcp_common_lib.raw_type_to_feature_format(
        feature_name, scalar_type
    )

    if (
        is_node
        and not primary_assigned
        and single_key_col is not None
        and expr_col == single_key_col
    ):
      semantic = schema_lib.FeatureSemantic.PRIMARY_ID
      primary_assigned = True
    elif is_array and scalar_type == TYPE_TIMESTAMP:
      semantic = schema_lib.FeatureSemantic.TIMESERIES
    elif not is_array and scalar_type == TYPE_TIMESTAMP:
      semantic = schema_lib.FeatureSemantic.TIMESTAMP
    elif vector_length is not None:
      semantic = schema_lib.FeatureSemantic.EMBEDDING
    else:
      semantic = schema_lib.FeatureSemantic.UNKNOWN

    if not is_array:
      shape = None
    elif vector_length is not None:
      shape = (vector_length,)
    else:
      shape = [None]

    features[feature_name] = schema_lib.FeatureSchema(
        format=feature_format,
        semantic=semantic,
        shape=shape,  # pyrefly: ignore[bad-argument-type]
        is_utf8_string=is_utf8_string,
    )

  return features


def json_object_sql(expressions: Mapping[str, str]) -> str:
  """Builds the SQL JSON_OBJECT of the named expressions."""
  fields = ", ".join(
      f"'{name}', {expression}" for name, expression in expressions.items()
  )
  return f"JSON_OBJECT({fields})"


@dataclasses.dataclass(frozen=True)
class NodeReadPlan:
  """Precomputed SQL query reading a nodeset: The node id, then the features."""

  nodeset_name: str
  table: NodeTable
  query: str
  features: Mapping[str, FeatureSpec]


@dataclasses.dataclass(frozen=True)
class EdgeReadPlan:
  """Precomputed SQL query reading an edgeset: Both node ids, then the features."""

  edgeset_name: str
  query: str
  features: Mapping[str, FeatureSpec]


@dataclasses.dataclass(frozen=True)
class GraphReadPlan:
  """Precomputed read plans for all nodesets and edgesets in a schema."""

  node_plans: tuple[NodeReadPlan, ...]
  edge_plans: tuple[EdgeReadPlan, ...]
  # Spanner type of each query parameter holding a filter value.
  parameter_types: Mapping[str, str]


def _filter_parameter(column_name: str) -> str:
  """Gets the name of the query parameter holding the filter value of a column."""
  return f"filter_{column_name}"


def filter_column_type(
    table_name: str,
    column_name: str,
    column_spanner_types: Mapping[tuple[str, str], str],
) -> str:
  """Gets the Spanner type of a column the rows of a table are filtered on."""
  gcp_common_lib.validate_identifier(column_name)
  if (table_name, column_name) not in column_spanner_types:
    raise ValueError(
        f"Cannot filter on {column_name!r}: The Spanner table"
        f" {table_name!r} has no such column."
    )
  raw_type = column_spanner_types[(table_name, column_name)]
  type_kind, is_array, _ = parse_spanner_type(raw_type)
  if is_array or type_kind not in _SUPPORTED_FILTER_TYPES:
    raise ValueError(
        f"Cannot filter on column {column_name!r} of {table_name!r} of type"
        f" {raw_type!r}. Supported types: {sorted(_SUPPORTED_FILTER_TYPES)}."
    )
  return type_kind


def _filter_sql(
    table_name: str,
    filters: Mapping[str, Any],
    column_spanner_types: Mapping[tuple[str, str], str],
    parameter_types: dict[str, str],
) -> list[str]:
  """Gets the WHERE conditions of a table, and adds their parameter types."""
  conditions = []
  for column_name in filters:
    type_kind = filter_column_type(
        table_name, column_name, column_spanner_types
    )
    parameter = _filter_parameter(column_name)
    if parameter_types.setdefault(parameter, type_kind) != type_kind:
      raise ValueError(
          f"Cannot filter on {column_name!r}: This column has the types"
          f" {parameter_types[parameter]} and {type_kind} in different tables."
      )
    conditions.append(f"t.`{column_name}` = @{parameter}")
  return conditions


def _select_features_sql(table: NodeTable | EdgeTable) -> list[str]:
  """Gets the SELECT expressions reading the features of a table."""
  return [
      f"{feature.column.to_json_sql(feature.column.sql('t'))} AS"
      f" `{feature_name}`"
      for feature_name, feature in table.features.items()
  ]


def resolve_read_plan(
    metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
    schema: schema_lib.GraphSchema,
    column_spanner_types: Mapping[tuple[str, str], str] | None = None,
    *,
    bytes_are_base64: bool = False,
    filters: Mapping[str, Any] | None = None,
) -> GraphReadPlan:
  """Precomputes validated SQL read queries and parsers for schema.

  Args:
    metadata: The definition of the property graph.
    schema: The DGF schema.
    column_spanner_types: Spanner type of each (table_name, column_name), used
      for the key columns that are not properties.
    bytes_are_base64: Whether the client returns the BYTES values in base64.
    filters: Only the rows whose column equals the value are read, for each
      {column_name: value}. All the tables read must have these columns.

  Returns:
    The read plans of the nodesets and edgesets of the schema.
  """
  filters = filters or {}
  if filters and column_spanner_types is None:
    raise ValueError("Filters require the Spanner types of the columns.")
  parameter_types: dict[str, str] = {}
  tables = resolve_graph_tables(
      metadata, schema, column_spanner_types, bytes_are_base64=bytes_are_base64
  )

  node_plans: list[NodeReadPlan] = []
  for nodeset_name, table in tables.node_tables.items():
    id_sql = key_bytes_sql(table.keys, [col.sql("t") for col in table.keys])
    select_sql = ", ".join([f"{id_sql} AS id", *_select_features_sql(table)])
    query = f"SELECT {select_sql} FROM `{table.table}` AS t"
    if filters:
      conditions = _filter_sql(
          table.table, filters, column_spanner_types or {}, parameter_types
      )
      query += f" WHERE {' AND '.join(conditions)}"
    node_plans.append(
        NodeReadPlan(
            nodeset_name=nodeset_name,
            table=table,
            query=query,
            features=table.features,
        )
    )

  edge_plans: list[EdgeReadPlan] = []
  for edgeset_name, table in tables.edge_tables.items():
    source_id_sql = key_bytes_sql(
        table.source, [col.sql("t") for col in table.source]
    )
    target_id_sql = key_bytes_sql(
        table.target, [col.sql("t") for col in table.target]
    )
    # Like in GQL, a row with a NULL reference to a node is not an edge.
    non_null_clauses = [
        f"{col.sql('t')} IS NOT NULL" for col in (*table.source, *table.target)
    ]
    where_sql = " AND ".join([
        *dict.fromkeys(non_null_clauses),
        *_filter_sql(
            table.table, filters, column_spanner_types or {}, parameter_types
        ),
    ])
    select_sql = ", ".join([
        f"{source_id_sql} AS source_id",
        f"{target_id_sql} AS target_id",
        *_select_features_sql(table),
    ])
    edge_plans.append(
        EdgeReadPlan(
            edgeset_name=edgeset_name,
            query=(
                f"SELECT {select_sql} FROM `{table.table}` AS t WHERE"
                f" {where_sql}"
            ),
            features=table.features,
        )
    )

  return GraphReadPlan(
      node_plans=tuple(node_plans),
      edge_plans=tuple(edge_plans),
      parameter_types=parameter_types,
  )


def _pack_node_ids(
    element_name: str, node_ids: list[Any], bytes_are_base64: bool
) -> np.ndarray:
  """Packs the node ids returned by Spanner for a nodeset or edgeset."""
  if None in node_ids:
    raise ValueError(f"{element_name!r} has a NULL key in its Spanner table.")
  if bytes_are_base64:
    node_ids = [base64.b64decode(node_id) for node_id in node_ids]
  return pack_bytes_ids(node_ids)


def in_memory_node_set(
    nodeset_schema: schema_lib.NodeSchema,
    node_table: NodeTable,
    node_ids: np.ndarray,
    columns: FeatureColumns,
) -> in_memory_graph_lib.InMemoryNodeSet:
  """Builds a nodeset from the ids and the feature values of its nodes."""
  features = columns.pack()
  if node_table.primary_key is not None:
    features[node_table.primary_key] = node_ids
  return in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=len(node_ids),
      features={
          feature_name: features[feature_name]
          for feature_name in nodeset_schema.features
      },
  )


def _read_node_set(
    plan: NodeReadPlan,
    nodeset_schema: schema_lib.NodeSchema,
    rows: Iterable[Sequence[Any]],
    bytes_are_base64: bool,
    verbose: int,
) -> tuple[in_memory_graph_lib.InMemoryNodeSet, io_ext.ByteIdToIdxMapper]:
  """Builds an InMemoryNodeSet and ByteIdToIdxMapper from SQL result rows."""
  node_ids: list[Any] = []
  columns = FeatureColumns(plan.features)
  if verbose >= 1:
    rows = tqdm.tqdm(rows, desc=f"Loading nodes for {plan.nodeset_name}")
  for row in rows:
    node_ids.append(row[0])
    columns.append(itertools.islice(row, 1, None))

  id_array = _pack_node_ids(plan.nodeset_name, node_ids, bytes_are_base64)
  if verbose >= 1:
    log.info("Loaded %d nodes in nodeset %r", len(id_array), plan.nodeset_name)
  return (
      in_memory_node_set(nodeset_schema, plan.table, id_array, columns),
      io_ext.ByteIdToIdxMapper(id_array),
  )


def _read_edge_set(
    plan: EdgeReadPlan,
    rows: Iterable[Sequence[Any]],
    source_mapper: io_ext.ByteIdToIdxMapper,
    target_mapper: io_ext.ByteIdToIdxMapper,
    bytes_are_base64: bool,
    verbose: int,
) -> in_memory_graph_lib.InMemoryEdgeSet:
  """Builds an InMemoryEdgeSet from SQL result rows."""
  source_ids: list[Any] = []
  target_ids: list[Any] = []
  columns = FeatureColumns(plan.features)
  if verbose >= 1:
    rows = tqdm.tqdm(rows, desc=f"Loading edges for {plan.edgeset_name}")
  for row in rows:
    source_ids.append(row[0])
    target_ids.append(row[1])
    columns.append(itertools.islice(row, 2, None))

  if source_ids:
    source_arr = _pack_node_ids(plan.edgeset_name, source_ids, bytes_are_base64)
    target_arr = _pack_node_ids(plan.edgeset_name, target_ids, bytes_are_base64)
    adjacency, mismatch_src, mismatch_trg = io_ext.PairMapping(
        source_mapper,
        target_mapper,
        source_arr,
        target_arr,
        min(32, os.cpu_count() or 1),
    )
    if mismatch_src != -1:
      raise ValueError(
          f"Source Node ID {source_arr[mismatch_src]!r} not found for edge set"
          f" {plan.edgeset_name!r}."
      )
    if mismatch_trg != -1:
      raise ValueError(
          f"Target Node ID {target_arr[mismatch_trg]!r} not found for edge set"
          f" {plan.edgeset_name!r}."
      )
  else:
    adjacency = np.zeros((2, 0), dtype=np.int64)

  if verbose >= 1:
    log.info(
        "Loaded %d edges in edgeset %r", len(source_ids), plan.edgeset_name
    )
  return in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=adjacency, features=columns.pack()
  )


def read_graph_with_executor(
    metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
    execute_fn: Callable[
        [str, Mapping[str, Any], Mapping[str, str]], Iterable[Sequence[Any]]
    ],
    *,
    bytes_are_base64: bool,
    schema: schema_lib.GraphSchema | None = None,
    column_spanner_types: Mapping[tuple[str, str], str] | None = None,
    filters: Mapping[str, Any] | None = None,
    verbose: int = 1,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads all the nodesets and edgesets in memory, with one query each.

  Args:
    metadata: The definition of the property graph.
    execute_fn: Runs a SQL query with the Spanner client and returns its rows.
      Takes the query, the values of its parameters and their Spanner types.
    bytes_are_base64: Whether the client returns the BYTES values in base64.
    schema: The nodesets, edgesets and features to read. If None, reads the
      schema returned by graph_schema_from_metadata.
    column_spanner_types: Spanner type of each (table_name, column_name).
    filters: Only the rows whose column equals the value are read, for each
      {column_name: value}. All the tables read must have these columns.
    verbose: Verbosity (0 = silent, 1 = progress, 2 = debug).

  Returns:
    The in-memory graph and its schema.
  """
  start_time = time.time()
  if verbose >= 2:
    log.info("meta-data:\n%s", metadata)

  if schema is None:
    schema = graph_schema_from_metadata(
        metadata, column_spanner_types=column_spanner_types
    )

  if verbose >= 2:
    log.info("%s", print_schema_lib.print_schema(schema, return_output=True))
  if verbose >= 1:
    log.info("Reading nodesets and edgesets from Spanner base tables via SQL")

  read_plan = resolve_read_plan(
      metadata=metadata,
      schema=schema,
      column_spanner_types=column_spanner_types,
      bytes_are_base64=bytes_are_base64,
      filters=filters,
  )
  parameters = {
      _filter_parameter(column_name): value
      for column_name, value in (filters or {}).items()
  }

  node_sets: dict[str, in_memory_graph_lib.InMemoryNodeSet] = {}
  node_mappers: dict[str, io_ext.ByteIdToIdxMapper] = {}
  for node_plan in read_plan.node_plans:
    if verbose >= 2:
      log.info("node SQL query:\n%s", node_plan.query)
    node_sets[node_plan.nodeset_name], node_mappers[node_plan.nodeset_name] = (
        _read_node_set(
            plan=node_plan,
            nodeset_schema=schema.node_sets[node_plan.nodeset_name],
            rows=execute_fn(
                node_plan.query, parameters, read_plan.parameter_types
            ),
            bytes_are_base64=bytes_are_base64,
            verbose=verbose,
        )
    )

  edge_sets: dict[str, in_memory_graph_lib.InMemoryEdgeSet] = {}
  for edge_plan in read_plan.edge_plans:
    if verbose >= 2:
      log.info("edge SQL query:\n%s", edge_plan.query)
    edgeset_schema = schema.edge_sets[edge_plan.edgeset_name]
    edge_sets[edge_plan.edgeset_name] = _read_edge_set(
        plan=edge_plan,
        rows=execute_fn(edge_plan.query, parameters, read_plan.parameter_types),
        source_mapper=node_mappers[edgeset_schema.source],
        target_mapper=node_mappers[edgeset_schema.target],
        bytes_are_base64=bytes_are_base64,
        verbose=verbose,
    )

  log.info(
      "Graph read in memory in %s",
      util_lib.format_duration(time.time() - start_time),
  )
  return (
      in_memory_graph_lib.InMemoryGraph(
          node_sets=node_sets, edge_sets=edge_sets
      ),
      schema,
  )
