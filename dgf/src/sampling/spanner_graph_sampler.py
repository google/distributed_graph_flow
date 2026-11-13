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

"""Graph sampler from Spanner Graph.

This module contains the client-independent part of the Spanner Graph samplers.
"""

import abc
import base64
import collections
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import enum
import os
from typing import Any, TypeVar

from dgf.src.analyse import schema as schema_analyse_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.sampling import config as config_lib
import numpy as np

# Indices of the BYTES id columns in the result rows:
# (seed_id, node_id, element_type, element_class, source_id, target_id, json).
ID_COLUMN_INDICES = (0, 1, 4, 5)
SEED_IDS_PARAMETER = "seed_ids"
SAMPLING_SEED_PARAMETER = "sampling_seed"

# The CTEs are prefixed to not shadow the tables of the database.
_SOURCE_NODE_CTE = "_dgf_source_node"

_ELEMENT_TYPE_NODE = "node"
_ELEMENT_TYPE_EDGE = "edge"

_TYPE_BYTES = "BYTES"
_TYPE_STRING = "STRING"
_TYPE_TIMESTAMP = "TIMESTAMP"
_BYTES_DTYPE = np.dtype(
    feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
        schema_lib.FeatureFormat.BYTES
    ]
)

_ElementTableT = TypeVar(
    "_ElementTableT",
    spanner_graph_metadata_lib.NodeTable,
    spanner_graph_metadata_lib.EdgeTable,
)

_COMPATIBLE_TYPES: dict[schema_lib.FeatureFormat, frozenset[str]] = {
    schema_lib.FeatureFormat.BOOL: frozenset({"BOOL"}),
    schema_lib.FeatureFormat.BYTES: frozenset({_TYPE_STRING, _TYPE_BYTES}),
    schema_lib.FeatureFormat.FLOAT_32: frozenset({"FLOAT32", "FLOAT64"}),
    schema_lib.FeatureFormat.FLOAT_64: frozenset({"FLOAT32", "FLOAT64"}),
    schema_lib.FeatureFormat.INTEGER_32: frozenset({"INT32", "INT64"}),
    schema_lib.FeatureFormat.INTEGER_64: frozenset(
        {"INT32", "INT64", _TYPE_TIMESTAMP}
    ),
}


class QueryEngine(enum.Enum):
  """Implementation of the sampling query."""

  AUTO = "AUTO"
  SQL = "SQL"
  GQL = "GQL"


@dataclasses.dataclass(frozen=True)
class _Column:
  """Spanner column backing a DGF feature or key."""

  name: str
  type_kind: str
  is_array: bool = False

  def __post_init__(self) -> None:
    gcp_common_lib.validate_identifier(self.name)

  def sql(self, table_alias: str) -> str:
    """Gets the SQL reference to this column in table_alias."""
    return f"{table_alias}.`{self.name}`"

  def to_json_sql(self, expression: str) -> str:
    """Gets the SQL expression storing expression in a JSON object."""
    if self.type_kind == _TYPE_TIMESTAMP:
      if self.is_array:
        return (
            f"IF({expression} IS NULL, NULL, ARRAY(SELECT UNIX_MICROS(t) FROM"
            f" UNNEST({expression}) AS t WITH OFFSET off ORDER BY off))"
        )
      return f"UNIX_MICROS({expression})"
    return expression

  def to_bytes_sql(self, expression: str) -> str:
    """Gets the SQL expression casting expression to BYTES."""
    if self.type_kind == _TYPE_STRING:
      return f"CAST({expression} AS BYTES)"
    return expression


@dataclasses.dataclass(frozen=True)
class _Feature:
  """Spanner column and precomputed parser/array spec of a DGF feature."""

  column: _Column
  dtype: np.dtype
  is_ragged: bool
  empty_array: np.ndarray
  parse_value: Callable[[Any], Any]


@dataclasses.dataclass(frozen=True)
class _NodeTable:
  """Spanner table of a nodeset."""

  table: str
  label: str
  # Whether other node tables have this label.
  label_is_shared: bool
  key: _Column
  key_property: str
  # Non-key features (the primary key feature is populated from node_id).
  features: dict[str, _Feature]

  def __post_init__(self) -> None:
    gcp_common_lib.validate_identifier(self.table)
    gcp_common_lib.validate_identifier(self.label)
    gcp_common_lib.validate_identifier(self.key_property)
    for feature_name in self.features:
      gcp_common_lib.validate_identifier(feature_name)


@dataclasses.dataclass(frozen=True)
class _EdgeTable:
  """Spanner table of an edgeset."""

  table: str
  label: str
  # Whether other edge tables have this label.
  label_is_shared: bool
  source: _Column
  target: _Column
  features: dict[str, _Feature]

  def __post_init__(self) -> None:
    gcp_common_lib.validate_identifier(self.table)
    gcp_common_lib.validate_identifier(self.label)
    for feature_name in self.features:
      gcp_common_lib.validate_identifier(feature_name)


@dataclasses.dataclass(frozen=True)
class GraphTables:
  """Spanner tables of all the nodesets and edgesets of a DGF schema."""

  node_tables: dict[str, _NodeTable]
  edge_tables: dict[str, _EdgeTable]

  def __post_init__(self) -> None:
    for nodeset_name in self.node_tables:
      gcp_common_lib.validate_identifier(nodeset_name)
    for edgeset_name in self.edge_tables:
      gcp_common_lib.validate_identifier(edgeset_name)


def _parse_type(type_name: str) -> tuple[str, bool]:
  """Parses a Spanner type into (element_type_kind, is_array)."""
  if type_name.startswith("ARRAY<") and type_name.endswith(">"):
    return type_name[len("ARRAY<") : -1], True
  return type_name, False


def _element_table(
    element_name: str, element_tables: Sequence[_ElementTableT]
) -> tuple[_ElementTableT, str, bool]:
  """Finds the element table of a nodeset or edgeset.

  A nodeset or edgeset is named after a label defined on a single element
  table, or after an element table with a single label.

  Args:
    element_name: Name of the nodeset or edgeset.
    element_tables: The node tables or the edge tables of the property graph.

  Returns:
    The element table, its label, and whether other element tables have this
    label.
  """
  labeled_tables = [
      table for table in element_tables if element_name in table.label_names
  ]
  if len(labeled_tables) == 1:
    return labeled_tables[0], element_name, False
  if labeled_tables:
    raise ValueError(
        f"The label {element_name!r} is defined on {len(labeled_tables)}"
        " element tables of the Spanner Graph:"
        f" {sorted(table.name for table in labeled_tables)}. Name the nodeset"
        " or edgeset after one of these element tables instead."
    )
  named_tables = [
      table for table in element_tables if table.name == element_name
  ]
  if len(named_tables) != 1:
    labels = {label for table in element_tables for label in table.label_names}
    raise ValueError(
        f"{element_name!r} is neither a label nor an element table of the"
        f" Spanner Graph. Available labels: {sorted(labels)}. Available"
        f" element tables: {sorted(table.name for table in element_tables)}."
    )
  table = named_tables[0]
  if len(table.label_names) != 1:
    raise ValueError(
        f"The element table {element_name!r} has {len(table.label_names)}"
        f" labels: {table.label_names}. Name the nodeset or edgeset after one"
        " of them instead."
    )
  label = table.label_names[0]
  num_tables_with_label = sum(
      label in other_table.label_names for other_table in element_tables
  )
  return table, label, num_tables_with_label > 1


def _default_scalar(feature_format: schema_lib.FeatureFormat) -> Any:
  """Gets the DGF default value for a NULL scalar."""
  if feature_format == schema_lib.FeatureFormat.BYTES:
    return b""
  if feature_format == schema_lib.FeatureFormat.BOOL:
    return False
  if feature_format.is_float():
    return 0.0
  if feature_format.is_integer():
    return 0
  raise NotImplementedError(f"Unsupported feature format: {feature_format}.")


def _make_value_parser(
    element_name: str,
    feature_name: str,
    column: _Column,
    dtype: np.dtype,
    default_scalar: Any,
    expected_len: int | None,
) -> Callable[[Any], Any]:
  """Builds the per-row JSON value parser for a feature."""
  if column.type_kind == _TYPE_BYTES:
    parse_scalar = lambda v: b"" if v is None else base64.b64decode(v)
  elif column.type_kind == _TYPE_STRING:
    parse_scalar = lambda v: b"" if v is None else v.encode("utf-8")
  else:
    parse_scalar = None

  if not column.is_array:
    if parse_scalar is not None:
      return parse_scalar
    return lambda v: default_scalar if v is None else v

  if expected_len is None:
    empty_element = np.empty(0, dtype=dtype)

    def parse_ragged(value: Any) -> np.ndarray:
      if value is None:
        return empty_element
      if parse_scalar is not None:
        value = [parse_scalar(item) for item in value]
      elif None in value:
        value = [default_scalar if item is None else item for item in value]
      return np.asarray(value, dtype=dtype)

    return parse_ragged

  default_list = [default_scalar] * expected_len

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
      return [default_scalar if item is None else item for item in value]
    return value

  return parse_static_array


def _resolve_feature(
    element_name: str,
    feature_name: str,
    feature_schema: schema_lib.FeatureSchema,
    expressions: Mapping[str, str],
    property_types: Mapping[str, str],
) -> _Feature:
  """Resolves and validates the column and parser of a feature."""
  gcp_common_lib.validate_identifier(feature_name)
  if feature_name not in expressions:
    raise ValueError(
        f"Feature {feature_name!r} of {element_name!r} is not a property of"
        f" the Spanner Graph. Available properties: {sorted(expressions)}"
    )
  column_name = expressions[feature_name]
  if not gcp_common_lib.IDENTIFIER_REGEX.match(column_name):
    raise NotImplementedError(
        f"Feature {feature_name!r} of {element_name!r} is defined by the"
        f" expression {column_name!r} in the Spanner Graph. Only properties"
        " defined by a plain column are supported."
    )
  type_kind, is_array = _parse_type(property_types[feature_name])

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

  column = _Column(name=column_name, type_kind=type_kind, is_array=is_array)
  dtype = np.dtype(
      feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[feature_schema.format]
  )
  expected_len = shape[0] if shape else None
  is_ragged = bool(shape) and expected_len is None
  if is_ragged:
    empty_array = np.empty(0, dtype=np.object_)
  elif expected_len is not None:
    empty_array = np.empty((0, expected_len), dtype=dtype)
  else:
    empty_array = np.empty(0, dtype=dtype)
  default_scalar = _default_scalar(feature_schema.format)
  return _Feature(
      column=column,
      dtype=dtype,
      is_ragged=is_ragged,
      empty_array=empty_array,
      parse_value=_make_value_parser(
          element_name,
          feature_name,
          column,
          dtype,
          default_scalar,
          expected_len,
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
      definition.property_declaration_name: definition.value_expression_sql
      for definition in element_table.property_definitions or ()
  }


def _resolve_node_table(
    nodeset_name: str,
    nodeset_schema: schema_lib.NodeSchema,
    node_table: spanner_graph_metadata_lib.NodeTable,
    label: str,
    label_is_shared: bool,
    property_types: Mapping[str, str],
) -> _NodeTable:
  """Resolves the Spanner table, key, and features of a nodeset."""
  if len(node_table.key_columns) != 1:
    raise NotImplementedError(
        f"Nodeset {nodeset_name!r} has a composite key"
        f" {node_table.key_columns}. Only single-column keys are supported."
    )
  expressions = _property_expressions(node_table)
  features = {
      feature_name: _resolve_feature(
          nodeset_name,
          feature_name,
          feature_schema,
          expressions,
          property_types,
      )
      for feature_name, feature_schema in nodeset_schema.features.items()
  }
  primary_key = schema_analyse_lib.primary_feature(nodeset_name, nodeset_schema)
  key = features.pop(primary_key).column
  if key.name != node_table.key_columns[0]:
    raise ValueError(
        f"The primary key feature {primary_key!r} of {nodeset_name!r} is"
        f" stored in column {key.name!r}, but the key of the Spanner node"
        f" table is {node_table.key_columns[0]!r}."
    )
  if key.type_kind not in (_TYPE_STRING, _TYPE_BYTES) or key.is_array:
    raise ValueError(
        f"The primary key {primary_key!r} of {nodeset_name!r} has type"
        f" {key.type_kind}. Only STRING and BYTES primary keys are supported."
    )
  return _NodeTable(
      table=node_table.base_table_name,
      label=label,
      label_is_shared=label_is_shared,
      key=key,
      key_property=primary_key,
      features=features,
  )


def _resolve_endpoint(
    edgeset_name: str,
    expected_nodeset: str,
    reference: spanner_graph_metadata_lib.NodeTableRef,
    node_table_names: Mapping[str, str],
    node_tables: Mapping[str, _NodeTable],
) -> _Column:
  """Resolves the edge table column referencing the key of expected_nodeset."""
  if reference.node_table_name != node_table_names[expected_nodeset]:
    raise ValueError(
        f"Edgeset {edgeset_name!r} references the node table"
        f" {reference.node_table_name!r} in the Spanner Graph, but the"
        f" nodeset {expected_nodeset!r} in the schema is the node table"
        f" {node_table_names[expected_nodeset]!r}."
    )
  node_key = node_tables[expected_nodeset].key
  if reference.node_table_columns != [node_key.name]:
    raise ValueError(
        f"Edgeset {edgeset_name!r} references columns"
        f" {reference.node_table_columns} of {expected_nodeset!r} instead of"
        f" its key {node_key.name!r}."
    )
  if len(reference.edge_table_columns) != 1:
    raise NotImplementedError(
        f"Edgeset {edgeset_name!r} has composite endpoint columns"
        f" {reference.edge_table_columns}."
    )
  return _Column(
      name=reference.edge_table_columns[0], type_kind=node_key.type_kind
  )


def resolve_graph_tables(
    metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
    schema: schema_lib.GraphSchema,
) -> GraphTables:
  """Resolves the Spanner tables and columns of all the schema elements."""
  property_types = metadata.property_types or {}
  node_tables = {}
  node_table_names = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    gcp_common_lib.validate_identifier(nodeset_name)
    node_table, label, label_is_shared = _element_table(
        nodeset_name, metadata.node_tables
    )
    node_table_names[nodeset_name] = node_table.name
    node_tables[nodeset_name] = _resolve_node_table(
        nodeset_name,
        nodeset_schema,
        node_table,
        label,
        label_is_shared,
        property_types,
    )

  edge_tables = {}
  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    gcp_common_lib.validate_identifier(edgeset_name)
    edge_table, label, label_is_shared = _element_table(
        edgeset_name, metadata.edge_tables
    )
    expressions = _property_expressions(edge_table)
    edge_tables[edgeset_name] = _EdgeTable(
        table=edge_table.base_table_name,
        label=label,
        label_is_shared=label_is_shared,
        source=_resolve_endpoint(
            edgeset_name,
            edgeset_schema.source,
            edge_table.source_node_table,
            node_table_names,
            node_tables,
        ),
        target=_resolve_endpoint(
            edgeset_name,
            edgeset_schema.target,
            edge_table.destination_node_table,
            node_table_names,
            node_tables,
        ),
        features={
            feature_name: _resolve_feature(
                edgeset_name,
                feature_name,
                feature_schema,
                expressions,
                property_types,
            )
            for feature_name, feature_schema in edgeset_schema.features.items()
        },
    )
  return GraphTables(node_tables=node_tables, edge_tables=edge_tables)


def _features_json_sql(
    table_alias: str, features: Mapping[str, _Feature]
) -> str:
  """Builds a SQL JSON_OBJECT expression for a row's features."""
  if not features:
    return "CAST(NULL AS JSON)"
  fields = ", ".join(
      f"'{feature_name}',"
      f" {feature.column.to_json_sql(feature.column.sql(table_alias))}"
      for feature_name, feature in features.items()
  )
  return f"JSON_OBJECT({fields})"


def _gql_features_json_sql(
    variable: str, features: Mapping[str, _Feature]
) -> str:
  """Builds a GQL JSON_OBJECT expression for an element's properties."""
  if not features:
    return "CAST(NULL AS JSON)"
  fields = ", ".join(
      f"'{feature_name}',"
      f" {feature.column.to_json_sql(f'{variable}.`{feature_name}`')}"
      for feature_name, feature in features.items()
  )
  return f"JSON_OBJECT({fields})"


class _QueryGenerator:
  """Compiles a SamplingPlan into a parameterized Spanner query.

  Spanner production databases enforce strict name resolution: all columns are
  qualified with a table alias.
  """

  def __init__(
      self,
      tables: GraphTables,
      plan: config_lib.SamplingPlan,
      engine: QueryEngine,
      graph_name: str,
      debug_sampling: bool,
  ):
    if engine == QueryEngine.AUTO:
      raise ValueError("The engine must be resolved before generating a query.")
    gcp_common_lib.validate_identifier(graph_name)
    self._tables = tables
    self._plan = plan
    self._engine = engine
    self._graph_name = graph_name
    self._debug_sampling = debug_sampling
    self._seed_key = tables.node_tables[plan.root.nodeset].key
    self._num_hops = 0
    self._ctes: list[str] = []
    self._node_queries: dict[str, list[str]] = collections.defaultdict(list)
    self._edge_queries: dict[str, list[str]] = collections.defaultdict(list)

  def generate(self) -> str:
    """Builds the parameterized sampling query."""
    root_nodeset = self._plan.root.nodeset
    root_table = self._tables.node_tables[root_nodeset]
    root_key = root_table.key.sql("n")
    self._ctes.append(
        f"  {_SOURCE_NODE_CTE} AS (\n"
        f"    SELECT {root_key} AS seed_id, {root_key} AS n0_id\n"
        f"    FROM `{root_table.table}` AS n\n"
        f"    WHERE {root_key} IN UNNEST(@{SEED_IDS_PARAMETER})\n"
        "  )"
    )
    self._node_queries[root_nodeset].append(
        f"SELECT s.seed_id, s.n0_id AS node_id FROM {_SOURCE_NODE_CTE} AS s"
    )
    self._visit(
        self._plan.root, parent_cte=_SOURCE_NODE_CTE, parent_column="n0_id"
    )

    results = [
        self._nodes_query(nodeset_name, queries)
        for nodeset_name, queries in self._node_queries.items()
    ] + [
        self._edges_query(edgeset_name, queries)
        for edgeset_name, queries in self._edge_queries.items()
    ]
    return (
        "WITH\n" + ",\n".join(self._ctes) + "\n" + "\nUNION ALL\n".join(results)
    )

  def _visit(
      self, node: config_lib.PlanNode, parent_cte: str, parent_column: str
  ) -> None:
    """Appends one hop CTE per child of node, recursively."""
    parent_table = self._tables.node_tables[node.nodeset]
    for edge in node.children:
      if edge.hop_width < 0:
        raise ValueError(
            f"hop_width must be non-negative, got {edge.hop_width} on"
            f" {edge.edgeset!r}."
        )
      if edge.hop_width == 0:
        continue
      self._num_hops += 1
      cte_name = f"_dgf_hop_{self._num_hops}"
      child_column = f"n{self._num_hops}_id"
      if self._engine == QueryEngine.SQL:
        neighbors, child_id = self._sql_neighbors(edge, parent_column)
      else:
        neighbors, child_id = self._gql_neighbors(node, edge, parent_column)
      child_key = self._tables.node_tables[edge.node.nodeset].key.to_bytes_sql(
          child_id
      )
      if self._debug_sampling:
        order = child_key
      else:
        # Hash with @sampling_seed instead of GENERATE_UUID() so every reference
        # to a CTE sees the same sampled neighbors. The hop makes a node
        # reached by several hops sample different neighbors in each of them.
        order = (
            f"FARM_FINGERPRINT(CONCAT(@{SAMPLING_SEED_PARAMETER},"
            f" b'{self._num_hops}:',"
            f" {self._seed_key.to_bytes_sql('p.seed_id')},"
            f" {parent_table.key.to_bytes_sql(f'p.{parent_column}')},"
            f" {child_key}))"
        )
      self._ctes.append(
          f"  {cte_name} AS (\n"
          f"    SELECT p.seed_id, p.{parent_column} AS parent_id,"
          f" c.child_id AS {child_column}, c.edge_json\n"
          f"    FROM (SELECT DISTINCT q.seed_id, q.{parent_column} FROM"
          f" {parent_cte} AS q) AS p,\n"
          "    UNNEST(ARRAY(\n"
          f"{neighbors}\n"
          # The d format rejects non-integer values.
          f"      ORDER BY {order} LIMIT {edge.hop_width:d}\n"
          "    )) AS c\n"
          "  )"
      )

      self._node_queries[edge.node.nodeset].append(
          f"SELECT h.seed_id, h.{child_column} AS node_id FROM {cte_name} AS h"
      )
      if edge.reversed:
        source_column, target_column = child_column, "parent_id"
      else:
        source_column, target_column = "parent_id", child_column
      self._edge_queries[edge.edgeset].append(
          f"SELECT h.seed_id, h.{source_column} AS source_id,"
          f" h.{target_column} AS target_id, h.edge_json FROM {cte_name} AS h"
      )
      self._visit(edge.node, parent_cte=cte_name, parent_column=child_column)

  def _sql_neighbors(
      self, edge: config_lib.PlanEdge, parent_column: str
  ) -> tuple[str, str]:
    """Gets the SQL query of the neighbors of a parent, and their id column."""
    edge_table = self._tables.edge_tables[edge.edgeset]
    if edge.reversed:
      join_column, child_column = edge_table.target, edge_table.source
    else:
      join_column, child_column = edge_table.source, edge_table.target
    child_id = child_column.sql("e")
    edge_json = _features_json_sql("e", edge_table.features)
    query = "\n".join([
        (
            f"      SELECT AS STRUCT {child_id} AS child_id, {edge_json} AS"
            " edge_json"
        ),
        f"      FROM `{edge_table.table}` AS e",
        (
            f"      WHERE {join_column.sql('e')} = p.{parent_column}"
            f" AND {child_id} IS NOT NULL"
        ),
    ])
    return query, child_id

  def _gql_neighbors(
      self,
      node: config_lib.PlanNode,
      edge: config_lib.PlanEdge,
      parent_column: str,
  ) -> tuple[str, str]:
    """Gets the GQL query of the neighbors of a parent, and their id column."""
    edge_table = self._tables.edge_tables[edge.edgeset]
    parent_table = self._tables.node_tables[node.nodeset]
    child_table = self._tables.node_tables[edge.node.nodeset]
    for element_table in (parent_table, edge_table, child_table):
      if element_table.label_is_shared:
        raise NotImplementedError(
            f"The label {element_table.label!r} is defined on several element"
            " tables of the Spanner Graph, and a GQL query would match all of"
            " them. Use query_engine SQL instead."
        )
    parent = f"(gp:`{parent_table.label}`)"
    child = f"(gc:`{child_table.label}`)"
    if edge.reversed:
      pattern = f"{parent}<-[ge:`{edge_table.label}`]-{child}"
    else:
      pattern = f"{parent}-[ge:`{edge_table.label}`]->{child}"
    child_key = f"gc.`{child_table.key_property}`"
    edge_json = _gql_features_json_sql("ge", edge_table.features)
    query = "\n".join([
        "      SELECT AS STRUCT g.child_id, g.edge_json",
        f"      FROM GRAPH_TABLE(`{self._graph_name}`",
        f"        MATCH {pattern}",
        f"        WHERE gp.`{parent_table.key_property}` = p.{parent_column}",
        f"        RETURN {child_key} AS child_id, {edge_json} AS edge_json",
        "      ) AS g",
    ])
    return query, "g.child_id"

  def _nodes_query(self, nodeset_name: str, queries: list[str]) -> str:
    """Builds the SELECT query returning the sampled nodes of nodeset_name."""
    node_table = self._tables.node_tables[nodeset_name]
    cte_name = f"_dgf_nodes_{nodeset_name}"
    union = "\n      UNION ALL\n      ".join(queries)
    self._ctes.append(
        f"  {cte_name} AS (\n"
        "    SELECT DISTINCT a.seed_id, a.node_id FROM (\n"
        f"      {union}\n"
        "    ) AS a\n"
        "  )"
    )
    # The node table is joined even without features to read: An edge to a node
    # missing from its node table is then reported instead of creating a node.
    return f"""\
SELECT
  {self._seed_key.to_bytes_sql('u.seed_id')} AS seed_id,
  {node_table.key.to_bytes_sql('u.node_id')} AS node_id,
  '{_ELEMENT_TYPE_NODE}' AS element_type,
  '{nodeset_name}' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  {_features_json_sql('n', node_table.features)} AS properties_json
FROM {cte_name} AS u
INNER JOIN `{node_table.table}` AS n ON u.node_id = {node_table.key.sql('n')}"""

  def _edges_query(self, edgeset_name: str, queries: list[str]) -> str:
    """Builds the SELECT query returning the sampled edges of edgeset_name."""
    edge_table = self._tables.edge_tables[edgeset_name]
    union = "\n  UNION ALL\n  ".join(queries)
    return f"""\
SELECT
  {self._seed_key.to_bytes_sql('e.seed_id')} AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  '{_ELEMENT_TYPE_EDGE}' AS element_type,
  '{edgeset_name}' AS element_class,
  {edge_table.source.to_bytes_sql('e.source_id')} AS source_id,
  {edge_table.target.to_bytes_sql('e.target_id')} AS target_id,
  e.edge_json AS properties_json
FROM (
  {union}
) AS e"""


def generate_query(
    tables: GraphTables,
    plan: config_lib.SamplingPlan,
    engine: QueryEngine,
    graph_name: str,
    debug_sampling: bool = False,
) -> str:
  """Compiles a sampling plan into a parameterized Spanner query.

  Args:
    tables: The Spanner tables of the schema.
    plan: The sampling plan.
    engine: Implementation of the query (SQL or GQL).
    graph_name: Name of the Spanner property graph.
    debug_sampling: If true, the sampled neighbors are the ones with the
      smallest ids.

  Returns:
    The query, with the parameters @seed_ids and (without debug_sampling)
    @sampling_seed.
  """
  return _QueryGenerator(
      tables, plan, engine, graph_name, debug_sampling
  ).generate()


def _append_parsed_features(
    properties: Mapping[str, Any] | None,
    features: Mapping[str, _Feature],
    output: dict[str, list[Any]],
) -> None:
  """Parses the properties of an element and appends them to output."""
  if not features:
    return
  if properties is None:
    raise ValueError("Expected JSON properties object, got None.")
  for feature_name, feature in features.items():
    output[feature_name].append(feature.parse_value(properties[feature_name]))


def _pack_feature(feature: _Feature, values: list[Any]) -> np.ndarray:
  """Packs parsed per-element values into a single feature array."""
  if not values:
    return feature.empty_array
  if feature.is_ragged:
    stacked = np.empty(len(values), dtype=np.object_)
    for index, item in enumerate(values):
      stacked[index] = item
    return stacked
  return np.array(values, dtype=feature.dtype)


@dataclasses.dataclass
class _RawNodeSet:
  """Nodes of a nodeset of a subgraph being assembled."""

  node_index: dict[bytes, int] = dataclasses.field(default_factory=dict)
  features: dict[str, list[Any]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )


@dataclasses.dataclass
class _RawEdgeSet:
  """Edges of an edgeset of a subgraph being assembled."""

  edges: set[tuple[int, int]] = dataclasses.field(default_factory=set)
  sources: list[int] = dataclasses.field(default_factory=list)
  targets: list[int] = dataclasses.field(default_factory=list)
  features: dict[str, list[Any]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )


class _SubgraphBuilder:
  """Assembles the subgraph of a seed from the rows of the sampling query."""

  def __init__(self, schema: schema_lib.GraphSchema, tables: GraphTables):
    self._schema = schema
    self._tables = tables
    self._node_sets = collections.defaultdict(_RawNodeSet)
    self._edge_sets = collections.defaultdict(_RawEdgeSet)

  def add_node(
      self,
      nodeset_name: str,
      node_id: bytes,
      properties: Mapping[str, Any] | None,
  ) -> None:
    """Adds a node, failing if it was already added."""
    nodeset = self._node_sets[nodeset_name]
    if node_id in nodeset.node_index:
      raise ValueError(
          f"Duplicate node {node_id!r} in nodeset {nodeset_name!r}."
      )
    nodeset.node_index[node_id] = len(nodeset.node_index)
    _append_parsed_features(
        properties,
        self._tables.node_tables[nodeset_name].features,
        nodeset.features,
    )

  def add_edge(
      self,
      edgeset_name: str,
      source_id: bytes,
      target_id: bytes,
      properties: Mapping[str, Any] | None,
  ) -> None:
    """Adds an edge, unless it is a parallel edge of an already added edge."""
    edgeset_schema = self._schema.edge_sets[edgeset_name]
    edgeset = self._edge_sets[edgeset_name]
    edge = (
        self._node_index(edgeset_name, edgeset_schema.source, source_id),
        self._node_index(edgeset_name, edgeset_schema.target, target_id),
    )
    if edge in edgeset.edges:
      return
    edgeset.edges.add(edge)
    edgeset.sources.append(edge[0])
    edgeset.targets.append(edge[1])
    _append_parsed_features(
        properties,
        self._tables.edge_tables[edgeset_name].features,
        edgeset.features,
    )

  def _node_index(
      self, edgeset_name: str, nodeset_name: str, node_id: bytes
  ) -> int:
    """Gets the subgraph index of an edge endpoint."""
    node_index = self._node_sets[nodeset_name].node_index
    if node_id not in node_index:
      raise ValueError(
          f"Edge of {edgeset_name!r} references node {node_id!r} which is"
          f" missing from the Spanner node table of {nodeset_name!r}."
      )
    return node_index[node_id]

  def build(self) -> in_memory_graph_lib.InMemoryGraph:
    """Builds the InMemoryGraph of the assembled subgraph."""
    node_sets = {}
    for nodeset_name, nodeset_schema in self._schema.node_sets.items():
      nodeset = self._node_sets[nodeset_name]
      node_table = self._tables.node_tables[nodeset_name]
      key_array = np.array(list(nodeset.node_index), dtype=_BYTES_DTYPE)
      features = {}
      for feature_name in nodeset_schema.features:
        if feature_name == node_table.key_property:
          features[feature_name] = key_array
        else:
          features[feature_name] = _pack_feature(
              node_table.features[feature_name], nodeset.features[feature_name]
          )
      node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
          num_nodes=len(nodeset.node_index), features=features
      )

    edge_sets = {}
    for edgeset_name, edgeset_schema in self._schema.edge_sets.items():
      edgeset = self._edge_sets[edgeset_name]
      edge_table = self._tables.edge_tables[edgeset_name]
      edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
          adjacency=np.array(
              [edgeset.sources, edgeset.targets], dtype=np.int64
          ).reshape(2, -1),
          features={
              feature_name: _pack_feature(
                  edge_table.features[feature_name],
                  edgeset.features[feature_name],
              )
              for feature_name in edgeset_schema.features
          },
      )
    return in_memory_graph_lib.InMemoryGraph(
        node_sets=node_sets, edge_sets=edge_sets
    )


def rows_to_in_memory_graphs(
    rows: Iterable[Sequence[Any]],
    schema: schema_lib.GraphSchema,
    tables: GraphTables,
    seed_nodeset: str,
    seed_ids: Sequence[bytes],
) -> list[in_memory_graph_lib.InMemoryGraph]:
  """Converts the rows of the sampling query into one subgraph per seed.

  Args:
    rows: Rows of the sampling query, as returned by
      SpannerGraphSampler._execute.
    schema: The DGF schema.
    tables: The Spanner tables of the schema.
    seed_nodeset: The seed nodeset.
    seed_ids: The seed ids, in the order of the returned subgraphs.

  Returns:
    One subgraph per seed. The seed node is at index 0 of the seed nodeset.
  """
  expected_seeds = set(seed_ids)
  found_seeds = set()
  node_rows = collections.defaultdict(list)
  edge_rows = collections.defaultdict(list)
  for row in rows:
    (
        seed_id,
        node_id,
        element_type,
        element_class,
        source_id,
        target_id,
        properties,
    ) = row
    if seed_id not in expected_seeds:
      raise ValueError(f"Unexpected seed_id {seed_id!r} in query result.")
    if element_type == _ELEMENT_TYPE_NODE:
      if element_class == seed_nodeset and node_id == seed_id:
        found_seeds.add(seed_id)
      node_rows[seed_id].append((element_class, node_id, properties))
    elif element_type == _ELEMENT_TYPE_EDGE:
      edge_rows[seed_id].append(
          (element_class, source_id, target_id, properties)
      )
    else:
      raise ValueError(f"Unknown element type: {element_type!r}")

  missing_seeds = [
      seed_id for seed_id in seed_ids if seed_id not in found_seeds
  ]
  if missing_seeds:
    raise ValueError(
        f"Seed nodes {missing_seeds!r} not found in nodeset {seed_nodeset!r}."
    )

  graphs = []
  for seed_id in seed_ids:
    builder = _SubgraphBuilder(schema, tables)
    # Add the seed node first so it sits at index 0 of the seed nodeset.
    seed_entries, other_entries = [], []
    for entry in node_rows[seed_id]:
      is_seed = entry[0] == seed_nodeset and entry[1] == seed_id
      (seed_entries if is_seed else other_entries).append(entry)
    for nodeset_name, node_id, properties in seed_entries + other_entries:
      builder.add_node(nodeset_name, node_id, properties)
    for edgeset_name, source_id, target_id, properties in edge_rows[seed_id]:
      builder.add_edge(edgeset_name, source_id, target_id, properties)
    graphs.append(builder.build())
  return graphs


class SpannerGraphSampler(abc.ABC):
  """Base class of the Spanner Graph samplers."""

  def __init__(
      self,
      metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
      graph: str,
      plan: config_lib.SimpleSamplingConfig | config_lib.SamplingPlan,
      schema: schema_lib.GraphSchema,
      debug_sampling: bool,
      query_engine: QueryEngine | str,
      sampler_name: str,
  ):
    if isinstance(plan, config_lib.SimpleSamplingConfig):
      plan = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)
    config_lib.check_supported_options(
        plan,
        schema,
        sampler_name,
        support_non_with_replacement=True,
        support_non_multi_visit=True,
    )
    query_engine = QueryEngine(query_engine)
    if query_engine == QueryEngine.AUTO:
      query_engine = QueryEngine.SQL
    self._plan = plan
    self._schema = schema
    self._debug_sampling = debug_sampling
    self._query_engine = query_engine
    self._tables = resolve_graph_tables(metadata, schema)
    self._query = generate_query(
        self._tables, plan, query_engine, graph, debug_sampling
    )
    self._root_key_is_string = (
        self._tables.node_tables[plan.root.nodeset].key.type_kind
        == _TYPE_STRING
    )

  @property
  def query_engine(self) -> QueryEngine:
    """Gets the engine of the sampling query (never AUTO)."""
    return self._query_engine

  @property
  def query(self) -> str:
    """Gets the compiled sampling query."""
    return self._query

  def sample(
      self, seed_ids: Sequence[str | bytes] | np.ndarray
  ) -> list[in_memory_graph_lib.InMemoryGraph]:
    """Samples one subgraph per seed node, in a single query.

    Args:
      seed_ids: The primary ids (as bytes or UTF-8 str) of the seed nodes.

    Returns:
      One subgraph per seed, in the order of seed_ids. The seed node is at
      index 0 of the seed nodeset.
    """
    seed_bytes = [
        seed_id.encode("utf-8") if isinstance(seed_id, str) else bytes(seed_id)
        for seed_id in seed_ids
    ]
    if not seed_bytes:
      return []
    if self._root_key_is_string:
      seed_values = [seed_id.decode("utf-8") for seed_id in seed_bytes]
    else:
      seed_values = seed_bytes
    rows = self._execute(
        seed_values=seed_values,
        sampling_seed=None if self._debug_sampling else os.urandom(8),
    )
    return rows_to_in_memory_graphs(
        rows,
        self._schema,
        self._tables,
        self._plan.root.nodeset,
        seed_bytes,
    )

  @abc.abstractmethod
  def _execute(
      self,
      seed_values: list[bytes] | list[str],
      sampling_seed: bytes | None,
  ) -> Iterable[Sequence[Any]]:
    """Runs the sampling query and returns its rows.

    Args:
      seed_values: Value of the @seed_ids parameter (an ARRAY<STRING> if the
        values are str, an ARRAY<BYTES> otherwise).
      sampling_seed: Value of the @sampling_seed BYTES parameter, or None when
        debug_sampling is enabled.

    Returns:
      The rows (seed_id, node_id, element_type, element_class, source_id,
      target_id, properties) of the query. The ids are raw bytes or None. The
      properties are a parsed JSON object or None.
    """
