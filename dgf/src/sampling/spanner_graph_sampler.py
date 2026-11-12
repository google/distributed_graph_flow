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
import collections
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import enum
import os
from typing import Any

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import spanner_graph as spanner_graph_io_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.sampling import config as config_lib
import numpy as np

# Indices of the BYTES id columns in the result rows:
# (seed_id, node_id, element_type, element_class, source_id, target_id, json).
ID_COLUMN_INDICES = (0, 1, 4, 5)
SEED_IDS_PARAMETER = "seed_ids"
SAMPLING_SEED_PARAMETER = "sampling_seed"

# Types of the query parameters holding the seed keys.
PARAM_TYPE_BOOL = "BOOL"
PARAM_TYPE_BYTES = "BYTES"
PARAM_TYPE_INT64 = "INT64"
PARAM_TYPE_STRING = "STRING"

# The CTEs are prefixed to not shadow the tables of the database.
_SOURCE_NODE_CTE = "_dgf_source_node"

_ELEMENT_TYPE_NODE = "node"
_ELEMENT_TYPE_EDGE = "edge"

# Key column types whose values are bound as query parameters of another type.
_SEED_PARAMETER_CASTS: dict[str, str] = {
    spanner_graph_io_lib.TYPE_TIMESTAMP: "TIMESTAMP_MICROS({})",
    spanner_graph_io_lib.TYPE_DATE: "CAST({} AS DATE)",
    spanner_graph_io_lib.TYPE_INT32: "CAST({} AS INT32)",
}

Column = spanner_graph_io_lib.Column
FeatureSpec = spanner_graph_io_lib.FeatureSpec
GraphTables = spanner_graph_io_lib.GraphTables


class QueryEngine(enum.Enum):
  """Implementation of the sampling query."""

  AUTO = "AUTO"
  SQL = "SQL"
  GQL = "GQL"


def resolve_graph_tables(
    metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
    schema: schema_lib.GraphSchema,
    column_spanner_types: Mapping[tuple[str, str], str] | None = None,
) -> GraphTables:
  """Resolves the Spanner tables and columns of all the schema elements.

  Args:
    metadata: The definition of the property graph.
    schema: The DGF schema.
    column_spanner_types: Spanner type of each (table_name, column_name), used
      for the key columns that are not properties.

  Returns:
    The Spanner tables of the nodesets and edgesets of the schema.
  """
  tables = spanner_graph_io_lib.resolve_graph_tables(
      metadata, schema, column_spanner_types, bytes_are_base64=True
  )
  for element_name, element_table in (
      *tables.node_tables.items(),
      *tables.edge_tables.items(),
  ):
    if element_table.label is None:
      raise ValueError(
          f"The element table {element_name!r} has several labels. Name the"
          " nodeset or edgeset after one of them instead."
      )
  # The nodes are identified by their encoded key in the sampling query.
  for nodeset_name, node_table in tables.node_tables.items():
    if node_table.primary_key is None:
      raise ValueError(
          f"Nodeset {nodeset_name!r} must have a BYTES primary key feature"
          " stored in the key of its Spanner node table."
      )
  return tables


def _key_columns(prefix: str, keys: Sequence[Column]) -> list[str]:
  """Gets the query columns of a key: <prefix>_id, or one per key column."""
  if len(keys) == 1:
    return [f"{prefix}_id"]
  return [f"{prefix}_k{index}" for index in range(len(keys))]


def _select_sql(
    alias: str, columns: Sequence[str], names: Sequence[str] | None = None
) -> str:
  """Gets the SELECT list of the columns of alias, optionally renamed."""
  if names is None:
    return ", ".join(f"{alias}.{column}" for column in columns)
  return ", ".join(
      f"{alias}.{column} AS {name}" for column, name in zip(columns, names)
  )


def _key_bytes_sql(keys: Sequence[Column], alias: str, prefix: str) -> str:
  """Gets the SQL BYTES id of the key stored in the prefix columns of alias."""
  return spanner_graph_io_lib.key_bytes_sql(
      keys, [f"{alias}.{column}" for column in _key_columns(prefix, keys)]
  )


def _seed_parameter_type(key: Column) -> str:
  """Gets the type of the query parameter holding the seed values of key."""
  if key.type_kind in (
      spanner_graph_io_lib.TYPE_STRING,
      spanner_graph_io_lib.TYPE_DATE,
  ):
    return PARAM_TYPE_STRING
  if key.type_kind == spanner_graph_io_lib.TYPE_BYTES:
    return PARAM_TYPE_BYTES
  if key.type_kind == spanner_graph_io_lib.TYPE_BOOL:
    return PARAM_TYPE_BOOL
  return PARAM_TYPE_INT64


def _seed_value_sql(key: Column, parameter: str) -> str:
  """Gets the SQL value of key read from its seed query parameter."""
  return _SEED_PARAMETER_CASTS.get(key.type_kind, "{}").format(parameter)


def seed_parameter_types(keys: Sequence[Column]) -> dict[str, str]:
  """Gets the name and type of the query parameter of each seed key column."""
  if len(keys) == 1:
    return {SEED_IDS_PARAMETER: _seed_parameter_type(keys[0])}
  return {
      f"{SEED_IDS_PARAMETER}_{index}": _seed_parameter_type(key)
      for index, key in enumerate(keys)
  }


def seed_parameter_values(
    keys: Sequence[Column], seed_ids: Sequence[bytes]
) -> dict[str, list[Any]]:
  """Decodes the seed ids into the values of the seed query parameters."""
  column_types = [key.type_kind for key in keys]
  components = [
      spanner_graph_io_lib.decode_key(seed_id, column_types)
      for seed_id in seed_ids
  ]
  return {
      name: [component[index] for component in components]
      for index, name in enumerate(seed_parameter_types(keys))
  }


def _features_json_sql(
    table_alias: str, features: Mapping[str, FeatureSpec]
) -> str:
  """Builds a SQL JSON_OBJECT expression for a row's features."""
  if not features:
    return "CAST(NULL AS JSON)"
  return spanner_graph_io_lib.json_object_sql(
      {name: feature.column for name, feature in features.items()}, table_alias
  )


def _gql_features_json_sql(
    variable: str, features: Mapping[str, FeatureSpec]
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

  A key is carried through the CTEs in its native columns (<prefix>_id, or
  <prefix>_k<i> for a composite key), so that joins use the keys of the tables.
  Keys are encoded into BYTES ids in the final result only.
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
    spanner_graph_io_lib.validate_identifier(graph_name)
    self._tables = tables
    self._plan = plan
    self._engine = engine
    self._graph_name = graph_name
    self._debug_sampling = debug_sampling
    self._seed_keys = tables.node_tables[plan.root.nodeset].keys
    self._seed_columns = _key_columns("seed", self._seed_keys)
    self._num_hops = 0
    self._ctes: list[str] = []
    self._node_queries: dict[str, list[str]] = collections.defaultdict(list)
    self._edge_queries: dict[str, list[str]] = collections.defaultdict(list)

  def generate(self) -> str:
    """Builds the parameterized sampling query."""
    root_nodeset = self._plan.root.nodeset
    root_columns = _key_columns("n0", self._seed_keys)
    node_columns = _key_columns("node", self._seed_keys)
    self._ctes.append(self._source_node_cte())
    self._node_queries[root_nodeset].append(
        f"SELECT {_select_sql('s', self._seed_columns)},"
        f" {_select_sql('s', root_columns, node_columns)} FROM"
        f" {_SOURCE_NODE_CTE} AS s"
    )
    self._visit(
        self._plan.root,
        parent_cte=_SOURCE_NODE_CTE,
        parent_columns=root_columns,
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

  def _source_node_cte(self) -> str:
    """Builds the CTE of the seed nodes found in their node table."""
    root_table = self._tables.node_tables[self._plan.root.nodeset]
    root_keys = [key.sql("n") for key in self._seed_keys]
    select = ", ".join(
        f"{root_key} AS {column}"
        for root_key, column in zip(
            root_keys * 2,
            self._seed_columns + _key_columns("n0", self._seed_keys),
        )
    )
    parameters = list(seed_parameter_types(self._seed_keys))
    has_cast = any(
        key.type_kind in _SEED_PARAMETER_CASTS for key in self._seed_keys
    )
    if len(parameters) == 1 and not has_cast:
      return (
          f"  {_SOURCE_NODE_CTE} AS (\n"
          f"    SELECT {select}\n"
          f"    FROM `{root_table.table}` AS n\n"
          f"    WHERE {root_keys[0]} IN UNNEST(@{parameters[0]})\n"
          "  )"
      )
    # The seed parameters are parallel arrays, with one array per key column.
    unnests = " INNER JOIN ".join(
        f"UNNEST(@{parameter}) AS k{index} WITH OFFSET AS o{index}"
        + (f" ON o{index} = o0" if index else "")
        for index, parameter in enumerate(parameters)
    )
    seed_columns = ", ".join(f"k{index}" for index in range(len(parameters)))
    join = " AND ".join(
        f"{root_key} = {_seed_value_sql(key, f's.k{index}')}"
        for index, (root_key, key) in enumerate(zip(root_keys, self._seed_keys))
    )
    return (
        f"  {_SOURCE_NODE_CTE} AS (\n"
        f"    SELECT {select}\n"
        f"    FROM (SELECT DISTINCT {seed_columns} FROM {unnests}) AS s\n"
        f"    INNER JOIN `{root_table.table}` AS n ON {join}\n"
        "  )"
    )

  def _visit(
      self,
      node: config_lib.PlanNode,
      parent_cte: str,
      parent_columns: Sequence[str],
  ) -> None:
    """Appends one hop CTE per child of node, recursively."""
    parent_table = self._tables.node_tables[node.nodeset]
    hop_parent_columns = _key_columns("parent", parent_table.keys)
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
      child_keys = self._tables.node_tables[edge.node.nodeset].keys
      child_columns = _key_columns(f"n{self._num_hops}", child_keys)
      if self._engine == QueryEngine.SQL:
        neighbors, child_ids = self._sql_neighbors(edge, parent_columns)
      else:
        neighbors, child_ids = self._gql_neighbors(node, edge, parent_columns)
      child_key = spanner_graph_io_lib.key_bytes_sql(child_keys, child_ids)
      if self._debug_sampling:
        order = child_key
      else:
        # Hash with @sampling_seed instead of GENERATE_UUID() so every reference
        # to a CTE sees the same sampled neighbors. The hop makes a node
        # reached by several hops sample different neighbors in each of them.
        seed_key = _key_bytes_sql(self._seed_keys, "p", "seed")
        parent_key = spanner_graph_io_lib.key_bytes_sql(
            parent_table.keys, [f"p.{column}" for column in parent_columns]
        )
        order = (
            f"FARM_FINGERPRINT(CONCAT(@{SAMPLING_SEED_PARAMETER},"
            f" b'{self._num_hops}:', {seed_key}, {parent_key}, {child_key}))"
        )
      parents = _select_sql("q", self._seed_columns + list(parent_columns))
      select = ", ".join([
          _select_sql("p", self._seed_columns),
          _select_sql("p", parent_columns, hop_parent_columns),
          _select_sql("c", _key_columns("child", child_keys), child_columns),
          "c.edge_json",
      ])
      self._ctes.append(
          f"  {cte_name} AS (\n"
          f"    SELECT {select}\n"
          f"    FROM (SELECT DISTINCT {parents} FROM {parent_cte} AS q) AS p,\n"
          "    UNNEST(ARRAY(\n"
          f"{neighbors}\n"
          # The d format rejects non-integer values.
          f"      ORDER BY {order} LIMIT {edge.hop_width:d}\n"
          "    )) AS c\n"
          "  )"
      )

      seeds = _select_sql("h", self._seed_columns)
      nodes = _select_sql("h", child_columns, _key_columns("node", child_keys))
      self._node_queries[edge.node.nodeset].append(
          f"SELECT {seeds}, {nodes} FROM {cte_name} AS h"
      )
      if edge.reversed:
        source_keys, source_columns = child_keys, child_columns
        target_keys, target_columns = parent_table.keys, hop_parent_columns
      else:
        source_keys, source_columns = parent_table.keys, hop_parent_columns
        target_keys, target_columns = child_keys, child_columns
      sources = _select_sql(
          "h", source_columns, _key_columns("source", source_keys)
      )
      targets = _select_sql(
          "h", target_columns, _key_columns("target", target_keys)
      )
      self._edge_queries[edge.edgeset].append(
          f"SELECT {seeds}, {sources}, {targets}, h.edge_json FROM"
          f" {cte_name} AS h"
      )
      self._visit(edge.node, parent_cte=cte_name, parent_columns=child_columns)

  def _sql_neighbors(
      self, edge: config_lib.PlanEdge, parent_columns: Sequence[str]
  ) -> tuple[str, list[str]]:
    """Gets the SQL query of the neighbors of a parent and their key columns."""
    edge_table = self._tables.edge_tables[edge.edgeset]
    if edge.reversed:
      join_columns, child_columns = edge_table.target, edge_table.source
    else:
      join_columns, child_columns = edge_table.source, edge_table.target
    child_ids = [column.sql("e") for column in child_columns]
    children = ", ".join(
        f"{child_id} AS {name}"
        for child_id, name in zip(
            child_ids, _key_columns("child", child_columns)
        )
    )
    edge_json = _features_json_sql("e", edge_table.features)
    conditions = [
        f"{join_column.sql('e')} = p.{parent_column}"
        for join_column, parent_column in zip(join_columns, parent_columns)
    ] + [f"{child_id} IS NOT NULL" for child_id in child_ids]
    query = "\n".join([
        f"      SELECT AS STRUCT {children}, {edge_json} AS edge_json",
        f"      FROM `{edge_table.table}` AS e",
        f"      WHERE {' AND '.join(conditions)}",
    ])
    return query, child_ids

  def _gql_neighbors(
      self,
      node: config_lib.PlanNode,
      edge: config_lib.PlanEdge,
      parent_columns: Sequence[str],
  ) -> tuple[str, list[str]]:
    """Gets the GQL query of the neighbors of a parent and their key columns."""
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
    for nodeset_name, node_table in (
        (node.nodeset, parent_table),
        (edge.node.nodeset, child_table),
    ):
      if None in node_table.key_properties:
        raise NotImplementedError(
            f"Some key columns of {nodeset_name!r} are not properties of the"
            " Spanner Graph, and a GQL query cannot read them. Use"
            " query_engine SQL instead."
        )
    parent = f"(gp:`{parent_table.label}`)"
    child = f"(gc:`{child_table.label}`)"
    if edge.reversed:
      pattern = f"{parent}<-[ge:`{edge_table.label}`]-{child}"
    else:
      pattern = f"{parent}-[ge:`{edge_table.label}`]->{child}"
    child_columns = _key_columns("child", child_table.keys)
    conditions = " AND ".join(
        f"gp.`{key_property}` = p.{parent_column}"
        for key_property, parent_column in zip(
            parent_table.key_properties, parent_columns
        )
    )
    children = ", ".join(
        f"gc.`{key_property}` AS {child_column}"
        for key_property, child_column in zip(
            child_table.key_properties, child_columns
        )
    )
    edge_json = _gql_features_json_sql("ge", edge_table.features)
    child_ids = [f"g.{child_column}" for child_column in child_columns]
    query = "\n".join([
        f"      SELECT AS STRUCT {', '.join(child_ids)}, g.edge_json",
        f"      FROM GRAPH_TABLE(`{self._graph_name}`",
        f"        MATCH {pattern}",
        f"        WHERE {conditions}",
        f"        RETURN {children}, {edge_json} AS edge_json",
        "      ) AS g",
    ])
    return query, child_ids

  def _nodes_query(self, nodeset_name: str, queries: list[str]) -> str:
    """Builds the SELECT query returning the sampled nodes of nodeset_name."""
    node_table = self._tables.node_tables[nodeset_name]
    node_columns = _key_columns("node", node_table.keys)
    cte_name = f"_dgf_nodes_{nodeset_name}"
    union = "\n      UNION ALL\n      ".join(queries)
    columns = _select_sql("a", self._seed_columns + node_columns)
    self._ctes.append(
        f"  {cte_name} AS (\n"
        f"    SELECT DISTINCT {columns} FROM (\n"
        f"      {union}\n"
        "    ) AS a\n"
        "  )"
    )
    join = " AND ".join(
        f"u.{node_column} = {key.sql('n')}"
        for node_column, key in zip(node_columns, node_table.keys)
    )
    # The node table is joined even without features to read: An edge to a node
    # missing from its node table is then reported instead of creating a node.
    return f"""\
SELECT
  {_key_bytes_sql(self._seed_keys, 'u', 'seed')} AS seed_id,
  {_key_bytes_sql(node_table.keys, 'u', 'node')} AS node_id,
  '{_ELEMENT_TYPE_NODE}' AS element_type,
  '{nodeset_name}' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  {_features_json_sql('n', node_table.features)} AS properties_json
FROM {cte_name} AS u
INNER JOIN `{node_table.table}` AS n ON {join}"""

  def _edges_query(self, edgeset_name: str, queries: list[str]) -> str:
    """Builds the SELECT query returning the sampled edges of edgeset_name."""
    edge_table = self._tables.edge_tables[edgeset_name]
    union = "\n  UNION ALL\n  ".join(queries)
    return f"""\
SELECT
  {_key_bytes_sql(self._seed_keys, 'e', 'seed')} AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  '{_ELEMENT_TYPE_EDGE}' AS element_type,
  '{edgeset_name}' AS element_class,
  {_key_bytes_sql(edge_table.source, 'e', 'source')} AS source_id,
  {_key_bytes_sql(edge_table.target, 'e', 'target')} AS target_id,
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
    features: Mapping[str, FeatureSpec],
    output: dict[str, list[Any]],
) -> None:
  """Parses the properties of an element and appends them to output."""
  if not features:
    return
  if properties is None:
    raise ValueError("Expected JSON properties object, got None.")
  for feature_name, feature in features.items():
    output[feature_name].append(feature.parse_value(properties[feature_name]))


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
      key_array = spanner_graph_io_lib.pack_bytes_ids(list(nodeset.node_index))
      features = {}
      for feature_name in nodeset_schema.features:
        if feature_name == node_table.primary_key:
          features[feature_name] = key_array
        else:
          features[feature_name] = spanner_graph_io_lib.pack_feature(
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
              feature_name: spanner_graph_io_lib.pack_feature(
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
      column_spanner_types: Mapping[tuple[str, str], str],
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
    self._tables = resolve_graph_tables(metadata, schema, column_spanner_types)
    self._query = generate_query(
        self._tables, plan, query_engine, graph, debug_sampling
    )
    self._seed_keys = self._tables.node_tables[plan.root.nodeset].keys
    # Type (PARAM_TYPE_*) of the elements of each seed query parameter.
    self._seed_parameter_types = seed_parameter_types(self._seed_keys)

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
    rows = self._execute(
        seed_parameters=seed_parameter_values(self._seed_keys, seed_bytes),
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
      seed_parameters: Mapping[str, list[Any]],
      sampling_seed: bytes | None,
  ) -> Iterable[Sequence[Any]]:
    """Runs the sampling query and returns its rows.

    Args:
      seed_parameters: Values of the seed query parameters, which are arrays of
        the types in self._seed_parameter_types.
      sampling_seed: Value of the @sampling_seed BYTES parameter, or None when
        debug_sampling is enabled.

    Returns:
      The rows (seed_id, node_id, element_type, element_class, source_id,
      target_id, properties) of the query. The ids are raw bytes or None. The
      properties are a parsed JSON object or None.
    """
