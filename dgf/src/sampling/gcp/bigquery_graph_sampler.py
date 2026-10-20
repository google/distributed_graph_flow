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

"""Samples subgraphs from a BigQuery Graph with a single SQL query.

The DGF `GraphSchema` defines which nodesets, edgesets and features are
sampled and how values are parsed. The BigQuery Graph metadata is only used to
locate the data: table paths, key and endpoint columns, and the column behind
each feature (i.e., the BigQuery property with the same name as the feature).

How a `SamplingPlan` is mapped to SQL:
A `SamplingPlan` is a tree rooted at the seed nodeset where each tree edge
`u -> v` traverses an edgeset (forward, or backward if `reversed`) and keeps up
to `hop_width` neighbors per parent node. The tree is compiled into Common
Table Expressions (CTEs):
- `source_node` selects the seed rows (`@seed_ids`) and tags each one with its
  own `seed_id`, so all the seeds of a batch are sampled by the same query.
- Each tree edge (in pre-order) becomes a CTE `hop_<i>` that joins the
  distinct `(seed_id, parent)` pairs of the parent CTE with the edge table and
  keeps up to `hop_width` random edges per `(seed_id, parent)` with
  `QUALIFY ROW_NUMBER() OVER (PARTITION BY seed_id, parent ORDER BY <hash>)`
  where `<hash>` hashes the edge with a random `@sampling_seed` drawn for each
  query (deterministic within a query, as BigQuery re-evaluates CTEs).
  Hops only carry node keys (and the edge features), not node features.
- For each nodeset, a `nodes_<nodeset>` CTE deduplicates the `(seed_id, node)`
  pairs of all the CTEs producing this nodeset. The node table is then joined
  once to fetch the node features.
- The nodes and the deduplicated edges of all the nodesets and edgesets are
  returned as a single stream of rows `(seed_id, node_id, element_type,
  element_class, source_id, target_id, properties_json)` where ids are BYTES
  and features are serialized with `TO_JSON_STRING`.

Structure of this file:
1. Resolution of the schema against the BigQuery metadata (`_GraphTables`).
2. `QueryGenerator`: Compiles a `SamplingPlan` into the SQL query.
3. Subgraph assembly: Converts the query rows into one `InMemoryGraph` per
   seed, with the seed node at index 0 of the seed nodeset.
4. `BigQueryGraphSampler` and `create_bigquery_graph_sampler`.

Limitations:
- Node primary keys must be a single STRING or BYTES column. Composite and
  integer keys are not supported.
- Supported BigQuery types: BOOL, INT64, FLOAT64, STRING, BYTES, TIMESTAMP and
  1D arrays of those.
- Parallel edges (same source and target) are merged into a single edge, as in
  the in-memory sampler. Its features are those of one of the parallel edges.
- A node reached several times at the same step of the plan is expanded once
  (i.e., `multi_visit=False`). Sampling with replacement and temporal sampling
  are not supported.
- Each `sample()` call runs a BigQuery query job (a few seconds of latency),
  which suits online inference or small batches rather than training.
"""

import base64
import collections
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json
import random
from typing import Any

from dgf.src.analyse import schema as schema_analyse_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io.gcp import bigquery_graph as bigquery_graph_lib
from dgf.src.io.gcp import bigquery_graph_metadata as bigquery_graph_metadata_lib
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.sampling import config as config_lib
from dgf.src.util import log
from google.cloud import bigquery
import numpy as np

# Columns of the rows returned by the sampling query.
_SEED_ID_COLUMN = "seed_id"
_NODE_ID_COLUMN = "node_id"
_ELEMENT_TYPE_COLUMN = "element_type"
_ELEMENT_CLASS_COLUMN = "element_class"
_SOURCE_ID_COLUMN = "source_id"
_TARGET_ID_COLUMN = "target_id"
_PROPERTIES_JSON_COLUMN = "properties_json"

# Intermediate columns of the hop CTEs.
_PARENT_ID_COLUMN = "parent_id"
_EDGE_JSON_COLUMN = "edge_json"

_ELEMENT_TYPE_NODE = "node"
_ELEMENT_TYPE_EDGE = "edge"
_SEED_IDS_PARAMETER = "seed_ids"
_SAMPLING_SEED_PARAMETER = "sampling_seed"
_SOURCE_NODE_CTE = "source_node"

_BQ_TYPE_ARRAY = "ARRAY"
_BQ_TYPE_BOOL = "BOOL"
_BQ_TYPE_BYTES = "BYTES"
_BQ_TYPE_FLOAT64 = "FLOAT64"
_BQ_TYPE_INT64 = "INT64"
_BQ_TYPE_STRING = "STRING"
_BQ_TYPE_TIMESTAMP = "TIMESTAMP"

# BigQuery column types that can back each DGF feature format.
_COMPATIBLE_BQ_TYPES: dict[schema_lib.FeatureFormat, frozenset[str]] = {
    schema_lib.FeatureFormat.BOOL: frozenset({_BQ_TYPE_BOOL}),
    schema_lib.FeatureFormat.BYTES: frozenset({_BQ_TYPE_STRING, _BQ_TYPE_BYTES}),
    schema_lib.FeatureFormat.FLOAT_32: frozenset({_BQ_TYPE_FLOAT64}),
    schema_lib.FeatureFormat.FLOAT_64: frozenset({_BQ_TYPE_FLOAT64}),
    schema_lib.FeatureFormat.INTEGER_32: frozenset({_BQ_TYPE_INT64}),
    schema_lib.FeatureFormat.INTEGER_64: frozenset(
        {_BQ_TYPE_INT64, _BQ_TYPE_TIMESTAMP}
    ),
}
_KEY_BQ_TYPES = frozenset({_BQ_TYPE_STRING, _BQ_TYPE_BYTES})
_SQL_NULL_BYTES = "CAST(NULL AS BYTES)"
_SQL_NULL_STRING = "CAST(NULL AS STRING)"


@dataclasses.dataclass(frozen=True)
class _Column:
  """BigQuery column backing a DGF feature or key."""

  name: str
  # Type of the column, or of its elements if `is_array`.
  type_kind: str
  is_array: bool = False

  def sql(self, table_alias: str) -> str:
    """Returns the SQL reference to this column in `table_alias`."""
    return f"{table_alias}.`{self.name}`"

  def to_bytes_sql(self, expression: str) -> str:
    """Converts `expression`, a value of this key column, to BYTES."""
    if self.type_kind == _BQ_TYPE_STRING:
      return f"CAST({expression} AS BYTES)"
    return expression


@dataclasses.dataclass(frozen=True)
class _NodeTable:
  """BigQuery table of a nodeset."""

  table_ref: str
  key: _Column
  features: dict[str, _Column]


@dataclasses.dataclass(frozen=True)
class _EdgeTable:
  """BigQuery table of an edgeset."""

  table_ref: str
  source: _Column
  target: _Column
  features: dict[str, _Column]


@dataclasses.dataclass(frozen=True)
class _GraphTables:
  """BigQuery tables of all the nodesets and edgesets of a DGF schema."""

  node_tables: dict[str, _NodeTable]
  edge_tables: dict[str, _EdgeTable]


def _table_ref(data_source: bigquery_graph_metadata_lib.DataSourceTable) -> str:
  """Returns the `project.dataset.table` reference of a BigQuery table."""
  gcp_common_lib.validate_identifier(data_source.dataset_id)
  gcp_common_lib.validate_identifier(data_source.table_id)
  return (
      f"{data_source.project_id}.{data_source.dataset_id}"
      f".{data_source.table_id}"
  )


def _properties_by_name(
    label_and_properties: Sequence[
        bigquery_graph_metadata_lib.LabelAndProperties
    ],
) -> dict[str, bigquery_graph_metadata_lib.Property]:
  """Indexes the properties of all the labels of a BigQuery table by name."""
  properties = {}
  for label_and_property in label_and_properties:
    for property_ in label_and_property.properties or ():
      properties[property_.name] = property_
  return properties


def _feature_column(
    element_name: str,
    feature_name: str,
    feature_schema: schema_lib.FeatureSchema,
    properties: Mapping[str, bigquery_graph_metadata_lib.Property],
) -> _Column:
  """Returns the column of a feature, checking it matches the feature schema."""
  if feature_name not in properties:
    raise ValueError(
        f"Feature {feature_name!r} of {element_name!r} is not a property of"
        f" the BigQuery Graph. Available properties: {list(properties)}"
    )
  property_ = properties[feature_name]
  # Only plain columns (possibly renamed) are supported as property expressions.
  gcp_common_lib.validate_identifier(property_.expression)

  data_type = property_.data_type
  is_array = data_type.type_kind == _BQ_TYPE_ARRAY
  if is_array:
    if data_type.array_element_type is None:
      raise ValueError(f"Missing array element type for {feature_name!r}")
    type_kind = data_type.array_element_type.type_kind
  else:
    type_kind = data_type.type_kind

  shape = feature_schema.shape or ()
  if len(shape) > 1:
    raise NotImplementedError(
        f"Feature {feature_name!r} of {element_name!r} has shape {shape}."
        " Only scalar and 1D features are supported."
    )
  if (
      type_kind not in _COMPATIBLE_BQ_TYPES[feature_schema.format]
      or is_array != bool(shape)
  ):
    raise ValueError(
        f"Feature {feature_name!r} of {element_name!r} (format"
        f" {feature_schema.format}, shape {shape}) is not compatible with the"
        f" BigQuery type {data_type.resolved_data_type()}."
    )
  return _Column(name=property_.expression, type_kind=type_kind, is_array=is_array)


def _resolve_node_table(
    nodeset_name: str,
    nodeset_schema: schema_lib.NodeSchema,
    node_table: bigquery_graph_metadata_lib.NodeTable,
) -> _NodeTable:
  """Resolves the BigQuery columns of a nodeset."""
  if len(node_table.key_columns) != 1:
    raise NotImplementedError(
        f"Nodeset {nodeset_name!r} has a composite key"
        f" {node_table.key_columns}. Only single-column keys are supported."
    )
  properties = _properties_by_name(node_table.label_and_properties)
  features = {
      feature_name: _feature_column(
          nodeset_name, feature_name, feature_schema, properties
      )
      for feature_name, feature_schema in nodeset_schema.features.items()
  }

  primary_key = schema_analyse_lib.primary_feature(nodeset_name, nodeset_schema)
  key = features[primary_key]
  if key.name != node_table.key_columns[0]:
    raise ValueError(
        f"The primary key feature {primary_key!r} of {nodeset_name!r} is"
        f" stored in column {key.name!r}, but the key of the BigQuery node"
        f" table is {node_table.key_columns[0]!r}."
    )
  if key.type_kind not in _KEY_BQ_TYPES or key.is_array:
    raise ValueError(
        f"The primary key {primary_key!r} of {nodeset_name!r} has type"
        f" {key.type_kind}. Only STRING and BYTES primary keys are supported."
    )
  return _NodeTable(
      table_ref=_table_ref(node_table.data_source_table),
      key=key,
      features=features,
  )


def _resolve_endpoint(
    edgeset_name: str,
    expected_nodeset: str,
    reference: bigquery_graph_metadata_lib.NodeReference,
    node_tables: Mapping[str, _NodeTable],
) -> _Column:
  """Returns the edge table column referencing the key of `expected_nodeset`."""
  if reference.node_table != expected_nodeset:
    raise ValueError(
        f"Edgeset {edgeset_name!r} references nodeset"
        f" {reference.node_table!r} in BigQuery but {expected_nodeset!r} in the"
        " schema."
    )
  node_key = node_tables[expected_nodeset].key
  if reference.node_table_columns != [node_key.name]:
    raise ValueError(
        f"Edgeset {edgeset_name!r} references columns"
        f" {reference.node_table_columns} of {expected_nodeset!r} instead of"
        f" its key {node_key.name!r}."
    )
  (column,) = reference.edge_table_columns
  gcp_common_lib.validate_identifier(column)
  return _Column(name=column, type_kind=node_key.type_kind)


def _resolve_graph_tables(
    metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata,
    schema: schema_lib.GraphSchema,
) -> _GraphTables:
  """Resolves the BigQuery tables and columns of all the schema elements."""
  bq_node_tables = {table.name: table for table in metadata.node_tables}
  bq_edge_tables = {table.name: table for table in metadata.edge_tables}

  node_tables = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    gcp_common_lib.validate_identifier(nodeset_name)
    if nodeset_name not in bq_node_tables:
      raise ValueError(
          f"Nodeset {nodeset_name!r} not found in the BigQuery Graph."
          f" Available node tables: {list(bq_node_tables)}"
      )
    node_tables[nodeset_name] = _resolve_node_table(
        nodeset_name, nodeset_schema, bq_node_tables[nodeset_name]
    )

  edge_tables = {}
  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    gcp_common_lib.validate_identifier(edgeset_name)
    if edgeset_name not in bq_edge_tables:
      raise ValueError(
          f"Edgeset {edgeset_name!r} not found in the BigQuery Graph."
          f" Available edge tables: {list(bq_edge_tables)}"
      )
    edge_table = bq_edge_tables[edgeset_name]
    properties = _properties_by_name(edge_table.label_and_properties)
    edge_tables[edgeset_name] = _EdgeTable(
        table_ref=_table_ref(edge_table.data_source_table),
        source=_resolve_endpoint(
            edgeset_name,
            edgeset_schema.source,
            edge_table.source_node_reference,
            node_tables,
        ),
        target=_resolve_endpoint(
            edgeset_name,
            edgeset_schema.target,
            edge_table.destination_node_reference,
            node_tables,
        ),
        features={
            feature_name: _feature_column(
                edgeset_name, feature_name, feature_schema, properties
            )
            for feature_name, feature_schema in edgeset_schema.features.items()
        },
    )
  return _GraphTables(node_tables=node_tables, edge_tables=edge_tables)


def _features_json_sql(table_alias: str, features: Mapping[str, _Column]) -> str:
  """Returns a SQL expression serializing the features of a row to JSON."""
  if not features:
    return _SQL_NULL_STRING
  fields = ", ".join(
      f"{column.sql(table_alias)} AS `{feature_name}`"
      for feature_name, column in features.items()
  )
  return f"TO_JSON_STRING(STRUCT({fields}))"


class QueryGenerator:
  """Compiles a `SamplingPlan` into a parameterized BigQuery SQL query.

  Each instance generates a single query.
  """

  def __init__(
      self,
      tables: _GraphTables,
      plan: config_lib.SamplingPlan,
      debug_sampling: bool = False,
  ):
    self._tables = tables
    self._plan = plan
    self._debug_sampling = debug_sampling
    self._num_hops = 0
    self._ctes: list[str] = []
    # (seed_id, node_id) queries for each nodeset.
    self._node_queries: dict[str, list[str]] = collections.defaultdict(list)
    # (seed_id, source_id, target_id, edge_json) queries for each edgeset.
    self._edge_queries: dict[str, list[str]] = collections.defaultdict(list)

  def generate(self) -> str:
    """Returns the SQL query. The seeds are the `@seed_ids` parameter."""
    root_nodeset = self._plan.root.nodeset
    root_table = self._tables.node_tables[root_nodeset]
    root_key = root_table.key.sql("n0")
    self._ctes.append(
        f"  {_SOURCE_NODE_CTE} AS (\n"
        f"    SELECT {root_key} AS {_SEED_ID_COLUMN}, {root_key} AS n0_id\n"
        f"    FROM `{root_table.table_ref}` AS n0\n"
        f"    WHERE {root_key} IN UNNEST(@{_SEED_IDS_PARAMETER})\n"
        "  )"
    )
    self._node_queries[root_nodeset].append(
        f"SELECT {_SEED_ID_COLUMN}, n0_id AS {_NODE_ID_COLUMN}"
        f" FROM {_SOURCE_NODE_CTE}"
    )
    self._visit(self._plan.root, parent_cte=_SOURCE_NODE_CTE, parent_column="n0_id")

    seed_key = root_table.key
    results = [
        self._nodes_query(nodeset_name, queries, seed_key)
        for nodeset_name, queries in self._node_queries.items()
    ] + [
        self._edges_query(edgeset_name, queries, seed_key)
        for edgeset_name, queries in self._edge_queries.items()
    ]
    return "WITH\n" + ",\n".join(self._ctes) + "\n" + "\nUNION ALL\n".join(results)

  def _visit(
      self, node: config_lib.PlanNode, parent_cte: str, parent_column: str
  ) -> None:
    """Emits one `hop_<i>` CTE per child of `node`, recursively."""
    for edge in node.children:
      self._num_hops += 1
      cte_name = f"hop_{self._num_hops}"
      child_column = f"n{self._num_hops}_id"
      edge_table = self._tables.edge_tables[edge.edgeset]
      if edge.reversed:
        join_column, child_key = edge_table.target, edge_table.source
      else:
        join_column, child_key = edge_table.source, edge_table.target

      qualify = ""
      if edge.hop_width > 0:
        if self._debug_sampling:
          order = child_key.sql("e")
        else:
          # Not `RAND()`: BigQuery re-evaluates a CTE each time it is
          # referenced, and all evaluations must sample the same neighbors.
          order = (
              "FARM_FINGERPRINT(TO_JSON_STRING(STRUCT("
              f"@{_SAMPLING_SEED_PARAMETER}, p.{_SEED_ID_COLUMN},"
              f" p.{parent_column}, {child_key.sql('e')})))"
          )
        qualify = (
            "\n    QUALIFY ROW_NUMBER() OVER (PARTITION BY"
            f" p.{_SEED_ID_COLUMN}, p.{parent_column} ORDER BY {order})"
            f" <= {edge.hop_width}"
        )
      edge_json = _features_json_sql("e", edge_table.features)
      self._ctes.append(
          f"  {cte_name} AS (\n"
          f"    SELECT p.{_SEED_ID_COLUMN}, p.{parent_column} AS"
          f" {_PARENT_ID_COLUMN}, {child_key.sql('e')} AS {child_column},"
          f" {edge_json} AS {_EDGE_JSON_COLUMN}\n"
          f"    FROM (SELECT DISTINCT {_SEED_ID_COLUMN}, {parent_column} FROM"
          f" {parent_cte}) AS p\n"
          f"    INNER JOIN `{edge_table.table_ref}` AS e ON p.{parent_column} ="
          f" {join_column.sql('e')}{qualify}\n"
          "  )"
      )

      self._node_queries[edge.node.nodeset].append(
          f"SELECT {_SEED_ID_COLUMN}, {child_column} AS {_NODE_ID_COLUMN} FROM"
          f" {cte_name}"
      )
      if edge.reversed:
        source_column, target_column = child_column, _PARENT_ID_COLUMN
      else:
        source_column, target_column = _PARENT_ID_COLUMN, child_column
      self._edge_queries[edge.edgeset].append(
          f"SELECT {_SEED_ID_COLUMN}, {source_column} AS {_SOURCE_ID_COLUMN},"
          f" {target_column} AS {_TARGET_ID_COLUMN}, {_EDGE_JSON_COLUMN} FROM"
          f" {cte_name}"
      )
      self._visit(edge.node, parent_cte=cte_name, parent_column=child_column)

  def _nodes_query(
      self, nodeset_name: str, queries: list[str], seed_key: _Column
  ) -> str:
    """Returns the result rows of the sampled nodes of a nodeset."""
    node_table = self._tables.node_tables[nodeset_name]
    cte_name = f"nodes_{nodeset_name}"
    union = "\n      UNION ALL\n      ".join(queries)
    self._ctes.append(
        f"  {cte_name} AS (\n"
        f"    SELECT DISTINCT {_SEED_ID_COLUMN}, {_NODE_ID_COLUMN} FROM (\n"
        f"      {union}\n"
        "    )\n"
        "  )"
    )
    return (
        "SELECT\n"
        f"  {seed_key.to_bytes_sql(f'u.{_SEED_ID_COLUMN}')} AS"
        f" {_SEED_ID_COLUMN},\n"
        f"  {node_table.key.to_bytes_sql(f'u.{_NODE_ID_COLUMN}')} AS"
        f" {_NODE_ID_COLUMN},\n"
        f"  '{_ELEMENT_TYPE_NODE}' AS {_ELEMENT_TYPE_COLUMN},\n"
        f"  '{nodeset_name}' AS {_ELEMENT_CLASS_COLUMN},\n"
        f"  {_SQL_NULL_BYTES} AS {_SOURCE_ID_COLUMN},\n"
        f"  {_SQL_NULL_BYTES} AS {_TARGET_ID_COLUMN},\n"
        f"  {_features_json_sql('n', node_table.features)} AS"
        f" {_PROPERTIES_JSON_COLUMN}\n"
        f"FROM {cte_name} AS u\n"
        f"INNER JOIN `{node_table.table_ref}` AS n ON u.{_NODE_ID_COLUMN} ="
        f" {node_table.key.sql('n')}"
    )

  def _edges_query(
      self, edgeset_name: str, queries: list[str], seed_key: _Column
  ) -> str:
    """Returns the result rows of the sampled (deduplicated) edges."""
    edge_table = self._tables.edge_tables[edgeset_name]
    union = "\n    UNION ALL\n    ".join(queries)
    return (
        "SELECT\n"
        f"  {seed_key.to_bytes_sql(f'e.{_SEED_ID_COLUMN}')} AS"
        f" {_SEED_ID_COLUMN},\n"
        f"  {_SQL_NULL_BYTES} AS {_NODE_ID_COLUMN},\n"
        f"  '{_ELEMENT_TYPE_EDGE}' AS {_ELEMENT_TYPE_COLUMN},\n"
        f"  '{edgeset_name}' AS {_ELEMENT_CLASS_COLUMN},\n"
        f"  {edge_table.source.to_bytes_sql(f'e.{_SOURCE_ID_COLUMN}')} AS"
        f" {_SOURCE_ID_COLUMN},\n"
        f"  {edge_table.target.to_bytes_sql(f'e.{_TARGET_ID_COLUMN}')} AS"
        f" {_TARGET_ID_COLUMN},\n"
        f"  e.{_EDGE_JSON_COLUMN} AS {_PROPERTIES_JSON_COLUMN}\n"
        "FROM (\n"
        # Parallel edges are merged, keeping the features of one of them.
        f"  SELECT {_SEED_ID_COLUMN}, {_SOURCE_ID_COLUMN}, {_TARGET_ID_COLUMN},"
        f" MIN({_EDGE_JSON_COLUMN}) AS {_EDGE_JSON_COLUMN} FROM (\n"
        f"    {union}\n"
        "  )\n"
        f"  GROUP BY {_SEED_ID_COLUMN}, {_SOURCE_ID_COLUMN}, {_TARGET_ID_COLUMN}\n"
        ") AS e"
    )


def _parse_json_value(value: Any, column: _Column) -> Any:
  """Converts a `TO_JSON_STRING` value of `column` to a DGF python value."""
  if value is None:
    return None
  if column.is_array:
    return [_parse_json_scalar(item, column.type_kind) for item in value]
  return _parse_json_scalar(value, column.type_kind)


def _parse_json_scalar(value: Any, type_kind: str) -> Any:
  """Converts a non-null `TO_JSON_STRING` scalar to a DGF python value."""
  if type_kind == _BQ_TYPE_BYTES:
    return base64.b64decode(value)  # TO_JSON_STRING encodes BYTES in base64.
  if type_kind == _BQ_TYPE_STRING:
    return value.encode("utf-8")
  return value


def _parse_features(
    properties_json: str,
    feature_schemas: schema_lib.FeatureSetSchema,
    columns: Mapping[str, _Column],
) -> dict[str, np.ndarray]:
  """Parses the features of a row serialized by `_features_json_sql`."""
  values = json.loads(properties_json)
  features = {}
  for feature_name, feature_schema in feature_schemas.items():
    value = gcp_common_lib.parse_property_value_to_feature(
        feature_schema,
        _parse_json_value(values[feature_name], columns[feature_name]),
    )
    features[feature_name] = np.asarray(
        value,
        dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
            feature_schema.format
        ],
    )
  return features


@dataclasses.dataclass
class _RawNodeSet:
  """Nodes of a nodeset of a subgraph being assembled."""

  node_index: dict[bytes, int] = dataclasses.field(default_factory=dict)
  features: dict[str, list[np.ndarray]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )


@dataclasses.dataclass
class _RawEdgeSet:
  """Edges of an edgeset of a subgraph being assembled."""

  sources: list[int] = dataclasses.field(default_factory=list)
  targets: list[int] = dataclasses.field(default_factory=list)
  features: dict[str, list[np.ndarray]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )


class _SubgraphBuilder:
  """Assembles the subgraph of a seed from the rows of the sampling query."""

  def __init__(self, schema: schema_lib.GraphSchema, tables: _GraphTables):
    self._schema = schema
    self._tables = tables
    self._node_sets = collections.defaultdict(_RawNodeSet)
    self._edge_sets = collections.defaultdict(_RawEdgeSet)

  def add_node(
      self, nodeset_name: str, node_id: bytes, properties_json: str
  ) -> None:
    """Adds a node. Fails if the node was already added."""
    nodeset = self._node_sets[nodeset_name]
    if node_id in nodeset.node_index:
      raise ValueError(
          f"Duplicate node {node_id!r} in nodeset {nodeset_name!r}. The key"
          " of the BigQuery node table is not unique."
      )
    nodeset.node_index[node_id] = len(nodeset.node_index)
    features = _parse_features(
        properties_json,
        self._schema.node_sets[nodeset_name].features,
        self._tables.node_tables[nodeset_name].features,
    )
    for feature_name, feature_value in features.items():
      nodeset.features[feature_name].append(feature_value)

  def add_edge(
      self,
      edgeset_name: str,
      source_id: bytes,
      target_id: bytes,
      properties_json: str | None,
  ) -> None:
    """Adds an edge. Its endpoints must have been added already."""
    edgeset_schema = self._schema.edge_sets[edgeset_name]
    edgeset = self._edge_sets[edgeset_name]
    edgeset.sources.append(
        self._node_index(edgeset_name, edgeset_schema.source, source_id)
    )
    edgeset.targets.append(
        self._node_index(edgeset_name, edgeset_schema.target, target_id)
    )
    if edgeset_schema.features:
      assert properties_json is not None
      features = _parse_features(
          properties_json,
          edgeset_schema.features,
          self._tables.edge_tables[edgeset_name].features,
      )
      for feature_name, feature_value in features.items():
        edgeset.features[feature_name].append(feature_value)

  def _node_index(
      self, edgeset_name: str, nodeset_name: str, node_id: bytes
  ) -> int:
    """Returns the index of an edge endpoint in the subgraph."""
    node_index = self._node_sets[nodeset_name].node_index
    if node_id not in node_index:
      raise ValueError(
          f"Edge of {edgeset_name!r} references node {node_id!r} which is"
          f" missing from the BigQuery node table of {nodeset_name!r}."
      )
    return node_index[node_id]

  def build(self) -> in_memory_graph_lib.InMemoryGraph:
    """Returns the assembled subgraph."""
    node_sets = {}
    for nodeset_name, nodeset_schema in self._schema.node_sets.items():
      nodeset = self._node_sets[nodeset_name]
      node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
          num_nodes=len(nodeset.node_index),
          features={
              feature_name: gcp_common_lib.stack_feature_values(
                  feature_schema, nodeset.features[feature_name]
              )
              for feature_name, feature_schema in nodeset_schema.features.items()
          },
      )
    edge_sets = {}
    for edgeset_name, edgeset_schema in self._schema.edge_sets.items():
      edgeset = self._edge_sets[edgeset_name]
      edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
          adjacency=np.array(
              [edgeset.sources, edgeset.targets], dtype=np.int64
          ).reshape(2, -1),
          features={
              feature_name: gcp_common_lib.stack_feature_values(
                  feature_schema, edgeset.features[feature_name]
              )
              for feature_name, feature_schema in edgeset_schema.features.items()
          },
      )
    return in_memory_graph_lib.InMemoryGraph(
        node_sets=node_sets, edge_sets=edge_sets
    )


def _rows_to_in_memory_graphs(
    rows: Iterable[Mapping[str, Any]],
    schema: schema_lib.GraphSchema,
    tables: _GraphTables,
    seed_nodeset: str,
    seed_ids: Sequence[bytes],
) -> list[in_memory_graph_lib.InMemoryGraph]:
  """Converts the rows of the sampling query into one subgraph per seed."""
  node_rows = collections.defaultdict(list)
  edge_rows = collections.defaultdict(list)
  for row in rows:
    element_type = row[_ELEMENT_TYPE_COLUMN]
    if element_type == _ELEMENT_TYPE_NODE:
      node_rows[row[_SEED_ID_COLUMN]].append(row)
    elif element_type == _ELEMENT_TYPE_EDGE:
      edge_rows[row[_SEED_ID_COLUMN]].append(row)
    else:
      raise ValueError(f"Unknown element type: {element_type!r}")

  missing_seeds = [seed_id for seed_id in seed_ids if seed_id not in node_rows]
  if missing_seeds:
    raise ValueError(
        f"Seed nodes {missing_seeds!r} not found in nodeset {seed_nodeset!r}."
    )

  graphs = []
  for seed_id in seed_ids:
    builder = _SubgraphBuilder(schema, tables)
    # The seed node is added first to be at index 0 of the seed nodeset.
    seed_rows, other_rows = [], []
    for row in node_rows[seed_id]:
      is_seed = (
          row[_ELEMENT_CLASS_COLUMN] == seed_nodeset
          and row[_NODE_ID_COLUMN] == seed_id
      )
      (seed_rows if is_seed else other_rows).append(row)
    for row in seed_rows + other_rows:
      builder.add_node(
          row[_ELEMENT_CLASS_COLUMN],
          row[_NODE_ID_COLUMN],
          row[_PROPERTIES_JSON_COLUMN],
      )
    for row in edge_rows[seed_id]:
      builder.add_edge(
          row[_ELEMENT_CLASS_COLUMN],
          row[_SOURCE_ID_COLUMN],
          row[_TARGET_ID_COLUMN],
          row[_PROPERTIES_JSON_COLUMN],
      )
    graphs.append(builder.build())
  return graphs


def _check_table_clustering(
    client: bigquery.Client,
    tables: _GraphTables,
    schema: schema_lib.GraphSchema,
    plan: config_lib.SamplingPlan,
) -> None:
  """Logs a recommendation for each traversed table that is not clustered.

  Without clustering on the key columns, each hop scans the full table.
  BigQuery cannot cluster BYTES columns, so BYTES-keyed tables are skipped.

  Args:
    client: BigQuery client.
    tables: The BigQuery tables of the graph.
    schema: The schema of the graph.
    plan: The sampling plan.
  """
  edgesets = sorted(config_lib.edgesets_in_plan(plan))
  nodesets = {plan.root.nodeset}
  for edgeset_name in edgesets:
    nodesets.add(schema.edge_sets[edgeset_name].source)
    nodesets.add(schema.edge_sets[edgeset_name].target)

  recommendations = []  # (table_ref, columns)
  for nodeset_name in sorted(nodesets):
    node_table = tables.node_tables[nodeset_name]
    recommendations.append((node_table.table_ref, [node_table.key]))
  for edgeset_name in edgesets:
    edge_table = tables.edge_tables[edgeset_name]
    recommendations.append(
        (edge_table.table_ref, [edge_table.source, edge_table.target])
    )

  for table_ref, columns in recommendations:
    if any(column.type_kind == _BQ_TYPE_BYTES for column in columns):
      continue
    clustering = client.get_table(table_ref).clustering_fields or []
    if clustering and clustering[0] in {column.name for column in columns}:
      continue
    columns_sql = ", ".join(f"`{column.name}`" for column in columns)
    log.warning(
        f"BigQuery table `{table_ref}` is not clustered by {columns_sql}"
        f" (current clustering: {clustering}). Sampling queries will scan the"
        " full table. To speed up sampling, run in BigQuery:\n"
        f"  CREATE OR REPLACE TABLE `{table_ref}` CLUSTER BY {columns_sql} AS"
        f" SELECT * FROM `{table_ref}`;"
    )


class BigQueryGraphSampler:
  """Samples subgraphs directly from a BigQuery Graph using SQL queries.

  `BigQueryGraphSampler` translates a `SamplingPlan` into a single parameterized
  BigQuery SQL query over the tables of a BigQuery Property Graph.

  Usage example:

  ```python
  import dgf

  schema = dgf.io.read_bigquery_graph_schema(
      project="my-project",
      dataset="my_dataset",
      graph="my_graph",
  )
  plan = dgf.sampling.SimpleSamplingConfig(
      seed_nodeset="nodes",
      num_hops=2,
      hop_width=5,
  )
  sampler = dgf.sampling.create_bigquery_graph_sampler(
      project="my-project",
      dataset="my_dataset",
      graph="my_graph",
      plan=plan,
      schema=schema,
  )
  samples = sampler.sample(["10", "11"])
  ```
  """

  def __init__(
      self,
      project: str,
      dataset: str,
      graph: str,
      plan: config_lib.SimpleSamplingConfig | config_lib.SamplingPlan,
      schema: schema_lib.GraphSchema,
      *,
      metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata | None = None,
      quota_project: str | None = None,
      debug_sampling: bool = False,
      check_optimizations: bool = True,
  ):
    self._schema = schema
    self._quota_project = quota_project or project
    self._debug_sampling = debug_sampling

    if isinstance(plan, config_lib.SimpleSamplingConfig):
      plan = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)
    if plan.with_replacement or plan.temporal_sampling:
      raise NotImplementedError(
          "BigQueryGraphSampler does not support sampling with replacement"
          " or temporal sampling."
      )
    self._plan = plan

    if metadata is None:
      metadata = bigquery_graph_lib.get_metadata(project, dataset, graph)
    self._tables = _resolve_graph_tables(metadata, schema)
    self._seed_key = self._tables.node_tables[plan.root.nodeset].key
    self._query = QueryGenerator(
        self._tables, plan, debug_sampling=debug_sampling
    ).generate()
    self._client: bigquery.Client | None = None

    if check_optimizations:
      _check_table_clustering(self._get_client(), self._tables, schema, plan)

  def _get_client(self) -> bigquery.Client:
    """Returns a cached BigQuery client."""
    if self._client is None:
      self._client = bigquery.Client(project=self._quota_project)
    return self._client

  @property
  def query(self) -> str:
    """Returns the generated SQL query used for sampling."""
    return self._query

  def sample(
      self, seed_ids: Sequence[str | bytes] | np.ndarray
  ) -> list[in_memory_graph_lib.InMemoryGraph]:
    """Samples subgraphs around the given seed node IDs.

    Args:
      seed_ids: Primary keys of the seed nodes, as `str` or `bytes`.

    Returns:
      One `InMemoryGraph` per seed, in the order of `seed_ids`, with the seed
      node at index 0 of the seed nodeset.
    """
    seed_bytes = [
        seed_id.encode("utf-8") if isinstance(seed_id, str) else bytes(seed_id)
        for seed_id in seed_ids
    ]
    if not seed_bytes:
      return []
    if self._seed_key.type_kind == _BQ_TYPE_STRING:
      parameter_values = [seed_id.decode("utf-8") for seed_id in seed_bytes]
    else:
      parameter_values = seed_bytes

    query_parameters: list[
        bigquery.ArrayQueryParameter | bigquery.ScalarQueryParameter
    ] = [
        bigquery.ArrayQueryParameter(
            _SEED_IDS_PARAMETER, self._seed_key.type_kind, parameter_values
        )
    ]
    if not self._debug_sampling:
      query_parameters.append(
          bigquery.ScalarQueryParameter(
              _SAMPLING_SEED_PARAMETER, _BQ_TYPE_INT64, random.getrandbits(63)
          )
      )
    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    rows = self._get_client().query(self._query, job_config=job_config).result()
    return _rows_to_in_memory_graphs(
        rows,
        self._schema,
        self._tables,
        seed_nodeset=self._plan.root.nodeset,
        seed_ids=seed_bytes,
    )


def create_bigquery_graph_sampler(
    project: str,
    dataset: str,
    graph: str,
    plan: config_lib.SimpleSamplingConfig | config_lib.SamplingPlan,
    schema: schema_lib.GraphSchema,
    *,
    metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata | None = None,
    quota_project: str | None = None,
    debug_sampling: bool = False,
    check_optimizations: bool = True,
) -> BigQueryGraphSampler:
  """Creates a `BigQueryGraphSampler` instance to sample subgraphs from BigQuery.

  The sampler queries the underlying BigQuery tables of a BigQuery Property
  Graph using a single parameterized SQL query and returns subgraphs as
  `InMemoryGraph` objects compatible with models trained on graphs exported
  via `dgf.io.export_bigquery_graph_to_disk`.

  Usage example:

  ```python
  import dgf

  sampler = dgf.sampling.create_bigquery_graph_sampler(
      project="my-project",
      dataset="my_dataset",
      graph="my_graph",
      plan=model.data().sampling_plan,
      schema=model.data().schema,
  )
  samples = sampler.sample(["10", "11"])
  predictions = model.predict_on_graph_sample_batch(samples)
  ```

  Args:
    project: The GCP project ID containing the BigQuery dataset.
    dataset: The BigQuery dataset ID containing the property graph.
    graph: The name of the BigQuery Property Graph.
    plan: The sampling configuration (`SimpleSamplingConfig` or `SamplingPlan`).
    schema: The `GraphSchema` of the graph. Defines the sampled features. Each
      nodeset needs a feature with `semantic=PRIMARY_ID`.
    metadata: Optional pre-fetched `BigQueryGraphMetadata`. If `None`, metadata
      is fetched from `INFORMATION_SCHEMA.PROPERTY_GRAPHS`.
    quota_project: Optional GCP project ID to bill BigQuery jobs to. Defaults to
      `project`.
    debug_sampling: If `True`, orders neighbors deterministically by node ID
      instead of `RAND()` for reproducible testing.
    check_optimizations: If `True`, inspects BigQuery table clustering and logs
      actionable warnings if node or edge tables are not clustered on their key
      columns.

  Returns:
    A configured `BigQueryGraphSampler` instance.
  """
  return BigQueryGraphSampler(
      project=project,
      dataset=dataset,
      graph=graph,
      plan=plan,
      schema=schema,
      metadata=metadata,
      quota_project=quota_project,
      debug_sampling=debug_sampling,
      check_optimizations=check_optimizations,
  )
