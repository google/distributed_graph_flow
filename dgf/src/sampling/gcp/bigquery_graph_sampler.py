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

"""BigQuery Graph Sampler implementation using standard SQL CTEs.

How a `SamplingPlan` is mapped to SQL:
A `SamplingPlan` is a rooted tree where the root represents the seed nodeset
and each directed tree edge `e = (u -> v)` specifies an edgeset traversal from
parent `PlanNode` `u` to child `PlanNode` `v` with fanout `k = e.num_neighbors`
and direction `e.reversed`. The plan tree is compiled into a DAG of Common Table
Expressions (CTEs) evaluated in topological order:
- The root `PlanNode` becomes `source_node`, which unnest-joins `@seed_ids`
  against the seed node table and tags every seed with its own `seed_id` so all
  seeds in a batch are traversed in parallel within a single query.
- Each tree edge `e_i` (traversed in pre-order) becomes a CTE `hop_i` that
  joins the parent CTE's `(seed_id, node_id)` pairs with the underlying edge
  table (on the source key if `reversed=False`, or target key if
  `reversed=True`). To sample up to `k` neighbors per `(seed_id, parent_node)`
  without replacement, `hop_i` applies a window filter
  `QUALIFY ROW_NUMBER() OVER (PARTITION BY seed_id, parent_id ORDER BY RAND()) <= k`.
- To avoid fetching heavy node features repeatedly during traversal, `hop_i`
  only propagates `(seed_id, neighbor_id)` keys. For each distinct nodeset
  visited across the plan tree, a `nodes_<nodeset>` CTE unions the discovered
  `(seed_id, node_id)` pairs from all CTEs producing that nodeset, deduplicates
  them (`SELECT DISTINCT`), and joins the underlying node table once to
  serialize node features via `TO_JSON_STRING(STRUCT(...))`.
- Finally, all `nodes_<nodeset>` CTEs and deduplicated `edges_<edgeset>` CTEs
  are combined with `UNION ALL` into a single 7-column stream
  `(seed_id, node_id, element_type, element_class, source_id, target_id, properties_json)`.

Structure of this file:
1. SQL & metadata helpers: Quote identifiers, resolve node/edge tables from
   `BigQueryGraphMetadata`, and build key and `TO_JSON_STRING` property
   expressions.
2. `CteQueryGenerator`: Compiles a `SamplingPlan` into a single parameterized
   GoogleSQL CTE query (`source_node`, per-hop `hop_k` CTEs with
   `QUALIFY ROW_NUMBER()`, per-nodeset `nodes_<name>` deduplication CTEs, and a
   unified 7-column `UNION ALL` result stream).
3. Subgraph assembly (`_RawGraph`, `_cte_result_to_in_memory_graphs`): Groups
   streamed rows by `seed_id`, parses JSON properties into NumPy arrays, places
   each seed node at index 0 of the seed nodeset, and remaps edge endpoints to
   0-based local node indices.
4. `_check_table_clustering` & `BigQueryGraphSampler`: Inspects table clustering
   to log actionable `CLUSTER BY` warnings and executes parameterized sampling
   queries.

Limitations:
- Non-UTF-8 raw binary (`BYTES`) columns are not supported (`BYTES` columns are
  converted via `SAFE_CONVERT_BYTES_TO_STRING` for JSON serialization).
- Multi-dimensional arrays (>1D) and `STRUCT` feature columns are not supported.
- Temporal sampling (`temporal_sampling=True`), sampling with replacement,
  parallel multi-edges between identical `(source, target)` pairs in the same
  edgeset, and merged-subgraph sampling (`subgraph()`) are not supported.
- Each `.sample()` call runs a BigQuery query job (~1.5-4s latency), which is
  suited for online inference or small-batch inspection rather than training.
"""

import collections
from collections.abc import Iterable, Sequence
import dataclasses
import json
import re
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

_DGF_SCHEMA_ATTRIBUTE_FEATURE_NAME = "__attributes__"
_ELEMENT_TYPE_NODE = "node"
_ELEMENT_TYPE_EDGE = "edge"
_INT_BQ_TYPES = frozenset({"INT64", "INT32", "INTEGER"})
_BQ_COLUMN_REGEX = re.compile(r"^[a-zA-Z_#][a-zA-Z0-9_]*$")


def _quote_bq_col(col: str) -> str:
  """Validates and quotes a BigQuery column identifier if needed."""
  if not _BQ_COLUMN_REGEX.match(col):
    raise ValueError(
        f"Invalid BigQuery column identifier: {col!r}. Must start with a"
        " letter, underscore, or '#' and contain only letters, numbers, and"
        " underscores."
    )
  return f"`{col}`" if col.startswith("#") else col


def _format_col(table_alias: str, col: str) -> str:
  """Formats a qualified BigQuery column reference."""
  return f"{table_alias}.{_quote_bq_col(col)}"


def _format_table_path(
    data_source: bigquery_graph_metadata_lib.DataSourceTable,
) -> str:
  """Formats a BigQuery DataSourceTable as a backtick-quoted SQL table path."""
  gcp_common_lib.validate_identifier(data_source.dataset_id)
  gcp_common_lib.validate_identifier(data_source.table_id)
  return (
      f"`{data_source.project_id}.{data_source.dataset_id}"
      f".{data_source.table_id}`"
  )


def _column_bq_type(
    label_and_properties: Sequence[
        bigquery_graph_metadata_lib.LabelAndProperties
    ],
    col_name: str,
) -> str:
  """Returns the uppercase BigQuery type_kind for a column/property, or ''."""
  for lp in label_and_properties:
    for prop in lp.properties or ():
      if prop.name == col_name or prop.expression == col_name:
        return prop.data_type.type_kind.strip().upper()
  return ""


def _is_bytes_key(node_table: bigquery_graph_metadata_lib.NodeTable) -> bool:
  """Returns True if the node table has a single key column of type BYTES."""
  if len(node_table.key_columns) != 1:
    return False
  return (
      _column_bq_type(
          node_table.label_and_properties, node_table.key_columns[0]
      )
      == "BYTES"
  )


def _cast_to_string_sql(expr: str, is_bytes: bool = False) -> str:
  """Formats a SQL expression converting a key/column to STRING."""
  if is_bytes:
    return f"SAFE_CONVERT_BYTES_TO_STRING({expr})"
  return f"CAST({expr} AS STRING)"


def _build_key_expr(
    table_alias: str,
    columns: Sequence[str],
    cast_to_string: bool = False,
    label_and_properties: Sequence[
        bigquery_graph_metadata_lib.LabelAndProperties
    ] = (),
) -> str:
  """Builds a SQL expression for a single or composite key."""
  if len(columns) == 1:
    col = columns[0]
    expr = _format_col(table_alias, col)
    is_bytes = _column_bq_type(label_and_properties, col) == "BYTES"
    return _cast_to_string_sql(expr, is_bytes) if cast_to_string else expr
  cast_cols = ", ".join(
      _cast_to_string_sql(
          _format_col(table_alias, col),
          _column_bq_type(label_and_properties, col) == "BYTES",
      )
      for col in columns
  )
  return f"ARRAY_TO_STRING([{cast_cols}], '_')"


def _build_properties_json_expr(
    table_alias: str,
    label_and_properties: Sequence[
        bigquery_graph_metadata_lib.LabelAndProperties
    ],
    feature_schemas: schema_lib.FeatureSetSchema,
) -> str:
  """Builds a SQL expression that serializes required element properties to JSON."""
  if not feature_schemas:
    return "CAST(NULL AS STRING)"

  combine_as_json = _DGF_SCHEMA_ATTRIBUTE_FEATURE_NAME in feature_schemas
  prop_exprs: dict[str, str] = {}
  for lp in label_and_properties:
    for prop in lp.properties or ():
      if combine_as_json or prop.name in feature_schemas:
        col_sql = _format_col(table_alias, prop.expression)
        type_kind = prop.data_type.type_kind.strip().upper()
        if type_kind == "BYTES":
          col_sql = f"SAFE_CONVERT_BYTES_TO_STRING({col_sql})"
        elif (
            type_kind == "ARRAY"
            and prop.data_type.array_element_type is not None
            and prop.data_type.array_element_type.type_kind.strip().upper()
            == "BYTES"
        ):
          col_sql = (
              "ARRAY(SELECT SAFE_CONVERT_BYTES_TO_STRING(x)"
              f" FROM UNNEST({col_sql}) AS x)"
          )
        prop_exprs[prop.name] = f"{col_sql} AS `{prop.name}`"

  for feat_name in feature_schemas:
    if feat_name == _DGF_SCHEMA_ATTRIBUTE_FEATURE_NAME or feat_name == "#id":
      continue
    if feat_name not in prop_exprs:
      col_sql = _format_col(table_alias, feat_name)
      prop_exprs[feat_name] = f"{col_sql} AS `{feat_name}`"

  if not prop_exprs:
    return "CAST(NULL AS STRING)"

  fields_sql = ", ".join(prop_exprs.values())
  if combine_as_json:
    return f"TO_JSON_STRING(STRUCT({fields_sql}))"
  return f"TO_JSON_STRING(STRUCT(STRUCT({fields_sql}) AS properties))"


def _resolve_node_table(
    metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata,
    nodeset_name: str,
) -> bigquery_graph_metadata_lib.NodeTable:
  """Resolves a nodeset name in GraphSchema to its NodeTable metadata."""
  for nt in metadata.node_tables:
    if nt.name == nodeset_name or nt.data_source_table.table_id == nodeset_name:
      return nt
  raise ValueError(
      f"Nodeset {nodeset_name!r} not found in BigQuery graph metadata. "
      f"Available node tables: {[nt.name for nt in metadata.node_tables]}"
  )


def _resolve_edge_table(
    metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata,
    edgeset_name: str,
) -> bigquery_graph_metadata_lib.EdgeTable:
  """Resolves an edgeset name in GraphSchema to its EdgeTable metadata."""
  for et in metadata.edge_tables:
    if et.name == edgeset_name or et.data_source_table.table_id == edgeset_name:
      return et
  raise ValueError(
      f"Edgeset {edgeset_name!r} not found in BigQuery graph metadata. "
      f"Available edge tables: {[et.name for et in metadata.edge_tables]}"
  )


def _seed_key_bq_type(
    node_table: bigquery_graph_metadata_lib.NodeTable,
) -> str:
  """Returns the BigQuery parameter type ('INT64', 'BYTES', or 'STRING') for seed_ids."""
  if len(node_table.key_columns) != 1:
    return "STRING"
  type_kind = _column_bq_type(
      node_table.label_and_properties, node_table.key_columns[0]
  )
  if type_kind in _INT_BQ_TYPES:
    return "INT64"
  if type_kind == "BYTES":
    return "BYTES"
  return "STRING"


class CteQueryGenerator:
  """Generates a BigQuery Standard SQL CTE query from a DGF SamplingPlan."""

  def __init__(
      self,
      metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata,
      plan: config_lib.SamplingPlan,
      schema: schema_lib.GraphSchema,
      debug_sampling: bool = False,
  ):
    self._metadata = metadata
    self._plan = plan
    self._schema = schema
    self._debug_sampling = debug_sampling
    self._var_count = 0

  def generate(self) -> str:
    """Generates the parameterized BigQuery SQL query string."""
    ctes: list[str] = []
    node_sources_by_class: dict[str, list[str]] = collections.defaultdict(list)
    edge_sources_by_class: dict[str, list[str]] = collections.defaultdict(list)

    root_nodeset = self._plan.root.nodeset
    root_node_table = _resolve_node_table(self._metadata, root_nodeset)
    root_table_sql = _format_table_path(root_node_table.data_source_table)
    root_key_expr = _build_key_expr("n0", root_node_table.key_columns)

    source_cte = (
        "  source_node AS (\n"
        f"    SELECT {root_key_expr} AS seed_id, {root_key_expr} AS n0_id\n"
        f"    FROM {root_table_sql} AS n0\n"
        f"    WHERE {root_key_expr} IN UNNEST(@seed_ids)\n"
        "  )"
    )
    ctes.append(source_cte)
    node_sources_by_class[root_nodeset].append(
        "SELECT seed_id, n0_id AS node_id FROM source_node"
    )

    self._visit_node(
        self._plan.root,
        parent_cte="source_node",
        parent_col="n0_id",
        ctes=ctes,
        node_sources_by_class=node_sources_by_class,
        edge_sources_by_class=edge_sources_by_class,
    )

    union_queries: list[str] = []
    seed_is_bytes = _is_bytes_key(root_node_table)

    for nodeset_name, node_selects in node_sources_by_class.items():
      node_table = _resolve_node_table(self._metadata, nodeset_name)
      node_table_sql = _format_table_path(node_table.data_source_table)
      node_key_expr = _build_key_expr("n", node_table.key_columns)
      node_features_schema = self._schema.node_sets[nodeset_name].features
      node_json_expr = _build_properties_json_expr(
          "n", node_table.label_and_properties, node_features_schema
      )

      dedup_nodes_cte_name = f"nodes_{nodeset_name}"
      gcp_common_lib.validate_identifier(dedup_nodes_cte_name)
      union_distinct_sql = "\n      UNION DISTINCT\n      ".join(node_selects)
      ctes.append(
          f"  {dedup_nodes_cte_name} AS (\n"
          f"    {union_distinct_sql}\n"
          "  )"
      )

      node_is_bytes = _is_bytes_key(node_table)
      seed_col_sql = _cast_to_string_sql("u.seed_id", seed_is_bytes)
      node_col_sql = _cast_to_string_sql("u.node_id", node_is_bytes)
      union_queries.append(
          "SELECT\n"
          f"  {seed_col_sql} AS seed_id,\n"
          f"  {node_col_sql} AS node_id,\n"
          f"  '{_ELEMENT_TYPE_NODE}' AS element_type,\n"
          f"  '{nodeset_name}' AS element_class,\n"
          "  CAST(NULL AS STRING) AS source_id,\n"
          "  CAST(NULL AS STRING) AS target_id,\n"
          f"  {node_json_expr} AS properties_json\n"
          f"FROM {dedup_nodes_cte_name} AS u\n"
          f"JOIN {node_table_sql} AS n ON u.node_id = {node_key_expr}"
      )

    for edgeset_name, edge_selects in edge_sources_by_class.items():
      edge_schema = self._schema.edge_sets[edgeset_name]
      has_edge_features = bool(edge_schema.features)
      union_op = "UNION ALL" if has_edge_features else "UNION DISTINCT"
      combined_edges_sql = f"\n      {union_op}\n      ".join(edge_selects)
      src_is_bytes = _is_bytes_key(
          _resolve_node_table(self._metadata, edge_schema.source)
      )
      dst_is_bytes = _is_bytes_key(
          _resolve_node_table(self._metadata, edge_schema.target)
      )
      seed_col_sql = _cast_to_string_sql("e.seed_id", seed_is_bytes)
      src_col_sql = _cast_to_string_sql("e.source_id", src_is_bytes)
      dst_col_sql = _cast_to_string_sql("e.target_id", dst_is_bytes)
      union_queries.append(
          "SELECT\n"
          f"  {seed_col_sql} AS seed_id,\n"
          "  CAST(NULL AS STRING) AS node_id,\n"
          f"  '{_ELEMENT_TYPE_EDGE}' AS element_type,\n"
          f"  '{edgeset_name}' AS element_class,\n"
          f"  {src_col_sql} AS source_id,\n"
          f"  {dst_col_sql} AS target_id,\n"
          "  e.edge_json AS properties_json\n"
          f"FROM (\n      {combined_edges_sql}\n) AS e"
      )

    with_clause = "WITH\n" + ",\n".join(ctes)
    union_clause = "\nUNION ALL\n".join(union_queries)
    return (
        f"{with_clause}\n"
        "SELECT seed_id, node_id, element_type, element_class, source_id,"
        f" target_id, properties_json\nFROM (\n{union_clause}\n)"
    )

  def _visit_node(
      self,
      node: config_lib.PlanNode,
      parent_cte: str,
      parent_col: str,
      ctes: list[str],
      node_sources_by_class: dict[str, list[str]],
      edge_sources_by_class: dict[str, list[str]],
  ) -> None:
    """Recursively visits `PlanNode` children to emit `hop_k` CTEs."""
    for edge in node.children:
      self._var_count += 1
      cte_name = f"hop_{self._var_count}"
      node_alias = f"n{self._var_count}"

      target_nodeset = edge.node.nodeset
      edge_schema = self._schema.edge_sets[edge.edgeset]
      edge_table = _resolve_edge_table(self._metadata, edge.edgeset)
      edge_table_sql = _format_table_path(edge_table.data_source_table)

      src_cols = edge_table.source_node_reference.edge_table_columns
      dst_cols = edge_table.destination_node_reference.edge_table_columns
      e_src_expr = _build_key_expr("e", src_cols)
      e_dst_expr = _build_key_expr("e", dst_cols)

      if edge.reversed:
        join_expr = e_dst_expr
        child_expr = e_src_expr
      else:
        join_expr = e_src_expr
        child_expr = e_dst_expr

      if edge_schema.features:
        edge_json_expr = _build_properties_json_expr(
            "e", edge_table.label_and_properties, edge_schema.features
        )
      else:
        edge_json_expr = "CAST(NULL AS STRING)"

      limit = edge.hop_width
      qualify_clause = ""
      if limit > 0:
        order_expr = child_expr if self._debug_sampling else "RAND()"
        qualify_clause = (
            "\n    QUALIFY ROW_NUMBER() OVER ("
            f"PARTITION BY p.seed_id, p.{parent_col} ORDER BY {order_expr}"
            f") <= {limit}"
        )

      hop_cte = (
          f"  {cte_name} AS (\n"
          f"    SELECT p.seed_id, p.{parent_col} AS parent_id,"
          f" {child_expr} AS {node_alias}_id, {edge_json_expr} AS edge_json\n"
          f"    FROM (SELECT DISTINCT seed_id, {parent_col}"
          f" FROM {parent_cte}) AS p\n"
          f"    JOIN {edge_table_sql} AS e ON p.{parent_col} = {join_expr}"
          f"{qualify_clause}\n"
          "  )"
      )
      ctes.append(hop_cte)

      node_sources_by_class[target_nodeset].append(
          f"SELECT seed_id, {node_alias}_id AS node_id FROM {cte_name}"
      )

      if edge.reversed:
        edge_src_col = f"{node_alias}_id"
        edge_dst_col = "parent_id"
      else:
        edge_src_col = "parent_id"
        edge_dst_col = f"{node_alias}_id"

      edge_sources_by_class[edge.edgeset].append(
          f"SELECT seed_id, {edge_src_col} AS source_id,"
          f" {edge_dst_col} AS target_id, edge_json FROM {cte_name}"
      )

      self._visit_node(
          edge.node,
          parent_cte=cte_name,
          parent_col=f"{node_alias}_id",
          ctes=ctes,
          node_sources_by_class=node_sources_by_class,
          edge_sources_by_class=edge_sources_by_class,
      )


def _extract_features_from_json(
    element_name: str,
    element_type: str,
    json_str: str | None,
    graph_schema: schema_lib.GraphSchema,
) -> dict[str, np.ndarray]:
  """Extracts DGF numpy features from a serialized properties_json string."""
  if not json_str:
    return {}

  element_dict = json.loads(json_str)
  if element_type == _ELEMENT_TYPE_NODE:
    element_schema = graph_schema.node_sets[element_name]
    gcp_type = gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE
  else:
    element_schema = graph_schema.edge_sets[element_name]
    gcp_type = gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE

  combine_as_json = (
      _DGF_SCHEMA_ATTRIBUTE_FEATURE_NAME in element_schema.features
  )
  if not combine_as_json:
    if gcp_common_lib.GRAPH_ELEMENT_PROPERTIES_KEY not in element_dict:
      element_dict = {
          gcp_common_lib.GRAPH_ELEMENT_PROPERTIES_KEY: element_dict
      }
    props = element_dict[gcp_common_lib.GRAPH_ELEMENT_PROPERTIES_KEY]
    if isinstance(props, dict):
      element_dict = {
          gcp_common_lib.GRAPH_ELEMENT_PROPERTIES_KEY: {
              k: v for k, v in props.items() if v is not None
          }
      }

  return gcp_common_lib.graph_element_to_features(
      element_name,
      gcp_type,
      element_dict,
      graph_schema,
      combine_as_json,
  )


def _default_feature_value(
    feature_schema: schema_lib.FeatureSchema,
) -> np.ndarray:
  """Returns a default numpy array for a missing or null feature value."""
  dtype = feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[feature_schema.format]
  if not feature_schema.is_static_shape():
    return np.empty((0,), dtype=dtype)
  default_scalar = gcp_common_lib.parse_property_value_to_feature(
      schema_lib.FeatureSchema(format=feature_schema.format), None
  )
  if feature_schema.shape:
    dims = [int(d) for d in feature_schema.shape if d is not None]
    return np.full(dims, default_scalar, dtype=dtype)
  return np.array(default_scalar, dtype=dtype)


def _empty_feature_array(
    feature_schema: schema_lib.FeatureSchema,
) -> np.ndarray:
  """Creates an empty numpy array matching the given FeatureSchema."""
  if not feature_schema.is_static_shape():
    return np.empty((0,), dtype=np.object_)
  dtype = feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[feature_schema.format]
  dims = [0]
  if feature_schema.shape:
    dims.extend(int(d) for d in feature_schema.shape if d is not None)
  return np.empty(dims, dtype=dtype)


@dataclasses.dataclass
class _RawNodeSet:
  """Accumulates nodes and features for a single nodeset in a sampled subgraph."""

  num_nodes: int = 0
  features: dict[str, list[np.ndarray]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )


@dataclasses.dataclass
class _RawEdgeSet:
  """Accumulates edges and features for a single edgeset in a sampled subgraph."""

  sources: list[bytes] = dataclasses.field(default_factory=list)
  targets: list[bytes] = dataclasses.field(default_factory=list)
  seen_edges: set[tuple[bytes, bytes]] = dataclasses.field(default_factory=set)
  features: dict[str, list[np.ndarray]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )


@dataclasses.dataclass
class _RawGraph:
  """Accumulates node sets and edge sets for a single seed node's subgraph."""

  node_sets: dict[str, _RawNodeSet] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(_RawNodeSet)
  )
  edge_sets: dict[str, _RawEdgeSet] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(_RawEdgeSet)
  )
  node_id_to_idx: dict[str, dict[bytes, int]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(dict)
  )
  buffered_seed_nodes: list[tuple[bytes, str | None]] = dataclasses.field(
      default_factory=list
  )

  def add_node(
      self,
      nodeset_name: str,
      node_id: bytes,
      properties_json: str | None,
      schema: schema_lib.GraphSchema,
  ) -> None:
    """Adds a node and its features to `nodeset_name` if not already present."""
    ns_map = self.node_id_to_idx[nodeset_name]
    if node_id in ns_map:
      return
    ns = self.node_sets[nodeset_name]
    ns_map[node_id] = ns.num_nodes
    ns.num_nodes += 1

    features = _extract_features_from_json(
        nodeset_name, _ELEMENT_TYPE_NODE, properties_json, schema
    )
    ns_schema = schema.node_sets[nodeset_name]
    combine_as_json = _DGF_SCHEMA_ATTRIBUTE_FEATURE_NAME in ns_schema.features
    if not combine_as_json:
      primary_key = schema_analyse_lib.primary_feature_or_none(
          nodeset_name, ns_schema
      )
      if primary_key is None:
        primary_key = (
            schema_analyse_lib.infer_most_likely_primary_key_or_none(
                nodeset_name, ns_schema
            )
        )
      if primary_key is not None and primary_key not in features:
        pk_schema = ns_schema.features[primary_key]
        pk_dtype = feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
            pk_schema.format
        ]
        if pk_schema.format.is_integer():
          pk_val = np.array(int(node_id.decode("utf-8")), dtype=pk_dtype)
        else:
          pk_val = np.array(node_id, dtype=pk_dtype)
        features[primary_key] = pk_val

    for feat_name, feat_schema in ns_schema.features.items():
      if feat_name in features:
        ns.features[feat_name].append(features[feat_name])
      else:
        ns.features[feat_name].append(_default_feature_value(feat_schema))

  def add_edge(
      self,
      edgeset_name: str,
      source_id: bytes,
      target_id: bytes,
      properties_json: str | None,
      schema: schema_lib.GraphSchema,
  ) -> None:
    """Adds an edge and its features to `edgeset_name` if not already present."""
    es = self.edge_sets[edgeset_name]
    edge_key = (source_id, target_id)
    if edge_key in es.seen_edges:
      return
    es.seen_edges.add(edge_key)
    es.sources.append(source_id)
    es.targets.append(target_id)

    es_schema = schema.edge_sets[edgeset_name]
    if not es_schema.features:
      return
    features = _extract_features_from_json(
        edgeset_name, _ELEMENT_TYPE_EDGE, properties_json, schema
    )
    for feat_name, feat_schema in es_schema.features.items():
      if feat_name in features:
        es.features[feat_name].append(features[feat_name])
      else:
        es.features[feat_name].append(_default_feature_value(feat_schema))

  def to_in_memory_graph(
      self, schema: schema_lib.GraphSchema
  ) -> in_memory_graph_lib.InMemoryGraph:
    """Converts the accumulated raw data into an `InMemoryGraph`."""
    in_memory_node_sets: dict[str, in_memory_graph_lib.InMemoryNodeSet] = {}
    for nodeset_name, ns_schema in schema.node_sets.items():
      if nodeset_name not in self.node_sets:
        empty_features = {
            fname: _empty_feature_array(fschema)
            for fname, fschema in ns_schema.features.items()
        }
        in_memory_node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
            num_nodes=0, features=empty_features
        )
        continue

      raw_ns = self.node_sets[nodeset_name]
      stacked_features: dict[str, np.ndarray] = {}
      for fname, fschema in ns_schema.features.items():
        fvalues = raw_ns.features.get(fname, [])
        if not fvalues:
          stacked_features[fname] = _empty_feature_array(fschema)
        elif not fschema.is_static_shape():
          arr = np.empty(len(fvalues), dtype=np.object_)
          arr[:] = fvalues
          stacked_features[fname] = arr
        else:
          stacked_features[fname] = np.stack(
              fvalues,
              axis=0,
              dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
                  fschema.format
              ],
          )
      in_memory_node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
          num_nodes=raw_ns.num_nodes, features=stacked_features
      )

    in_memory_edge_sets: dict[str, in_memory_graph_lib.InMemoryEdgeSet] = {}
    for edgeset_name, es_schema in schema.edge_sets.items():
      if edgeset_name not in self.edge_sets:
        empty_features = {
            fname: _empty_feature_array(fschema)
            for fname, fschema in es_schema.features.items()
        }
        in_memory_edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
            adjacency=np.zeros((2, 0), dtype=np.int64), features=empty_features
        )
        continue

      raw_es = self.edge_sets[edgeset_name]
      src_map = self.node_id_to_idx[es_schema.source]
      dst_map = self.node_id_to_idx[es_schema.target]

      valid_indices: list[int] = []
      src_idxs: list[int] = []
      dst_idxs: list[int] = []
      for idx, (s_id, t_id) in enumerate(zip(raw_es.sources, raw_es.targets)):
        if s_id in src_map and t_id in dst_map:
          valid_indices.append(idx)
          src_idxs.append(src_map[s_id])
          dst_idxs.append(dst_map[t_id])

      if src_idxs:
        adjacency = np.array([src_idxs, dst_idxs], dtype=np.int64)
      else:
        adjacency = np.zeros((2, 0), dtype=np.int64)

      stacked_features: dict[str, np.ndarray] = {}
      for fname, fschema in es_schema.features.items():
        all_fvalues = raw_es.features.get(fname, [])
        fvalues = [all_fvalues[i] for i in valid_indices] if all_fvalues else []
        if not fvalues:
          stacked_features[fname] = _empty_feature_array(fschema)
        elif not fschema.is_static_shape():
          arr = np.empty(len(fvalues), dtype=np.object_)
          arr[:] = fvalues
          stacked_features[fname] = arr
        else:
          stacked_features[fname] = np.stack(
              fvalues,
              axis=0,
              dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
                  fschema.format
              ],
          )
      in_memory_edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
          adjacency=adjacency, features=stacked_features
      )

    return in_memory_graph_lib.InMemoryGraph(
        node_sets=in_memory_node_sets,
        edge_sets=in_memory_edge_sets,
    )


def _cte_result_to_in_memory_graphs(
    results: Iterable[Any],
    schema: schema_lib.GraphSchema,
    seed_nodeset: str,
    seed_ids: Sequence[str | int | bytes] | None = None,
) -> list[in_memory_graph_lib.InMemoryGraph]:
  """Converts BigQuery CTE query rows into a list of `InMemoryGraph` instances."""
  raw_graphs: dict[bytes, _RawGraph] = collections.defaultdict(_RawGraph)
  edge_rows_by_seed: dict[bytes, list[tuple[str, bytes, bytes, str | None]]] = (
      collections.defaultdict(list)
  )

  for row in results:
    if isinstance(row, dict):
      seed_id_val = row["seed_id"]
      node_id_val = row["node_id"]
      element_type = row["element_type"]
      element_class = row["element_class"]
      source_id_val = row["source_id"]
      target_id_val = row["target_id"]
      properties_json = row["properties_json"]
    else:
      seed_id_val = row[0]
      node_id_val = row[1]
      element_type = row[2]
      element_class = row[3]
      source_id_val = row[4]
      target_id_val = row[5]
      properties_json = row[6]

    if properties_json is not None and not isinstance(properties_json, str):
      properties_json = json.dumps(properties_json)

    seed_id_bytes = (
        seed_id_val
        if isinstance(seed_id_val, bytes)
        else str(seed_id_val).encode("utf-8")
    )
    raw_graph = raw_graphs[seed_id_bytes]

    if element_type == _ELEMENT_TYPE_NODE:
      node_id_bytes = (
          node_id_val
          if isinstance(node_id_val, bytes)
          else str(node_id_val).encode("utf-8")
      )
      if element_class == seed_nodeset and node_id_bytes == seed_id_bytes:
        # Ensure the seed node is always placed at index 0 of `seed_nodeset`.
        raw_graph.add_node(
            element_class, node_id_bytes, properties_json, schema
        )
      elif element_class == seed_nodeset and (
          seed_id_bytes not in raw_graph.node_id_to_idx[seed_nodeset]
      ):
        raw_graph.buffered_seed_nodes.append((node_id_bytes, properties_json))
      else:
        raw_graph.add_node(
            element_class, node_id_bytes, properties_json, schema
        )
    elif element_type == _ELEMENT_TYPE_EDGE:
      src_bytes = (
          source_id_val
          if isinstance(source_id_val, bytes)
          else str(source_id_val).encode("utf-8")
      )
      dst_bytes = (
          target_id_val
          if isinstance(target_id_val, bytes)
          else str(target_id_val).encode("utf-8")
      )
      edge_rows_by_seed[seed_id_bytes].append(
          (element_class, src_bytes, dst_bytes, properties_json)
      )
    else:
      raise ValueError(f"Unknown element type: {element_type!r}")

  for seed_id_bytes, raw_graph in raw_graphs.items():
    for node_id_bytes, prop_json in raw_graph.buffered_seed_nodes:
      raw_graph.add_node(seed_nodeset, node_id_bytes, prop_json, schema)
    raw_graph.buffered_seed_nodes.clear()
    for es_name, src_b, dst_b, prop_json in edge_rows_by_seed[seed_id_bytes]:
      raw_graph.add_edge(es_name, src_b, dst_b, prop_json, schema)

  if seed_ids is not None:
    ordered_keys = []
    seen_keys = set()
    for s in seed_ids:
      k = s if isinstance(s, bytes) else str(s).encode("utf-8")
      if k in raw_graphs:
        ordered_keys.append(k)
        seen_keys.add(k)
    for k in raw_graphs:
      if k not in seen_keys:
        ordered_keys.append(k)
  else:
    ordered_keys = list(raw_graphs.keys())

  return [
      raw_graphs[k].to_in_memory_graph(schema) for k in ordered_keys
  ]


def _check_table_clustering(
    client: bigquery.Client,
    metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata,
    plan: config_lib.SamplingPlan,
    schema: schema_lib.GraphSchema,
) -> None:
  """Checks BigQuery table clustering and logs actionable recommendations."""
  _ = schema
  checked_tables: set[str] = set()

  visited_nodesets = {plan.root.nodeset}
  visited_edgesets: dict[str, set[bool]] = collections.defaultdict(set)

  def _collect_plan_nodes(node: config_lib.PlanNode) -> None:
    visited_nodesets.add(node.nodeset)
    for edge in node.children:
      visited_edgesets[edge.edgeset].add(edge.reversed)
      _collect_plan_nodes(edge.node)

  _collect_plan_nodes(plan.root)

  for ns_name in sorted(visited_nodesets):
    nt = _resolve_node_table(metadata, ns_name)
    ds = nt.data_source_table
    table_ref = f"{ds.project_id}.{ds.dataset_id}.{ds.table_id}"
    if table_ref in checked_tables:
      continue
    checked_tables.add(table_ref)
    try:
      bq_table = client.get_table(table_ref)
    except Exception:  # pylint: disable=broad-exception-caught
      continue

    clustering = bq_table.clustering_fields or []
    expected_keys = list(nt.key_columns)
    if clustering[: len(expected_keys)] != expected_keys:
      cols_sql = ", ".join(_quote_bq_col(c) for c in expected_keys)
      log.warning(
          f"BigQuery node table `{table_ref}` (nodeset {ns_name!r}) is not"
          f" clustered by its key column(s) ({cols_sql}); current"
          f" clustering={clustering}. Sampling queries may perform full table"
          " scans. To speed up sampling, run in BigQuery:\n"
          f"  CREATE OR REPLACE TABLE `{table_ref}` CLUSTER BY {cols_sql} AS"
          f" SELECT * FROM `{table_ref}`;"
      )

  for es_name, directions in sorted(visited_edgesets.items()):
    et = _resolve_edge_table(metadata, es_name)
    ds = et.data_source_table
    table_ref = f"{ds.project_id}.{ds.dataset_id}.{ds.table_id}"
    if table_ref in checked_tables:
      continue
    checked_tables.add(table_ref)
    try:
      bq_table = client.get_table(table_ref)
    except Exception:  # pylint: disable=broad-exception-caught
      continue

    clustering = bq_table.clustering_fields or []
    src_cols = list(et.source_node_reference.edge_table_columns)
    dst_cols = list(et.destination_node_reference.edge_table_columns)
    recommended_cols = list(dict.fromkeys(src_cols + dst_cols))[:4]
    cols_sql = ", ".join(_quote_bq_col(c) for c in recommended_cols)

    if not clustering or (
        clustering[: len(src_cols)] != src_cols
        and clustering[: len(dst_cols)] != dst_cols
    ):
      log.warning(
          f"BigQuery edge table `{table_ref}` (edgeset {es_name!r}) is not"
          f" clustered by its source/target key column(s) ({cols_sql});"
          f" current clustering={clustering}. Sampling queries may perform full"
          " table scans. To speed up sampling, run in BigQuery:\n"
          f"  CREATE OR REPLACE TABLE `{table_ref}` CLUSTER BY {cols_sql} AS"
          f" SELECT * FROM `{table_ref}`;"
      )
    elif True in directions and not set(dst_cols).issubset(set(clustering)):
      dst_cols_sql = ", ".join(_quote_bq_col(c) for c in dst_cols)
      log.warning(
          f"BigQuery edge table `{table_ref}` (edgeset {es_name!r}) is"
          f" traversed in reverse (incoming edges on {dst_cols_sql}), but is"
          f" clustered by {clustering}. If reverse-hop sampling is slow,"
          " consider creating a search index or clustering by"
          f" ({dst_cols_sql})."
      )


class BigQueryGraphSampler:
  """Samples subgraphs directly from a BigQuery Graph using SQL queries.

  `BigQueryGraphSampler` translates a `SamplingPlan` into a single parameterized
  BigQuery GoogleSQL query with Common Table Expressions (CTEs) over the
  underlying node and edge tables defined in the BigQuery Property Graph.

  Usage example:

  ```python
  import dgf

  graph, schema = dgf.io.read_bigquery_graph(
      project="my-project",
      dataset="my_dataset",
      graph="my_graph",
      work_dir="gs://my-bucket/tmp",
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
    self._project = project
    self._dataset = dataset
    self._graph = graph
    self._schema = schema
    self._quota_project = quota_project or project
    self._debug_sampling = debug_sampling

    if isinstance(plan, config_lib.SimpleSamplingConfig):
      self._plan = config_lib.simple_sampling_config_to_sampling_plan(
          plan, schema
      )
    elif isinstance(plan, config_lib.SamplingPlan):
      self._plan = plan
    else:
      raise ValueError(f"Unsupported plan type: {type(plan)}")

    if metadata is None:
      metadata = bigquery_graph_lib.get_metadata(project, dataset, graph)
    self._metadata = metadata
    self._client: bigquery.Client | None = None

    root_node_table = _resolve_node_table(
        self._metadata, self._plan.root.nodeset
    )
    self._seed_param_type = _seed_key_bq_type(root_node_table)

    generator = CteQueryGenerator(
        self._metadata,
        self._plan,
        self._schema,
        debug_sampling=self._debug_sampling,
    )
    self._query = generator.generate()

    if check_optimizations:
      try:
        client = self._get_client()
        _check_table_clustering(
            client, self._metadata, self._plan, self._schema
        )
      except Exception:  # pylint: disable=broad-exception-caught
        pass

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
      self,
      seed_ids: Sequence[str | int | bytes] | np.ndarray,
  ) -> list[in_memory_graph_lib.InMemoryGraph]:
    """Samples subgraphs around the given seed node IDs.

    Args:
      seed_ids: A sequence or 1D numpy array of seed node primary key IDs.

    Returns:
      A list of `InMemoryGraph` instances, one for each found seed node ID in
      the order of `seed_ids`, with the seed node placed at index 0 of the
      seed nodeset.
    """
    if isinstance(seed_ids, np.ndarray):
      seed_ids_list = seed_ids.tolist()
    else:
      seed_ids_list = list(seed_ids)

    if not seed_ids_list:
      return []

    if self._seed_param_type == "INT64":
      param_values = [
          int(s.decode("utf-8") if isinstance(s, bytes) else s)
          for s in seed_ids_list
      ]
    elif self._seed_param_type == "BYTES":
      param_values = [
          s if isinstance(s, bytes) else str(s).encode("utf-8")
          for s in seed_ids_list
      ]
    else:
      param_values = [
          s.decode("utf-8") if isinstance(s, bytes) else str(s)
          for s in seed_ids_list
      ]

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "seed_ids", self._seed_param_type, param_values
            )
        ]
    )
    query_job = self._get_client().query(self._query, job_config=job_config)
    results = query_job.result()

    return _cte_result_to_in_memory_graphs(
        results,
        self._schema,
        seed_nodeset=self._plan.root.nodeset,
        seed_ids=seed_ids_list,
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
  Graph using a single parameterized GoogleSQL CTE query and returns subgraphs
  as `InMemoryGraph` objects compatible with models trained on graphs exported
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
    schema: The `GraphSchema` of the graph.
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
