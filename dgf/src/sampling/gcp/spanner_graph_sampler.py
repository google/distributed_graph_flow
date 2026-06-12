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

"""Graph sampling using Google Spanner Graph's GQL engine."""

import base64
import collections
import dataclasses
import json
from typing import Any, Dict, List, Set, Tuple, Union
from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
from dgf.src.sampling import config as config_lib
from google.cloud import spanner
import numpy as np

# Node IDs are always bytes in DGF.
NormalizedNodeId = bytes


def _normalize_node_id(
    node_id: Union[str, bytes],
    feature_schema: schema_lib.FeatureSchema,
    is_from_spanner: bool = False,
) -> bytes:
  if is_from_spanner and isinstance(node_id, bytes):
    try:
      node_id = base64.b64decode(node_id)
    except Exception:
      pass

  if isinstance(node_id, str):
    node_id = node_id.encode("utf-8")

  if feature_schema.format != schema_lib.FeatureFormat.BYTES:
    raise ValueError(
        f"Unsupported key format: {feature_schema.format}. Only BYTES (string)"
        " is supported."
    )

  return bytes(node_id)


def _node_id_expression(
    var: str, pk: str, nodeset: str, schema: schema_lib.GraphSchema
) -> str:
  feat_schema = schema.node_sets[nodeset].features[pk]
  if feat_schema.is_utf8_string:
    return f"CAST({var}.{pk} AS BYTES)"
  else:
    return f"{var}.{pk}"


def _sql_cast_to_bytes(
    col_name: str, nodeset_name: str, schema: schema_lib.GraphSchema
) -> str:
  pk = _get_primary_key(nodeset_name, schema)
  feat_schema = schema.node_sets[nodeset_name].features[pk]
  if feat_schema.is_utf8_string:
    return f"CAST({col_name} AS BYTES)"
  return col_name


RawFeatures = Dict[str, list]


@dataclasses.dataclass
class RawNodeset:
  features: RawFeatures
  num_nodes: int = 0


@dataclasses.dataclass
class RawEdgeset:
  features: RawFeatures
  adjacency: List[Tuple[int, int]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RawGraph:
  node_sets: Dict[str, RawNodeset]
  edge_sets: Dict[str, RawEdgeset]

  # Mapping from the Spanner graph internal node ids to the per-nodeset dense
  # node index in the raw graph.
  spanner_node_ids: Dict[bytes, int] = dataclasses.field(default_factory=dict)

  # Set of Spanner graph internal edge ids to avoid double counting edges.
  spanner_edge_ids: Set[Any] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class SpannerGraphId:
  project: str
  instance: str
  database: str
  graph: str


@dataclasses.dataclass
class SpannerGraphConnection:
  client: spanner.Client
  instance: Any
  database: Any


def _get_primary_key(nodeset_name: str, schema: schema_lib.GraphSchema) -> str:
  return analyse_schema_lib.primary_feature(
      nodeset_name, schema.node_sets[nodeset_name]
  )


class CteQueryGenerator:
  """Generates GQL/SQL queries that return both graph structure and features in a single call."""

  def __init__(
      self,
      graph_name: str,
      schema: schema_lib.GraphSchema,
      plan: config_lib.SamplingPlan,
      debug_sampling: bool,
  ):
    self.graph_name = graph_name
    self.schema = schema
    self.plan = plan
    self.debug_sampling = debug_sampling
    self.cte_defs = []
    self.union_queries = []
    self.var_count = 0
    self.order_by_str = "" if debug_sampling else " ORDER BY GENERATE_UUID()"

  def generate(self) -> str:
    root_nodeset = self.plan.root.nodeset
    root_pk = _get_primary_key(root_nodeset, self.schema)

    # Native ID in source_node CTE (no cast)
    source_node_cte = f"""source_node AS (
  SELECT n0_id AS seed_id, n0_id AS n0_id
  FROM GRAPH_TABLE({self.graph_name}
    MATCH (n0:{root_nodeset} WHERE n0.{root_pk} IN UNNEST(@seed_ids))
    RETURN n0.{root_pk} AS n0_id
  )
)"""
    self.cte_defs.append(source_node_cte)

    # Union seed nodes (cast to BYTES for the union, join to get features)
    seed_id_cast = _sql_cast_to_bytes("n0_id", root_nodeset, self.schema)
    self.union_queries.append(
        f"SELECT {seed_id_cast} AS seed_id, {seed_id_cast} AS node_id, 'node'"
        f" AS element_type, '{root_nodeset}' AS element_class, CAST(NULL AS"
        " BYTES) AS source_id, CAST(NULL AS BYTES) AS target_id, TO_JSON(n) AS"
        f" properties_json FROM source_node AS h JOIN {root_nodeset} AS n ON"
        f" h.n0_id = n.{root_pk}"
    )

    self._visit_node(self.plan.root, "source_node", "n0_id")

    cte_part = "WITH\n  " + ",\n  ".join(self.cte_defs)
    union_part = "\nUNION ALL\n".join(self.union_queries)
    return (
        f"{cte_part}\nSELECT seed_id, node_id, element_type, element_class,"
        f" source_id, target_id, properties_json FROM (\n{union_part}\n)"
    )

  def _visit_node(
      self,
      node: config_lib.PlanNode,
      parent_cte: str,
      parent_id_col: str,
  ):
    for edge in node.children:
      self.var_count += 1
      child_var = f"n{self.var_count}"
      cte_name = f"hop_{self.var_count}"

      child_nodeset = edge.node.nodeset
      parent_nodeset = node.nodeset

      parent_pk = _get_primary_key(parent_nodeset, self.schema)
      child_pk = _get_primary_key(child_nodeset, self.schema)

      limit = edge.hop_width
      gql_parent_var = "gp"
      gql_child_var = "gc"
      gql_edge_var = "ge"

      # Clean arrow syntax
      if edge.reversed:
        left_arrow = "<-"
        right_arrow = "-"
      else:
        left_arrow = "-"
        right_arrow = "->"

      match_pattern = f"({gql_parent_var}:{parent_nodeset}){left_arrow}[{gql_edge_var}:{edge.edgeset}]{right_arrow}({gql_child_var}:{child_nodeset})"

      # Native ID expressions (no cast)
      parent_id_expr_native = f"{gql_parent_var}.{parent_pk}"
      child_id_expr_native = f"{gql_child_var}.{child_pk}"

      # Localized IS_FIRST partition by gp.id (parent native ID)
      if self.order_by_str:
        over_clause = (
            "OVER (PARTITION BY"
            f" {gql_parent_var}.{parent_pk}{self.order_by_str})"
        )
      else:
        over_clause = f"OVER (PARTITION BY {gql_parent_var}.{parent_pk})"

      gql_query = f"""GRAPH_TABLE({self.graph_name}
      MATCH {match_pattern}
      FILTER IS_FIRST({limit}) {over_clause}
      RETURN {parent_id_expr_native} AS parent_id, TO_JSON({gql_edge_var}) AS edge_json, {child_id_expr_native} AS child_id
    )"""

      # Propagate native IDs in CTE, join on native IDs
      cte_def = f"""{cte_name} AS (
  SELECT p.seed_id, gt.edge_json, gt.child_id AS {child_var}_id, p.{parent_id_col} AS parent_id
  FROM {parent_cte} p
  JOIN {gql_query} AS gt ON p.{parent_id_col} = gt.parent_id
)"""
      self.cte_defs.append(cte_def)

      # Cast to BYTES for the final UNION ALL
      root_nodeset = self.plan.root.nodeset
      seed_id_cast = _sql_cast_to_bytes("seed_id", root_nodeset, self.schema)
      parent_id_cast = _sql_cast_to_bytes(
          "parent_id", parent_nodeset, self.schema
      )
      child_id_cast = _sql_cast_to_bytes(
          f"{child_var}_id", child_nodeset, self.schema
      )

      # Union edges
      if edge.reversed:
        self.union_queries.append(
            f"SELECT {seed_id_cast} AS seed_id, CAST(NULL AS BYTES) AS node_id,"
            f" 'edge' AS element_type, '{edge.edgeset}' AS element_class,"
            f" {child_id_cast} AS source_id, {parent_id_cast} AS target_id,"
            f" edge_json AS properties_json FROM {cte_name}"
        )
      else:
        self.union_queries.append(
            f"SELECT {seed_id_cast} AS seed_id, CAST(NULL AS BYTES) AS node_id,"
            f" 'edge' AS element_type, '{edge.edgeset}' AS element_class,"
            f" {parent_id_cast} AS source_id, {child_id_cast} AS target_id,"
            f" edge_json AS properties_json FROM {cte_name}"
        )

      # Union nodes (with features, JOIN child table)
      node_union_query = f"""
      SELECT 
        {seed_id_cast} AS seed_id, 
        {child_id_cast} AS node_id, 
        'node' AS element_type, 
        '{child_nodeset}' AS element_class, 
        CAST(NULL AS BYTES) AS source_id, 
        CAST(NULL AS BYTES) AS target_id, 
        TO_JSON(n) AS properties_json 
      FROM {cte_name} AS h
      JOIN {child_nodeset} AS n ON h.{child_var}_id = n.{child_pk}
      """
      self.union_queries.append(node_union_query)

      self._visit_node(edge.node, cte_name, f"{child_var}_id")


class SpannerGraphSampler:
  """Sampler that executes queries on Spanner directly to fetch subgraphs."""

  def __init__(
      self,
      graph: SpannerGraphId,
      plan: config_lib.SamplingPlan,
      schema: schema_lib.GraphSchema,
      debug_sampling: bool,
  ):
    self._graph = graph
    self._plan = plan
    self._schema = schema
    self._debug_sampling = debug_sampling
    self._connection = None

  def _get_connection(self) -> SpannerGraphConnection:
    if self._connection is None:
      client = spanner.Client(project=self._graph.project)
      instance = client.instance(self._graph.instance)
      database = instance.database(self._graph.database)
      self._connection = SpannerGraphConnection(
          client=client, instance=instance, database=database
      )
    return self._connection

  def _get_query(self) -> str:
    generator = CteQueryGenerator(
        graph_name=self._graph.graph,
        schema=self._schema,
        plan=self._plan,
        debug_sampling=self._debug_sampling,
    )
    return generator.generate()

  def sample(
      self, seed_ids: List[bytes]
  ) -> List[in_memory_graph_lib.InMemoryGraph]:
    """Samples subgraphs starting from the given seed nodes.

    Args:
      seed_ids: The list of node IDs (as bytes) to use as seeds.

    Returns:
      A list of InMemoryGraph objects corresponding to the subgraphs sampled for
      each seed ID.
    """
    connection = self._get_connection()
    query = self._get_query()

    # Prepare parameters for secure query execution.
    root_nodeset = self._plan.root.nodeset
    root_pk = _get_primary_key(root_nodeset, self._schema)
    root_pk_schema = self._schema.node_sets[root_nodeset].features[root_pk]
    normalized_seeds = [
        _normalize_node_id(sid, root_pk_schema) for sid in seed_ids
    ]

    if root_pk_schema.is_utf8_string:
      # Decode bytes to string for Spanner STRING type.
      params = {"seed_ids": [sid.decode("utf-8") for sid in normalized_seeds]}
      param_types = {
          "seed_ids": spanner.param_types.Array(spanner.param_types.STRING)
      }
    else:
      params = {"seed_ids": normalized_seeds}
      param_types = {
          "seed_ids": spanner.param_types.Array(spanner.param_types.BYTES)
      }

    with connection.database.snapshot() as snapshot:
      json_results = list(
          snapshot.execute_sql(
              query,
              params=params,
              param_types=param_types,
          )
      )

    return _cte_result_to_in_memory_graphs(
        json_results,
        self._schema,
        seed_ids,
        self._plan.root.nodeset,
    )


def create_graph_spanner_sampler(
    project: str,
    instance: str,
    database: str,
    graph: str,
    plan: Union[config_lib.SimpleSamplingConfig, config_lib.SamplingPlan],
    schema: schema_lib.GraphSchema,
    debug_sampling: bool = False,
) -> SpannerGraphSampler:
  """Creates a SpannerGraphSampler instance."""
  if isinstance(plan, config_lib.SimpleSamplingConfig):
    plan = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)

  return SpannerGraphSampler(
      plan=plan,
      schema=schema,
      debug_sampling=debug_sampling,
      graph=SpannerGraphId(
          project=project,
          instance=instance,
          database=database,
          graph=graph,
      ),
  )


def _json_features_to_features(
    values: Dict[str, List[Any]], schema: schema_lib.FeatureSetSchema
) -> Dict[str, np.ndarray]:
  """Converts spanner feature/propertie values into numpy feature values."""
  result = {}
  for feature_name, feature_schema in schema.items():
    dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[feature_schema.format]
    value = np.array(values[feature_name], dtype=dtype)
    if feature_schema.shape:
      resolved_shape = [0 if value.size == 0 else -1]
      for i, dim in enumerate(feature_schema.shape):
        if dim is None:
          if i + 1 < value.ndim:
            resolved_shape.append(value.shape[i + 1])
          else:
            resolved_shape.append(0)
        else:
          resolved_shape.append(dim)
      value = value.reshape(resolved_shape)
    result[feature_name] = value
  return result


def _cte_result_to_in_memory_graphs(
    result: Any,
    schema: schema_lib.GraphSchema,
    seed_ids: List[bytes],
    root_nodeset: str,
) -> List[in_memory_graph_lib.InMemoryGraph]:
  """Converts the flat CTE query results (with features) into InMemoryGraphs."""
  root_pk = _get_primary_key(root_nodeset, schema)
  root_pk_schema = schema.node_sets[root_nodeset].features[root_pk]
  normalized_seeds = [
      _normalize_node_id(sid, root_pk_schema) for sid in seed_ids
  ]

  # 1. Initialize RawGraphs for each seed
  raw_graphs_per_seed: Dict[bytes, RawGraph] = {}
  for seed_id in normalized_seeds:
    raw_graphs_per_seed[seed_id] = RawGraph(
        node_sets={
            nodeset_name: RawNodeset(
                num_nodes=0,
                features={
                    feature_name: [] for feature_name in nodeset_schema.features
                },
            )
            for nodeset_name, nodeset_schema in schema.node_sets.items()
        },
        edge_sets={
            edgeset_name: RawEdgeset(
                features={
                    feature_name: [] for feature_name in edgeset_schema.features
                }
            )
            for edgeset_name, edgeset_schema in schema.edge_sets.items()
        },
    )

  # Intermediate structure to store edges before we can resolve node indices.
  pending_edges = collections.defaultdict(list)

  # First Pass: Collect all nodes and their features, and store pending edges.
  for row in result:
    seed_id_bytes = row[0]
    if seed_id_bytes is None:
      continue
    seed_id = _normalize_node_id(
        seed_id_bytes, root_pk_schema, is_from_spanner=True
    )
    if seed_id not in raw_graphs_per_seed:
      continue
    raw_graph = raw_graphs_per_seed[seed_id]

    optype = row[2]  # element_type
    element_class = row[3]  # nodeset or edgeset name
    properties_json = row[6]  # properties_json (JSON)

    if isinstance(properties_json, str):
      properties = json.loads(properties_json) if properties_json else {}
    elif isinstance(properties_json, dict):
      properties = properties_json
    else:
      properties = {}

    if optype == "node":
      node_id_bytes = row[1]
      nodeset_name = element_class
      node_pk = _get_primary_key(nodeset_name, schema)
      node_pk_schema = schema.node_sets[nodeset_name].features[node_pk]
      node_id = _normalize_node_id(
          node_id_bytes, node_pk_schema, is_from_spanner=True
      )

      nodeset = raw_graph.node_sets[nodeset_name]

      if node_id not in raw_graph.spanner_node_ids:
        raw_graph.spanner_node_ids[node_id] = nodeset.num_nodes
        nodeset.num_nodes += 1
        for feature_name in nodeset.features:
          if feature_name == node_pk:
            val = node_id
          else:
            # Extract feature from properties.
            val = properties.get(feature_name)
          nodeset.features[feature_name].append(val)

    elif optype == "edge":
      edgeset_name = element_class
      source_id_bytes = row[4]
      target_id_bytes = row[5]

      edgeset_schema = schema.edge_sets[edgeset_name]
      src_pk = _get_primary_key(edgeset_schema.source, schema)
      trg_pk = _get_primary_key(edgeset_schema.target, schema)
      src_pk_schema = schema.node_sets[edgeset_schema.source].features[src_pk]
      trg_pk_schema = schema.node_sets[edgeset_schema.target].features[trg_pk]

      source_id = _normalize_node_id(
          source_id_bytes, src_pk_schema, is_from_spanner=True
      )
      target_id = _normalize_node_id(
          target_id_bytes, trg_pk_schema, is_from_spanner=True
      )

      pending_edges[seed_id].append(
          (edgeset_name, source_id, target_id, properties)
      )

  # Second Pass: Process edges and build final InMemoryGraphs
  in_memory_graphs = []
  for seed_id in normalized_seeds:
    raw_graph = raw_graphs_per_seed[seed_id]

    # 1. Process pending edges for this seed
    for edgeset_name, source_id, target_id, properties in pending_edges[
        seed_id
    ]:
      edgeset_schema = schema.edge_sets[edgeset_name]

      # Ensure source and target nodes exist in the graph.
      if (
          source_id not in raw_graph.spanner_node_ids
          or target_id not in raw_graph.spanner_node_ids
      ):
        continue

      source_idx = raw_graph.spanner_node_ids[source_id]
      target_idx = raw_graph.spanner_node_ids[target_id]

      edgeset = raw_graph.edge_sets[edgeset_name]

      # We need a unique edge ID to avoid duplicates
      edge_id = properties.get("identifier", f"{source_id}-{target_id}")
      if edge_id not in raw_graph.spanner_edge_ids:
        raw_graph.spanner_edge_ids.add(edge_id)
        edgeset.adjacency.append((source_idx, target_idx))

        for feature_name in edgeset.features:
          val = properties.get(feature_name)
          edgeset.features[feature_name].append(val)

    # 2. Reconstruct NodeSets and EdgeSets
    node_sets = {}
    for nodeset_name, nodeset_schema in schema.node_sets.items():
      raw_nodeset = raw_graph.node_sets[nodeset_name]
      features = _json_features_to_features(
          raw_nodeset.features, nodeset_schema.features
      )
      node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
          num_nodes=raw_nodeset.num_nodes, features=features
      )

    edge_sets = {}
    for edgeset_name, edgeset_schema in schema.edge_sets.items():
      raw_edgeset = raw_graph.edge_sets[edgeset_name]
      features = _json_features_to_features(
          raw_edgeset.features, edgeset_schema.features
      )
      if not raw_edgeset.adjacency:
        adjacency = np.zeros((2, 0), dtype=np.int64)
      else:
        adjacency = np.array(raw_edgeset.adjacency, dtype=np.int64).T
      edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
          adjacency=adjacency, features=features
      )

    in_memory_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets=node_sets, edge_sets=edge_sets
    )
    in_memory_graphs.append(in_memory_graph)

  return in_memory_graphs
