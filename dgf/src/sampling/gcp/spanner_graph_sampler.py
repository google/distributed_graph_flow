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

import dataclasses
import logging
from typing import Any, Dict, List, Set, Tuple, Union
from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
from dgf.src.sampling import config as config_lib
from google.cloud import spanner
import numpy as np

# The possible types of node ids.
NodeId = Union[str, int]

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

  # Mapping from the Spanner graph internal node ids (which is different from
  # the classical id nodes) to the per-nodeset dense node index in the raw
  # graph. "already_visited" is used to avoid double counting nodes, and to
  # create the adjacency lists.
  spanner_node_ids: Dict[str, int] = dataclasses.field(default_factory=dict)

  # Set of Spanner graph internal edge ids (or constructed keys) to avoid
  # double counting edges.
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

  def _get_gql_query(self, seed_ids: List[NodeId]) -> str:
    return _generate_gql_query(
        graph_name=self._graph.graph,
        plan=self._plan,
        schema=self._schema,
        seed_ids=seed_ids,
        debug_sampling=self._debug_sampling,
    )

  def sample_to_json(self, seed_ids: List[NodeId]):
    query = self._get_gql_query(seed_ids)
    connection = self._get_connection()
    with connection.database.snapshot() as snapshot:
      return list(snapshot.execute_sql(query))

  def sample(
      self, seed_ids: List[NodeId]
  ) -> List[in_memory_graph_lib.InMemoryGraph]:
    json_results = self.sample_to_json(seed_ids)
    return _json_to_in_memory_graphs(
        json_results, self._schema, seed_ids, self._plan.root.nodeset
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
  """Creates a SpannerGraphSampler instance.

  Example:
    ```python
    schema = schema_lib.GraphSchema(...)
    config = config_lib.SimpleSamplingConfig(
        seed_nodeset='users',
        num_hops=2,
        hop_width=10
    )
    sampler = create_graph_spanner_sampler(
        project='my-project',
        instance='my-instance',
        database='my-database',
        graph='my-graph',
        plan=config,
        schema=schema
    )
    subgraphs = sampler.sample(['user1', 'user2'])
    ```

  Args:
    project: Google Cloud project ID.
    instance: Spanner instance ID.
    database: Spanner database ID.
    graph: Spanner graph ID.
    plan: Sampling plan or simple sampling config.
    schema: Graph schema.
    debug_sampling: Whether to enable debug mode for sampling.

  Returns:
    A SpannerGraphSampler instance.
  """
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


def _generate_gql_query(
    graph_name: str,
    plan: config_lib.SamplingPlan,
    schema: schema_lib.GraphSchema,
    seed_ids: List[NodeId],
    debug_sampling: bool,
) -> str:
  """Generates a GQL query string for sampling from Spanner Graphs.

  The GQL query will return a different row for each sampling path i.e. leaf
  nodes.

  TODO(gbm): Can we avoid retuning multiple times the feature values (i.e.
  properties) of the non-leaf nodes?

  Args:
    graph_name: Spanner graph ID.
    plan: Sampling plan.
    schema: Graph schema.
    seed_ids: List of seed node IDs.
    debug_sampling: Whether to enable debug mode for sampling.

  Returns:
    A GQL query string.
  """

  primary_key_feature = analyse_schema_lib.primary_feature(
      plan.root.nodeset, schema.node_sets[plan.root.nodeset]
  )

  seed_ids_str = ", ".join([f"'{seed_id}'" for seed_id in seed_ids])
  query_parts = [f"GRAPH {graph_name}"]

  def count_edges(node):
    return len(node.children) + sum(count_edges(c.node) for c in node.children)

  total_steps = count_edges(plan.root)

  if len(seed_ids) == 1:
    root_match = f"MATCH (n0 {{{primary_key_feature}: '{seed_ids[0]}'}})"
  else:
    root_match = (
        f"MATCH (n0 WHERE n0.{primary_key_feature} IN ({seed_ids_str}))"
    )
  query_parts.append(root_match)

  if total_steps == 0:
    query_parts.append("RETURN DISTINCT TO_JSON(n0) AS n0")
    return "\n".join(query_parts)

  paths = []
  var_count = 0
  all_vars = ["n0"]

  def build_paths(
      node: config_lib.PlanNode, parent_var: str, current_path: str
  ):
    nonlocal var_count
    if not node.children:
      if current_path:
        paths.append(current_path)
      return

    for i, edge in enumerate(node.children):
      var_count += 1
      child_var = f"n{var_count}"
      edge_var = f"e{var_count}"
      all_vars.extend([child_var, edge_var])

      order_by = "" if debug_sampling else " ORDER BY GENERATE_UUID()"
      if edge.reversed:
        edge_pattern = (
            f"<-[{edge_var}:{edge.edgeset} WHERE {edge_var} IN {{\n    MATCH"
            f" <-[selected_e:{edge.edgeset}]-()\n    FILTER"
            f" IS_FIRST({edge.hop_width}) OVER (PARTITION BY"
            f" DESTINATION_NODE_ID(selected_e){order_by})\n    RETURN"
            f" selected_e\n  }}]-({child_var})"
        )
      else:
        edge_pattern = (
            f"-[{edge_var}:{edge.edgeset} WHERE {edge_var} IN {{\n    MATCH"
            f" -[selected_e:{edge.edgeset}]->()\n    FILTER"
            f" IS_FIRST({edge.hop_width}) OVER (PARTITION BY"
            f" SOURCE_NODE_ID(selected_e){order_by})\n    RETURN selected_e\n "
            f" }}]->({child_var})"
        )

      if i == 0:
        next_path = (
            f"{current_path}{edge_pattern}"
            if current_path == "(n0)"
            else f"{current_path} {edge_pattern}"
            if current_path
            else f"({parent_var}) {edge_pattern}"
        )
      else:
        next_path = f"({parent_var}) {edge_pattern}"

      build_paths(edge.node, child_var, next_path)

  build_paths(plan.root, "n0", "(n0)")

  query_parts.append("OPTIONAL MATCH " + "\n\nOPTIONAL MATCH ".join(paths))

  nodes_to_ret = [v for v in all_vars if v.startswith("n")]
  edges_to_ret = [v for v in all_vars if v.startswith("e")]
  ret_str = ", ".join(
      [f"TO_JSON({v}) AS {v}" for v in nodes_to_ret + edges_to_ret]
  )
  query_parts.append(f"RETURN DISTINCT {ret_str}")

  return "\n".join(query_parts)


# TODO(gbm): Use "_graph_element_to_features" in
# "src/io/gcp/common.py"
def _json_features_to_features(
    values: Dict[str, List[Any]], schema: schema_lib.FeatureSetSchema
) -> Dict[str, np.ndarray]:
  """Converts spanner feature/propertie values into numpy feature values."""
  result = {}
  for feature_name, feature_schema in schema.items():
    dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[feature_schema.format]
    value = np.array(values[feature_name], dtype=dtype)
    if feature_schema.shape:
      shape = [-1] + list(feature_schema.shape)
      value = value.reshape(shape)
    result[feature_name] = value
  return result


# TODO(gbm): This method converts a list of paths into a graph. This is slow.
# Ultimately, we want for the GQL query to return directly a graph (e.g., list
# of nodes and list of edges). When done, we will be able to use the
# methods in "dgf/src/io/gcp/common.py".
def _json_to_in_memory_graphs(
    result: Any,
    schema: schema_lib.GraphSchema,
    seed_ids: List[NodeId],
    root_nodeset: str,
) -> List[in_memory_graph_lib.InMemoryGraph]:
  """Converts the spanner graph JSON results into in memory graphs.

  Args:
    result: The raw JSON result from the Spanner GQL query.
    schema: The graph schema.
    seed_ids: List of seed node IDs.
    root_nodeset: Seeded nodeset.

  Returns:
    A list of InMemoryGraph objects.
  """

  primary_key_feature = analyse_schema_lib.primary_feature(
      root_nodeset, schema.node_sets[root_nodeset]
  )
  raw_graphs_per_seed: Dict[NodeId, RawGraph] = {}

  # Initialises the accumulators
  for seed_id in seed_ids:
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

  # For each sampling path.
  for row in result:
    seed = row[0]
    if not seed:
      continue

    seed_id = seed["properties"][primary_key_feature]
    raw_graph = raw_graphs_per_seed[seed_id]

    for n in range(len(row)):
      item = row[n]
      if not item:
        continue

      if item["kind"] == "node":
        # For each node in the sampling path.
        nodeset_name = item["labels"][0]
        node_id = item["identifier"]
        nodeset = raw_graph.node_sets[nodeset_name]
        raw_features = item["properties"]

        if node_id not in raw_graph.spanner_node_ids:
          raw_graph.spanner_node_ids[node_id] = nodeset.num_nodes
          nodeset.num_nodes += 1
          for feature_name in nodeset.features:
            nodeset.features[feature_name].append(raw_features[feature_name])

      if item["kind"] == "edge":
        # For each edge in the sampling path.
        edgeset_name = item["labels"][0]
        spanner_source_id = item["source_node_identifier"]
        spanner_target_id = item["destination_node_identifier"]

        edge_id = item["identifier"]

        if edge_id not in raw_graph.spanner_edge_ids:
          raw_graph.spanner_edge_ids.add(edge_id)

          edgeset = raw_graph.edge_sets[edgeset_name]
          raw_features = item["properties"]

          source_idx = raw_graph.spanner_node_ids[spanner_source_id]
          target_idx = raw_graph.spanner_node_ids[spanner_target_id]
          edgeset.adjacency.append((source_idx, target_idx))

          for feature_name in edgeset.features:
            edgeset.features[feature_name].append(raw_features[feature_name])

  in_memory_graphs = []

  for seed_id in seed_ids:
    raw_graph = raw_graphs_per_seed[seed_id]

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
