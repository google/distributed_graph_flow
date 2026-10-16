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

"""Canonicalizes an InMemoryGraph by sorting nodes and edges deterministically."""

from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
import numpy as np


def _resolve_node_primary_key(
    nodeset_name: str,
    nodeset_schema: schema_lib.NodeSchema,
) -> str:
  """Returns the primary key feature name for a nodeset."""
  primary_key = analyse_schema_lib.primary_feature_or_none(
      nodeset_name, nodeset_schema
  )
  if primary_key is not None:
    return primary_key
  return analyse_schema_lib.infer_most_likely_primary_key(
      nodeset_name, nodeset_schema
  )


def _resolve_edge_primary_key_or_none(
    edgeset_name: str,
    edgeset_schema: schema_lib.EdgeSchema,
) -> str | None:
  """Returns the primary key feature name for an edgeset, or None if absent."""
  primary_key = analyse_schema_lib.primary_feature_or_none(
      edgeset_name, edgeset_schema
  )
  if primary_key is not None:
    return primary_key
  return analyse_schema_lib.infer_most_likely_primary_key_or_none(
      edgeset_name, edgeset_schema
  )


def canonicalize_graph(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
) -> in_memory_graph_lib.InMemoryGraph:
  """Sorts nodes and edges of an in-memory graph into a canonical order.

  Nodes in each nodeset are sorted in ascending order by their primary key
  feature, and all node features are permuted accordingly. Edge endpoints are
  remapped to the new sorted node indices, and edges in each edgeset are sorted
  by their edge primary key (if present) or lexicographically by
  `(source_index, target_index)`.

  Usage example:

  ```python
  canon_graph = dgf.transform.canonicalize_graph(graph, schema)
  ```

  Args:
    graph: The input `InMemoryGraph` to canonicalize.
    schema: The `GraphSchema` describing `graph`.

  Returns:
    A new `InMemoryGraph` with canonically ordered nodes and edges.
  """
  new_node_sets: dict[str, in_memory_graph_lib.InMemoryNodeSet] = {}
  node_permutations: dict[str, np.ndarray] = {}

  for nodeset_name, nodeset_schema in schema.node_sets.items():
    nodeset = graph.node_sets[nodeset_name]
    num_nodes = nodeset.num_nodes or 0
    if num_nodes == 0:
      node_permutations[nodeset_name] = np.empty(0, dtype=np.int64)
      new_node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
          num_nodes=0,
          features={
              feature_name: nodeset.features[feature_name]
              for feature_name in nodeset_schema.features
          },
      )
      continue

    primary_key_name = _resolve_node_primary_key(nodeset_name, nodeset_schema)
    primary_key_values = nodeset.features[primary_key_name]
    sort_indices = np.argsort(primary_key_values, kind="stable")

    inverse_permutation = np.empty(num_nodes, dtype=np.int64)
    inverse_permutation[sort_indices] = np.arange(num_nodes, dtype=np.int64)
    node_permutations[nodeset_name] = inverse_permutation

    new_features = {
        feature_name: nodeset.features[feature_name][sort_indices]
        for feature_name in nodeset_schema.features
    }
    new_node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=num_nodes,
        features=new_features,
    )

  new_edge_sets: dict[str, in_memory_graph_lib.InMemoryEdgeSet] = {}
  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    edgeset = graph.edge_sets[edgeset_name]
    if edgeset.adjacency.shape[1] == 0:
      new_edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
          adjacency=np.zeros((2, 0), dtype=edgeset.adjacency.dtype),
          features={
              feature_name: edgeset.features[feature_name]
              for feature_name in edgeset_schema.features
          },
      )
      continue

    source_permutation = node_permutations[edgeset_schema.source]
    target_permutation = node_permutations[edgeset_schema.target]
    new_sources = source_permutation[edgeset.adjacency[0]]
    new_targets = target_permutation[edgeset.adjacency[1]]

    edge_primary_key = _resolve_edge_primary_key_or_none(
        edgeset_name, edgeset_schema
    )
    if edge_primary_key is not None and edge_primary_key in edgeset.features:
      edge_order = np.argsort(edgeset.features[edge_primary_key], kind="stable")
    else:
      edge_order = np.lexsort((new_targets, new_sources))

    new_adjacency = np.stack(
        [new_sources[edge_order], new_targets[edge_order]], axis=0
    ).astype(edgeset.adjacency.dtype, copy=False)
    new_features = {
        feature_name: edgeset.features[feature_name][edge_order]
        for feature_name in edgeset_schema.features
    }
    new_edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=new_adjacency,
        features=new_features,
    )

  return in_memory_graph_lib.InMemoryGraph(
      node_sets=new_node_sets,
      edge_sets=new_edge_sets,
      timestamp=graph.timestamp,
  )
