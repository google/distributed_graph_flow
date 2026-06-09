# Copyright 2024 Google LLC.
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
"""Topological anomaly detection using Betti numbers for InMemoryGraph.

This module provides a lightweight defense layer for detecting structural
anomalies in graph inputs by analysing their first Betti number (beta_1),
which counts independent cycles.  It is intended to be used as a
pre-processing filter or safety guard in graph neural network pipelines.
"""

from typing import Optional

import numpy as np

from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib

GraphSchema = schema_lib.GraphSchema
InMemoryGraph = in_memory_graph.InMemoryGraph
InMemoryEdgeSet = in_memory_graph.InMemoryEdgeSet


def _connected_components(
    adjacency: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
  """Returns connected-component labels for each node.

  Uses a simple union-find on the undirected view of the edge list.

  Args:
    adjacency: Integer array of shape [2, num_edges].
    num_nodes: Total number of nodes in the graph.

  Returns:
    Integer array of shape [num_nodes] where equal values denote nodes
    that belong to the same connected component.
  """
  parent = np.arange(num_nodes, dtype=np.int32)

  def _find(x: np.ndarray) -> np.ndarray:
    """Vectorised find with path compression."""
    # Iterative path compression – at most log* N iterations.
    while True:
      p = parent[x]
      unchanged = p == x
      x = np.where(unchanged, x, p)
      if np.all(unchanged):
        break
    return x

  # Union step for every undirected edge.
  if adjacency.shape[1] > 0:
    src = adjacency[0]
    tgt = adjacency[1]
    root_src = _find(src)
    root_tgt = _find(tgt)
    smaller = np.minimum(root_src, root_tgt)
    larger = np.maximum(root_src, root_tgt)
    parent[larger] = smaller
    # Second pass to compress any remaining paths.
    parent[:] = _find(np.arange(num_nodes))

  return parent


def _num_connected_components(
    adjacency: np.ndarray,
    num_nodes: int,
) -> int:
  """Returns the number of connected components."""
  labels = _connected_components(adjacency, num_nodes)
  return int(np.unique(labels).shape[0])


def calculate_betti_1(
    graph: InMemoryGraph,
    schema: GraphSchema,
) -> int:
  """Calculates the first Betti number (beta_1) of a homogeneous graph.

  beta_1 = |E| - |V| + C

  where |E| is the number of edges, |V| the number of vertices, and C the
  number of connected components.

  Args:
    graph: An InMemoryGraph.  Must be homogeneous (single node set, single
      edge set, source == target).
    schema: The GraphSchema for the graph.

  Returns:
    The integer beta_1 value.

  Raises:
    ValueError: If the graph is not homogeneous.
  """
  if not _is_homogeneous_graph(graph, schema):
    raise ValueError(
        "calculate_betti_1 currently only supports homogeneous graphs.")

  node_set = next(iter(graph.node_sets.values()))
  edge_set = next(iter(graph.edge_sets.values()))
  num_nodes = node_set.num_nodes or 0
  num_edges = edge_set.num_edges()

  if num_nodes == 0:
    return 0

  c = _num_connected_components(edge_set.adjacency, num_nodes)
  return num_edges - num_nodes + c


def is_anomalous(
    graph: InMemoryGraph,
    schema: GraphSchema,
    expected_max_betti: int = 1,
) -> bool:
  """Flags a graph whose first Betti number exceeds a threshold.

  Args:
    graph: An InMemoryGraph.  Must be homogeneous.
    schema: The GraphSchema for the graph.
    expected_max_betti: Maximum expected beta_1 for a benign graph.

  Returns:
    True if the graph is topologically anomalous.
  """
  betti_1 = calculate_betti_1(graph, schema)
  return betti_1 > expected_max_betti


def _is_homogeneous_graph(
    graph: InMemoryGraph,
    schema: GraphSchema,
) -> bool:
  """Returns True if the graph is homogeneous."""
  if (
      len(graph.node_sets) == 1
      and len(graph.edge_sets) == 1
      and len(schema.node_sets) == 1
      and len(schema.edge_sets) == 1
  ):
    edge_set = next(iter(schema.edge_sets.values()))
    node_set_name = next(iter(schema.node_sets.keys()))
    if node_set_name == edge_set.source and node_set_name == edge_set.target:
      return True
  return False
