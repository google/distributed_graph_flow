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

"""Defines the GlobalGraphTopology data class for graph statistics.

This module provides a structured data class to hold various global topological
metrics of a graph, such as node count, edge count, density, and connected
components. It also includes methods for auto-calculating derived statistics.
"""

import dataclasses
from typing import Dict, Optional
from dgf.src.analyse.topology import betti_defense
from dgf.src.analyse.topology import node_degree
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
import numpy as np

GraphSchema = schema_lib.GraphSchema
InMemoryGraph = in_memory_graph.InMemoryGraph


## TODO(tewariy): Move to a more appropriate location.
def is_homogeneous_graph(graph: InMemoryGraph, schema: GraphSchema) -> bool:
  """Returns True if the graph is homogeneous, False otherwise."""
  if (
      len(graph.node_sets) == 1
      and len(graph.edge_sets) == 1
      and len(schema.node_sets) == 1
      and len(schema.edge_sets) == 1
  ):
    edge_set = list(schema.edge_sets.values())[0]
    node_set_name = list(schema.node_sets.keys())[0]
    if node_set_name == edge_set.source and node_set_name == edge_set.target:
      return True
  return False


@dataclasses.dataclass
class GlobalGraphTopology:
  """Global graph topology for a given graph.

  Attributes:
    total_nodes: Total number of nodes in the graph.
    total_edges: Total number of edges in the graph.
    avg_degree: Average degree of the graph.
    graph_density: Graph density of the graph.
    num_connected_components: Number of connected components in the graph.
    largest_component_size: Size of the largest connected component.
    isolated_nodes: Number of isolated nodes in the graph.
    graph_diameter: Graph diameter of the graph.
    homophily_ratio: Homophily ratio of the graph.
    degree_distribution: Degree distribution of the graph.
  """

  total_nodes: int
  total_edges: int
  avg_degree: Optional[float] = None
  graph_density: Optional[float] = None
  num_connected_components: Optional[int] = None
  largest_component_size: Optional[int] = None
  isolated_nodes: Optional[int] = None
  graph_diameter: Optional[float] = None
  homophily_ratio: Optional[float] = None
  degree_distribution: Optional[Dict[int, int]] = None
  betti_1: Optional[int] = None

  ## No need to call post_init. Use update_graph_density() to auto-calculate
  ## derived statistics if not provided.
  def update_graph_density(self):
    """Auto-calculates derived statistics if not provided."""
    if self.graph_density is None and self.total_nodes > 1:
      # efficient calculation for undirected graph density
      # Density = 2 * |E| / (|V| * (|V| - 1))
      self.graph_density = (2 * self.total_edges) / (
          self.total_nodes * (self.total_nodes - 1)
      )


def get_in_memory_graph_topology(
    graph: InMemoryGraph,
    schema: GraphSchema,
) -> GlobalGraphTopology:
  """Returns global graph topology for a given in-memory graph.

  Args:
    graph: In-memory graph to compute the topology for.
    schema: Schema of the graph.

  Returns:
    GlobalGraphTopology object containing the topology statistics.
  """

  total_nodes = 0
  total_edges = 0

  for node_set in graph.node_sets.values():
    total_nodes += node_set.num_nodes
  for edge_set in graph.edge_sets.values():
    total_edges += edge_set.num_edges()

  ## Homogeneous graph only
  ## TODO(tewariy): Add support for heterogeneous graphs.
  if is_homogeneous_graph(graph, schema):
    ## Node Degree
    out_degree, in_degree = node_degree.node_degree_edge_list(
        next(iter(graph.edge_sets.values())).adjacency
    )
    node_degree_total = in_degree + out_degree
    average_degree = np.mean(node_degree_total)
    degree, counts = np.unique(node_degree_total, return_counts=True)
    degree_distribution = dict(zip(degree.tolist(), counts.tolist()))

    ## Connected Components
    adj = next(iter(graph.edge_sets.values())).adjacency
    num_nodes = next(iter(graph.node_sets.values())).num_nodes or 0
    cc_labels = betti_defense._connected_components(adj, num_nodes)
    num_cc, cc_counts = np.unique(cc_labels, return_counts=True)
    largest_cc = int(np.max(cc_counts))

    ## Betti-1
    betti_1 = edge_set.num_edges() - num_nodes + int(num_cc.shape[0])
  else:
    average_degree = None
    degree_distribution = None
    num_cc = None
    largest_cc = None
    betti_1 = None

  num_connected_components = num_cc.shape[0] if num_cc is not None else None
  ggt = GlobalGraphTopology(
      total_nodes=total_nodes,
      total_edges=total_edges,
      avg_degree=average_degree,
      num_connected_components=num_connected_components,
      largest_component_size=largest_cc,
      degree_distribution=degree_distribution,
      betti_1=betti_1,
  )

  ggt.update_graph_density()

  return ggt
