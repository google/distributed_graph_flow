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

"""Compute topology statistics, in process, on InMemoryGraph."""

import dataclasses
from typing import Iterable, List
from dgf.src.analyse import histogram as analyse_hist
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
import numpy as np


@dataclasses.dataclass
class _NodeSetAccumulator:
  node_counts: List[int]


@dataclasses.dataclass
class _EdgeSetAccumulator:
  edge_counts: List[int]
  in_degrees: List[np.ndarray]
  out_degrees: List[np.ndarray]


def _get_num_nodes(nodeset: in_memory_graph_lib.InMemoryNodeSet) -> int:
  if nodeset.num_nodes is not None:
    return nodeset.num_nodes
  if nodeset.features:
    return next(iter(nodeset.features.values())).shape[0]
  return 0


def topology_statistics(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    num_bins: int = 32,
) -> statistics_lib.GraphTopologyStatistics:
  """Computes topology statistics for a single InMemoryGraph.

  Usage example:

  ```python
  # Read a graph
  graph, schema = dgf.io.read_graph("/tmp/my_graph")

  # Compute the topology statistics
  topo_stats = dgf.analyse.topology_statistics(graph, schema)

  # Print the statistics (shows beautiful ASCII art histograms)
  print(topo_stats)
  ```

  Args:
    graph: An in-memory graph.
    schema: Schema of the graph.
    num_bins: Number of bins to use for degree distribution histograms.

  Returns:
    GraphTopologyStatistics object.
  """
  return topology_statistics_from_graphs(
      graphs=[graph],
      schema=schema,
      num_bins=num_bins,
  )


def topology_statistics_from_graphs(
    graphs: Iterable[in_memory_graph_lib.InMemoryGraph],
    schema: schema_lib.GraphSchema,
    num_bins: int = 32,
) -> statistics_lib.GraphTopologyStatistics:
  """Computes topology statistics for a set of InMemoryGraphs.

  Usage example:

  ```python
  # Read graphs
  graphs, schema = dgf.io.read_tfgnn_graphs("/my/data@10")

  # Compute the topology statistics
  topo_stats = dgf.analyse.topology_statistics_from_graphs(graphs, schema)

  # Print the statistics
  print(topo_stats)
  ```

  Args:
    graphs: An iterable of in-memory graphs.
    schema: Schema of the graphs.
    num_bins: Number of bins to use for histograms.

  Returns:
    GraphTopologyStatistics object.
  """
  # Initialize accumulators
  node_accumulators = {
      name: _NodeSetAccumulator(node_counts=[]) for name in schema.node_sets
  }
  edge_accumulators = {
      name: _EdgeSetAccumulator(edge_counts=[], in_degrees=[], out_degrees=[])
      for name in schema.edge_sets
  }

  num_graphs = 0
  for graph in graphs:
    num_graphs += 1

    # Collect node counts
    for name in schema.node_sets:
      nodeset = graph.node_sets[name]
      node_accumulators[name].node_counts.append(_get_num_nodes(nodeset))

    # Collect edge counts and degrees
    for name in schema.edge_sets:
      edgeset = graph.edge_sets[name]
      edge_accumulators[name].edge_counts.append(edgeset.num_edges())

      edge_schema = schema.edge_sets[name]
      source_nodeset = graph.node_sets[edge_schema.source]
      target_nodeset = graph.node_sets[edge_schema.target]

      num_sources = _get_num_nodes(source_nodeset)
      num_targets = _get_num_nodes(target_nodeset)

      # Compute in-degrees for this graph
      target_indices = edgeset.adjacency[1]
      in_degrees = np.zeros(num_targets, dtype=int)
      if target_indices.size > 0:
        unique_targets, counts = np.unique(target_indices, return_counts=True)
        in_degrees[unique_targets] = counts
      edge_accumulators[name].in_degrees.append(in_degrees)

      # Compute out-degrees for this graph
      source_indices = edgeset.adjacency[0]
      out_degrees = np.zeros(num_sources, dtype=int)
      if source_indices.size > 0:
        unique_sources, counts = np.unique(source_indices, return_counts=True)
        out_degrees[unique_sources] = counts
      edge_accumulators[name].out_degrees.append(out_degrees)

  if num_graphs == 0:
    raise ValueError("The input 'graphs' iterable was empty.")

  # Build final statistics
  node_sets_stats = {}
  for name in node_accumulators:
    acc = node_accumulators[name]
    nodes_arr = np.array(acc.node_counts)
    node_sets_stats[name] = statistics_lib.NodeSetTopologyStatistics(
        num_nodes=analyse_hist.make_histogram(
            nodes_arr, num_bins=num_bins, log_scale=True, is_integer=True
        )
    )

  edge_sets_stats = {}
  for name in edge_accumulators:
    acc = edge_accumulators[name]
    edges_arr = np.array(acc.edge_counts)
    num_edges_hist = analyse_hist.make_histogram(
        edges_arr, num_bins=num_bins, log_scale=True, is_integer=True
    )

    # Aggregate degrees by concatenating arrays from all graphs
    in_degrees_flat = np.concatenate(acc.in_degrees)
    out_degrees_flat = np.concatenate(acc.out_degrees)

    in_degree_dist = analyse_hist.make_histogram(
        in_degrees_flat, num_bins=num_bins, log_scale=True, is_integer=True
    )
    out_degree_dist = analyse_hist.make_histogram(
        out_degrees_flat, num_bins=num_bins, log_scale=True, is_integer=True
    )

    edge_sets_stats[name] = statistics_lib.EdgeSetTopologyStatistics(
        num_edges=num_edges_hist,
        in_degree_distribution=in_degree_dist,
        out_degree_distribution=out_degree_dist,
    )

  return statistics_lib.GraphTopologyStatistics(
      node_sets=node_sets_stats,
      edge_sets=edge_sets_stats,
      num_graphs=num_graphs,
  )
