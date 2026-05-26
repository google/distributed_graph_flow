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

"""Conver between edge types for heterogeneous graphs."""

import dataclasses
from typing import Iterable

from absl import logging
import apache_beam as beam
from dgf.src.data import distributed_graph

AdjacencyList = distributed_graph.AdjacencyList
EdgeFormat = distributed_graph.EdgeFormat
Edge = distributed_graph.Edge
Graph = distributed_graph.Graph
Neighbor = distributed_graph.Neighbor
PEdgeSet = distributed_graph.PEdgeSet
PAdjacencyList = distributed_graph.PAdjacencyList


# TODO(bmayer): Move this to a common (edge utils?) library.
class ExplodeAdjacencyList(beam.DoFn):
  """Explodes an adjacency edge set into multiple edge sets."""

  def process(self, edges: AdjacencyList) -> Iterable[Edge]:
    source_node_id = edges.source
    for neighbor in edges.neighbors:
      yield Edge(
          source=source_node_id,
          target=neighbor.target,
          id=neighbor.id,
          features=neighbor.features,
      )


# TODO(bmayer): Move this to a common (beam utils?) library.
# Can we make this less verbose and easier to use?
def build_beam_name(
    prefix: str = '',
    graph_piece_name: str = '',
    operation: str = '',
    delimiter: str = '_',
) -> str:
  """Builds a beam name for a graph piece with an operation.

  Args:
    prefix: The prefix to use for the beam name.
    graph_piece_name: The name of the graph piece to use for the beam name.
    operation: The operation to use for the beam name.
    delimiter: The delimiter to use for the beam name.

  Returns:
    The beam name for the graph piece with the operation.
  """
  to_join = []
  if prefix:
    to_join.append(prefix)
  if graph_piece_name:
    to_join.append(graph_piece_name)
  if operation:
    to_join.append(operation)

  if not to_join:
    return ''

  return delimiter.join(to_join)


def convert_to_flat_edges(hgraph: Graph) -> Graph:
  """Converts the edge type to ADJACENCY."""

  if hgraph.edge_format == EdgeFormat.FLAT:
    logging.warning('Edge set type is already ADJACENCY.')
    return hgraph
  elif hgraph.edge_format != EdgeFormat.ADJACENCY:
    raise ValueError('Currently only AJACENCY -> FLAT conversion is supported.')

  _TRANSFORM_PREFIX = 'ConvertToFlatEdges'  # pylint: disable=invalid-name
  flat_edge_sets = {}
  for edge_set_name, edge_set in hgraph.edge_sets.items():
    flat_edge_sets[edge_set_name] = edge_set | build_beam_name(
        prefix=_TRANSFORM_PREFIX,
        graph_piece_name=edge_set_name,
        operation='ExplodeAdjacencyList',
    ) >> beam.ParDo(ExplodeAdjacencyList())

  return dataclasses.replace(
      hgraph,
      edge_sets=flat_edge_sets,
      edge_format=EdgeFormat.FLAT,
  )


def edges_to_adjacency_list(
    source_id: bytes, edges: Iterable[Edge]
) -> AdjacencyList:
  """Helper function to convert a list of edges to an adjacency list.

  Args:
    source_id: The source id of the adjacency list.
    edges: The list of edges to convert.

  Returns:
    An adjacency list.
  """
  neighbors = []
  for edge in edges:
    neighbors.append(
        Neighbor(
            target=edge.target,
            id=edge.id,
            features=edge.features,
        )
    )
  return AdjacencyList(source=source_id, neighbors=neighbors)


class ConvertFlatEdgeSetToAdjacency(beam.PTransform):
  """Converts a flat edge set to an adjacency list."""

  def __init__(
      self,
      transform_prefix: str = 'ConvertFlatEdgeSetToAdjacency',
      edge_set_name: str = '',
  ):
    self._transform_prefix = transform_prefix
    self._edge_set_name = edge_set_name

    # Label the transform with the prefix and edge set name.
    super().__init__(
        build_beam_name(
            prefix=self._transform_prefix,
            graph_piece_name=self._edge_set_name,
        )
    )

  def build_beam_name(self, operation: str) -> str:
    return build_beam_name(
        prefix=self._transform_prefix,
        graph_piece_name=self._edge_set_name,
        operation=operation,
    )

  def expand(self, pcol: PEdgeSet) -> PAdjacencyList:
    return (
        pcol
        | self.build_beam_name('KeyBySourceId')
        >> beam.Map(lambda edge: (edge.source, edge))
        | self.build_beam_name('GroupBySourceID') >> beam.GroupByKey()
        | self.build_beam_name(
            'ConvertGroupedEdgesToAdjacencyList',
        )
        >> beam.MapTuple(edges_to_adjacency_list)
    )


def convert_to_adjacency_list(
    hgraph: Graph,
) -> Graph:
  """Converts the edge type to ADJACENCY."""

  if hgraph.edge_format == EdgeFormat.ADJACENCY:
    logging.info('Edge set type is already ADJACENCY.')
    return hgraph
  elif hgraph.edge_format != EdgeFormat.FLAT:
    raise ValueError(
        'Currently only FLAT -> ADJACENCY conversion is supported.'
    )

  _TRANSFORM_PREFIX = 'ConvertEdgesToAdjacencyList'  # pylint: disable=invalid-name
  adjacency_lists = {}
  for edge_set_name, edge_set in hgraph.edge_sets.items():
    adjacency_lists[edge_set_name] = edge_set | ConvertFlatEdgeSetToAdjacency(
        edge_set_name
    )

  return dataclasses.replace(
      hgraph,
      edge_sets=adjacency_lists,
      edge_format=EdgeFormat.ADJACENCY,
  )
