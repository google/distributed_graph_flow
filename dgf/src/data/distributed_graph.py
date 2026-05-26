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

"""Distributed graph data structures."""

import dataclasses
import enum
from typing import Dict, List, NamedTuple, Optional, Union

import apache_beam as beam
from apache_beam import coders

from dgf.src.data import beam_coders as _  # pylint: disable=unused-import
from dgf.src.data import in_memory_graph
from dgf.src.data import numpy_coder as _  # pylint: disable=unused-import
from dgf.src.data import schema as schema_lib


PCollection = beam.PCollection

# TODO(bmayer): Move the non-beam objects to a `graph.py` definition file.
NodeId = bytes | int
EdgeId = bytes | int
SourceId = bytes | int
TargetId = bytes | int

Features = in_memory_graph.Features


@dataclasses.dataclass(frozen=True)
class Node:
  id: NodeId
  features: Features | None = None


class EdgeFormat(enum.Enum):
  UNDEFINED = 0
  ADJACENCY = 1
  FLAT = 2


@dataclasses.dataclass(frozen=True)
class Edge:
  """A single flat edge."""

  source: SourceId
  target: TargetId

  # ID of this edge entry, e.g., join key with `edge_features`
  id: Optional[EdgeId] = None

  # Optionally pack the external `edge_features` directly into the edge or use
  # for special features such as weights and temporal (datetime) data.
  features: Optional[Features] = None


@dataclasses.dataclass(frozen=True)
class Neighbor:
  target: TargetId
  id: Optional[EdgeId] = None
  features: Optional[Features] = None


# Adjacency list of edges.
@dataclasses.dataclass(frozen=True)
class AdjacencyList:
  source: SourceId
  neighbors: List[Neighbor]


# Defined to optionally store edge features separately from topology.
@dataclasses.dataclass(frozen=True)
class EdgeFeatures:
  id: EdgeId
  features: Features


PNode = PCollection[Node]
# Flattened list of edges.
PEdge = PCollection[Edge]
PAdjacencyList = PCollection[AdjacencyList]
PEdgeSet = Union[PEdge, PAdjacencyList]
PEdgeFeatures = PCollection[EdgeFeatures]


@dataclasses.dataclass(frozen=True)
class HomogeneousGraph:
  """A (potentially distributed) homogeneous graph."""

  nodes: PNode
  edges: PEdgeSet
  edge_features: Optional[PEdgeFeatures] = None

  # Track the edge set format.
  edge_format: EdgeFormat = EdgeFormat.UNDEFINED


@dataclasses.dataclass(frozen=True)
class Graph:
  """A (potentially distributed) heterogeneous graph."""

  # List the named node sets, edge sets (with domain and range node sets).
  schema: schema_lib.GraphSchema

  # Name -> NodeSet
  node_sets: Dict[str, PNode]

  # Name -> EdgeSet
  edge_sets: Dict[str, PEdgeSet]

  edge_format: EdgeFormat = EdgeFormat.UNDEFINED

  # Optional named -> EdgeFeatures
  edge_features: Optional[Dict[str, PEdgeFeatures]] = None


def heterogeneous_graph_from_pieces(
    p: beam.pvalue.PBegin,
    schema: schema_lib.GraphSchema,
    node_sets: Dict[str, List[Node]],
    edge_sets: Dict[str, List[Edge]],
    edge_format: Optional[EdgeFormat] = EdgeFormat.UNDEFINED,
    edge_features: Optional[Dict[str, List[EdgeFeatures]]] = None,
    stage_prefix: str = "",
) -> Graph:
  """Creates a distributed Graph from in-memory pieces.

  This is mostly useful for testing purposes.

  Args:
    p: A beam PBegin instance indicating the start of a pipeline.
    schema: The graph schema.
    node_sets: A dictionary of node set name to list of nodes.
    edge_sets: A dictionary of edge set name to list of edges.
    edge_format: The format of the edges.
    edge_features: An optional dictionary of edge set name to list of edge
      features.
    stage_prefix: A prefix to add to the stage names.

  Returns:
    A distributed Graph.
  """
  pnode_sets = {}
  for name, nodes in node_sets.items():
    pnode_sets[name] = p | f"{stage_prefix}CreateNodeSet_{name}" >> beam.Create(
        nodes
    )

  pedge_sets = {}
  for name, edges in edge_sets.items():
    pedge_sets[name] = p | f"{stage_prefix}CreateEdgeset_{name}" >> beam.Create(
        edges
    )

  pedge_features = None
  if edge_features is not None:
    pedge_features = {}
    for name, edge_features in edge_features.items():
      pedge_features[name] = (
          p
          | f"{stage_prefix}CreateEdgeFeatures_{name}"
          >> beam.Create(edge_features)
      )

  return Graph(
      schema=schema,
      node_sets=pnode_sets,
      edge_sets=pedge_sets,
      edge_format=edge_format,
      edge_features=pedge_features,
  )


class KeyedInMemoryGraph(NamedTuple):
  key: Optional[bytes]
  graph: in_memory_graph.InMemoryGraph


PKeyedInMemoryGraph = beam.PCollection[KeyedInMemoryGraph]


coders.registry.register_coder(PEdge, coders.RowCoder)
coders.registry.register_coder(PNode, coders.RowCoder)
coders.registry.register_coder(Graph, coders.RowCoder)
