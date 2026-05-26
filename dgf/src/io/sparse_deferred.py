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

"""Import / export data to sparse deferred."""

from typing import Dict, Optional
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import schema as schema_lib
import jax.numpy as jnp
import numpy as np
import sparse_deferred as sd
from sparse_deferred.structs import graph_struct as sd_struct_lib


def _features_to_numpy(src: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
  return {k: np.asarray(v, copy=False) for k, v in src.items()}


def _features_to_jax(src: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
  return {k: jnp.asarray(v, copy=False) for k, v in src.items()}


def sparse_deferred_struct_to_graph(
    sd_graph_struct: sd_struct_lib.GraphStruct,
) -> in_memory_graph_lib.InMemoryGraph:
  """Converts a Sparse Deferred struct into an in-memory graph.

  Args:
    sd_graph_struct: The input graph in `sd_struct_lib.GraphStruct` format.

  Returns:
    An `InMemoryGraph` instance containing the same graph data.
  """
  in_memory_node_sets = {}
  for node_name, sd_node_features in sd_graph_struct.nodes.items():
    if sd_node_features:
      num_nodes = next(iter(sd_node_features.values())).shape[0]
    else:
      num_nodes = None
    in_memory_node_sets[node_name] = in_memory_graph_lib.InMemoryNodeSet(
        features=_features_to_numpy(sd_node_features),
        num_nodes=num_nodes,
    )

  in_memory_edge_sets = {}
  for edge_name, (endpoints, sd_edge_features) in sd_graph_struct.edges.items():
    sources, targets = endpoints
    # Stack sources and targets to get adjacency of shape [2, num_edges].
    adjacency = np.stack([sources, targets])
    in_memory_edge_sets[edge_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=adjacency,
        features=_features_to_numpy(sd_edge_features),
    )

  return in_memory_graph_lib.InMemoryGraph(
      node_sets=in_memory_node_sets, edge_sets=in_memory_edge_sets
  )


def sparse_deferred_struct_to_jax_graph(
    sd_graph_struct: sd_struct_lib.GraphStruct,
) -> jax_in_memory_graph_lib.JaxInMemoryGraph:
  """Converts a Sparse Deferred struct into a JAX in-memory graph.

  Args:
    sd_graph_struct: The input graph in `sd_struct_lib.GraphStruct` format.

  Returns:
    An `InMemoryGraph` instance containing the same graph data.
  """
  in_memory_node_sets = {}
  for node_name, sd_node_features in sd_graph_struct.nodes.items():
    if sd_node_features:
      num_nodes = next(iter(sd_node_features.values())).shape[0]
    else:
      num_nodes = None
    in_memory_node_sets[node_name] = jax_in_memory_graph_lib.JaxInMemoryNodeSet(
        features=_features_to_jax(sd_node_features),
        num_nodes=num_nodes,
    )

  in_memory_edge_sets = {}
  for edge_name, (endpoints, sd_edge_features) in sd_graph_struct.edges.items():
    sources, targets = endpoints
    # Stack sources and targets to get adjacency of shape [2, num_edges].
    adjacency = jnp.stack([sources, targets])
    in_memory_edge_sets[edge_name] = jax_in_memory_graph_lib.JaxInMemoryEdgeSet(
        adjacency=adjacency,
        features=_features_to_jax(sd_edge_features),
    )

  return jax_in_memory_graph_lib.JaxInMemoryGraph(
      node_sets=in_memory_node_sets, edge_sets=in_memory_edge_sets
  )


def graph_to_sparse_deferred_struct(
    in_memory_graph: in_memory_graph_lib.InMemoryGraph,
    schema: Optional[schema_lib.GraphSchema] = None,
) -> sd_struct_lib.GraphStruct:
  """Converts an in-memory graph into a Sparse Deferred struct.

  Args:
    in_memory_graph: The input graph in `InMemoryGraph` format.
    schema: An optional `GraphSchema` instance describing the graph structure.
      If provided, it will be converted to a Sparse Deferred schema.

  Returns:
    A `sd_struct_lib.GraphStruct` instance containing the same graph data.
  """

  sd_nodes = {
      node_name: in_memory_node_set.features
      for node_name, in_memory_node_set in in_memory_graph.node_sets.items()
  }

  sd_edges = {}
  for edge_name, in_memory_edge_set in in_memory_graph.edge_sets.items():
    # InMemoryEdgeSet.adjacency is shape [2, num_edges].
    # The first row contains source indices, the second row contains target
    # indices.
    sources = in_memory_edge_set.adjacency[0]
    targets = in_memory_edge_set.adjacency[1]
    endpoints = (sources, targets)
    sd_edges[edge_name] = (endpoints, in_memory_edge_set.features)

  if schema is not None:
    sd_schema = schema_to_sparse_deferred_schema(schema)
  else:
    sd_schema = None

  return sd_struct_lib.GraphStruct.new(
      nodes=sd_nodes,
      edges=sd_edges,
      schema=sd_schema,
  )


def jax_graph_to_sparse_deferred_struct(
    in_memory_graph: jax_in_memory_graph_lib.JaxInMemoryGraph,
    schema: Optional[schema_lib.GraphSchema] = None,
) -> sd_struct_lib.GraphStruct:
  """Converts a Jax in memory graph into a Sparse Deferred struct.

  Args:
    in_memory_graph: The input graph in `JaxInMemoryGraph` format.
    schema: An optional `GraphSchema` instance describing the graph structure.
      If provided, it will be converted to a Sparse Deferred schema.

  Returns:
    A `sd_struct_lib.GraphStruct` instance containing the same graph data.
  """

  sd_nodes = {
      node_name: in_memory_node_set.features
      for node_name, in_memory_node_set in in_memory_graph.node_sets.items()
  }

  sd_edges = {}
  for edge_name, in_memory_edge_set in in_memory_graph.edge_sets.items():
    # InMemoryEdgeSet.adjacency is shape [2, num_edges].
    # The first row contains source indices, the second row contains target
    # indices.
    sources = in_memory_edge_set.adjacency[0]
    targets = in_memory_edge_set.adjacency[1]
    endpoints = (sources, targets)
    sd_edges[edge_name] = (endpoints, in_memory_edge_set.features)

  if schema is not None:
    sd_schema = schema_to_sparse_deferred_schema(schema)
  else:
    sd_schema = None

  return sd_struct_lib.GraphStruct.new(
      nodes=sd_nodes,
      edges=sd_edges,
      schema=sd_schema,
  )


def schema_to_sparse_deferred_schema(
    schema: schema_lib.GraphSchema,
) -> sd_struct_lib.Schema:
  """Converts a DGF `GraphSchema` into a Sparse Deferred schema.

  Args:
    schema: The input schema in `schema_lib.GraphSchema` format.

  Returns:
    A `sd_struct_lib.Schema` instance representing the graph schema.
  """

  return {
      edge_name: (edge_schema.source, edge_schema.target)
      for edge_name, edge_schema in schema.edge_sets.items()
  }
