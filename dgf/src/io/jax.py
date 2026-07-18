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

"""Conversion to JAX related graph objects."""

from typing import Union
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import tf_in_memory_graph as tf_in_memory_graph_lib
import jax
import numpy as np


def graph_to_jax_graph(
    src: Union[
        in_memory_graph_lib.InMemoryGraph,
        tf_in_memory_graph_lib.TFInMemoryGraph,
    ],
    cast_arrays: bool = True,
) -> jax_in_memory_graph_lib.JaxInMemoryGraph:
  """Converts a (NumPy) in-memory graph into a JAX in-memory graph.

  Args:
    src: The source graph to convert.
    cast_arrays: Whether to cast the arrays to jax array. If False, the original
      arrays are returned. This can be used to interact with jax2tf.
  """
  # TODO(gbm): Add control for the device placement of features and adjacencies.

  def _asarray(x):
    if not cast_arrays:
      return x
    return jax.numpy.asarray(x)

  jax_node_sets = {}
  for node_set_name, node_set in src.node_sets.items():
    jax_features = {k: _asarray(v) for k, v in node_set.features.items()}
    jax_node_sets[node_set_name] = jax_in_memory_graph_lib.JaxInMemoryNodeSet(
        features=jax_features, num_nodes=node_set.num_nodes  # pyrefly: ignore[bad-argument-type]
    )

  jax_edge_sets = {}
  for edge_set_name, edge_set in src.edge_sets.items():
    jax_adjacency = _asarray(edge_set.adjacency)
    jax_features = {k: _asarray(v) for k, v in edge_set.features.items()}
    jax_edge_sets[edge_set_name] = jax_in_memory_graph_lib.JaxInMemoryEdgeSet(
        adjacency=jax_adjacency, features=jax_features
    )

  return jax_in_memory_graph_lib.JaxInMemoryGraph(
      node_sets=jax_node_sets, edge_sets=jax_edge_sets
  )


def jax_graph_to_graph(
    src: jax_in_memory_graph_lib.JaxInMemoryGraph,
) -> in_memory_graph_lib.InMemoryGraph:
  """Converts a jax in memory graph into a (Numpy) in memory graph."""
  node_sets = {}
  for node_set_name, node_set in src.node_sets.items():
    features = {k: np.asarray(v) for k, v in node_set.features.items()}
    node_sets[node_set_name] = in_memory_graph_lib.InMemoryNodeSet(
        features=features, num_nodes=node_set.num_nodes
    )

  edge_sets = {}
  for edge_set_name, edge_set in src.edge_sets.items():
    adjacency = np.asarray(edge_set.adjacency)
    features = {k: np.asarray(v) for k, v in edge_set.features.items()}
    edge_sets[edge_set_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=adjacency, features=features
    )

  return in_memory_graph_lib.InMemoryGraph(
      node_sets=node_sets, edge_sets=edge_sets
  )
