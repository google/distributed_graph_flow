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

"""Low level message passing primitives."""

from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
import jax.numpy as jnp
import sparse_deferred as sd
import sparse_deferred.jax as sdjax


def core_graph_to_sd_sparse_matrix(
    adjacency: jnp.ndarray, num_nodes_source: int, num_nodes_target: int
) -> sd.SparseMatrix:
  """Converts an adjacency list to a sparse_deferred SparseMatrix."""
  # TODO(gbm): Benchmark and validate.
  col_indices = adjacency[0]  # One-at-a-time is TPU-friendly.
  row_indices = adjacency[1]  # One-at-a-time is TPU-friendly.
  return sd.SparseMatrix(
      sdjax.engine,
      indices=(row_indices, col_indices),
      dense_shape=(
          num_nodes_target,
          num_nodes_source,
      ),
  )


def graph_to_sd_sparse_matrix(
    graph: jax_in_memory_graph.JaxInMemoryGraph,
    schema: schema_lib.GraphSchema,
    edgeset_name: str,
) -> sd.SparseMatrix:
  """Converts an adjacency list to a sparse_deferred SparseMatrix.

  Args:
    graph: The graph structure containing the adjacency list.
    schema: The graph schema.
    edgeset_name: The name of the edgeset to convert.

  Returns:
    A sparse_deferred SparseMatrix representing the adjacency matrix.
  """
  adjacency = graph.edge_sets[edgeset_name].adjacency
  source_nodeset = schema.edge_sets[edgeset_name].source
  target_nodeset = schema.edge_sets[edgeset_name].target
  return core_graph_to_sd_sparse_matrix(
      adjacency,
      graph.node_sets[source_nodeset].num_nodes,
      graph.node_sets[target_nodeset].num_nodes,
  )
