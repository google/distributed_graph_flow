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

"""Test utilities common to Jax and GNNs."""

from dgf.src.learning.jax.layers import homo_gnn_sparse_deferred as homo_gnn
import jax
import jax.numpy as jnp
from sparse_deferred.structs import graph_struct as graph_struct_lib

GraphStruct = graph_struct_lib.GraphStruct


# TODO(bmayer): Add tests with edge features.
def generate_test_graph(
    dim: int = 8,
    nodeset_name: str = homo_gnn.DEFAULT_NODESET_NAME,
    edgeset_name: str = homo_gnn.DEFAULT_EDGESET_NAME,
    node_feature_name: str = homo_gnn.DEFAULT_NODE_FEATURE_NAME,
) -> GraphStruct:
  num_nodes = 4
  key = jax.random.PRNGKey(0)
  return GraphStruct.new(
      nodes={
          nodeset_name: {
              node_feature_name: jax.random.normal(key, (num_nodes, dim))
          }
      },
      edges={
          edgeset_name: (
              (jnp.array([0, 0, 0, 1, 3]), jnp.array([1, 2, 3, 2, 0])),
              {},
          )
      },
      schema={edgeset_name: (nodeset_name, nodeset_name)},
  )
