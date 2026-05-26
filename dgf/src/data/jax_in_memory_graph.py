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

"""In memory graph like InMemoryGraph, using JAX instead of NP.

See in_memory_graph.py for the actual documentation.

Note: Parameterizing `InMemoryGraph` to support both numpy and jax
arrays would lead to more complex code, notably due to the increased number of
type annotations required for static checkers in both this file and user code.
"""

import dataclasses
from typing import Dict, Optional

import jax
from jax import tree_util as jax_tree_util


Features = Dict[str, jax.Array]


@jax_tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class JaxInMemoryNodeSet:
  """A Node Set.

  Attributes:
    features: Dictionary of feature names to feature values. Feature values are
      ArrayOrAnys of shape [num_nodes, ...]. If the graph is batched, the batch
      dimension is added first: [batch, num_nodes, ...].
    num_nodes: Number of nodes in the nodeset. Useful to determine the number of
      nodes in a nodeset without features.
  """

  features: Features
  num_nodes: Optional[int] = dataclasses.field(metadata=dict(static=True))


@jax_tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class JaxInMemoryEdgeSet:
  """An Edge Set.

  Attributes:
    adjacency: Array of shape [2, num_edges]. Dtype: int-like. If the graph is
      batched, the batch dimension is added first: [batch, num_edges].
    features: Dictionary of edge set features. Feature values are ArrayOrAnys of
      shape [num_edges, ...]. If the graph is batched, the batch dimension is
      added first: [batch, num_edges, ...].
  """

  adjacency: jax.Array
  features: Features = dataclasses.field(default_factory=dict)

  def num_edges(self) -> int:
    return self.adjacency.shape[1]


@jax_tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class JaxInMemoryGraph:
  """An in-memory generic graph.

  Attributes:
    node_sets: Dictionary of node set name and node set features.
    edge_sets: Dictionary of edge set name and edge set features.
  """

  node_sets: Dict[str, JaxInMemoryNodeSet]
  edge_sets: Dict[str, JaxInMemoryEdgeSet]
