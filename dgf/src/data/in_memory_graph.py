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

"""An in-memory generic graph."""

import dataclasses
from typing import Dict, Optional
import numpy as np

Features = Dict[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class InMemoryNodeSet:
  """A Node Set.

  Attributes:
    num_nodes: Number of nodes in the nodeset. Useful to determine the number of
      nodes in a nodeset without features.
    features: Dictionary of feature names to feature values. Feature values are
      ArrayOrAnys of shape [num_nodes, <feature shape>].
  """

  num_nodes: Optional[int]
  features: Features = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class InMemoryEdgeSet:
  """An Edge Set.

  Attributes:
    adjacency: Array of shape [2, num_edges]. Dtype: int-like. If the graph is
      batched, the batch dimension is added first: [batch, num_edges].
    features: Dictionary of edge set features. Feature values are ArrayOrAnys of
      shape [num_edges,  <feature shape>].
  """

  adjacency: np.ndarray
  features: Features = dataclasses.field(default_factory=dict)

  def num_edges(self) -> int:
    return self.adjacency.shape[1]


@dataclasses.dataclass(frozen=True)
class InMemoryGraph:
  """An in-memory generic graph.

  Attributes:
    node_sets: Dictionary of node set name and node set features.
    edge_sets: Dictionary of edge set name and edge set features.
  """

  node_sets: Dict[str, InMemoryNodeSet]
  edge_sets: Dict[str, InMemoryEdgeSet]
