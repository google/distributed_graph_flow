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

"""In memory graph like InMemoryGraph, using TF instead of NP.

See in_memory_graph.py for the actual documentation.
"""

from typing import Dict, Mapping, Union
import tensorflow as tf

Array = Union[
    tf.Tensor, tf.RaggedTensor, tf.SparseTensor
]  # Can be dense, sparse or ragged.
Features = Mapping[str, Array]


# A TF Graph Dict is equivalent to a TF Graph (i.e. TFInMemoryGraph class
# defined below), but where all the fields are flattened in a single dictionary.
# A TF Graph Dict is useful for passing graphs in some systems e.g. VertexAI.
TFInMemoryGraphDict = Dict[str, Array]


class TFInMemoryNodeSet(tf.experimental.ExtensionType):
  """A Node Set.

  Attributes:
    num_nodes: Number of nodes in the nodeset. Useful to determine the number of
      nodes in a nodeset without features.
    features: Dictionary of feature names to feature values. Feature values are
      ArrayOrAnys of shape [num_nodes, <feature shape>].
  """

  __name__ = 'dgf.data.TFInMemoryNodeSet'
  num_nodes: Union[int, Array]
  features: Features = {}


class TFInMemoryEdgeSet(tf.experimental.ExtensionType):
  """An Edge Set.

  Attributes:
    adjacency: Array of shape [2, num_edges]. Dtype: int-like. If the graph is
      batched, the batch dimension is added first: [batch, num_edges].
    features: Dictionary of edge set features. Feature values are ArrayOrAnys of
      shape [num_edges,  <feature shape>].
  """

  __name__ = 'dgf.data.TFInMemoryEdgeSet'
  adjacency: Array
  features: Features = {}


class TFInMemoryGraph(tf.experimental.ExtensionType):
  """An in-memory generic graph.

  Attributes:
    node_sets: Dictionary of node set name and node set features.
    edge_sets: Dictionary of edge set name and edge set features.
  """

  __name__ = 'dgf.data.TFInMemoryGraph'
  node_sets: Mapping[str, TFInMemoryNodeSet]
  edge_sets: Mapping[str, TFInMemoryEdgeSet]
