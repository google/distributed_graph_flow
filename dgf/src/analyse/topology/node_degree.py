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

"""Computes node degrees from graph representations.

This module provides functions to calculate node properties, such as in-degree
and out-degree, from various graph representations like edge lists.
"""

import numpy as np


def node_degree_edge_list(adj: np.ndarray, num_nodes: int | None = None):
  """Computes the in-degree and out-degree for each node from an edge list.

  Args:
    adj: An array of shape [2, num_edges] representing the edge list. adj[0]
      contains the source nodes and adj[1] contains the destination nodes.
    num_nodes: The total number of nodes in the graph. If None, it's inferred
      from the maximum node index in `adj`.

  Returns:
    A tuple containing:
      - out_degree: An array of shape [num_nodes] where each element is the
        out-degree of the corresponding node.
      - in_degree: An array of shape [num_nodes] where each element is the
        in-degree of the corresponding node.

  Raises:
    ValueError: If `adj` is not an integer type or does not have a shape of [2,
    Ne].
  """
  if not np.issubdtype(adj.dtype, np.integer):
    raise ValueError(f"Expected `adj` to be an integer type, got {adj.dtype}.")
  if adj.shape[0] != 2:
    raise ValueError(f"Expected a [2, Ne] shape, got {adj.shape}")

  n_nodes = num_nodes
  if n_nodes is None:
    # we assume node indices are on [0, n_nodes - 1] if num_nodes is not given.
    n_nodes = adj.max() + 1
  out_degree = np.bincount(adj[0], minlength=n_nodes)
  in_degree = np.bincount(adj[1], minlength=n_nodes)

  return out_degree, in_degree
