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

import math
from typing import Iterator, List, Optional
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib

# TODO(gbm): Implement Beam version.


def padding_from_graph_generator(
    schema: schema_lib.GraphSchema,
    graphs: Iterator[in_memory_graph_lib.InMemoryGraph],
    relative_margin: float = 0.1,
    absolute_margin: int = 1,
) -> padding_lib.Padding:
  """Creates a padding configuration from a set of in memory graphs.

  Usage example:

  ```python
  # Determine the padding
  graphs = iter(...) # Or, you can use the sampler
  schema = ...
  padding = padding_from_graph_generator(schema, graphs)

  # Later, the padding can be used to merge graphs
  merged_graph_samples, nodeset_offsets = dgf.transform.merge_graphs(
      [<some graph samples>],
      schema,
      padding=padding,
  )
  ```

  The padding size is:
    ceil((<maximum observed value> + absolute_margin) * (1 + relative_margin))

  Note: Having absolute_margin=1 (or more) is necessary if you need a sentinel
  node (e.g., when doing padded merging).

  Args:
    schema: The graph schema.
    graphs: An iterator over in-memory heterogeneous graphs.
    relative_margin: A relative margin.
    absolute_margin: An absolute margin.

  Returns:
    Padding configuration compatible with the graphs.
  """

  # Determine the maximum number of edges / nodes.
  max_nodes = {node_set_name: 0 for node_set_name in schema.node_sets}
  max_edges = {edge_set_name: 0 for edge_set_name in schema.edge_sets}

  num_graphs = 0
  for graph in graphs:
    for node_set_name in schema.node_sets:
      node_set = graph.node_sets[node_set_name]
      if node_set.num_nodes is not None:
        max_nodes[node_set_name] = max(
            max_nodes[node_set_name], node_set.num_nodes
        )

    for edge_set_name in schema.edge_sets:
      edge_set = graph.edge_sets[edge_set_name]
      # The number of edges is the second dimension of the adjacency matrix.
      num_edges = edge_set.adjacency.shape[1]
      max_edges[edge_set_name] = max(max_edges[edge_set_name], num_edges)
    num_graphs += 1

  if num_graphs == 0:
    raise ValueError("The input 'graphs' iterator was empty.")

  padded_node_sets = {}
  for node_set_name, max_n in max_nodes.items():
    padded_n = math.ceil((max_n + absolute_margin) * (1.0 + relative_margin))
    padded_node_sets[node_set_name] = padding_lib.NodeSetPadding(
        num_nodes=padded_n
    )

  # Create the padding with a margin.
  padded_edge_sets = {}
  for edge_set_name, max_e in max_edges.items():
    padded_e = math.ceil((max_e + absolute_margin) * (1.0 + relative_margin))
    padded_edge_sets[edge_set_name] = padding_lib.EdgeSetPadding(
        num_edges=padded_e
    )

  return padding_lib.Padding(
      node_sets=padded_node_sets, edge_sets=padded_edge_sets
  )


def print_padding(
    padding: padding_lib.Padding,
    return_output: bool = False,
    header: bool = True,
) -> Optional[str]:
  """Generates a human-readable string representation of a graph padding.

  Args:
    padding: The graph padding to print.
    return_output: If true, returns the output text instead of printing it.
    header: If true, print the "Graph Padding" header.

  Returns:
    A string containing the human-readable representation of the padding.
  """
  lines = []

  if header:
    lines.append("Graph Padding:\n")

  # Node Sets
  lines.append("Node Sets:")
  if not padding.node_sets:
    lines.append("  (No node sets)")
  else:
    for node_name in sorted(padding.node_sets.keys()):
      node_padding = padding.node_sets[node_name]
      lines.append(f"  {node_name}: {node_padding.num_nodes} nodes")

  # Edge Sets
  lines.append("\nEdge Sets:")
  if not padding.edge_sets:
    lines.append("  (No edge sets)")
  else:
    for edge_name in sorted(padding.edge_sets.keys()):
      edge_padding = padding.edge_sets[edge_name]
      lines.append(f"  {edge_name}: {edge_padding.num_edges} edges")

  text_content = "\n".join(lines)

  if return_output:
    return text_content
  else:
    print(text_content)
    return None
