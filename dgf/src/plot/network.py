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

"""Plotting of graph elements using the networkx library."""

import hashlib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
import graphviz


def _get_color(name: str) -> str:
  """Gets a consistent color for a given name."""
  colors = [
      "blue",
      "orange",
      "green",
      "red",
      "purple",
      "brown",
      "pink",
      "grey",
      "olive",
      "cyan",
  ]
  # Use a hash to pick a color consistently for each unique name.
  return colors[
      int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % len(colors)
  ]


def _graphviz_html_label(name: str, features: list[str]) -> str:
  """Creates a node/edge label."""
  if not features:
    return name
  return f"<<b>{name}</b><br/>" + "<br/>".join(features) + ">"


def plot_schema(
    schema: schema_lib.GraphSchema, features: bool = True
) -> graphviz.Digraph:
  """Plots the graphschema's meta-graph (i.e., its nodesets and edgesets).

  Usage example:

  ```python
  import graphviz

  schema = dgf.io.read_schema("path")
  # Or use "dgf.io.tfgnn_schema_to_schema" if you have a TF-GNN schema.

  dot = dgf.plot.plot_schema(schema)
  # Display in a colab
  dot
  # Save to file.
  dot.render('in_memory_graph', format='png')
  ```

  Args:
    schema: The `GraphSchema` object to plot.
    features: If true, display the node and edges features.

  Returns:
    A `graphviz.Digraph` object representing the graph schema.
  """
  dot = graphviz.Digraph(comment="Graph Schema")

  for node_set_name in sorted(schema.node_sets.keys()):
    if features:
      node_set_schema = schema.node_sets[node_set_name]
      feature_names = sorted(node_set_schema.features.keys())
      label = _graphviz_html_label(node_set_name, feature_names)
      dot.node(node_set_name, label=label, shape="box")
    else:
      dot.node(node_set_name, node_set_name, shape="ellipse")

  for edge_set_name in sorted(schema.edge_sets.keys()):
    edge_schema = schema.edge_sets[edge_set_name]
    if features:
      feature_names = sorted(edge_schema.features.keys())
      edge_label = _graphviz_html_label(edge_set_name, feature_names)
    else:
      edge_label = edge_set_name
    dot.edge(edge_schema.source, edge_schema.target, label=edge_label)

  return dot


def plot_graph(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    features: bool = True,
) -> graphviz.Digraph:
  """Plots an in-memory graph.

  Usage example:

  ```python
  import graphviz

  schema = dgf.io.read_schema("path")
  graph = dgf.io.tfgnn_graph_to_graph(...)
  # Or use the in-memory sampler to generate in_memory_graphb5s.

  dot = dgf.plot.plot_graph(graph, schema)
  # Display in a colab
  dot
  # Save to file.
  dot.render('in_memory_graph', format='png')
  ```

  Args:
    graph: The `InMemoryGraph` object to plot.
    schema: The `GraphSchema` object describing the graph.
    features: If true, display the node and edges features.

  Returns:
    A `graphviz.Digraph` object representing the graph schema.
  """
  dot = graphviz.Digraph(
      comment="In-Memory Graph", graph_attr={"rankdir": "LR"}
  )

  # Add nodes for each node set.
  for node_set_name in sorted(schema.node_sets.keys()):
    node_set_schema = schema.node_sets[node_set_name]
    if node_set_name not in graph.node_sets:
      print(f"Warning: Node set '{node_set_name}' not found in the graph.")
      continue
    node_set_data = graph.node_sets[node_set_name]
    if node_set_data.num_nodes is None:
      print(f"Warning: Node set '{node_set_name}' has no num_nodes. Skipping.")
      continue

    node_color = _get_color(node_set_name)
    for node_idx in range(node_set_data.num_nodes):
      node_id = f"{node_set_name}_{node_idx}"
      if features:
        feature_labels = []

        for feature_name in sorted(node_set_schema.features.keys()):
          feature_tensor = node_set_data.features[feature_name]
          if node_idx < feature_tensor.shape[0]:
            feature_value = feature_tensor[node_idx]
            feature_value_str = str(feature_value)
            if len(feature_value_str) > 50:
              feature_value_str = feature_value_str[:50] + "[...]"
            feature_labels.append(f"{feature_name}: {feature_value_str}")
        label = _graphviz_html_label(node_id, feature_labels)
        dot.node(
            node_id,
            label=label,
            shape="box",
            style="filled",
            fillcolor=node_color,
        )
      else:
        dot.node(
            node_id,
            label=node_id,
            shape="box",
            style="filled",
            fillcolor=node_color,
        )

  # Add edges for each edge set.
  for edge_set_name, edge_set_schema in sorted(schema.edge_sets.items()):
    if edge_set_name not in graph.edge_sets:
      print(f"Warning: Edge set '{edge_set_name}' not found in the graph.")
      continue
    edge_set_data = graph.edge_sets[edge_set_name]
    source_set = edge_set_schema.source
    target_set = edge_set_schema.target
    edge_color = _get_color(edge_set_name)

    adjacency = edge_set_data.adjacency
    # Adjacency is expected to be of shape [2, num_edges] for unbatched graphs.
    if adjacency.ndim == 2 and adjacency.shape[0] == 2:
      num_edges = adjacency.shape[1]
      for edge_idx in range(num_edges):
        source_idx = adjacency[0, edge_idx]
        target_idx = adjacency[1, edge_idx]
        source_node_id = f"{source_set}_{source_idx}"
        target_node_id = f"{target_set}_{target_idx}"

        edge_label = edge_set_name
        if features:
          feature_labels = []
          for feature_name, feature_tensor in edge_set_data.features.items():
            if edge_idx < feature_tensor.shape[0]:
              feature_value = feature_tensor[edge_idx]
              feature_labels.append(f"{feature_name}: {feature_value}")
          edge_label = _graphviz_html_label(edge_set_name, feature_labels)

        dot.edge(
            source_node_id,
            target_node_id,
            label=edge_label,
            color=edge_color,
            fontcolor=edge_color,
        )
    else:
      print(
          f"Warning: Unexpected adjacency shape for edge set '{edge_set_name}':"
          f" {adjacency.shape}. Expected [2, num_edges] for unbatched graphs."
          " Batched graphs are not currently supported by this plotting"
          " function."
      )

  return dot
