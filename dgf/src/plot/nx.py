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

"""Helpers to visualize NetworkX graphs."""

from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx


def plot_nx_graph(
    g: nx.Graph, label_name: Optional[str] = None, ax: Optional[plt.Axes] = None
):
  """Helper function to draw an nx graph.

  Args:
    g: The nx graph to draw.
    label_name: The name of the nx node data attribute to use as the name of the
      node in the visualization. If provided, nodes will be colored according to
      this label.
    ax: The matplotlib axes to draw on. If not provided, a new figure and axes
      will be created.
  """
  pos = nx.spring_layout(g)

  node_colors = None
  labels = {}
  if label_name is not None:
    unique_labels = sorted(
        list(set(data[label_name] for _, data in g.nodes(data=True)))
    )
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    node_colors = []
    for node_id, data in g.nodes(data=True):
      label = data[label_name]
      if isinstance(label, bytes):
        label = label.decode('utf-8')
      labels[node_id] = label
      node_colors.append(cmap(label_to_int[data[label_name]]))

  nx.draw_networkx_nodes(
      g, pos, node_size=300, ax=ax, node_color=node_colors, node_shape='o'
  )
  nx.draw_networkx_edges(g, pos, edge_color='gray', ax=ax)

  if label_name is None:
    nx.draw_networkx_labels(g, pos, font_size=10, font_weight='bold', ax=ax)
  else:
    nx.draw_networkx_labels(
        g, pos, labels=labels, font_size=10, font_weight='bold', ax=ax
    )
