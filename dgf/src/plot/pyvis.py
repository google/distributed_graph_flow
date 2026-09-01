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

"""Plotting utilities for graph schemas using pyvis."""

from typing import Any

from dgf.src.data import schema as schema_lib
from dgf.src.util import options as options_lib
from pyvis import network as pyvis_network

Context = options_lib.Context
options = options_lib.Manager("pyvis_options")
default_options = options.default_options
option_context = options.context


def _html_label(name: str, features: list[str]) -> str:
  """Creates a node/edge title for hover information."""
  if not features:
    return name
  return f"{name}\n" + "\n".join(features)


def plot_schema(
    schema: schema_lib.GraphSchema,
    features: bool = True,
    *,
    pyvis_kwargs: dict[Any, Any] | None = None,
) -> pyvis_network.Network:
  """Plots the graph schema's meta-graph (i.e., its nodesets and edgesets).

  Args:
    schema: The `GraphSchema` object to plot.
    features: If true, display the node and edges features in the title (hover).
    pyvis_kwargs: Additional keyword arguments to pass to the
      `pyvis.network.Network` constructor. These override any default or
      contextual pyvis options.

  Returns:
    A `pyvis.network.Network` object representing the graph schema.
  """
  final_kwargs: dict[str, Any] = {"directed": True}
  final_kwargs.update(options.to_dict())

  if pyvis_kwargs is not None:
    final_kwargs.update(pyvis_kwargs)

  clean_kwargs = {k: v for k, v in final_kwargs.items() if v is not None}
  net = pyvis_network.Network(**clean_kwargs)

  # Add nodes
  for node_set_name in sorted(schema.node_sets.keys()):
    if features:
      node_set_schema = schema.node_sets[node_set_name]
      feature_names = sorted(node_set_schema.features.keys())
      title = _html_label(node_set_name, feature_names)
      net.add_node(node_set_name, label=node_set_name, title=title, shape="box")
    else:
      net.add_node(node_set_name, label=node_set_name, shape="ellipse")

  # Add edges
  for edge_set_name in sorted(schema.edge_sets.keys()):
    edge_schema = schema.edge_sets[edge_set_name]
    if features:
      feature_names = sorted(edge_schema.features.keys())
      edge_title = _html_label(edge_set_name, feature_names)
    else:
      edge_title = edge_set_name

    net.add_edge(
        edge_schema.source,
        edge_schema.target,
        title=edge_title,
        label=edge_set_name,
    )

  return net
