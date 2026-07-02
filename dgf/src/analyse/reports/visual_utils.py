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

"""Utilities for generating visual inspection data for Graph Flow reports.

This module provides functions to convert NetworkX graphs into data structures
compatible with Vis.js (via PyVis) for interactive visualization in HTML
reports.
"""

import colorsys
import logging
from typing import Any, Dict, Union
import zlib

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
import networkx as nx
import numpy as np


def _get_color_for_value(value: Any) -> str:
  """Deterministically generates a color based on the value."""
  if value is None:
    return "#9AA0A6"  # Grey for None/Unknown

  # Simple deterministic hash
  # Use zlib.adler32 for stability across runs (python hash() is randomized)
  hash_val = zlib.adler32(str(value).encode("utf-8"))

  # Use Golden Ratio to spread hues uniformly (Golden Angle Approximation)
  golden_ratio_conjugate = 0.618033988749895
  hue = (hash_val * golden_ratio_conjugate) % 1.0

  # Vary Saturation and Value slightly based on hash bits for more variety
  # Saturation: [0.6, 0.95] - Avoid too pale
  saturation = 0.6 + ((hash_val & 0xFF) / 255.0) * 0.35
  # Value/Brightness: [0.75, 0.95] - Avoid too dark
  val_brightness = 0.75 + (((hash_val >> 8) & 0xFF) / 255.0) * 0.20

  rgb = colorsys.hsv_to_rgb(hue, saturation, val_brightness)

  return "#{:02x}{:02x}{:02x}".format(
      int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
  )


def graph_to_pyvis_data(
    graph: Union[nx.Graph, in_memory_graph_lib.InMemoryGraph],
    height: str = "500px",
    width: str = "100%",
    nodeset_namespace_delimiter: str = "_",
    color_by_attribute: str | None = None,
    node_label_attribute: str | None = None,
    graph_schema: schema_lib.GraphSchema | None = None,
) -> Dict[str, Any]:
  """Converts a graph (NetworkX or InMemoryGraph) to PyVis-compatible data.

  Args:
    graph: The graph (NetworkX or InMemoryGraph) to convert.
    height: Height of the visualization container (default: "500px").
    width: Width of the visualization container (default: "100%").
    nodeset_namespace_delimiter: Delimiter to use when namespacing node IDs
      across NodeSets.
    color_by_attribute: Optional attribute name to color nodes by (NetworkX
      only). For InMemoryGraph, nodes are automatically colored by their NodeSet
      name.
    node_label_attribute: Optional attribute to use for node labeling.
    graph_schema: Optional map from edge set name to (source, target) node sets.
      Only used for InMemoryGraph to resolve edge endpoints.

  Returns:
    A dictionary containing nodes, edges, options, height, width, and legend.
  """
  if isinstance(graph, in_memory_graph_lib.InMemoryGraph):
    return _in_memory_graph_to_pyvis_data(
        graph,
        height,
        width,
        nodeset_namespace_delimiter,
        color_by_attribute,
        node_label_attribute,
        graph_schema,
    )
  elif isinstance(graph, nx.Graph):
    return _nx_graph_to_pyvis_data(
        graph, height, width, color_by_attribute, node_label_attribute
    )
  else:
    raise ValueError(f"Unsupported graph type: {type(graph)}")


def _attr_str(attrs: Dict[str, Any]) -> str:
  """Helper to format attributes for tooltip."""
  if not attrs:
    return ""
  return "<br>".join([f"{k}: {v}" for k, v in attrs.items()])


def _namespaced_id(namespace: str, idx: int, delimiter: str = "_") -> str:
  """Returns id of the "graph piece" so it's generic for node, edge or context level items.

  Args:
    namespace: The namespace of the "graph piece" (node, edge or context level).
    idx: The index of the "graph piece" within the namespace.
    delimiter: The delimiter to use between the namespace and the index.

  Returns:
    A graph element ID for a nodes, edges or context level items.
  """
  return f"{namespace}{delimiter}{idx}"


def _nx_graph_to_pyvis_data(
    graph: nx.Graph,
    height: str,
    width: str,
    color_by_attribute: str | None = None,
    node_label_attribute: str | None = None,
) -> Dict[str, Any]:
  """Converts a NetworkX graph to PyVis data."""

  # Convert Nodes (NetworkX)
  nodes = []
  legend = {}  # Mapping from attribute value to color

  for node_id, attrs in graph.nodes(data=True):
    node_data = {
        "id": int(node_id),
        "label": str(node_id),
        "title": str(_attr_str(attrs)),
    }

    # Label Override
    if node_label_attribute and node_label_attribute in attrs:
      node_data["label"] = str(attrs[node_label_attribute])

    # Coloring Logic
    if color_by_attribute:
      val = attrs.get(color_by_attribute, "N/A")
      color = _get_color_for_value(val)
      node_data["color"] = color
      # Append coloring info to title for tooltip
      node_data["title"] += f"<br><b>{color_by_attribute}:</b> {val}"

      # Add to legend
      legend[str(val)] = color

    elif "color" in attrs:
      # Fallback to existing color if present and no override is requested.
      node_data["color"] = attrs["color"]

    if "size" in attrs:
      node_data["size"] = attrs["size"]

    # Store raw attributes for interactive inspection
    node_data["extra_data"] = attrs
    nodes.append(node_data)

  # Convert Edges
  edges = []
  for u, v, attrs in graph.edges(data=True):
    edge_data = {"from": int(u), "to": int(v)}
    if "weight" in attrs:
      edge_data["value"] = attrs["weight"]
      edge_data["title"] = f"Weight: {attrs['weight']}"  # pyrefly: ignore[bad-assignment]

    if "color" in attrs:
      edge_data["color"] = attrs["color"]

    # Store raw attributes for interactive inspection
    edge_data["extra_data"] = attrs
    edges.append(edge_data)

  # Default Options
  options = {
      "interaction": {"hover": True},
      "physics": {"enabled": True, "stabilization": {"iterations": 200}},
      "nodes": {
          "shape": "dot",
          "size": 10,
          "font": {"size": 14, "face": "Roboto"},
      },
      "edges": {"color": {"inherit": True}, "smooth": False},
  }

  ## TODO(tewariy): Wrap this return object in a dataclass.
  return {
      "nodes": nodes,
      "edges": edges,
      "options": options,
      "height": height,
      "width": width,
      "legend": legend,
  }


def _in_memory_graph_to_pyvis_data(
    graph: in_memory_graph_lib.InMemoryGraph,
    height: str,
    width: str,
    nodeset_namespace_delimiter: str = "_",
    color_by_attribute: str | None = None,
    node_label_attribute: str | None = None,
    graph_schema: schema_lib.GraphSchema | None = None,
) -> Dict[str, Any]:
  """Converts an InMemoryGraph to PyVis data, coloring nodes by NodeSet."""
  nodes = []
  edges = []
  legend = {}

  # --- Nodes ---
  for ns_name, ns in graph.node_sets.items():
    # Determine count
    count = ns.num_nodes
    if count is None and ns.features:
      count = list(ns.features.values())[0].shape[0]
    if count is None:
      continue

    # Default Color and Legend
    default_color = _get_color_for_value(ns_name)
    if not color_by_attribute:
      legend[ns_name] = default_color

    # Pre-fetch features to avoid repeated lookups
    feature_keys = list(ns.features.keys())

    for i in range(count):
      # Namespace the ID to ensure uniqueness across NodeSets
      node_id = _namespaced_id(ns_name, i, nodeset_namespace_delimiter)

      # Default label to index
      label = str(i)

      # Priority 1: User-specified attribute
      if node_label_attribute and node_label_attribute in ns.features:
        val = ns.features[node_label_attribute][i]
        if hasattr(val, "item"):
          val = val.item()
        if isinstance(val, bytes):
          val = val.decode("utf-8", "ignore")
        label = str(val)

      # Priority 2: Implicit ID features (if no user override)
      else:
        for id_key in ["#id", "id"]:
          if id_key in ns.features:
            val = ns.features[id_key][i]
            if hasattr(val, "item"):
              val = val.item()
            if isinstance(val, bytes):
              val = val.decode("utf-8", "ignore")
            label = str(val)
            break

      extra_data = {"type": ns_name}
      title_parts = [f"Type: {ns_name}"]

      for key in feature_keys:
        val = ns.features[key][i]
        if isinstance(val, np.ndarray):
          if val.ndim == 0 or val.size == 1:
            val = val.item()
          else:
            val = str(val.tolist())
        elif hasattr(val, "item"):
          val = val.item()

        if isinstance(val, bytes):
          val = val.decode("utf-8", "ignore")

        extra_data[key] = val
        title_parts.append(f"{key}: {val}")

      # Coloring Logic
      color = default_color
      if color_by_attribute:
        val = extra_data.get(color_by_attribute)
        color = _get_color_for_value(val)
        legend[str(val)] = color

      node_dict = {
          "id": node_id,
          "label": label,
          "color": color,
          "title": "<br>".join(title_parts),
          # "group": ns_name,  # Removed to prevent vis.js from overriding color
          "extra_data": extra_data,
      }

      nodes.append(node_dict)

  # --- Edges ---
  single_node_set_name = (
      list(graph.node_sets.keys())[0] if len(graph.node_sets) == 1 else None
  )

  for es_name, es in graph.edge_sets.items():
    adj = es.adjacency
    if adj.shape[0] != 2:
      continue

    # Resolve Source and Target NodeSets
    src_ns = None
    tgt_ns = None

    if graph_schema:
      if es_name in graph_schema.edge_sets:
        edge_def = graph_schema.edge_sets[es_name]
        src_ns = edge_def.source
        tgt_ns = edge_def.target
      else:
        logging.warning(
            "EdgeSet '%s' not found in GraphSchema. Available keys: %s",
            es_name,
            list(graph_schema.edge_sets.keys()),
        )
    elif single_node_set_name:
      src_ns = single_node_set_name
      tgt_ns = single_node_set_name

    if not src_ns or not tgt_ns:
      continue

    count = adj.shape[1]
    feature_keys = list(es.features.keys())

    for i in range(count):
      u_idx = adj[0, i].item()
      v_idx = adj[1, i].item()

      # Construct namespaced IDs
      u_id = _namespaced_id(src_ns, u_idx, nodeset_namespace_delimiter)
      v_id = _namespaced_id(tgt_ns, v_idx, nodeset_namespace_delimiter)

      extra_data = {"type": es_name}
      edge_dict = {"from": u_id, "to": v_id, "extra_data": extra_data}

      for key in feature_keys:
        val = es.features[key][i]
        if isinstance(val, np.ndarray):
          if val.ndim == 0 or val.size == 1:
            val = val.item()
          else:
            val = str(val.tolist())
        elif hasattr(val, "item"):
          val = val.item()

        if isinstance(val, bytes):
          val = val.decode("utf-8", "ignore")

        extra_data[key] = val
        if key == "color":
          edge_dict["color"] = val
        elif key == "weight":
          edge_dict["value"] = val
          edge_dict["title"] = f"Weight: {val}"

      edges.append(edge_dict)

  # Default Options
  options = {
      "interaction": {"hover": True},
      "physics": {"enabled": True, "stabilization": {"iterations": 200}},
      "nodes": {
          "shape": "dot",
          "size": 10,
          "font": {"size": 14, "face": "Roboto"},
      },
      "edges": {"color": {"inherit": True}, "smooth": False},
  }

  ## TODO(tewariy): Wrap this return object in a dataclass.
  return {
      "nodes": nodes,
      "edges": edges,
      "options": options,
      "height": height,
      "width": width,
      "legend": legend,
  }
