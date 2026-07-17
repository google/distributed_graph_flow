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

"""Import / export data to networkx."""

from typing import Tuple

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
import networkx as nx
import numpy as np


def graph_to_networkx(
    in_memory_graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    for_graphml: bool = False,
) -> nx.MultiDiGraph:
  """Converts an InMemoryGraph into a NetworkX MultiDiGraph.

  Usage:
    ```python
    # Normal conversion
    nx_graph = dgf.convert.graph_to_networkx(graph, schema)

    # Convert and write to GraphML
    nx_graph = dgf.convert.graph_to_networkx(graph, schema, for_graphml=True)
    nx.write_graphml(nx_graph, "/tmp/graph.graphml")
    ```

  Args:
    in_memory_graph: The input graph in `InMemoryGraph` format.
    schema: A `GraphSchema` instance describing the graph structure. Required to
      determine the source and target node sets for edges.
    for_graphml: If True, convert complex node and edge features (such as NumPy
      arrays and NumPy bytes) into basic strings and scalars that are strictly
      supported by the GraphML export format. Setting this to True breaks
      `networkx_to_graph`, as multidimensional data is degraded to strings.

  Returns:
    A NetworkX `MultiDiGraph` containing the graph data.
  """

  def _get_element_id(set_name: str, index: int):
    if not for_graphml:
      return (set_name, index)
    return f"{set_name}_{index}"

  def _get_feature_value(v):
    if not for_graphml:
      return v
    if isinstance(v, (bytes, np.bytes_)):
      return v.decode("utf-8", errors="replace")
    if isinstance(v, (list, np.ndarray)):
      return ",".join(map(str, np.array(v).flatten()))
    if isinstance(v, np.generic):
      return v.item()
    return v

  nx_graph = nx.MultiDiGraph()

  for node_set_name, node_set in in_memory_graph.node_sets.items():
    num_nodes = node_set.num_nodes
    assert num_nodes is not None

    for i in range(num_nodes):
      node_id = _get_element_id(node_set_name, i)
      node_attrs = {"node_set": node_set_name}
      for feat_name, feat_val in node_set.features.items():
        node_attrs[feat_name] = _get_feature_value(feat_val[i])
      nx_graph.add_node(node_id, **node_attrs)

  for edge_set_name, edge_set in in_memory_graph.edge_sets.items():
    if edge_set_name not in schema.edge_sets:
      raise ValueError(f"Edge set {edge_set_name} not found in schema.")
    edge_schema = schema.edge_sets[edge_set_name]
    source_set_name = edge_schema.source
    target_set_name = edge_schema.target

    sources = edge_set.adjacency[0]
    targets = edge_set.adjacency[1]

    for i in range(edge_set.num_edges()):
      src_node = _get_element_id(source_set_name, sources[i])
      tgt_node = _get_element_id(target_set_name, targets[i])
      edge_attrs = {"edge_set": edge_set_name}
      edge_key = f"{edge_set_name}_{i}" if for_graphml else None

      for feat_name, feat_val in edge_set.features.items():
        edge_attrs[feat_name] = _get_feature_value(feat_val[i])
      nx_graph.add_edge(src_node, tgt_node, key=edge_key, **edge_attrs)

  return nx_graph


def _infer_feature_schema(arr: np.ndarray) -> schema_lib.FeatureSchema:
  if np.issubdtype(arr.dtype, np.integer):
    fmt = schema_lib.FeatureFormat.INTEGER_64
  elif np.issubdtype(arr.dtype, np.floating):
    fmt = schema_lib.FeatureFormat.FLOAT_32
  elif np.issubdtype(arr.dtype, np.bool_):
    fmt = schema_lib.FeatureFormat.BOOL
  else:
    fmt = schema_lib.FeatureFormat.BYTES

  shape = _get_feature_shape(arr)
  return schema_lib.FeatureSchema(format=fmt, shape=shape)


def _get_feature_shape(arr: np.ndarray) -> Tuple[int, ...]:
  if arr.ndim > 1:
    return arr.shape[1:]
  return ()


def networkx_to_graph(
    nx_graph: nx.MultiDiGraph,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Converts a NetworkX graph into an InMemoryGraph and its schema.

  Usage:
    ```
    in_memory_graph, schema = dgf.convert.networkx_to_graph(nx_graph)
    ```

  Args:
    nx_graph: The input graph in NetworkX format.

  Returns:
    A tuple of (`InMemoryGraph`, `GraphSchema`).
  """
  node_sets_info = {}
  node_mapping = {}

  for node_id, attrs in nx_graph.nodes(data=True):
    node_set_name = attrs.get("node_set")
    if node_set_name is None:
      if (
          isinstance(node_id, tuple)
          and len(node_id) == 2
          and isinstance(node_id[0], str)
      ):
        node_set_name = node_id[0]
      else:
        node_set_name = "nodes"

    if node_set_name not in node_sets_info:
      node_sets_info[node_set_name] = {"nodes": [], "features": []}

    node_sets_info[node_set_name]["nodes"].append(node_id)
    node_sets_info[node_set_name]["features"].append(attrs)

  in_memory_node_sets = {}
  node_schemas = {}
  for node_set_name, info in node_sets_info.items():
    nodes = info["nodes"]
    num_nodes = len(nodes)

    feature_names = set()
    for attrs in info["features"]:
      for k in attrs.keys():
        if k != "node_set":
          feature_names.add(k)

    features = {k: [] for k in feature_names}
    for idx, (node_id, attrs) in enumerate(zip(nodes, info["features"])):
      node_mapping[node_id] = (node_set_name, idx)
      for k in feature_names:
        features[k].append(attrs.get(k))

    np_features = {k: np.array(v) for k, v in features.items()}
    in_memory_node_sets[node_set_name] = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=num_nodes, features=np_features
    )

    node_schemas[node_set_name] = schema_lib.NodeSchema(
        features={k: _infer_feature_schema(v) for k, v in np_features.items()}
    )

  edge_sets_info = {}
  for u, v, attrs in nx_graph.edges(data=True):
    edge_set_name = attrs.get("edge_set", "edges")

    if edge_set_name not in edge_sets_info:
      edge_sets_info[edge_set_name] = {
          "sources": [],
          "targets": [],
          "features": [],
          "source_set": None,
          "target_set": None,
      }

    u_mapped = node_mapping.get(u)
    v_mapped = node_mapping.get(v)
    if u_mapped is None or v_mapped is None:
      raise ValueError(f"Edge references unknown node(s): {u} -> {v}")

    # Record edge endpoint types if not done yet
    if edge_sets_info[edge_set_name]["source_set"] is None:
      edge_sets_info[edge_set_name]["source_set"] = u_mapped[0]
      edge_sets_info[edge_set_name]["target_set"] = v_mapped[0]

    edge_sets_info[edge_set_name]["sources"].append(u_mapped[1])
    edge_sets_info[edge_set_name]["targets"].append(v_mapped[1])
    edge_sets_info[edge_set_name]["features"].append(attrs)

  in_memory_edge_sets = {}
  edge_schemas = {}
  for edge_set_name, info in edge_sets_info.items():
    sources = info["sources"]
    targets = info["targets"]
    adjacency = np.array([sources, targets], dtype=np.int64)
    if len(sources) == 0:
      adjacency = adjacency.reshape(2, 0)

    feature_names = set()
    for attrs in info["features"]:
      for k in attrs.keys():
        if k != "edge_set":
          feature_names.add(k)

    features = {k: [] for k in feature_names}
    for attrs in info["features"]:
      for k in feature_names:
        features[k].append(attrs.get(k))

    np_features = {k: np.array(v) for k, v in features.items()}
    in_memory_edge_sets[edge_set_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=adjacency, features=np_features
    )

    source_set = info["source_set"]
    target_set = info["target_set"]
    edge_schemas[edge_set_name] = schema_lib.EdgeSchema(
        source=source_set,
        target=target_set,
        features={k: _infer_feature_schema(v) for k, v in np_features.items()},
    )

  return (
      in_memory_graph_lib.InMemoryGraph(
          node_sets=in_memory_node_sets, edge_sets=in_memory_edge_sets
      ),
      schema_lib.GraphSchema(node_sets=node_schemas, edge_sets=edge_schemas),
  )
