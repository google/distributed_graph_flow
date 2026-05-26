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

"""Conversion to TF related graph objects."""

import re
from typing import List, Optional
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import tf_in_memory_graph as tf_in_memory_graph_lib
from dgf.src.io import feature_format as feature_format_lib
import tensorflow as tf

BEGIN_CODE = "begincode"
END_CODE = "endcode"


def _encode_name(name: str) -> str:
  """Encodes a nodeset, edgeset, or feature name."""
  encoded = []
  for char in name:
    if char.isalnum():
      encoded.append(char)
    else:
      encoded.append(f"{BEGIN_CODE}{ord(char):02x}{END_CODE}")
  return "".join(encoded)


def _decode_name(name: str) -> str:
  """Decodes a nodeset, edgeset, or feature name."""
  pattern = rf"{BEGIN_CODE}([0-9a-f]{{2}}){END_CODE}"
  return re.sub(pattern, lambda match: chr(int(match.group(1), 16)), name)


def _has_encoded_pattern(name: str) -> bool:
  """Checks if a name contains the encoding pattern."""
  pattern = rf"{BEGIN_CODE}[0-9a-f]{{2}}{END_CODE}"
  return re.search(pattern, name) is not None


def schema_to_spec(
    schema_: schema_lib.GraphSchema,
) -> tf_in_memory_graph_lib.TFInMemoryGraph.Spec:
  """Converts a GraphSchema to a TFInMemoryGraph.Spec for tf serialization."""
  node_sets = {}
  for node_set_name, node_schema in schema_.node_sets.items():
    if _has_encoded_pattern(node_set_name):
      raise ValueError(
          f"Node set name '{node_set_name}' contains invalid substring"
          " matching encoding pattern"
      )
    features = {}
    for feat_name, feat_schema in node_schema.features.items():
      if _has_encoded_pattern(feat_name):
        raise ValueError(
            f"Feature name '{feat_name}' in node set '{node_set_name}' contains"
            " invalid substring matching encoding pattern"
        )
      dtype = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[feat_schema.format]
      shape = [None] + (list(feat_schema.shape) if feat_schema.shape else [])
      features[feat_name] = tf.TensorSpec(shape=shape, dtype=dtype)
    node_sets[node_set_name] = tf_in_memory_graph_lib.TFInMemoryNodeSet.Spec(
        num_nodes=tf.TensorSpec(shape=(), dtype=tf.int32),
        features=features,
    )

  edge_sets = {}
  for edge_set_name, edge_schema in schema_.edge_sets.items():
    if _has_encoded_pattern(edge_set_name):
      raise ValueError(
          f"Edge set name '{edge_set_name}' contains invalid substring"
          " matching encoding pattern"
      )
    features = {}
    for feat_name, feat_schema in edge_schema.features.items():
      if _has_encoded_pattern(feat_name):
        raise ValueError(
            f"Feature name '{feat_name}' in edge set '{edge_set_name}' contains"
            " invalid substring matching encoding pattern"
        )
      dtype = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[feat_schema.format]
      shape = [None] + (list(feat_schema.shape) if feat_schema.shape else [])
      features[feat_name] = tf.TensorSpec(shape=shape, dtype=dtype)
    edge_sets[edge_set_name] = tf_in_memory_graph_lib.TFInMemoryEdgeSet.Spec(
        adjacency=tf.TensorSpec(shape=[2, None], dtype=tf.int64),
        features=features,
    )
  return tf_in_memory_graph_lib.TFInMemoryGraph.Spec(
      node_sets=node_sets, edge_sets=edge_sets
  )


def schema_to_dict_spec(
    schema_: schema_lib.GraphSchema,
) -> List[tf.TensorSpec]:
  """Converts a GraphSchema to a dict of TensorSpec for TFInMemoryGraphDict."""
  result = []
  for node_set_name, node_schema in schema_.node_sets.items():
    encoded_node_set_name = _encode_name(node_set_name)
    key = f"nodes_{encoded_node_set_name}_reserved_size"
    result.append(tf.TensorSpec(shape=(), dtype=tf.int32, name=key))
    for feat_name, feat_schema in node_schema.features.items():
      dtype = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[feat_schema.format]
      shape = [None] + (list(feat_schema.shape) if feat_schema.shape else [])
      encoded_feat_name = _encode_name(feat_name)
      key = f"nodes_{encoded_node_set_name}_{encoded_feat_name}"
      result.append(tf.TensorSpec(shape=shape, dtype=dtype, name=key))

  for edge_set_name, edge_schema in schema_.edge_sets.items():
    encoded_edge_set_name = _encode_name(edge_set_name)
    key = f"edges_{encoded_edge_set_name}_reserved_adjacency"
    result.append(tf.TensorSpec(shape=[2, None], dtype=tf.int64, name=key))
    for feat_name, feat_schema in edge_schema.features.items():
      dtype = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[feat_schema.format]
      shape = [None] + (list(feat_schema.shape) if feat_schema.shape else [])
      encoded_feat_name = _encode_name(feat_name)
      key = f"edges_{encoded_edge_set_name}_{encoded_feat_name}"
      result.append(tf.TensorSpec(shape=shape, dtype=dtype, name=key))
  return result


def graph_to_tf_graph(
    src: in_memory_graph_lib.InMemoryGraph,
    schema: Optional[schema_lib.GraphSchema] = None,
) -> tf_in_memory_graph_lib.TFInMemoryGraph:
  """Converts a graph to a TF in memory graph.

  Args:
    src: The source graph to convert.
    schema: Optional graph schema to enforce typing (especially useful for empty
      arrays).

  Returns:
    A `TFInMemoryGraph` representation of the input graph.
  """

  # Convert InMemoryGraph to TFInMemoryGraph
  tf_node_sets = {}
  for node_set_name, node_set in src.node_sets.items():
    tf_features = {}
    for k, v in node_set.features.items():
      target_dtype = None
      if (
          schema is not None
          and node_set_name in schema.node_sets
          and k in schema.node_sets[node_set_name].features
      ):
        target_dtype = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[
            schema.node_sets[node_set_name].features[k].format
        ]

      def numpy_to_tf(v) -> tf.Tensor:
        if getattr(v, "dtype", None) == object or isinstance(v, (list, tuple)):
          # ragged arrays or object arrays (like variable-length strings)
          if target_dtype is not None:
            return tf.ragged.constant(v, dtype=target_dtype)
          return tf.ragged.constant(v)
        return tf.constant(v)

      tensor = numpy_to_tf(v)
      if target_dtype is not None and tensor.dtype != target_dtype:
        tensor = tf.cast(tensor, target_dtype)
      tf_features[k] = tensor

    tf_node_sets[node_set_name] = tf_in_memory_graph_lib.TFInMemoryNodeSet(
        features=tf_features,
        num_nodes=tf.convert_to_tensor(node_set.num_nodes, dtype=tf.int32),
    )

  tf_edge_sets = {}
  for edge_set_name, edge_set in src.edge_sets.items():
    tf_adjacency = tf.convert_to_tensor(edge_set.adjacency)
    tf_features = {}
    for k, v in edge_set.features.items():
      target_dtype = None
      if (
          schema is not None
          and edge_set_name in schema.edge_sets
          and k in schema.edge_sets[edge_set_name].features
      ):
        target_dtype = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[
            schema.edge_sets[edge_set_name].features[k].format
        ]

      def numpy_to_tf_edge(v) -> tf.Tensor:
        if getattr(v, "dtype", None) == object or isinstance(v, (list, tuple)):
          # ragged arrays or object arrays
          if target_dtype is not None:
            return tf.ragged.constant(v, dtype=target_dtype)
          return tf.ragged.constant(v)
        return tf.constant(v)

      tensor = numpy_to_tf_edge(v)
      if target_dtype is not None and tensor.dtype != target_dtype:
        tensor = tf.cast(tensor, target_dtype)
      tf_features[k] = tensor

    tf_edge_sets[edge_set_name] = tf_in_memory_graph_lib.TFInMemoryEdgeSet(
        adjacency=tf_adjacency, features=tf_features
    )

  return tf_in_memory_graph_lib.TFInMemoryGraph(
      node_sets=tf_node_sets, edge_sets=tf_edge_sets
  )


def tf_graph_to_tf_graph_dict(
    src: tf_in_memory_graph_lib.TFInMemoryGraph,
) -> tf_in_memory_graph_lib.TFInMemoryGraphDict:
  """Converts a TFInMemoryGraph into a flattened TFInMemoryGraphDict.

  Usage example:

  ```python
  tf_graph = ...  # A TFInMemoryGraph instance
  graph_dict = dgf.api.convert.tf_graph_to_tf_graph_dict(tf_graph)
  ```

  See the "Graph formats" documentation page for details about the tf graph dict
  format.

  Args:
    src: The source TFInMemoryGraph to convert.

  Returns:
    A TFInMemoryGraphDict with flattened keys and tensor values.
  """

  result = {}
  for node_set_name, node_set in src.node_sets.items():
    encoded_node_set_name = _encode_name(node_set_name)
    result[f"nodes_{encoded_node_set_name}_reserved_size"] = (
        tf.convert_to_tensor(node_set.num_nodes)
    )
    for feat_name, feat_val in node_set.features.items():
      encoded_feat_name = _encode_name(feat_name)
      result[f"nodes_{encoded_node_set_name}_{encoded_feat_name}"] = feat_val

  for edge_set_name, edge_set in src.edge_sets.items():
    encoded_edge_set_name = _encode_name(edge_set_name)
    result[f"edges_{encoded_edge_set_name}_reserved_adjacency"] = (
        edge_set.adjacency
    )
    for feat_name, feat_val in edge_set.features.items():
      encoded_feat_name = _encode_name(feat_name)
      result[f"edges_{encoded_edge_set_name}_{encoded_feat_name}"] = feat_val
  return result


def tf_graph_dict_to_tf_graph(
    src: tf_in_memory_graph_lib.TFInMemoryGraphDict,
) -> tf_in_memory_graph_lib.TFInMemoryGraph:
  """Converts a flattened TFInMemoryGraphDict back into a TFInMemoryGraph.

  Usage example:

  ```python
  graph_dict = {
      "nodes_n1_reserved_size": tf.constant([2], dtype=tf.int32),
      "nodes_n1_feat": tf.constant([[1.0], [2.0]]),
      "edges_e1_reserved_adjacency": tf.constant([[0, 0], [0, 1]],
      dtype=tf.int64),
  }
  tf_graph = dgf.api.convert.tf_graph_dict_to_tf_graph(graph_dict)
  ```

  See the "Graph formats" documentation page for details about the tf graph dict
  format.

  Args:
    src: The source TFInMemoryGraphDict to convert.

  Returns:
    A reconstructed TFInMemoryGraph.
  """

  nodeset_features = {}
  nodeset_num_nodes = {}
  edgeset_features = {}
  edgeset_adjacency = {}

  for key, val in src.items():
    if key.startswith("nodes_"):
      parts = key[len("nodes_") :].split("_", 1)
      nodeset_name = _decode_name(parts[0])
      attr = parts[1]
      if attr == "reserved_size":
        nodeset_num_nodes[nodeset_name] = val
      else:
        attr = _decode_name(attr)
        if nodeset_name not in nodeset_features:
          nodeset_features[nodeset_name] = {}
        nodeset_features[nodeset_name][attr] = val
    elif key.startswith("edges_"):
      parts = key[len("edges_") :].split("_", 1)
      edgeset_name = _decode_name(parts[0])
      attr = parts[1]
      if attr == "reserved_adjacency":
        edgeset_adjacency[edgeset_name] = val
      else:
        attr = _decode_name(attr)
        if edgeset_name not in edgeset_features:
          edgeset_features[edgeset_name] = {}
        edgeset_features[edgeset_name][attr] = val

  tf_node_sets = {}
  for name in set(nodeset_num_nodes.keys()).union(nodeset_features.keys()):
    tf_node_sets[name] = tf_in_memory_graph_lib.TFInMemoryNodeSet(
        num_nodes=nodeset_num_nodes.get(name),
        features=nodeset_features.get(name, {}),
    )

  tf_edge_sets = {}
  for name in set(edgeset_adjacency.keys()).union(edgeset_features.keys()):
    tf_edge_sets[name] = tf_in_memory_graph_lib.TFInMemoryEdgeSet(
        adjacency=edgeset_adjacency.get(name),
        features=edgeset_features.get(name, {}),
    )

  return tf_in_memory_graph_lib.TFInMemoryGraph(
      node_sets=tf_node_sets, edge_sets=tf_edge_sets
  )
