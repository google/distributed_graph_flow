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

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import tf_in_memory_graph as tf_in_memory_graph_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf

# Key naming of the "TF GNN Graph Sample" format.
#
# A TF GNN Graph Sample is a `tf.train.Example` proto (or an equivalent
# dictionary of flat arrays / tensors, called a "TF GNN graph dict") encoding a
# single graph with the following keys:
#
#   nodes/<nodeset>.#size      : Number of nodes in the nodeset.
#   nodes/<nodeset>.<feature>  : Flattened node feature values.
#   edges/<edgeset>.#size      : Number of edges in the edgeset.
#   edges/<edgeset>.#source    : Source node index of each edge.
#   edges/<edgeset>.#target    : Target node index of each edge.
#   edges/<edgeset>.<feature>  : Flattened edge feature values.
#
# Since a `tf.train.Example` only stores flat lists of int64s, float32s or
# bytes:
#   - Multi-dimensional features are flattened (in C / row-major order).
#   - Variable-length (i.e. "ragged") dimensions are encoded with an extra
#     "<feature>.d<i>" key containing the length of each of the rows of the
#     i-th dimension of the feature (the 0-th dimension being the item i.e.
#     node or edge dimension). For example, a feature with a schema shape
#     (None, 2), i.e. of effective shape [num_items, None, 2], is encoded with
#     the flat feature values and a "<feature>.d1" key containing `num_items`
#     row lengths.
#   - Feature values are stored with the `tf.train.Example` compatible dtype
#     (int64, float32 or bytes) of their schema format. The conversion to the
#     exact schema dtype (e.g. int32, bool) is done when converting the TF GNN
#     Graph Sample into a DGF object (e.g. `InMemoryGraph`, `TFInMemoryGraph`).

TFGNN_NODES_PREFIX = "nodes/"
TFGNN_EDGES_PREFIX = "edges/"
TFGNN_SIZE_KEY = "#size"
TFGNN_SOURCE_KEY = "#source"
TFGNN_TARGET_KEY = "#target"

# Key naming of the DGF "TF Graph Dict" format (`TFInMemoryGraphDict`), i.e. a
# `TFInMemoryGraph` flattened into a single dictionary of tensors:
#
#   nodes_<nodeset>_reserved_size       : Number of nodes in the nodeset.
#   nodes_<nodeset>_<feature>           : Node feature values.
#   edges_<edgeset>_reserved_adjacency  : Adjacency of the edgeset.
#   edges_<edgeset>_<feature>           : Edge feature values.
#
# Contrary to the TF GNN Graph Sample format, values are not flattened, they use
# the dtype of the schema, and variable-length features are represented with
# ragged tensors.

TF_GRAPH_DICT_NODES_PREFIX = "nodes_"
TF_GRAPH_DICT_EDGES_PREFIX = "edges_"
TF_GRAPH_DICT_SIZE_KEY = "reserved_size"
TF_GRAPH_DICT_ADJACENCY_KEY = "reserved_adjacency"

# A "TF GNN graph dict" i.e. the dictionary of flat (possibly sparse) tensors of
# a TF GNN Graph Sample. For instance, the output of `tf.io.parse_example` with
# the spec returned by `schema_to_tfgnn_graph_parsing_spec`.
TFGNNGraphDict = Mapping[str, Any]


def tfgnn_node_key(nodeset_name: str, suffix: str) -> str:
  """Key of a node feature (or of "#size") in a TF GNN Graph Sample."""
  return f"{TFGNN_NODES_PREFIX}{nodeset_name}.{suffix}"


def tfgnn_edge_key(edgeset_name: str, suffix: str) -> str:
  """Key of an edge feature (or of "#size" / "#source" / "#target")."""
  return f"{TFGNN_EDGES_PREFIX}{edgeset_name}.{suffix}"


def tfgnn_ragged_dim_key(feature_key: str, dim_idx: int) -> str:
  """Key of the row lengths of the `dim_idx`-th dimension of a feature.

  Args:
    feature_key: Key of the feature e.g. "nodes/n1.f1".
    dim_idx: Index of the ragged dimension. The 0-th dimension is the item (i.e.
      node or edge) dimension, and cannot be ragged. Therefore, `dim_idx` is
      always greater than zero.

  Returns:
    The key of the row lengths e.g. "nodes/n1.f1.d1".
  """
  if dim_idx <= 0:
    raise ValueError(
        "The item dimension (0) of a feature cannot be ragged. Got"
        f" dim_idx={dim_idx} for the feature {feature_key!r}."
    )
  return f"{feature_key}.d{dim_idx}"


_RESERVED_GRAPH_DICT_SUFFIXES = frozenset({
    TF_GRAPH_DICT_SIZE_KEY,
    TF_GRAPH_DICT_ADJACENCY_KEY,
})

# Matches a valid escaped entity name: non-empty, all '_' escaped as '__', and
# not ending with '_' (so entity/feature '_' boundaries remain unambiguous).
_ESCAPED_ENTITY_RE = re.compile(r"^(?:(?:__)*[^_])+$")
_ESCAPED_SEGMENT_RE = re.compile(r"^(?:[^_]|__)+$")
_SINGLE_UNDERSCORE_SPLIT_RE = re.compile(r"^((?:(?:__)*[^_])+)_((?:[^_]|__)+)$")


def _validate_graph_dict_entity_name(name: str, kind: str) -> None:
  if not name:
    raise ValueError(f"{kind} name cannot be empty.")
  if name.endswith("_"):
    raise ValueError(
        f"{kind} name {name!r} cannot end with '_' in TF Graph Dict keys."
    )


def _escape_graph_dict_key_part(name: str, is_reserved: bool = False) -> str:
  # Escape '_' as '__' so a single '_' unambiguously separates key segments
  # (e.g., ('a_b', 'c') -> 'a__b_c' vs. ('a', 'b_c') -> 'a_b__c').
  if not name:
    raise ValueError("TF Graph Dict key segment cannot be empty.")
  return name if is_reserved else name.replace("_", "__")


def tf_graph_dict_node_key(nodeset_name: str, suffix: str) -> str:
  """Key of a node feature (or of the size) in a TF Graph Dict."""
  _validate_graph_dict_entity_name(nodeset_name, "Node set")
  escaped_nodeset = _escape_graph_dict_key_part(nodeset_name)
  escaped_suffix = _escape_graph_dict_key_part(
      suffix, is_reserved=suffix in _RESERVED_GRAPH_DICT_SUFFIXES
  )
  return f"{TF_GRAPH_DICT_NODES_PREFIX}{escaped_nodeset}_{escaped_suffix}"


def tf_graph_dict_edge_key(edgeset_name: str, suffix: str) -> str:
  """Key of an edge feature (or of the adjacency) in a TF Graph Dict."""
  _validate_graph_dict_entity_name(edgeset_name, "Edge set")
  escaped_edgeset = _escape_graph_dict_key_part(edgeset_name)
  escaped_suffix = _escape_graph_dict_key_part(
      suffix, is_reserved=suffix in _RESERVED_GRAPH_DICT_SUFFIXES
  )
  return f"{TF_GRAPH_DICT_EDGES_PREFIX}{escaped_edgeset}_{escaped_suffix}"


def _feature_to_spec(
    feature_schema: schema_lib.FeatureSchema,
) -> tf.TensorSpec | tf.RaggedTensorSpec:
  """Spec of a feature value in a `TFInMemoryGraph`.

  Args:
    feature_schema: Schema of the feature.

  Returns:
    A `tf.TensorSpec` of shape [num_items, *feature_schema.shape] for
    fixed-shape features, or a `tf.RaggedTensorSpec` for variable-length
    features.
  """
  dtype = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[feature_schema.format]
  shape = list(feature_schema.shape) if feature_schema.shape else []
  if feature_schema.is_static_shape():
    return tf.TensorSpec(shape=[None] + shape, dtype=dtype)
  # Variable-length features are represented with ragged tensors. All the
  # dimensions up to the inner-most ragged one are encoded in the ragged tensor
  # (as ragged or uniform dimensions).
  # Note: `feature_schema.shape` does not contain the item dimension, so the
  # index of a dimension in the feature value is its index in the schema shape
  # plus one.
  ragged_rank = max(idx for idx, dim in enumerate(shape) if dim is None) + 1
  return tf.RaggedTensorSpec(
      shape=[None] + shape, dtype=dtype, ragged_rank=ragged_rank
  )


def schema_to_spec(
    schema_: schema_lib.GraphSchema,
) -> tf_in_memory_graph_lib.TFInMemoryGraph.Spec:
  """Converts a GraphSchema to a TFInMemoryGraph.Spec for tf serialization."""
  node_sets = {}
  for node_set_name, node_schema in schema_.node_sets.items():
    features = {}
    for feat_name, feat_schema in node_schema.features.items():
      features[feat_name] = _feature_to_spec(feat_schema)
    node_sets[node_set_name] = tf_in_memory_graph_lib.TFInMemoryNodeSet.Spec(
        num_nodes=tf.TensorSpec(shape=(), dtype=tf.int32),
        features=features,
    )

  edge_sets = {}
  for edge_set_name, edge_schema in schema_.edge_sets.items():
    features = {}
    for feat_name, feat_schema in edge_schema.features.items():
      features[feat_name] = _feature_to_spec(feat_schema)
    edge_sets[edge_set_name] = tf_in_memory_graph_lib.TFInMemoryEdgeSet.Spec(
        adjacency=tf.TensorSpec(shape=[2, None], dtype=tf.int64),
        features=features,
    )
  return tf_in_memory_graph_lib.TFInMemoryGraph.Spec(
      node_sets=node_sets, edge_sets=edge_sets
  )


def schema_to_dict_spec(
    schema_: schema_lib.GraphSchema,
) -> dict[str, tf.TensorSpec | tf.RaggedTensorSpec]:
  """Converts a GraphSchema to the specs of a `TFInMemoryGraphDict`.

  Args:
    schema_: The graph schema.

  Returns:
    A dictionary of TF Graph Dict keys to tensor specs.
  """
  result = {}
  for node_set_name, node_schema in schema_.node_sets.items():
    key = tf_graph_dict_node_key(node_set_name, TF_GRAPH_DICT_SIZE_KEY)
    result[key] = tf.TensorSpec(shape=(), dtype=tf.int32)
    for feat_name, feat_schema in node_schema.features.items():
      key = tf_graph_dict_node_key(node_set_name, feat_name)
      result[key] = _feature_to_spec(feat_schema)

  for edge_set_name, edge_schema in schema_.edge_sets.items():
    key = tf_graph_dict_edge_key(edge_set_name, TF_GRAPH_DICT_ADJACENCY_KEY)
    result[key] = tf.TensorSpec(shape=[2, None], dtype=tf.int64)
    for feat_name, feat_schema in edge_schema.features.items():
      key = tf_graph_dict_edge_key(edge_set_name, feat_name)
      result[key] = _feature_to_spec(feat_schema)
  return result


def graph_to_tf_graph(
    src: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema | None = None,
) -> tf_in_memory_graph_lib.TFInMemoryGraph:
  """Converts a graph to a TF in-memory graph.

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
  graph_dict = dgf.convert.tf_graph_to_tf_graph_dict(tf_graph)
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
    result[tf_graph_dict_node_key(node_set_name, TF_GRAPH_DICT_SIZE_KEY)] = (
        tf.convert_to_tensor(node_set.num_nodes)
    )
    for feat_name, feat_val in node_set.features.items():
      result[tf_graph_dict_node_key(node_set_name, feat_name)] = feat_val

  for edge_set_name, edge_set in src.edge_sets.items():
    result[
        tf_graph_dict_edge_key(edge_set_name, TF_GRAPH_DICT_ADJACENCY_KEY)
    ] = edge_set.adjacency
    for feat_name, feat_val in edge_set.features.items():
      result[tf_graph_dict_edge_key(edge_set_name, feat_name)] = feat_val
  return result


def _unescape_graph_dict_key_part(name: str) -> str:
  return name.replace("__", "_")


def _declared_escaped_entities(
    src: tf_in_memory_graph_lib.TFInMemoryGraphDict,
    prefix: str,
    reserved_suffix: str,
) -> set[str]:
  """Returns the escaped entity names of the size / adjacency keys of `src`."""
  suffix = f"_{reserved_suffix}"
  entities = set()
  for key in src:
    if key.startswith(prefix) and key.endswith(suffix):
      escaped_name = key[len(prefix) : -len(suffix)]
      if _ESCAPED_ENTITY_RE.match(escaped_name) is not None:
        entities.add(escaped_name)
  return entities


def _split_graph_dict_key(
    rest: str,
    prefix: str,
    src: tf_in_memory_graph_lib.TFInMemoryGraphDict,
    declared_escaped_entities: set[str],
) -> tuple[str, str]:
  """Splits a TF Graph Dict key body into (entity_name, attribute_name).

  Args:
    rest: The key without its "nodes_" / "edges_" prefix.
    prefix: The "nodes_" / "edges_" prefix of the key (for error messages and
      lookups in `src`).
    src: The TF Graph Dict the key belongs to.
    declared_escaped_entities: The escaped entity names declared by the size /
      adjacency keys of `src` (see `_declared_escaped_entities`).

  Returns:
    The unescaped entity (node set / edge set) name and attribute (feature or
    reserved) name.
  """
  for reserved_key in (TF_GRAPH_DICT_SIZE_KEY, TF_GRAPH_DICT_ADJACENCY_KEY):
    if rest.endswith(f"_{reserved_key}"):
      escaped_name = rest[: -len(reserved_key) - 1]
      if _ESCAPED_ENTITY_RE.match(escaped_name) is None:
        raise ValueError(
            f"Invalid entity name in key '{prefix}{rest}' in"
            " TFInMemoryGraphDict."
        )
      return _unescape_graph_dict_key_part(escaped_name), reserved_key

  match = _SINGLE_UNDERSCORE_SPLIT_RE.match(rest)
  if match is not None:
    return (
        _unescape_graph_dict_key_part(match.group(1)),
        _unescape_graph_dict_key_part(match.group(2)),
    )

  # `@tf.function` sanitizes '#' to '_' in parameter names, so the key of a
  # '#'-prefixed feature (e.g. '#id', '#idx') is received as
  # '{entity}__{feature}' instead of '{entity}_#{feature}'. Recover it by
  # splitting on the '__' that separates a declared entity name from the
  # feature name.
  candidates = []
  for sep_match in re.finditer("__", rest):
    escaped_entity = rest[: sep_match.start()]
    escaped_feat = rest[sep_match.end() :]
    if (
        _ESCAPED_ENTITY_RE.match(escaped_entity) is not None
        and _ESCAPED_SEGMENT_RE.match(escaped_feat) is not None
        and (
            not declared_escaped_entities
            or escaped_entity in declared_escaped_entities
        )
        # The un-sanitized key must not already be present.
        and f"{prefix}{escaped_entity}_#{escaped_feat}" not in src
    ):
      candidates.append((escaped_entity, escaped_feat))
  if len(candidates) > 1:
    raise ValueError(
        f"Ambiguous sanitized key '{prefix}{rest}' in TFInMemoryGraphDict."
    )
  if candidates:
    escaped_entity, escaped_feat = candidates[0]
    return (
        _unescape_graph_dict_key_part(escaped_entity),
        f"#{_unescape_graph_dict_key_part(escaped_feat)}",
    )
  raise ValueError(f"Unrecognized key '{prefix}{rest}' in TFInMemoryGraphDict.")


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
  tf_graph = dgf.convert.tf_graph_dict_to_tf_graph(graph_dict)
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

  known_escaped_nodesets = _declared_escaped_entities(
      src, TF_GRAPH_DICT_NODES_PREFIX, TF_GRAPH_DICT_SIZE_KEY
  )
  known_escaped_edgesets = _declared_escaped_entities(
      src, TF_GRAPH_DICT_EDGES_PREFIX, TF_GRAPH_DICT_ADJACENCY_KEY
  )

  for key, val in src.items():
    if key.startswith(TF_GRAPH_DICT_NODES_PREFIX):
      nodeset_name, attr = _split_graph_dict_key(
          key[len(TF_GRAPH_DICT_NODES_PREFIX) :],
          TF_GRAPH_DICT_NODES_PREFIX,
          src,
          known_escaped_nodesets,
      )
      if attr == TF_GRAPH_DICT_SIZE_KEY:
        nodeset_num_nodes[nodeset_name] = val
      else:
        if nodeset_name not in nodeset_features:
          nodeset_features[nodeset_name] = {}
        nodeset_features[nodeset_name][attr] = val
    elif key.startswith(TF_GRAPH_DICT_EDGES_PREFIX):
      edgeset_name, attr = _split_graph_dict_key(
          key[len(TF_GRAPH_DICT_EDGES_PREFIX) :],
          TF_GRAPH_DICT_EDGES_PREFIX,
          src,
          known_escaped_edgesets,
      )
      if attr == TF_GRAPH_DICT_ADJACENCY_KEY:
        edgeset_adjacency[edgeset_name] = val
      else:
        if edgeset_name not in edgeset_features:
          edgeset_features[edgeset_name] = {}
        edgeset_features[edgeset_name][attr] = val
    else:
      raise ValueError(
          f"Unrecognized key {key!r} in TFInMemoryGraphDict; expected prefix"
          f" {TF_GRAPH_DICT_NODES_PREFIX!r} or {TF_GRAPH_DICT_EDGES_PREFIX!r}."
      )

  tf_node_sets = {}
  for name in set(nodeset_num_nodes).union(nodeset_features):
    tf_node_sets[name] = tf_in_memory_graph_lib.TFInMemoryNodeSet(
        num_nodes=nodeset_num_nodes.get(name),
        features=nodeset_features.get(name, {}),
    )

  tf_edge_sets = {}
  for name in set(edgeset_adjacency).union(edgeset_features):
    tf_edge_sets[name] = tf_in_memory_graph_lib.TFInMemoryEdgeSet(
        adjacency=edgeset_adjacency.get(name),
        features=edgeset_features.get(name, {}),
    )

  return tf_in_memory_graph_lib.TFInMemoryGraph(
      node_sets=tf_node_sets, edge_sets=tf_edge_sets
  )


def schema_to_tfgnn_graph_parsing_spec(
    schema_: schema_lib.GraphSchema,
) -> dict[str, tf.io.VarLenFeature]:
  """Builds the parsing spec of a TF GNN Graph Sample from a graph schema.

  The returned spec is meant to be used with `tf.io.parse_example` or
  `tf.io.parse_single_example` on a serialized TF GNN Graph Sample. The result
  of the parsing (called a "TF GNN graph dict") can then be converted into a DGF
  graph with `tfgnn_graph_dict_to_tf_graph`.

  Usage example:

  ```python
  spec = dgf.convert.schema_to_tfgnn_graph_parsing_spec(schema)
  tfgnn_graph_dict = tf.io.parse_single_example(serialized_example, spec)
  tf_graph = dgf.convert.tfgnn_graph_dict_to_tf_graph(tfgnn_graph_dict, schema)
  ```

  Note that features are parsed with the dtype used to store them in the
  `tf.train.Example` proto (i.e. int64, float32 or bytes), and not with the
  dtype of the schema (e.g. int32, bool). The cast to the schema dtype is done
  by `tfgnn_graph_dict_to_tf_graph`.

  Args:
    schema_: The graph schema.

  Returns:
    A dictionary of TF GNN Graph Sample keys to `tf.io.VarLenFeature`.
  """
  feature_spec = {}

  def add_feature_to_spec(key: str, feature_schema: schema_lib.FeatureSchema):
    feature_spec[key] = tf.io.VarLenFeature(
        feature_format_lib.FEATURE_FORMAT_TO_TFGNN_TF_DTYPE[
            feature_schema.format
        ]
    )
    if not feature_schema.is_static_shape():
      for dim_idx, dim in enumerate(feature_schema.shape or []):
        if dim is None:
          # Note: The 0-th dimension of a feature value is the item dimension.
          feature_spec[tfgnn_ragged_dim_key(key, dim_idx + 1)] = (
              tf.io.VarLenFeature(tf.int64)
          )

  for node_set_name, node_set_schema in schema_.node_sets.items():
    feature_spec[tfgnn_node_key(node_set_name, TFGNN_SIZE_KEY)] = (
        tf.io.VarLenFeature(tf.int64)
    )
    for feature_name, feature_schema in node_set_schema.features.items():
      add_feature_to_spec(
          tfgnn_node_key(node_set_name, feature_name), feature_schema
      )

  for edge_set_name, edge_set_schema in schema_.edge_sets.items():
    feature_spec[tfgnn_edge_key(edge_set_name, TFGNN_SOURCE_KEY)] = (
        tf.io.VarLenFeature(tf.int64)
    )
    feature_spec[tfgnn_edge_key(edge_set_name, TFGNN_TARGET_KEY)] = (
        tf.io.VarLenFeature(tf.int64)
    )
    feature_spec[tfgnn_edge_key(edge_set_name, TFGNN_SIZE_KEY)] = (
        tf.io.VarLenFeature(tf.int64)
    )
    for feature_name, feature_schema in edge_set_schema.features.items():
      add_feature_to_spec(
          tfgnn_edge_key(edge_set_name, feature_name), feature_schema
      )
  return feature_spec


def _tfgnn_get(src: TFGNNGraphDict, key: str) -> tf.Tensor:
  """Returns the flat and dense tensor of a TF GNN Graph Sample key."""
  if key not in src:
    raise ValueError(
        f"The key {key!r} is missing from the TF GNN graph dict. Available"
        f" keys: {sorted(src.keys())}. The graph dict should be parsed with the"
        " spec returned by `schema_to_tfgnn_graph_parsing_spec`."
    )
  value = src[key]
  if isinstance(value, tf.SparseTensor):
    value = tf.sparse.to_dense(value)
  # Values of a TF GNN Graph Sample are always flat.
  return tf.reshape(value, [-1])


def _tfgnn_num_items(src: TFGNNGraphDict, key: str) -> tf.Tensor:
  """Returns a "#size" value of a TF GNN Graph Sample as an int32 scalar."""
  size = _tfgnn_get(src, key)
  tf.debugging.assert_less_equal(
      tf.size(size),
      1,
      message=f"The key {key} should contain at most one value.",
  )
  # Note: `reduce_sum` (instead of `size[0]`) also supports the case of a
  # missing (i.e. empty) size, which means "no items".
  return tf.cast(tf.reduce_sum(size), tf.int32)


def _group_uniform_dims(
    values: tf.Tensor | tf.RaggedTensor, dims: list[int]
) -> tf.Tensor | tf.RaggedTensor:
  """Groups the outer-most dimension of `values` into the `dims` dimensions.

  For example, if `values` is of shape [12, 5] and `dims` is [3], the result is
  of shape [4, 3, 5].

  Args:
    values: A dense or ragged tensor.
    dims: The static dimensions to extract from the outer-most dimension of
      `values`, from the outer-most to the inner-most one.

  Returns:
    The re-grouped tensor.
  """
  if isinstance(values, tf.RaggedTensor):
    # Note: The inner-most dimension has to be added first.
    for dim in reversed(dims):
      values = tf.RaggedTensor.from_uniform_row_length(values, dim)
    return values
  return tf.reshape(values, [-1] + dims)


def _tfgnn_feature_to_tensor(
    src: TFGNNGraphDict,
    feature_key: str,
    feature_schema: schema_lib.FeatureSchema,
) -> tf.Tensor | tf.RaggedTensor:
  """Converts a TF GNN Graph Sample feature into a dense or ragged tensor."""

  values = _tfgnn_get(src, feature_key)
  target_dtype = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[
      feature_schema.format
  ]
  if values.dtype != target_dtype:
    values = tf.cast(values, target_dtype)

  shape = list(feature_schema.shape) if feature_schema.shape else []
  if feature_schema.is_static_shape():
    # Note: The first (i.e. item) dimension is implicit.
    return tf.reshape(values, [-1] + shape)

  # Re-build the feature values, from the inner-most dimension to the outer-most
  # one. `pending_uniform_dims` contains the static dimensions that are not yet
  # applied on the accumulated `result`.
  result = values
  pending_uniform_dims: list[int] = []
  for dim_idx in range(len(shape), 0, -1):
    dim = shape[dim_idx - 1]
    if dim is None:
      result = _group_uniform_dims(result, pending_uniform_dims)
      pending_uniform_dims = []
      # Note: The 0-th dimension of a feature value is the item dimension.
      row_lengths = tf.cast(
          _tfgnn_get(src, tfgnn_ragged_dim_key(feature_key, dim_idx)), tf.int64
      )
      result = tf.RaggedTensor.from_row_lengths(result, row_lengths)
    else:
      pending_uniform_dims.insert(0, dim)
  # Note: The remaining outer-most dimension is the item dimension.
  return _group_uniform_dims(result, pending_uniform_dims)


def tfgnn_graph_dict_to_tf_graph(
    src: TFGNNGraphDict,
    schema_: schema_lib.GraphSchema,
) -> tf_in_memory_graph_lib.TFInMemoryGraph:
  """Converts a parsed TF GNN Graph Sample into a `TFInMemoryGraph`.

  This function only uses TensorFlow operations: it can be traced in a
  `tf.function` and exported in a TF SavedModel, which makes it suitable for
  model inference. This function is the TensorFlow equivalent of
  `dgf.convert.graph_dict_to_graph` (which uses NumPy).

  Usage example:

  ```python
  spec = dgf.convert.schema_to_tfgnn_graph_parsing_spec(schema)
  tfgnn_graph_dict = tf.io.parse_single_example(serialized_example, spec)
  tf_graph = dgf.convert.tfgnn_graph_dict_to_tf_graph(tfgnn_graph_dict, schema)
  ```

  The values are read according to the schema: fixed-shape features are reshaped
  into [num_items, *shape] dense tensors, and variable-length features are
  converted into ragged tensors using the "<feature>.d<i>" row lengths. Values
  are cast from their storage dtype (int64, float32 or bytes) to the dtype of
  the schema (e.g. int32, bool).

  Args:
    src: A TF GNN graph dict containing a single graph e.g. the output of
      `tf.io.parse_single_example` with the spec returned by
      `schema_to_tfgnn_graph_parsing_spec`. Values can be dense or sparse
      tensors. Batches of graphs are not supported.
    schema_: The schema of the graph.

  Returns:
    A `TFInMemoryGraph`.
  """

  node_sets = {}
  for node_set_name, node_set_schema in schema_.node_sets.items():
    features = {}
    for feature_name, feature_schema in node_set_schema.features.items():
      features[feature_name] = _tfgnn_feature_to_tensor(
          src, tfgnn_node_key(node_set_name, feature_name), feature_schema
      )
    node_sets[node_set_name] = tf_in_memory_graph_lib.TFInMemoryNodeSet(
        num_nodes=_tfgnn_num_items(
            src, tfgnn_node_key(node_set_name, TFGNN_SIZE_KEY)
        ),
        features=features,
    )

  edge_sets = {}
  for edge_set_name, edge_set_schema in schema_.edge_sets.items():
    source = tf.cast(
        _tfgnn_get(src, tfgnn_edge_key(edge_set_name, TFGNN_SOURCE_KEY)),
        tf.int64,
    )
    target = tf.cast(
        _tfgnn_get(src, tfgnn_edge_key(edge_set_name, TFGNN_TARGET_KEY)),
        tf.int64,
    )
    features = {}
    for feature_name, feature_schema in edge_set_schema.features.items():
      features[feature_name] = _tfgnn_feature_to_tensor(
          src, tfgnn_edge_key(edge_set_name, feature_name), feature_schema
      )
    edge_sets[edge_set_name] = tf_in_memory_graph_lib.TFInMemoryEdgeSet(
        adjacency=tf.stack([source, target], axis=0),
        features=features,
    )

  return tf_in_memory_graph_lib.TFInMemoryGraph(
      node_sets=node_sets, edge_sets=edge_sets
  )


def serialized_tfgnn_graph_to_tf_graph(
    src: tf.Tensor,
    schema_: schema_lib.GraphSchema,
) -> tf_in_memory_graph_lib.TFInMemoryGraph:
  """Converts a serialized TF GNN Graph Sample into a `TFInMemoryGraph`.

  This function only uses TensorFlow operations: it can be traced in a
  `tf.function` and exported in a TF SavedModel, which makes it suitable for
  model inference.

  Usage example:

  ```python
  serialized_example = ...  # A scalar string tensor.
  tf_graph = dgf.convert.serialized_tfgnn_graph_to_tf_graph(
      serialized_example, schema)
  ```

  When converting multiple graph samples, and to avoid re-creating the parsing
  spec for each of them, use `schema_to_tfgnn_graph_parsing_spec` with
  `tfgnn_graph_dict_to_tf_graph` instead.

  Args:
    src: A scalar string tensor containing a serialized `tf.train.Example` proto
      in the TF GNN Graph Sample format.
    schema_: The schema of the graph.

  Returns:
    A `TFInMemoryGraph`.
  """
  return tfgnn_graph_dict_to_tf_graph(
      tf.io.parse_single_example(
          src, schema_to_tfgnn_graph_parsing_spec(schema_)
      ),
      schema_,
  )
