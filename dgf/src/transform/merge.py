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

"""Batching of graphs for training."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from dgf.src.data import in_memory_graph
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import tf_in_memory_graph
from dgf.src.transform import timeseries_padding
from dgf.src.util import temporal as temporal_util
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
import numpy as np


class InsufficientPaddingError(ValueError):
  pass


class GraphMerger:
  """Merges multiple `InMemoryGraph` instances into a single graph.

  Graphs are concatenated sequentially. Node and edge indices from each graph
  are offset to prevent collisions in the merged graph.

  Padding nodes and edges are added according to the `padding` configuration.
  Padding edges are defined to connect to the last node index of their
  respective destination node set. Therefore, ensure that each node set's
  padding size is at least one greater than the maximum number of nodes expected
  before padding.

  If `padding` is None, the edges/nodes are not padded.

  If `padding` is provided, but not large enough to encode all the graphs
  (e.g., the padding has space for 100 nodes but 120 nodes are provided),
  raises an `InsufficientPaddingError` exception.

  Attributes:
    schema: The original input GraphSchema.
    padding: The padding configuration to apply.
    sentinel_offset: Where to include the offset of sentinel nodes / edges added
      for the padding. Only used when `padding` is set.
    schema_cache: Pre-computed timeseries schema cache.
  """

  def __init__(
      self,
      schema: schema_lib.GraphSchema,
      padding: Optional[padding_lib.Padding] = None,
      sentinel_offset: bool = True,
      schema_cache: Optional[temporal_util.TimeseriesSchemaCache] = None,
  ):
    self.schema = schema
    self.padding = padding
    self.sentinel_offset = sentinel_offset

    if padding is not None:
      unknown_node_sets = set(padding.node_sets) - set(schema.node_sets)
      if unknown_node_sets:
        raise ValueError(
            f"Padding specifies unknown node sets: {sorted(unknown_node_sets)}."
        )
      unknown_edge_sets = set(padding.edge_sets) - set(schema.edge_sets)
      if unknown_edge_sets:
        raise ValueError(
            f"Padding specifies unknown edge sets: {sorted(unknown_edge_sets)}."
        )

    self._has_timeseries_padding = (
        timeseries_padding.has_timeseries_padding(padding)
        and temporal_util.schema_has_timeseries_features(schema)
    )

    if self._has_timeseries_padding:
      assert padding is not None
      if schema_cache is None:
        schema_cache = temporal_util.extract_timeseries_schema_cache(schema)
      self._padded_schema = timeseries_padding.pad_timeseries_schema(
          schema,
          padding=padding,
          schema_cache=schema_cache,
      )
    else:
      self._padded_schema = schema

    self.schema_cache = schema_cache

  def output_schema(self) -> schema_lib.GraphSchema:
    """Returns the schema of the merged graph."""
    return self._padded_schema

  def __call__(
      self,
      graphs: List[in_memory_graph.InMemoryGraph],
  ) -> Tuple[in_memory_graph.InMemoryGraph, Dict[str, np.ndarray]]:
    """Merges a list of graphs into a single graph.

    Args:
      graphs: A list of graphs to merge.

    Returns:
      A tuple containing:
        - The merged `InMemoryGraph`.
        - A dictionary mapping each node set name to a NumPy array. The array
          at `node_set_offsets[node_set_name][i]` contains the starting node
          index of the `i`-th graph from the input `graphs` list within the
          merged graph's node set named `node_set_name`. If
          sentinel_offset=True, assuming the `graphs` contains `n` values, all
          the `node_set_offsets[node_set_name]` will contain `n+1` values. This
          extra value represents the fake node used to pad edges. If
          sentinel_offset=False, the `node_set_offsets[node_set_name]` will
          contain `n` values.
    """
    if not graphs:
      return in_memory_graph.InMemoryGraph({}, {}), {}

    if self._has_timeseries_padding:
      assert self.padding is not None
      # TODO(simonmeierhans): Avoid additional memory copy for timeseries
      # padding.
      graphs = [
          timeseries_padding.pad_timeseries_graph(
              graph,
              self.schema,
              padding=self.padding,
              schema_cache=self.schema_cache,
          )
          for graph in graphs
      ]

    schema = self._padded_schema
    padding = self.padding
    sentinel_offset = self.sentinel_offset

    # TODO(gbm): Failure / warning / crop if the padding is not enought.
    # TODO(gbm): TF compatible.
    # TODO(gbm): Make padding optionnal.
    # TODO(gbm): Implement the graph merging in c++ as part of the sampler.

    # Index of the first node of each graph in the merged graph, for each
    # nodeset.
    node_set_offsets: Dict[str, np.ndarray] = {}
    for node_set_name in schema.node_sets:
      num_nodes = [
          graph.node_sets[node_set_name].num_nodes for graph in graphs
      ]
      offsets = np.cumsum(np.array(num_nodes, dtype=np.int32))
      node_set_offsets[node_set_name] = np.insert(offsets, 0, 0)

    # Determine the edgeset offsets.
    # Index of the first edge of each graph in the merged graph, for each
    # nodeset.
    edge_set_offsets: Dict[str, np.ndarray] = {}
    for edge_set_name in schema.edge_sets:
      num_edges = [
          graph.edge_sets[edge_set_name].adjacency.shape[1] for graph in graphs
      ]
      offsets = np.cumsum(np.array(num_edges, dtype=np.int32))
      edge_set_offsets[edge_set_name] = np.insert(offsets, 0, 0)

    # Merge the nodesets + some more nodeset related compute.
    node_set_sentinel_idx: Dict[str, int] = {}
    merged_node_sets: Dict[str, in_memory_graph.InMemoryNodeSet] = {}
    for node_set_name, node_set_schema in schema.node_sets.items():
      merged_features = {}
      node_offsets = node_set_offsets[node_set_name]
      # Note: node_offsets[-1] is the number of real nodes
      num_real_nodes = node_offsets[-1]
      num_sentinel_nodes = 0

      target_num_nodes = (
          padding.node_sets[node_set_name].num_nodes
          if padding and node_set_name in padding.node_sets
          else None
      )

      # Pad the nodeset with sentinel nodes if a padding configuration is
      # provided.
      if target_num_nodes is not None:
        num_nodes = target_num_nodes
        # Note: node_offsets[-1] is the number of real nodes
        num_sentinel_nodes = num_nodes - num_real_nodes
        # The sentinel node is the last one.
        node_set_sentinel_idx[node_set_name] = num_nodes - 1
        if num_sentinel_nodes < 1:
          raise InsufficientPaddingError(
              f"Padding for node set '{node_set_name}' is insufficient. Required"
              f" at least {num_real_nodes + 1} nodes (including the sentinel"
              f" node), but the padder only defines {num_nodes}."
          )
      else:
        num_nodes = num_real_nodes

      for feature_name, _ in node_set_schema.features.items():
        # Collect the feature values
        merged_values = [
            graph.node_sets[node_set_name].features[feature_name]
            for graph in graphs
        ]

        if target_num_nodes is not None:
          # Create a feature padding item
          first_value = merged_values[0]
          padding_shape = list(first_value.shape)
          padding_shape[0] = num_sentinel_nodes

          if first_value.dtype.type is np.bytes_:
            padding_item = np.full(padding_shape, b"", dtype=first_value.dtype)
          elif first_value.dtype == np.object_:
            padding_item = np.empty(padding_shape, dtype=np.object_)
          else:
            padding_item = np.zeros(
                shape=padding_shape, dtype=first_value.dtype
            )
          merged_values.append(padding_item)

        # Merge together the feature of all the graphs (+ the padding).
        merged_features[feature_name] = np.concatenate(merged_values, axis=0)

      merged_node_sets[node_set_name] = in_memory_graph.InMemoryNodeSet(
          features=merged_features,
          num_nodes=num_nodes,
      )

    # Merge the edges
    merged_edge_sets: Dict[str, in_memory_graph.InMemoryEdgeSet] = {}
    for edge_set_name, edge_set_schema in schema.edge_sets.items():
      merged_features = {}
      # Collect the adjacency of all the graph and apply the offset.
      merged_adjacency = []
      for graph_idx, graph in enumerate(graphs):
        adjacency = graph.edge_sets[edge_set_name].adjacency
        adjacency_offset = np.array([
            [node_set_offsets[edge_set_schema.source][graph_idx]],
            [node_set_offsets[edge_set_schema.target][graph_idx]],
        ])
        merged_adjacency.append(adjacency + adjacency_offset)

      edge_offsets = edge_set_offsets[edge_set_name]
      num_real_edges = edge_offsets[-1]
      num_padding = 0

      target_num_edges = (
          padding.edge_sets[edge_set_name].num_edges
          if padding and edge_set_name in padding.edge_sets
          else None
      )

      # Pad the edgeset with sentinel edges if a padding configuration is
      # provided.
      if target_num_edges is not None:
        num_edges = target_num_edges
        num_padding = num_edges - num_real_edges
        if num_padding < 0:
          raise InsufficientPaddingError(
              f"Padding for edge set '{edge_set_name}' is insufficient. "
              f"Required at least {num_real_edges} edges, but the padder only "
              f"defines {num_edges}."
          )

        # Add padding edges pointing to sentinel nodes.
        if (
            edge_set_schema.source not in node_set_sentinel_idx
            or edge_set_schema.target not in node_set_sentinel_idx
        ):
          raise ValueError(
              f"Padding for edge set '{edge_set_name}' requires sentinel nodes"
              f" on both source '{edge_set_schema.source}' and target"
              f" '{edge_set_schema.target}', but one or both node sets are not"
              " padded with sentinel nodes."
          )
        padding_node_src = node_set_sentinel_idx[edge_set_schema.source]
        padding_node_trg = node_set_sentinel_idx[edge_set_schema.target]
        adj_dtype = graphs[0].edge_sets[edge_set_name].adjacency.dtype
        padding_edge = np.array(
            [[padding_node_src], [padding_node_trg]], dtype=adj_dtype
        )
        padding_block = np.tile(padding_edge, (1, num_padding))
        merged_adjacency.append(padding_block)

      for feature_name, _ in edge_set_schema.features.items():
        # TODO(simonmeierhans): Remove this check once the sampler supports
        # edge features. Subgraphs from samplers currently omit edge features.
        if feature_name not in graphs[0].edge_sets[edge_set_name].features:
          continue
        # Collect the feature values
        merged_values = [
            graph.edge_sets[edge_set_name].features[feature_name]
            for graph in graphs
        ]

        if target_num_edges is not None:
          # Create a feature padding item
          first_value = merged_values[0]
          padding_shape = list(first_value.shape)
          padding_shape[0] = num_padding

          if first_value.dtype.type is np.bytes_:
            padding_item = np.full(padding_shape, b"", dtype=first_value.dtype)
          elif first_value.dtype == np.object_:
            padding_item = np.empty(padding_shape, dtype=np.object_)
          else:
            padding_item = np.zeros(
                shape=padding_shape, dtype=first_value.dtype
            )
          merged_values.append(padding_item)

        # Merge together the feature of all the graphs (+ the padding).
        merged_features[feature_name] = np.concatenate(merged_values, axis=0)

      # Merge all the adjacencies + padding.
      merged_adjacency = np.concatenate(merged_adjacency, axis=1)

      merged_edge_sets[edge_set_name] = in_memory_graph.InMemoryEdgeSet(
          adjacency=merged_adjacency,
          features=merged_features,
      )

    if not sentinel_offset:
      node_set_offsets = {k: v[:-1] for k, v in node_set_offsets.items()}

    return (
        in_memory_graph.InMemoryGraph(merged_node_sets, merged_edge_sets),
        node_set_offsets,
    )





def create_padding_item(value, num_padding_items):
  """Creates a padding item for a given value."""
  if isinstance(value, tf.RaggedTensor):
    padding_flat_shape = tf.concat(
        [[0], tf.shape(value.flat_values)[1:]], axis=0
    )
    if value.dtype == tf.string:
      padding_flat_values = tf.fill(padding_flat_shape, "")
    else:
      padding_flat_values = tf.zeros(padding_flat_shape, dtype=value.dtype)

    padding_row_splits = tf.zeros(
        [num_padding_items + 1], dtype=value.row_splits.dtype
    )
    return tf.RaggedTensor.from_row_splits(
        values=padding_flat_values,
        row_splits=padding_row_splits,
    )
  else:
    padding_shape = tf.concat(
        [[num_padding_items], tf.shape(value)[1:]], axis=0
    )

    if value.dtype == tf.string:
      return tf.fill(padding_shape, "")
    else:
      return tf.zeros(padding_shape, dtype=value.dtype)


def pad_graph_tensorflow(
    graph: tf_in_memory_graph.TFInMemoryGraph,
    schema: schema_lib.GraphSchema,
    padding: padding_lib.Padding,
) -> tf_in_memory_graph.TFInMemoryGraph:
  """Pads a TFInMemoryGraph with sentinels according to the padding.

  Similar to the padding implemented in GraphMerger with sentinel_offsets and
  padder, but work on TFInMemoryGraph instead of
  (Numpy)InMemoryGraphs.

  Args:
    graph: The graph to pad.
    schema: Graph schema.
    padding: The padding configuration to apply.

  Returns:
    The padded `TFInMemoryGraph`.
  """
  if temporal_util.schema_has_timeseries_features(schema):
    raise NotImplementedError(
        "pad_graph_tensorflow does not support timeseries padding yet."
    )

  assert padding.node_sets
  assert padding.edge_sets

  merged_node_sets: Dict[str, tf_in_memory_graph.TFInMemoryNodeSet] = {}
  for node_set_name, node_set_schema in schema.node_sets.items():
    merged_features = {}
    node_set_padding = padding.node_sets[node_set_name]
    if node_set_padding.num_nodes is None:
      raise ValueError(
          f"Node set '{node_set_name}' does not have num_nodes defined in"
          " padding."
      )
    num_nodes = node_set_padding.num_nodes
    num_real_nodes = tf.cast(graph.node_sets[node_set_name].num_nodes, tf.int32)

    num_sentinel_nodes = num_nodes - num_real_nodes

    tf.debugging.assert_greater_equal(
        num_sentinel_nodes,
        1,
        message=f"Padding for node set '{node_set_name}' is insufficient.",
    )

    for feature_name, _ in node_set_schema.features.items():
      value = graph.node_sets[node_set_name].features[feature_name]
      padding_item = create_padding_item(value, num_sentinel_nodes)
      merged_feat = tf.concat([value, padding_item], axis=0)
      if not isinstance(merged_feat, tf.RaggedTensor):
        shape = merged_feat.shape.as_list()
        shape[0] = num_nodes
        merged_feat = tf.ensure_shape(merged_feat, shape)
      merged_features[feature_name] = merged_feat

    merged_node_sets[node_set_name] = tf_in_memory_graph.TFInMemoryNodeSet(
        features=merged_features,
        num_nodes=num_nodes,
    )

  merged_edge_sets: Dict[str, tf_in_memory_graph.TFInMemoryEdgeSet] = {}
  for edge_set_name, edge_set_schema in schema.edge_sets.items():
    adjacency = graph.edge_sets[edge_set_name].adjacency

    edge_set_padding = padding.edge_sets[edge_set_name]
    if edge_set_padding.num_edges is None:
      raise ValueError(
          f"Edge set '{edge_set_name}' does not have num_edges defined in"
          " padding."
      )
    num_edges = edge_set_padding.num_edges
    num_real_edges = tf.shape(adjacency)[1]
    num_padding = num_edges - num_real_edges

    tf.debugging.assert_greater_equal(
        num_padding,
        0,
        message=f"Padding for edge set '{edge_set_name}' is insufficient.",
    )

    src_padding = padding.node_sets[edge_set_schema.source]
    trg_padding = padding.node_sets[edge_set_schema.target]
    if src_padding.num_nodes is None or trg_padding.num_nodes is None:
      raise ValueError(
          f"Padding for edge set '{edge_set_name}' requires sentinel nodes on"
          f" both source '{edge_set_schema.source}' and target"
          f" '{edge_set_schema.target}'."
      )
    padding_node_src = src_padding.num_nodes - 1
    padding_node_trg = trg_padding.num_nodes - 1

    padding_edge = tf.constant(
        [[padding_node_src], [padding_node_trg]], dtype=adjacency.dtype
    )
    padding_block = tf.tile(padding_edge, [1, num_padding])

    merged_adjacency = tf.concat([adjacency, padding_block], axis=1)
    merged_adjacency = tf.ensure_shape(merged_adjacency, [2, num_edges])

    merged_features = {}
    for feature_name, _ in edge_set_schema.features.items():
      # TODO(simonmeierhans): Remove this check once the sampler supports
      # edge features. Subgraphs from samplers currently omit edge features.
      if feature_name not in graph.edge_sets[edge_set_name].features:
        continue
      value = graph.edge_sets[edge_set_name].features[feature_name]
      padding_item = create_padding_item(value, num_padding)
      merged_feat = tf.concat([value, padding_item], axis=0)
      if not isinstance(merged_feat, tf.RaggedTensor):
        shape = merged_feat.shape.as_list()
        shape[0] = num_edges
        merged_feat = tf.ensure_shape(merged_feat, shape)
      merged_features[feature_name] = merged_feat

    merged_edge_sets[edge_set_name] = tf_in_memory_graph.TFInMemoryEdgeSet(
        adjacency=merged_adjacency,
        features=merged_features,
    )

  return tf_in_memory_graph.TFInMemoryGraph(merged_node_sets, merged_edge_sets)


def remove_padding_sentinels(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    offsets: Dict[str, np.ndarray],
) -> in_memory_graph.InMemoryGraph:
  """Removes the sentinel nodes and edges added by `GraphMerger`.

  The `graph` and `offsets` arguments are the two return values from
  `GraphMerger`. This method requires `GraphMerger` to have been called with
  `sentinel_offset=True`.

  Usage example:

  ```python
    padded_graph, offsets = GraphMerger(
        schema, padding, sentinel_offset=True
    )([graph])
    unpadded_graph = remove_padding_sentinels(padded_graph, schema, offsets)
    assert unpadded_graph == graph
  ```

  This method can be used to more easily inspect or plot padded graphs, as
  padding makes the interpretation harder.

  Args:
    graph: The padded graph.
    schema: Graph schema.
    offsets: Node set offsets returned by `GraphMerger`.

  Returns:
    A new `InMemoryGraph` with sentinel nodes and edges removed.
  """
  unpadded_node_sets: Dict[str, in_memory_graph.InMemoryNodeSet] = {}
  for node_set_name, node_set_schema in schema.node_sets.items():
    node_set = graph.node_sets[node_set_name]
    num_real_nodes = int(offsets[node_set_name][-1])

    unpadded_features = {}
    for feature_name, _ in node_set_schema.features.items():
      feature_value = node_set.features[feature_name]
      unpadded_features[feature_name] = feature_value[:num_real_nodes]

    unpadded_node_sets[node_set_name] = in_memory_graph.InMemoryNodeSet(
        features=unpadded_features,
        num_nodes=num_real_nodes,
    )

  unpadded_edge_sets: Dict[str, in_memory_graph.InMemoryEdgeSet] = {}
  for edge_set_name, edge_set_schema in schema.edge_sets.items():
    edge_set = graph.edge_sets[edge_set_name]
    num_real_nodes_src = int(offsets[edge_set_schema.source][-1])
    num_real_nodes_trg = int(offsets[edge_set_schema.target][-1])

    adjacency = edge_set.adjacency
    mask = (adjacency[0] < num_real_nodes_src) & (
        adjacency[1] < num_real_nodes_trg
    )
    num_real_edges = int(np.sum(mask))

    unpadded_adjacency = adjacency[:, :num_real_edges]
    unpadded_features = {}
    for feature_name, _ in edge_set_schema.features.items():
      # TODO(simonmeierhans): Remove this check once the sampler supports
      # edge features. Subgraphs from samplers currently omit edge features.
      if feature_name not in edge_set.features:
        continue
      feature_value = edge_set.features[feature_name]
      unpadded_features[feature_name] = feature_value[:num_real_edges]

    unpadded_edge_sets[edge_set_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=unpadded_adjacency,
        features=unpadded_features,
    )

  return in_memory_graph.InMemoryGraph(unpadded_node_sets, unpadded_edge_sets)
