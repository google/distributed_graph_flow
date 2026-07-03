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

"""Reading and writing HGraph in Beam."""

import logging
import os
from typing import Iterator, Optional, Tuple

import apache_beam as beam
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import hgraph_in_memory
from dgf.src.util import filesystem
from dgf.src.util import proto as proto_lib
from dgf.src.util import shard as shard_lib
from dgf.src.util import weak_dep
import numpy as np
import tensorflow as tf


def read_graphai_hgraph(
    pbegin: beam.Pipeline,
    path: str,
    container_type: (
        hgraph_in_memory.HGraphContainerType | str
    ) = hgraph_in_memory.HGraphContainerType.TF_RECORD,
    node_id_column: Optional[str] = None,
    edge_id_column: Optional[str] = None,
    override_schema: Optional[schema_lib.GraphSchema] = None,
    remove_dangling_edges: bool = False,
) -> distributed_graph.Graph:
  """Reads a distributed HGraph using Beam.

  This PTransform reads a HGraph where data components like node features,
  edge features, and adjacencies are stored in a distributed format.

  For small HGraph that can fit in memory, using
  "dgf.io.read_graphai_hgraph" (i.e., loading the HGraph in memory)
  might be easier to use and more efficient on small dataset.

  Usage example:

  ```python
  graph = dgf.beam.io.read_graphai_hgraph(pbegin, "gs:/my/dataset")
  ```

  Note: Currently only support TFRecord HGraph.

  Args:
    pbegin: Beam pbegin.
    path: The path to the HGraph directory.
    container_type: The type of container for the HGraph data.
    node_id_column: Column name containing the node id. If using a sstable (or
      another format with native key) and `node_id_column=None`, use the native
      key. If using a tfrecord (or another format without native key) and
      `node_id_column=None`, node_id_column defaults to '#id'. If
      `node_id_column` is set, use this column as id. This column is not
      necesseraly a feature defined in the grpah schema.
    edge_id_column: Column name containing the edge id. If None, the edges have
      no ID. Edge IDs are necessary for edge features.
    override_schema: Schema of the HGraph. If not provided, the schema is
      inferred from the TF GNN schema contained in the HGraph. Specifying the
      format allows to only includes a subset of features / nodesets / edgesets.

  Returns:
    A distributed graph.
  """
  return pbegin | f"Read {path}" >> ReadFromHGraph(
      path=path,
      container_type=container_type,
      node_id_column=node_id_column,
      edge_id_column=edge_id_column,
      override_schema=override_schema,
      remove_dangling_edges=remove_dangling_edges,
  )


class ReadFromHGraph(beam.PTransform):
  """Reads a distributed HGraph using Beam."""

  def __init__(
      self,
      path: str,
      container_type: hgraph_in_memory.HGraphContainerType | str,
      node_id_column: Optional[str],
      edge_id_column: Optional[str],
      override_schema: Optional[schema_lib.GraphSchema],
      remove_dangling_edges: bool = False,
  ):
    """Initializes the ReadFromHGraph PTransform."""
    if isinstance(container_type, str):
      container_type = hgraph_in_memory.HGraphContainerType[container_type]
    self.path = path
    self.container_type = container_type
    self.node_id_column = node_id_column
    self.edge_id_column = edge_id_column
    self.schema = override_schema
    self.remove_dangling_edges = remove_dangling_edges

    # TODO(gbm): Add support for AdjacencyList format.
    self.edge_format = distributed_graph.EdgeFormat.FLAT

  def expand(self, pbegin: beam.pvalue.PBegin) -> distributed_graph.Graph:
    """Reads the HGraph file and returns a PCollection of its lines."""

    # Import TF-GNN schema proto
    if self.schema is None:
      tf_gnn_proto = weak_dep.import_tf_gnn_proto()
      tfgnn_schema = proto_lib.read_text_proto(
          os.path.join(self.path, hgraph_in_memory.PATH_GRAPH_SCHEMA),
          tf_gnn_proto.GraphSchema,
      )
      schema = hgraph_in_memory.tfgnn_schema_to_schema(tfgnn_schema)
    else:
      schema = self.schema

    extension = hgraph_in_memory.get_extension(self.container_type)

    # Read the node features
    # TODO(gbm): Add support for graph without node features.
    node_sets = {}
    for nodeset_name, nodeset_def in schema.node_sets.items():
      file_pattern = shard_lib.shard_pattern_to_glob(
          os.path.join(
              self.path, hgraph_in_memory.PATH_NODE_FEATURE, nodeset_name
          ),
          extension,
      )

      node_sets[nodeset_name] = (
          pbegin
          | f"Read nodeset {nodeset_name}"
          >> ReadNodeSet(
              file_pattern=file_pattern,
              container_type=self.container_type,
              schema=nodeset_def,
              node_id_column=self.node_id_column,
          )
      )

    edge_sets = {}
    for edgeset_name, edgeset_def in schema.edge_sets.items():
      file_pattern = shard_lib.shard_pattern_to_glob(
          os.path.join(self.path, hgraph_in_memory.PATH_EDGES, edgeset_name),
          extension,
      )
      edge_sets[edgeset_name] = (
          pbegin
          | f"Read edgeset {edgeset_name}"
          >> ReadEdgeSet(
              file_pattern=file_pattern,
              container_type=self.container_type,
              edge_id_column=self.edge_id_column,
          )
      )

    if self.remove_dangling_edges:
      for edgeset_name, edgeset_def in schema.edge_sets.items():
        edges = edge_sets[edgeset_name]
        source_nodes = node_sets[edgeset_def.source]
        target_nodes = node_sets[edgeset_def.target]

        source_node_ids = (
            source_nodes
            | f"Get source IDs for {edgeset_name}"
            >> beam.Map(lambda node: (node.id, True))
        )
        target_node_ids = (
            target_nodes
            | f"Get target IDs for {edgeset_name}"
            >> beam.Map(lambda node: (node.id, True))
        )

        edges_by_source = edges | f"EdgesBySource {edgeset_name}" >> beam.Map(
            lambda edge: (edge.source, edge)
        )
        joined_source = {
            "edges": edges_by_source,
            "nodes": source_node_ids,
        } | f"JoinBySource {edgeset_name}" >> beam.CoGroupByKey()
        filtered_by_source = (
            joined_source
            | f"FilterDanglingSource {edgeset_name}"
            >> beam.FlatMap(
                lambda element: element[1]["edges"]
                if element[1]["nodes"]
                else []
            )
        )

        edges_by_target = (
            filtered_by_source
            | f"EdgesByTarget {edgeset_name}"
            >> beam.Map(lambda edge: (edge.target, edge))
        )
        joined_target = {
            "edges": edges_by_target,
            "nodes": target_node_ids,
        } | f"JoinByTarget {edgeset_name}" >> beam.CoGroupByKey()
        filtered_edges = (
            joined_target
            | f"FilterDanglingTarget {edgeset_name}"
            >> beam.FlatMap(
                lambda element: element[1]["edges"]
                if element[1]["nodes"]
                else []
            )
        )

        orig_count = (
            edges
            | f"CountOriginal {edgeset_name}" >> beam.combiners.Count.Globally()
        )
        filt_count = (
            filtered_edges
            | f"CountFiltered {edgeset_name}" >> beam.combiners.Count.Globally()
        )

        def log_removed_fn(element, orig, name=edgeset_name):
          removed = orig - element
          if removed > 0:
            logging.warning(
                "Removed %d dangling edges in edgeset %r.",
                removed,
                name,
            )

        _ = filt_count | f"LogRemoved {edgeset_name}" >> beam.Map(
            log_removed_fn, orig=beam.pvalue.AsSingleton(orig_count)
        )

        edge_sets[edgeset_name] = filtered_edges

    # TODO(gbm): Add support for edge features.
    # TODO(gbm): Add support for edge weights.
    return distributed_graph.Graph(
        schema=schema,
        node_sets=node_sets,
        edge_sets=edge_sets,
        edge_format=self.edge_format,
    )


class ReadTfExampleContainer(beam.PTransform):
  """Reads a container of tf.train.Example."""

  def __init__(
      self,
      file_pattern: str,
      container_type: hgraph_in_memory.HGraphContainerType,
  ):
    self.file_pattern = file_pattern
    self.container_type = container_type

  def expand(
      self, pbegin: beam.pvalue.PBegin
  ) -> beam.PCollection[tf.train.Example]:
    tfe_coder = beam.coders.ProtoCoder(tf.train.Example)
    if self.container_type == hgraph_in_memory.HGraphContainerType.TF_RECORD:
      return (
          pbegin
          | f"Read {self.file_pattern}"
          >> beam.io.tfrecordio.ReadFromTFRecord(
              file_pattern=self.file_pattern,
              coder=tfe_coder,
              compression_type=beam.io.filesystem.CompressionTypes.GZIP,
          )
      )
    else:
      raise ValueError(f"Unsupported container type: {self.container_type}")


class ReadNodeSet(beam.PTransform):
  """Reads a container of nodes."""

  def __init__(
      self,
      file_pattern: str,
      container_type: hgraph_in_memory.HGraphContainerType,
      schema: schema_lib.NodeSchema,
      node_id_column: Optional[str],
  ):
    self.file_pattern = file_pattern
    self.container_type = container_type
    self.node_id_column = node_id_column
    self.schema = schema

  def expand(
      self, pbegin: beam.pvalue.PBegin
  ) -> beam.PCollection[distributed_graph.Node]:
    tfe_coder = beam.coders.ProtoCoder(tf.train.Example)

    if self.container_type == hgraph_in_memory.HGraphContainerType.TF_RECORD:

      if self.node_id_column is None:
        node_id_column = hgraph_in_memory.DEFAULT_KEY_ID
      else:
        node_id_column = self.node_id_column

      return (
          pbegin
          | f"Read {self.file_pattern}"
          >> beam.io.tfrecordio.ReadFromTFRecord(
              file_pattern=self.file_pattern,
              coder=tfe_coder,
              compression_type=beam.io.filesystem.CompressionTypes.GZIP,
          )
          | "Build nodes"
          >> beam.Map(
              nonkeyed_tf_example_to_node,
              schema=self.schema,
              node_id_column=node_id_column,
          )
      )

    else:
      raise ValueError(f"Unsupported container type: {self.container_type}")


def tf_feature_to_feature(
    example: tf.train.Example,
    key: str,
    feature_schema: schema_lib.FeatureSchema,
) -> np.ndarray:
  """Extracts features from a tf.train.Example feature.

  Args:
    example: A tf.train.Example.
    key: The key of the feature to extract.
    feature_schema: Schema of the feature.

  Returns:
    A numpy array containing the feature values.
  """

  feature = example.features.feature.get(key)
  if feature is None:
    raise ValueError(f"Missing feature {key}")
  if feature.HasField("int64_list"):
    value = np.array(feature.int64_list.value, dtype=np.int64)
  elif feature.HasField("float_list"):
    value = np.array(feature.float_list.value, dtype=np.float32)
  elif feature.HasField("bytes_list"):
    value = np.array(feature.bytes_list.value, dtype=np.bytes_)
  else:
    raise ValueError("Non supported type")

  if feature_schema.shape is None or feature_schema.shape == ():
    if value.shape[0] != 1:
      raise ValueError(
          f"Expected scalar value for feature '{key}' but got value with shape"
          f" {value.shape}. If the feature is multi-dimentionnal, its `shape`"
          " should be specified in the Graph Schema. Note: If you cannot fix"
          " the schema file, use the `override_schema` or `schema_transformer`"
          " argument of the `read_graphai_hgraph` function."
      )
    value = np.squeeze(value, axis=0)
  elif value.ndim != 1:
    value = np.reshape(value, feature_schema.shape)  # pyrefly: ignore[no-matching-overload]
  return value


def tf_feature_to_bytes(example: tf.train.Example, key: str) -> bytes:
  """Extracts a byte value from a tf.train.Example feature.

  Args:
    example: A tf.train.Example.
    key: The key of the feature to extract.

  Returns:
    A numpy array containing the feature values.
  """
  feature = example.features.feature.get(key)
  if feature is None:
    raise ValueError(f"Missing feature {key}")
  if feature.HasField("bytes_list"):
    if len(feature.bytes_list.value) != 1:
      raise ValueError(
          f"Expected a single bytes value for {key}. Instead got"
          f" {len(feature.bytes_list.value)} values."
      )
    return feature.bytes_list.value[0]
  elif feature.HasField("int64_list"):
    if len(feature.int64_list.value) != 1:
      raise ValueError(
          f"Expected a single int value for {key}. Instead got"
          f" {len(feature.bytes_list.value)} values."
      )
    return feature.int64_list.value[0]
  else:
    raise ValueError("Non supported type")


def nonkeyed_tf_example_to_node(
    example: tf.train.Example,
    schema: schema_lib.NodeSchema,
    node_id_column: str,
) -> distributed_graph.Node:
  """Build a node from a tf example."""
  node_features = {}
  for feature_name, feature_schema in schema.features.items():
    if feature_name == node_id_column:
      continue
    node_features[feature_name] = tf_feature_to_feature(
        example, feature_name, feature_schema
    )
  node_id = tf_feature_to_bytes(example, node_id_column)
  return distributed_graph.Node(id=node_id, features=node_features)


def keyed_tf_example_to_node(
    keyed_example: Tuple[bytes, tf.train.Example],
    schema: schema_lib.NodeSchema,
    node_id_column: Optional[str],
) -> distributed_graph.Node:
  """Build a node from a tf example."""
  key, example = keyed_example
  node_features = {}
  for feature_name, feature_schema in schema.features.items():
    if feature_name == node_id_column:
      continue
    node_features[feature_name] = tf_feature_to_feature(
        example, feature_name, feature_schema
    )
  if node_id_column is not None:
    node_features[node_id_column] = np.array([key], dtype=np.bytes_)
  return distributed_graph.Node(id=key, features=node_features)


def tf_example_to_edge(
    example: tf.train.Example,
    edge_id_column: Optional[str],
) -> distributed_graph.Edge:
  """Extracts edge adjacency from a tf example.

  Args:
    example: A tf.train.Example.
    edge_id_column: Column containing the edge id.

  Returns:
    Edge adjacency.
  """
  edge_target = tf_feature_to_bytes(example, hgraph_in_memory.KEY_TARGET)
  edge_source = tf_feature_to_bytes(example, hgraph_in_memory.KEY_SOURCE)

  if edge_id_column is not None and edge_id_column in example.features.feature:
    # TODO(gbm): How to generate an error where the ID is "missing" while
    # allowing the automatic detection of id.
    edge_id = tf_feature_to_bytes(example, edge_id_column)
  else:
    edge_id = None

  return distributed_graph.Edge(
      source=edge_source, target=edge_target, id=edge_id
  )


class ReadEdgeSet(beam.PTransform):
  """Reads flat edge sets from various formats."""

  def __init__(
      self,
      file_pattern: str,
      container_type: hgraph_in_memory.HGraphContainerType,
      edge_id_column: Optional[str],
  ):
    self.file_pattern = file_pattern
    self.container_type = container_type
    self.edge_id_column = edge_id_column

  def expand(
      self, pbegin: beam.pvalue.PBegin
  ) -> beam.PCollection[distributed_graph.Edge]:

    if self.edge_id_column is None:
      edge_id_column = hgraph_in_memory.DEFAULT_KEY_ID
    else:
      edge_id_column = self.edge_id_column

    if self.container_type == hgraph_in_memory.HGraphContainerType.TF_RECORD:
      return (
          pbegin
          | ReadTfExampleContainer(
              file_pattern=self.file_pattern,
              container_type=self.container_type,
          )
          | "Import edgeset"
          >> beam.Map(tf_example_to_edge, edge_id_column=edge_id_column)
      )
    else:
      raise ValueError(f"Unsupported container type: {self.container_type}")


class WriteTfExampleContainer(beam.PTransform):
  """Writes a container of tf.train.Example."""

  def __init__(
      self,
      file_path_prefix: str,
      extension: str,
      container_type: hgraph_in_memory.HGraphContainerType,
  ):
    self.file_path_prefix = file_path_prefix
    self.extension = extension
    self.container_type = container_type

  def expand(
      self, pcoll: beam.PCollection[tf.train.Example]
  ) -> beam.pvalue.PDone:
    tfe_coder = beam.coders.ProtoCoder(tf.train.Example)
    if self.container_type == hgraph_in_memory.HGraphContainerType.TF_RECORD:
      return (
          pcoll
          | f"Write {self.file_path_prefix}"
          >> beam.io.tfrecordio.WriteToTFRecord(
              file_path_prefix=self.file_path_prefix,
              file_name_suffix=self.extension,
              coder=tfe_coder,
              compression_type=beam.io.filesystem.CompressionTypes.GZIP,
          )
      )
    else:
      raise ValueError(f"Unsupported container type: {self.container_type}")


def node_to_tf_example(
    node: distributed_graph.Node,
    node_id_column: Optional[str],
    nodeset_schema: schema_lib.NodeSchema,
) -> tf.train.Example:
  """Converts node features to a tf example.

  Args:
    node: Input distributed node.
    node_id_column: Optional tf example column where to export the node id.
    nodeset_schema: Schema of the node set.

  Returns:
    A tf.train.Example.
  """

  example = tf.train.Example()

  for feature_name, feature_schema in nodeset_schema.features.items():
    if feature_name == node_id_column:
      value = [node.id]
    else:
      value = node.features[feature_name]  # pyrefly: ignore[unsupported-operation]
      if value.ndim == 0:
        value = np.expand_dims(value, axis=0)

    if feature_schema.format.is_integer():
      example.features.feature[feature_name].int64_list.value.extend(value)
    elif feature_schema.format.is_float():
      example.features.feature[feature_name].float_list.value.extend(value)
    elif feature_schema.format == schema_lib.FeatureFormat.BYTES:
      example.features.feature[feature_name].bytes_list.value.extend(value)
    else:
      raise ValueError(f"Non supported type {feature_schema.format}")

  return example


def set_tf_scalar(dst, value, format: schema_lib.FeatureFormat):
  if format.is_integer():
    dst.int64_list.value.append(value)
  elif format == schema_lib.FeatureFormat.BYTES:
    dst.bytes_list.value.append(value)
  else:
    raise ValueError(f"Unsupported format: {format}")


def edge_to_tf_example(
    edge: distributed_graph.Edge,
    edge_id_column: Optional[str],
    edge_schema: schema_lib.EdgeSchema,
    schema: schema_lib.GraphSchema,
) -> tf.train.Example:
  """Converts edge adjacency to a tf example.

  Args:
    edge: a distributed_graph.Edge instance.
    edge_id_column: Optional tf example column where to export the edge id.
    edge_schema: Schema of the edge set.
    schema: Schema of the entire graph.

  Returns:
    A tf.train.Example.
  """

  example = tf.train.Example()

  source_format = (
      schema.node_sets[edge_schema.source]
      .features[hgraph_in_memory.DEFAULT_KEY_ID]
      .format
  )
  target_format = (
      schema.node_sets[edge_schema.target]
      .features[hgraph_in_memory.DEFAULT_KEY_ID]
      .format
  )

  set_tf_scalar(
      example.features.feature[hgraph_in_memory.KEY_SOURCE],
      edge.source,
      source_format,
  )
  set_tf_scalar(
      example.features.feature[hgraph_in_memory.KEY_TARGET],
      edge.target,
      target_format,
  )

  if (
      edge.id is not None
      and edge_id_column is not None
      and edge_id_column not in example.features.feature
  ):
    example.features.feature[edge_id_column].bytes_list.value.append(edge.id)
  return example


def write_graphai_hgraph(
    graph: distributed_graph.Graph,
    path: str,
    container_type: (
        hgraph_in_memory.HGraphContainerType | str
    ) = hgraph_in_memory.HGraphContainerType.TF_RECORD,
    node_id_column: Optional[str] = None,
    edge_id_column: Optional[str] = None,
):
  """Initializes the WriteToHGraph PTransform.

  Args:
    graph: Graph to write.
    path: The path to the HGraph directory.
    container_type: The type of container for the HGraph data.
    node_id_column: If provided, the node ID is exported as a column with this
      name. If not provided, for indexed formats (e.g., SSTable), the node ID is
      used as the native key, and for formats without native keys (e.g.,
      TFRecord), the node ID is exported as a feature named `"#id"`.
    edge_id_column: If provided, the edge ID is exported as a feature with this
      name.
  """
  if isinstance(container_type, str):
    container_type = hgraph_in_memory.HGraphContainerType[container_type]

  filesystem.makedirs(path)

  # Write the schema
  tfgnn_schema = hgraph_in_memory.schema_to_tfgnn_schema(graph.schema)
  proto_lib.write_text_proto(
      os.path.join(path, hgraph_in_memory.PATH_GRAPH_SCHEMA), tfgnn_schema
  )

  extension = hgraph_in_memory.get_extension(container_type)

  if container_type == hgraph_in_memory.HGraphContainerType.TF_RECORD:
    if node_id_column is None:
      node_id_column = hgraph_in_memory.DEFAULT_KEY_ID
    if edge_id_column is None:
      edge_id_column = hgraph_in_memory.DEFAULT_KEY_ID

  # Write the node features
  for nodeset_name, nodeset_schema in graph.schema.node_sets.items():
    nodeset = graph.node_sets[nodeset_name]
    _ = (
        nodeset
        | f"Export nodeset {nodeset_name}"
        >> beam.Map(
            node_to_tf_example,
            node_id_column=node_id_column,
            nodeset_schema=nodeset_schema,
        )
        | f"Write nodeset {nodeset_name}"
        >> WriteTfExampleContainer(
            file_path_prefix=os.path.join(
                path, hgraph_in_memory.PATH_NODE_FEATURE, nodeset_name
            ),
            extension=extension,
            container_type=container_type,
        )
    )

  # Write the edge adjacency
  for edgeset_name, edgeset_schema in graph.schema.edge_sets.items():
    edgeset = graph.edge_sets[edgeset_name]
    _ = (
        edgeset
        | f"Export edgeset {edgeset_name}"
        >> beam.Map(
            edge_to_tf_example,
            edge_id_column=edge_id_column,
            edge_schema=edgeset_schema,
            schema=graph.schema,
        )
        | f"Write edgeset {edgeset_name}"
        >> WriteTfExampleContainer(
            file_path_prefix=os.path.join(
                path, hgraph_in_memory.PATH_EDGES, edgeset_name
            ),
            extension=extension,
            container_type=container_type,
        )
    )
