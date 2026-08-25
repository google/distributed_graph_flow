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

"""Common utilities for Beam-based TF-example graph processing."""

from __future__ import annotations

from typing import Optional, Tuple

from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import tf_graph_common
from dgf.src.util.weak_dep.weak_dep_apache_beam import PTransform, beam
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
import numpy as np


class ReadTfExampleContainer(PTransform):
  """Reads a container format outputting a unified PCollection of tf.train.Example."""

  def __init__(
      self,
      file_pattern: str,
      container_type: tf_graph_common.TfExampleContainer,
  ):
    self.file_pattern = file_pattern
    self.container_type = container_type

  def expand(
      self, pbegin: beam.pvalue.PBegin
  ) -> beam.PCollection[tf.train.Example]:
    tfe_coder = beam.coders.ProtoCoder(tf.train.Example)
    if self.container_type == tf_graph_common.TfExampleContainer.TF_RECORD:
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


class WriteTfExampleContainer(PTransform):
  """Writes a PCollection of tf.train.Example to an underlying static container."""

  def __init__(
      self,
      file_path_prefix: str,
      extension: str,
      container_type: tf_graph_common.TfExampleContainer,
      num_shards: int,
  ):
    self.file_path_prefix = file_path_prefix
    self.extension = extension
    self.container_type = container_type
    self.num_shards = num_shards

  def expand(
      self, pcoll: beam.PCollection[tf.train.Example]
  ) -> beam.pvalue.PDone:
    tfe_coder = beam.coders.ProtoCoder(tf.train.Example)
    if self.container_type == tf_graph_common.TfExampleContainer.TF_RECORD:
      return (
          pcoll
          | f"Write {self.file_path_prefix}"
          >> beam.io.tfrecordio.WriteToTFRecord(
              file_path_prefix=self.file_path_prefix,
              file_name_suffix=self.extension,
              coder=tfe_coder,
              num_shards=self.num_shards,
              compression_type=beam.io.filesystem.CompressionTypes.GZIP,
          )
      )
    else:
      raise ValueError(f"Unsupported container type: {self.container_type}")


def nonkeyed_tf_example_to_node(
    example: tf.train.Example,
    schema: schema_lib.NodeSchema,
    node_id_column: str,
    ignore_keys: tuple[str, ...],
) -> distributed_graph.Node:
  """Build a node from a tf example."""
  node_features = tf_graph_common.extract_features(
      example, schema.features, ignore_keys
  )
  node_id = tf_graph_common.tf_feature_to_bytes(example, node_id_column)
  return distributed_graph.Node(id=node_id, features=node_features)


def keyed_tf_example_to_node(
    keyed_example: Tuple[bytes, tf.train.Example],
    schema: schema_lib.NodeSchema,
    node_id_column: Optional[str],
    ignore_keys: tuple[str, ...],
) -> distributed_graph.Node:
  """Build a node from a tf example."""
  key, example = keyed_example
  node_features = tf_graph_common.extract_features(
      example, schema.features, ignore_keys
  )
  if node_id_column is not None:
    node_features[node_id_column] = np.array([key], dtype=np.bytes_)
  return distributed_graph.Node(id=key, features=node_features)


def tf_example_to_edge(
    example: tf.train.Example,
    edge_id_column: Optional[str],
    schema: schema_lib.EdgeSchema,
    ignore_keys: tuple[str, ...],
) -> distributed_graph.Edge:
  """Extracts edge adjacency from a tf example."""
  edge_target = tf_graph_common.tf_feature_to_bytes(
      example, tf_graph_common.KEY_TARGET
  )
  edge_source = tf_graph_common.tf_feature_to_bytes(
      example, tf_graph_common.KEY_SOURCE
  )

  if edge_id_column is not None and edge_id_column in example.features.feature:
    edge_id = tf_graph_common.tf_feature_to_bytes(example, edge_id_column)
  else:
    edge_id = None

  edge_features = tf_graph_common.extract_features(
      example,
      schema.features,
      ignore_keys,
  )

  return distributed_graph.Edge(
      source=edge_source,
      target=edge_target,
      id=edge_id,
      features=edge_features if edge_features else None,
  )


def node_to_tf_example(
    node: distributed_graph.Node,
    node_id_column: Optional[str],
    nodeset_schema: schema_lib.NodeSchema,
) -> tf.train.Example:
  """Converts node features to a tf example."""

  features = {}
  if node.features is not None:
    for k, v in node.features.items():
      if isinstance(v, (list, tuple)):
        features[k] = np.array([v])
      elif isinstance(v, np.ndarray):
        features[k] = np.array([v])
      else:
        features[k] = np.array([v])

  if node_id_column is not None:
    features[tf_graph_common.DEFAULT_KEY_ID] = np.array([node.id])

  return tf_graph_common.in_memory_node_to_tf_example(
      0, nodeset_schema.features, features, node_id_column
  )


def edge_to_tf_example(
    edge: distributed_graph.Edge,
    edge_id_column: Optional[str],
    edge_schema: schema_lib.EdgeSchema,
    source_format: schema_lib.FeatureFormat,
    target_format: schema_lib.FeatureFormat,
) -> tf.train.Example:
  """Converts edge adjacency to a tf example."""
  features = {}
  if edge.features is not None:
    for k, v in edge.features.items():
      if k in (tf_graph_common.KEY_SOURCE, tf_graph_common.KEY_TARGET):
        continue
      if isinstance(v, (list, tuple)):
        features[k] = np.array([v])
      elif isinstance(v, np.ndarray):
        features[k] = np.array([v])
      else:
        features[k] = np.array([v])
  # Do not pass id via 'features' if it's not in schema, instead set it directly.
  example = tf_graph_common.in_memory_edge_to_tf_example(
      0,
      edge_schema.features,
      edge.source,
      source_format,
      edge.target,
      target_format,
      features,
      edge_id_column,
  )
  if (
      edge.id is not None
      and edge_id_column is not None
      and edge_id_column not in example.features.feature
  ):
    # use raw tf assignment since edge.id is bytes
    if isinstance(edge.id, bytes):
      example.features.feature[edge_id_column].bytes_list.value.extend(
          [edge.id]
      )
    elif isinstance(edge.id, (int, np.integer)):
      example.features.feature[edge_id_column].int64_list.value.extend(
          [edge.id]
      )
    elif isinstance(edge.id, (str,)):
      example.features.feature[edge_id_column].bytes_list.value.extend(
          [edge.id.encode("utf-8")]
      )
  return example
