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

from __future__ import annotations

import logging
import os
from typing import Iterator, Optional, Tuple
from typing import TYPE_CHECKING

from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import beam_tf_graph_common
from dgf.src.io import graph_in_memory
from dgf.src.io import hgraph_in_memory
from dgf.src.io import tf_graph_common
from dgf.src.util import filesystem
from dgf.src.util import proto as proto_lib
from dgf.src.util import shard as shard_lib
from dgf.src.util.weak_dep.weak_dep_apache_beam import PTransform, beam
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
from dgf.src.util.weak_dep.weak_dep_tensorflow_gnn import tf_gnn_proto
import numpy as np


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

  Args:
    pbegin: Beam pbegin.
    path: The path to the HGraph directory.
    container_type: The type of container for the HGraph data.
    node_id_column: Column name containing the node id. If using a sstable (or
      another format with native key) and `node_id_column=None`, use the native
      key. If using a tfrecord (or another format without native key) and
      `node_id_column=None`, node_id_column defaults to '#id'. If
      `node_id_column` is set, use this column as id. This column is not
      necessarily a feature defined in the graph schema.
    edge_id_column: Column name containing the edge id. If None, the edges have
      no ID. Edge IDs are necessary for edge features.
    override_schema: Schema of the HGraph. If not provided, the schema is
      inferred from the TF GNN schema contained in the HGraph. Specifying the
      format allows to only include a subset of features / nodesets / edgesets.

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


class ReadFromHGraph(PTransform):
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
      tfgnn_schema = proto_lib.read_text_proto(
          os.path.join(self.path, hgraph_in_memory.PATH_GRAPH_SCHEMA),
          tf_gnn_proto.GraphSchema,
      )
      schema = hgraph_in_memory.tfgnn_schema_to_schema(tfgnn_schema)
    else:
      schema = self.schema

    extension = graph_in_memory.get_extension(self.container_type)

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
              schema=edgeset_def,
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


class ReadNodeSet(PTransform):
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
      node_id_column = (
          self.node_id_column
          if self.node_id_column is not None
          else hgraph_in_memory.DEFAULT_KEY_ID
      )
      return (
          pbegin
          | beam_tf_graph_common.ReadTfExampleContainer(
              file_pattern=self.file_pattern,
              container_type=tf_graph_common.TfExampleContainer.TF_RECORD,
          )
          | "Build nodes"
          >> beam.Map(
              beam_tf_graph_common.nonkeyed_tf_example_to_node,
              schema=self.schema,
              node_id_column=node_id_column,
              ignore_keys=(node_id_column,) if node_id_column else (),
          )
      )

    else:
      raise ValueError(f"Unsupported container type: {self.container_type}")


class ReadEdgeSet(PTransform):
  """Reads flat edge sets from various formats."""

  def __init__(
      self,
      file_pattern: str,
      container_type: hgraph_in_memory.HGraphContainerType,
      edge_id_column: Optional[str],
      schema: schema_lib.EdgeSchema,
  ):
    self.file_pattern = file_pattern
    self.container_type = container_type
    self.edge_id_column = edge_id_column
    self.schema = schema

  def expand(
      self, pbegin: beam.pvalue.PBegin
  ) -> beam.PCollection[distributed_graph.Edge]:

    if self.edge_id_column is None:
      edge_id_column = hgraph_in_memory.DEFAULT_KEY_ID
    else:
      edge_id_column = self.edge_id_column

    if self.container_type in (
        hgraph_in_memory.HGraphContainerType.TF_RECORD,
    ):
      return (
          pbegin
          | beam_tf_graph_common.ReadTfExampleContainer(
              file_pattern=self.file_pattern,
              container_type=tf_graph_common.TfExampleContainer[
                  self.container_type.name
              ],
          )
          | "Import edgeset"
          >> beam.Map(
              beam_tf_graph_common.tf_example_to_edge,
              edge_id_column=edge_id_column,
              schema=self.schema,
              ignore_keys=(
                  tf_graph_common.KEY_SOURCE,
                  tf_graph_common.KEY_TARGET,
                  edge_id_column,
              )
              if edge_id_column
              else (
                  tf_graph_common.KEY_SOURCE,
                  tf_graph_common.KEY_TARGET,
              ),
          )
      )
    else:
      raise ValueError(f"Unsupported container type: {self.container_type}")


def write_graphai_hgraph(
    graph: distributed_graph.Graph,
    path: str,
    container_type: (
        hgraph_in_memory.HGraphContainerType | str
    ) = hgraph_in_memory.HGraphContainerType.TF_RECORD,
    node_id_column: Optional[str] = None,
    edge_id_column: Optional[str] = None,
):
  """Writes a distributed HGraph using Beam.

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

  extension = graph_in_memory.get_extension(container_type)

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
            beam_tf_graph_common.node_to_tf_example,
            node_id_column=node_id_column,
            nodeset_schema=nodeset_schema,
        )
        | f"Write nodeset {nodeset_name}"
        >> beam_tf_graph_common.WriteTfExampleContainer(
            file_path_prefix=os.path.join(
                path, hgraph_in_memory.PATH_NODE_FEATURE, nodeset_name
            ),
            extension=extension,
            container_type=tf_graph_common.TfExampleContainer[
                container_type.name
            ],
            num_shards=0,
        )
    )

  # Write the edge adjacency
  for edgeset_name, edgeset_schema in graph.schema.edge_sets.items():
    edgeset = graph.edge_sets[edgeset_name]
    source_format = (
        graph.schema.node_sets[edgeset_schema.source]
        .features[hgraph_in_memory.DEFAULT_KEY_ID]
        .format
    )
    target_format = (
        graph.schema.node_sets[edgeset_schema.target]
        .features[hgraph_in_memory.DEFAULT_KEY_ID]
        .format
    )
    _ = (
        edgeset
        | f"Export edgeset {edgeset_name}"
        >> beam.Map(
            beam_tf_graph_common.edge_to_tf_example,
            edge_id_column=edge_id_column,
            edge_schema=edgeset_schema,
            source_format=source_format,
            target_format=target_format,
        )
        | f"Write edgeset {edgeset_name}"
        >> beam_tf_graph_common.WriteTfExampleContainer(
            file_path_prefix=os.path.join(
                path, hgraph_in_memory.PATH_EDGES, edgeset_name
            ),
            extension=extension,
            container_type=tf_graph_common.TfExampleContainer[
                container_type.name
            ],
            num_shards=0,
        )
    )
