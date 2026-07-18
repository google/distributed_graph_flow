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

"""Reader and writer for GF Graphs to/from Beam distributed graphs."""

import functools
import os
from typing import Any, Dict, List, Optional
import apache_beam as beam
from apache_beam.io import parquetio
from dgf.src.analyse import schema as schema_analyse_lib
from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.data import gf_metadata as gf_metadata_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import graph_constants
from dgf.src.io import schema as schema_io_lib
from dgf.src.transform import schema as schema_filter_lib
from dgf.src.util import filesystem
from dgf.src.util import shard as shard_lib
import numpy as np
import pyarrow

FILENAME_SCHEMA = graph_constants.FILENAME_SCHEMA
FILENAME_METADATA = graph_constants.FILENAME_METADATA
FILENAME_NODE_FEATURE = graph_constants.FILENAME_NODE_FEATURE
FILENAME_EDGES_ADJACENCIES = graph_constants.FILENAME_EDGES_ADJACENCIES
PARQUET_EXTENSION = graph_constants.PARQUET_EXTENSION
KEY_SOURCE = graph_constants.KEY_SOURCE
KEY_TARGET = graph_constants.KEY_TARGET
MAX_SUPPORTED_GF_VERSION = graph_constants.MAX_SUPPORTED_GF_VERSION


FEATURE_FORMAT_TO_PY_ARROW_DTYPE: Dict[schema_lib.FeatureFormat, Any] = {
    schema_lib.FeatureFormat.INTEGER_64: pyarrow.int64(),
    schema_lib.FeatureFormat.INTEGER_32: pyarrow.int32(),
    schema_lib.FeatureFormat.FLOAT_32: pyarrow.float32(),
    schema_lib.FeatureFormat.FLOAT_64: pyarrow.float64(),
    schema_lib.FeatureFormat.BYTES: pyarrow.binary(),
    schema_lib.FeatureFormat.BOOL: pyarrow.bool_(),
}


def read_graph(
    pbegin: beam.Pipeline,
    path: str,
    *,
    override_schema: Optional[schema_lib.GraphSchema] = None,
    schema_filter: Optional[schema_lib.GraphSchemaFilter] = None,
    beam_namespace: str = "",
) -> distributed_graph_lib.Graph:
  """Reads a GF graph into a distributed graph.

  Usage example:

  ```python
  with beam.Pipeline() as pbegin:
    graph = dgf.beam.io.read_graph(pbegin,
    "/tmp/my_gf_graph")
    # Further process the graph...
  ```

  The GF Graph format is an efficient format to store both small and large
  graphs, both for in-process and distributed computation. See the "File
  formats" documentation page for details.

  Use the :convert_hgraph_to_gf_graph CLI to convert GraphAI HGraphs into GF
  Graphs.

  Args:
    pbegin: The Beam pipeline root.
    path: Path to GF graph directory.
    override_schema: If specified, overrides the schema. This can be used to
      load only a subset of nodesets/edgesets/features.
    schema_filter: Optional filter to apply to the schema before loading. Only
      matching nodesets, edgesets, and features will be loaded.
    beam_namespace: Optional namespace to prepend to Beam stage names.

  Returns:
    A distributed graph.
  """

  # Schema
  if override_schema is not None:
    schema = override_schema
  else:
    schema = schema_io_lib.read_schema(os.path.join(path, FILENAME_SCHEMA))

  if schema_filter:
    schema = schema_filter_lib.filter_schema(schema, schema_filter)

  schema_analyse_lib.fix_schema(schema)

  # Metadata
  with filesystem.open_read(os.path.join(path, FILENAME_METADATA)) as f:
    metadata = gf_metadata_lib.GFGraphMetadata.from_json(f.read())  # pyrefly: ignore[missing-attribute]

  if metadata.version > MAX_SUPPORTED_GF_VERSION:
    raise NotImplementedError(
        f"Unsupported metadata version: {metadata.version}. Only versions <="
        f" {MAX_SUPPORTED_GF_VERSION} are supported."
    )

  node_sets = {}
  for nodeset_name, nodeset_def in schema.node_sets.items():
    file_pattern = shard_lib.shard_pattern_to_glob(
        os.path.join(path, FILENAME_NODE_FEATURE, nodeset_name),
        PARQUET_EXTENSION,
    )

    node_sets[nodeset_name] = (
        pbegin
        | f"{beam_namespace}Read nodeset {nodeset_name!r}"
        >> read_node_set_features(
            file_pattern=file_pattern,
            schema=nodeset_def,
            nodeset_name=nodeset_name,
        )
    )

  edge_sets = {}
  for edgeset_name, edgeset_def in schema.edge_sets.items():
    file_pattern = shard_lib.shard_pattern_to_glob(
        os.path.join(path, FILENAME_EDGES_ADJACENCIES, edgeset_name),
        PARQUET_EXTENSION,
    )
    edge_sets[edgeset_name] = (
        pbegin
        | f"{beam_namespace}Read edgeset {edgeset_name!r}"
        >> read_edge_set_features(
            file_pattern=file_pattern,
            schema=edgeset_def,
            edgeset_name=edgeset_name,
        )
    )

  return distributed_graph_lib.Graph(
      schema=schema,
      node_sets=node_sets,
      edge_sets=edge_sets,
  )


def _raw_to_node(
    row: Dict[str, Any], schema: schema_lib.NodeSchema, primary_key: str
) -> distributed_graph_lib.Node:
  node_features = {}
  node_id = None
  for feature_name, feature_schema in schema.features.items():
    raw_value = row[feature_name]
    if feature_name == primary_key:
      node_id = raw_value

    node_features[feature_name] = np.array(
        raw_value,
        dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
            feature_schema.format
        ],
    )

  if node_id is None:
    raise ValueError(
        f"Node ID ({primary_key!r}) not found in row={row} with"
        f" schema={schema.features}"
    )
  return distributed_graph_lib.Node(id=node_id, features=node_features)


@beam.ptransform_fn
def read_node_set_features(
    pbegin: beam.pvalue.PBegin,
    file_pattern: str,
    schema: schema_lib.NodeSchema,
    nodeset_name: str,
) -> beam.PCollection[distributed_graph_lib.Node]:
  """Reads Nodes from a Parquet file."""
  primary_key = schema_analyse_lib.primary_feature(nodeset_name, schema)
  return (
      pbegin
      | "Read Parquet file" >> parquetio.ReadFromParquet(file_pattern)
      | "Convert to Node"
      >> beam.Map(
          functools.partial(
              _raw_to_node, schema=schema, primary_key=primary_key
          )
      )
  )


def _raw_to_edge(
    row: Dict[str, Any],
    schema: schema_lib.EdgeSchema,
    primary_key: Optional[str],
) -> distributed_graph_lib.Edge:
  edge_features = {}
  edge_id = None
  for feature_name, feature_schema in schema.features.items():
    raw_value = row[feature_name]
    if primary_key is not None and feature_name == primary_key:
      edge_id = raw_value

    if feature_schema.shape and not isinstance(raw_value, list):
      raw_value = [raw_value]
    edge_features[feature_name] = np.array(
        raw_value,
        dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
            feature_schema.format
        ],
    )

  return distributed_graph_lib.Edge(
      source=row[KEY_SOURCE],
      target=row[KEY_TARGET],
      id=edge_id,
      features=edge_features if edge_features else None,
  )


@beam.ptransform_fn
def read_edge_set_features(
    pbegin: beam.pvalue.PBegin,
    file_pattern: str,
    schema: schema_lib.EdgeSchema,
    edgeset_name: str,
) -> beam.PCollection[distributed_graph_lib.Edge]:
  """Reads Edges from a Parquet file."""
  primary_key = schema_analyse_lib.primary_feature_or_none(edgeset_name, schema)
  return (
      pbegin
      | "Read Parquet file" >> parquetio.ReadFromParquet(file_pattern)
      | "Convert to Edge"
      >> beam.Map(
          functools.partial(
              _raw_to_edge, schema=schema, primary_key=primary_key
          )
      )
  )


def write_graph(
    graph: distributed_graph_lib.Graph,
    path: str,
    beam_namespace: str = "",
    num_node_shards: int = 0,
    num_edge_shards: int = 0,
    compression: str = "snappy",
) -> beam.pvalue.PDone:
  """Writes a GF Graph from a distributed graph (beam).

  Usage example:

  ```python
  with beam.Pipeline() as root:
    graph = read_graph(root, "/tmp/my_gf_graph")
    write_graph(graph, "/tmp/my_new_gf_graph")
    # Further process the graph...
  ```

  The GF Graph format is an efficient format to store both small and large
  graphs, both for in-process and distributed computation. See the "File
  formats" documentation page for details.

  Args:
    graph: Graph to write.
    path: Path to the output GF graph directory.
    beam_namespace: Optional namespace to prepend to Beam stage names.

  Returns:
    A PCollection of type `beam.pvalue.PDone` that represents the completion
    of all write operations.
  """
  filesystem.makedirs(path)

  # Write Schema
  schema_io_lib.write_schema(graph.schema, os.path.join(path, FILENAME_SCHEMA))

  # Write Metadata
  metadata = gf_metadata_lib.GFGraphMetadata(version=MAX_SUPPORTED_GF_VERSION)
  metadata_path = os.path.join(path, FILENAME_METADATA)
  with filesystem.open_write(metadata_path) as f:
    f.write(metadata.to_json(indent=2))  # pyrefly: ignore[missing-attribute]

  write_results = []

  # Write Node Sets
  nodeset_dir = os.path.join(path, FILENAME_NODE_FEATURE)
  filesystem.makedirs(nodeset_dir)
  for nodeset_name, nodeset_schema in graph.schema.node_sets.items():
    pnode_collection = graph.node_sets[nodeset_name]
    file_path_prefix = os.path.join(nodeset_dir, nodeset_name)
    write_result = (
        pnode_collection
        | f"{beam_namespace}NodeToRaw_{nodeset_name}"
        >> beam.Map(_node_to_raw, schema=nodeset_schema)
        | f"{beam_namespace}WriteNodeset_{nodeset_name}"
        >> parquetio.WriteToParquet(
            file_path_prefix=file_path_prefix,
            file_name_suffix=PARQUET_EXTENSION,
            schema=_node_schema_to_parquet_schema(nodeset_schema),
            codec=compression,
            num_shards=num_node_shards,
        )
    )
    write_results.append(write_result)

  # Write Edge Sets
  edgeset_dir = os.path.join(path, FILENAME_EDGES_ADJACENCIES)
  filesystem.makedirs(edgeset_dir)
  for edgeset_name, edgeset_schema in graph.schema.edge_sets.items():
    pedge_collection = graph.edge_sets[edgeset_name]
    file_path_prefix = os.path.join(edgeset_dir, edgeset_name)
    write_result = (
        pedge_collection
        | f"{beam_namespace}EdgeToRaw_{edgeset_name}"
        >> beam.Map(_edge_to_raw, schema=edgeset_schema)
        | f"{beam_namespace}WriteEdgeset_{edgeset_name}"
        >> parquetio.WriteToParquet(
            file_path_prefix=file_path_prefix,
            file_name_suffix=PARQUET_EXTENSION,
            schema=_edge_schema_to_parquet_schema(edgeset_schema, graph.schema),
            codec=compression,
            num_shards=num_edge_shards,
        )
    )
    write_results.append(write_result)

  return (
      write_results | f"{beam_namespace}FlattenWriteResults" >> beam.Flatten()
  )


def _feature_schema_to_parquet_fields(
    feature_schema: schema_lib.FeatureSchema,
) -> List[pyarrow.Field]:
  """Creates the schema for the parquet node container."""
  fields = []
  # Note: The schema has the node "#id".
  for feature_name, feature_schema in feature_schema.items():  # pyrefly: ignore[missing-attribute]
    pa_type = FEATURE_FORMAT_TO_PY_ARROW_DTYPE[feature_schema.format]
    shape = feature_schema.shape
    if shape is None:
      shape = []
    for s in reversed(shape):
      if s is None:
        pa_type = pyarrow.list_(pa_type)
      else:
        pa_type = pyarrow.list_(pa_type, list_size=s)
    fields.append(pyarrow.field(feature_name, pa_type))
  return fields


def _node_schema_to_parquet_schema(
    node_schema: schema_lib.NodeSchema,
) -> pyarrow.Schema:
  """Creates the schema for the parquet node container."""
  return pyarrow.schema(_feature_schema_to_parquet_fields(node_schema.features))  # pyrefly: ignore[bad-argument-type]


def _edge_schema_to_parquet_schema(
    edge_schema: schema_lib.EdgeSchema,
    schema: schema_lib.GraphSchema,
) -> pyarrow.Schema:
  """Creates the schema for the parquet edge container."""
  source_nodeset_primary_key = schema_analyse_lib.primary_feature(
      edge_schema.source, schema.node_sets[edge_schema.source]
  )
  source_node_format = (
      schema.node_sets[edge_schema.source]
      .features[source_nodeset_primary_key]
      .format
  )

  target_nodeset_primary_key = schema_analyse_lib.primary_feature(
      edge_schema.target, schema.node_sets[edge_schema.target]
  )
  target_node_format = (
      schema.node_sets[edge_schema.target]
      .features[target_nodeset_primary_key]
      .format
  )

  fields = [
      pyarrow.field(
          KEY_SOURCE, FEATURE_FORMAT_TO_PY_ARROW_DTYPE[source_node_format]
      ),
      pyarrow.field(
          KEY_TARGET, FEATURE_FORMAT_TO_PY_ARROW_DTYPE[target_node_format]
      ),
  ] + _feature_schema_to_parquet_fields(edge_schema.features)  # pyrefly: ignore[bad-argument-type]
  return pyarrow.schema(fields)


def _node_to_raw(
    node: distributed_graph_lib.Node, schema: schema_lib.NodeSchema
) -> Dict[str, Any]:
  """Converts a Node to a raw dictionary for Parquet writing."""
  primary_key = schema_analyse_lib.primary_feature_or_none("", schema)
  raw_dict = {}
  for feature_name in schema.features:
    if feature_name == primary_key:
      raw_dict[feature_name] = node.id
    else:
      feature_values = node.features[feature_name]  # pyrefly: ignore[unsupported-operation]
      raw_dict[feature_name] = feature_values.tolist()
  return raw_dict


def _edge_to_raw(
    edge: distributed_graph_lib.Edge, schema: schema_lib.EdgeSchema
) -> Dict[str, Any]:
  """Converts an Edge to a raw dictionary for Parquet writing."""
  raw_dict = {
      KEY_SOURCE: edge.source,
      KEY_TARGET: edge.target,
  }
  for feature_name in schema.features:
    feature_values = edge.features[feature_name]  # pyrefly: ignore[unsupported-operation]
    raw_dict[feature_name] = feature_values.tolist()
  return raw_dict
