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

"""Utilities to read and write Avro files efficiently in memory."""

# TODO(liuyanchen): Add the Avrio IO to the IO benchmark.

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.util import filesystem
from dgf.src.util import shard as shard_lib
import fastavro
import numpy as np
import tqdm


parse_schema = fastavro.parse_schema
fastavro_reader = fastavro.reader
fastavro_writer = fastavro.writer
tqdm = tqdm.tqdm


def _get_avro_type_for_shape(
    shape: Tuple[Optional[int], ...], base_avro_type: str
) -> Any:
  """Recursively builds a nested Avro array type from a shape tuple."""
  if not shape:
    # Base case: empty shape () is a scalar
    return base_avro_type

  # Recursive step: build nested arrays
  inner_type = _get_avro_type_for_shape(shape[1:], base_avro_type)
  return {"type": "array", "items": inner_type}


def _get_avro_type_for_feature(feature_schema: schema_lib.FeatureSchema) -> Any:
  """Converts a FeatureSchema into a valid Avro type definition."""
  base_avro_type = feature_format_lib.FEATURE_FORMAT_TO_AVRO_DTYPE[
      feature_schema.format
  ]
  shape = feature_schema.shape
  if shape is None or not list(shape):
    # Scalar feature
    return base_avro_type

  # Array feature
  return _get_avro_type_for_shape(shape, base_avro_type)


def _serialize_numpy_value(value: Union[np.ndarray, np.generic, Any]) -> Any:
  """Converts a numpy scalar or array to a JSON-serializable Python type."""
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  # Fallback for non-numpy types (like bytes)
  return value


def _generate_node_records(
    feature_items: List[Tuple[str, Any]],
    start_index: int,
    end_index: int,
    name: str,
    verbose: bool,
) -> Any:
  """Generates records for writing node sets to Avro."""
  iterator = range(start_index, end_index)
  if verbose:
    iterator = tqdm(
        iterator,
        desc=f"  - Writing nodes for '{name}'",
        unit="node",
        total=end_index - start_index,
    )
  for i in iterator:
    yield {
        f_name: _serialize_numpy_value(f_array[i])
        for f_name, f_array in feature_items
    }


def _generate_edge_records(
    feature_items: List[Tuple[str, Any]],
    source_array: np.ndarray,
    target_array: np.ndarray,
    start_index: int,
    end_index: int,
    key_source: str,
    key_target: str,
    name: str,
    verbose: bool,
) -> Any:
  """Generates records for writing edge sets to Avro."""
  iterator = range(start_index, end_index)
  if verbose:
    iterator = tqdm(
        iterator,
        desc=f"  - Writing edges for '{name}'",
        unit="edge",
        total=end_index - start_index,
    )
  for i in iterator:
    record_to_write = {
        key_source: _serialize_numpy_value(source_array[i]),
        key_target: _serialize_numpy_value(target_array[i]),
    }
    record_to_write.update({
        f_name: _serialize_numpy_value(f_array[i])
        for f_name, f_array in feature_items
    })
    yield record_to_write


def write_avro_node_sets(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    directory: str,
    extension: str,
    verbose: bool,
):
  """Writes node sets to Avro files."""

  for nodeset_name, node_schema in schema.node_sets.items():
    nodeset = graph.node_sets[nodeset_name]
    avro_fields = []
    for feature_name, feature_schema in node_schema.features.items():
      avro_fields.append({
          "name": feature_name,
          "type": _get_avro_type_for_feature(feature_schema),
      })
    avro_schema_dict = {
        "type": "record",
        "name": f"{nodeset_name}_node",
        "fields": avro_fields,
    }
    parsed_schema = fastavro.parse_schema(avro_schema_dict)
    feature_items = list(nodeset.features.items())
    num_shards, num_nodes_per_shard = shard_lib.estimate_num_node_shards(
        nodeset.num_nodes
    )
    for shard_index in range(num_shards):
      filename = shard_lib.sharded_filename(
          filename=nodeset_name,
          shard=shard_index,
          num_shards=num_shards,
          extension=extension,
      )
      filepath = os.path.join(directory, filename)
      start_index = shard_index * num_nodes_per_shard
      end_index = min(
          (shard_index + 1) * num_nodes_per_shard, nodeset.num_nodes
      )
      with filesystem.open_write(filepath, binary=True) as f_out:
        fastavro.writer(
            f_out,
            parsed_schema,
            _generate_node_records(
                feature_items,
                start_index,
                end_index,
                nodeset_name,
                verbose,
            ),
        )


def write_avro_edge_sets(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    directory: str,
    extension: str,
    node_id_column: str,
    key_source: str,
    key_target: str,
    verbose: bool,
):
  """Writes edge sets to Avro files."""
  for edgeset_name, edge_schema in schema.edge_sets.items():
    edgeset = graph.edge_sets[edgeset_name]
    source_nodeset_name = edge_schema.source
    target_nodeset_name = edge_schema.target
    source_node_id_feat_schema = schema.node_sets[source_nodeset_name].features[
        node_id_column
    ]
    target_node_id_feat_schema = schema.node_sets[target_nodeset_name].features[
        node_id_column
    ]
    avro_fields = [
        {
            "name": key_source,
            "type": _get_avro_type_for_feature(source_node_id_feat_schema),
        },
        {
            "name": key_target,
            "type": _get_avro_type_for_feature(target_node_id_feat_schema),
        },
    ]
    if edge_schema.features:
      for feature_name, feature_schema in edge_schema.features.items():
        avro_fields.append({
            "name": feature_name,
            "type": _get_avro_type_for_feature(feature_schema),
        })
    avro_schema_dict = {
        "type": "record",
        "name": f"{edgeset_name}_edge",
        "fields": avro_fields,
    }
    parsed_schema = fastavro.parse_schema(avro_schema_dict)
    num_edges = edgeset.adjacency.shape[1]
    feature_items = list(edgeset.features.items()) if edgeset.features else []

    source_ids = graph.node_sets[source_nodeset_name].features[node_id_column][
        edgeset.adjacency[0]
    ]
    target_ids = graph.node_sets[target_nodeset_name].features[node_id_column][
        edgeset.adjacency[1]
    ]

    num_shards, num_edges_per_shard = shard_lib.estimate_num_edge_shards(
        num_edges
    )
    for shard_index in range(num_shards):
      filename = shard_lib.sharded_filename(
          filename=edgeset_name,
          shard=shard_index,
          num_shards=num_shards,
          extension=extension,
      )
      filepath = os.path.join(directory, filename)
      start_index = shard_index * num_edges_per_shard
      end_index = min((shard_index + 1) * num_edges_per_shard, num_edges)
      with filesystem.open_write(filepath, binary=True) as f_out:
        fastavro.writer(
            f_out,
            parsed_schema,
            _generate_edge_records(
                feature_items,
                source_ids,
                target_ids,
                start_index,
                end_index,
                key_source,
                key_target,
                edgeset_name,
                verbose,
            ),
        )


def read_avro_record(
    paths: List[str],
    columns: Dict[str, Tuple[str, Tuple[Optional[int], ...]]],
    verbose: bool,
) -> Tuple[Dict[str, np.ndarray], int]:
  """Reads an Avro file and updates the feature builders."""
  feature_builders: Dict[str, List[Any]] = {
      f_name: [] for f_name in columns.keys()
  }
  num_records = 0

  for avro_file in paths:
    with filesystem.open_read(avro_file, binary=True) as f_in:
      reader = fastavro_reader(f_in)
      record_iterator = reader
      if verbose:
        record_iterator = tqdm(
            reader,
            desc=f"  - Reading records from {avro_file}",
            unit="record",
        )
      for record in record_iterator:
        num_records += 1
        for feature_name in feature_builders.keys():
          feature_builders[feature_name].append(record[feature_name])

  # Convert lists to numpy arrays
  final_features = {}
  for feature_name, data_list in feature_builders.items():
    dtype = columns[feature_name][0]
    if not data_list:
      shape = (0,) + tuple(
          d for d in (columns[feature_name][1] or ()) if d is not None
      )
      final_features[feature_name] = np.empty(shape, dtype=dtype)
    else:
      final_features[feature_name] = np.array(data_list, dtype=dtype)
  return final_features, num_records
