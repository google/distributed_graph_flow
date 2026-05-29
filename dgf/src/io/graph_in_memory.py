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

"""Reads a GF graph from a directory to an in-memory graph.

This module provides functionality to load a graph stored in the Google Format
(GF) from a specified directory into an InMemoryGraph object.
It handles sharded Parquet files for node and edge sets and converts node IDs
to 0-based indices in the edge set adjacencies.
"""

from collections.abc import Sequence
import os
import time
from typing import Dict, Optional, Tuple
from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import gf_metadata as gf_metadata_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_constants
from dgf.src.io import io_ext
from dgf.src.io import parquet as parquet_lib
from dgf.src.io import schema as schema_io_lib
from dgf.src.transform import schema as schema_filter_lib
from dgf.src.util import filesystem
from dgf.src.util import log
from dgf.src.util import shard as shard_lib
from dgf.src.util import util as util_lib
import numpy as np

FILENAME_SCHEMA = graph_constants.FILENAME_SCHEMA
FILENAME_METADATA = graph_constants.FILENAME_METADATA
FILENAME_NODE_FEATURE = graph_constants.FILENAME_NODE_FEATURE
FILENAME_EDGES_ADJACENCIES = graph_constants.FILENAME_EDGES_ADJACENCIES
PARQUET_EXTENSION = graph_constants.PARQUET_EXTENSION
KEY_SOURCE = graph_constants.KEY_SOURCE
KEY_TARGET = graph_constants.KEY_TARGET
MAX_SUPPORTED_GF_VERSION = graph_constants.MAX_SUPPORTED_GF_VERSION


IMPLICIT_EDGE_FEATURES = {KEY_SOURCE, KEY_TARGET}

# Highest GF version number supported


def _read_sharded_parquet(
    base_path: str, name: str, columns: Sequence[str], verbose: bool
) -> Tuple[Dict[str, np.ndarray], int]:
  """Reads sharded Parquet files for a given node or edge set."""
  sharded_files = shard_lib.list_paths(
      base_path, PARQUET_EXTENSION, allow_bq_fallback=True
  )
  if not sharded_files:
    raise ValueError(
        "No files found matching pattern"
        f" {shard_lib.shard_pattern_to_glob(base_path, PARQUET_EXTENSION)} or "
        f"{base_path}-*{PARQUET_EXTENSION}"
    )

  try:
    return parquet_lib.read_parquet_to_numpy_dict(
        paths=sharded_files, columns=columns, verbose=verbose
    )
  except Exception as e:
    raise ValueError(
        f"Failed to read sharded Parquet file at {base_path} for '{name}'"
    ) from e


def _read_node_set(
    path: str,
    nodeset_name: str,
    nodeset_def: schema_lib.NodeSchema,
    verbose: bool,
) -> tuple[dict[str, np.ndarray], int, io_ext.ByteIdToIdxMapper]:
  """Reads a node set from a GF graph."""
  nodeset_path = os.path.join(path, FILENAME_NODE_FEATURE, nodeset_name)
  columns = list(nodeset_def.features.keys())
  primary_key = analyse_schema_lib.primary_feature(nodeset_name, nodeset_def)

  try:
    features, node_count = _read_sharded_parquet(
        nodeset_path, nodeset_name, columns, verbose=verbose
    )
  except ValueError as e:
    raise ValueError(
        f"Failed to read nodeset {nodeset_name} from {nodeset_path}"
    ) from e

  # Index the nodeset ids for mapping to 0-based indices in the edge set
  # adjacencies.
  node_raw_ids = features[primary_key]
  if node_raw_ids.ndim != 1:
    raise ValueError(
        f"Expected node raw IDs for nodeset '{nodeset_name}' to be a 1D array"
        f" but got shape: {node_raw_ids.shape}"
    )
  # TODO(gbm): Implement in C++.

  if node_raw_ids.dtype.kind == "S":
    mapper = io_ext.ByteIdToIdxMapper(node_raw_ids)
  else:
    # A slow version of "ByteIdToIdxMapper" for integer values.
    mapping = {id.item(): idx for idx, id in enumerate(node_raw_ids)}

    def mapper(ids: np.ndarray) -> Tuple[np.ndarray, int]:

      idxs = np.empty(shape=[ids.shape[0]], dtype=np.int64)
      missmatch = -1
      for i, id_value in enumerate(ids):
        value = mapping.get(id_value)
        if value is None:
          value = -1
          missmatch = i
        idxs[i] = value
      return idxs, missmatch

  return features, node_count, mapper


def _read_edge_set(
    path: str,
    edgeset_name: str,
    edgeset_def: schema_lib.EdgeSchema,
    nodeset_mapping: dict[str, io_ext.ByteIdToIdxMapper],
    remove_dangling_edges: bool,
    verbose: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """Reads an edge set from a GF graph."""
  edge_set_path = os.path.join(path, FILENAME_EDGES_ADJACENCIES, edgeset_name)
  columns = [
      KEY_SOURCE,
      KEY_TARGET,
      *edgeset_def.features.keys(),
  ]
  try:
    features, _ = _read_sharded_parquet(
        edge_set_path, edgeset_name, columns, verbose=verbose
    )
  except ValueError as e:
    raise ValueError(
        f"Failed to read edgeset {edgeset_name} from {edge_set_path}"
    ) from e

  # TODO: b/454335246 - Avoid creating a python dict + copying the data to
  # numpy by doing the same computation faster directly in parquet.
  source_ids = features[KEY_SOURCE]
  target_ids = features[KEY_TARGET]
  source_nodeset_name = edgeset_def.source
  target_nodeset_name = edgeset_def.target
  source_mapper = nodeset_mapping[source_nodeset_name]
  target_mapper = nodeset_mapping[target_nodeset_name]

  if isinstance(source_mapper, io_ext.ByteIdToIdxMapper) and isinstance(
      target_mapper, io_ext.ByteIdToIdxMapper
  ):
    # Efficient path
    adjacency, missmatch_src, missmatch_trg = io_ext.PairMapping(
        source_mapper,
        target_mapper,
        source_ids,
        target_ids,
        min(32, os.cpu_count()),
    )
  else:
    # Slow path
    source_idxs, missmatch_src = source_mapper(source_ids)
    target_idxs, missmatch_trg = target_mapper(target_ids)
    adjacency = np.stack([source_idxs, target_idxs])

  if remove_dangling_edges:
    if missmatch_src != -1 or missmatch_trg != -1:
      valid_indices = (adjacency[0] != -1) & (adjacency[1] != -1)
      adjacency = adjacency[:, valid_indices]
      # Also filter features if they exist.
      for key in features:
        features[key] = features[key][valid_indices]
  else:
    if missmatch_src != -1:
      bad_id = source_ids[missmatch_src].item().decode()
      raise ValueError(
          f"Node ID {bad_id!r} not found in nodeset {source_nodeset_name!r}."
      )
    if missmatch_trg != -1:
      bad_id = target_ids[missmatch_trg].item().decode()
      raise ValueError(
          f"Node ID {bad_id!r} not found in nodeset {target_nodeset_name!r}."
      )

  if KEY_SOURCE not in edgeset_def.features:
    del features[KEY_SOURCE]
  if KEY_TARGET not in edgeset_def.features:
    del features[KEY_TARGET]
  return adjacency, features


def read_graph(
    path: str,
    *,
    override_schema: schema_lib.GraphSchema | None = None,
    verbose: bool = False,
    remove_dangling_edges: bool = False,
    schema_filter: Optional[schema_lib.GraphSchemaFilter] = None,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a GF graph from a directory to an in-memory graph.

  Usage example:

  ```python
  graph, schema = gdf.io.read_graph("/tmp/my_hgraph")
  print(f"Loaded graph with schema: {schema}")
  print(f"Number of nodes in 'n1': {graph.node_sets['n1'].num_nodes}")
  ```

  See the "File formats" documentation page for details about this file format,
  and how you can create GF Graphs manually.

  Args:
    path: Path to the GF Graph directory.
    override_schema: If specified, overrides the schema. This can be used to
      load only a subset of nodesets/edgesets/features.
    verbose: If True, print progress information to stdout.
    remove_dangling_edges: If False (default), fails if an edge is dangling
      (i.e., it refers to non-existing nodes). If True, dangling edges are
      removed.
    schema_filter: Optional filter to apply to the schema before loading. Only
      matching nodesets, edgesets, and features will be loaded.

  Returns:
    An in-memory heterogeneous graph.
  """

  start_time = time.monotonic()

  # Schema
  if override_schema is None:
    if verbose:
      log.info("Reading schema from %s", path)
    schema = schema_io_lib.read_schema(os.path.join(path, FILENAME_SCHEMA))
  else:
    if verbose:
      log.info("Using override schema")
    schema = override_schema

  if schema_filter:
    schema = schema_filter_lib.filter_schema(schema, schema_filter)

  analyse_schema_lib.fix_schema(schema)

  if verbose:
    num_features = sum(
        len(ns.features) for ns in schema.node_sets.values()
    ) + sum(len(es.features) for es in schema.edge_sets.values())
    log.info(
        "Reading %d nodeset(s), %d edgeset(s), and %d feature(s)",
        len(schema.node_sets),
        len(schema.edge_sets),
        num_features,
    )

  # Metadata
  with filesystem.open_read(os.path.join(path, FILENAME_METADATA)) as f:
    if verbose:
      log.info("Reading metadata from %s", path)
    metadata = gf_metadata_lib.GFGraphMetadata.from_json(f.read())

  if metadata.version > MAX_SUPPORTED_GF_VERSION:
    raise NotImplementedError(
        f"Unsupported metadata version: {metadata.version}. Only versions <="
        f" {MAX_SUPPORTED_GF_VERSION} are supported."
    )

  # Node sets
  node_sets = {}

  # Maps each node set name to a vectorized function that converts raw byte
  # IDs to integer indices.
  nodeset_mapping = {}

  for nodeset_name, nodeset_def in schema.node_sets.items():
    if verbose:
      log.info("Reading nodeset %s from %s", nodeset_name, path)
    features, num_nodes, mapper = _read_node_set(
        path, nodeset_name, nodeset_def, verbose
    )
    node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
        features=features,
        num_nodes=num_nodes,
    )
    nodeset_mapping[nodeset_name] = mapper

  # Edge sets
  edge_sets = {}
  for edgeset_name, edgeset_def in schema.edge_sets.items():
    if verbose:
      log.info("Reading edgeset %s from %s", edgeset_name, path)
    adjacency, features = _read_edge_set(
        path,
        edgeset_name,
        edgeset_def,
        nodeset_mapping,
        remove_dangling_edges=remove_dangling_edges,
        verbose=verbose,
    )
    edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=adjacency,
        features=features,
    )

  end_time = time.monotonic()
  log.info(
      "Graph read in memory in %s",
      util_lib.format_duration(end_time - start_time),
  )

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets=node_sets, edge_sets=edge_sets
  )
  return graph, schema


def write_graph(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    path: str,
    verbose: bool = False,
    max_num_shards: Optional[int] = None,
    compression: str = "snappy",
):
  """Writes an in-memory graph and schema to a GF Graph directory.

  This function is the inverse of `read_graph`, writing the provided
  `InMemoryGraph` and `GraphSchema` to disk in the Google Format (GF).
  The raw node IDs used in the edge adjacencies are taken from the `KEY_ID`
  feature within each node set of the `InMemoryGraph`.

  Args:
    graph: The in-memory graph to write.
    schema: The schema of the graph.
    path: The path to the GF Graph directory.
    verbose: If True, print progress information.
    max_num_shards: If provided, limits the maximum number of shards used when
      writing the Parquet files for each node and edge set.
  """
  start_time = time.monotonic()

  filesystem.makedirs(path)

  # Write Schema
  schema_path = os.path.join(path, FILENAME_SCHEMA)
  if verbose:
    log.info("Writing schema to %s", schema_path)
  schema_io_lib.write_schema(schema, schema_path)

  # Write Metadata
  metadata = gf_metadata_lib.GFGraphMetadata(version=MAX_SUPPORTED_GF_VERSION)
  metadata_path = os.path.join(path, FILENAME_METADATA)
  if verbose:
    log.info("Writing metadata to %s", metadata_path)
  with filesystem.open_write(metadata_path) as f:
    f.write(metadata.to_json(indent=2))

  # Write Node Sets
  node_dir = os.path.join(path, FILENAME_NODE_FEATURE)
  filesystem.makedirs(node_dir)
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    node_set = graph.node_sets[nodeset_name]

    # Check the existance of a primary key.
    _ = analyse_schema_lib.primary_feature(nodeset_name, nodeset_schema)

    if verbose:
      log.info("Writing nodeset %s to %s", nodeset_name, node_dir)

    num_shards, _ = shard_lib.estimate_num_node_shards(node_set.num_nodes)
    if max_num_shards is not None:
      num_shards = min(num_shards, max_num_shards)

    features_to_write = node_set.features
    if features_to_write is None:
      raise ValueError(f"Node set {nodeset_name} has no features to write.")

    parquet_lib.write_numpy_dict_to_parquet(
        features_to_write,
        nodeset_name,
        node_dir,
        nodeset_schema.features,
        num_shards,
        verbose,
        compression=compression,
    )

  # Write Edge Sets
  edge_dir = os.path.join(path, FILENAME_EDGES_ADJACENCIES)
  filesystem.makedirs(edge_dir)
  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    edge_set = graph.edge_sets[edgeset_name]
    if verbose:
      log.info("Writing edgeset %s to %s", edgeset_name, edge_dir)

    source_primary_key = analyse_schema_lib.primary_feature(
        edgeset_schema.source, schema.node_sets[edgeset_schema.source]
    )
    target_primary_key = analyse_schema_lib.primary_feature(
        edgeset_schema.target, schema.node_sets[edgeset_schema.target]
    )

    source_ids = graph.node_sets[edgeset_schema.source].features[
        source_primary_key
    ]
    target_ids = graph.node_sets[edgeset_schema.target].features[
        target_primary_key
    ]

    source_idxs = edge_set.adjacency[0]
    target_idxs = edge_set.adjacency[1]

    features_to_write = edge_set.features.copy() if edge_set.features else {}
    features_to_write[KEY_SOURCE] = source_ids[source_idxs]
    features_to_write[KEY_TARGET] = target_ids[target_idxs]

    features_schema = {**edgeset_schema.features}
    features_schema[KEY_SOURCE] = schema.node_sets[
        edgeset_schema.source
    ].features[source_primary_key]
    features_schema[KEY_TARGET] = schema.node_sets[
        edgeset_schema.target
    ].features[target_primary_key]

    num_shards, _ = shard_lib.estimate_num_node_shards(edge_set.num_edges())
    if max_num_shards is not None:
      num_shards = min(num_shards, max_num_shards)

    parquet_lib.write_numpy_dict_to_parquet(
        features_to_write,
        edgeset_name,
        edge_dir,
        features_schema,
        num_shards,
        verbose,
        compression=compression,
    )

  end_time = time.monotonic()
  log.info(
      "Graph written from memory in %s",
      util_lib.format_duration(end_time - start_time),
  )
