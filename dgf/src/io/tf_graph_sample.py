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

"""Import TF GNN Graph Samples."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
import enum
import os
import typing
from typing import Any

from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import tf as io_tf_lib
from dgf.src.io import tf_graph_sample_ext
from dgf.src.util import shard as shard_lib
from dgf.src.util.weak_dep.weak_dep_apache_beam import PTransform, beam
from dgf.src.util.weak_dep.weak_dep_bagz import bag_io, bagz
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
import numpy as np


class TFGraphSampleContainerType(enum.Enum):
  TF_RECORD = 1
  SSTABLE = 2
  BAGZ = 3
  RECORDIO = 4


def tfgnn_graph_to_graph(
    example: tf.train.Example,
    schema: schema_lib.GraphSchema,
    import_node_ids: str | None = None,
    import_edge_ids: str | None = None,
) -> in_memory_graph.InMemoryGraph:
  """Converts a TF GNN Graph Sample to an InMemoryGraph."""
  feature_dict = {}
  for key, value in example.features.feature.items():
    if value.HasField("bytes_list"):
      feature_dict[key] = np.array(value.bytes_list.value)
    elif value.HasField("float_list"):
      feature_dict[key] = np.array(value.float_list.value)
    elif value.HasField("int64_list"):
      feature_dict[key] = np.array(value.int64_list.value)
  return graph_dict_to_graph(
      feature_dict, schema, import_node_ids, import_edge_ids
  )


def _check_at_most_one_ragged_dim(
    feature_key: str, shape: list[int | None]
) -> None:
  """Fails if the feature has more than one variable-length dimension.

  Nesting the arrays of objects that encode a ragged dimension would need a
  convention DGF does not define. The TensorFlow conversions use
  `tf.RaggedTensor` and have no such limit.

  Args:
    feature_key: The feature name, used in the error message.
    shape: The feature shape, excluding the item dimension.

  Raises:
    ValueError: If more than one dimension is variable-length.
  """
  num_ragged_dims = sum(1 for dim in shape if dim is None)
  if num_ragged_dims > 1:
    raise ValueError(
        f"The feature '{feature_key}' has {num_ragged_dims} variable-length"
        f" dimensions (shape {tuple(shape)}). The NumPy in-memory format"
        " supports at most one. Use the TensorFlow conversion"
        " (`serialized_tfgnn_graph_to_tf_graph` or"
        " `tfgnn_graph_dict_to_tf_graph`) for such a feature."
    )


def _split_ragged_dim(
    values: np.ndarray, row_lengths: np.ndarray
) -> np.ndarray:
  """Splits the outer-most dimension of `values` into rows.

  Args:
    values: An array (possibly an array of objects) of shape [total_rows, ...].
    row_lengths: The length of each of the returned rows. Sums to `total_rows`.

  Returns:
    An array of objects of length `len(row_lengths)`.
  """
  offsets = np.cumsum(row_lengths)
  if offsets.size and offsets[-1] != len(values):
    raise ValueError(
        f"The row lengths sum to {offsets[-1]} while the value contains"
        f" {len(values)} rows."
    )
  rows = np.split(values, offsets[:-1]) if offsets.size else []
  # Note: Make sure numpy does not merge the arrays.
  result = np.empty(len(rows), dtype=np.object_)
  result[:] = rows
  return result


def _group_static_dims(values: np.ndarray, dims: list[int]) -> np.ndarray:
  """Groups the outer-most dimension of `values` into the `dims` dimensions.

  For example, if `values` is of shape [12] and `dims` is [3], the result is of
  shape [4, 3].

  Args:
    values: An array (possibly an array of objects).
    dims: The static dimensions to extract from the outer-most dimension of
      `values`, from the outer-most to the inner-most one.

  Returns:
    The re-grouped array.
  """
  if not dims:
    return values
  if values.dtype == np.object_:
    # The values are ragged: group them without merging the sub-arrays.
    num_items = len(values) // int(np.prod(dims))
    result = np.empty([num_items] + list(dims), dtype=np.object_)
    result[:] = values.reshape([num_items] + list(dims))
    return result
  return values.reshape([-1] + list(dims))


def _tfgnn_feature_to_array(
    example: dict[str, np.ndarray],
    feature_key: str,
    feature_schema: schema_lib.FeatureSchema,
    num_items: int,
) -> np.ndarray:
  """Converts a TF GNN Graph Sample feature into a numpy array."""

  values = np.asarray(example[feature_key])
  target_dtype = feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
      feature_schema.format
  ]
  if values.dtype != target_dtype:
    values = values.astype(target_dtype)

  shape = list(feature_schema.shape) if feature_schema.shape else []
  if feature_schema.is_static_shape():
    static_shape = typing.cast(list[int], shape)
    return values.reshape([num_items] + static_shape)

  _check_at_most_one_ragged_dim(feature_key, shape)

  # Re-build the feature values, from the inner-most dimension to the outer-most
  # one. `pending_static_dims` contains the static dimensions that are not yet
  # applied on the accumulated `result`.
  result = values
  pending_static_dims: list[int] = []
  for dim_idx in range(len(shape), 0, -1):
    dim = shape[dim_idx - 1]
    if dim is None:
      result = _group_static_dims(result, pending_static_dims)
      pending_static_dims = []
      # Note: The 0-th dimension of a feature value is the item dimension.
      row_lengths = example[
          io_tf_lib.tfgnn_ragged_dim_key(feature_key, dim_idx)
      ]
      result = _split_ragged_dim(result, row_lengths)
    else:
      pending_static_dims.insert(0, dim)
  # Note: The remaining outer-most dimension is the item dimension.
  result = _group_static_dims(result, pending_static_dims)
  if len(result) != num_items:
    raise ValueError(
        f"The feature '{feature_key}' contains {len(result)} items while the"
        f" node/edge set contains {num_items} items."
    )
  return result


def graph_dict_to_graph(
    example: dict[str, np.ndarray],
    schema: schema_lib.GraphSchema,
    import_node_ids: str | None = None,
    import_edge_ids: str | None = None,
) -> in_memory_graph.InMemoryGraph:
  """Converts a TF GNN Graph Sample Dict to an InMemoryGraph.

  This function is the NumPy equivalent of
  `dgf.convert.tfgnn_graph_dict_to_tf_graph` (which only uses TensorFlow
  operations, and can therefore be exported in a TF SavedModel).

  Args:
    example: A TF GNN Graph Sample Dict i.e. a dictionary of flat numpy arrays.
    schema: The schema of the graph.
    import_node_ids: If set, name of the node feature containing the node ids.
    import_edge_ids: If set, name of the edge feature containing the edge ids.

  Returns:
    An `InMemoryGraph`.
  """
  node_sets = {}
  edge_sets = {}

  for node_set_name, node_set in schema.node_sets.items():
    node_features = {}
    num_nodes = example[
        io_tf_lib.tfgnn_node_key(node_set_name, io_tf_lib.TFGNN_SIZE_KEY)
    ][0].item()
    for feature_name, feature_schema in node_set.features.items():
      node_features[feature_name] = _tfgnn_feature_to_array(
          example,
          io_tf_lib.tfgnn_node_key(node_set_name, feature_name),
          feature_schema,
          num_nodes,
      )
    if import_node_ids:
      node_features[import_node_ids] = example[
          io_tf_lib.tfgnn_node_key(node_set_name, import_node_ids)
      ]
    node_sets[node_set_name] = in_memory_graph.InMemoryNodeSet(
        features=node_features,
        num_nodes=num_nodes,
    )

  for edge_set_name, edge_set in schema.edge_sets.items():
    adjacency = np.array([
        example[
            io_tf_lib.tfgnn_edge_key(edge_set_name, io_tf_lib.TFGNN_SOURCE_KEY)
        ],
        example[
            io_tf_lib.tfgnn_edge_key(edge_set_name, io_tf_lib.TFGNN_TARGET_KEY)
        ],
    ])
    edge_features = {}
    num_edges = example[
        io_tf_lib.tfgnn_edge_key(edge_set_name, io_tf_lib.TFGNN_SIZE_KEY)
    ][0].item()
    for feature_name, feature_schema in edge_set.features.items():
      edge_features[feature_name] = _tfgnn_feature_to_array(
          example,
          io_tf_lib.tfgnn_edge_key(edge_set_name, feature_name),
          feature_schema,
          num_edges,
      )
    if import_edge_ids:
      edge_features[import_edge_ids] = example[
          io_tf_lib.tfgnn_edge_key(edge_set_name, import_edge_ids)
      ]
    edge_sets[edge_set_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=adjacency, features=edge_features
    )

  return in_memory_graph.InMemoryGraph(node_sets=node_sets, edge_sets=edge_sets)


def graph_to_tfgnn_graph(
    graph: in_memory_graph.InMemoryGraph, schema: schema_lib.GraphSchema
) -> tf.train.Example:
  """Converts an InMemoryGraph to a TF GNN Graph Sample."""
  feature_dict = graph_to_tfgnn_graph_dict(graph, schema)

  example = tf.train.Example()
  for key, value in feature_dict.items():
    flat_values = value.flatten().tolist()
    if np.issubdtype(value.dtype, np.floating):
      example.features.feature[key].float_list.value.extend(flat_values)
    elif np.issubdtype(value.dtype, np.integer):
      example.features.feature[key].int64_list.value.extend(flat_values)
    elif value.dtype.kind == "S":
      example.features.feature[key].bytes_list.value.extend(flat_values)
    else:
      raise ValueError(f"Unsupported dtype: {value.dtype} for key {key}")
  return example


def _flatten_feature_value(value: Any, np_dtype: Any) -> np.ndarray:
  """Flattens a (possibly nested) feature value into a 1D array of `np_dtype`."""
  array = np.asarray(value)
  if array.dtype == np.object_:
    # Note: The nested arrays are merged (which is only possible if the nested
    # values all have the same shape).
    array = np.asarray(array.tolist(), dtype=np_dtype)
  return np.ravel(array).astype(np_dtype, copy=False)


def _feature_to_tfgnn_values(
    feature_value: np.ndarray,
    feature_schema: schema_lib.FeatureSchema,
    feature_key: str,
) -> dict[str, np.ndarray]:
  """Converts a feature of an InMemoryGraph into TF GNN Graph Sample values.

  Args:
    feature_value: The feature values of shape [num_items, *shape].
    feature_schema: The schema of the feature.
    feature_key: The key of the feature in the TF GNN Graph Sample e.g.
      "nodes/n1.f1".

  Returns:
    A dictionary containing the flat feature values and, for variable-length
    features, the row lengths of each of the ragged dimensions.
  """
  np_dtype = feature_format_lib.FEATURE_FORMAT_TO_TFGNN_NP_DTYPE[
      feature_schema.format
  ]

  if feature_schema.is_static_shape():
    return {feature_key: _flatten_feature_value(feature_value, np_dtype)}

  if feature_value.dtype != np.object_:
    raise ValueError(
        f"The feature '{feature_key}' has a dynamic shape"
        f" {feature_schema.shape} but its values are not a numpy array of"
        f" dtype object. Found dtype: {feature_value.dtype}."
    )

  shape = list(feature_schema.shape or [])
  flat_values: list[np.ndarray] = []
  row_lengths: dict[int, list[int]] = {}

  def collect(rows: Any, dim_idx: int) -> None:
    """Collects the row lengths and flat values of the `dim_idx`-th dim."""
    # Note: The 0-th dimension of a feature value is the item dimension, so
    # `dim_idx` is in [1, len(shape)].
    dim = shape[dim_idx - 1]
    if dim is None:
      row_lengths.setdefault(dim_idx, []).extend(len(row) for row in rows)
    else:
      for row in rows:
        if len(row) != dim:
          raise ValueError(
              f"The dimension {dim_idx} of the feature '{feature_key}' is"
              f" expected to be of size {dim}, but a value of size"
              f" {len(row)} was found."
          )
    if all(sub_dim is not None for sub_dim in shape[dim_idx:]):
      # All the deeper dimensions are static: the rows can be flattened.
      for row in rows:
        flat_values.append(_flatten_feature_value(row, np_dtype))
      return
    collect([item for row in rows for item in row], dim_idx + 1)

  _check_at_most_one_ragged_dim(feature_key, shape)
  collect(feature_value, 1)

  result = {
      feature_key: (
          np.concatenate(flat_values, axis=0, dtype=np_dtype)
          if flat_values
          else np.array([], dtype=np_dtype)
      )
  }
  for dim_idx, lengths in row_lengths.items():
    result[io_tf_lib.tfgnn_ragged_dim_key(feature_key, dim_idx)] = np.array(
        lengths, dtype=np.int64
    )
  return result


def graph_to_tfgnn_graph_dict(
    graph: in_memory_graph.InMemoryGraph, schema: schema_lib.GraphSchema
) -> dict[str, np.ndarray]:
  """Converts an InMemoryGraph to a TF GNN Graph Sample Dict.

  The values are stored with the dtype used in a `tf.train.Example` proto (i.e.
  int64, float32 or bytes), and not with the dtype of the schema (e.g. int32,
  bool). Multi-dimensional features are flattened, and variable-length features
  are accompanied by the row lengths of each of their ragged dimensions.

  Args:
    graph: The graph to convert.
    schema: The schema of the graph.

  Returns:
    A TF GNN Graph Sample Dict i.e. a dictionary of flat numpy arrays.
  """
  feature_dict = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    node_set = graph.node_sets[nodeset_name]
    feature_dict[
        io_tf_lib.tfgnn_node_key(nodeset_name, io_tf_lib.TFGNN_SIZE_KEY)
    ] = np.array([node_set.num_nodes], dtype=np.int64)
    for feature_name, feature_schema in nodeset_schema.features.items():
      feature_dict.update(
          _feature_to_tfgnn_values(
              node_set.features[feature_name],
              feature_schema,
              io_tf_lib.tfgnn_node_key(nodeset_name, feature_name),
          )
      )

  for edge_set_name, edge_set_schema in schema.edge_sets.items():
    edge_set = graph.edge_sets[edge_set_name]
    feature_dict[
        io_tf_lib.tfgnn_edge_key(edge_set_name, io_tf_lib.TFGNN_SIZE_KEY)
    ] = np.array([edge_set.adjacency.shape[1]], dtype=np.int64)
    feature_dict[
        io_tf_lib.tfgnn_edge_key(edge_set_name, io_tf_lib.TFGNN_SOURCE_KEY)
    ] = np.asarray(edge_set.adjacency[0], dtype=np.int64)
    feature_dict[
        io_tf_lib.tfgnn_edge_key(edge_set_name, io_tf_lib.TFGNN_TARGET_KEY)
    ] = np.asarray(edge_set.adjacency[1], dtype=np.int64)
    for feature_name, feature_schema in edge_set_schema.features.items():
      feature_dict.update(
          _feature_to_tfgnn_values(
              edge_set.features[feature_name],
              feature_schema,
              io_tf_lib.tfgnn_edge_key(edge_set_name, feature_name),
          )
      )

  return feature_dict


def read_tfgnn_graphs_beam(
    pbegin: beam.Pipeline,
    path: str,
    schema: schema_lib.GraphSchema,
    *,
    container_type: (
        TFGraphSampleContainerType | str
    ) = TFGraphSampleContainerType.TF_RECORD,
    import_node_ids: str | None = None,
    import_edge_ids: str | None = None,
) -> distributed_graph_lib.PKeyedInMemoryGraph:
  """Read a collection of TF GNN Graphs.

  Usage example:

  ```python
  with beam.Pipeline() as p:
    schema = dgf.io.read_schema("/cns/../schema.json")
    graphs = dgf.io.beam.read_tfgnn_graphs(
        p,
        path="/cns/../data@.tfr",
        schema=schema)
  ```

  Args:
    pbegin: A beam pbegin.
    path: The path to the HGraph directory.
    schema: Schema.
    container_type: The type of container for the HGraph data.
    import_node_ids: Whether to import the ids of the nodes.
    import_edge_ids: Whether to import the ids of the edges. If a list, only
      import the ids of the specified edgeset names.

  Returns:
    A PCollection of keyyed in memory graphs.
  """
  return pbegin | f"Read {path}" >> ReadFromTFGraphSample(
      path=path,
      schema=schema,
      container_type=container_type,
      import_node_ids=import_node_ids,
      import_edge_ids=import_edge_ids,
  )


class ReadFromTFGraphSample(PTransform):
  """Read a collection of TF GNN Graphs."""

  def __init__(
      self,
      path: str,
      schema: schema_lib.GraphSchema,
      container_type: TFGraphSampleContainerType | str,
      import_node_ids: str | None,
      import_edge_ids: str | None,
  ):
    """Initializes the PTransform."""
    if isinstance(container_type, str):
      container_type = TFGraphSampleContainerType[container_type]
    self.path = shard_lib.shard_path_to_glob(path)
    self.container_type = container_type
    self.schema = schema
    self.import_node_ids = import_node_ids
    self.import_edge_ids = import_edge_ids

  def expand(
      self, pbegin: beam.pvalue.PBegin
  ) -> distributed_graph_lib.PKeyedInMemoryGraph:

    coder = beam.coders.ProtoCoder(tf.train.Example)
    if self.container_type == TFGraphSampleContainerType.TF_RECORD:
      keyed_tf_examples = (
          pbegin
          | f"Read TF Record {self.path}"
          >> beam.io.ReadFromTFRecord(
              self.path,
              coder=coder,
              compression_type=beam.io.filesystem.CompressionTypes.GZIP,
          )
          # Note: The examples are not keyyed.
          | "Add None Keys" >> beam.Map(lambda x: (None, x))
      )
    elif self.container_type == TFGraphSampleContainerType.BAGZ:

      keyed_tf_examples = (
          pbegin
          | f"Read Bagz {self.path}"
          >> bag_io.ReadFromBag(
              self.path,
              columns=(bag_io.Column(coder=coder),),
          )
          # Note: The examples are not keyyed.
          # TODO(gbm): Create a separate beam function / argument to extract a
          # key from one of the feature.
          | "Add None Keys" >> beam.Map(lambda x: (None, x))
      )
    else:
      raise ValueError(f"Unsupported container type: {self.container_type}")

    return keyed_tf_examples | "ToInMemoryGraph" >> beam.MapTuple(
        lambda key, example: distributed_graph_lib.KeyedInMemoryGraph(
            key,
            tfgnn_graph_to_graph(
                example,
                schema=self.schema,
                import_node_ids=self.import_node_ids,
                import_edge_ids=self.import_edge_ids,
            ),
        )
    )


def write_tfgnn_graphs_beam(
    graphs: distributed_graph_lib.PKeyedInMemoryGraph,
    path: str,
    schema: schema_lib.GraphSchema,
    container_type: (
        TFGraphSampleContainerType | str
    ) = TFGraphSampleContainerType.TF_RECORD,
) -> beam.PTransform:
  """Writes a collection of TF Graph Samples on disk.

  This function does not add a reshuffeling stage. Don't forget to add a
  reshuffle beam operation before the write if the data is unevenly
  distributed. If you are only doing a format conversion, or are only applying
  some 1:1 maps, reshuffeling is likely not benefitial.

  Args:
    graphs: A PCollection of `KeyedInMemoryGraph` to write.
    path: The sharded path to write the TF Graph Samples to. Supports sharding
      (e.g., "/path/to/data@10", "/path/to/data@*.sst").
    schema: The graph schema.
    container_type: The container format to use. Can be a
      `TFGraphSampleContainerType` enum or a string ("TF_RECORD", "SSTABLE",
      "BAGZ").
  """
  if isinstance(container_type, str):
    container_type = TFGraphSampleContainerType[container_type]

  # Check if schema has any dynamic shapes
  has_dynamic_shape = False
  for nodeset_schema in schema.node_sets.values():
    for feature_schema in nodeset_schema.features.values():
      if not feature_schema.is_static_shape():
        has_dynamic_shape = True
        break
    if has_dynamic_shape:
      break
  if not has_dynamic_shape:
    for edgeset_schema in schema.edge_sets.values():
      for feature_schema in edgeset_schema.features.values():
        if not feature_schema.is_static_shape():
          has_dynamic_shape = True
          break
      if has_dynamic_shape:
        break

  if not has_dynamic_shape:
    coder = beam.coders.BytesCoder()
    tf_examples = graphs | "ToSerializedBytes" >> beam.MapTuple(
        # pytype: disable=module-attr
        lambda key, graph: (key, tf_graph_sample_ext.serialize_graph(graph))
        # pytype: enable=module-attr
    )
  else:
    coder = beam.coders.ProtoCoder(tf.train.Example)
    tf_examples = graphs | "ToTFExample" >> beam.MapTuple(
        lambda key, graph: (key, graph_to_tfgnn_graph(graph, schema=schema))
    )
  basepath, num_shards, extension = shard_lib.parse_sharded_filename(path)
  if container_type == TFGraphSampleContainerType.TF_RECORD:
    return (
        tf_examples
        | "Remove Keys" >> beam.Values()
        | f"Write TF Record {path}"
        >> beam.io.WriteToTFRecord(
            file_path_prefix=basepath,
            file_name_suffix=extension,
            num_shards=num_shards or 0,
            coder=coder,
            compression_type=beam.io.filesystem.CompressionTypes.GZIP,
        )
    )
  elif container_type == TFGraphSampleContainerType.BAGZ:

    return (
        tf_examples
        | "Remove Keys" >> beam.Values()
        | f"Write Bagz {path}"
        >> bag_io.WriteToBag(
            path,
            columns=(bag_io.Column(coder=coder),),
        )
    )
  else:
    raise ValueError(f"Unsupported container type: {container_type}")


def write_tfgnn_graphs(
    graphs: Generator[in_memory_graph.InMemoryGraph, None, None],
    path: str,
    *,
    schema: schema_lib.GraphSchema,
    container_type: (
        TFGraphSampleContainerType | str
    ) = TFGraphSampleContainerType.TF_RECORD,
    compression: str = "GZIP",
    num_shards: int = 10,
):
  """Writes a set of in-memory graphs to disk as TF Examples.

  The writing is done in process, which is different from
  "write_tfgnn_graphs" which runs with Beam.

  Usage example:

  ```python
  # A generator of in-memory-graphs
  def generator():
    for _ in range(10):
      # You can use the graph sampler here.
      yield dgf.data.InMemoryGraph(...)

  # Generates the in memory graphs and write them to disk.
  write_tfgnn_graphs(generator(),
  "/my/data@10")

  # The examples can then be reloaded
  for graph in read_tfgnn_graphs(
        "/my/data@10"):
    pass
  ```

  Note that TF Graph Samples are very slow to read/write. For temporary storage
  of `InMemoryGraph` batches, using pickle is significantly faster.

  Args:
    graphs: An iterator of `InMemoryGraph` to write.
    path: The sharded path to write the TF Graph Samples to. Support sharding.
    schema: The graph schema.
    container_type: Container format.
    compression: TFRecord compression level. Can be "ZLIB", "GZIP", or "" (no
      compression).
    num_shards: Number of shards if the "path" is a sharded path without defined
      number of shards e.g. data@*.rio.  Ignored for other container types.
  """

  if isinstance(container_type, str):
    container_type = TFGraphSampleContainerType[container_type]

  paths = shard_lib.expand_output_paths(path, num_shards=num_shards)

  if container_type == TFGraphSampleContainerType.TF_RECORD:
    writers = [
        tf.io.TFRecordWriter(
            p, options=tf.io.TFRecordOptions(compression_type=compression)
        )
        for p in paths
    ]
  elif container_type == TFGraphSampleContainerType.BAGZ:

    writers = [bagz.Writer(p) for p in paths]
  else:
    raise ValueError("Non supported container type")

  try:
    for i, graph in enumerate(graphs):
      example = graph_to_tfgnn_graph(graph, schema=schema)
      serialized_example = example.SerializeToString()
      writer_idx = i % len(writers)
      writers[writer_idx].write(serialized_example)
  finally:
    for writer in writers:
      writer.close()


def write_tfgnn_graphs_single_file(
    graphs: Generator[in_memory_graph.InMemoryGraph, None, None],
    path: str,
    schema: schema_lib.GraphSchema,
    *,
    container_type: (
        TFGraphSampleContainerType | str
    ) = TFGraphSampleContainerType.TF_RECORD,
    compression: str = "GZIP",
):
  """Writes a set of graphs to a single file on disk as TF Examples."""

  if isinstance(container_type, str):
    container_type = TFGraphSampleContainerType[container_type]

  if container_type == TFGraphSampleContainerType.TF_RECORD:
    writer = tf.io.TFRecordWriter(
        path, options=tf.io.TFRecordOptions(compression_type=compression)
    )
  elif container_type == TFGraphSampleContainerType.BAGZ:

    writer = bagz.Writer(path)
  else:
    raise ValueError("Non supported container type")

  try:
    for graph in graphs:
      example = graph_to_tfgnn_graph(graph, schema=schema)
      serialized_example = example.SerializeToString()
      writer.write(serialized_example)
  finally:
    writer.close()


# The parsing spec of a TF GNN Graph Sample is defined in `dgf.src.io.tf` (so
# that it can be used for model inference without depending on this module and
# its heavy dependencies e.g. beam).
schema_to_tfgnn_graph_parsing_spec = (
    io_tf_lib.schema_to_tfgnn_graph_parsing_spec
)


def read_tfgnn_graphs(
    path: str | Sequence[str],
    schema: schema_lib.GraphSchema,
    import_node_ids: str | None = None,
    import_edge_ids: str | None = None,
    container_type: (
        TFGraphSampleContainerType | str
    ) = TFGraphSampleContainerType.TF_RECORD,
    compression: str = "GZIP",
) -> Generator[in_memory_graph.InMemoryGraph, None, None]:
  """Reads a set of in-memory graphs from disk stored as TF Examples.

  The reading is done in process, which is different from
  "ReadFromTFGraphSample" which runs with Beam.

  Usage example:

  ```python
  # A generator of in-memory-graphs
  def generator():
    for _ in range(10):
      # You can use the graph sampler here.
      yield dgf.data.InMemoryGraph(...)

  # Generates the in memory graphs and write them to disk.
  write_tfgnn_graphs_from_in_memory_graphs_in_process(generator(),
  "/my/data@10")

  # Reads the graphs
  for graph in read_tfgnn_graphs(
        "/my/data@10"):
    pass
  ```

  Note that TF Graph Samples are very slow to read/write. For temporary storage
  of `InMemoryGraph` batches, using pickle is significantly faster.

  Args:
    path: The sharded path to read the TF Graph Samples from. Supports sharding.
    schema: TF GNN schema.
    import_node_ids: Whether to import the ids of the nodes.
    import_edge_ids: Whether to import the ids of the edges.
    container_type: Container format.
    compression: TFRecord compression level. Can be "ZLIB", "GZIP", or "" (no
      compression). Ignored for other container types.
  """

  if isinstance(container_type, str):
    container_type = TFGraphSampleContainerType[container_type]
  paths = shard_lib.expand_input_paths(path)
  path_dataset = tf.data.Dataset.from_tensor_slices(paths)  # pyrefly: ignore[bad-argument-type]

  # Build the tf parsing spec.
  feature_spec = schema_to_tfgnn_graph_parsing_spec(schema)

  if container_type == TFGraphSampleContainerType.TF_RECORD:

    def read_serialized_proto_dataset(path):
      return tf.data.TFRecordDataset(path, compression_type=compression)  # pyrefly: ignore[bad-instantiation]

  else:
    raise ValueError("Non supported container type")

  dataset = path_dataset.interleave(
      read_serialized_proto_dataset,
      cycle_length=tf.data.AUTOTUNE,
      num_parallel_calls=tf.data.AUTOTUNE,
  )

  def parse_examples(x):
    x = tf.io.parse_example(x, feature_spec)
    x = {k: tf.sparse.to_dense(v) for k, v in x.items()}
    return x

  dataset = dataset.map(
      parse_examples,
      num_parallel_calls=tf.data.AUTOTUNE,
  )

  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  for tf_dict in dataset:
    np_dict = {}
    for key, tf_value in tf_dict.items():
      np_value = tf_value.numpy()
      if np_value.dtype == object:
        np_value = np_value.astype(np.bytes_)
      np_dict[key] = np_value

    in_memory_example = graph_dict_to_graph(
        np_dict,
        schema,
        import_node_ids=import_node_ids,
        import_edge_ids=import_edge_ids,
    )
    yield in_memory_example


def graph_to_serialized_tfgnn_graph(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema | None = None,
) -> bytes:
  """Converts an InMemoryGraph into a serialized TF-GNN graph sample proto.

  This function is equivalent to, but significantly faster than, calling:
  `graph_to_tfgnn_graph(graph, schema).SerializeToString()`.
  The performance improvement comes from reduced data copies and the
  implementation being entirely in C++.

  When serializing multiple graphs (e.g., a collection of graph samples), use
  the `graphs_to_serialized_tfgnn_graphs` method for even faster computation.

  Usage example:

  ```python
  graph, schema = gdf.io.read_graph("/tmp/my_graph")
  serialized_graph = dgf.convert.graph_to_serialized_tfgnn_graph(graph,schema)
  ```

  Args:
    graph: The input InMemoryGraph.
    schema: An optional and currently unused graph schema. This argument is
      included to ensure API consistency with other graph serialization
      functions, and it may be used in future implementations.

  Returns:
    Bytes of a serialized `tf.train.Example` proto containing the graph data
    in the TF-GNN format.
  """
  del schema
  return tf_graph_sample_ext.serialize_graph(graph)


def graphs_to_serialized_tfgnn_graphs(
    graphs: Sequence[in_memory_graph.InMemoryGraph],
    schema: schema_lib.GraphSchema | None = None,
    *,
    num_threads: int = os.cpu_count() * 2,  # pyrefly: ignore[unsupported-operation]
) -> list[bytes]:
  """Converts a sequence of InMemoryGraphs into serialized TF-GNN graph sample protos.

  This function is significantly faster than calling
  `graph_to_tfgnn_graph(graph, schema).SerializeToString()` or
  `graph_to_tfgnn_graph` in a loop.

  ```python
  def graph_generator():
    for graph in <sampler>:
      yield graph
  serialized_graphs =
  dgf.convert.graphs_to_serialized_tfgnn_graphs(graph_generator,schema)
  ```

  Args:
    graphs: The input sequence of InMemoryGraphs.
    schema: An optional and currently unused graph schema. This argument is
      included to ensure API consistency with other graph serialization
      functions, and it may be used in future implementations.
    num_threads: The number of threads to use for serialization. If negative,
      the GIL will be released, but a single thread will be used.

  Returns:
    A list of bytes, where each element is a serialized `tf.train.Example` proto
    containing the data for one graph in the TF-GNN format.
  """

  del schema
  return tf_graph_sample_ext.serialize_graphs(graphs, num_threads)


# TODO(gbm): Add efficient serialized proto TF-gnn graph into a GF graph?
