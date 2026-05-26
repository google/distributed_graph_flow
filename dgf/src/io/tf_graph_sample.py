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

from collections.abc import Mapping
import enum
import os
from typing import Dict, Generator, List, Optional, Sequence
import apache_beam as beam
import bagz
from bagz.beam import bagzio as bag_io
from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import tf_graph_sample_ext
from dgf.src.util import shard as shard_lib
import numpy as np
import tensorflow as tf


class TFGraphSampleContainerType(enum.Enum):
  TF_RECORD = 1
  SSTABLE = 2
  BAGZ = 3


def tfgnn_graph_to_graph(
    example: tf.train.Example,
    schema: schema_lib.GraphSchema,
    import_node_ids: Optional[str] = None,
    import_edge_ids: Optional[str] = None,
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


def graph_dict_to_graph(
    example: Dict[str, np.ndarray],
    schema: schema_lib.GraphSchema,
    import_node_ids: Optional[str] = None,
    import_edge_ids: Optional[str] = None,
) -> in_memory_graph.InMemoryGraph:
  """Converts a TF GNN Graph Sample Dict to an InMemoryGraph."""
  node_sets = {}
  edge_sets = {}

  def _parse_feature(
      src_feature_name: str,
      dst_feature_name: str,
      feature_schema: schema_lib.FeatureSchema,
      num_items: int,
      dst_dict: Dict[str, np.ndarray],
  ) -> None:
    feature_value = example[src_feature_name]

    if feature_schema.is_static_shape():
      feature_value = feature_value.reshape(
          (num_items,) + (feature_schema.shape or ())
      )
      dst_dict[dst_feature_name] = feature_value
    else:
      # TODO(gbm): Add support for more than one ragged dimension.
      dim_value = example[f"{src_feature_name}.d1"]
      assert len(dim_value) == num_items
      list_values = []
      begin_idx = 0
      num_sclars_per_item = feature_schema.static_size()
      target_shape = feature_schema.shape or ()
      target_shape = tuple([x if x is not None else -1 for x in target_shape])
      for item_idx in range(num_items):
        end_idx = begin_idx + dim_value[item_idx] * num_sclars_per_item
        list_values.append(
            feature_value[begin_idx:end_idx].reshape(target_shape)
        )
        begin_idx = end_idx

      # Note: Make sure numpy does not merge the arrays.
      array_of_object = np.empty(len(list_values), dtype=np.object_)
      array_of_object[:] = list_values

      dst_dict[dst_feature_name] = array_of_object

  for node_set_name, node_set in schema.node_sets.items():
    node_features = {}
    num_nodes = example[f"nodes/{node_set_name}.#size"][0].item()
    for feature_name, feature_schema in node_set.features.items():
      _parse_feature(
          f"nodes/{node_set_name}.{feature_name}",
          feature_name,
          feature_schema,
          num_nodes,
          node_features,
      )
    if import_node_ids:
      node_features[import_node_ids] = example[
          f"nodes/{node_set_name}.{import_node_ids}"
      ]
    node_sets[node_set_name] = in_memory_graph.InMemoryNodeSet(
        features=node_features,
        num_nodes=num_nodes,
    )

  for edge_set_name, edge_set in schema.edge_sets.items():
    source_feature_key = f"edges/{edge_set_name}.#source"
    target_feature_key = f"edges/{edge_set_name}.#target"
    adjacency = np.array(
        [example[source_feature_key], example[target_feature_key]]
    )
    edge_features = {}
    num_edges = example[f"edges/{edge_set_name}.#size"][0].item()
    for feature_name, feature_schema in edge_set.features.items():
      _parse_feature(
          f"edges/{edge_set_name}.{feature_name}",
          feature_name,
          feature_schema,
          num_edges,
          edge_features,
      )
    if import_edge_ids:
      edge_features[import_edge_ids] = example[
          f"edges/{edge_set_name}.{import_edge_ids}"
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


def graph_to_tfgnn_graph_dict(
    graph: in_memory_graph.InMemoryGraph, schema: schema_lib.GraphSchema
) -> Dict[str, np.ndarray]:
  """Converts an InMemoryGraph to a TF GNN Graph Sample Dict."""
  feature_dict = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    node_set = graph.node_sets[nodeset_name]
    feature_dict[f"nodes/{nodeset_name}.#size"] = np.array(
        [node_set.num_nodes], dtype=np.int64
    )
    for feature_name, feature_schema in nodeset_schema.features.items():
      feature_value = node_set.features[feature_name]
      tf_feature_name = f"nodes/{nodeset_name}.{feature_name}"
      if feature_schema.is_static_shape():
        feature_dict[tf_feature_name] = feature_value
      else:
        if feature_value.dtype != object:
          raise ValueError(
              f"Feature '{feature_name}' in node set '{nodeset_name}' has a "
              "dynamic shape but is not a numpy array of dtype object. "
              f"Found type: {type(feature_value)}, dtype: {feature_value.dtype}"
          )
        feature_dict[tf_feature_name] = np.concatenate(
            [value.flatten() for value in feature_value],
            axis=0,
            dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
                feature_schema.format
            ],
        )
        # TODO(gbm): Add support for more than one "ragged" dimension.
        feature_dict[f"nodes/{nodeset_name}.{feature_name}.d1"] = np.array(
            [len(value) for value in feature_value],
            dtype=np.int64,
        )

  for edge_set_name, edge_set_schema in schema.edge_sets.items():
    edge_set = graph.edge_sets[edge_set_name]
    feature_dict[f"edges/{edge_set_name}.#size"] = np.array(
        [edge_set.adjacency.shape[1]], dtype=np.int64
    )
    feature_dict[f"edges/{edge_set_name}.#source"] = edge_set.adjacency[0]
    feature_dict[f"edges/{edge_set_name}.#target"] = edge_set.adjacency[1]
    for feature_name, feature_schema in edge_set_schema.features.items():
      feature_value = edge_set.features[feature_name]
      tf_feature_name = f"edges/{edge_set_name}.{feature_name}"
      if feature_schema.is_static_shape():
        feature_dict[tf_feature_name] = feature_value
      else:
        if feature_value.dtype != object:
          raise ValueError(
              f"Feature '{feature_name}' in edge set '{edge_set_name}' has a "
              "dynamic shape but is not a numpy array of dtype object. "
              f"Found type: {type(feature_value)}, dtype: {feature_value.dtype}"
          )
        feature_dict[tf_feature_name] = np.concatenate(
            [value.flatten() for value in feature_value], axis=0
        )
        # TODO(gbm): Add support for more than one "ragged" dimension.
        feature_dict[f"edges/{edge_set_name}.{feature_name}.d1"] = np.array(
            [len(value) for value in feature_value]
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
    import_node_ids: Optional[str] = None,
    import_edge_ids: Optional[str] = None,
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


class ReadFromTFGraphSample(beam.PTransform):
  """Read a collection of TF GNN Graphs."""

  def __init__(
      self,
      path: str,
      schema: schema_lib.GraphSchema,
      container_type: TFGraphSampleContainerType | str,
      import_node_ids: Optional[str],
      import_edge_ids: Optional[str],
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


def build_tfgnn_feature_spec(
    schema: schema_lib.GraphSchema,
) -> Dict[str, tf.io.VarLenFeature]:
  """Builds the tf parsing spec from the graph schema."""
  feature_spec = {}

  def add_feature_to_spec(key: str, feature_schema: schema_lib.FeatureSchema):
    tf_dtype = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[
        feature_schema.format
    ]
    feature_spec[key] = tf.io.VarLenFeature(tf_dtype)
    if not feature_schema.is_static_shape():
      feature_spec[f"{key}.d1"] = tf.io.VarLenFeature(tf.int64)

  for node_set_name, node_set_schema in schema.node_sets.items():
    feature_spec[f"nodes/{node_set_name}.#size"] = tf.io.VarLenFeature(tf.int64)
    for feature_name, feature_schema in node_set_schema.features.items():
      add_feature_to_spec(
          f"nodes/{node_set_name}.{feature_name}", feature_schema
      )
  for edge_set_name, edge_set_schema in schema.edge_sets.items():
    feature_spec[f"edges/{edge_set_name}.#source"] = tf.io.VarLenFeature(
        tf.int64
    )
    feature_spec[f"edges/{edge_set_name}.#target"] = tf.io.VarLenFeature(
        tf.int64
    )
    feature_spec[f"edges/{edge_set_name}.#size"] = tf.io.VarLenFeature(tf.int64)
    for feature_name, feature_schema in edge_set_schema.features.items():
      add_feature_to_spec(
          f"edges/{edge_set_name}.{feature_name}", feature_schema
      )
  return feature_spec


def read_tfgnn_graphs(
    path: str,
    schema: schema_lib.GraphSchema,
    import_node_ids: Optional[str] = None,
    import_edge_ids: Optional[str] = None,
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
  path_dataset = tf.data.Dataset.from_tensor_slices(paths)

  # Build the tf parsing spec.
  feature_spec = build_tfgnn_feature_spec(schema)

  if container_type == TFGraphSampleContainerType.TF_RECORD:

    def read_serialized_proto_dataset(path):
      return tf.data.TFRecordDataset(path, compression_type=compression)

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
    num_threads: int = os.cpu_count() * 2,
) -> List[bytes]:
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
