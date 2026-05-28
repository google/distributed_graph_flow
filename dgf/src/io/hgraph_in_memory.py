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

"""Reading and writing HGraph in memory."""

import enum
import logging
import os
import time
from typing import Callable, Dict, Optional, TYPE_CHECKING, Tuple
from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import hgraph_in_avro
from dgf.src.io import tfrecord as tfrecord_lib
from dgf.src.util import filesystem
from dgf.src.util import proto as proto_lib
from dgf.src.util import shard as shard_lib
from dgf.src.util import weak_dep
import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
  from tensorflow_gnn import proto as tf_gnn_proto

PATH_GRAPH_SCHEMA = "graph_schema.pbtxt"
PATH_NODE_FEATURE = "node_features"
PATH_EDGES = "edges"
DEFAULT_KEY_ID = "#id"
KEY_SOURCE = "#source"
KEY_TARGET = "#target"


class HGraphContainerType(enum.Enum):
  """Type of container for HGraph data."""

  TF_RECORD = 1
  SSTABLE = 2
  AVRO = 3
  # TODO(gbm): Add an "AUTO" option.


def get_extension(container_type: HGraphContainerType) -> str:
  """Returns the file extension for the given container type."""
  return {
      HGraphContainerType.TF_RECORD: ".tfrecord.gz",
      HGraphContainerType.SSTABLE: ".sst",
      HGraphContainerType.AVRO: ".avro",
  }[container_type]


def tfgnn_schema_to_schema(
    tfgnn_schema: "tf_gnn_proto.GraphSchema",
) -> schema_lib.GraphSchema:
  """Converts a TF-GNN schema proto into a GraphSchema object.

  Args:
    tfgnn_schema: A TF-GNN schema proto.

  Returns:
    A GraphSchema object.
  """

  def convert_feature(gnn_feature) -> schema_lib.FeatureSchema:
    feature_format = feature_format_lib.TF_DTYPE_TO_FEATURE_FORMAT[
        tf.dtypes.as_dtype(gnn_feature.dtype)
    ]
    feature_schema = schema_lib.FeatureSchema(format=feature_format)
    if gnn_feature.HasField("shape"):
      # GF represent unknown dimensions with None, while TF-GNN represents
      # them with -1.
      shape = [d.size if d.size >= 0 else None for d in gnn_feature.shape.dim]
      feature_schema.shape = tuple(shape)
    return feature_schema

  node_sets = {}
  for node_set_name, node_set in tfgnn_schema.node_sets.items():
    features = {}
    for feature_name, feature in node_set.features.items():
      features[feature_name] = convert_feature(feature)
    node_sets[node_set_name] = schema_lib.NodeSchema(features=features)

  edge_sets = {}
  for edge_set_name, edge_set in tfgnn_schema.edge_sets.items():
    features = {}
    for feature_name, feature in edge_set.features.items():
      features[feature_name] = convert_feature(feature)
    edge_sets[edge_set_name] = schema_lib.EdgeSchema(
        features=features, source=edge_set.source, target=edge_set.target
    )

  schema = schema_lib.GraphSchema(node_sets=node_sets, edge_sets=edge_sets)
  analyse_schema_lib.fix_schema(schema, create_pound_id_as_fall_back=True)
  return schema


def schema_to_tfgnn_schema(
    schema: schema_lib.GraphSchema,
    add_reverse_edges: bool = False,
) -> "tf_gnn_proto.GraphSchema":
  """Converts a GraphSchema object into a TF-GNN schema proto.

  Args:
    schema: A GraphSchema object.
    add_reverse_edges: If true, for each edge set in the schema, a corresponding
      reverse edge set is added. For example, an edge set named "my_edge" will
      result in two edge sets in the output TF-GNN schema: "my_edge" (the
      original) and "reverse_my_edge" (with source and target swapped).

  Returns:
    A TF-GNN schema proto.
  """

  tf_gnn_proto = weak_dep.import_tf_gnn_proto()

  def convert_feature_schema(feature: schema_lib.FeatureSchema):
    feature_type = feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[
        feature.format
    ].as_datatype_enum
    feature_schema = tf_gnn_proto.Feature(dtype=feature_type)
    if feature.shape is not None:
      feature_schema.shape.SetInParent()
      for d in feature.shape:
        feature_schema.shape.dim.add(size=d if d is not None else -1)
    return feature_schema

  node_sets = {}
  for node_set_name, node_set in schema.node_sets.items():
    features = {}
    for feature_name, feature in node_set.features.items():
      features[feature_name] = convert_feature_schema(feature)
    node_sets[node_set_name] = tf_gnn_proto.NodeSet(features=features)

  edge_sets = {}
  for edge_set_name, edge_set in schema.edge_sets.items():
    features = None
    if edge_set.features is not None:
      features = {}
      for feature_name, feature in edge_set.features.items():
        features[feature_name] = convert_feature_schema(feature)
    edge_sets[edge_set_name] = tf_gnn_proto.EdgeSet(
        features=features, source=edge_set.source, target=edge_set.target
    )
    if add_reverse_edges:
      edge_sets[f"reverse_{edge_set_name}"] = tf_gnn_proto.EdgeSet(
          features=features, target=edge_set.source, source=edge_set.target
      )

  return tf_gnn_proto.GraphSchema(node_sets=node_sets, edge_sets=edge_sets)


def read_graphai_hgraph(
    path: str,
    container_type: HGraphContainerType | str = HGraphContainerType.TF_RECORD,
    verbose: bool = True,
    node_id_column: Optional[str] = None,
    edge_id_column: Optional[str] = None,
    schema_transformer: Optional[
        Callable[[schema_lib.GraphSchema], schema_lib.GraphSchema]
    ] = None,
    override_schema: Optional[schema_lib.GraphSchema] = None,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads an on-disk HGraph into an in-memory representation.

  This function is suitable for datasets that can fully fit in memory. Loading
  the graph as an `dgf.data.InMemoryGraph` enables faster training
  and sampling operations compared to distributed processing.

  For larger HGraphs that do not fit in memory, use `dgf.beam.io.ReadFromHGraph`
  to read the graph using Apache Beam.

  Usage example:

    ```python
    graph, schema = gdf.io.read_graphai_hgraph("/tmp/my_hgraph")
    print(f"Loaded graph with schema: {schema}")
    print(f"Number of nodes in 'n1': {graph.node_sets['n1'].num_nodes}")
    ```

  TODO(gbm): Allow selective loading of features/edge values.
  TODO(gbm): Add control for the data backend (e.g., jax, numpy).
  TODO(gbm): Parallelize IO operations.

  Args:
    path: Path to the HGraph directory.
    container_type: Container format used by the HGraph directory.
    verbose: If true, display reading progress.
    node_id_column: Column name containing the node id. See "ReadFromHGraph"
      documentation for "node_id_column" for full details.
    edge_id_column: Column name containing the node id. See "ReadFromHGraph"
      documentation for "edge_id_column" for full details.
    schema_transformer: An optional callable that takes the read
      `schema_lib.GraphSchema` and returns a transformed
      `schema_lib.GraphSchema`. This allows for on-the-fly modifications of the
      schema before loading the graph data. Applied both if "schema" is provided
      or not.
    override_schema: Schema of the HGraph. If not provided, the schema is loaded
      from the TF GNN or AVRO schema contained in the HGraph. Specifying the
      format allows to only includes a subset of features / nodesets / edgesets.

  Returns:
    An in-memory heterogeneous graph.
  """
  start_time = time.monotonic()

  if isinstance(container_type, str):
    container_type = HGraphContainerType[container_type]
  # TODO(gbm): Add support for other formats.
  if container_type not in [
      HGraphContainerType.TF_RECORD,
      HGraphContainerType.AVRO,
  ]:
    raise ValueError(
        f"Unsupported container type for in-memory graph: {container_type}. "
        "Only HGraphContainerType.TF_RECORD and HGraphContainerType.AVRO are"
        " currently supported."
    )
  extension = get_extension(container_type)

  # Import TF-GNN schema proto
  if override_schema is None:
    if verbose:
      print("Loading schema")
    tf_gnn_proto = weak_dep.import_tf_gnn_proto()
    tfgnn_schema = proto_lib.read_text_proto(
        os.path.join(path, PATH_GRAPH_SCHEMA),
        tf_gnn_proto.GraphSchema,
    )
    schema = tfgnn_schema_to_schema(tfgnn_schema)
  else:
    schema = override_schema
  if schema_transformer is not None:
    schema = schema_transformer(schema)

  # Final container of nodesets / edgesets.
  node_sets: Dict[str, in_memory_graph_lib.InMemoryNodeSet] = {}
  edge_sets: Dict[str, in_memory_graph_lib.InMemoryEdgeSet] = {}

  # Maps each node set name to a vectorized function that converts raw byte
  # IDs to integer indices.
  nodeset_mapping = {}

  # Default id column values. We assume this is a TF_RECORD or AVRO file.
  if node_id_column is None:
    node_id_column = DEFAULT_KEY_ID

  def default_shape(shape):
    if shape is None:
      return ()
    if None in shape:
      raise ValueError(
          "read_graphai_hgraph does not support shapes with None values:"
          f" {shape}"
      )
    return shape

  def _feature_format_to_container_dtype(
      container_type: HGraphContainerType,
      feature_format: schema_lib.FeatureFormat,
  ) -> tf.DType | str:
    """Returns the dtype for the given container type and feature format."""
    if container_type == HGraphContainerType.TF_RECORD:
      return feature_format_lib.FEATURE_FORMAT_TO_TF_DTYPE[feature_format]
    elif container_type == HGraphContainerType.AVRO:
      return feature_format_lib.FEATURE_FORMAT_TO_AVRO_DTYPE[feature_format]
    else:
      raise ValueError(
          f"Unsupported container type: {container_type}. Only"
          " HGraphContainerType.TF_RECORD and HGraphContainerType.AVRO are"
          " supported."
      )

  def _read_container(
      paths: list[str],
      container_type: HGraphContainerType,
      columns: Dict[str, Tuple[tf.DType | str, Tuple[Optional[int], ...]]],
      verbose: bool,
  ) -> Tuple[Dict[str, np.ndarray], int]:
    """Reads features given a container type."""
    if container_type == HGraphContainerType.TF_RECORD:
      features, num_records = tfrecord_lib.read_tf_record(
          paths=paths, columns=columns, verbose=verbose, preserve_order=False
      )
    elif container_type == HGraphContainerType.AVRO:
      features, num_records = hgraph_in_avro.read_avro_record(
          paths=paths, columns=columns, verbose=verbose
      )
    else:
      raise ValueError(
          f"Unsupported container type for in-memory graph: {container_type}. "
          "Only HGraphContainerType.TF_RECORD and HGraphContainerType.AVRO are"
          " currently supported."
      )
    return features, num_records

  # Load the nodesets
  for nodeset_name, nodeset_def in schema.node_sets.items():
    if verbose:
      print(f"Loading nodeset: {nodeset_name}")
    paths = shard_lib.list_paths(
        os.path.join(path, PATH_NODE_FEATURE, nodeset_name),
        extension,
    )

    if not paths:
      raise FileNotFoundError(
          f"Nodeset '{nodeset_name}' defined in schema but file not found at"
          f" {os.path.join(path, PATH_NODE_FEATURE)}."
      )

    columns = {
        k: (
            _feature_format_to_container_dtype(container_type, v.format),
            default_shape(v.shape),
        )
        for k, v in nodeset_def.features.items()
    }
    if node_id_column not in columns:
      # If the user does not define the dtype of the node id,
      # set it to string.
      columns[node_id_column] = (
          _feature_format_to_container_dtype(
              container_type, schema_lib.FeatureFormat.BYTES
          ),
          (),
      )
      logging.info(
          "No id feature in the schema. Adding a virtual string id feature"
      )

    node_features, num_nodes = _read_container(
        paths=paths,
        container_type=container_type,
        columns=columns,
        verbose=verbose,
    )

    # Index the nodeset ids
    if verbose:
      print("Build index")
    node_raw_ids = node_features[node_id_column]
    if node_raw_ids.ndim != 1:
      raise ValueError(
          f"Expected node raw IDs for nodeset '{nodeset_name}' to be a 1D array"
          f" but got shape: {node_raw_ids.shape}"
      )
    mapping = {id.item(): idx for idx, id in enumerate(node_raw_ids)}

    def mapper(x, mapping=mapping, nodeset_name=nodeset_name):
      v = mapping.get(x)
      if v is None:
        raise ValueError(
            f"Node ID '{x.decode()}' not found in nodeset '{nodeset_name}'."
            f" Some values: {list(mapping.items())[:10]!r}"
        )
      return v

    nodeset_mapping[nodeset_name] = np.vectorize(
        mapper,
        otypes=["int64"],
    )

    # Create the nodeset
    if node_id_column not in nodeset_def.features:
      # TODO(gbm): Add argument to keep the node id event if it is not part of
      # the schema.
      del node_features[node_id_column]
    node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
        features=node_features,
        num_nodes=num_nodes,
    )

  # Load the edgesets
  for edge_set_name, edge_set in schema.edge_sets.items():
    if verbose:
      print(f"Loading edgeset: {edge_set_name}")
    paths = shard_lib.list_paths(
        os.path.join(path, PATH_EDGES, edge_set_name),
        extension,
    )
    if not paths:
      raise FileNotFoundError(
          f"Edgeset '{edge_set_name}' defined in schema but file not found at"
          f" {os.path.join(path, PATH_EDGES, edge_set_name)}."
      )

    # Get the id dtype for the source/target nodesets
    source_node_id_dtype = (
        schema.node_sets[edge_set.source].features[node_id_column].format
    )
    target_node_id_dtype = (
        schema.node_sets[edge_set.target].features[node_id_column].format
    )

    columns = {
        KEY_SOURCE: (
            _feature_format_to_container_dtype(
                container_type, source_node_id_dtype
            ),
            (),
        ),
        KEY_TARGET: (
            _feature_format_to_container_dtype(
                container_type, target_node_id_dtype
            ),
            (),
        ),
    }

    if edge_set.features:
      for k, v in edge_set.features.items():
        columns[k] = (
            _feature_format_to_container_dtype(container_type, v.format),
            default_shape(v.shape),
        )

    if edge_id_column is not None and edge_id_column not in columns:
      columns[edge_id_column] = (
          _feature_format_to_container_dtype(
              container_type, schema_lib.FeatureFormat.BYTES
          ),
          (),
      )

    raw_edges, _ = _read_container(
        paths=paths,
        container_type=container_type,
        columns=columns,
        verbose=verbose,
    )

    if verbose:
      print("Compute adjacency")

    adjacency = np.stack([
        nodeset_mapping[edge_set.source](raw_edges[KEY_SOURCE]),
        nodeset_mapping[edge_set.target](raw_edges[KEY_TARGET]),
    ])

    features = raw_edges
    # Deleting source and target from features because they have been converted
    # to adjacencies.
    del features[KEY_SOURCE]
    del features[KEY_TARGET]
    # Delete empty edge_id_column
    if edge_id_column is not None and (
        edge_id_column not in features or features[edge_id_column].size == 0
    ):
      del features[edge_id_column]
    # TODO(gbm): Add argument to keep the edge id.
    edge_sets[edge_set_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=adjacency, features=features
    )

  end_time = time.monotonic()
  print(f"HGraph read in memory in {end_time - start_time:.2f} seconds")

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets=node_sets, edge_sets=edge_sets
  )
  return graph, schema


def in_memory_node_to_tf_example(
    node_index: int,
    feature_schema: schema_lib.FeatureSetSchema,
    features: dict[str, np.ndarray] | None,
    node_id_column: Optional[str],
) -> tf.train.Example:
  """Converts in-memory node features to a tf example, using feature schema

  as the blueprint for features to be converted.

  In particular, the #id field in features is reserved for node id.
  """
  example = tf.train.Example()
  if features is not None:
    for feature_name in feature_schema.keys():
      # in memory graph uses #id as the node id column name.
      if feature_name == DEFAULT_KEY_ID:
        continue
      if feature_name not in features:
        continue
      feature_values = features[feature_name]
      if feature_values.ndim == 1:
        feature_value = [feature_values[node_index]]
      else:
        feature_value = feature_values[node_index]
      if np.issubdtype(feature_values.dtype, np.integer):
        example.features.feature[feature_name].int64_list.value.extend(
            feature_value
        )
      elif np.issubdtype(feature_values.dtype, np.floating):
        example.features.feature[feature_name].float_list.value.extend(
            feature_value
        )
      elif feature_values.dtype.kind == "S":
        example.features.feature[feature_name].bytes_list.value.extend(
            feature_value
        )
      else:
        raise ValueError(f"Non supported type {feature_values.dtype}")
    # For nodes, #id is required.
  if (
      node_id_column is not None
      and node_id_column not in example.features.feature
  ):
    if np.issubdtype(features[DEFAULT_KEY_ID].dtype, np.integer):
      example.features.feature[node_id_column].int64_list.value.append(
          features[DEFAULT_KEY_ID][node_index]
      )
    elif features[DEFAULT_KEY_ID].dtype.kind == "S":
      example.features.feature[node_id_column].bytes_list.value.append(
          features[DEFAULT_KEY_ID][node_index]
      )
    else:
      raise ValueError(f"Non supported type {features[DEFAULT_KEY_ID]}")
  return example


def set_tf_scalar(dst, value, format: schema_lib.FeatureFormat):
  if format.is_integer():
    dst.int64_list.value.append(value)
  elif format == schema_lib.FeatureFormat.BYTES:
    dst.bytes_list.value.append(value)
  else:
    raise ValueError(f"Unsupported format: {format}")


def in_memory_edge_to_tf_example(
    edge_index: int,
    feature_schema: schema_lib.FeatureSetSchema,
    source: int | bytes,
    source_format: schema_lib.FeatureFormat,
    target: int | bytes,
    target_format: schema_lib.FeatureFormat,
    features: dict[str, np.ndarray] | None,
    edge_id_column: Optional[str],
) -> tf.train.Example:
  """Converts in-memory edge adjacency to a tf example."""
  example = tf.train.Example()
  set_tf_scalar(
      example.features.feature[KEY_SOURCE],
      source,
      source_format,
  )
  set_tf_scalar(
      example.features.feature[KEY_TARGET],
      target,
      target_format,
  )
  if features is not None:
    for feature_name in feature_schema.keys():
      if feature_name == DEFAULT_KEY_ID:
        continue
      if feature_name not in features:
        continue
      feature_values = features[feature_name]
      if feature_values.ndim == 1:
        feature_value = [feature_values[edge_index]]
      else:
        feature_value = feature_values[edge_index]
      if np.issubdtype(feature_values.dtype, np.integer):
        example.features.feature[feature_name].int64_list.value.extend(
            feature_value
        )
      elif np.issubdtype(feature_values.dtype, np.floating):
        example.features.feature[feature_name].float_list.value.extend(
            feature_value
        )
      elif feature_values.dtype.kind == "S":
        example.features.feature[feature_name].bytes_list.value.extend(
            feature_value
        )
      else:
        raise ValueError(f"Non supported type {feature_values.dtype}")

  # For edges, #id is optional.
  if (
      edge_id_column is not None
      and edge_id_column in feature_schema
      and features is not None
      and DEFAULT_KEY_ID in features
      and edge_id_column not in example.features.feature
  ):
    if np.issubdtype(features[DEFAULT_KEY_ID].dtype, np.integer):
      example.features.feature[edge_id_column].int64_list.value.append(
          features[DEFAULT_KEY_ID][edge_index]
      )
    elif features[DEFAULT_KEY_ID].dtype.kind == "S":
      example.features.feature[edge_id_column].bytes_list.value.append(
          features[DEFAULT_KEY_ID][edge_index]
      )
    else:
      raise ValueError(f"Non supported type {features[DEFAULT_KEY_ID]}")
  return example


def _validate_in_memory_graph_to_hgraph_request(
    graph: in_memory_graph_lib.InMemoryGraph,
    container_type: HGraphContainerType | str,
):
  """Validates the arguments for writing an in-memory graph to HGraph.

  Args:
    graph: The in-memory graph to write.
    container_type: The type of container for the HGraph data.

  Raises:
    ValueError: If the container type is not TF_RECORD or AVRO, if a nodeset is
    missing the number of nodes or features, or if the node ID column is not
    found in a nodeset's features.
  """
  if container_type not in (
      HGraphContainerType.TF_RECORD,
      HGraphContainerType.AVRO,
  ):
    raise ValueError(
        "Only TF_RECORD and AVRO are supported for in-memory graph to HGraph"
        " writing."
    )

  for nodeset_name, nodeset in graph.node_sets.items():
    if nodeset.num_nodes is None:
      raise ValueError(
          f"Node set {nodeset_name} has no number of nodes. Please provide the"
          " number of nodes in the node set."
      )
    if nodeset.features is None:
      raise ValueError(
          f"Node set {nodeset_name} has no features. Please provide the"
          " features in the node set."
      )
    if DEFAULT_KEY_ID not in nodeset.features:
      raise ValueError(
          f"Node set {nodeset_name} does not have a node ID column"
          f" {DEFAULT_KEY_ID}. Please provide the node IDs"
          " as features in the node set."
      )


def _write_tfrecord_node_sets(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    directory: str,
    extension: str,
    node_id_column: str,
):
  """Writes node sets to TFRecord files."""
  for nodeset_name, nodeset in graph.node_sets.items():
    num_shards, num_nodes_per_shard = shard_lib.estimate_num_node_shards(
        nodeset.num_nodes
    )
    for shard_index in range(num_shards):
      examples = []
      filename = shard_lib.sharded_filename(
          filename=nodeset_name,
          shard=shard_index,
          num_shards=num_shards,
          extension=extension,
      )
      for node_index in range(
          shard_index * num_nodes_per_shard,
          min(
              (shard_index + 1) * num_nodes_per_shard,
              nodeset.num_nodes,
          ),
      ):
        examples.append(
            in_memory_node_to_tf_example(
                node_index,
                schema.node_sets[nodeset_name].features,
                nodeset.features if nodeset.features else None,
                node_id_column,
            )
        )
        # TODO(canliu): parallelize the serialization for better performance.
        # Example: cl/813227631.
      tfrecord_lib.write_tf_record(os.path.join(directory, filename), examples)


def _write_tfrecord_edge_sets(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    directory: str,
    extension: str,
    node_id_column: str,
    edge_id_column: str,
):
  """Writes edge sets to TFRecord files."""
  for edgeset_name, edgeset in graph.edge_sets.items():
    num_edges = edgeset.adjacency.shape[1]
    source_format = (
        schema.node_sets[schema.edge_sets[edgeset_name].source]
        .features[node_id_column]
        .format
    )
    target_format = (
        schema.node_sets[schema.edge_sets[edgeset_name].target]
        .features[node_id_column]
        .format
    )
    num_shards, num_edges_per_shard = shard_lib.estimate_num_edge_shards(
        num_edges
    )
    for shard_index in range(num_shards):
      examples = []
      filename = shard_lib.sharded_filename(
          filename=edgeset_name,
          shard=shard_index,
          num_shards=num_shards,
          extension=extension,
      )
      for edge_index in range(
          shard_index * num_edges_per_shard,
          min(
              (shard_index + 1) * num_edges_per_shard,
              num_edges,
          ),
      ):
        source_index, target_index = edgeset.adjacency[:, edge_index]
        source_nodeset = schema.edge_sets[edgeset_name].source
        target_nodeset = schema.edge_sets[edgeset_name].target
        source = graph.node_sets[source_nodeset].features[node_id_column][
            source_index
        ]
        target = graph.node_sets[target_nodeset].features[node_id_column][
            target_index
        ]

        examples.append(
            in_memory_edge_to_tf_example(
                edge_index,
                schema.edge_sets[edgeset_name].features,
                source,
                source_format,
                target,
                target_format,
                edgeset.features if edgeset.features else None,
                edge_id_column,
            )
        )
      tfrecord_lib.write_tf_record(os.path.join(directory, filename), examples)


def write_graphai_hgraph(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    path: str,
    container_type: HGraphContainerType | str = HGraphContainerType.TF_RECORD,
    node_id_column: Optional[str] = None,
    edge_id_column: Optional[str] = None,
    verbose: bool = True,
):
  """Writes an in-memory heterogeneous graph to an HGraph directory.

  Args:
    graph: The in-memory graph to write.
    schema: The schema of the graph.
    path: The path to the HGraph directory.
    container_type: The type of container for the HGraph data. Currently only
      TF_RECORD and AVRO are supported.
    node_id_column: If provided, the node ID is exported as a column with this
      name. If not provided, for indexed formats (e.g., SSTable), the node ID is
      used as the native key, and for formats without native keys (e.g.,
      TFRecord), the node ID is exported as a feature named `"#id"`.
    edge_id_column: If provided, the edge ID is exported as a feature with this
      name.
    verbose: If true, display writing progress.
  """
  if isinstance(container_type, str):
    container_type = HGraphContainerType[container_type]
  extension = get_extension(container_type)
  tfgnn_schema = schema_to_tfgnn_schema(schema)

  _validate_in_memory_graph_to_hgraph_request(graph, container_type)

  if node_id_column is None:
    node_id_column = DEFAULT_KEY_ID
  if edge_id_column is None:
    edge_id_column = DEFAULT_KEY_ID

  filesystem.makedirs(path)
  proto_lib.write_text_proto(
      os.path.join(path, PATH_GRAPH_SCHEMA), tfgnn_schema
  )

  node_directory = os.path.join(path, PATH_NODE_FEATURE)
  edge_directory = os.path.join(path, PATH_EDGES)
  filesystem.makedirs(node_directory)
  filesystem.makedirs(edge_directory)

  if container_type == HGraphContainerType.TF_RECORD:
    _write_tfrecord_node_sets(
        graph, schema, node_directory, extension, node_id_column
    )
    _write_tfrecord_edge_sets(
        graph, schema, edge_directory, extension, node_id_column, edge_id_column
    )
  elif container_type == HGraphContainerType.AVRO:
    hgraph_in_avro.write_avro_node_sets(
        graph, schema, node_directory, extension, verbose
    )
    hgraph_in_avro.write_avro_edge_sets(
        graph,
        schema,
        edge_directory,
        extension,
        node_id_column,
        KEY_SOURCE,
        KEY_TARGET,
        verbose,
    )
