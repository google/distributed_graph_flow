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

"""Library for working with Google Cloud Native Graphs (Spanner and BigQuery)."""

from collections import defaultdict
import json
import os
import re
from typing import Any, Dict, Final, List, Literal, Tuple

from dgf.src.analyse import schema as schema_analyse_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import io_ext
from dgf.src.util import log
import numpy as np
import pandas as pd
import tqdm

_DGF_SCHEMA_ATTRIBUTE_FEATURE_NAME = "__attributes__"

GRAPH_ELEMENT_TYPE_NODE: Literal["NODE"] = "NODE"
GRAPH_ELEMENT_TYPE_EDGE: Literal["EDGE"] = "EDGE"
GRAPH_ELEMENT_PROPERTIES_KEY = "properties"

GRAPH_ELEMENT_ID_KEY = "id"
DGF_GRAPH_ELEMENT_ID_KEY = "_DGF_ID_"
GRAPH_ELEMENT_SOURCE_ID_KEY = "source_id"
DGF_GRAPH_ELEMENT_SOURCE_ID_KEY = "_DGF_SOURCE_"
GRAPH_ELEMENT_TARGET_ID_KEY = "target_id"
DGF_GRAPH_ELEMENT_TARGET_ID_KEY = "_DGF_TARGET_"
GRAPH_ELEMENT_JSON_KEY = "graph_element"

GCS_PREFIX_NODESETS = "nodesets"
GCS_PREFIX_EDGESETS = "edgesets"

TIMESTAMP_FEATURE_TYPE = "TIMESTAMP"

GqlFeatureType = str


def raw_type_to_feature_format(
    feature_name: str, feature_type: str
) -> Tuple[schema_lib.FeatureFormat, bool]:
  """Converts a Spanner type to (FeatureFormat, is_utf8_string).

  Args:
    spanner_type: A Spanner type.

  Returns:
    A tuple containing the FeatureFormat and a boolean indicating if it is a
    UTF-8 string.
  """

  feature_type = feature_type.strip().upper()

  ## TODO(goelshreya): Add support for other spanner types.
  match feature_type:
    case s if s.startswith("ARRAY"):
      # The format is expected to be "ARRAY<{data_type}>".
      # Slicing [6:-1] removes the prefix "ARRAY<"(length 6) and the suffix ">".
      data_type = feature_type[6:-1].strip()
      return raw_type_to_feature_format(
          f"{feature_name}'s sub-array", data_type
      )
    case s if s.startswith("PROTO"):
      return schema_lib.FeatureFormat.BYTES, False
    case "BOOL":
      return schema_lib.FeatureFormat.BOOL, False
    case "STRING" | "JSON":
      # Flag it as utf-8 bytes.
      return schema_lib.FeatureFormat.BYTES, True
    case "BYTES" | "DATE":
      return schema_lib.FeatureFormat.BYTES, False
    case "INT32":
      return schema_lib.FeatureFormat.INTEGER_32, False
    case "INT64" | "TIMESTAMP":
      return schema_lib.FeatureFormat.INTEGER_64, False
    case "FLOAT32":
      return schema_lib.FeatureFormat.FLOAT_32, False
    case "FLOAT64" | "NUMERIC":
      return schema_lib.FeatureFormat.FLOAT_64, False
    case "STRUCT":
      raise ValueError(
          f"The STRUCT type is not supported for column {feature_name!r}."
          " Convert it to an ARRAY first. For example:  `ARRAY(SELECT"
          " x.element FROM UNNEST(<table>.<column>.list) AS x)`"
      )
    case _:
      raise ValueError(
          f"Unsupported type {feature_type!r} for column {feature_name!r}"
      )


def is_semantic_timestamp(feature_type: GqlFeatureType) -> bool:
  """Returns true if the feature type is a timestamp."""
  return schema_lib.FeatureSemantic.TIMESTAMP.value in feature_type


def is_semantic_array(feature_type: GqlFeatureType) -> bool:
  """Returns true if the feature type is an array."""
  re_pattern = r"ARRAY<(\w+)>"
  match = re.match(re_pattern, feature_type)
  return match is not None


def is_semantic_timeseries(feature_type: GqlFeatureType) -> bool:
  """Returns true if the feature type is a timeseries."""
  re_pattern = r"ARRAY<(\w+)>"
  match = re.match(re_pattern, feature_type)
  if match:
    return is_semantic_timestamp(match.group(1))
  return False


def is_pk_fk_aligned(
    node_table_columns: List[str],
    node_table_key_columns: List[str],
) -> bool:
  """Returns true if the edge table columns are PK-FK aligned with the node table columns."""
  return node_table_columns == node_table_key_columns


def infer_feature_set_schema(
    graph_element_table: Dict[str, str],
    key_columns: List[str],
    combine_as_json: bool,
    skip_primary_keys: bool = False,
) -> schema_lib.FeatureSetSchema:
  """Converts a list of node/edge properties into a set of DGF feature schemas.

  The features are generated from the property definitions in the graph
  element table.

  Args:
    graph_element_table: The node/edge property graph element table.
    key_columns: A list of column names that form the primary key.
    combine_as_json: Whether to combine the features as JSON.

  Returns:
    A dictionary of feature names to feature schemas mapping for the given
    nodeset or edgeset.
  """
  features = {}
  if combine_as_json:
    features[_DGF_SCHEMA_ATTRIBUTE_FEATURE_NAME] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.UNKNOWN,
        shape=(),
        num_categorical_values=None,
    )
  else:
    # TODO(tewariy): Add a checks for graph_element_table. It can be empty

    create_primary_key = len(key_columns) > 1

    if create_primary_key and not skip_primary_keys:
      # Since DGF does not support multi-primary key, we generate a new feature
      # computed as the concatenation of all the primary keys.
      features["#id"] = schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
      )

    for feature_name, feature_type in graph_element_table.items():
      feature_format, _ = raw_type_to_feature_format(feature_name, feature_type)

      is_primary = feature_name in key_columns
      semantic = schema_lib.FeatureSemantic.UNKNOWN
      if is_primary:
        if skip_primary_keys:
          continue
        if not create_primary_key:
          semantic = schema_lib.FeatureSemantic.PRIMARY_ID
      elif is_semantic_timeseries(feature_type):
        semantic = schema_lib.FeatureSemantic.TIMESERIES
      elif is_semantic_timestamp(feature_type):
        semantic = schema_lib.FeatureSemantic.TIMESTAMP

      # Set shape to [None] for array features.
      if is_semantic_array(feature_type):
        shape = [None]
      else:
        shape = None

      features[feature_name] = schema_lib.FeatureSchema(
          format=feature_format, semantic=semantic, shape=shape
      )

  return features


def gql_base(
    graph_id: str,
    graph_element_type: str,
    graph_element_labels_string: str,
    graph_element_table_name: str,
) -> str:
  """Create a GQL graph query string to read graph elements (nodes or edges) from BQ Graph or Spanner Graph.

  Args:
    graph_id: The fully qualified graph id.
    graph_element_type: The type of the graph element (NODE or EDGE).
    graph_element_labels_string: The labels of the graph element.
    graph_element_table_name: The name of the graph element table.

  Returns:
    A BQ/Spanner Graph query string.
  """

  element_table_name_where_clause = (
      f"ELEMENT_DEFINITION_NAME(ge) = '{graph_element_table_name}'"
  )

  if graph_element_type == GRAPH_ELEMENT_TYPE_NODE:
    graph_element = f"(ge:{graph_element_labels_string})"
    return_clause = """
        ELEMENT_ID(ge) as id,
        TO_JSON_STRING(TO_JSON(ge)) as graph_element
    """
  elif graph_element_type == GRAPH_ELEMENT_TYPE_EDGE:
    graph_element = f"-[ge:{graph_element_labels_string}]->"
    return_clause = """
        ELEMENT_ID(ge) as id,
        SOURCE_NODE_ID(ge) as source_id,
        DESTINATION_NODE_ID(ge) as target_id,
        TO_JSON_STRING(TO_JSON(ge)) as graph_element
    """
    # TODO(tewariy): Expand to_json to individual elements. ??
  else:
    raise ValueError(
        "graph_element_type must be either 'NODE' or 'EDGE', "
        f"found {graph_element_type}"
    )

  return f"""
    GRAPH {graph_id}
      MATCH {graph_element}
      WHERE {element_table_name_where_clause}
    RETURN {return_clause}
  """


def parse_timestamp_to_micros(timestamp: str) -> int:
  """Converts a timestamp string into microseconds."""
  # TODO(tewariy): Switch to native python datetime parsing.
  return (
      pd.to_datetime(timestamp, utc=True)
      .to_numpy()
      .astype("datetime64[us]")
      .view(int)
  )


def parse_property_value_to_feature(
    feature_schema: schema_lib.FeatureSchema,
    property_value: Any = None,
) -> Any:
  """# TODO(tewariy): Add docstring."""
  feature_format = feature_schema.format

  if property_value is None:
    if feature_schema.shape == [None]:
      property_value = []  # missing array value.
    elif feature_format.is_integer():
      property_value = 0  # C++ default for global int.
    elif feature_format.is_float():
      property_value = 0.0  # C++ default for global float.
    elif feature_format == schema_lib.FeatureFormat.BOOL:
      property_value = False  # C++ default for global boolean.
    elif feature_format == schema_lib.FeatureFormat.BYTES:
      property_value = ""  # C++ default for global string.
    else:
      raise ValueError("Unsupported feature format: %s" % feature_schema)

  elif feature_schema.semantic == schema_lib.FeatureSemantic.TIMESTAMP:
    property_value = parse_timestamp_to_micros(property_value)

  elif feature_schema.semantic == schema_lib.FeatureSemantic.TIMESERIES:
    # dummy feature schema to handle missing values in timeseries.
    feature_ = schema_lib.FeatureSchema(
        format=feature_format,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=None,
        num_categorical_values=None,
    )
    property_value = [
        parse_property_value_to_feature(feature_, value)
        for value in property_value
    ]

  elif feature_schema.shape or feature_schema.shape == [None]:
    # dummy feature schema to handle missing values in array features.
    # TODO(tewariy): Add checks and unit tests for various shapes.
    feature_ = schema_lib.FeatureSchema(
        format=feature_format,
        semantic=feature_schema.semantic,
        shape=None,
        num_categorical_values=None,
    )
    property_value = [
        parse_property_value_to_feature(feature_, value)
        for value in property_value
    ]
  # if none of the above conditions are met, property_value is returned as is.
  return property_value


def graph_element_to_features(
    graph_element_name: str,
    graph_element_type: Literal[
        GRAPH_ELEMENT_TYPE_NODE, GRAPH_ELEMENT_TYPE_EDGE
    ],
    graph_element: Dict[str, Any],
    graph_schema: schema_lib.GraphSchema,
    combine_as_json: bool,
) -> in_memory_graph_lib.Features:
  """Returns a DGF Features from a GCP property graph element.

  Args:
    graph_element_name: The name of the graph element.
    graph_element_type: The type of the graph element (NODE or EDGE).
    graph_element: The GCP graph element record in the form of Dict[str, Any].
    graph_schema: The DGF schema of the graph.
    combine_as_json: Whether to combine the features as JSON.

  Returns:
    A DGF Features object.
  """
  features = {}
  # TODO(tewariy): Add checks for graph_element.
  if combine_as_json:
    # GCP Graph element json will be stored as a single feature in the DGF
    # Graph. It will be parsed into the actual features in the DGF
    # pipeline during model building.
    feature = np.array(json.dumps(graph_element).encode("utf-8"))
    features[_DGF_SCHEMA_ATTRIBUTE_FEATURE_NAME] = feature
  else:
    if GRAPH_ELEMENT_PROPERTIES_KEY in graph_element:
      if graph_element_type == GRAPH_ELEMENT_TYPE_NODE:
        element_schema = graph_schema.node_sets[graph_element_name]
      elif graph_element_type == GRAPH_ELEMENT_TYPE_EDGE:
        element_schema = graph_schema.edge_sets[graph_element_name]
      else:
        raise ValueError(
            "graph_element_type must be either 'NODE' or 'EDGE', "
            f"found {graph_element_type}"
        )

      feature_values = graph_element[GRAPH_ELEMENT_PROPERTIES_KEY]

      for feature_name, feature_schema in element_schema.features.items():
        if feature_name in feature_values:
          feature_value = feature_values[feature_name]
          feature_value = parse_property_value_to_feature(
              feature_schema, feature_value
          )
          # TODO(gbm): Don't return a numpy array. Instead, return the raw
          # python object.
          features[feature_name] = np.array(
              feature_value,
              dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
                  feature_schema.format
              ],
          )

  return features


def create_in_memory_node_set(
    nodeset_name: str,
    graph_schema: schema_lib.GraphSchema,
    query_results: List[Dict[str, Any]],
    combine_as_json: bool,
    verbose: int,
) -> Tuple[in_memory_graph_lib.InMemoryNodeSet, io_ext.ByteIdToIdxMapper]:
  """Returns a DGF InMemoryNodeSet from a GCP property graph element."""

  node_set_features = defaultdict(list)
  node_ids = []

  primary_key = None
  if not combine_as_json:
    nodeset_schema = graph_schema.node_sets[nodeset_name]
    primary_key = schema_analyse_lib.primary_feature(
        nodeset_name, nodeset_schema
    )

  # TODO(tewariy): Fix query_results type List / Iterator ?.
  iterator = query_results
  if verbose >= 1:
    iterator = tqdm.tqdm(iterator, desc=f"Loading nodes for {nodeset_name}")

  for node_row in iterator:
    element_id = np.array(node_row[GRAPH_ELEMENT_ID_KEY].encode("utf-8"))
    features = graph_element_to_features(
        nodeset_name,
        GRAPH_ELEMENT_TYPE_NODE,
        json.loads(node_row[GRAPH_ELEMENT_JSON_KEY]),
        graph_schema,
        combine_as_json,
    )
    if primary_key is not None and primary_key not in features:
      features[primary_key] = np.array(
          node_row[GRAPH_ELEMENT_ID_KEY].encode("utf-8"),
          dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
              schema_lib.FeatureFormat.BYTES
          ],
      )
    for feature_name, feature_value in features.items():
      node_set_features[feature_name].append(feature_value)
    node_ids.append(element_id)

  num_nodes = len(node_ids)

  for feature_name, feature_values in node_set_features.items():
    feature_schema = graph_schema.node_sets[nodeset_name].features[feature_name]

    if not feature_schema.is_static_shape():
      # Variable-length features must be stored as an object array.
      arr = np.empty(len(feature_values), dtype=np.object_)
      arr[:] = feature_values
      node_set_features[feature_name] = arr
    else:
      node_set_features[feature_name] = np.stack(
          feature_values,
          axis=0,
          dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
              feature_schema.format
          ],
      )

  if verbose >= 1:
    log.info(f"Loaded {num_nodes} nodes in nodeset {nodeset_name!r}")
  return (
      in_memory_graph_lib.InMemoryNodeSet(
          num_nodes=num_nodes,
          features=node_set_features,
      ),
      io_ext.ByteIdToIdxMapper(
          np.array(
              node_ids,
              dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
                  schema_lib.FeatureFormat.BYTES
              ],
          )
      ),
  )


def _optimized_adjacency(
    graph_element_name: str,
    source_mapper: io_ext.ByteIdToIdxMapper,
    target_mapper: io_ext.ByteIdToIdxMapper,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
) -> np.ndarray:
  """Use c++ optimized index mapper to create adjacency matrix."""

  adjacency, missmatch_src, missmatch_trg = io_ext.PairMapping(
      source_mapper,
      target_mapper,
      source_ids,
      target_ids,
      min(32, os.cpu_count()),
  )

  if missmatch_src != -1:
    bad_id = source_ids[missmatch_src].decode()
    raise ValueError(
        f"Source Node ID {bad_id!r} not found for edge set"
        f" {graph_element_name!r}."
    )
  if missmatch_trg != -1:
    bad_id = target_ids[missmatch_trg].decode()
    raise ValueError(
        f"Target Node ID {bad_id!r} not found for edge set"
        f" {graph_element_name!r}."
    )
  return adjacency


def create_in_memory_edge_set(
    edgeset_name: str,
    graph_schema: schema_lib.GraphSchema,
    query_results: List[Dict[str, Any]],
    source_node_id_index_map: io_ext.ByteIdToIdxMapper,
    target_node_id_index_map: io_ext.ByteIdToIdxMapper,
    combine_as_json: bool,
    verbose: int,
) -> in_memory_graph_lib.InMemoryEdgeSet:
  """Returns a DGF InMemoryEdgeSet from a GCP property graph element."""
  edge_set_features = defaultdict(list)

  source_ids = []
  target_ids = []

  # TODO(tewariy): Fix query_results type List / Iterator ?.
  iterator = query_results
  if verbose >= 1:
    iterator = tqdm.tqdm(iterator, desc=f"Loading edges for {edgeset_name}")
  for edge_row in iterator:
    source_ids.append(edge_row[GRAPH_ELEMENT_SOURCE_ID_KEY].encode("utf-8"))
    target_ids.append(edge_row[GRAPH_ELEMENT_TARGET_ID_KEY].encode("utf-8"))
    features = graph_element_to_features(
        edgeset_name,
        GRAPH_ELEMENT_TYPE_EDGE,
        json.loads(edge_row[GRAPH_ELEMENT_JSON_KEY]),
        graph_schema,
        combine_as_json,
    )
    for feature_name, feature_value in features.items():
      edge_set_features[feature_name].append(feature_value)

  num_edges = len(source_ids)

  for feature_name, feature_values in edge_set_features.items():
    feature_schema = graph_schema.edge_sets[edgeset_name].features[feature_name]

    if not feature_schema.is_static_shape():
      # Variable-length features must be stored as an object array.
      arr = np.empty(len(feature_values), dtype=np.object_)
      arr[:] = feature_values
      edge_set_features[feature_name] = arr
    else:
      edge_set_features[feature_name] = np.stack(
          feature_values,
          axis=0,
          dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
              feature_schema.format
          ],
      )

  if verbose >= 1:
    log.info(f"Loaded {num_edges} edges in the edgeset {edgeset_name!r}")

  return in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=_optimized_adjacency(
          edgeset_name,
          source_node_id_index_map,
          target_node_id_index_map,
          np.array(
              source_ids,
              dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
                  # TODO(gbm): Format from schema.
                  schema_lib.FeatureFormat.BYTES
              ],
          ),
          np.array(
              target_ids,
              dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[
                  # TODO(gbm): Format from schema.
                  schema_lib.FeatureFormat.BYTES
              ],
          ),
      ),
      features=edge_set_features,
  )
