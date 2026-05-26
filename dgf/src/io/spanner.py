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

"""Library for working with Cloud Spanner (Graph) databases."""

import enum
import functools
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, TypeVar

from absl import logging
import apache_beam as beam
from apache_beam import coders
import apache_beam.io.gcp.spanner as beam_spanner_io
from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from google.api_core import exceptions
from google.cloud import spanner as gcp_spanner
import numpy as np

# A default value for Spanner ID columns that can be overridden by the user.
# Don't use `#id` (TF-GNN convention) as a column name, spanner will treat the
# `#` as a comment and needlessly complicate queries. Node sets must have an
# `id` column. Edge sets can provide an `id` column optionally to support
# external edge features or hypergraphs.
_DEFAULT_ID_KEY = "id"

# Name of the id feature for both edgesets and nodesets.
_DEFAULT_ID_KEY_GNN = "#id"

_NAME_PREFIX_TABLE = "t"
_NAME_PREFIX_COLUMN = "c"

T = TypeVar("T")


class Precision(enum.Enum):
  SINGLE = 1
  DOUBLE = 2


def feature_format_to_spanner_type(
    feature_format: schema_lib.FeatureFormat,
    shape: Tuple[int, ...],
    is_utf8_string: bool = False,
    max_bytes_length: Optional[int] = None,
    float_precision: Precision = Precision.SINGLE,
    int_precision: Precision = Precision.DOUBLE,
) -> str:
  """Converts a FeatureFormat to a string for Spanner.

  We assume that float features in the schema map to ARRAY<FLOAT> (single
  precision) and integer features map to ARRAY<INT64> (double precision).

  Args:
    feature_format: A FeatureFormat.
    shape: Feature shape.
    is_utf8_string: Whether the feature is a UTF-8 string.
    max_bytes_length: The maximum length of a BYTES feature. If None, the
      maximum length is specified.
    float_precision: The precision of a FLOAT feature. Defaults to SINGLE.
    int_precision: The precision of an INTEGER feature. Defaults to DOUBLE.

  Returns:
    A string for Spanner.

  Raises:
    ValueError: If the feature format is not supported.
  """

  # go/spanner-schema#data-types
  match feature_format:
    case schema_lib.FeatureFormat.BYTES:
      if is_utf8_string:
        return "STRING(MAX)"
      else:
        return f"BYTES({max_bytes_length if max_bytes_length else 'MAX'})"
    case schema_lib.FeatureFormat.FLOAT_32 | schema_lib.FeatureFormat.FLOAT_64:
      if float_precision == Precision.SINGLE:
        base = "FLOAT32"
      elif float_precision == Precision.DOUBLE:
        base = "FLOAT64"
      else:
        raise ValueError(f"Unsupported float precision: {float_precision}")
    case schema_lib.FeatureFormat.INTEGER_64:
      if int_precision == Precision.SINGLE:
        base = "INT32"
      elif int_precision == Precision.DOUBLE:
        base = "INT64"
      else:
        raise ValueError(f"Unsupported int precision: {int_precision}")
    case schema_lib.FeatureFormat.BOOL:
      base = "BOOL"
    case _:
      raise ValueError(f"Unsupported feature format: {feature_format}")

  if shape is None or shape == ():
    return base
  else:
    return f"ARRAY<{base}>"


def feature_format_to_type_hint(
    feature_format: schema_lib.FeatureFormat,
    shape: Tuple[int, ...],
) -> type[Any]:
  """Converts a FeatureFormat to a type hint.

  Args:
    feature_format: A FeatureFormat.
    shape: Feature shape.

  Returns:
    A type hint.

  Raises:
    ValueError: If the feature format is not supported.
  """
  match feature_format:
    case schema_lib.FeatureFormat.INTEGER_64:
      base = int
    case schema_lib.FeatureFormat.FLOAT_32:
      base = float
    case schema_lib.FeatureFormat.BYTES:
      base = bytes
    case schema_lib.FeatureFormat.BOOL:
      base = bool
    case _:
      raise ValueError(f"Unsupported feature format: {feature_format}")

  if shape is None or shape == ():
    return base
  else:
    return List[base]


def sanitize_name(name, prefix):
  """Sanitizes a table or column name for Spanner."""
  sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
  if re.match(r"^[0-9_]", sanitized_name):
    sanitized_name = prefix + sanitized_name
  return sanitized_name


def schema_to_spanner_ddl(
    schema: schema_lib.GraphSchema,
    max_bytes_length: Optional[int] = None,
    enforce_foreign_keys: bool = False,
):
  r"""Converts a GraphSchema to a string of CREATE statements for Spanner.

  Useful for creating a Spanner database that matches the schema of a hgraph.

  Can use something like `\n.join(schema_to_spanner_ddl(schema).values())` to
  get a single string with all the CREATE statements.

  Assumptions:
    * An `#id` BYTES(MAX) column is added to each node set table an acts as the
      primary key for node features.
    * Edges are stored in a flat (source, target, features) format rather
      than adjacency list.
    * Edges will have a nullable `id` BYTES(MAX) column that is also used as the
      primary key. Hypergraphs can only be supported if all edge tuples have an
      associated `#id`.
    * Each edge must have a (source, target) specification - NOT NULL
    constraints are added to the table schema (DDL).


  Args:
    schema: A GraphSchema.
    max_bytes_length: Optional max byte length for BYTES columns. Defaults to
      "MAX".
    enforce_foreign_keys: Whether to enforce foreign keys in the edge tables. If
      True, the edge tables will have foreign key constraints. If False, the
      edge tables will not have foreign key constraints.

  Returns:
    A dictionary mapping node and edge set names to the corresponding Spanner
    CREATE TABLE statements.

  Raises:
    ValueError: If the max_str_length is not a valid value.
  """
  max_bytes_length = max_bytes_length if max_bytes_length else "MAX"

  ddl_statements: Dict[str, str] = {}

  for node_set_name, node_set_schema in schema.node_sets.items():
    node_set_name = sanitize_name(node_set_name, _NAME_PREFIX_TABLE)
    current_ddl_statement = f"CREATE TABLE {node_set_name} ("

    feature_ddls = []

    primary_id_feature_name = analyse_schema_lib.primary_feature(
        node_set_name, node_set_schema
    )
    sanitized_primary_id_feature_name = sanitize_name(
        primary_id_feature_name, _NAME_PREFIX_COLUMN
    )

    for feature_name, feature_schema in node_set_schema.features.items():
      is_id = feature_name == primary_id_feature_name
      ## Handle feature names with special characters and reserve words.
      feature_name = sanitize_name(feature_name, _NAME_PREFIX_COLUMN)

      sql_type = feature_format_to_spanner_type(
          feature_schema.format,
          feature_schema.shape,
          max_bytes_length=max_bytes_length,
          is_utf8_string=feature_schema.is_utf8_string,
      )
      if is_id:
        feature_ddls.append(f"{feature_name} {sql_type} NOT NULL")
      else:
        feature_ddls.append(f"{feature_name} {sql_type}")

    current_ddl_statement += ",".join(feature_ddls) + ")"
    current_ddl_statement += (
        f" PRIMARY KEY ({sanitized_primary_id_feature_name})"
    )
    print(
        f"current_ddl_statement for node set {node_set_name}:"
        f" {current_ddl_statement}"
    )
    ddl_statements[node_set_name] = current_ddl_statement
    print(
        f"ddl_statements for node set {node_set_name}:"
        f" {ddl_statements[node_set_name]}"
    )

  for edge_set_name, edge_set_schema in schema.edge_sets.items():
    edge_set_name = sanitize_name(edge_set_name, _NAME_PREFIX_TABLE)
    print(f"edge_set_name: {edge_set_name}")
    print(f"edge_set_schema: {edge_set_schema}")

    src_node_schema = schema.node_sets[edge_set_schema.source]
    src_primary_feature_name = analyse_schema_lib.primary_feature(
        edge_set_schema.source, src_node_schema
    )

    source_feature_id = src_node_schema.features[src_primary_feature_name]

    source_type = feature_format_to_spanner_type(
        source_feature_id.format,
        source_feature_id.shape,
        max_bytes_length=max_bytes_length,
        is_utf8_string=source_feature_id.is_utf8_string,
    )

    tgt_node_schema = schema.node_sets[edge_set_schema.target]
    tgt_primary_feature_name = analyse_schema_lib.primary_feature(
        edge_set_schema.target, tgt_node_schema
    )

    target_feature_id = tgt_node_schema.features[tgt_primary_feature_name]

    target_type = feature_format_to_spanner_type(
        target_feature_id.format,
        target_feature_id.shape,
        max_bytes_length=max_bytes_length,
        is_utf8_string=target_feature_id.is_utf8_string,
    )

    current_ddl_statement = (
        f"CREATE TABLE {edge_set_name} ("
        f"source {source_type} NOT NULL,"
        f"target {target_type} NOT NULL,"
    )

    feature_ddls = []

    try:
      edge_primary_feature_name = analyse_schema_lib.primary_feature(
          edge_set_name, edge_set_schema
      )
      edge_has_id = True
    except ValueError:
      edge_primary_feature_name = None
      edge_has_id = False

    if edge_set_schema.features:
      for feature_name, feature_schema in edge_set_schema.features.items():
        is_id = feature_name == edge_primary_feature_name
        feature_name = sanitize_name(feature_name, _NAME_PREFIX_COLUMN)

        sql_type = feature_format_to_spanner_type(
            feature_schema.format,
            feature_schema.shape,
            is_utf8_string=feature_schema.is_utf8_string,
        )
        if is_id:
          feature_ddls.append(f"  {feature_name} {sql_type} NOT NULL")
        else:
          feature_ddls.append(f"  {feature_name} {sql_type}")

    if enforce_foreign_keys:
      feature_ddls.append(
          f"CONSTRAINT fk_{edge_set_name}_source FOREIGN KEY(source) REFERENCES"
          f" {edge_set_schema.source}({sanitize_name(src_primary_feature_name, _NAME_PREFIX_COLUMN)})"
      )
      feature_ddls.append(
          f"CONSTRAINT fk_{edge_set_name}_target FOREIGN KEY(target) REFERENCES"
          f" {edge_set_schema.target}({sanitize_name(tgt_primary_feature_name, _NAME_PREFIX_COLUMN)})"
      )
    current_ddl_statement += ",".join(feature_ddls) + ")"

    if edge_has_id:
      sanitized_edge_primary_feature_name = sanitize_name(
          edge_primary_feature_name, _NAME_PREFIX_COLUMN
      )
      current_ddl_statement += (
          " PRIMARY KEY (source, target,"
          f" {sanitized_edge_primary_feature_name})"
      )
    else:
      current_ddl_statement += " PRIMARY KEY (source, target)"

    print(
        f"current_ddl_statement for edge set {edge_set_name}:"
        f" {current_ddl_statement}"
    )
    ddl_statements[edge_set_name] = current_ddl_statement

  return ddl_statements


def create_spanner_tables_from_graph_schema(
    schema: schema_lib.GraphSchema,
    project_id: str,
    instance_id: str,
    database_id: str,
    spanner_client: Optional[gcp_spanner.Client] = None,
    ddl_timeout_seconds: int = 30,
):
  """Creates Spanner tables for a graph schema.

  Args:
    schema: A GraphSchema.
    project_id: The GCP project ID.
    instance_id: Spanner instance ID.
    database_id: Spanner database ID.
    spanner_client: Optional Spanner client. If None, a new client is created.
    ddl_timeout_seconds: Timeout for DDL operation.
  """
  client = spanner_client
  if client is None:
    client = gcp_spanner.Client(project=project_id)

  database = client.instance(instance_id).database(database_id)
  if not database.exists():
    print("Destination Spanner Graph database does not exist. Creating one.")
    op = database.create()
    op.result(timeout=120)

  ddl_statements = schema_to_spanner_ddl(schema)
  for table_name, ddl in ddl_statements.items():
    try:
      operation = database.update_ddl([ddl])
      operation.result(timeout=ddl_timeout_seconds)
      logging.info("Created table: %s", table_name)
    except (exceptions.AlreadyExists, exceptions.FailedPrecondition):
      logging.warning("Table %s already exists. Skipping creation.", table_name)


# TODO(bmayer): Add support to make all features support either scalar or array.
def features_to_dict(features: distributed_graph.Features) -> Dict[str, Any]:
  """Converts a Features to a dictionary."""
  ret = {}
  for k, v in features.items():
    # We should always have an np.ndarray?
    assert isinstance(v, np.ndarray), "Features must be np.ndarrays."
    # Note: If the number array is a scalar, "tolist" returns a scalar.
    ret[k] = v.tolist()
  return ret


def node_to_spanner_row(
    node: distributed_graph.Node, cls: Type[T], id_key: str = _DEFAULT_ID_KEY
) -> T:
  """Converts a Node to a dictionary for SpannerSink.

  Given a dynamically generated row type hint (typically a NamedTuple class),
  return the instantiated row.

  Args:
    node: A Node.
    cls: The row type hint (NamedTuple type constructor).
    id_key: The name of the id column.

  Returns:
    An instance a `cls` object with the node data.
  """
  node_dict = features_to_dict(node.features)
  node_dict[id_key] = node.id
  return cls(**node_dict)


def write_node_set_to_spanner(
    nodes: distributed_graph.PNode,
    node_set_name: str,
    node_row_type: Type[NamedTuple],
    project_id: str,
    instance_id: str,
    database_id: str,
    table_id: str,
    id_key: str = _DEFAULT_ID_KEY,
    **kwargs,
) -> beam.pvalue.PDone:
  """Writes a node set to a Spanner table using SpannerInsertOrUpdate.

  Args:
    nodes: A PCollection of Nodes.
    node_set_name: The name of the node set to write. Only used to generate
      unique beam stage names.
    node_row_type: The Spanner row type (NamedTuple type constructor) for the
      node set. See `create_spanner_row_type_from_node_schema` for more details.
    project_id: The GCP project ID.
    instance_id: The Spanner instance ID.
    database_id: The Spanner database ID.
    table_id: The Spanner table name for the node set.
    id_key: The name of the id column. All node sets must have a value for the
      ID column. Defaults to `id`.
    **kwargs: Additional arguments to pass to
      `apache_beam.io.gcp.spanner.SpannerInsertOrUpdate`.

  Returns:
    A PDone.
  """

  return (
      nodes
      | f"NodesToSpannerRows_{node_set_name}"
      >> beam.Map(
          functools.partial(  # pytype: disable=wrong-arg-count
              node_to_spanner_row, cls=node_row_type, id_key=id_key
          )
      ).with_output_types(node_row_type)
      | f"WriteNodesToSpanner_{node_set_name}"
      >> beam_spanner_io.SpannerInsertOrUpdate(
          table=table_id,
          project_id=project_id,
          instance_id=instance_id,
          database_id=database_id,
          **kwargs,
      )
  )


def edge_to_spanner_row(
    edge: distributed_graph.Edge, cls: Type[T], id_key: str = _DEFAULT_ID_KEY
) -> T:
  """Converts an Edge to a dictionary for SpannerSink.

  Given an spanner row type (typically a NamedTuple class) instantiate
  the row representation for the `edge`.

  Args:
    edge: An Edge.
    cls: The row type hint (NamedTuple type constructor).
    id_key: The name of the id column.

  Returns:
    An instance of the `cls` object with the edge row data.
  """

  edge_dict = {}
  if edge.features is not None:
    edge_dict.update(features_to_dict(edge.features))

  if edge.id is not None:
    edge_dict[id_key] = edge.id

  edge_dict["source"] = edge.source
  edge_dict["target"] = edge.target

  return cls(**edge_dict)


def write_edge_set_to_spanner(
    edges: distributed_graph.PEdge,
    edge_set_name: str,
    edge_row_type: Type[NamedTuple],
    project_id: str,
    instance_id: str,
    database_id: str,
    table_id: str,
    id_key: str = _DEFAULT_ID_KEY,
    **kwargs,
) -> beam.pvalue.PDone:
  """Writes an edge set to a Spanner table using SpannerInsertOrUpdate.

  Args:
    edges: A PCollection of Edges.
    edge_set_name: The name of the edge set to write. Only used to generate
      unique beam stage names.
    edge_row_type: The Spanner row type (NamedTuple type constructor) for the
      edge set. See `create_spanner_row_type_from_edge_schema` for more details.
    project_id: The GCP project ID.
    instance_id: The Spanner instance ID.
    database_id: The Spanner database ID.
    table_id: The Spanner table name for the node set.
    id_key: The name of the id column. Will only be added if the EdgeSet
      contains an id value.
    **kwargs: Additional arguments to pass to
      `apache_beam.io.gcp.spanner.SpannerInsertOrUpdate`.

  Returns:
    A PDone.
  """
  return (
      edges
      | f"EdgesToSpannerRows_{edge_set_name}"
      >> beam.Map(
          functools.partial(  # pytype: disable=wrong-arg-count
              edge_to_spanner_row, cls=edge_row_type, id_key=id_key
          )
      ).with_output_types(edge_row_type)
      | f"WriteEdgesToSpanner_{edge_set_name}"
      >> beam_spanner_io.SpannerInsertOrUpdate(
          table=table_id,
          project_id=project_id,
          instance_id=instance_id,
          database_id=database_id,
          **kwargs,
      )
  )


class CreateSpannerTables(beam.DoFn):
  """Creates Spanner tables for a graph schema.

  This is meant to be used similarly to a `source` transform run on a
  PCollection with a single (dummy) element (see spanner_test.py for an
  example). This ensures that remote
  controllers can create tables by deferring work to the platform which should
  have credentials to create tables. This ParDo merely defers the work of
  `create_graph_tables_from_schema` a remote stage in the beam pipeline.

  **IF** Dataflow creates backup workers, the table creation call **might** be
  OK since we handle the TableAlreadyExists exception.
  """

  def __init__(
      self,
      schema: schema_lib.GraphSchema,
      project_id: str,
      instance_id: str,
      database_id: str,
      ddl_timeout_seconds: int = 30,
  ):
    """Initializes the CreateSpannerTables transform.

    Args:
      schema: A GraphSchema.
      project_id: The GCP project ID.
      instance_id: Spanner instance ID.
      database_id: Spanner database ID.
      ddl_timeout_seconds: Timeout for DDL operation.
    """
    self.schema = schema
    self.project_id = project_id
    self.instance_id = instance_id
    self.database_id = database_id
    self.ddl_timeout_seconds = ddl_timeout_seconds

  def process(self, element: Any):
    """Creates Spanner tables for a graph schema."""
    create_spanner_tables_from_graph_schema(
        schema=self.schema,
        project_id=self.project_id,
        instance_id=self.instance_id,
        database_id=self.database_id,
        ddl_timeout_seconds=self.ddl_timeout_seconds,
    )
    return element


def create_spanner_row_type_from_node_schema(
    node_set_name: str,
    node_schema: schema_lib.NodeSchema,
    id_column: str = _DEFAULT_ID_KEY,
    name_suffix: str = "NodeSet_SpannerRow",
) -> Type[NamedTuple]:
  """Creates a Spanner row type for a node schema.

  Uses the NamedTuple functional API to satisfy the Beam type hint requirement
  for Spanner API
  https://beam.apache.org/releases/pydoc/current/apache_beam.io.gcp.spanner.html.

  For node sets, the ID column (bytes) is required and defaults to `id`.

  Args:
    node_set_name: The name of the node set to write. Only used to generate
      unique beam stage names.
    node_schema: The node schema to use.
    id_column: The name of the id column.
    name_suffix: The suffix to append to the name of the NamedTuple.

  Returns:
    A NamedTuple representing the Spanner row type.
  """

  column_names_and_types = []
  for feature_name, feature_schema in node_schema.features.items():
    column_names_and_types.append((
        feature_name,
        feature_format_to_type_hint(
            feature_schema.format, feature_schema.shape
        ),
    ))
  return NamedTuple(f"{node_set_name}_{name_suffix}", column_names_and_types)


def create_spanner_row_type_from_edge_schema(
    edge_set_name: str,
    edge_schema: schema_lib.EdgeSchema,
    schema: schema_lib.GraphSchema,
    name_suffix: str = "EdgeSet_SpannerRow",
) -> type[NamedTuple]:
  """Creates a Spanner row type for an edge schema.

  Args:
    edge_set_name: The name of the edge set to write. Only used to generate
      unique beam stage names.
    edge_schema: The edge schema to use.
    name_suffix: The suffix to append to the name of the NamedTuple.

  Returns:
    A NamedTuple representing the Spanner row type.
  """

  source_feature_id = schema.node_sets[edge_schema.source].features[
      _DEFAULT_ID_KEY_GNN
  ]
  source_hint = feature_format_to_type_hint(
      source_feature_id.format, source_feature_id.shape
  )
  target_feature_id = schema.node_sets[edge_schema.target].features[
      _DEFAULT_ID_KEY_GNN
  ]
  target_hint = feature_format_to_type_hint(
      target_feature_id.format, target_feature_id.shape
  )
  column_names_and_types = [("source", source_hint), ("target", target_hint)]

  for feature_name, feature_schema in edge_schema.features.items():
    column_names_and_types.append((
        feature_name,
        feature_format_to_type_hint(
            feature_schema.format, feature_schema.shape
        ),
    ))
  return NamedTuple(f"{edge_set_name}_{name_suffix}", column_names_and_types)


def create_spanner_row_types_from_schema(
    schema: schema_lib.GraphSchema,
    node_set_name_suffix: str = "NodeSet_SpannerRow",
    edge_set_name_suffix: str = "EdgeSet_SpannerRow",
    register_row_coders: bool = True,
) -> Dict[str, Type[NamedTuple]]:
  """Creates a Spanner row type for a graph schema and maybe register coders.

  **NOTE** THIS MUST BE CALLED BEFORE THE DATAFLOW BEAM PIPELINE THAT USES
  SPANNER TRANSFORMS IS CREATED/RUN.

  Given a graph schema, dynamically generate the expected Spanner row type for
  each node/edge table as a NamedTuple class. The returned value maps the piece
  name to the row type which can be instaniated by calling the values of the
  return type (a NamedTuple class with type hints) with kw args (e.g., a
  **dict).

  Args:
    schema: A GraphSchema.
    node_set_name_suffix: The suffix to append to the name of the NamedTuple for
      node sets.
    edge_set_name_suffix: The suffix to append to the name of the NamedTuple for
      edge sets.
    register_row_coders: Whether to register the row coders. Defaults to True.

  Returns:
    A dictionary mapping node and edge set names to the corresponding NamedTuple
    Spanner row types and optionally registers the row coders.
  """
  row_types: Dict[str, Type[NamedTuple]] = {}
  for node_set_name, node_schema in schema.node_sets.items():
    row_types[node_set_name] = create_spanner_row_type_from_node_schema(
        node_set_name, node_schema, name_suffix=node_set_name_suffix
    )
  for edge_set_name, edge_schema in schema.edge_sets.items():
    row_types[edge_set_name] = create_spanner_row_type_from_edge_schema(
        edge_set_name, edge_schema, schema, name_suffix=edge_set_name_suffix
    )

  if register_row_coders:
    for row_type in row_types.values():
      coders.registry.register_coder(row_type, coders.RowCoder)

  return row_types


# TODO(bmayer): Add a unit-test if
# https://yaqs.corp.google.com/cloud/q/8465600717619462144 is resolved.
def write_spanner(
    graph: distributed_graph.Graph,
    spanner_row_types: Dict[str, Type[NamedTuple]],
    project_id: str,
    instance_id: str,
    database_id: str,
    create_tables: bool,
    ddl_timeout_seconds: int = 30,
    **kwargs,
) -> Dict[str, beam.pvalue.PDone]:
  """Writes a heterogeneous graph to Spanner.

  **NOTE** `create_spanner_row_types_from_schema` MUST BE CALLED ON THE DRIVER
  PRIOR TO PIPELINE CREATION IF THIS SINK IS USED!

  **NOTE** Spanner really doesn't like `#id` as a column name even though it is
  used in the TF-GNN library. Avoid it!

  Args:
    graph: A Graph.
    spanner_row_types: A dictionary mapping node and edge set names to the
      corresponding Spanner row types.
    project_id: The GCP project ID.
    instance_id: The Spanner instance ID.
    database_id: The Spanner database ID.
    create_tables: Whether to create the Spanner tables from the schema.
    ddl_timeout_seconds: Timeout for DDL operation.
    **kwargs: Additional arguments to pass to
      `apache_beam.io.gcp.spanner.SpannerInsertOrUpdate`. A useful extra
      argument is `expansion_service` to pass an expansion service address for
      testing.

  Returns:
    A dictionary mapping node and edge set names to the corresponding PDone
    objects.
  """

  if create_tables:
    create_spanner_tables_from_graph_schema(
        schema=graph.schema,
        project_id=project_id,
        instance_id=instance_id,
        database_id=database_id,
        ddl_timeout_seconds=ddl_timeout_seconds,
    )

  pdones: Dict[str, beam.pvalue.PDone] = {}
  for node_set_name, node_set in graph.node_sets.items():
    pdones[node_set_name] = write_node_set_to_spanner(
        node_set,
        node_set_name,
        spanner_row_types[node_set_name],
        project_id=project_id,
        instance_id=instance_id,
        database_id=database_id,
        table_id=node_set_name,
        **kwargs,
    )

  for edge_set_name, edge_set in graph.edge_sets.items():
    pdones[edge_set_name] = write_edge_set_to_spanner(
        edge_set,
        edge_set_name,
        spanner_row_types[edge_set_name],
        project_id=project_id,
        instance_id=instance_id,
        database_id=database_id,
        table_id=edge_set_name,
        **kwargs,
    )

  return pdones
