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

"""Utility to generate graph data for tests."""

import json
import os
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Tuple
from unittest import mock

from dgf.src.util import weak_dep

if TYPE_CHECKING:
  from tensorflow_gnn import proto as tf_gnn_proto

import bagz
from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.data import gf_metadata as gf_metadata_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import tf_in_memory_graph as tf_in_memory_graph_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.io import schema as schema_io_lib
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import spanner_graph_metadata as sgm
from dgf.src.util import filesystem
from dgf.src.util import proto as proto_lib
from dgf.src.util import shard as shard_lib
import fastavro
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf


parse_schema = fastavro.parse_schema

_ID_COLUMN_NAME = "_id"
_SOURCE_ID_COLUMN_NAME = "_source_id"
_TARGET_ID_COLUMN_NAME = "_target_id"
_SPANNER_GRAPH_ELEMENT_KEY = "graph_element"
DEFAULT_KEY_ID = "#id"

Node = distributed_graph_lib.Node
Edge = distributed_graph_lib.Edge


def _write_sharded_tfrecord_tfexample(
    directory: str,
    base_filename: str,
    shard_idx: int,
    num_shards: int,
    extension: str,
    examples_pbtxt: list[str],
):
  """Create a TFRecord sharded file, and write some tf examples."""
  filename = shard_lib.sharded_filename(
      base_filename, shard_idx, num_shards, extension
  )
  with tf.io.TFRecordWriter(
      os.path.join(directory, filename), options="GZIP"
  ) as writer:
    for example_pbtxt in examples_pbtxt:
      example = proto_lib.parse_text_proto(example_pbtxt, tf.train.Example)
      writer.write(example.SerializeToString())


def _write_sharded_avro(
    directory: str,
    base_filename: str,
    shard_idx: int,
    num_shards: int,
    extension: str,
    schema: Dict[str, Any],
    records: List[Dict[str, Any]],
):
  """Create an Avro sharded file, and write some records."""
  filename = shard_lib.sharded_filename(
      base_filename, shard_idx, num_shards, extension
  )
  with filesystem.open_write(
      os.path.join(directory, filename), binary=True
  ) as out_file:
    fastavro.writer(out_file, schema, records, codec="deflate")


def generate_schema(
    node_ids: bool = False,
    edge_ids: bool = False,
    semantic: bool = False,
    variable_length: bool = True,
    bytes_feature: bool = True,
) -> schema_lib.GraphSchema:
  """Generates a graph schema."""
  n1_features = {
      "f2": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          shape=(2,),  # Fixed size [2]
      ),
  }
  n2_features = {
      "f3": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          shape=None,  # Same as ()
      ),
      "f4": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          shape=(),  # Scalar
      ),
  }
  if bytes_feature:
    n1_features["f1"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        shape=(1,),  # Fixed size [1]
    )
  if variable_length:
    n2_features["f5"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        shape=(None,),  # Variable length
    )

  if semantic:
    if bytes_feature:
      n1_features["f1"].semantic = schema_lib.FeatureSemantic.CATEGORICAL
    n1_features["f2"].semantic = schema_lib.FeatureSemantic.EMBEDDING
    n2_features["f3"].semantic = schema_lib.FeatureSemantic.NUMERICAL
    n2_features["f4"].semantic = schema_lib.FeatureSemantic.NUMERICAL
    if variable_length:
      n2_features["f5"].semantic = schema_lib.FeatureSemantic.NUMERICAL

  if node_ids:
    n1_features["#id"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES
    )
    n2_features["#id"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64
    )
    if semantic:
      n1_features["#id"].semantic = schema_lib.FeatureSemantic.PRIMARY_ID
      n2_features["#id"].semantic = schema_lib.FeatureSemantic.PRIMARY_ID

  n1_schema = schema_lib.NodeSchema(features=n1_features)
  n2_schema = schema_lib.NodeSchema(features=n2_features)

  e1_schema = schema_lib.EdgeSchema(source="n1", target="n1")
  e2_schema = schema_lib.EdgeSchema(source="n1", target="n2")

  if edge_ids:
    e1_schema.features = {
        "#id": schema_lib.FeatureSchema(format=schema_lib.FeatureFormat.BYTES)
    }
    e2_schema.features = {
        "#id": schema_lib.FeatureSchema(format=schema_lib.FeatureFormat.BYTES)
    }
    if semantic:
      e1_schema.features["#id"].semantic = schema_lib.FeatureSemantic.PRIMARY_ID
      e2_schema.features["#id"].semantic = schema_lib.FeatureSemantic.PRIMARY_ID

  return schema_lib.GraphSchema(
      node_sets={
          "n1": n1_schema,
          "n2": n2_schema,
      },
      edge_sets={
          "e1": e1_schema,
          "e2": e2_schema,
      },
  )


def generate_tf_gnn_graph_schema(
    node_id: bool = False,
    edge_id: bool = False,
    variable_length: bool = True,
) -> "tf_gnn_proto.GraphSchema":
  """Generates a tf_gnn_proto.GraphSchema.

  Args:
    node_id: If true, adds a "#id" feature to each node set.
    edge_id: If true, adds a "#id" feature to each edge set.
    variable_length: Generate the f5 variable len feature.

  Returns:
    A tf_gnn_proto.GraphSchema.
  """

  node_n1_id_feature = (
      """
        features {
          key: "#id"
          value {
            dtype: DT_STRING
          }
        }
  """
      if node_id
      else ""
  )
  node_n2_id_feature = (
      """
        features {
          key: "#id"
          value {
            dtype: DT_INT64
          }
        }
  """
      if node_id
      else ""
  )

  edge_id_feature = (
      """
        features {
          key: "#id"
          value {
            dtype: DT_STRING
          }
        }
  """
      if edge_id
      else ""
  )

  if variable_length:
    f5_feature = """
        features {
          key: "f5"
          value {
            dtype: DT_INT64
            shape { dim { size: -1 } }
          }
        }
    """
  else:
    f5_feature = ""

  graph_schema_pbtxt = (
      """
    node_sets {
      key: "n1"
      value {
  """
      + node_n1_id_feature
      + """
        features {
          key: "f1"
          value {
            dtype: DT_STRING
            shape { dim { size: 1 } }
          }
        }
        features {
          key: "f2"
          value {
            dtype: DT_FLOAT
            shape { dim { size: 2 } }
          }
        }
      }
    }
    node_sets {
      key: "n2"
      value {
  """
      + node_n2_id_feature
      + """
        features {
          key: "f3"
          value {
            dtype: DT_INT64
          }
        }
        features {
          key: "f4"
          value {
            dtype: DT_INT64
            shape { }
          }
        }
        """
      + f5_feature
      + """
      }
    }
    edge_sets {
      key: "e1"
      value {
        source: "n1"
        target: "n1"
        """
      + edge_id_feature
      + """
      }
    }
    edge_sets {
      key: "e2"
      value {
        source: "n1"
        target: "n2"
        """
      + edge_id_feature
      + """
      }
    }
  """
  )

  tf_gnn_proto = weak_dep.import_tf_gnn_proto()
  graph_schema = proto_lib.parse_text_proto(
      graph_schema_pbtxt, tf_gnn_proto.GraphSchema
  )
  return graph_schema


def generate_gf_graph(
    path: str,
    edge_ids: bool,
    variable_length: bool = True,
    insert_dangling_edges: bool = False,
):
  """Genenerates a GF Graph on disk."""
  os.makedirs(os.path.join(path, "nodesets"), exist_ok=True)
  os.makedirs(os.path.join(path, "edgesets"), exist_ok=True)

  # Schema
  schema = generate_schema(
      node_ids=True,
      edge_ids=edge_ids,
      semantic=True,
      variable_length=variable_length,
  )
  schema_io_lib.write_schema(schema, os.path.join(path, "schema.json"))

  # Metadata
  metadata = gf_metadata_lib.GFGraphMetadata(version=0)
  with open(os.path.join(path, "metadata.json"), "w") as f:
    f.write(metadata.to_json(indent=2))

  # Node features
  pq.write_table(
      pa.Table.from_arrays(
          [
              pa.array([b"1", b"2"]),
              pa.array([[b"blue"], [b"red"]], type=pa.list_(pa.binary(), 1)),
              pa.array(
                  [[0.0, 1.0], [2.0, 3.0]], type=pa.list_(pa.float32(), 2)
              ),
          ],
          names=["#id", "f1", "f2"],
      ),
      os.path.join(path, "nodesets", "n1-00000-of-00001.parquet"),
  )
  # Note: We split the n2 node data in 2 shards.
  pq.write_table(
      pa.Table.from_arrays(
          [
              pa.array([1]),
              pa.array([4]),
              pa.array([10]),
          ]
          + ([pa.array([[11, 12]])] if variable_length else []),
          names=["#id", "f3", "f4"] + (["f5"] if variable_length else []),
      ),
      os.path.join(path, "nodesets", "n2-00000-of-00002.parquet"),
  )
  pq.write_table(
      pa.Table.from_arrays(
          [
              pa.array([2]),
              pa.array([5]),
              pa.array([11]),
          ]
          + ([pa.array([[12, 13, 14]])] if variable_length else []),
          names=["#id", "f3", "f4"] + (["f5"] if variable_length else []),
      ),
      os.path.join(path, "nodesets", "n2-00001-of-00002.parquet"),
  )

  # Edge adjacencies
  pq.write_table(
      pa.Table.from_arrays(
          [
              pa.array([b"1", b"1"]),
              pa.array([b"1", b"2"]),
          ]
          + (
              [
                  pa.array([b"a", b"b"]),
              ]
              if edge_ids
              else []
          ),
          names=["#source", "#target"] + (["#id"] if edge_ids else []),
      ),
      os.path.join(path, "edgesets", "e1-00000-of-00001.parquet"),
  )

  second_source_id = b"missing" if insert_dangling_edges else b"1"

  pq.write_table(
      pa.Table.from_arrays(
          [
              pa.array([b"1", second_source_id]),
              pa.array([1, 2]),
          ]
          + (
              [
                  pa.array([b"A", b"B"]),
              ]
              if edge_ids
              else []
          ),
          names=["#source", "#target"] + (["#id"] if edge_ids else []),
      ),
      os.path.join(path, "edgesets", "e2-00000-of-00001.parquet"),
  )


def generate_hgraph(
    path: str,
    node_id: bool = False,
    edge_id: bool = False,
    variable_length: bool = True,
):
  """Generates a TFRecord based HGraph on disk.

  Follow the same schema as "generate_schema".

  Args:
    path: Path to generated hgraph directory.
    node_id: Create a node id in the schema.
    edge_id: Create a edge id in the schema.
    variable_length: Generate the f5 variable len feature.
  """

  os.makedirs(os.path.join(path, "node_features"), exist_ok=True)
  os.makedirs(os.path.join(path, "edges"), exist_ok=True)

  graph_schema = generate_tf_gnn_graph_schema(
      node_id=node_id, edge_id=edge_id, variable_length=variable_length
  )

  # Graph schema
  proto_lib.write_text_proto(
      os.path.join(path, "graph_schema.pbtxt"), graph_schema
  )

  # Node n1
  _write_sharded_tfrecord_tfexample(
      directory=os.path.join(path, "node_features"),
      base_filename="n1",
      shard_idx=0,
      num_shards=2,
      extension=".tfrecord.gz",
      examples_pbtxt=["""
      features {
        feature {
          key: "#id"
          value {
            bytes_list {
              value: "1"
            }
          }
        }
        feature {
          key: "f1"
          value {
            bytes_list {
              value: "blue"
            }
          }
        }
        feature {
          key: "f2"
          value {
            float_list {
              value: 0.0
              value: 1.0
            }
          }
        }
      }
    """],
  )
  _write_sharded_tfrecord_tfexample(
      directory=os.path.join(path, "node_features"),
      base_filename="n1",
      shard_idx=1,
      num_shards=2,
      extension=".tfrecord.gz",
      examples_pbtxt=["""
      features {
        feature {
          key: "#id"
          value {
            bytes_list {
              value: "2"
            }
          }
        }
        feature {
          key: "f1"
          value {
            bytes_list {
              value: "red"
            }
          }
        }
        feature {
          key: "f2"
          value {
            float_list {
              value: 2.0
              value: 3.0
            }
          }
        }
      }
    """],
  )

  # Node n2

  if variable_length:
    f5_1 = """
    feature {
          key: "f5"
          value {
            int64_list {
            value: 11
            value: 12
            }
          }
        }
    """
    f5_2 = """
    feature {
          key: "f5"
          value {
            int64_list {
              value: 12
              value: 13
              value: 14
            }
          }
        }
    """
  else:
    f5_1 = ""
    f5_2 = ""

  _write_sharded_tfrecord_tfexample(
      directory=os.path.join(path, "node_features"),
      base_filename="n2",
      shard_idx=0,
      num_shards=1,
      extension=".tfrecord.gz",
      examples_pbtxt=[
          """
      features {
        feature {
          key: "#id"
          value {
            int64_list {
              value: 1
            }
          }
        }
        feature {
          key: "f3"
          value {
            int64_list {
            value: 4
            }
          }
        }
        feature {
          key: "f4"
          value {
            int64_list {
            value: 10
            }
          }
        }
        """
          + f5_1
          + """
      }
    """,
          """
      features {
        feature {
          key: "#id"
          value {
            int64_list {
              value: 2
            }
          }
        }
        feature {
          key: "f3"
          value {
            int64_list {
              value: 5
            }
          }
        }
        feature {
          key: "f4"
          value {
            int64_list {
              value: 11
            }
          }
        }
        """
          + f5_2
          + """
      }
    """,
      ],
  )

  # Edge e1
  _write_sharded_tfrecord_tfexample(
      directory=os.path.join(path, "edges"),
      base_filename="e1",
      shard_idx=0,
      num_shards=1,
      extension=".tfrecord.gz",
      examples_pbtxt=[
          """
          features {
            feature {
              key: "#id"
              value {
                bytes_list {
                  value: "a"
                }
              }
            }
            feature {
              key: "#source"
              value {
                bytes_list {
                  value: "1"
                }
              }
            }
            feature {
              key: "#target"
              value {
                bytes_list {
                  value: "1"
                }
              }
            }
          }
    """,
          """
          features {
            feature {
              key: "#id"
              value {
                bytes_list {
                  value: "b"
                }
              }
            }
            feature {
              key: "#source"
              value {
                bytes_list {
                  value: "1"
                }
              }
            }
            feature {
              key: "#target"
              value {
                bytes_list {
                  value: "2"
                }
              }
            }
          }
    """,
      ],
  )

  # Edge e2
  _write_sharded_tfrecord_tfexample(
      directory=os.path.join(path, "edges"),
      base_filename="e2",
      shard_idx=0,
      num_shards=1,
      extension=".tfrecord.gz",
      examples_pbtxt=[
          """
          features {
            feature {
              key: "#source"
              value {
                bytes_list {
                  value: "1"
                }
              }
            }
            feature {
              key: "#target"
              value {
                int64_list {
                  value: 1
                }
              }
            }
          }
    """,
          """
          features {
            feature {
              key: "#source"
              value {
                bytes_list {
                  value: "1"
                }
              }
            }
            feature {
              key: "#target"
              value {
                int64_list {
                  value: 2
                }
              }
            }
          }
    """,
      ],
  )


def generate_avro_graph(
    path: str,
    node_id: bool = False,
    edge_id: bool = False,
    variable_length: bool = True,
):
  """Generates an Avro based HGraph on disk.

  Follow the same schema as "generate_schema".

  Args:
    path: Path to generated avro graph directory.
    node_id: Create a node id in the schema.
    edge_id: Create a edge id in the schema.
    variable_length: Generate the f5 variable len feature.
  """

  os.makedirs(os.path.join(path, "node_features"), exist_ok=True)
  os.makedirs(os.path.join(path, "edges"), exist_ok=True)

  graph_schema = generate_tf_gnn_graph_schema(
      node_id=node_id, edge_id=edge_id, variable_length=variable_length
  )

  proto_lib.write_text_proto(
      os.path.join(path, "graph_schema.pbtxt"), graph_schema
  )

  # Node n1
  n1_schema_dict = {
      "type": "record",
      "name": "n1",
      "fields": [
          {"name": "#id", "type": "bytes"},
          {"name": "f1", "type": {"type": "array", "items": "bytes"}},
          {"name": "f2", "type": {"type": "array", "items": "float"}},
      ],
  }
  n1_schema = parse_schema(n1_schema_dict)
  _write_sharded_avro(
      directory=os.path.join(path, "node_features"),
      base_filename="n1",
      shard_idx=0,
      num_shards=2,
      extension=".avro",
      schema=n1_schema,
      records=[{
          "#id": b"1",
          "f1": [b"blue"],
          "f2": [0.0, 1.0],
      }],
  )
  _write_sharded_avro(
      directory=os.path.join(path, "node_features"),
      base_filename="n1",
      shard_idx=1,
      num_shards=2,
      extension=".avro",
      schema=n1_schema,
      records=[{
          "#id": b"2",
          "f1": [b"red"],
          "f2": [2.0, 3.0],
      }],
  )

  # Node n2
  n2_fields = [
      {"name": "#id", "type": "long"},
      {"name": "f3", "type": "long"},
      {"name": "f4", "type": "long"},
  ]
  if variable_length:
    n2_fields.append({"name": "f5", "type": {"type": "array", "items": "long"}})
  n2_schema_dict = {"type": "record", "name": "n2", "fields": n2_fields}
  n2_schema = parse_schema(n2_schema_dict)

  n2_record1 = {"#id": 1, "f3": 4, "f4": 10}
  if variable_length:
    n2_record1["f5"] = [11, 12]

  n2_record2 = {"#id": 2, "f3": 5, "f4": 11}
  if variable_length:
    n2_record2["f5"] = [12, 13, 14]

  _write_sharded_avro(
      directory=os.path.join(path, "node_features"),
      base_filename="n2",
      shard_idx=0,
      num_shards=1,
      extension=".avro",
      schema=n2_schema,
      records=[n2_record1, n2_record2],
  )

  # Edge e1
  e1_schema_dict = {
      "type": "record",
      "name": "e1",
      "fields": [
          {"name": "#id", "type": "bytes"},
          {"name": "#source", "type": "bytes"},
          {"name": "#target", "type": "bytes"},
      ],
  }
  e1_schema = parse_schema(e1_schema_dict)

  _write_sharded_avro(
      directory=os.path.join(path, "edges"),
      base_filename="e1",
      shard_idx=0,
      num_shards=1,
      extension=".avro",
      schema=e1_schema,
      records=[
          {"#id": b"a", "#source": b"1", "#target": b"1"},
          {"#id": b"b", "#source": b"1", "#target": b"2"},
      ],
  )

  # Edge e2
  e2_schema_dict = {
      "type": "record",
      "name": "e2",
      "fields": [
          {"name": "#source", "type": "bytes"},
          {"name": "#target", "type": "long"},
      ],
  }
  e2_schema = parse_schema(e2_schema_dict)
  _write_sharded_avro(
      directory=os.path.join(path, "edges"),
      base_filename="e2",
      shard_idx=0,
      num_shards=1,
      extension=".avro",
      schema=e2_schema,
      records=[
          {"#source": b"1", "#target": 1},
          {"#source": b"1", "#target": 2},
      ],
  )


def generate_tf_graph_sample(
    node_ids: bool, edge_ids: bool, variable_length: bool = False
) -> tf.train.Example:
  """Generates a tf.train.Example graph sample.

  The generate graph sample does not have the variable-length f5 feature.

  Args:
    node_ids: if true, adds the "#id" features for nodes.
    edge_ids: if true, adds the "#id" features for edges "e1" (but not "e2").
    variable_length: Generate a variable length feature f5.

  Returns:
    A tf.train.Example graph sample.
  """

  features = ""
  if node_ids:
    features += """
        feature {
          key: "nodes/n1.#id"
          value {
            bytes_list {
              value: "1"
              value: "2"
            }
          }
        }
        feature {
          key: "nodes/n2.#id"
          value {
            int64_list {
              value: 1
              value: 2
            }
          }
        }
    """
  if edge_ids:
    features += """
        feature {
          key: "edges/e1.#id"
          value {
            bytes_list {
              value: "a"
              value: "b"
            }
          }
        }
        feature {
          key: "edges/e2.#id"
          value {
            bytes_list {
              value: "A"
              value: "B"
            }
          }
        }
    """
  features += """
    feature {
          key: "nodes/n1.#size"
          value {
            int64_list {
              value: 2
            }
          }
        }
        feature {
          key: "nodes/n1.f1"
          value {
            bytes_list {
              value: "blue"
              value: "red"
            }
          }
        }
        feature {
          key: "nodes/n1.f2"
          value {
            float_list {
              value: 0.0
              value: 1.0
              value: 2.0
              value: 3.0
            }
          }
        }
        feature {
          key: "nodes/n2.#size"
          value {
            int64_list {
              value: 2
            }
          }
        }
        feature {
          key: "nodes/n2.f3"
          value {
            int64_list {
              value: 4
              value: 5
            }
          }
        }
        feature {
          key: "nodes/n2.f4"
          value {
            int64_list {
              value: 10
              value: 11
            }
          }
        }
        feature {
          key: "edges/e1.#size"
          value {
            int64_list {
              value: 2
            }
          }
        }
        feature {
          key: "edges/e1.#source"
          value {
            int64_list {
              value: 0
              value: 0
            }
          }
        }
        feature {
          key: "edges/e1.#target"
          value {
            int64_list {
              value: 0
              value: 1
            }
          }
        }
        feature {
          key: "edges/e2.#size"
          value {
            int64_list {
              value: 2
            }
          }
        }
        feature {
          key: "edges/e2.#source"
          value {
            int64_list {
              value: 0
              value: 0
            }
          }
        }
        feature {
          key: "edges/e2.#target"
          value {
            int64_list {
              value: 0
              value: 1
            }
          }
        }
        """

  if variable_length:
    features += """
  feature {
    key: "nodes/n2.f5"
    value {
      int64_list {
        value: 11
        value: 12
        value: 12
        value: 13
        value: 14
      }
    }
  }
  feature {
    key: "nodes/n2.f5.d1"
    value {
      int64_list {
        value: 2
        value: 3
      }
    }
  }
        """

  base_pbtxt = f"""
  features {{
    {features}
  }}
  """
  sample = proto_lib.parse_text_proto(base_pbtxt, tf.train.Example)
  return sample


def generate_tf_graph_sample_in_tf_record(
    path: str,
    node_ids: bool = False,
    edge_ids: bool = False,
    container_type: Literal["TF_RECORD", "BAGZ"] = "TF_RECORD",
    variable_length: bool = False,
):
  """Generates a set of graph samples in a tf record."""
  sample = generate_tf_graph_sample(
      node_ids, edge_ids, variable_length=variable_length
  )
  if container_type == "TF_RECORD":
    with tf.io.TFRecordWriter(path, options="GZIP") as writer:
      for _ in range(2):
        writer.write(sample.SerializeToString())
  elif container_type == "BAGZ":
    with bagz.Writer(path) as writer:
      for _ in range(2):
        writer.write(sample.SerializeToString())
  else:
    raise ValueError(f"Unknown container_type: {container_type}")


def generate_tf_graph_sample_dict(
    node_ids: bool,
    edge_ids: bool,
) -> Dict[str, np.ndarray]:
  """Generates a dict of numpy arrays representing a graph sample.

  Args:
    node_ids: if true, adds the "#id" features for nodes.
    edge_ids: if true, adds the "#id" features for edges.

  Returns:
    A dict of numpy arrays representing a graph sample.
  """
  graph_dict = {
      "nodes/n1.#size": np.array([2], dtype=np.int64),
      "nodes/n1.f1": np.array([[b"blue"], [b"red"]], dtype=np.object_),
      "nodes/n1.f2": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
      "nodes/n2.#size": np.array([2], dtype=np.int64),
      "nodes/n2.f3": np.array([4, 5], dtype=np.int32),
      "nodes/n2.f4": np.array([10, 11], dtype=np.int32),
      "edges/e1.#size": np.array([2], dtype=np.int64),
      "edges/e1.#source": np.array([0, 0], dtype=np.int64),
      "edges/e1.#target": np.array([0, 1], dtype=np.int64),
      "edges/e2.#size": np.array([2], dtype=np.int64),
      "edges/e2.#source": np.array([0, 0], dtype=np.int64),
      "edges/e2.#target": np.array([0, 1], dtype=np.int64),
  }
  if node_ids:
    graph_dict["nodes/n1.#id"] = np.array([b"1", b"2"], dtype=np.object_)
    graph_dict["nodes/n2.#id"] = np.array([1, 2], dtype=np.int64)
  if edge_ids:
    graph_dict["edges/e1.#id"] = np.array([b"a", b"b"], dtype=np.object_)
    graph_dict["edges/e2.#id"] = np.array([b"A", b"B"], dtype=np.object_)
  return graph_dict


def generate_in_memory_graph(
    node_ids: bool = False,
    edge_ids: bool = False,
    variable_length: bool = True,
    bytes_feature: bool = True,
    if_spanner_graph: bool = False,
) -> in_memory_graph_lib.InMemoryGraph:
  """Generates an in-memory graph.

  Does not generate the f5 feature.

  Args:
    node_ids: if true, adds the "#id" features for nodes.
    edge_ids: if true, adds the "#id" features for edges.
    variable_length: Generate the f5 variable len feature.
    bytes_feature: Generate the f1 bytes feature.
    if_spanner_graph: If true, memory graph is generated assuming it is coming
      from a Spanner graph.

  Returns:
    An in-memory graph.
  """
  n1_features = {
      "f2": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
  }
  n2_features = {
      "f3": np.array([4, 5], dtype=np.int64),
      "f4": np.array([10, 11], dtype=np.int64),
  }
  if bytes_feature:
    n1_features["f1"] = np.array([[b"blue"], [b"red"]])
  if node_ids:
    n1_features["#id"] = np.array([b"1", b"2"])
    n2_features["#id"] = np.array([1, 2])
  if variable_length:
    n2_features["f5"] = np.array(
        [np.array([11, 12]), np.array([12, 13, 14])], dtype=np.object_
    )

  e1_features = {}
  e2_features = {}
  if edge_ids:
    e1_features["#id"] = np.array([b"a", b"b"])
    e2_features["#id"] = np.array([b"A", b"B"])

  if if_spanner_graph:
    # Add source and target id features to the edge sets.
    e1_features["source_id"] = np.array([b"1", b"1"])
    e1_features["target_id"] = np.array([b"1", b"2"])
    e2_features["source_id"] = np.array([b"1", b"1"])
    e2_features["target_id"] = np.array([b"1", b"2"])

  return in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "n1": in_memory_graph_lib.InMemoryNodeSet(
              features=n1_features,
              num_nodes=2,
          ),
          "n2": in_memory_graph_lib.InMemoryNodeSet(
              features=n2_features,
              num_nodes=2,
          ),
      },
      edge_sets={
          "e2": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.array([[0, 0], [0, 1]]), features=e2_features
          ),
          "e1": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.array([[0, 0], [0, 1]]), features=e1_features
          ),
      },
  )


def generate_tf_in_memory_graph(
    variable_length: bool,
    tensor_type: Literal["DENSE", "SPARSE", "RAGGED"],
    num_nodes_as_tensor: bool,
    node_ids: bool = False,
    edge_ids: bool = False,
) -> tf_in_memory_graph_lib.TFInMemoryGraph:
  """Generates a TF in-memory graph.

  Does not generate the f5 feature if variable_length is False.

  Args:
    node_ids: bool = if true, adds the "#id" features for nodes.
    edge_ids: if true, adds the "#id" features for edges.
    variable_length: Generate the f5 variable len feature.
    tensor_type: The tensor type for all features and adjacencies.
    num_nodes_as_tensor: If true, num_nodes is a tf.Tensor. Otherwise, an int.

  Returns:
    A TF in-memory graph.
  """

  def _convert(tensor):
    if tensor_type == "DENSE":
      if isinstance(tensor, tf.RaggedTensor):
        return tensor.to_tensor()
      return tensor
    elif tensor_type == "SPARSE":
      if isinstance(tensor, tf.RaggedTensor):
        return tensor.to_sparse()
      return tf.sparse.from_dense(tensor)
    elif tensor_type == "RAGGED":
      if isinstance(tensor, tf.RaggedTensor):
        return tensor
      if len(tensor.shape) > 1:
        return tf.RaggedTensor.from_tensor(tensor)
      return tensor
    else:
      raise ValueError(f"Unknown tensor_type: {tensor_type}")

  n1_features = {
      "f1": _convert(tf.constant([[b"blue"], [b"red"]], dtype=tf.string)),
      "f2": _convert(tf.constant([[0.0, 1.0], [2.0, 3.0]], dtype=tf.float32)),
  }
  n2_features = {
      "f3": _convert(tf.constant([4, 5], dtype=tf.int64)),
      "f4": _convert(tf.constant([10, 11], dtype=tf.int64)),
  }
  if node_ids:
    n1_features["#id"] = _convert(tf.constant([b"1", b"2"], dtype=tf.string))
    n2_features["#id"] = _convert(tf.constant([1, 2], dtype=tf.int64))
  if variable_length:
    n2_features["f5"] = _convert(
        tf.ragged.constant([[11, 12], [12, 13, 14]], dtype=tf.int64)
    )

  e1_features = {}
  e2_features = {}
  if edge_ids:
    e1_features["#id"] = _convert(tf.constant([b"a", b"b"], dtype=tf.string))
    e2_features["#id"] = _convert(tf.constant([b"A", b"B"], dtype=tf.string))

  num_nodes = tf.constant(2, dtype=tf.int32) if num_nodes_as_tensor else 2

  return tf_in_memory_graph_lib.TFInMemoryGraph(
      node_sets={
          "n1": tf_in_memory_graph_lib.TFInMemoryNodeSet(
              features=n1_features, num_nodes=num_nodes
          ),
          "n2": tf_in_memory_graph_lib.TFInMemoryNodeSet(
              features=n2_features, num_nodes=num_nodes
          ),
      },
      edge_sets={
          "e2": tf_in_memory_graph_lib.TFInMemoryEdgeSet(
              adjacency=_convert(tf.constant([[0, 0], [0, 1]], dtype=tf.int64)),
              features=e2_features,
          ),
          "e1": tf_in_memory_graph_lib.TFInMemoryEdgeSet(
              adjacency=_convert(tf.constant([[0, 0], [0, 1]], dtype=tf.int64)),
              features=e1_features,
          ),
      },
  )


def generate_graph_pieces(
    schema_node_ids: bool = True, schema_edge_ids: bool = True
) -> Tuple[
    Dict[str, List[Node]], Dict[str, List[Edge]], schema_lib.GraphSchema
]:
  """Generates some node sets, edge sets and a schema.

  Returns:
    A tuple of (node sets, edge sets, schema).
  """
  node_sets = {
      "n1": [
          Node(
              id=b"1",
              features={
                  "f1": np.array([b"blue"]),
                  "f2": np.array([0.0, 1.0], dtype=np.float32),
              },
          ),
          Node(
              id=b"2",
              features={
                  "f1": np.array([b"red"]),
                  "f2": np.array([2.0, 3.0], dtype=np.float32),
              },
          ),
      ],
      "n2": [
          Node(
              id=b"a",
              features={
                  "f3": np.array(4, dtype=np.int64),
                  "f4": np.array(10, dtype=np.int64),
                  "f5": np.array([11, 12], dtype=np.int64),
              },
          ),
          Node(
              id=b"b",
              features={
                  "f3": np.array(5, dtype=np.int64),
                  "f4": np.array(11, dtype=np.int64),
                  "f5": np.array([12, 13, 14], dtype=np.int64),
              },
          ),
      ],
  }

  edge_sets = {
      "e1": [
          Edge(source=b"1", target=b"1"),
          Edge(source=b"1", target=b"2"),
      ],
      "e2": [
          Edge(source=b"a", target=1),
          Edge(source=b"b", target=2),
      ],
  }

  return (
      node_sets,
      edge_sets,
      generate_schema(node_ids=schema_node_ids, edge_ids=schema_edge_ids),
  )


def get_spanner_graph_metadata_dict() -> Dict[str, Any]:
  """Returns the Spanner Graph metadata as a dictionary."""
  return {
      "catalog": "",
      "name": "SpannerGraph",
      "propertyDeclarations": [
          {"name": "f1", "type": "ARRAY<STRING>"},
          {"name": "f2", "type": "ARRAY<FLOAT64>"},
          {"name": "f3", "type": "INT64"},
          {"name": "f4", "type": "INT64"},
      ],
      "nodeTables": [
          {
              "baseCatalogName": "",
              "baseSchemaName": "",
              "baseTableName": "n1",
              "name": "n1",
              "kind": "NODE",
              "labelNames": ["n1"],
              "keyColumns": ["id"],
              "propertyDefinitions": [
                  {"propertyDeclarationName": "f1", "valueExpressionSql": "f1"},
                  {"propertyDeclarationName": "f2", "valueExpressionSql": "f2"},
                  {"propertyDeclarationName": "id", "valueExpressionSql": "id"},
              ],
          },
          {
              "baseCatalogName": "",
              "baseSchemaName": "",
              "baseTableName": "n2",
              "name": "n2",
              "kind": "NODE",
              "labelNames": ["n2"],
              "keyColumns": ["id"],
              "propertyDefinitions": [
                  {"propertyDeclarationName": "f3", "valueExpressionSql": "f3"},
                  {"propertyDeclarationName": "f4", "valueExpressionSql": "f4"},
                  {"propertyDeclarationName": "id", "valueExpressionSql": "id"},
              ],
          },
      ],
      "edgeTables": [
          {
              "baseCatalogName": "",
              "baseSchemaName": "",
              "baseTableName": "e1",
              "name": "e1",
              "kind": "EDGE",
              "labelNames": ["e1"],
              "keyColumns": ["id", "source_id", "target_id"],
              "sourceNodeTable": {
                  "nodeTableName": "n1",
                  "edgeTableColumns": ["source_id"],
                  "nodeTableColumns": ["id"],
              },
              "destinationNodeTable": {
                  "nodeTableName": "n1",
                  "edgeTableColumns": ["target_id"],
                  "nodeTableColumns": ["id"],
              },
              "propertyDefinitions": [
                  {"propertyDeclarationName": "id", "valueExpressionSql": "id"}
              ],
          },
          {
              "baseCatalogName": "",
              "baseSchemaName": "",
              "baseTableName": "e2",
              "name": "e2",
              "kind": "EDGE",
              "labelNames": ["e2"],
              "keyColumns": ["id", "source_id", "target_id"],
              "sourceNodeTable": {
                  "nodeTableName": "n1",
                  "edgeTableColumns": ["source_id"],
                  "nodeTableColumns": ["id"],
              },
              "destinationNodeTable": {
                  "nodeTableName": "n2",
                  "edgeTableColumns": ["target_id"],
                  "nodeTableColumns": ["id"],
              },
              "propertyDefinitions": [
                  {"propertyDeclarationName": "id", "valueExpressionSql": "id"}
              ],
          },
      ],
      "labels": [
          {
              "name": "n1",
              "propertyDeclarationNames": ["f1", "f2", "id"],
          },
          {
              "name": "n2",
              "propertyDeclarationNames": ["f3", "f4", "id"],
          },
          {
              "name": "e1",
              "propertyDeclarationNames": [],
          },
          {
              "name": "e2",
              "propertyDeclarationNames": [],
          },
      ],
      "schema": "",
  }


def _get_spanner_graph_metadata_and_features():
  """Loads the Spanner Graph metadata and feature format mapping."""

  metadata_json = get_spanner_graph_metadata_dict()

  feature_formats = {
      prop["name"]: gcp_common_lib.raw_type_to_feature_format(
          prop["name"], prop["type"]
      )[0]
      for prop in metadata_json["propertyDeclarations"]
  }
  feature_semantics = {
      "f1": schema_lib.FeatureSemantic.CATEGORICAL,
      "f2": schema_lib.FeatureSemantic.EMBEDDING,
  }
  feature_shapes = {
      "f1": (1,),
      "f2": (2,),
  }

  return (
      sgm.SpannerGraphMetadata.from_json(json.dumps(metadata_json)),
      feature_formats,
      feature_semantics,
      feature_shapes,
  )


def generate_spanner_graph() -> Tuple[Any, Dict[str, List[Dict[str, Any]]]]:
  """Generates a SpannerGraph config and corresponding raw data for mocking.

  Returns:
    A tuple containing:
      - A dummy config instance holding metadata properties.
      - A dictionary mapping edge/node set names to a list of Spanner result
      rows.
  """
  metadata, feature_formats, feature_semantics, feature_shapes = (
      _get_spanner_graph_metadata_and_features()
  )

  spanner_graph = mock.MagicMock()
  spanner_graph.project_id = "test_project"
  spanner_graph.instance_id = "test_instance"
  spanner_graph.database_id = "test_database"
  spanner_graph.graph_id = "SpannerGraph"
  spanner_graph.feature_shapes = feature_shapes
  spanner_graph.feature_semantics = feature_semantics
  spanner_graph.feature_formats = feature_formats
  spanner_graph.spanner_graph_metadata = metadata

  test_data = {}

  # --- Node Set n1 ---
  # Spanner graph n1: 2 nodes.
  # f1: [[b"blue"], [b"red"]]
  # f2: [[0.0, 1.0], [2.0, 3.0]]
  test_data["n1"] = [
      {
          _ID_COLUMN_NAME: "1",
          "id": "1",
          "f1": ["blue"],
          "f2": [0.0, 1.0],
      },
      {
          _ID_COLUMN_NAME: "2",
          "id": "2",
          "f1": ["red"],
          "f2": [2.0, 3.0],
      },
  ]

  # --- Node Set n2 ---
  # generic graph n2: 2 nodes.
  # f3: [4, 5]
  # f4: [10, 11]
  # f5 (variable length): [[11, 12], [12, 13, 14]]
  test_data["n2"] = [
      {
          _ID_COLUMN_NAME: "1",
          "id": 1,
          "f3": 4,
          "f4": 10,
      },
      {
          _ID_COLUMN_NAME: "2",
          "id": 2,
          "f3": 5,
          "f4": 11,
      },
  ]

  # --- Edge Set e1 ---
  # generic graph e1: Adjacency [[0, 0], [0, 1]] (Source Index -> Target Index)
  # Maps to: n1(id=1)->n1(id=1) and n1(id=1)->n1(id=2)
  test_data["e1"] = [
      {
          _SOURCE_ID_COLUMN_NAME: "1",
          _TARGET_ID_COLUMN_NAME: "1",
          "id": "a",
          "source_id": "1",
          "target_id": "1",
      },
      {
          _SOURCE_ID_COLUMN_NAME: "1",
          _TARGET_ID_COLUMN_NAME: "2",
          "id": "b",
          "source_id": "1",
          "target_id": "2",
      },
  ]

  # --- Edge Set e2 ---
  # generic graph e2: Adjacency [[0, 0], [0, 1]]
  # Maps to: n1(id=1)->n2(id=1) and n1(id=1)->n2(id=2)
  test_data["e2"] = [
      {
          _SOURCE_ID_COLUMN_NAME: "1",
          _TARGET_ID_COLUMN_NAME: "1",
          "id": "A",
          "source_id": "1",
          "target_id": "1",
      },
      {
          _SOURCE_ID_COLUMN_NAME: "1",
          _TARGET_ID_COLUMN_NAME: "2",
          "id": "B",
          "source_id": "1",
          "target_id": "2",
      },
  ]

  return spanner_graph, test_data


def generate_schema_for_tfgnn_classification(
    node_ids: bool = False,
    edge_ids: bool = False,
) -> schema_lib.GraphSchema:
  """Returns a graph schema sample for TF-GNN classification."""

  n1_features = {
      "f1": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          shape=(2,),
          semantic=schema_lib.FeatureSemantic.EMBEDDING,
      ),
      "f2": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_32,
          shape=None,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
          num_categorical_values=10,
      ),
  }

  n2_features = {
      "f3": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_32,
          shape=(),
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
          num_categorical_values=3,
      )
  }

  if node_ids:
    n1_features["#id"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES
    )
    n2_features["#id"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64
    )

  e1_schema = schema_lib.EdgeSchema(source="n1", target="n1")
  e2_schema = schema_lib.EdgeSchema(source="n1", target="n2")

  if edge_ids:
    e1_schema.features = {
        "#id": schema_lib.FeatureSchema(format=schema_lib.FeatureFormat.BYTES)
    }
    e2_schema.features = {
        "#id": schema_lib.FeatureSchema(format=schema_lib.FeatureFormat.BYTES)
    }

  return schema_lib.GraphSchema(
      node_sets={
          "n1": schema_lib.NodeSchema(features=n1_features),
          "n2": schema_lib.NodeSchema(features=n2_features),
      },
      edge_sets={
          "e1": e1_schema,
          "e2": e2_schema,
      },
  )


def generate_tf_graph_sample_for_tfgnn_classification(
    node_ids: bool = True,
    edge_ids: bool = True,
) -> tf.train.Example:
  """Generates a single tf.train.Example for TF-GNN classification."""

  features_dict = {
      # Node Set n1
      "nodes/n1.#size": tf.train.Feature(
          int64_list=tf.train.Int64List(value=[2])
      ),
      "nodes/n1.f1": tf.train.Feature(
          float_list=tf.train.FloatList(value=[0.1, 0.2, 0.3, 0.4])
      ),
      "nodes/n1.f2": tf.train.Feature(
          int64_list=tf.train.Int64List(value=[5, 8])
      ),
      # Node Set n2
      "nodes/n2.#size": tf.train.Feature(
          int64_list=tf.train.Int64List(value=[1])
      ),
      "nodes/n2.f3": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
      # Edge Set e1 (n1 -> n1)
      "edges/e1.#size": tf.train.Feature(
          int64_list=tf.train.Int64List(value=[2])
      ),
      "edges/e1.#source": tf.train.Feature(
          int64_list=tf.train.Int64List(value=[0, 1])
      ),
      "edges/e1.#target": tf.train.Feature(
          int64_list=tf.train.Int64List(value=[1, 0])
      ),
      # Edge Set e2 (n1 -> n2)
      "edges/e2.#size": tf.train.Feature(
          int64_list=tf.train.Int64List(value=[2])
      ),
      "edges/e2.#source": tf.train.Feature(
          int64_list=tf.train.Int64List(value=[0, 1])
      ),
      "edges/e2.#target": tf.train.Feature(
          int64_list=tf.train.Int64List(value=[0, 0])
      ),
  }

  if node_ids:
    features_dict["nodes/n1.#id"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[b"1", b"2"])
    )
    features_dict["nodes/n2.#id"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[99])
    )

  if edge_ids:
    features_dict["edges/e1.#id"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[b"a", b"b"])
    )
    features_dict["edges/e2.#id"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[b"A", b"B"])
    )

  return tf.train.Example(features=tf.train.Features(feature=features_dict))


def generate_semantic_tfrecord_for_tfgnn_classification(
    path: str,
    num_samples: int = 2,
    node_ids: bool = True,
    edge_ids: bool = True,
) -> None:
  """Writes compliant samples to a GZIP TFRecord file."""
  sample = generate_tf_graph_sample_for_tfgnn_classification(
      node_ids=node_ids, edge_ids=edge_ids
  )
  with tf.io.TFRecordWriter(path, options="GZIP") as writer:
    for _ in range(num_samples):
      writer.write(sample.SerializeToString())


def gen_toy_classification_dataset(
    num_n1_nodes: int = 100,
    num_n2_nodes: int = 200,
    random_seed: int = 0,
    max_num_edges_per_n1_nodes: int = 20,
    accuracy: float = 0.8,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Generates a toy classification dataset with a learnable pattern.

  The dataset contains two node sets, N1 and N2. N1 contains the label feature.

  The label for an N1 node is true if and only if the ratio of connected N2
  nodes with feature "f2" > 0.5 is greater than or equal to the N1 node's "f1"
  feature value. The "f3" feature contains no relevant information for
  classification. The relation between N1 nodes are not interesting.


  Args:
    num_n1_nodes: The number of nodes in node set N1.
    num_n2_nodes: The number of nodes in node set N2.
    random_seed: The random seed for data generation.
    max_num_edges_per_n1_nodes: Max number of edges per N1 node.
    accuracy: The target accuracy of the learnable pattern. Noise is injected
      into the labels such that a model cannot achieve better than this accuracy
      on a test dataset.

  Returns:
    A tuple containing the generated graph and its schema.
  """
  schema = schema_lib.GraphSchema(
      node_sets={
          "N1": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "label": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                  ),
                  "f1": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
                  "f3": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          ),
          "N2": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "f2": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          ),
      },
      edge_sets={
          "N1_to_N1": schema_lib.EdgeSchema(source="N1", target="N1"),
          "N1_to_N2": schema_lib.EdgeSchema(source="N1", target="N2"),
      },
  )
  rng = np.random.default_rng(random_seed)

  # Add N1 nodes
  n1_ids = [f"N1_{i}".encode() for i in range(num_n1_nodes)]
  n1_f1 = rng.integers(0, 5, size=num_n1_nodes)
  n1_f3 = rng.random(size=num_n1_nodes).astype(np.float32)
  n1_labels = np.zeros(shape=num_n1_nodes, dtype=bool)
  n1_node_set = in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=num_n1_nodes,
      features={
          "#id": np.array(n1_ids),
          "f1": n1_f1,
          "f3": n1_f3,
      },
  )

  # Add N2 nodes
  n2_ids = [f"N2_{i}".encode() for i in range(num_n2_nodes)]
  n2_f2 = rng.random(size=num_n2_nodes).astype(np.float32)
  n2_node_set = in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=num_n2_nodes,
      features={
          "#id": np.array(n2_ids),
          "f2": n2_f2,
      },
  )

  # Add N1_to_N2 edges and determine N1 labels
  n1_to_n2_source = []
  n1_to_n2_target = []

  for i in range(num_n1_nodes):
    # Each N1 node is connected to a random subset of N2 nodes.
    num_edges = rng.integers(0, max_num_edges_per_n1_nodes)
    target_indices = rng.choice(num_n2_nodes, size=num_edges, replace=False)

    n1_to_n2_source.extend([i] * num_edges)
    n1_to_n2_target.extend(target_indices)

    connected_n2_f2 = n2_f2[target_indices]
    num_connected_f2_gt_0_5 = np.sum(connected_n2_f2 > 0.5)
    ratio_num_connected_f2_gt_0_5 = num_connected_f2_gt_0_5 / num_edges

    if ratio_num_connected_f2_gt_0_5 >= n1_f1[i]:
      n1_labels[i] = True

    # Randomly flip the label with probability 1 - accuracy
    if rng.random() > accuracy:
      n1_labels[i] = not n1_labels[i]

  n1_node_set.features["label"] = n1_labels.astype(np.int64)

  n1_to_n2_edge_set = in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=np.array([n1_to_n2_source, n1_to_n2_target], dtype=np.int64),
      features={},
  )

  # Add some random N1_to_N1 edges (not relevant for the label)
  num_n1_to_n1_edges = num_n1_nodes * 2
  n1_to_n1_source = rng.choice(num_n1_nodes, size=num_n1_to_n1_edges)
  n1_to_n1_target = rng.choice(num_n1_nodes, size=num_n1_to_n1_edges)
  n1_to_n1_edge_set = in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=np.array([n1_to_n1_source, n1_to_n1_target], dtype=np.int64),
      features={},
  )

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "N1": n1_node_set,
          "N2": n2_node_set,
      },
      edge_sets={
          "N1_to_N2": n1_to_n2_edge_set,
          "N1_to_N1": n1_to_n1_edge_set,
      },
  )

  return graph, schema


def gen_toy_regression_dataset(
    num_n1_nodes: int = 100,
    num_n2_nodes: int = 200,
    random_seed: int = 0,
    max_num_edges_per_n1_nodes: int = 20,
    label_dtype: Literal["float32", "int64"] = "float32",
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Generates a toy regression dataset with a learnable pattern.

  The dataset contains two node sets, N1 and N2. N1 contains the label feature.
  The label for an N1 node is based on the features of connected N2 nodes.

  Args:
    num_n1_nodes: The number of nodes in node set N1.
    num_n2_nodes: The number of nodes in node set N2.
    random_seed: The random seed for data generation.
    max_num_edges_per_n1_nodes: Max number of edges per N1 node.
    label_dtype: The dtype for the regression label.

  Returns:
    A tuple containing the generated graph and its schema.
  """
  schema_label_format = (
      schema_lib.FeatureFormat.FLOAT_32
      if label_dtype == "float32"
      else schema_lib.FeatureFormat.INTEGER_64
  )

  schema = schema_lib.GraphSchema(
      node_sets={
          "N1": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "label": schema_lib.FeatureSchema(
                      format=schema_label_format,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
                  "f1": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
                  "f3": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          ),
          "N2": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "f2": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          ),
      },
      edge_sets={
          "N1_to_N1": schema_lib.EdgeSchema(source="N1", target="N1"),
          "N1_to_N2": schema_lib.EdgeSchema(source="N1", target="N2"),
      },
  )
  rng = np.random.default_rng(random_seed)

  # Add N1 nodes
  n1_ids = [f"N1_{i}".encode() for i in range(num_n1_nodes)]
  n1_f1 = rng.integers(0, 5, size=num_n1_nodes)
  n1_f3 = rng.random(size=num_n1_nodes).astype(np.float32)
  np_label_dtype = np.float32 if label_dtype == "float32" else np.int64
  n1_labels = np.zeros(shape=num_n1_nodes, dtype=np_label_dtype)
  n1_node_set = in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=num_n1_nodes,
      features={
          "#id": np.array(n1_ids),
          "f1": n1_f1,
          "f3": n1_f3,
      },
  )

  # Add N2 nodes
  n2_ids = [f"N2_{i}".encode() for i in range(num_n2_nodes)]
  n2_f2 = rng.random(size=num_n2_nodes).astype(np.float32)
  n2_node_set = in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=num_n2_nodes,
      features={
          "#id": np.array(n2_ids),
          "f2": n2_f2,
      },
  )

  # Add N1_to_N2 edges and determine N1 labels
  n1_to_n2_source = []
  n1_to_n2_target = []

  for i in range(num_n1_nodes):
    # Each N1 node is connected to a random subset of N2 nodes.
    num_edges = rng.integers(0, max_num_edges_per_n1_nodes)
    target_indices = rng.choice(num_n2_nodes, size=num_edges, replace=False)

    n1_to_n2_source.extend([i] * num_edges)
    n1_to_n2_target.extend(target_indices)

    if num_edges > 0:
      connected_n2_f2 = n2_f2[target_indices]
      # Some regression target function
      label_val = n1_f1[i] * 2.0 + np.sum(connected_n2_f2) + rng.random()
    else:
      label_val = n1_f1[i] * 2.0 + rng.random()

    n1_labels[i] = np_label_dtype(label_val)

  n1_node_set.features["label"] = n1_labels

  n1_to_n2_edge_set = in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=np.array([n1_to_n2_source, n1_to_n2_target], dtype=np.int64),
      features={},
  )

  # Add some random N1_to_N1 edges
  num_n1_to_n1_edges = num_n1_nodes * 2
  n1_to_n1_source = rng.choice(num_n1_nodes, size=num_n1_to_n1_edges)
  n1_to_n1_target = rng.choice(num_n1_nodes, size=num_n1_to_n1_edges)
  n1_to_n1_edge_set = in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=np.array([n1_to_n1_source, n1_to_n1_target], dtype=np.int64),
      features={},
  )

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "N1": n1_node_set,
          "N2": n2_node_set,
      },
      edge_sets={
          "N1_to_N2": n1_to_n2_edge_set,
          "N1_to_N1": n1_to_n1_edge_set,
      },
  )

  return graph, schema


def generate_temporal_in_memory_graph(
    include_e2: bool,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Generates a temporal InMemoryGraph for testing."""

  edge_sets_schema = {
      "e1": schema_lib.EdgeSchema(
          source="n1",
          target="n1",
          features={
              "timestamp": schema_lib.FeatureSchema(
                  format=schema_lib.FeatureFormat.INTEGER_64,
                  semantic=schema_lib.FeatureSemantic.TIMESTAMP,
              )
          },
      ),
  }
  if include_e2:
    edge_sets_schema["e2"] = schema_lib.EdgeSchema(
        source="n1",
        target="n1",
        features={},
    )

  schema = schema_lib.GraphSchema(
      node_sets={
          "n1": schema_lib.NodeSchema(
              features={
                  "timestamp": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                  ),
                  "feat": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      shape=(2,),
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                  ),
              }
          )
      },
      edge_sets=edge_sets_schema,
  )

  n1_nodes = in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=4,
      features={
          "timestamp": np.array([10, 20, 30, 40], dtype=np.int64),
          "feat": np.array(
              [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32
          ),
      },
  )

  e1_edges = in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=np.array([[0, 0, 1], [1, 2, 3]], dtype=np.int64),
      features={"timestamp": np.array([15, 25, 35], dtype=np.int64)},
  )

  edge_sets_data = {"e1": e1_edges}
  if include_e2:
    e2_edges = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=np.array([[2, 3], [0, 1]], dtype=np.int64),
        features={},
    )
    edge_sets_data["e2"] = e2_edges

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={"n1": n1_nodes}, edge_sets=edge_sets_data
  )
  return graph, schema


def generate_recommender_like_in_memory_graph() -> (
    Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
):
  """Generates a recommender-like in-memory graph and schema.

  Graph structure (e2 edges from n1 to n2):
    n1 nodes               n2 nodes
    --------               --------
      ( 0 ) ---------------> ( 0 )
        |
        +------------------> ( 1 )
                               ^
      ( 1 ) -------------------+
        |
        +------------------> ( 2 )

      ( 2 )                  ( 3 )
                             ( 4 )
                             ( 5 )
  """
  schema = schema_lib.GraphSchema(
      node_sets={
          "n1": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "f1": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          ),
          "n2": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "f2": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          ),
      },
      edge_sets={
          "e2": schema_lib.EdgeSchema(source="n1", target="n2"),
      },
  )
  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "n1": in_memory_graph_lib.InMemoryNodeSet(
              features={
                  "#id": np.array([0, 1, 2], dtype=np.int64),
                  "f1": np.array([0.0, 1.0, 2.0], dtype=np.float32),
              },
              num_nodes=3,
          ),
          "n2": in_memory_graph_lib.InMemoryNodeSet(
              features={
                  "#id": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
                  "f2": np.array(
                      [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
                  ),
              },
              num_nodes=6,
          ),
      },
      edge_sets={
          "e2": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.array([[0, 0, 1, 1], [0, 1, 1, 2]])
          ),
      },
  )
  return graph, schema
