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

import os
import struct
import tempfile
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import tf_graph_sample
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np
import tensorflow as tf

test_util.disable_diff_truncation()


def get_file_extension(container_type: str):
  if container_type == "TF_RECORD":
    return "tfr.gz"
  if container_type == "BAGZ":
    return "bagz"
  assert False


def _get_runner():
  return None


class TfGnnGraphSampleTest(parameterized.TestCase):

  @parameterized.product(node_ids=[None, "#id"], edge_ids=[None, "#id"])
  def test_tfgnn_graph_to_graph(self, node_ids, edge_ids):
    expected_graph = gen_test_graph.generate_in_memory_graph(
        node_ids, edge_ids, variable_length=True
    )
    tf_sample = gen_test_graph.generate_tf_graph_sample(
        node_ids, edge_ids, variable_length=True
    )
    schema = gen_test_graph.generate_schema(
        variable_length=True, node_ids=node_ids, edge_ids=edge_ids
    )
    graph = tf_graph_sample.tfgnn_graph_to_graph(
        tf_sample, schema, node_ids, edge_ids
    )
    test_util.assert_are_equal(self, graph, expected_graph)

  @parameterized.product(node_ids=[None, "#id"], edge_ids=[None, "#id"])
  def test_graph_to_tfgnn_graph(self, node_ids, edge_ids):
    schema = gen_test_graph.generate_schema(
        variable_length=True, node_ids=node_ids, edge_ids=edge_ids
    )
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids, edge_ids, variable_length=True
    )
    expected_tf_sample = gen_test_graph.generate_tf_graph_sample(
        node_ids, edge_ids, variable_length=True
    )
    tf_sample = tf_graph_sample.graph_to_tfgnn_graph(graph, schema=schema)
    test_util.assert_are_equal(self, tf_sample, expected_tf_sample)

  @parameterized.product(
      node_ids=[None, "#id"],
      edge_ids=[None, "#id"],
      container_type=["TF_RECORD"],
  )
  def test_read_tf_graph_sample(self, node_ids, edge_ids, container_type):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      extension = get_file_extension(container_type)
      path = os.path.join(tmpdir, f"samples@*.{extension}")
      gen_test_graph.generate_tf_graph_sample_in_tf_record(
          os.path.join(
              tmpdir,
              f"samples-00000-of-00001.{extension}",
          ),
          node_ids,
          edge_ids,
          container_type=container_type,
      )
      schema = gen_test_graph.generate_schema(
          variable_length=False, node_ids=node_ids, edge_ids=edge_ids
      )

      expected_in_mem_graphs = [
          distributed_graph_lib.KeyedInMemoryGraph(
              None,
              gen_test_graph.generate_in_memory_graph(
                  node_ids, edge_ids, variable_length=False
              ),
          ),
          distributed_graph_lib.KeyedInMemoryGraph(
              None,
              gen_test_graph.generate_in_memory_graph(
                  node_ids, edge_ids, variable_length=False
              ),
          ),
      ]

      with beam.Pipeline(_get_runner()) as p:
        in_mem_graphs = tf_graph_sample.read_tfgnn_graphs_beam(
            p,
            path,
            schema=schema,
            import_node_ids=node_ids,
            import_edge_ids=edge_ids,
            container_type=container_type,
        )
        util.assert_that(
            in_mem_graphs,
            util.equal_to(expected_in_mem_graphs, test_util.are_equal),
        )

  @parameterized.product(
      node_ids=[None, "#id"],
      edge_ids=[None, "#id"],
      container_type=["TF_RECORD"],
  )
  def test_write_tf_graph_sample(self, node_ids, edge_ids, container_type):
    with tempfile.TemporaryDirectory() as tmpdir:
      extension = get_file_extension(container_type)
      os.makedirs(tmpdir, exist_ok=True)
      path = os.path.join(tmpdir, f"samples@*.{extension}")

      # Generate some toy data
      in_mem_graphs = [
          distributed_graph_lib.KeyedInMemoryGraph(
              None,
              gen_test_graph.generate_in_memory_graph(
                  node_ids, edge_ids, variable_length=False
              ),
          ),
      ]
      expected_tf_samples = [
          gen_test_graph.generate_tf_graph_sample(node_ids, edge_ids),
      ]

      with beam.Pipeline(_get_runner()) as p:
        p_in_mem_graph = p | beam.Create(in_mem_graphs)
        tf_graph_sample.write_tfgnn_graphs_beam(
            p_in_mem_graph,
            path,
            schema=gen_test_graph.generate_schema(
                node_ids, edge_ids, variable_length=False
            ),
            container_type=container_type,
        )

      with beam.Pipeline(_get_runner()) as p:
        if container_type == "TF_RECORD":
          tf_samples = p | beam.io.tfrecordio.ReadFromTFRecord(
              os.path.join(tmpdir, f"samples-00000-of-00001.{extension}"),
              coder=beam.coders.ProtoCoder(tf.train.Example),
              compression_type=beam.io.filesystem.CompressionTypes.GZIP,
          )
        else:
          assert False
        util.assert_that(
            tf_samples,
            util.equal_to(expected_tf_samples, test_util.are_equal),
        )

  def test_write_tfgnn_graphs(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "samples@2.tfr.gz")
      paths = [
          os.path.join(tmpdir, "samples-00000-of-00002.tfr.gz"),
          os.path.join(tmpdir, "samples-00001-of-00002.tfr.gz"),
      ]

      # Generate some toy data
      graph = gen_test_graph.generate_in_memory_graph(variable_length=True)
      schema = gen_test_graph.generate_schema(variable_length=True)

      def in_mem_graphs():
        yield graph
        yield graph
        yield graph

      tf_graph_sample.write_tfgnn_graphs(
          in_mem_graphs(),
          path,
          schema=schema,
      )

      num_read_examples = 0
      expected_example = tf_graph_sample.graph_to_tfgnn_graph(
          graph, schema=schema
      )
      for current_path in paths:
        for tensor in tf.data.TFRecordDataset(
            current_path, compression_type="GZIP"
        ):
          read_example = tf.train.Example.FromString(tensor.numpy())
          self.assertEqual(read_example, expected_example)
          num_read_examples += 1
      self.assertEqual(num_read_examples, 3)

  @parameterized.parameters(("TF_RECORD",))
  def test_read_tfgnn_graphs(self, container_type):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(
          tmpdir, f"samples@2.{get_file_extension(container_type)}"
      )

      # Generate some toy data
      graph = gen_test_graph.generate_in_memory_graph(variable_length=False)
      schema = gen_test_graph.generate_schema(variable_length=False)

      def in_mem_graphs():
        yield graph
        yield graph
        yield graph

      tf_graph_sample.write_tfgnn_graphs(
          in_mem_graphs(), path, schema=schema, container_type=container_type
      )

      num_read_examples = 0
      for read_graph in tf_graph_sample.read_tfgnn_graphs(
          path,
          gen_test_graph.generate_schema(variable_length=False),
          container_type=container_type,
      ):
        test_util.assert_are_equal(self, read_graph, graph)
        num_read_examples += 1
      self.assertEqual(num_read_examples, 3)

  def test_bool_feature_round_trip(self):
    """BOOL features are stored as int64s in a TF GNN Graph Sample."""
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BOOL
                    )
                }
            )
        },
        edge_sets={},
    )
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=3,
                features={"f1": np.array([True, False, True])},
            )
        },
        edge_sets={},
    )

    example = tf_graph_sample.graph_to_tfgnn_graph(graph, schema)

    self.assertEqual(
        list(example.features.feature["nodes/n1.f1"].int64_list.value),
        [1, 0, 1],
    )
    test_util.assert_are_equal(
        self, tf_graph_sample.tfgnn_graph_to_graph(example, schema), graph
    )

  def test_multi_ragged_feature_is_rejected(self):
    """The NumPy format supports at most one variable-length dimension."""
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        shape=(None, None),
                    )
                }
            )
        },
        edge_sets={},
    )

    def as_object_array(rows):
      result = np.empty(len(rows), dtype=np.object_)
      result[:] = rows
      return result

    values = as_object_array([
        as_object_array([np.array([1, 2]), np.array([3])]),
        as_object_array([np.array([4, 5, 6])]),
    ])
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=2, features={"f1": values}
            )
        },
        edge_sets={},
    )

    with self.assertRaisesRegex(ValueError, "has 2 variable-length dimensions"):
      tf_graph_sample.graph_to_tfgnn_graph_dict(graph, schema)

    with self.assertRaisesRegex(ValueError, "has 2 variable-length dimensions"):
      tf_graph_sample.graph_dict_to_graph(
          {
              "nodes/n1.#size": np.array([2]),
              "nodes/n1.f1": np.array([1, 2, 3, 4, 5, 6]),
              "nodes/n1.f1.d1": np.array([2, 1]),
              "nodes/n1.f1.d2": np.array([2, 1, 3]),
          },
          schema,
      )


def _all_formats_features() -> dict[str, schema_lib.FeatureSchema]:
  """Features with all the formats, and many static and variable shapes."""
  features = {}
  for format_ in schema_lib.FeatureFormat:
    for shape_name, shape in [
        ("scalar", None),
        ("vector", (3,)),
        ("matrix", (2, 3)),
        ("tensor", (2, 1, 3)),
        ("ragged", (None,)),
        ("ragged_vector", (None, 2)),
        ("ragged_matrix", (None, 2, 3)),
        ("vector_ragged", (2, None)),
        ("vector_ragged_vector", (2, None, 3)),
    ]:
      features[f"{format_.name.lower()}_{shape_name}"] = (
          schema_lib.FeatureSchema(format=format_, shape=shape)
      )
  return features


def _all_formats_schema() -> schema_lib.GraphSchema:
  """A schema with all the feature formats and many shapes."""
  fmt = schema_lib.FeatureFormat
  node_features = _all_formats_features()
  node_features["#id"] = schema_lib.FeatureSchema(format=fmt.BYTES)
  edge_features = _all_formats_features()
  edge_features["#id"] = schema_lib.FeatureSchema(format=fmt.FLOAT_32)
  return schema_lib.GraphSchema(
      node_sets={
          "n1": schema_lib.NodeSchema(features=node_features),
          "n2": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(format=fmt.INTEGER_64),
              }
          ),
          "n3": schema_lib.NodeSchema(features={}),
      },
      edge_sets={
          "e1": schema_lib.EdgeSchema(
              source="n1", target="n2", features=edge_features
          ),
          "e2": schema_lib.EdgeSchema(
              source="n1",
              target="n1",
              features={"#id": schema_lib.FeatureSchema(format=fmt.INTEGER_32)},
          ),
      },
  )


def _random_values(
    rng: np.random.Generator,
    format_: schema_lib.FeatureFormat,
    shape: list[int],
) -> np.ndarray:
  """Random values of a given format and shape."""
  fmt = schema_lib.FeatureFormat
  if format_ == fmt.BYTES:
    size = int(np.prod(shape))
    values = [
        bytes(rng.integers(1, 256, size=rng.integers(0, 5), dtype=np.uint8))
        for _ in range(size)
    ]
    return np.array(values, dtype=np.bytes_).reshape(shape)
  if format_ == fmt.BOOL:
    return rng.integers(0, 2, size=shape).astype(np.bool_)
  if format_ in (fmt.FLOAT_32, fmt.FLOAT_64):
    # Note: FLOAT_64 values are stored as float32 in a TF GNN Graph Sample.
    return (
        rng.normal(size=shape)
        .astype(np.float32)
        .astype(feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[format_])
    )
  return rng.integers(-1000, 1000, size=shape).astype(
      feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[format_]
  )


def _as_object_array(items: list[Any], shape: list[int]) -> np.ndarray:
  result = np.empty(len(items), dtype=np.object_)
  result[:] = items
  return result.reshape(shape)


def _random_feature(
    rng: np.random.Generator,
    feature_schema: schema_lib.FeatureSchema,
    num_items: int,
) -> np.ndarray:
  """Random values of a feature."""
  shape = list(feature_schema.shape or [])
  # Note: Only one dimension can be variable-length.
  ragged_idx = shape.index(None) if None in shape else len(shape)
  outer_shape = [num_items] + [int(dim or 0) for dim in shape[:ragged_idx]]
  if feature_schema.is_static_shape():
    return _random_values(rng, feature_schema.format, outer_shape)
  inner_shape = [int(dim or 0) for dim in shape[ragged_idx + 1 :]]
  rows = [
      _random_values(
          rng, feature_schema.format, [int(rng.integers(0, 4))] + inner_shape
      )
      for _ in range(int(np.prod(outer_shape)))
  ]
  return _as_object_array(rows, outer_shape)


def _random_graph(
    rng: np.random.Generator, schema: schema_lib.GraphSchema
) -> in_memory_graph.InMemoryGraph:
  """A random graph following a schema."""
  node_sets = {}
  for node_set_name, node_set_schema in schema.node_sets.items():
    num_nodes = int(rng.integers(0, 4))
    node_sets[node_set_name] = in_memory_graph.InMemoryNodeSet(
        num_nodes=num_nodes,
        features={
            feature_name: _random_feature(rng, feature_schema, num_nodes)
            for feature_name, feature_schema in node_set_schema.features.items()
        },
    )
  edge_sets = {}
  for edge_set_name, edge_set_schema in schema.edge_sets.items():
    num_edges = int(rng.integers(0, 4))
    edge_sets[edge_set_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=rng.integers(0, 3, size=[2, num_edges]),
        features={
            feature_name: _random_feature(rng, feature_schema, num_edges)
            for feature_name, feature_schema in edge_set_schema.features.items()
        },
    )
  return in_memory_graph.InMemoryGraph(node_sets=node_sets, edge_sets=edge_sets)


def _write_records(path: str, records: list[bytes]) -> None:
  """Writes serialized records in a GZIP TFRecord file."""
  with tf.io.TFRecordWriter(
      path, options=tf.io.TFRecordOptions(compression_type="GZIP")
  ) as writer:
    for record in records:
      writer.write(record)


def _varint(value: int) -> bytes:
  """Encodes a proto varint."""
  value &= (1 << 64) - 1
  result = bytearray()
  while True:
    byte = value & 0x7F
    value >>= 7
    if value:
      result.append(byte | 0x80)
    else:
      result.append(byte)
      return bytes(result)


def _length_delimited(field: int, payload: bytes) -> bytes:
  """Encodes a length-delimited proto field."""
  return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _unpacked_example(features: list[tuple[str, str, list[Any]]]) -> bytes:
  """Encodes a tf.train.Example with non-packed int64 and float values.

  Args:
    features: List of (key, kind, values) where kind is "int64" or "float". Keys
      can be repeated.

  Returns:
    The serialized tf.train.Example.
  """
  entries = b""
  for key, kind, values in features:
    if kind == "int64":
      value_list = b"".join(_varint(1 << 3 | 0) + _varint(v) for v in values)
      feature = _length_delimited(3, value_list)
    elif kind == "float":
      value_list = b"".join(
          _varint(1 << 3 | 5) + struct.pack("<f", v) for v in values
      )
      feature = _length_delimited(2, value_list)
    else:
      raise ValueError(kind)
    entry = _length_delimited(1, key.encode()) + _length_delimited(2, feature)
    entries += _length_delimited(1, entry)
  return _length_delimited(1, entries)


class ReadTfgnnGraphsImplementationTest(parameterized.TestCase):
  """Checks that the implementations of `read_tfgnn_graphs` are identical."""

  def _read_all_implementations(
      self, path: str, schema: schema_lib.GraphSchema, **kwargs
  ) -> list[in_memory_graph.InMemoryGraph]:
    """Reads graphs with all the implementations, and checks they match."""
    kwargs.setdefault("deterministic", True)
    results = {
        implementation: list(
            tf_graph_sample.read_tfgnn_graphs(
                path, schema, implementation=implementation, **kwargs
            )
        )
        for implementation in ["python", "cpp", "auto"]
    }
    for implementation in ["cpp", "auto"]:
      test_util.assert_are_equal(
          self, results["python"], results[implementation], strict=True
      )
    return results["cpp"]

  @parameterized.product(
      node_ids=[None, "#id"],
      edge_ids=[None, "#id"],
      variable_length=[False, True],
  )
  def test_gen_test_graph(self, node_ids, edge_ids, variable_length):
    schema = gen_test_graph.generate_schema(
        node_ids=node_ids, edge_ids=edge_ids, variable_length=variable_length
    )
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids, edge_ids, variable_length=variable_length
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "samples@3.tfr.gz")
      tf_graph_sample.write_tfgnn_graphs(
          (graph for _ in range(10)), path, schema=schema
      )
      read_graphs = self._read_all_implementations(
          path, schema, import_node_ids=node_ids, import_edge_ids=edge_ids
      )
    self.assertLen(read_graphs, 10)
    for read_graph in read_graphs:
      test_util.assert_are_equal(self, read_graph, graph)

  @parameterized.parameters(1, 4)
  def test_all_formats(self, num_threads):
    schema = _all_formats_schema()
    rng = np.random.default_rng(seed=1)
    graphs = [_random_graph(rng, schema) for _ in range(300)]
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "samples@1.tfr.gz")
      tf_graph_sample.write_tfgnn_graphs(
          (graph for graph in graphs), path, schema=schema
      )
      read_graphs = self._read_all_implementations(
          path, schema, num_threads=num_threads
      )
    self.assertLen(read_graphs, len(graphs))
    for read_graph, graph in zip(read_graphs, graphs, strict=True):
      test_util.assert_are_equal(self, read_graph, graph)

  def test_all_formats_imported_ids(self):
    schema = _all_formats_schema()
    # Note: The imported id feature should be defined in all the sets.
    del schema.node_sets["n3"]
    rng = np.random.default_rng(seed=2)
    graphs = [_random_graph(rng, schema) for _ in range(50)]
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "samples@2.tfr.gz")
      tf_graph_sample.write_tfgnn_graphs(
          (graph for graph in graphs), path, schema=schema
      )
      read_graphs = self._read_all_implementations(
          path, schema, import_node_ids="#id", import_edge_ids="#id"
      )
    self.assertLen(read_graphs, len(graphs))
    # The imported ids are flat, and use their storage dtype.
    for read_graph in read_graphs:
      self.assertEqual(
          read_graph.edge_sets["e2"].features["#id"].dtype, np.int64
      )

  def test_multi_dimensional_and_variable_size_features(self):
    fmt = schema_lib.FeatureFormat
    schema = schema_lib.GraphSchema(
        node_sets={
            "n": schema_lib.NodeSchema(
                features={
                    "m": schema_lib.FeatureSchema(
                        format=fmt.INTEGER_32, shape=(2, 2)
                    ),
                    "r": schema_lib.FeatureSchema(
                        format=fmt.FLOAT_32, shape=(None, 2)
                    ),
                    "s": schema_lib.FeatureSchema(
                        format=fmt.BYTES, shape=(2, None)
                    ),
                }
            )
        },
        edge_sets={
            "e": schema_lib.EdgeSchema(
                source="n",
                target="n",
                features={
                    "r": schema_lib.FeatureSchema(
                        format=fmt.INTEGER_64, shape=(None,)
                    )
                },
            )
        },
    )
    r = np.empty(2, dtype=np.object_)
    r[:] = [
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
    ]
    s = np.empty((2, 2), dtype=np.object_)
    s[0, 0] = np.array([b"a", b"bc"])
    s[0, 1] = np.array([], dtype=np.bytes_)
    s[1, 0] = np.array([b"d"])
    s[1, 1] = np.array([b"e", b"f", b"g"])
    edge_r = np.empty(3, dtype=np.object_)
    edge_r[:] = [np.array([5]), np.array([], dtype=np.int64), np.array([6, 7])]
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n": in_memory_graph.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "m": (
                        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(
                            np.int32
                        )
                    ),
                    "r": r,
                    "s": s,
                },
            )
        },
        edge_sets={
            "e": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0, 1, 1], [1, 0, 1]]),
                features={"r": edge_r},
            )
        },
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "samples.tfr.gz")
      tf_graph_sample.write_tfgnn_graphs(
          (graph for _ in range(1)), path, schema=schema
      )
      (read_graph,) = self._read_all_implementations(path, schema)
    test_util.assert_are_equal(self, read_graph, graph)
    self.assertEqual(read_graph.node_sets["n"].features["m"].shape, (2, 2, 2))
    self.assertEqual(read_graph.node_sets["n"].features["s"].shape, (2, 2))

  @parameterized.parameters(True, False)
  def test_order(self, deterministic):
    graphs = [
        in_memory_graph.InMemoryGraph(
            node_sets={
                "n": in_memory_graph.InMemoryNodeSet(
                    num_nodes=1, features={"f": np.array([i])}
                )
            },
            edge_sets={},
        )
        for i in range(1000)
    ]
    schema = schema_lib.GraphSchema(
        node_sets={
            "n": schema_lib.NodeSchema(
                features={
                    "f": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64
                    )
                }
            )
        },
        edge_sets={},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "samples@7.tfr.gz")
      tf_graph_sample.write_tfgnn_graphs(
          (graph for graph in graphs), path, schema=schema
      )
      results = {}
      for implementation in ["python", "cpp"]:
        results[implementation] = [
            graph.node_sets["n"].features["f"][0]
            for graph in tf_graph_sample.read_tfgnn_graphs(
                path,
                schema,
                implementation=implementation,
                deterministic=deterministic,
            )
        ]
    for values in results.values():
      self.assertCountEqual(values, range(1000))
    if deterministic:
      self.assertEqual(results["python"], results["cpp"])

  def test_non_packed_and_repeated_values(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n": schema_lib.NodeSchema(
                features={
                    "i": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_32, shape=(2,)
                    ),
                    "f": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64
                    ),
                    "b": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BOOL
                    ),
                }
            )
        },
        edge_sets={"e": schema_lib.EdgeSchema(source="n", target="n")},
    )
    records = [
        _unpacked_example([
            ("nodes/n.#size", "int64", [2, 5]),
            ("nodes/n.i", "int64", [1, -2, 3, 1 << 40]),
            ("nodes/n.f", "float", [1.5, -2.25]),
            # Note: The last value of a repeated key wins.
            ("nodes/n.b", "int64", [9, 9, 9]),
            ("nodes/n.b", "int64", [0, 7]),
            ("edges/e.#size", "int64", [1]),
            ("edges/e.#source", "int64", [0]),
            ("edges/e.#target", "int64", [1]),
            ("unused", "float", [1.0]),
        ])
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "samples.tfr.gz")
      _write_records(path, records)
      (graph,) = self._read_all_implementations(path, schema)
    test_util.assert_are_equal(
        self,
        graph,
        in_memory_graph.InMemoryGraph(
            node_sets={
                "n": in_memory_graph.InMemoryNodeSet(
                    num_nodes=2,
                    features={
                        "i": np.array(
                            [[1, -2], [3, 0]], dtype=np.int32
                        ),  # Wrapping cast.
                        "f": np.array([1.5, -2.25], dtype=np.float64),
                        "b": np.array([False, True]),
                    },
                )
            },
            edge_sets={
                "e": in_memory_graph.InMemoryEdgeSet(
                    adjacency=np.array([[0], [1]]), features={}
                )
            },
        ),
        strict=True,
    )

  @parameterized.parameters(
      (
          [("nodes/n.#size", "int64", [2]), ("nodes/n.f", "float", [1.0, 2.0])],
          "int64_list is expected",
      ),
      (
          [("nodes/n.#size", "int64", [2]), ("nodes/n.f", "int64", [1])],
          "cannot be reshaped",
      ),
      ([("nodes/n.f", "int64", [])], "missing"),
  )
  def test_cpp_invalid_graph_sample(self, features, error):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n": schema_lib.NodeSchema(
                features={
                    "f": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64
                    )
                }
            )
        },
        edge_sets={},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "samples.tfr.gz")
      _write_records(path, [_unpacked_example(features)])
      with self.assertRaisesRegex(ValueError, error):
        list(
            tf_graph_sample.read_tfgnn_graphs(
                path, schema, implementation="cpp"
            )
        )

  def test_cpp_corrupted_graph_sample(self):
    schema = gen_test_graph.generate_schema(variable_length=False)
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "samples.tfr.gz")
      _write_records(path, [b"\x0a\xff\xff"])
      with self.assertRaisesRegex(ValueError, "Cannot parse"):
        list(
            tf_graph_sample.read_tfgnn_graphs(
                path, schema, implementation="cpp"
            )
        )

  @parameterized.parameters("python", "cpp")
  def test_unknown_imported_ids(self, implementation):
    schema = gen_test_graph.generate_schema(variable_length=False)
    with self.assertRaisesRegex(ValueError, "is not defined in the schema"):
      list(
          tf_graph_sample.read_tfgnn_graphs(
              "/unused@1",
              schema,
              import_node_ids="#id",
              implementation=implementation,
          )
      )

  def test_parser_submit(self):
    schema = gen_test_graph.generate_schema(variable_length=True)
    graph = gen_test_graph.generate_in_memory_graph(variable_length=True)
    serialized = tf_graph_sample.graph_to_tfgnn_graph(
        graph, schema
    ).SerializeToString()
    parser = tf_graph_sample.create_tfgnn_graph_parser(schema, num_threads=2)
    futures = [parser.submit([serialized] * i) for i in range(5)]
    for i, future in enumerate(futures):
      graphs = future.result()
      self.assertLen(graphs, i)
      for parsed_graph in graphs:
        test_util.assert_are_equal(self, parsed_graph, graph)
      with self.assertRaisesRegex(ValueError, "already consumed"):
        future.result()
    test_util.assert_are_equal(self, parser.parse([serialized]), [graph])
    with self.assertRaisesRegex(ValueError, "Expecting a sequence of bytes"):
      parser.parse(["not bytes"])
    # Futures can be destroyed before being consumed.
    del futures
    parser.submit([serialized] * 100)


if __name__ == "__main__":
  absltest.main()
