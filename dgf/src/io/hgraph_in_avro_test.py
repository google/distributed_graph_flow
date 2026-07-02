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

"""Tests for avro.py."""

import os
import tempfile

from absl.testing import absltest
from dgf.src.data import schema as schema_lib
from dgf.src.io import hgraph_in_avro as avro_lib
from dgf.src.util import gen_test_graph
import fastavro
import numpy as np


class HGraphAvroTest(absltest.TestCase):
  """Tests for reading and writing graphs from/to Avro format."""

  def test_get_avro_type_for_shape(self):
    # Test scalar
    self.assertEqual(avro_lib._get_avro_type_for_shape((), "long"), "long")
    # Test 1D array
    self.assertEqual(
        avro_lib._get_avro_type_for_shape((None,), "long"),
        {"type": "array", "items": "long"},
    )
    # Test 2D array
    self.assertEqual(
        avro_lib._get_avro_type_for_shape((None, 5), "long"),
        {"type": "array", "items": {"type": "array", "items": "long"}},
    )

  def test_get_avro_type_for_feature(self):
    # Scalar feature
    feature_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64
    )
    self.assertEqual(
        avro_lib._get_avro_type_for_feature(feature_schema), "long"
    )

    # 1D array feature
    feature_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64, shape=(None,)
    )
    self.assertEqual(
        avro_lib._get_avro_type_for_feature(feature_schema),
        {"type": "array", "items": "long"},
    )

    # 2D array feature
    feature_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32, shape=(None, 3)
    )
    self.assertEqual(
        avro_lib._get_avro_type_for_feature(feature_schema),
        {"type": "array", "items": {"type": "array", "items": "float"}},
    )

  def test_serialize_numpy_value(self):
    # numpy scalar
    self.assertEqual(avro_lib._serialize_numpy_value(np.int32(42)), 42)
    self.assertIsInstance(avro_lib._serialize_numpy_value(np.int32(42)), int)
    # numpy array
    self.assertEqual(
        avro_lib._serialize_numpy_value(np.array([1, 2, 3])), [1, 2, 3]
    )
    # non-numpy
    self.assertEqual(avro_lib._serialize_numpy_value(b"test"), b"test")

  def test_generate_node_records(self):
    features = {
        "f1": np.array([1, 2, 3]),
        "f2": np.array([[0.1], [0.2], [0.3]]),
    }
    feature_items = list(features.items())
    records = list(
        avro_lib._generate_node_records(
            feature_items,
            start_index=0,
            end_index=3,
            name="test",
            verbose=False,
        )
    )
    expected_records = [
        {"f1": 1, "f2": [0.1]},
        {"f1": 2, "f2": [0.2]},
        {"f1": 3, "f2": [0.3]},
    ]
    self.assertEqual(records, expected_records)

  def test_generate_edge_records(self):
    features = {"weight": np.array([0.5, 0.6])}
    feature_items = list(features.items())
    source = np.array([0, 1])
    target = np.array([1, 2])
    records = list(
        avro_lib._generate_edge_records(
            feature_items,
            source,
            target,
            start_index=0,
            end_index=2,
            name="test",
            key_source="#source",
            key_target="#target",
            verbose=False,
        )
    )
    expected_records = [
        {"#source": 0, "#target": 1, "weight": 0.5},
        {"#source": 1, "#target": 2, "weight": 0.6},
    ]
    self.assertEqual(records, expected_records)

  def test_read_avro_record(self):
    schema = {
        "type": "record",
        "name": "Test",
        "fields": [
            {"name": "f1", "type": "long"},
            {"name": "f2", "type": "float"},
        ],
    }
    records = [
        {"f1": 1, "f2": 1.0},
        {"f1": 2, "f2": 2.0},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.avro")
      with open(path, "wb") as f:
        fastavro.writer(f, schema, records)

      columns = {
          "f1": ("int64", ()),
          "f2": ("float32", ()),
      }
      data, num_records = avro_lib.read_avro_record([path], columns, False)  # pyrefly: ignore[bad-argument-type]

      self.assertEqual(num_records, 2)
      np.testing.assert_array_equal(data["f1"], np.array([1, 2], dtype="int64"))
      np.testing.assert_array_equal(
          data["f2"], np.array([1.0, 2.0], dtype="float32")
      )

  def test_read_avro_record_sharded(self):
    schema = {
        "type": "record",
        "name": "Test",
        "fields": [{"name": "f1", "type": "long"}],
    }
    records1 = [{"f1": 1}, {"f1": 2}]
    records2 = [{"f1": 3}]

    with tempfile.TemporaryDirectory() as tmpdir:
      path1 = os.path.join(tmpdir, "shard1.avro")
      path2 = os.path.join(tmpdir, "shard2.avro")
      with open(path1, "wb") as f:
        fastavro.writer(f, schema, records1)
      with open(path2, "wb") as f:
        fastavro.writer(f, schema, records2)

      columns = {"f1": ("int64", ())}
      data, num_records = avro_lib.read_avro_record(
          [path1, path2], columns, False  # pyrefly: ignore[bad-argument-type]
      )

      self.assertEqual(num_records, 3)
      np.testing.assert_array_equal(data["f1"], np.array([1, 2, 3]))

  def test_read_avro_record_empty(self):
    schema = {
        "type": "record",
        "name": "Test",
        "fields": [{"name": "f1", "type": "long"}],
    }
    records = []
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.avro")
      with open(path, "wb") as f:
        fastavro.writer(f, schema, records)

      columns = {"f1": ("int64", ())}
      data, num_records = avro_lib.read_avro_record([path], columns, False)  # pyrefly: ignore[bad-argument-type]

      self.assertEqual(num_records, 0)
      self.assertEqual(data["f1"].shape, (0,))
      self.assertEqual(data["f1"].dtype, "int64")

  def test_write_avro_node_sets(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=True, variable_length=False
      )
      schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=True, variable_length=False
      )
      avro_lib.write_avro_node_sets(
          graph, schema, tmpdir, extension=".avro", verbose=False
      )

      # Check n1 nodes
      n1_path = os.path.join(tmpdir, "n1-00000-of-00001.avro")
      self.assertTrue(os.path.exists(n1_path))
      n1_cols = {
          "#id": ("bytes", ()),
          "f1": ("bytes", (1,)),
          "f2": ("float32", (2,)),
      }
      n1_data, n1_num_records = avro_lib.read_avro_record(
          [n1_path], n1_cols, False  # pyrefly: ignore[bad-argument-type]
      )
      self.assertEqual(n1_num_records, 2)
      np.testing.assert_array_equal(n1_data["#id"], np.array([b"1", b"2"]))
      np.testing.assert_array_equal(
          n1_data["f1"], np.array([[b"blue"], [b"red"]])
      )
      np.testing.assert_array_equal(
          n1_data["f2"], np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
      )

      # Check n2 nodes
      n2_path = os.path.join(tmpdir, "n2-00000-of-00001.avro")
      self.assertTrue(os.path.exists(n2_path))
      n2_cols = {"#id": ("int64", ()), "f3": ("int64", ()), "f4": ("int64", ())}
      n2_data, n2_num_records = avro_lib.read_avro_record(
          [n2_path], n2_cols, False  # pyrefly: ignore[bad-argument-type]
      )
      self.assertEqual(n2_num_records, 2)
      np.testing.assert_array_equal(n2_data["#id"], np.array([1, 2]))
      np.testing.assert_array_equal(n2_data["f3"], np.array([4, 5]))
      np.testing.assert_array_equal(n2_data["f4"], np.array([10, 11]))

  def test_write_avro_edge_sets(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=True, variable_length=False
      )
      schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=True, variable_length=False
      )
      avro_lib.write_avro_edge_sets(
          graph,
          schema,
          tmpdir,
          extension=".avro",
          node_id_column="#id",
          key_source="#source",
          key_target="#target",
          verbose=False,
      )

      # Check e1 edges
      e1_path = os.path.join(tmpdir, "e1-00000-of-00001.avro")
      self.assertTrue(os.path.exists(e1_path))
      e1_cols = {
          "#source": ("bytes", ()),
          "#target": ("bytes", ()),
          "#id": ("bytes", ()),
      }
      e1_data, e1_num_records = avro_lib.read_avro_record(
          [e1_path], e1_cols, False  # pyrefly: ignore[bad-argument-type]
      )
      self.assertEqual(e1_num_records, 2)
      np.testing.assert_array_equal(e1_data["#source"], np.array([b"1", b"1"]))
      np.testing.assert_array_equal(e1_data["#target"], np.array([b"1", b"2"]))
      np.testing.assert_array_equal(e1_data["#id"], np.array([b"a", b"b"]))

      # Check e2 edges
      e2_path = os.path.join(tmpdir, "e2-00000-of-00001.avro")
      self.assertTrue(os.path.exists(e2_path))
      e2_cols = {
          "#source": ("bytes", ()),
          "#target": ("int64", ()),
          "#id": ("bytes", ()),
      }
      e2_data, e2_num_records = avro_lib.read_avro_record(
          [e2_path], e2_cols, False  # pyrefly: ignore[bad-argument-type]
      )
      self.assertEqual(e2_num_records, 2)
      np.testing.assert_array_equal(e2_data["#source"], np.array([b"1", b"1"]))
      np.testing.assert_array_equal(e2_data["#target"], np.array([1, 2]))
      np.testing.assert_array_equal(e2_data["#id"], np.array([b"A", b"B"]))


if __name__ == "__main__":
  absltest.main()
