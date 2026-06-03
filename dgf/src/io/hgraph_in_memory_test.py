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
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import hgraph_in_memory
from dgf.src.io import tfexample as tfexample_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import proto as proto_lib
from dgf.src.util import test_util
import numpy as np
import tensorflow as tf
from tensorflow_gnn import proto as tf_gnn_proto

test_util.disable_diff_truncation()
Edge = distributed_graph.Edge


class ConvertSchemaTest(absltest.TestCase):

  def test_import_tf_gnn_schema(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "hgraph")
      gen_test_graph.generate_hgraph(path)

      # Convert the tfgnn schema into a schema
      tfgnn_schema = proto_lib.read_text_proto(
          os.path.join(path, "graph_schema.pbtxt"),
          tf_gnn_proto.GraphSchema,
      )
      schema = hgraph_in_memory.tfgnn_schema_to_schema(tfgnn_schema)

      # Check value
      expected_schema = gen_test_graph.generate_schema(node_ids=False)
      for nodeset_def in expected_schema.node_sets.values():
        nodeset_def.features["#id"] = schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BYTES,
            semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
        )
      self.assertEqual(schema, expected_schema)

  def test_export_tf_gnn_schema(self):
    schema = gen_test_graph.generate_schema(node_ids=True, semantic=False)
    for nodeset_def in schema.node_sets.values():
      nodeset_def.features["#id"].semantic = (
          schema_lib.FeatureSemantic.PRIMARY_ID
      )
    tfgnn_schema = hgraph_in_memory.schema_to_tfgnn_schema(schema)
    schema_again = hgraph_in_memory.tfgnn_schema_to_schema(tfgnn_schema)
    self.assertEqual(schema, schema_again)


class ReadHGraphInMemoryTest(absltest.TestCase):

  # TODO(gbm): Add parametrization with nodeids=True.
  def test_read_graphai_hgraph_tf_record(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "hgraph")
      gen_test_graph.generate_hgraph(path, node_id=True, variable_length=False)

      graph, schema = hgraph_in_memory.read_graphai_hgraph(path)

      # Test schema
      expected_schema = gen_test_graph.generate_schema(
          node_ids=True, variable_length=False
      )
      for nodeset_def in expected_schema.node_sets.values():
        nodeset_def.features["#id"].semantic = (
            schema_lib.FeatureSemantic.PRIMARY_ID
        )
      test_util.assert_are_equal(self, schema, expected_schema)

      # Test graph
      expected_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, variable_length=False
      )
      test_util.assert_are_equal(self, graph, expected_graph)

  def test_read_graphai_hgraph_avro(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "avro_hgraph")
      gen_test_graph.generate_avro_graph(
          path, node_id=True, variable_length=False
      )

      graph, schema = hgraph_in_memory.read_graphai_hgraph(
          path, container_type=hgraph_in_memory.HGraphContainerType.AVRO
      )

      # Test schema
      expected_schema = gen_test_graph.generate_schema(
          node_ids=True, variable_length=False
      )
      for nodeset_def in expected_schema.node_sets.values():
        nodeset_def.features["#id"].semantic = (
            schema_lib.FeatureSemantic.PRIMARY_ID
        )
      test_util.assert_are_equal(self, schema, expected_schema)

      # Test graph
      expected_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, variable_length=False
      )
      test_util.assert_are_equal(self, graph, expected_graph)


class WriteHGraphTest(parameterized.TestCase):

  # NOTE: node_ids is required for in-memory graph to HGraph writing.
  @parameterized.product(node_ids=[True], edge_ids=[True, False])
  def test_write_in_memory_graph_to_hgraph_tf_record(
      self, node_ids: bool, edge_ids: bool
  ):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate a toy in-memory graph
      output_path = os.path.join(tmpdir, "output_hgraph_in_memory")
      in_memory_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=node_ids, edge_ids=edge_ids, variable_length=False
      )
      hgraph_in_memory.write_graphai_hgraph(
          in_memory_graph,
          # Set variable_length to False because read_graphai_hgraph
          # does not support shape of (None,).
          gen_test_graph.generate_schema(
              node_ids=node_ids, edge_ids=edge_ids, variable_length=False
          ),
          output_path,
      )
      output_in_memory_graph, output_schema = (
          hgraph_in_memory.read_graphai_hgraph(
              output_path, edge_id_column="#id" if edge_ids else None
          )
      )
      test_util.assert_are_equal(self, output_in_memory_graph, in_memory_graph)
      expected_schema = gen_test_graph.generate_schema(
          node_ids=node_ids, edge_ids=edge_ids, variable_length=False
      )
      for nodeset_def in expected_schema.node_sets.values():
        if "#id" in nodeset_def.features:
          nodeset_def.features["#id"].semantic = (
              schema_lib.FeatureSemantic.PRIMARY_ID
          )
      for edgeset_def in expected_schema.edge_sets.values():
        if "#id" in edgeset_def.features:
          edgeset_def.features["#id"].semantic = (
              schema_lib.FeatureSemantic.PRIMARY_ID
          )
      test_util.assert_are_equal(self, output_schema, expected_schema)

      expected_files = [
          "/graph_schema.pbtxt",
          "/node_features/n1-00000-of-00001.tfrecord.gz",
          "/node_features/n2-00000-of-00001.tfrecord.gz",
          "/edges/e1-00000-of-00001.tfrecord.gz",
          "/edges/e2-00000-of-00001.tfrecord.gz",
      ]
      actual_files = []
      for dirpath, _, filenames in os.walk(output_path):
        for filename in filenames:
          actual_files.append(
              os.path.join(dirpath, filename).removeprefix(output_path)
          )
      self.assertSameElements(sorted(actual_files), sorted(expected_files))

  @parameterized.product(node_ids=[True], edge_ids=[True, False])
  def test_write_in_memory_graph_to_hgraph_avro(
      self, node_ids: bool, edge_ids: bool
  ):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate a toy in-memory graph
      output_path = os.path.join(tmpdir, "output_hgraph_in_memory_avro")
      in_memory_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=node_ids, edge_ids=edge_ids, variable_length=False
      )
      hgraph_in_memory.write_graphai_hgraph(
          in_memory_graph,
          # Set variable_length to False because read_graphai_hgraph
          # does not support shape of (None,).
          gen_test_graph.generate_schema(
              node_ids=node_ids, edge_ids=edge_ids, variable_length=False
          ),
          output_path,
          container_type=hgraph_in_memory.HGraphContainerType.AVRO,
      )
      output_in_memory_graph, output_schema = (
          hgraph_in_memory.read_graphai_hgraph(
              output_path,
              edge_id_column="#id" if edge_ids else None,
              container_type=hgraph_in_memory.HGraphContainerType.AVRO,
          )
      )
      test_util.assert_are_equal(self, output_in_memory_graph, in_memory_graph)
      expected_schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=edge_ids, variable_length=False
      )
      for nodeset_def in expected_schema.node_sets.values():
        if "#id" in nodeset_def.features:
          nodeset_def.features["#id"].semantic = (
              schema_lib.FeatureSemantic.PRIMARY_ID
          )
      for edgeset_def in expected_schema.edge_sets.values():
        if "#id" in edgeset_def.features:
          edgeset_def.features["#id"].semantic = (
              schema_lib.FeatureSemantic.PRIMARY_ID
          )
      test_util.assert_are_equal(self, output_schema, expected_schema)

      expected_files = [
          "/graph_schema.pbtxt",
          "/node_features/n1-00000-of-00001.avro",
          "/node_features/n2-00000-of-00001.avro",
          "/edges/e1-00000-of-00001.avro",
          "/edges/e2-00000-of-00001.avro",
      ]
      actual_files = []
      for dirpath, _, filenames in os.walk(output_path):
        for filename in filenames:
          actual_files.append(
              os.path.join(dirpath, filename).removeprefix(output_path)
          )
      self.assertSameElements(sorted(actual_files), sorted(expected_files))


if __name__ == "__main__":
  absltest.main()
