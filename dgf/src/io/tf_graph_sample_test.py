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
import apache_beam as beam
from apache_beam.testing import util
from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.io import tf_graph_sample
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
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
        for tensor in tf.data.TFRecordDataset(  # pyrefly: ignore[bad-instantiation]
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


if __name__ == "__main__":
  absltest.main()
