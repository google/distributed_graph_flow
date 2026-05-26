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

"""Tests for TF Graph Sample Extensions."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.io import tf_graph_sample
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import tensorflow as tf


class TfGraphExtTest(parameterized.TestCase):

  def test_tf_sample_serialize(self):
    schema = gen_test_graph.generate_schema(variable_length=False)
    graph = gen_test_graph.generate_in_memory_graph(variable_length=False)
    py_example = tf_graph_sample.graph_to_tfgnn_graph(graph, schema=schema)
    serialized_graph = tf_graph_sample.graph_to_serialized_tfgnn_graph(graph)

    ex = tf.train.Example()
    ex.ParseFromString(serialized_graph)
    test_util.assertProto2Equal(self, py_example, ex)

  @parameterized.named_parameters(
      {
          'testcase_name': 'single_thread',
          'num_threads': -1,
      },
      {
          'testcase_name': 'multithreaded',
          'num_threads': 2,
      },
  )
  def test_tf_sample_serialize_list(self, num_threads):
    schema = gen_test_graph.generate_schema(variable_length=False)

    expected_graphs = []
    graphs = []
    for _ in range(10):
      graph = gen_test_graph.generate_in_memory_graph(variable_length=False)
      graphs.append(graph)
      expected_graphs.append(
          tf_graph_sample.graph_to_tfgnn_graph(graph, schema=schema)
      )

    serialized_graphs = tf_graph_sample.graphs_to_serialized_tfgnn_graphs(graphs, num_threads)
    self.assertLen(serialized_graphs, len(expected_graphs))
    for serialized_graph, expected_graph in zip(
        serialized_graphs, expected_graphs
    ):
      ex = tf.train.Example()
      ex.ParseFromString(serialized_graph)
      test_util.assertProto2Equal(self, expected_graph, ex)


if __name__ == '__main__':
  absltest.main()
