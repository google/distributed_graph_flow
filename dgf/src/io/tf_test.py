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

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.data import tf_in_memory_graph as tf_in_memory_graph_lib
from dgf.src.io import tf as tf_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import tensorflow as tf


class TfTest(parameterized.TestCase):

  def test_graph_to_tf_graph(self):
    in_memory_graph = gen_test_graph.generate_in_memory_graph(
        variable_length=False
    )
    expected_tf_in_memory_graph = gen_test_graph.generate_tf_in_memory_graph(
        variable_length=False,
        tensor_type="DENSE",
        num_nodes_as_tensor=True,
    )
    tf_in_memory_graph = tf_lib.graph_to_tf_graph(in_memory_graph)
    test_util.assert_are_equal(
        self, tf_in_memory_graph, expected_tf_in_memory_graph
    )

  def test_tf_graph_to_tf_graph_dict(self):
    tf_in_memory_graph = gen_test_graph.generate_tf_in_memory_graph(
        variable_length=False,
        tensor_type="DENSE",
        num_nodes_as_tensor=True,
    )
    graph_dict = tf_lib.tf_graph_to_tf_graph_dict(tf_in_memory_graph)

    # Verify some keys
    self.assertIn("nodes_n1_reserved_size", graph_dict)
    self.assertIn("nodes_n1_f1", graph_dict)
    self.assertIn("edges_e1_reserved_adjacency", graph_dict)

  def test_tf_graph_dict_to_tf_graph(self):
    tf_in_memory_graph = gen_test_graph.generate_tf_in_memory_graph(
        variable_length=False,
        tensor_type="DENSE",
        num_nodes_as_tensor=True,
    )
    graph_dict = tf_lib.tf_graph_to_tf_graph_dict(tf_in_memory_graph)
    reconstructed_graph = tf_lib.tf_graph_dict_to_tf_graph(graph_dict)

    test_util.assert_are_equal(self, reconstructed_graph, tf_in_memory_graph)

  def test_tf_graph_to_tf_graph_dict_with_underscores(self):
    tf_graph = tf_in_memory_graph_lib.TFInMemoryGraph(
        node_sets={
            "#my_nodeset": tf_in_memory_graph_lib.TFInMemoryNodeSet(
                num_nodes=tf.constant(2, dtype=tf.int32),
                features={"my_#feat": tf.constant([[1.0], [2.0]])},
            )
        },
        edge_sets={
            "#my_edgeset": tf_in_memory_graph_lib.TFInMemoryEdgeSet(
                adjacency=tf.constant([[0, 0], [0, 1]], dtype=tf.int64),
                features={"#my_edge_feat": tf.constant([0.5, 0.8])},
            )
        },
    )
    graph_dict = tf_lib.tf_graph_to_tf_graph_dict(tf_graph)

    h = f"{tf_lib.BEGIN_CODE}23{tf_lib.END_CODE}"
    u = f"{tf_lib.BEGIN_CODE}5f{tf_lib.END_CODE}"
    # Verify keys have both # and underscores replaced
    self.assertIn(f"nodes_{h}my{u}nodeset_reserved_size", graph_dict)
    self.assertIn(f"nodes_{h}my{u}nodeset_my{u}{h}feat", graph_dict)
    self.assertIn(f"edges_{h}my{u}edgeset_reserved_adjacency", graph_dict)
    self.assertIn(f"edges_{h}my{u}edgeset_{h}my{u}edge{u}feat", graph_dict)

    reconstructed_graph = tf_lib.tf_graph_dict_to_tf_graph(graph_dict)
    test_util.assert_are_equal(self, reconstructed_graph, tf_graph)

  def test_schema_to_spec_invalid_names(self):
    inv = f"{tf_lib.BEGIN_CODE}23{tf_lib.END_CODE}"
    # Node set name invalid
    schema = schema_lib.GraphSchema(
        node_sets={f"n_{inv}_1": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    with self.assertRaises(ValueError):
      tf_lib.schema_to_spec(schema)

    # Edge set name invalid
    schema = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={
            f"e_{inv}_1": schema_lib.EdgeSchema(
                source="n1", target="n1", features={}
            )
        },
    )
    with self.assertRaises(ValueError):
      tf_lib.schema_to_spec(schema)

    # Feature name invalid in node set
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    f"f_{inv}_1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    )
                }
            )
        },
        edge_sets={},
    )
    with self.assertRaises(ValueError):
      tf_lib.schema_to_spec(schema)

    # Feature name invalid in edge set
    schema = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    f"f_{inv}_1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    )
                },
            )
        },
    )
    with self.assertRaises(ValueError):
      tf_lib.schema_to_spec(schema)


if __name__ == "__main__":
  absltest.main()
