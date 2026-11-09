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

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import tf_in_memory_graph as tf_in_memory_graph_lib
from dgf.src.io import tf as tf_lib
from dgf.src.io import tf_graph_sample as tf_graph_sample_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np
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
    schema = schema_lib.GraphSchema(
        node_sets={
            "#my_nodeset": schema_lib.NodeSchema(
                features={
                    "my_#feat": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32, shape=(1,)
                    ),
                    "user_id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32, shape=(1,)
                    ),
                    "_leading_feat": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32, shape=(1,)
                    ),
                }
            )
        },
        edge_sets={
            "#my_edgeset": schema_lib.EdgeSchema(
                source="#my_nodeset",
                target="#my_nodeset",
                features={
                    "#my_edge_feat": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    )
                },
            )
        },
    )
    tf_graph = tf_in_memory_graph_lib.TFInMemoryGraph(
        node_sets={
            "#my_nodeset": tf_in_memory_graph_lib.TFInMemoryNodeSet(
                num_nodes=tf.constant(2, dtype=tf.int32),
                features={
                    "my_#feat": tf.constant([[1.0], [2.0]]),
                    "user_id": tf.constant([[10.0], [20.0]]),
                    "_leading_feat": tf.constant([[3.0], [4.0]]),
                },
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

    self.assertIn("nodes_#my__nodeset_reserved_size", graph_dict)
    self.assertIn("nodes_#my__nodeset_my__#feat", graph_dict)
    self.assertIn("nodes_#my__nodeset_user__id", graph_dict)
    self.assertIn("nodes_#my__nodeset___leading__feat", graph_dict)
    self.assertIn("edges_#my__edgeset_reserved_adjacency", graph_dict)
    self.assertIn("edges_#my__edgeset_#my__edge__feat", graph_dict)

    # Verify that ('a_b', 'c') and ('a', 'b_c') produce distinct keys.
    self.assertEqual(tf_lib.tf_graph_dict_edge_key("a_b", "c"), "edges_a__b_c")
    self.assertEqual(tf_lib.tf_graph_dict_edge_key("a", "b_c"), "edges_a_b__c")

    reconstructed_graph = tf_lib.tf_graph_dict_to_tf_graph(graph_dict)
    test_util.assert_are_equal(self, reconstructed_graph, tf_graph)

    dict_spec = tf_lib.schema_to_dict_spec(schema)
    self.assertEqual(set(dict_spec.keys()), set(graph_dict.keys()))

    with self.assertRaisesRegex(ValueError, "Unrecognized key"):
      tf_lib.tf_graph_dict_to_tf_graph({"invalid_key": tf.constant([1.0])})

  def test_tf_graph_dict_rejects_trailing_underscore_entity_names(self):
    with self.assertRaisesRegex(ValueError, "cannot end with '_'"):
      tf_lib.tf_graph_dict_node_key("client_", "feat")
    with self.assertRaisesRegex(ValueError, "cannot end with '_'"):
      tf_lib.tf_graph_dict_edge_key("edgeset_", "feat")
    with self.assertRaisesRegex(ValueError, "cannot be empty"):
      tf_lib.tf_graph_dict_node_key("", "feat")
    with self.assertRaisesRegex(ValueError, "cannot be empty"):
      tf_lib.tf_graph_dict_node_key("client", "")
    with self.assertRaisesRegex(ValueError, "Invalid entity name"):
      tf_lib.tf_graph_dict_to_tf_graph(
          {"nodes_client___reserved_size": tf.constant(2, dtype=tf.int32)}
      )
    with self.assertRaisesRegex(ValueError, "Unrecognized key"):
      tf_lib.tf_graph_dict_to_tf_graph({"nodes_a_b_c": tf.constant([1.0])})

  def test_tf_graph_dict_to_tf_graph_sanitized_keys(self):
    # "nodes_a__b__id" is the `tf.function`-sanitized key of both
    # ("a_b", "#id") and ("a", "#b_id"): the size keys disambiguate.
    graph = tf_lib.tf_graph_dict_to_tf_graph({
        "nodes_a__b_reserved_size": tf.constant(2, dtype=tf.int32),
        "nodes_a__b__id": tf.constant([b"x", b"y"]),
        "nodes_a__b__idx": tf.constant([0, 1], dtype=tf.int64),
    })
    self.assertEqual(set(graph.node_sets), {"a_b"})
    self.assertEqual(set(graph.node_sets["a_b"].features), {"#id", "#idx"})

    with self.assertRaisesRegex(ValueError, "Ambiguous sanitized key"):
      tf_lib.tf_graph_dict_to_tf_graph({"nodes_a__b__id": tf.constant([b"x"])})

  def test_tf_graph_dict_to_tf_graph_in_tf_function_with_hash_features(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "my_nodeset": schema_lib.NodeSchema(
                features={
                    "#id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES
                    ),
                    "#idx": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64
                    ),
                    "#propagated_creation_time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64
                    ),
                    "user_id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    ),
                }
            )
        },
        edge_sets={
            "my_edgeset": schema_lib.EdgeSchema(
                source="my_nodeset",
                target="my_nodeset",
                features={
                    "#my_edge_feat": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    )
                },
            )
        },
    )
    tf_graph = tf_in_memory_graph_lib.TFInMemoryGraph(
        node_sets={
            "my_nodeset": tf_in_memory_graph_lib.TFInMemoryNodeSet(
                num_nodes=tf.constant(2, dtype=tf.int32),
                features={
                    "#id": tf.constant([b"a", b"b"]),
                    "#idx": tf.constant([0, 1], dtype=tf.int64),
                    "#propagated_creation_time": tf.constant(
                        [100, 200], dtype=tf.int64
                    ),
                    "user_id": tf.constant([1.0, 2.0], dtype=tf.float32),
                },
            )
        },
        edge_sets={
            "my_edgeset": tf_in_memory_graph_lib.TFInMemoryEdgeSet(
                adjacency=tf.constant([[0, 0], [0, 1]], dtype=tf.int64),
                features={
                    "#my_edge_feat": tf.constant([0.5, 0.8], dtype=tf.float32)
                },
            )
        },
    )

    @tf.function(autograph=False)
    def trace_fn(**kwargs):
      # `tf.function` sanitizes the '#' of the parameter names to '_'.
      self.assertNotIn("nodes_my__nodeset_#id", kwargs)
      self.assertIn("nodes_my__nodeset__id", kwargs)
      reconstructed = tf_lib.tf_graph_dict_to_tf_graph(kwargs)
      self.assertEqual(set(reconstructed.node_sets), {"my_nodeset"})
      self.assertEqual(set(reconstructed.edge_sets), {"my_edgeset"})
      node_set = reconstructed.node_sets["my_nodeset"]
      self.assertEqual(
          set(node_set.features),
          {"#id", "#idx", "#propagated_creation_time", "user_id"},
      )
      edge_set = reconstructed.edge_sets["my_edgeset"]
      self.assertEqual(set(edge_set.features), {"#my_edge_feat"})
      return (
          node_set.features["#id"],
          node_set.features["#idx"]
          + node_set.features["#propagated_creation_time"],
          edge_set.features["#my_edge_feat"],
      )

    concrete_fn = trace_fn.get_concrete_function(
        **tf_lib.schema_to_dict_spec(schema)
    )
    ids, idxs_plus_times, edge_feat = concrete_fn(
        **tf_lib.tf_graph_to_tf_graph_dict(tf_graph)
    )
    np.testing.assert_array_equal(ids.numpy(), np.array([b"a", b"b"]))
    np.testing.assert_array_equal(idxs_plus_times.numpy(), np.array([100, 201]))
    np.testing.assert_allclose(edge_feat.numpy(), np.array([0.5, 0.8]))


class TfgnnGraphSampleTest(parameterized.TestCase):
  """Tests of the conversion from TF GNN Graph Samples to TFInMemoryGraphs."""

  def test_schema_to_tfgnn_graph_parsing_spec(self):
    schema = gen_test_graph.generate_schema(variable_length=True)
    spec = tf_lib.schema_to_tfgnn_graph_parsing_spec(schema)

    self.assertEqual(
        set(spec.keys()),
        {
            "nodes/n1.#size",
            "nodes/n1.f1",
            "nodes/n1.f2",
            "nodes/n2.#size",
            "nodes/n2.f3",
            "nodes/n2.f4",
            "nodes/n2.f5",
            "nodes/n2.f5.d1",
            "nodes/n2.f6",
            "nodes/n2.f6.d1",
            "edges/e1.#size",
            "edges/e1.#source",
            "edges/e1.#target",
            "edges/e2.#size",
            "edges/e2.#source",
            "edges/e2.#target",
        },
    )
    # Only the dtypes supported by `tf.io.parse_example` are used.
    self.assertEqual(spec["nodes/n1.f1"].dtype, tf.string)
    self.assertEqual(spec["nodes/n1.f2"].dtype, tf.float32)
    self.assertEqual(spec["nodes/n2.f3"].dtype, tf.int64)
    self.assertEqual(spec["nodes/n2.f5.d1"].dtype, tf.int64)

  def test_tfgnn_graph_dict_to_tf_graph(self):
    schema = gen_test_graph.generate_schema(variable_length=True)
    example = gen_test_graph.generate_tf_graph_sample(
        node_ids=False, edge_ids=False, variable_length=True
    )
    expected_graph = gen_test_graph.generate_tf_in_memory_graph(
        variable_length=True,
        tensor_type="NATURAL",
        num_nodes_as_tensor=True,
    )

    graph = tf_lib.serialized_tfgnn_graph_to_tf_graph(
        tf.constant(example.SerializeToString()), schema
    )

    test_util.assert_are_equal(self, graph, expected_graph)
    # The multi-dimensional ragged feature "f6" keeps its inner dimension.
    self.assertEqual(
        graph.node_sets["n2"].features["f6"].to_list(),
        [[[11, 12], [13, 14]], [[15, 16], [17, 18], [19, 20]]],
    )

  def test_tfgnn_graph_dict_to_tf_graph_round_trip(self):
    schema = gen_test_graph.generate_schema(variable_length=True)
    expected_graph = gen_test_graph.generate_in_memory_graph(
        variable_length=True
    )
    graph_dict = tf_graph_sample_lib.graph_to_tfgnn_graph_dict(
        expected_graph, schema
    )

    tf_graph = tf_lib.tfgnn_graph_dict_to_tf_graph(
        {k: tf.constant(v) for k, v in graph_dict.items()}, schema
    )

    test_util.assert_are_equal(
        self,
        tf_graph,
        gen_test_graph.generate_tf_in_memory_graph(
            variable_length=True,
            tensor_type="NATURAL",
            num_nodes_as_tensor=True,
        ),
    )

  def test_tfgnn_graph_dict_to_tf_graph_casts_dtypes(self):
    """Values are cast from their storage dtype to the schema dtype."""
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "int32": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_32
                    ),
                    "float64": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64
                    ),
                    "bool": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BOOL
                    ),
                }
            )
        },
        edge_sets={},
    )
    graph_dict = {
        "nodes/n1.#size": tf.constant([2], dtype=tf.int64),
        "nodes/n1.int32": tf.constant([1, 2], dtype=tf.int64),
        "nodes/n1.float64": tf.constant([1.0, 2.0], dtype=tf.float32),
        "nodes/n1.bool": tf.constant([0, 1], dtype=tf.int64),
    }

    graph = tf_lib.tfgnn_graph_dict_to_tf_graph(graph_dict, schema)

    features = graph.node_sets["n1"].features
    self.assertEqual(features["int32"].dtype, tf.int32)
    self.assertEqual(features["float64"].dtype, tf.float64)
    self.assertEqual(features["bool"].dtype, tf.bool)
    test_util.assert_are_equal(
        self, features["bool"], tf.constant([False, True])
    )

  def test_tfgnn_graph_dict_to_tf_graph_ragged_bytes(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES, shape=(None,)
                    )
                }
            )
        },
        edge_sets={},
    )
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "f1": np.array(
                        [
                            np.array([b"a", b"b"]),
                            np.array([b"c"]),
                        ],
                        dtype=np.object_,
                    )
                },
            )
        },
        edge_sets={},
    )
    example = tf_graph_sample_lib.graph_to_tfgnn_graph(graph, schema)

    tf_graph = tf_lib.serialized_tfgnn_graph_to_tf_graph(
        tf.constant(example.SerializeToString()), schema
    )

    self.assertEqual(
        tf_graph.node_sets["n1"].features["f1"].to_list(),
        [[b"a", b"b"], [b"c"]],
    )
    # The numpy and C++ readers return the same values.
    test_util.assert_are_equal(
        self,
        tf_graph_sample_lib.tfgnn_graph_to_graph(example, schema),
        graph,
    )
    test_util.assert_are_equal(
        self,
        tf_graph_sample_lib.create_tfgnn_graph_parser(schema).parse(
            [example.SerializeToString()]
        ),
        [graph],
    )

  def test_tfgnn_graph_dict_to_tf_graph_empty_sets(self):
    """A missing "#size" means an empty node/edge set."""
    schema = gen_test_graph.generate_schema(variable_length=True)
    graph_dict = tf.io.parse_single_example(
        tf.train.Example().SerializeToString(),
        tf_lib.schema_to_tfgnn_graph_parsing_spec(schema),
    )

    graph = tf_lib.tfgnn_graph_dict_to_tf_graph(graph_dict, schema)

    self.assertEqual(int(graph.node_sets["n1"].num_nodes), 0)
    self.assertEqual(int(graph.node_sets["n2"].num_nodes), 0)
    self.assertEqual(graph.edge_sets["e1"].adjacency.shape, (2, 0))
    self.assertEmpty(graph.node_sets["n2"].features["f5"].to_list())

  def test_tfgnn_graph_dict_to_tf_graph_missing_key(self):
    schema = gen_test_graph.generate_schema(variable_length=True)
    with self.assertRaisesRegex(ValueError, "is missing from the TF GNN"):
      tf_lib.tfgnn_graph_dict_to_tf_graph({}, schema)

  def test_tfgnn_graph_dict_to_tf_graph_in_tf_function(self):
    """The conversion is exportable in a TF SavedModel."""
    schema = gen_test_graph.generate_schema(variable_length=True)
    parsing_spec = tf_lib.schema_to_tfgnn_graph_parsing_spec(schema)

    class Module(tf.Module):

      @tf.function(
          input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)],
          autograph=False,
      )
      def __call__(self, serialized_example: tf.Tensor) -> tf.Tensor:
        graph = tf_lib.tfgnn_graph_dict_to_tf_graph(
            tf.io.parse_single_example(serialized_example, parsing_spec),
            schema,
        )
        # The number of values of the ragged "f5" feature.
        return tf.size(graph.node_sets["n2"].features["f5"].flat_values)

    module = Module()
    with tempfile.TemporaryDirectory() as tmpdir:
      tf.saved_model.save(module, tmpdir)
      loaded = tf.saved_model.load(tmpdir)

      # Note: The two graphs have a different number of nodes: the traced
      # function should support graphs of any size.
      small_example = gen_test_graph.generate_tf_graph_sample(
          node_ids=False, edge_ids=False, variable_length=True
      )
      self.assertEqual(
          loaded(tf.constant(small_example.SerializeToString())).numpy(), 5
      )
      self.assertEqual(
          loaded(tf.constant(tf.train.Example().SerializeToString())).numpy(), 0
      )


if __name__ == "__main__":
  absltest.main()
