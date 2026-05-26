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

"""Test converting between heterogeneous graph edge types."""

from absl.testing import absltest
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import padding as padding_lib
from dgf.src.data import tf_in_memory_graph
from dgf.src.transform import merge as merge_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np
import tensorflow as tf


class BatchTest(absltest.TestCase):

  def test_batch_with_padding(self):
    graphs = [
        gen_test_graph.generate_in_memory_graph(False, False),
        gen_test_graph.generate_in_memory_graph(False, False),
    ]
    schema = gen_test_graph.generate_schema(False, False, variable_length=False)
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(num_nodes=5 + 1),
            "n2": padding_lib.NodeSetPadding(num_nodes=6 + 1),
        },
        edge_sets={
            "e1": padding_lib.EdgeSetPadding(num_edges=5),
            "e2": padding_lib.EdgeSetPadding(num_edges=6),
        },
    )
    merged_graph, offsets = merge_lib.merge_graphs(graphs, schema, padding)
    expected_merged_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                features={
                    "f1": np.array([
                        [b"blue"],
                        [b"red"],
                        [b"blue"],
                        [b"red"],
                        [b""],
                        [b""],
                    ]),
                    "f2": np.array([
                        [0.0, 1.0],
                        [2.0, 3.0],
                        [0.0, 1.0],
                        [2.0, 3.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]),
                },
                num_nodes=5 + 1,
            ),
            "n2": in_memory_graph_lib.InMemoryNodeSet(
                features={
                    "f3": np.array([4, 5, 4, 5, 0, 0, 0]),
                    "f4": np.array([10, 11, 10, 11, 0, 0, 0]),
                },
                num_nodes=6 + 1,
            ),
        },
        edge_sets={
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([
                    [0, 0, 2, 2, 5],
                    [0, 1, 2, 3, 5],
                ]),
                features={},
            ),
            "e2": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([
                    [0, 0, 2, 2, 5, 5],
                    [0, 1, 2, 3, 6, 6],
                ]),
                features={},
            ),
        },
    )
    test_util.assert_are_equal(
        self, offsets, {"n2": np.array([0, 2, 4]), "n1": np.array([0, 2, 4])}
    )
    test_util.assert_are_equal(self, merged_graph, expected_merged_graph)

  def test_batch_no_padding(self):
    graphs = [
        gen_test_graph.generate_in_memory_graph(False, False),
        gen_test_graph.generate_in_memory_graph(False, False),
    ]
    schema = gen_test_graph.generate_schema(False, False, variable_length=False)
    merged_graph, offsets = merge_lib.merge_graphs(graphs, schema, padding=None)
    expected_merged_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                features={
                    "f1": np.array([
                        [b"blue"],
                        [b"red"],
                        [b"blue"],
                        [b"red"],
                    ]),
                    "f2": np.array([
                        [0.0, 1.0],
                        [2.0, 3.0],
                        [0.0, 1.0],
                        [2.0, 3.0],
                    ]),
                },
                num_nodes=4,
            ),
            "n2": in_memory_graph_lib.InMemoryNodeSet(
                features={
                    "f3": np.array([4, 5, 4, 5]),
                    "f4": np.array([10, 11, 10, 11]),
                },
                num_nodes=4,
            ),
        },
        edge_sets={
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([
                    [0, 0, 2, 2],
                    [0, 1, 2, 3],
                ]),
                features={},
            ),
            "e2": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([
                    [0, 0, 2, 2],
                    [0, 1, 2, 3],
                ]),
                features={},
            ),
        },
    )
    test_util.assert_are_equal(
        self, offsets, {"n2": np.array([0, 2, 4]), "n1": np.array([0, 2, 4])}
    )
    test_util.assert_are_equal(self, merged_graph, expected_merged_graph)

  def test_batch_padding_too_small(self):
    graphs = [
        # Each nodeset has 2 nodes, each edgesets has 2 edges.
        gen_test_graph.generate_in_memory_graph(False, False),
        gen_test_graph.generate_in_memory_graph(False, False),
    ]
    schema = gen_test_graph.generate_schema(False, False, variable_length=False)
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(
                num_nodes=4 + 0
            ),  # Not enough for the sentinel.
            "n2": padding_lib.NodeSetPadding(num_nodes=6 + 1),
        },
        edge_sets={
            "e1": padding_lib.EdgeSetPadding(num_edges=5),
            "e2": padding_lib.EdgeSetPadding(num_edges=6),
        },
    )
    with self.assertRaisesRegex(
        merge_lib.InsufficientPaddingError,
        r"Required at least 5 nodes \(including the sentinel node\), but the"
        r" padder only defines 4.",
    ):
      _ = merge_lib.merge_graphs(graphs, schema, padding)

  def test_batch_with_padding_no_sentinel_offset(self):
    graphs = [
        gen_test_graph.generate_in_memory_graph(False, False),
        gen_test_graph.generate_in_memory_graph(False, False),
    ]
    schema = gen_test_graph.generate_schema(False, False, variable_length=False)
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(num_nodes=5 + 1),
            "n2": padding_lib.NodeSetPadding(num_nodes=6 + 1),
        },
        edge_sets={
            "e1": padding_lib.EdgeSetPadding(num_edges=5),
            "e2": padding_lib.EdgeSetPadding(num_edges=6),
        },
    )
    merged_graph, offsets = merge_lib.merge_graphs(
        graphs, schema, padding, sentinel_offset=False
    )
    expected_merged_graph, _ = merge_lib.merge_graphs(
        graphs, schema, padding, sentinel_offset=True
    )
    test_util.assert_are_equal(self, merged_graph, expected_merged_graph)
    test_util.assert_are_equal(
        self, offsets, {"n2": np.array([0, 2]), "n1": np.array([0, 2])}
    )

  def test_batch_no_padding_no_sentinel_offset(self):
    graphs = [
        gen_test_graph.generate_in_memory_graph(False, False),
        gen_test_graph.generate_in_memory_graph(False, False),
    ]
    schema = gen_test_graph.generate_schema(False, False, variable_length=False)
    merged_graph, offsets = merge_lib.merge_graphs(
        graphs, schema, padding=None, sentinel_offset=False
    )
    expected_merged_graph, _ = merge_lib.merge_graphs(
        graphs, schema, padding=None, sentinel_offset=True
    )
    test_util.assert_are_equal(self, merged_graph, expected_merged_graph)
    test_util.assert_are_equal(
        self, offsets, {"n2": np.array([0, 2]), "n1": np.array([0, 2])}
    )

  def test_pad_graph_tensorflow(self):
    tf_graph = gen_test_graph.generate_tf_in_memory_graph(
        variable_length=False,
        tensor_type="DENSE",
        num_nodes_as_tensor=True,
    )
    schema = gen_test_graph.generate_schema(False, False, variable_length=False)
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(num_nodes=5 + 1),
            "n2": padding_lib.NodeSetPadding(num_nodes=6 + 1),
        },
        edge_sets={
            "e1": padding_lib.EdgeSetPadding(num_edges=5),
            "e2": padding_lib.EdgeSetPadding(num_edges=6),
        },
    )

    @tf.function(autograph=False)
    def pad(graph):
      return merge_lib.pad_graph_tensorflow(graph, schema, padding)

    merged_graph = pad(tf_graph)

    expected_merged_graph = tf_in_memory_graph.TFInMemoryGraph(
        node_sets={
            "n1": tf_in_memory_graph.TFInMemoryNodeSet(
                features={
                    "f1": tf.constant([
                        [b"blue"],
                        [b"red"],
                        [b""],
                        [b""],
                        [b""],
                        [b""],
                    ]),
                    "f2": tf.constant([
                        [0.0, 1.0],
                        [2.0, 3.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]),
                },
                num_nodes=5 + 1,
            ),
            "n2": tf_in_memory_graph.TFInMemoryNodeSet(
                features={
                    "f3": tf.constant([4, 5, 0, 0, 0, 0, 0], dtype=tf.int64),
                    "f4": tf.constant([10, 11, 0, 0, 0, 0, 0], dtype=tf.int64),
                },
                num_nodes=6 + 1,
            ),
        },
        edge_sets={
            "e1": tf_in_memory_graph.TFInMemoryEdgeSet(
                adjacency=tf.constant(
                    [
                        [0, 0, 5, 5, 5],
                        [0, 1, 5, 5, 5],
                    ],
                    dtype=tf.int64,
                ),
                features={},
            ),
            "e2": tf_in_memory_graph.TFInMemoryEdgeSet(
                adjacency=tf.constant(
                    [
                        [0, 0, 5, 5, 5, 5],
                        [0, 1, 6, 6, 6, 6],
                    ],
                    dtype=tf.int64,
                ),
                features={},
            ),
        },
    )
    test_util.assert_are_equal(self, merged_graph, expected_merged_graph)

    for node_set_name, node_set in merged_graph.node_sets.items():
      for feature_name, feature in node_set.features.items():
        self.assertTrue(
            feature.shape.is_fully_defined(),
            f"Feature {feature_name} in nodeset {node_set_name} should have"
            " fully defined shape.",
        )
    for edge_set_name, edge_set in merged_graph.edge_sets.items():
      for feature_name, feature in edge_set.features.items():
        self.assertTrue(
            feature.shape.is_fully_defined(),
            f"Feature {feature_name} in edgeset {edge_set_name} should have"
            " fully defined shape.",
        )
      self.assertTrue(
          edge_set.adjacency.shape.is_fully_defined(),
          f"Adjacency in edgeset {edge_set_name} should have fully defined"
          " shape.",
      )

  def test_remove_padding_sentinels_with_padding(self):
    graphs = [
        gen_test_graph.generate_in_memory_graph(False, False),
        gen_test_graph.generate_in_memory_graph(False, False),
    ]
    schema = gen_test_graph.generate_schema(False, False, variable_length=False)
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(num_nodes=5 + 1),
            "n2": padding_lib.NodeSetPadding(num_nodes=6 + 1),
        },
        edge_sets={
            "e1": padding_lib.EdgeSetPadding(num_edges=5),
            "e2": padding_lib.EdgeSetPadding(num_edges=6),
        },
    )
    merged_graph, offsets = merge_lib.merge_graphs(graphs, schema, padding)
    unpadded_graph = merge_lib.remove_padding_sentinels(
        merged_graph, schema, offsets
    )

    expected_unpadded_graph, _ = merge_lib.merge_graphs(
        graphs, schema, padding=None
    )
    test_util.assert_are_equal(self, unpadded_graph, expected_unpadded_graph)


if __name__ == "__main__":
  absltest.main()
