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

import dataclasses
from absl.testing import absltest
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import tf_in_memory_graph
from dgf.src.transform import merge as merge_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import temporal as temporal_util
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
    merged_graph, offsets = merge_lib.merge_graphs(
        graphs, schema, padding=padding
    )
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
      _ = merge_lib.merge_graphs(graphs, schema, padding=padding)

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
        graphs, schema, sentinel_offset=False, padding=padding
    )
    expected_merged_graph, _ = merge_lib.merge_graphs(
        graphs, schema, sentinel_offset=True, padding=padding
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

  def test_pad_graph_tensorflow_timeseries_unsupported(self):
    tf_graph = gen_test_graph.generate_tf_in_memory_graph(
        variable_length=False,
        tensor_type="DENSE",
        num_nodes_as_tensor=True,
    )
    schema = gen_test_graph.generate_schema(
        False, False, variable_length=False
    )
    schema.node_sets["n1"].features["f1"] = dataclasses.replace(
        schema.node_sets["n1"].features["f1"], is_timeseries=True
    )
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(
                num_nodes=5,
                features={
                    "f1": padding_lib.FeaturePadding(max_timeseries_len=10)
                },
            )
        },
        edge_sets={},
    )
    with self.assertRaisesRegex(
        NotImplementedError, "does not support timeseries padding"
    ):
      merge_lib.pad_graph_tensorflow(tf_graph, schema, padding)

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
    merged_graph, offsets = merge_lib.merge_graphs(
        graphs, schema, padding=padding
    )
    unpadded_graph = merge_lib.remove_padding_sentinels(
        merged_graph, schema, offsets
    )

    expected_unpadded_graph, _ = merge_lib.merge_graphs(
        graphs, schema, padding=None
    )
    test_util.assert_are_equal(self, unpadded_graph, expected_unpadded_graph)

  def test_batch_with_timeseries_padding(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        shape=(None,),
                    ),
                }
            ),
        },
        edge_sets={},
    )
    g1 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "ts": np.array(
                        [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])],
                        dtype=object,
                    ),
                },
            ),
        },
        edge_sets={},
    )
    g2 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1,
                features={
                    "ts": np.array(
                        [np.array([6.0, 7.0, 8.0, 9.0, 10.0])], dtype=object
                    ),
                },
            ),
        },
        edge_sets={},
    )
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(
                num_nodes=4 + 1,
                features={
                    "ts": padding_lib.FeaturePadding(max_timeseries_len=3)
                },
            )
        },
        edge_sets={},
    )
    # Check that schema cache is no longer required and can be omitted.
    merged_graph_no_cache, _ = merge_lib.merge_graphs(
        [g1, g2], schema, padding=padding, schema_cache=None
    )
    schema_cache = temporal_util.extract_timeseries_schema_cache(schema)
    merged_graph, _ = merge_lib.merge_graphs(
        [g1, g2],
        schema,
        padding=padding,
        schema_cache=schema_cache,
    )
    test_util.assert_are_equal(self, merged_graph_no_cache, merged_graph)
    # Total real nodes: 2 + 1 = 3. Sentinel nodes: 5 - 3 = 2. Total nodes: 5.
    self.assertEqual(merged_graph.node_sets["n1"].num_nodes, 5)
    ts_feat = merged_graph.node_sets["n1"].features["ts"]
    mask_feat = merged_graph.node_sets["n1"].features["ts_mask"]
    self.assertEqual(ts_feat.shape, (5, 3))
    self.assertEqual(mask_feat.shape, (5, 3))
    # g1 node 0: [1.0, 2.0] -> padded to [0.0, 1.0, 2.0], mask: [F, T, T]
    np.testing.assert_array_equal(ts_feat[0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(mask_feat[0], np.array([False, True, True]))
    # g1 node 1: [3.0, 4.0, 5.0] -> [3.0, 4.0, 5.0], mask: [T, T, T]
    np.testing.assert_array_equal(ts_feat[1], np.array([3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(mask_feat[1], np.array([True, True, True]))
    # g2 node 0: [6.0, 7.0, 8.0, 9.0, 10.0] -> capped to [8.0, 9.0, 10.0]
    np.testing.assert_array_equal(ts_feat[2], np.array([8.0, 9.0, 10.0]))
    np.testing.assert_array_equal(mask_feat[2], np.array([True, True, True]))
    # Sentinel nodes: [0.0, 0.0, 0.0]
    np.testing.assert_array_equal(ts_feat[3], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(ts_feat[4], np.array([0.0, 0.0, 0.0]))

  def test_batch_with_edge_set_timeseries_padding(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "weight_ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        shape=(None,),
                    ),
                },
            ),
        },
    )
    g1 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(num_nodes=2, features={}),
        },
        edge_sets={
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0], [1]], dtype=np.int32),
                features={
                    "weight_ts": np.array(
                        [np.array([1.0, 2.0])], dtype=object
                    ),
                },
            ),
        },
    )
    g2 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(num_nodes=2, features={}),
        },
        edge_sets={
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0], [1]], dtype=np.int32),
                features={
                    "weight_ts": np.array(
                        [np.array([3.0, 4.0, 5.0, 6.0])], dtype=object
                    ),
                },
            ),
        },
    )
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(num_nodes=4 + 1),
        },
        edge_sets={
            "e1": padding_lib.EdgeSetPadding(
                num_edges=3,
                features={
                    "weight_ts": padding_lib.FeaturePadding(
                        max_timeseries_len=3
                    )
                },
            ),
        },
    )
    schema_cache = temporal_util.extract_timeseries_schema_cache(schema)
    merged_graph, _ = merge_lib.merge_graphs(
        [g1, g2],
        schema,
        padding=padding,
        schema_cache=schema_cache,
    )
    # Total real edges: 1 + 1 = 2. Total edges with padding: 3.
    edge_set = merged_graph.edge_sets["e1"]
    self.assertEqual(edge_set.adjacency.shape, (2, 3))
    ts_feat = edge_set.features["weight_ts"]
    mask_feat = edge_set.features["weight_ts_mask"]
    self.assertEqual(ts_feat.shape, (3, 3))
    self.assertEqual(mask_feat.shape, (3, 3))
    # Edge 0 (g1): [1.0, 2.0] -> [0.0, 1.0, 2.0]
    np.testing.assert_array_equal(ts_feat[0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(mask_feat[0], np.array([False, True, True]))
    # Edge 1 (g2): [3.0, 4.0, 5.0, 6.0] -> [4.0, 5.0, 6.0]
    np.testing.assert_array_equal(ts_feat[1], np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(mask_feat[1], np.array([True, True, True]))
    # Padding edge (sentinel): [0.0, 0.0, 0.0]
    np.testing.assert_array_equal(ts_feat[2], np.array([0.0, 0.0, 0.0]))

  def test_batch_with_timeseries_only_padding(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        shape=(None,),
                    ),
                }
            ),
        },
        edge_sets={},
    )
    g1 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "ts": np.array(
                        [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])],
                        dtype=object,
                    ),
                },
            ),
        },
        edge_sets={},
    )
    g2 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1,
                features={
                    "ts": np.array(
                        [np.array([6.0, 7.0, 8.0, 9.0, 10.0])], dtype=object
                    ),
                },
            ),
        },
        edge_sets={},
    )
    # num_nodes is None: node topology is unpadded, only timeseries is.
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(
                features={
                    "ts": padding_lib.FeaturePadding(max_timeseries_len=3)
                },
            )
        },
        edge_sets={},
    )
    schema_cache = temporal_util.extract_timeseries_schema_cache(schema)
    merged_graph, offsets = merge_lib.merge_graphs(
        [g1, g2],
        schema,
        padding=padding,
        schema_cache=schema_cache,
    )
    # Total real nodes: 2 + 1 = 3. No sentinel nodes added.
    self.assertEqual(merged_graph.node_sets["n1"].num_nodes, 3)
    ts_feat = merged_graph.node_sets["n1"].features["ts"]
    mask_feat = merged_graph.node_sets["n1"].features["ts_mask"]
    self.assertEqual(ts_feat.shape, (3, 3))
    self.assertEqual(mask_feat.shape, (3, 3))
    np.testing.assert_array_equal(offsets["n1"], np.array([0, 2, 3]))
    # g1 node 0: [1.0, 2.0] -> padded to [0.0, 1.0, 2.0], mask: [F, T, T]
    np.testing.assert_array_equal(ts_feat[0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(mask_feat[0], np.array([False, True, True]))
    # g1 node 1: [3.0, 4.0, 5.0] -> [3.0, 4.0, 5.0], mask: [T, T, T]
    np.testing.assert_array_equal(ts_feat[1], np.array([3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(mask_feat[1], np.array([True, True, True]))
    # g2 node 0: [6.0, 7.0, 8.0, 9.0, 10.0] -> capped to [8.0, 9.0, 10.0]
    np.testing.assert_array_equal(ts_feat[2], np.array([8.0, 9.0, 10.0]))
    np.testing.assert_array_equal(mask_feat[2], np.array([True, True, True]))

  def test_graph_merger_class_and_output_schema(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        shape=(None,),
                    ),
                }
            ),
        },
        edge_sets={},
    )
    g1 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "ts": np.array(
                        [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])],
                        dtype=np.object_,
                    ),
                },
            ),
        },
        edge_sets={},
    )
    g2 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1,
                features={
                    "ts": np.array(
                        [np.array([6.0, 7.0, 8.0, 9.0, 10.0])],
                        dtype=np.object_,
                    ),
                },
            ),
        },
        edge_sets={},
    )
    padding = padding_lib.Padding(
        node_sets={
            "n1": padding_lib.NodeSetPadding(
                num_nodes=5,
                features={
                    "ts": padding_lib.FeaturePadding(max_timeseries_len=3)
                },
            )
        },
        edge_sets={},
    )
    merger = merge_lib.GraphMerger(schema=schema, padding=padding)
    output_schema = merger.output_schema()
    self.assertIn("ts", output_schema.node_sets["n1"].features)
    self.assertIn("ts_mask", output_schema.node_sets["n1"].features)
    self.assertEqual(
        output_schema.node_sets["n1"].features["ts"].shape, (3,)
    )

    merged_graph, offsets = merger([g1, g2])
    self.assertEqual(merged_graph.node_sets["n1"].num_nodes, 5)

    # Test merge_graph function alias
    merged_graph_alias, offsets_alias = merge_lib.merge_graph(
        [g1, g2], schema=schema, padding=padding
    )
    test_util.assert_are_equal(self, merged_graph, merged_graph_alias)
    test_util.assert_are_equal(self, offsets, offsets_alias)

  def test_unknown_padding_keys_raise_value_error(self):
    schema = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    bad_node_padding = padding_lib.Padding(
        node_sets={"unknown_node_set": padding_lib.NodeSetPadding(num_nodes=5)},
        edge_sets={},
    )
    with self.assertRaisesRegex(ValueError, "unknown node sets"):
      merge_lib.GraphMerger(schema=schema, padding=bad_node_padding)

    bad_edge_padding = padding_lib.Padding(
        node_sets={"n1": padding_lib.NodeSetPadding(num_nodes=5)},
        edge_sets={"unknown_edge_set": padding_lib.EdgeSetPadding(num_edges=5)},
    )
    with self.assertRaisesRegex(ValueError, "unknown edge sets"):
      merge_lib.GraphMerger(schema=schema, padding=bad_edge_padding)


if __name__ == "__main__":
  absltest.main()
