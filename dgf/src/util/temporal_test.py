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

"""Tests for temporal schema cache extraction utilities."""

from absl.testing import absltest
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.util import temporal
import numpy as np


class TemporalTest(absltest.TestCase):

  def test_extract_timeseries_schema_cache_empty(self):
    schema = schema_lib.GraphSchema(
        node_sets={"nodes": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    cache = temporal.extract_timeseries_schema_cache(schema)
    self.assertEqual(cache.node_sets, {"nodes": []})
    self.assertEqual(cache.edge_sets, {})
    self.assertFalse(cache.has_timeseries)

  def test_extract_timeseries_schema_cache_no_timeseries_features(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "feat1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=False,
                    )
                }
            )
        },
        edge_sets={
            "edges": schema_lib.EdgeSchema(
                source="nodes",
                target="nodes",
                features={
                    "feat2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=False,
                    )
                },
            )
        },
    )
    cache = temporal.extract_timeseries_schema_cache(schema)
    self.assertEqual(cache.node_sets, {"nodes": []})
    self.assertEqual(cache.edge_sets, {"edges": []})

  def test_extract_timeseries_schema_cache_grouped(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "hardware": schema_lib.NodeSchema(
                features={
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        group="g1",
                    ),
                    "signal": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        group="g1",
                    ),
                    "non_ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=False,
                    ),
                }
            )
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="hardware",
                target="hardware",
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        group="e1_g",
                    ),
                    "weight_series": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        group="e1_g",
                    ),
                },
            )
        },
    )

    cache = temporal.extract_timeseries_schema_cache(schema)
    self.assertTrue(cache.has_timeseries)

    self.assertIn("hardware", cache.node_sets)
    self.assertLen(cache.node_sets["hardware"], 1)
    hw_group = cache.node_sets["hardware"][0]
    self.assertEqual(hw_group.timestamp_feature_name, "time")
    self.assertCountEqual(hw_group.feature_names, ["time", "signal"])

    self.assertIn("e1", cache.edge_sets)
    self.assertLen(cache.edge_sets["e1"], 1)
    e1_group = cache.edge_sets["e1"][0]
    self.assertEqual(e1_group.timestamp_feature_name, "ts")
    self.assertCountEqual(e1_group.feature_names, ["ts", "weight_series"])

  def test_extract_timeseries_schema_cache_ungrouped_creation_time(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                    ),
                    "other_ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                    ),
                }
            )
        },
        edge_sets={},
    )
    cache = temporal.extract_timeseries_schema_cache(schema)
    self.assertTrue(cache.has_timeseries)
    self.assertIn("nodes", cache.node_sets)
    self.assertLen(cache.node_sets["nodes"], 2)

    time_group = next(
        g for g in cache.node_sets["nodes"] if "time" in g.feature_names
    )
    self.assertEqual(time_group.timestamp_feature_name, "time")
    self.assertCountEqual(time_group.feature_names, ["time"])

    other_group = next(
        g for g in cache.node_sets["nodes"] if "other_ts" in g.feature_names
    )
    self.assertIsNone(other_group.timestamp_feature_name)
    self.assertCountEqual(other_group.feature_names, ["other_ts"])

  def test_feature_schema_helpers(self):
    scalar_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(None,),
        is_timeseries=True,
    )
    self.assertEqual(temporal.get_timeseries_step_shape(scalar_ts), ())
    self.assertEqual(temporal.with_sequence_length(scalar_ts, 30).shape, (30,))
    self.assertTrue(temporal.with_sequence_length(scalar_ts, 30).is_timeseries)

    vector_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(None, 8),
        is_timeseries=True,
    )
    self.assertEqual(temporal.get_timeseries_step_shape(vector_ts), (8,))
    self.assertEqual(
        temporal.with_sequence_length(vector_ts, 30).shape, (30, 8)
    )

    unknown_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=None,
        is_timeseries=True,
    )
    with self.assertRaisesRegex(
        ValueError,
        r"Timeseries feature schema must have at least 1 dimension \(sequence"
        r" length at shape\[0\]\), but got shape=None\.",
    ):
      temporal.get_timeseries_step_shape(unknown_ts)

    empty_tuple_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(),
        is_timeseries=True,
    )
    with self.assertRaisesRegex(
        ValueError,
        r"Timeseries feature schema must have at least 1 dimension \(sequence"
        r" length at shape\[0\]\), but got shape=\(\)\.",
    ):
      temporal.get_timeseries_step_shape(empty_tuple_ts)

    non_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(10,),
        is_timeseries=False,
    )
    with self.assertRaisesRegex(
        ValueError, r"Feature schema must be a timeseries feature\."
    ):
      temporal.get_timeseries_step_shape(non_ts)

  def test_expand_mask_dims(self):
    # Create a mask with alternating True/False entries.
    mask = np.arange(50).reshape(5, 10) % 2 == 0
    target_2d = np.zeros((5, 10), dtype=np.float32)
    target_4d = np.ones((5, 10, 3, 4), dtype=np.float32) * 42.0

    # Check 2D target (no extra dimensions needed).
    expanded_2d = temporal.expand_mask_dims(mask, target_2d)
    self.assertIs(expanded_2d, mask)
    self.assertEqual(expanded_2d.shape, (5, 10))

    # Check 4D target (two extra dimensions added).
    expanded_4d = temporal.expand_mask_dims(mask, target_4d)
    self.assertEqual(expanded_4d.shape, (5, 10, 1, 1))
    np.testing.assert_array_equal(expanded_4d[:, :, 0, 0], mask)

  def test_timeseries_group(self):
    schemas = {
        "time": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.TIMESTAMP,
            is_timeseries=True,
            is_creation_time=True,
            group="group",
        ),
        "feature": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            is_timeseries=True,
            group="group",
        ),
    }

    self.assertEqual(schemas["feature"].group, "group")

  def test_group_creation_time_feature_name(self):
    schemas = {
        "explicit_time": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.TIMESTAMP,
            is_timeseries=True,
            is_creation_time=True,
            group="explicit_group",
        ),
    }

    self.assertEqual(
        temporal.group_creation_time_feature_name(
            "explicit_group", schemas
        ),
        "explicit_time",
    )

  def test_edgeset_creation_time_feature_name_heterogeneous(self):
    node_ts = lambda: schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_creation_time=True,
    )
    edge_ts = lambda: schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_creation_time=False,
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={"t1": node_ts()}),
            "n2": schema_lib.NodeSchema(features={"t2": node_ts()}),
        },
        edge_sets={},
    )
    es = lambda f: schema_lib.EdgeSchema(
        source="n1", target="n2", features={f: edge_ts()}
    )
    self.assertEqual(
        temporal.edgeset_creation_time_feature_name(es("t1"), schema),
        "t1",
    )
    self.assertEqual(
        temporal.edgeset_creation_time_feature_name(es("t2"), schema),
        "t2",
    )

  def test_entity_set_timestamp_features(self):
    node_ts = lambda: schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_creation_time=True,
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={"ts_n": node_ts()}),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1", target="n1", features={"ts_e": node_ts()}
            ),
        },
    )
    self.assertEqual(
        temporal.nodeset_timestamp_features(schema), {"n1": "ts_n"}
    )

  def test_schema_has_timeseries_features(self):
    schema_no_ts = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        is_timeseries=False,
                    )
                }
            ),
        },
        edge_sets={},
    )
    self.assertFalse(temporal.schema_has_timeseries_features(schema_no_ts))

    schema_with_node_ts = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                    )
                }
            ),
        },
        edge_sets={},
    )
    self.assertTrue(
        temporal.schema_has_timeseries_features(schema_with_node_ts)
    )

    schema_with_edge_ts = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "ts_e": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                    )
                },
            )
        },
    )
    self.assertTrue(
        temporal.schema_has_timeseries_features(schema_with_edge_ts)
    )

  def test_schema_has_dynamic_timeseries_features(self):
    schema_static_ts = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                        shape=(10,),
                    )
                }
            ),
        },
        edge_sets={},
    )
    self.assertFalse(
        temporal.schema_has_dynamic_timeseries_features(schema_static_ts)
    )

    schema_dynamic_ts = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                        shape=(None,),
                    )
                }
            ),
        },
        edge_sets={},
    )
    self.assertTrue(
        temporal.schema_has_dynamic_timeseries_features(schema_dynamic_ts)
    )

    schema_dynamic_edge_ts = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "ts_e": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        is_timeseries=True,
                        shape=(None,),
                    )
                },
            )
        },
    )
    self.assertTrue(
        temporal.schema_has_dynamic_timeseries_features(schema_dynamic_edge_ts)
    )

  def test_expand_batch_seed_timestamps_heterogeneous(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes_a": schema_lib.NodeSchema(features={}),
            "nodes_b": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "edges": schema_lib.EdgeSchema(
                source="nodes_a", target="nodes_b", features={}
            ),
        },
    )
    sample = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes_a": in_memory_graph_lib.InMemoryNodeSet(
                features={}, num_nodes=5
            ),
            "nodes_b": in_memory_graph_lib.InMemoryNodeSet(
                features={}, num_nodes=3
            ),
        },
        edge_sets={
            "edges": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 1, 3, 4], [0, 0, 1, 2]]),
                features={},
            ),
        },
    )
    # Batch of 2 subgraphs:
    # Subgraph 0 has 2 nodes_a and 1 nodes_b.
    # Subgraph 1 has 3 nodes_a and 2 nodes_b.
    merge_offsets = {
        "nodes_a": np.array([0, 2, 5], dtype=np.int32),
        "nodes_b": np.array([0, 1, 3], dtype=np.int32),
    }
    seed_timestamps = np.array([1000, 2000], dtype=np.int64)

    expanded = temporal.expand_batch_seed_timestamps(
        sample=sample,
        merge_offsets=merge_offsets,
        schema=schema,
        seed_timestamps=seed_timestamps,
    )

    expected_nodes_a = np.array([1000, 1000, 2000, 2000, 2000], dtype=np.int64)
    expected_nodes_b = np.array([1000, 2000, 2000], dtype=np.int64)
    # Edges connect nodes_a [0, 1, 3, 4] -> timestamps [1000, 1000, 2000, 2000]
    expected_edges = np.array([1000, 1000, 2000, 2000], dtype=np.int64)

    np.testing.assert_array_equal(expanded["nodes_a"], expected_nodes_a)
    np.testing.assert_array_equal(expanded["nodes_b"], expected_nodes_b)
    np.testing.assert_array_equal(expanded["edges"], expected_edges)

  def test_expand_batch_seed_timestamps_padded_and_empty_edge(self):
    schema = schema_lib.GraphSchema(
        node_sets={"nodes": schema_lib.NodeSchema(features={})},
        edge_sets={
            "empty_edges": schema_lib.EdgeSchema(
                source="nodes", target="nodes", features={}
            ),
        },
    )
    # Total 4 nodes (2 real + 2 padding)
    sample = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                features={}, num_nodes=4
            )
        },
        edge_sets={
            "empty_edges": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.zeros((2, 0), dtype=np.int32),
                features={},
            ),
        },
    )
    # 2 subgraphs: Subgraph 0 has 1 node, Subgraph 1 has 1 node, plus padding
    merge_offsets = {
        "nodes": np.array([0, 1, 2, 4], dtype=np.int32),
    }
    seed_timestamps = np.array([100, 200], dtype=np.int64)

    expanded = temporal.expand_batch_seed_timestamps(
        sample=sample,
        merge_offsets=merge_offsets,
        schema=schema,
        seed_timestamps=seed_timestamps,
    )

    # 2 real nodes get [100, 200], padding nodes get 0-th timestamp [100, 100]
    expected_nodes = np.array([100, 200, 100, 100], dtype=np.int64)
    expected_empty_edges = np.zeros((0,), dtype=np.int64)

    np.testing.assert_array_equal(expanded["nodes"], expected_nodes)
    np.testing.assert_array_equal(expanded["empty_edges"], expected_empty_edges)

  def test_expand_batch_seed_timestamps_insufficient_offsets(self):
    schema = schema_lib.GraphSchema(
        node_sets={"nodes": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    sample = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                features={}, num_nodes=4
            )
        },
        edge_sets={},
    )
    # 2 subgraphs requires at least 3 offsets (e.g. [0, 2, 4]), but provide
    # only 2 offsets.
    merge_offsets = {
        "nodes": np.array([0, 2], dtype=np.int32),
    }
    seed_timestamps = np.array([100, 200], dtype=np.int64)

    with self.assertRaises(AssertionError):
      temporal.expand_batch_seed_timestamps(
          sample=sample,
          merge_offsets=merge_offsets,
          schema=schema,
          seed_timestamps=seed_timestamps,
      )

  def test_expand_batch_seed_timestamps_invalid_ndim(self):
    schema = schema_lib.GraphSchema(
        node_sets={"nodes": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    sample = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                features={}, num_nodes=4
            )
        },
        edge_sets={},
    )
    merge_offsets = {
        "nodes": np.array([0, 2, 4], dtype=np.int32),
    }

    # 2D array should trigger assertion error
    with self.assertRaises(AssertionError):
      temporal.expand_batch_seed_timestamps(
          sample=sample,
          merge_offsets=merge_offsets,
          schema=schema,
          seed_timestamps=np.array([[100], [200]], dtype=np.int64),
      )

    # 0D scalar should trigger assertion error
    with self.assertRaises(AssertionError):
      temporal.expand_batch_seed_timestamps(
          sample=sample,
          merge_offsets=merge_offsets,
          schema=schema,
          seed_timestamps=np.array(100, dtype=np.int64),
      )


if __name__ == "__main__":
  absltest.main()
