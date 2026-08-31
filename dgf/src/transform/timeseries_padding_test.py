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

"""Tests for padding and capping timeseries sequence features."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.transform import timeseries_padding
from dgf.src.util import test_util
import numpy as np


def _make_graph_and_schema(
    values: dict[str, np.ndarray],
    schemas: dict[str, schema_lib.FeatureSchema],
    node_set_name: str = "hardware",
    num_nodes: int = 1,
    edge_values: dict[str, np.ndarray] | None = None,
    edge_schemas: dict[str, schema_lib.FeatureSchema] | None = None,
    edge_set_name: str = "edges",
) -> tuple[in_memory_graph.InMemoryGraph, schema_lib.GraphSchema]:
  edge_sets = {}
  edge_set_schemas = {}
  if edge_values is not None and edge_schemas is not None:
    edge_sets[edge_set_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=np.array([[0], [0]]),
        features=dict(edge_values),
    )
    edge_set_schemas[edge_set_name] = schema_lib.EdgeSchema(
        source=node_set_name,
        target=node_set_name,
        features=edge_schemas,
    )

  return (
      in_memory_graph.InMemoryGraph(
          node_sets={
              node_set_name: in_memory_graph.InMemoryNodeSet(
                  num_nodes=num_nodes, features=dict(values)
              )
          },
          edge_sets=edge_sets,
      ),
      schema_lib.GraphSchema(
          node_sets={node_set_name: schema_lib.NodeSchema(features=schemas)},
          edge_sets=edge_set_schemas,
      ),
  )


def _ts_schema(
    fmt: schema_lib.FeatureFormat = schema_lib.FeatureFormat.FLOAT_32,
    sem: schema_lib.FeatureSemantic = schema_lib.FeatureSemantic.NUMERICAL,
    group: str | None = None,
    is_creation_time: bool = False,
    shape: schema_lib.Shape = (None,),
) -> schema_lib.FeatureSchema:
  return schema_lib.FeatureSchema(
      format=fmt,
      semantic=sem,
      is_timeseries=True,
      group=group,
      is_creation_time=is_creation_time,
      shape=shape,
  )


class TimeseriesPaddingTest(parameterized.TestCase):

  def test_capping_and_padding_with_padding_config(self):
    graph, schema = _make_graph_and_schema(
        values={
            "time": np.array(
                [np.array([10, 20, 30, 40, 50]), np.array([5, 15])],
                dtype=np.object_,
            ),
            "signal": np.array(
                [
                    np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
                    np.array([0.5, 1.5], dtype=np.float32),
                ],
                dtype=np.object_,
            ),
            "id": np.array([101, 102]),
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                is_creation_time=True,
                group="time",
            ),
            "signal": _ts_schema(group="time"),
            "id": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.INTEGER_64,
                semantic=schema_lib.FeatureSemantic.NUMERICAL,
            ),
        },
        num_nodes=2,
    )
    padding = padding_lib.Padding(
        node_sets={
            "hardware": padding_lib.NodeSetPadding(
                num_nodes=2,
                features={
                    "time": padding_lib.FeaturePadding(max_timeseries_len=3),
                    "signal": padding_lib.FeaturePadding(max_timeseries_len=3),
                },
            )
        },
        edge_sets={},
    )
    new_graph = timeseries_padding.pad_timeseries_graph(
        graph=graph, schema=schema, padding=padding
    )
    new_schema = timeseries_padding.pad_timeseries_schema(
        schema=schema, padding=padding
    )
    hw_val = new_graph.node_sets["hardware"]
    hw_sch = new_schema.node_sets["hardware"]

    expected_features = {
        "time": np.array([[30, 40, 50], [0, 5, 15]], dtype=np.int64),
        "signal": np.array(
            [[3.0, 4.0, 5.0], [0.0, 0.5, 1.5]], dtype=np.float32
        ),
        "time_mask": np.array([[True, True, True], [False, True, True]]),
        "id": np.array([101, 102]),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)

    self.assertEqual(hw_sch.features["time"].shape, (3,))
    self.assertTrue(hw_sch.features["time"].is_timeseries)
    self.assertTrue(hw_sch.features["time_mask"].is_timeseries)
    self.assertEqual(
        hw_sch.features["time_mask"].semantic, schema_lib.FeatureSemantic.MASK
    )

  def test_edge_sets_and_non_timeseries(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "user": in_memory_graph.InMemoryNodeSet(
                num_nodes=1, features={"age": np.array([30], dtype=np.int64)}
            ),
        },
        edge_sets={
            "clicks": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0], [0]], dtype=np.int64),
                features={
                    "time": np.array([np.array([100, 200])], dtype=np.object_)
                },
            ),
            "static_edge": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0], [0]], dtype=np.int64),
                features={"weight": np.array([1.0], dtype=np.float32)},
            ),
        },
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "user": schema_lib.NodeSchema(
                features={
                    "age": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                    )
                }
            )
        },
        edge_sets={
            "clicks": schema_lib.EdgeSchema(
                source="user",
                target="user",
                features={
                    "time": _ts_schema(
                        fmt=schema_lib.FeatureFormat.INTEGER_64,
                        sem=schema_lib.FeatureSemantic.TIMESTAMP,
                        group="time",
                    )
                },
            ),
            "static_edge": schema_lib.EdgeSchema(
                source="user",
                target="user",
                features={
                    "weight": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                    )
                },
            ),
        },
    )
    padding = padding_lib.Padding(
        node_sets={},
        edge_sets={
            "clicks": padding_lib.EdgeSetPadding(
                num_edges=1,
                features={
                    "time": padding_lib.FeaturePadding(max_timeseries_len=2)
                },
            )
        },
    )
    new_graph = timeseries_padding.pad_timeseries_graph(
        graph=graph, schema=schema, padding=padding
    )
    new_schema = timeseries_padding.pad_timeseries_schema(
        schema=schema, padding=padding
    )

    np.testing.assert_array_equal(
        new_graph.node_sets["user"].features["age"], [30]
    )
    np.testing.assert_array_equal(
        new_graph.edge_sets["clicks"].features["time"][0], [100, 200]
    )
    self.assertTrue(
        new_schema.edge_sets["clicks"].features["time"].is_timeseries
    )

  def test_multidimensional_sequence(self):
    graph, schema = _make_graph_and_schema(
        values={
            "emb": np.array(
                [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
                dtype=np.object_,
            )
        },
        schemas={
            "emb": _ts_schema(
                sem=schema_lib.FeatureSemantic.EMBEDDING,
                group="emb",
                shape=(None, 2),
            )
        },
    )
    padding = padding_lib.Padding(
        node_sets={
            "hardware": padding_lib.NodeSetPadding(
                num_nodes=1,
                features={
                    "emb": padding_lib.FeaturePadding(max_timeseries_len=3)
                },
            )
        },
        edge_sets={},
    )
    new_graph = timeseries_padding.pad_timeseries_graph(
        graph=graph, schema=schema, padding=padding
    )
    new_schema = timeseries_padding.pad_timeseries_schema(
        schema=schema, padding=padding
    )
    hw_val = new_graph.node_sets["hardware"]
    hw_sch = new_schema.node_sets["hardware"]

    expected_features = {
        "emb": np.array(
            [[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]], dtype=np.float32
        ),
        "emb_mask": np.array([[False, True, True]]),
    }
    test_util.assert_are_equal(self, hw_val.features, expected_features)
    self.assertEqual(hw_sch.features["emb"].shape, (3, 2))
    self.assertEqual(hw_sch.features["emb_mask"].shape, (3,))
    self.assertEqual(
        hw_sch.features["emb_mask"].semantic, schema_lib.FeatureSemantic.MASK
    )

  def test_empty_padding_returns_original_schema(self):
    _, schema = _make_graph_and_schema(
        values={
            "signal": np.array(
                [np.array([1.0, 2.0], dtype=np.float32)], dtype=np.object_
            ),
        },
        schemas={
            "signal": _ts_schema(group="sig", shape=(None,)),
        },
    )
    empty_features_padding = padding_lib.Padding(
        node_sets={
            "hardware": padding_lib.NodeSetPadding(
                num_nodes=1,
                features={},
            )
        },
        edge_sets={},
    )
    # Empty feature padding returns original schema unchanged
    schema_empty_features = timeseries_padding.pad_timeseries_schema(
        schema, padding=empty_features_padding
    )
    self.assertEqual(schema_empty_features, schema)

  def test_inconsistent_group_padding_raises_error(self):
    _, schema = _make_graph_and_schema(
        values={
            "time": np.array([np.array([1, 2])], dtype=np.object_),
            "signal": np.array([np.array([1.0, 2.0])], dtype=np.object_),
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                group="sensor",
            ),
            "signal": _ts_schema(
                fmt=schema_lib.FeatureFormat.FLOAT_32,
                group="sensor",
            ),
        },
    )
    inconsistent_padding = padding_lib.Padding(
        node_sets={
            "hardware": padding_lib.NodeSetPadding(
                features={
                    "time": padding_lib.FeaturePadding(max_timeseries_len=5),
                },
            )
        },
        edge_sets={},
    )
    with self.assertRaisesRegex(
        ValueError, "inconsistent padding configuration"
    ):
      timeseries_padding.pad_timeseries_schema(
          schema, padding=inconsistent_padding
      )

  def test_conflicting_group_sequence_lengths_raises_error(self):
    _, schema = _make_graph_and_schema(
        values={
            "time": np.array([np.array([1, 2])], dtype=np.object_),
            "signal": np.array([np.array([1.0, 2.0])], dtype=np.object_),
        },
        schemas={
            "time": _ts_schema(
                fmt=schema_lib.FeatureFormat.INTEGER_64,
                sem=schema_lib.FeatureSemantic.TIMESTAMP,
                group="sensor",
            ),
            "signal": _ts_schema(
                fmt=schema_lib.FeatureFormat.FLOAT_32,
                group="sensor",
            ),
        },
    )
    conflicting_padding = padding_lib.Padding(
        node_sets={
            "hardware": padding_lib.NodeSetPadding(
                num_nodes=1,
                features={
                    "time": padding_lib.FeaturePadding(max_timeseries_len=5),
                    "signal": padding_lib.FeaturePadding(max_timeseries_len=10),
                },
            )
        },
        edge_sets={},
    )
    with self.assertRaisesRegex(ValueError, "conflicting max_timeseries_len"):
      timeseries_padding.pad_timeseries_schema(
          schema, padding=conflicting_padding
      )


  def test_bytes_timeseries_padding(self):
    graph, schema = _make_graph_and_schema(
        values={
            "tag": np.array(
                [np.array([b"alpha", b"beta"]), np.array([b"gamma"])],
                dtype=np.object_,
            ),
        },
        schemas={
            "tag": _ts_schema(
                fmt=schema_lib.FeatureFormat.BYTES,
                sem=schema_lib.FeatureSemantic.CATEGORICAL,
                group="tag",
            ),
        },
        num_nodes=2,
    )
    padding = padding_lib.Padding(
        node_sets={
            "hardware": padding_lib.NodeSetPadding(
                num_nodes=2,
                features={
                    "tag": padding_lib.FeaturePadding(max_timeseries_len=3)
                },
            )
        },
        edge_sets={},
    )
    new_graph = timeseries_padding.pad_timeseries_graph(
        graph=graph, schema=schema, padding=padding
    )
    hw_val = new_graph.node_sets["hardware"]
    np.testing.assert_array_equal(
        hw_val.features["tag"],
        np.array([[b"", b"alpha", b"beta"], [b"", b"", b"gamma"]]),
    )
    np.testing.assert_array_equal(
        hw_val.features["tag_mask"],
        np.array([[False, True, True], [False, False, True]]),
    )

  def test_empty_entities_timeseries_padding(self):
    graph, schema = _make_graph_and_schema(
        values={
            "sig": np.empty((0,), dtype=np.object_),
        },
        schemas={
            "sig": _ts_schema(group="sig"),
        },
        num_nodes=0,
    )
    padding = padding_lib.Padding(
        node_sets={
            "hardware": padding_lib.NodeSetPadding(
                num_nodes=0,
                features={
                    "sig": padding_lib.FeaturePadding(max_timeseries_len=3)
                },
            )
        },
        edge_sets={},
    )
    new_graph = timeseries_padding.pad_timeseries_graph(
        graph=graph, schema=schema, padding=padding
    )
    hw_val = new_graph.node_sets["hardware"]
    self.assertEqual(hw_val.features["sig"].shape, (0, 3))
    self.assertEqual(hw_val.features["sig_mask"].shape, (0, 3))


if __name__ == "__main__":
  absltest.main()
