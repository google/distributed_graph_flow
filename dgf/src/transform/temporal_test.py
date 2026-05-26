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

"""Tests for temporal transformations."""

from absl.testing import absltest
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.transform import temporal
from dgf.src.util import test_util
import numpy as np


class TemporalTest(absltest.TestCase):

  def test_propagate_timestamp_to_edges_basic(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=2, features={"timestamps": np.array([10.0, 20.0])}
            ),
            "n2": in_memory_graph.InMemoryNodeSet(
                num_nodes=2, features={"timestamps": np.array([5.0, 25.0])}
            ),
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0, 1], [1, 0]])
            )
        },
    )

    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "timestamps": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    )
                }
            ),
            "n2": schema_lib.NodeSchema(
                features={
                    "timestamps": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    )
                }
            ),
        },
        edge_sets={"e1": schema_lib.EdgeSchema(source="n1", target="n2")},
    )

    new_graph, new_schema = temporal.propagate_timestamp_to_edges(graph, schema)

    expected_edge_ts = np.array([25.0, 20.0])
    test_util.assert_are_equal(
        self, new_graph.edge_sets["e1"].features["timestamps"], expected_edge_ts
    )
    self.assertIn("timestamps", new_schema.edge_sets["e1"].features)
    self.assertEqual(
        new_schema.edge_sets["e1"].features["timestamps"].semantic,
        schema_lib.FeatureSemantic.TIMESTAMP,
    )

  def test_propagate_timestamp_to_edges_single_node_timestamp(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=2, features={"timestamps": np.array([10.0, 20.0])}
            ),
            "n2": in_memory_graph.InMemoryNodeSet(num_nodes=2, features={}),
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0, 1], [1, 0]])
            )
        },
    )

    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "timestamps": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    )
                }
            ),
            "n2": schema_lib.NodeSchema(features={}),
        },
        edge_sets={"e1": schema_lib.EdgeSchema(source="n1", target="n2")},
    )

    new_graph, _ = temporal.propagate_timestamp_to_edges(graph, schema)

    expected_edge_ts = np.array([10.0, 20.0])
    test_util.assert_are_equal(
        self, new_graph.edge_sets["e1"].features["timestamps"], expected_edge_ts
    )

  def test_propagate_timestamp_to_edges_fail_no_timestamps(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(num_nodes=2, features={}),
            "n2": in_memory_graph.InMemoryNodeSet(num_nodes=2, features={}),
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0, 1], [1, 0]])
            )
        },
    )

    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
            "n2": schema_lib.NodeSchema(features={}),
        },
        edge_sets={"e1": schema_lib.EdgeSchema(source="n1", target="n2")},
    )

    with self.assertRaisesRegex(ValueError, "Neither source nodeset"):
      temporal.propagate_timestamp_to_edges(graph, schema)

  def test_propagate_timestamp_to_edges_custom_names(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=2, features={"time": np.array([10.0, 20.0])}
            ),
            "n2": in_memory_graph.InMemoryNodeSet(
                num_nodes=2, features={"ts": np.array([5.0, 25.0])}
            ),
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0, 1], [1, 0]])
            )
        },
    )

    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    )
                }
            ),
            "n2": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    )
                }
            ),
        },
        edge_sets={"e1": schema_lib.EdgeSchema(source="n1", target="n2")},
    )

    new_graph, new_schema = temporal.propagate_timestamp_to_edges(
        graph,
        schema,
        node_timestamps={"n1": "time", "n2": "ts"},
        target_feature="edge_ts",
    )

    expected_edge_ts = np.array([25.0, 20.0])
    test_util.assert_are_equal(
        self, new_graph.edge_sets["e1"].features["edge_ts"], expected_edge_ts
    )
    self.assertIn("edge_ts", new_schema.edge_sets["e1"].features)

  def test_propagate_timestamp_to_edges_fail_existing_feature(self):
    graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=2, features={"timestamps": np.array([10.0, 20.0])}
            ),
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0, 1]]),
                features={"timestamps": np.array([1.0])},
            )
        },
    )

    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "timestamps": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                    )
                }
            ),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "timestamps": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_64,
                    )
                },
            )
        },
    )

    with self.assertRaisesRegex(ValueError, "already exists in edgeset"):
      temporal.propagate_timestamp_to_edges(graph, schema)


if __name__ == "__main__":
  absltest.main()
