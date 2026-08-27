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
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.gbbs import loader
from dgf.src.io import graph_in_memory
import numpy as np


class LoaderTest(parameterized.TestCase):

  def test_read_gbbs_graph_from_parquet(self):
    loader.set_num_parlay_workers(1)
    work_dir = self.create_tempdir().full_path

    # Create a small homogeneous graph schema
    schema = schema_lib.GraphSchema(
        node_sets={
            "V": schema_lib.NodeSchema(
                features={
                    "#id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    )
                }
            )
        },
        edge_sets={
            "E": schema_lib.EdgeSchema(
                "V",
                "V",
                features={
                    "weight": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                    )
                },
            )
        },
    )

    # Create corresponding in-memory graph
    graph = in_memory_graph_lib.InMemoryGraph(node_sets={}, edge_sets={})
    graph.node_sets["V"] = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=3, features={"#id": np.array([10, 20, 30], dtype=np.int64)}
    )
    graph.edge_sets["E"] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=np.array([[0, 1], [1, 2]], dtype=np.int64),
        features={"weight": np.array([1.5, 2.5], dtype=np.float32)},
    )

    # Write to disk
    graph_in_memory.write_graph(graph, schema, work_dir)

    # Load into gbbs graph handle directly using read_gbbs_projection
    graph_handle = loader.read_gbbs_projection(
        work_dir, node_set_name="V", edge_set_name="E", weight_key="weight"
    )

    self.assertEqual(graph_handle.num_nodes(), 3)
    self.assertEqual(graph_handle.num_edges(), 4)

  def test_validate_projection_success(self):
    schema = schema_lib.GraphSchema(
        node_sets={"V": schema_lib.NodeSchema(features={})},
        edge_sets={"E": schema_lib.EdgeSchema("V", "V", features={})},
    )
    # Should not raise
    loader.validate_projection(schema, "V", "E")

  def test_validate_projection_success_with_weights(self):
    schema = schema_lib.GraphSchema(
        node_sets={"V": schema_lib.NodeSchema(features={})},
        edge_sets={
            "E": schema_lib.EdgeSchema(
                "V",
                "V",
                features={
                    "weight": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    )
                },
            )
        },
    )
    # Should not raise
    loader.validate_projection(schema, "V", "E", weight_key="weight")

  def test_validate_projection_missing_node_set(self):
    schema = schema_lib.GraphSchema(
        node_sets={"V": schema_lib.NodeSchema(features={})},
        edge_sets={"E": schema_lib.EdgeSchema("V", "V", features={})},
    )
    with self.assertRaisesRegex(
        ValueError, "Node set 'MissingNode' not found in schema"
    ):
      loader.validate_projection(schema, "MissingNode", "E")

  def test_validate_projection_missing_edge_set(self):
    schema = schema_lib.GraphSchema(
        node_sets={"V": schema_lib.NodeSchema(features={})},
        edge_sets={"E": schema_lib.EdgeSchema("V", "V", features={})},
    )
    with self.assertRaisesRegex(
        ValueError, "Edge set 'MissingEdge' not found in schema"
    ):
      loader.validate_projection(schema, "V", "MissingEdge")

  def test_validate_projection_heterogeneous_edge_set(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "V1": schema_lib.NodeSchema(features={}),
            "V2": schema_lib.NodeSchema(features={}),
        },
        edge_sets={"E": schema_lib.EdgeSchema("V1", "V2", features={})},
    )
    with self.assertRaisesRegex(ValueError, "not homogeneous"):
      loader.validate_projection(schema, "V1", "E")

  def test_validate_projection_missing_weight_key(self):
    schema = schema_lib.GraphSchema(
        node_sets={"V": schema_lib.NodeSchema(features={})},
        edge_sets={
            "E": schema_lib.EdgeSchema(
                "V",
                "V",
                features={
                    "weight": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32
                    )
                },
            )
        },
    )
    with self.assertRaisesRegex(
        ValueError, "Weight key 'missing_weight' not found in edge set"
    ):
      loader.validate_projection(schema, "V", "E", weight_key="missing_weight")

  def test_validate_projection_non_numeric_weight_key(self):
    schema = schema_lib.GraphSchema(
        node_sets={"V": schema_lib.NodeSchema(features={})},
        edge_sets={
            "E": schema_lib.EdgeSchema(
                "V",
                "V",
                features={
                    "weight": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES
                    )
                },
            )
        },
    )
    with self.assertRaisesRegex(ValueError, "has non-numeric feature type"):
      loader.validate_projection(schema, "V", "E", weight_key="weight")


if __name__ == "__main__":
  absltest.main()
