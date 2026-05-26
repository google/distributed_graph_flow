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

import json
import os
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling.gcp import spanner_graph_sampler as spanner_graph_sampler_lib
from dgf.src.util import test_util
import numpy as np

InMemoryGraph = in_memory_graph_lib.InMemoryGraph
InMemoryNodeSet = in_memory_graph_lib.InMemoryNodeSet
InMemoryEdgeSet = in_memory_graph_lib.InMemoryEdgeSet


def _toy_schema(has_feat: bool = False):
  features = {
      "id": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
      ),
      "labels": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
      ),
      "year": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      ),
  }
  if has_feat:
    features["feat"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        shape=(128,),
    )
  return schema_lib.GraphSchema(
      node_sets={"nodes": schema_lib.NodeSchema(features=features)},
      edge_sets={
          "edges": schema_lib.EdgeSchema(source="nodes", target="nodes")
      },
  )


class SpannerGraphSamplerTest(parameterized.TestCase):

  def test_generate_gql_query(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "N1": schema_lib.NodeSchema(
                features={
                    "id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    )
                }
            ),
            "N2": schema_lib.NodeSchema(
                features={
                    "id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    )
                }
            ),
        },
        edge_sets={
            "E1": schema_lib.EdgeSchema(source="N1", target="N1"),
            "E2": schema_lib.EdgeSchema(source="N1", target="N2"),
        },
    )
    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="N1", num_hops=2, hop_width=2, reverse=True
    )
    plan_tree = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)

    query = spanner_graph_sampler_lib._generate_gql_query(
        graph_name="my_graph",
        plan=plan_tree,
        schema=schema,
        seed_ids=["10", "11"],
        debug_sampling=False,
    )

    expected_query_path = os.path.join(
        test_util.dgf_test_data_path(),
        "spanner_graph_sampler_test_generate_gql_query.txt",
    )
    with open(expected_query_path, "r") as f:
      expected_query = f.read()

    self.assertEqual(query, expected_query)

  def test_json_to_in_memory_graphs(self):
    # Note: "spanner_graph_sampler_test_json_to_in_memory_graphs" as been
    # generated using the e2e manual test below.
    raw_query_path = os.path.join(
        test_util.dgf_test_data_path(),
        "spanner_graph_sampler_test_json_to_in_memory_graphs.json",
    )
    with open(raw_query_path, "r") as f:
      query = json.load(f)
    schema = _toy_schema()
    graphs = spanner_graph_sampler_lib._json_to_in_memory_graphs(
        query, schema, ["10", "11"], "nodes"
    )

    test_util.assert_are_equal(
        self,
        graphs,
        [
            InMemoryGraph(
                node_sets={
                    "nodes": InMemoryNodeSet(
                        num_nodes=5,
                        features={
                            "id": np.array(
                                [b"10", b"10090", b"92331", b"25350", b"46510"],
                            ),
                            "labels": np.array([24, 34, 8, 8, 24]),
                            "year": np.array([2012, 2007, 2013, 2011, 2013]),
                        },
                    )
                },
                edge_sets={
                    "edges": InMemoryEdgeSet(
                        adjacency=np.array([[0, 2, 0, 4], [1, 0, 3, 0]])
                    )
                },
            ),
            InMemoryGraph(
                node_sets={
                    "nodes": InMemoryNodeSet(
                        num_nodes=3,
                        features={
                            "id": np.array([b"11", b"54035", b"142591"]),
                            "labels": np.array([36, 36, 36]),
                            "year": np.array([2015, 2012, 2012]),
                        },
                    )
                },
                edge_sets={
                    "edges": InMemoryEdgeSet(
                        adjacency=np.array([[0, 0], [1, 2]])
                    )
                },
            ),
        ],
    )

  def test_json_to_in_memory_graphs_deduplication(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    )
                }
            )
        },
        edge_sets={
            "edges": schema_lib.EdgeSchema(source="nodes", target="nodes")
        },
    )

    # Mock JSON result with duplicate edges.
    # Row 0: Seed '10', Path with edge 'e1' from '10' to '20'.
    # Row 1: Seed '10', Another path with the SAME edge 'e1' from '10' to '20'.
    query = [
        [
            {
                "kind": "node",
                "labels": ["nodes"],
                "identifier": "10",
                "properties": {"id": "10"},
            },
            {
                "kind": "node",
                "labels": ["nodes"],
                "identifier": "20",
                "properties": {"id": "20"},
            },
            {
                "kind": "edge",
                "labels": ["edges"],
                "identifier": "e1",
                "source_node_identifier": "10",
                "destination_node_identifier": "20",
                "properties": {},
            },
        ],
        [
            {
                "kind": "node",
                "labels": ["nodes"],
                "identifier": "10",
                "properties": {"id": "10"},
            },
            {
                "kind": "node",
                "labels": ["nodes"],
                "identifier": "20",
                "properties": {"id": "20"},
            },
            {
                "kind": "edge",
                "labels": ["edges"],
                "identifier": "e1",
                "source_node_identifier": "10",
                "destination_node_identifier": "20",
                "properties": {},
            },
        ],
    ]

    graphs = spanner_graph_sampler_lib._json_to_in_memory_graphs(
        query, schema, ["10"], "nodes"
    )

    test_util.assert_are_equal(
        self,
        graphs,
        [
            InMemoryGraph(
                node_sets={
                    "nodes": InMemoryNodeSet(
                        num_nodes=2,
                        features={
                            "id": np.array([b"10", b"20"]),
                        },
                    )
                },
                edge_sets={
                    "edges": InMemoryEdgeSet(adjacency=np.array([[0], [1]]))
                },
            ),
        ],
    )


if __name__ == "__main__":
  absltest.main()
