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

  def test_generate_cte_query(self):
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

    query = spanner_graph_sampler_lib.CteQueryGenerator(
        graph_name="my_graph",
        schema=schema,
        plan=plan_tree,
        debug_sampling=False,
    ).generate()

    test_util.assert_golden_string(
        self,
        query,
        "spanner_graph_sampler_test_generate_cte_query.txt",
        strip=True,
    )

  def test_generate_cte_query_directed(self):
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
        seed_nodeset="N1", num_hops=2, hop_width=2, reverse=False
    )
    plan_tree = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)

    query = spanner_graph_sampler_lib.CteQueryGenerator(
        graph_name="my_graph",
        schema=schema,
        plan=plan_tree,
        debug_sampling=False,
    ).generate()

    test_util.assert_golden_string(
        self,
        query,
        "spanner_graph_sampler_test_generate_cte_query_directed.txt",
        strip=True,
    )

  def test_cte_result_to_in_memory_graphs(self):
    # Mock CTE flat rows.
    # Columns: [seed_id, node_id, element_type, element_class, source_id, target_id, properties_json]
    # IDs are base64 encoded bytes as returned by Spanner for BYTES columns.
    # '10' -> b'MTA='
    # '11' -> b'MTE='
    # '10090' -> b'MTAwOTA='
    # '92331' -> b'OTIzMzE='
    query_results = [
        # Seed 10
        [
            b"MTA=",
            b"MTA=",
            "node",
            "nodes",
            None,
            None,
            '{"labels": 24, "year": 2012}',
        ],
        [b"MTA=", None, "edge", "edges", b"MTA=", b"MTAwOTA=", "{}"],
        [
            b"MTA=",
            b"MTAwOTA=",
            "node",
            "nodes",
            None,
            None,
            '{"labels": 34, "year": 2007}',
        ],
        [b"MTA=", None, "edge", "edges", b"MTA=", b"OTIzMzE=", "{}"],
        [
            b"MTA=",
            b"OTIzMzE=",
            "node",
            "nodes",
            None,
            None,
            '{"labels": 8, "year": 2013}',
        ],
        # Seed 11
        [
            b"MTE=",
            b"MTE=",
            "node",
            "nodes",
            None,
            None,
            '{"labels": 36, "year": 2015}',
        ],
    ]
    schema = _toy_schema()
    graphs = spanner_graph_sampler_lib._cte_result_to_in_memory_graphs(
        query_results, schema, [b"10", b"11"], "nodes"
    )

    test_util.assert_are_equal(
        self,
        graphs,
        [
            InMemoryGraph(
                node_sets={
                    "nodes": InMemoryNodeSet(
                        num_nodes=3,
                        features={
                            "id": np.array([b"10", b"10090", b"92331"]),
                            "labels": np.array([24, 34, 8]),
                            "year": np.array([2012, 2007, 2013]),
                        },
                    )
                },
                edge_sets={
                    "edges": InMemoryEdgeSet(
                        adjacency=np.array([[0, 0], [1, 2]])
                    )
                },
            ),
            InMemoryGraph(
                node_sets={
                    "nodes": InMemoryNodeSet(
                        num_nodes=1,
                        features={
                            "id": np.array([b"11"]),
                            "labels": np.array([36]),
                            "year": np.array([2015]),
                        },
                    )
                },
                edge_sets={
                    "edges": InMemoryEdgeSet(
                        adjacency=np.zeros((2, 0), dtype=np.int64)
                    )
                },
            ),
        ],
    )


if __name__ == "__main__":
  absltest.main()
