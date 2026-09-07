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
from dgf.src.analyse import padding as padding_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import padding as padding_data_lib
from dgf.src.data import schema as schema_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np

test_util.disable_diff_truncation()


class PaddingTest(absltest.TestCase):

  def test_basic(self):
    schema = gen_test_graph.generate_schema()
    graphs = [
        gen_test_graph.generate_in_memory_graph(),
        gen_test_graph.generate_in_memory_graph(),
        gen_test_graph.generate_in_memory_graph(),
    ]
    padding = padding_lib.padding_from_graph_generator(schema, iter(graphs))
    expected_padding = padding_data_lib.Padding(
        node_sets={
            "n1": padding_data_lib.NodeSetPadding(num_nodes=4),
            "n2": padding_data_lib.NodeSetPadding(num_nodes=4),
        },
        edge_sets={
            "e1": padding_data_lib.EdgeSetPadding(num_edges=4),
            "e2": padding_data_lib.EdgeSetPadding(num_edges=4),
        },
    )
    test_util.assert_are_equal(self, padding, expected_padding)

  def test_timeseries_padding(self):
    # 3 features: 1 static timeseries, 2 variable timeseries sharing a group.
    schema = schema_lib.GraphSchema(
        node_sets={
            "n": schema_lib.NodeSchema(
                features={
                    "static_ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=(5,),
                        is_timeseries=True,
                    ),
                    "var_ts1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=(None,),
                        is_timeseries=True,
                        group="g1",
                    ),
                    "var_ts2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        shape=(None,),
                        is_timeseries=True,
                        is_creation_time=True,
                        group="g1",
                    ),
                }
            )
        },
        edge_sets={
            "e": schema_lib.EdgeSchema(
                source="n",
                target="n",
                features={
                    "edge_ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=(None,),
                        is_timeseries=True,
                    ),
                },
            )
        },
    )
    g1 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "static_ts": np.zeros((2, 5), dtype=np.float32),
                    "var_ts1": np.array(
                        [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])],
                        dtype=object,
                    ),
                    "var_ts2": np.array(
                        [
                            np.array([10, 20], dtype=np.int64),
                            np.array([20, 30, 40], dtype=np.int64),
                        ],
                        dtype=object,
                    ),
                },
            )
        },
        edge_sets={
            "e": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0], [1]], dtype=np.int32),
                features={
                    "edge_ts": np.array([np.array([1.0, 2.0])], dtype=object),
                },
            )
        },
    )
    padding = padding_lib.padding_from_graph_generator(
        schema, iter([g1]), max_timeseries_len=4
    )
    expected_padding = padding_data_lib.Padding(
        node_sets={
            "n": padding_data_lib.NodeSetPadding(
                num_nodes=4,
                features={
                    "static_ts": padding_data_lib.FeaturePadding(
                        max_timeseries_len=4
                    ),
                    "var_ts1": padding_data_lib.FeaturePadding(
                        max_timeseries_len=4
                    ),
                    "var_ts2": padding_data_lib.FeaturePadding(
                        max_timeseries_len=4
                    ),
                },
            )
        },
        edge_sets={
            "e": padding_data_lib.EdgeSetPadding(
                num_edges=3,
                features={
                    "edge_ts": padding_data_lib.FeaturePadding(
                        max_timeseries_len=4
                    ),
                },
            )
        },
    )
    test_util.assert_are_equal(self, padding, expected_padding)

  def test_print_padding(self):
    padding = padding_data_lib.Padding(
        node_sets={
            "n1": padding_data_lib.NodeSetPadding(num_nodes=10),
            "n2": padding_data_lib.NodeSetPadding(num_nodes=20),
        },
        edge_sets={
            "e1": padding_data_lib.EdgeSetPadding(num_edges=100),
        },
    )
    output = padding_lib.print_padding(padding, return_output=True)
    expected_output = """Graph Padding:

Node Sets:
  n1: 10 nodes
  n2: 20 nodes

Edge Sets:
  e1: 100 edges"""
    self.assertEqual(output, expected_output)

  def test_feature_padding_from_schema(self):
    features_schema = {
        "f": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            is_timeseries=True,
        ),
    }
    feature_padding = padding_lib._feature_padding_from_schema(
        features_schema, max_timeseries_len=5
    )
    expected_padding = {
        "f": padding_data_lib.FeaturePadding(max_timeseries_len=5),
    }
    self.assertEqual(feature_padding, expected_padding)


if __name__ == "__main__":
  absltest.main()

