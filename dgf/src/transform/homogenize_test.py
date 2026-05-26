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
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import jax as jax_lib
from dgf.src.transform import homogenize as homogenize_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import jax
import jax.numpy as jnp
import numpy as np


class HomogenizeTest(absltest.TestCase):

  def _expectd_apply_features_numpy_output(self):
    return in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=2,
                features={},
            ),
            "n2": in_memory_graph.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "f34": np.array(
                        [[4.0, 10.0], [5.0, 11.0]], dtype=np.float32
                    )
                },
            ),
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0, 0], [0, 1]]), features={}
            )
        },
    ), schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
            "n2": schema_lib.NodeSchema(
                features={
                    "f34": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.UNKNOWN,
                        shape=(2,),
                    )
                }
            ),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1", features={})
        },
    )

  def _input_homogenize_numpy_output(self):
    return in_memory_graph.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "f1": np.array([1, 2], dtype=np.float32),
                    "f2": np.array([[3, 4], [5, 6]], dtype=np.float32),
                },
            ),
            "n2": in_memory_graph.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "f1": np.array([7, 8], dtype=np.float32),
                    "f2": np.array([[9, 10], [11, 12]], dtype=np.float32),
                },
            ),
        },
        edge_sets={
            "e1": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0, 0], [0, 1]]),
                features={
                    "f3": np.array([3, 4], dtype=np.float32),
                },
            ),
            "e2": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.array([[0, 0], [0, 1]]),
                features={
                    "f3": np.array([5, 6], dtype=np.float32),
                },
            ),
        },
    ), schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                    ),
                    "f2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=(2,),
                    ),
                }
            ),
            "n2": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                    ),
                    "f2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=(2,),
                    ),
                }
            ),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "f3": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                    ),
                },
            ),
            "e2": schema_lib.EdgeSchema(
                source="n1",
                target="n2",
                features={
                    "f3": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                    ),
                },
            ),
        },
    )

  def _expectd_homogenize_numpy_output(self):
    return (
        in_memory_graph.InMemoryGraph(
            node_sets={
                "nodes": in_memory_graph.InMemoryNodeSet(
                    num_nodes=4,
                    features={
                        "f1": np.array([1.0, 2.0, 7.0, 8.0], dtype=np.float32),
                        "f2": np.array(
                            [[3.0, 4.0], [5.0, 6.0], [9.0, 10.0], [11.0, 12.0]],
                            dtype=np.float32,
                        ),
                    },
                )
            },
            edge_sets={
                "edges": in_memory_graph.InMemoryEdgeSet(
                    adjacency=np.array([[0, 0, 0, 0], [0, 1, 2, 3]]),
                    features={
                        "f3": np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
                    },
                )
            },
        ),
        schema_lib.GraphSchema(
            node_sets={
                "nodes": schema_lib.NodeSchema(
                    features={
                        "f1": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32
                        ),
                        "f2": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32, shape=(2,)
                        ),
                    }
                )
            },
            edge_sets={
                "edges": schema_lib.EdgeSchema(
                    source="nodes",
                    target="nodes",
                    features={
                        "f3": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32
                        )
                    },
                )
            },
        ),
        {"n1": 0, "n2": 2},
    )

  def test_apply_features_numpy(self):
    graph = gen_test_graph.generate_in_memory_graph(node_ids=False)
    schema = gen_test_graph.generate_schema(node_ids=False)

    def process_n1(features, schemas, num_nodes):
      return {}, {}

    def process_n2(features, schemas, num_nodes):
      return {
          "f34": np.stack(
              [features["f3"], features["f4"]], axis=1, dtype=np.float32
          )
      }, {
          "f34": schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.FLOAT_32,
              shape=(2,),
          )
      }

    def process_e1(features, schemas, num_edges):
      return features, schemas

    dst_graph, dst_schema = homogenize_lib.apply_feature(
        graph,
        schema,
        process_nodesets={"n1": process_n1, "n2": process_n2},
        process_edgesets={"e1": process_e1},
    )
    in_memory_graph_validate_lib.validate_graph(
        dst_graph, dst_schema, raise_on_warning=False
    )
    expectd_graph, expected_schema = self._expectd_apply_features_numpy_output()
    test_util.assert_are_equal(self, dst_graph, expectd_graph)
    test_util.assert_are_equal(self, dst_schema, expected_schema)

  def test_apply_features_jax(self):
    numpy_graph = gen_test_graph.generate_in_memory_graph(
        node_ids=False, variable_length=False
    )
    schema = gen_test_graph.generate_schema(
        node_ids=False, variable_length=False
    )

    del numpy_graph.node_sets["n1"].features["f1"]
    del schema.node_sets["n1"].features["f1"]
    jax_graph = jax_lib.graph_to_jax_graph(numpy_graph)

    def process_n1(features, schemas, num_nodes):
      return {}, {}

    def process_n2(features, schemas, num_nodes):
      return {
          "f34": jnp.stack(
              [features["f3"], features["f4"]], axis=1, dtype=np.float32
          )
      }, {
          "f34": schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.FLOAT_32,
              shape=(2,),
          )
      }

    def process_e1(features, schemas, num_edges):
      return features, schemas

    dst_jax_graph, dst_schema = homogenize_lib.apply_feature(
        jax_graph,
        schema,
        process_nodesets={"n1": process_n1, "n2": process_n2},
        process_edgesets={"e1": process_e1},
    )
    dst_numpy_graph = jax_lib.jax_graph_to_graph(dst_jax_graph)
    in_memory_graph_validate_lib.validate_graph(
        dst_numpy_graph, dst_schema, raise_on_warning=False
    )
    expectd_graph, expected_schema = self._expectd_apply_features_numpy_output()
    test_util.assert_are_equal(self, dst_numpy_graph, expectd_graph)
    test_util.assert_are_equal(self, dst_schema, expected_schema)

  def test_homogenize_numpy(self):
    graph, schema = self._input_homogenize_numpy_output()
    dst_graph, dst_schema, nodeset_offsets = homogenize_lib.homogenize(
        graph, schema
    )
    in_memory_graph_validate_lib.validate_graph(
        dst_graph, dst_schema, raise_on_warning=False
    )
    expectd_graph, expected_schema, expected_nodeset_offsets = (
        self._expectd_homogenize_numpy_output()
    )
    test_util.assert_are_equal(self, dst_graph, expectd_graph)
    test_util.assert_are_equal(self, dst_schema, expected_schema)
    test_util.assert_are_equal(self, nodeset_offsets, expected_nodeset_offsets)

  def test_homogenize_jax(self):
    numpy_graph, schema = self._input_homogenize_numpy_output()
    jax_graph = jax_lib.graph_to_jax_graph(numpy_graph)

    dst_schema = None

    @jax.jit
    def my_homogenize(jax_graph):
      nonlocal dst_schema
      dst_jax_graph, dst_schema, nodeset_offsets = homogenize_lib.homogenize(
          jax_graph, schema
      )
      return dst_jax_graph, nodeset_offsets

    dst_jax_graph, nodeset_offsets = my_homogenize(jax_graph)

    dst_numpy_graph = jax_lib.jax_graph_to_graph(dst_jax_graph)
    in_memory_graph_validate_lib.validate_graph(
        dst_numpy_graph, dst_schema, raise_on_warning=False
    )
    expectd_graph, expected_schema, expected_nodeset_offsets = (
        self._expectd_homogenize_numpy_output()
    )
    test_util.assert_are_equal(self, dst_numpy_graph, expectd_graph)
    test_util.assert_are_equal(self, dst_schema, expected_schema)
    test_util.assert_are_equal(self, nodeset_offsets, expected_nodeset_offsets)


if __name__ == "__main__":
  absltest.main()
