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
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import tf_in_memory_graph as tf_in_memory_graph_lib
from dgf.src.io import jax as jax_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import jax.numpy as jnp
import numpy as np


class JaxTest(parameterized.TestCase):

  def test_graph_to_jax_graph(self):
    in_memory_graph = gen_test_graph.generate_in_memory_graph(
        variable_length=False
    )
    del in_memory_graph.node_sets["n1"].features["f1"]
    del in_memory_graph.node_sets["n2"].features["f4"]
    expected_jax_in_memory_graph = jax_in_memory_graph_lib.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph_lib.JaxInMemoryNodeSet(
                features={
                    "f2": jnp.array([[0.0, 1.0], [2.0, 3.0]]),
                },
                num_nodes=2,
            ),
            "n2": jax_in_memory_graph_lib.JaxInMemoryNodeSet(
                features={"f3": jnp.array([4, 5])}, num_nodes=2
            ),
        },
        edge_sets={
            "e2": jax_in_memory_graph_lib.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 0], [0, 1]]), features={}
            ),
            "e1": jax_in_memory_graph_lib.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 0], [0, 1]]), features={}
            ),
        },
    )
    jax_in_memory_graph = jax_lib.graph_to_jax_graph(in_memory_graph)
    test_util.assert_are_equal(
        self, jax_in_memory_graph, expected_jax_in_memory_graph
    )

  def test_jax_graph_to_graph(self):
    # Create a JaxInMemoryGraph to convert.
    jax_in_memory_graph = jax_in_memory_graph_lib.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph_lib.JaxInMemoryNodeSet(
                features={
                    "f2": jnp.array([[0.0, 1.0], [2.0, 3.0]]),
                },
                num_nodes=2,
            ),
            "n2": jax_in_memory_graph_lib.JaxInMemoryNodeSet(
                features={"f3": jnp.array([4, 5])}, num_nodes=2
            ),
        },
        edge_sets={
            "e2": jax_in_memory_graph_lib.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 0], [0, 1]]), features={}
            ),
            "e1": jax_in_memory_graph_lib.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 0], [0, 1]]), features={}
            ),
        },
    )

    # Define the expected InMemoryGraph after conversion.
    expected_in_memory_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                features={
                    "f2": np.array([[0.0, 1.0], [2.0, 3.0]]),
                },
                num_nodes=2,
            ),
            "n2": in_memory_graph_lib.InMemoryNodeSet(
                features={"f3": np.array([4, 5])}, num_nodes=2
            ),
        },
        edge_sets={
            "e2": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 0], [0, 1]]), features={}
            ),
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 0], [0, 1]]), features={}
            ),
        },
    )

    # Perform the conversion.
    in_memory_graph = jax_lib.jax_graph_to_graph(jax_in_memory_graph)

    # Assert that the converted graph matches the expected graph.
    test_util.assert_are_equal(self, in_memory_graph, expected_in_memory_graph)

  def test_tf_graph_to_jax_graph(self):
    tf_in_memory_graph = gen_test_graph.generate_tf_in_memory_graph(
        tensor_type="DENSE",
        num_nodes_as_tensor=True,
        variable_length=False,
    )
    new_features = dict(tf_in_memory_graph.node_sets["n1"].features)
    del new_features["f1"]
    new_node_set = tf_in_memory_graph_lib.TFInMemoryNodeSet(
        num_nodes=tf_in_memory_graph.node_sets["n1"].num_nodes,
        features=new_features,
    )
    new_node_sets = dict(tf_in_memory_graph.node_sets)
    new_node_sets["n1"] = new_node_set
    tf_in_memory_graph = tf_in_memory_graph_lib.TFInMemoryGraph(
        node_sets=new_node_sets, edge_sets=tf_in_memory_graph.edge_sets
    )

    expected_jax_in_memory_graph = jax_in_memory_graph_lib.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph_lib.JaxInMemoryNodeSet(
                features={
                    "f2": jnp.array([[0.0, 1.0], [2.0, 3.0]]),
                },
                num_nodes=2,
            ),
            "n2": jax_in_memory_graph_lib.JaxInMemoryNodeSet(
                features={
                    "f3": jnp.array([4, 5]),
                    "f4": jnp.array([10, 11], dtype=jnp.int32),
                },
                num_nodes=2,
            ),
        },
        edge_sets={
            "e2": jax_in_memory_graph_lib.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 0], [0, 1]]), features={}
            ),
            "e1": jax_in_memory_graph_lib.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 0], [0, 1]]), features={}
            ),
        },
    )
    jax_in_memory_graph = jax_lib.graph_to_jax_graph(tf_in_memory_graph)
    test_util.assert_are_equal(
        self, jax_in_memory_graph, expected_jax_in_memory_graph
    )


if __name__ == "__main__":
  absltest.main()
