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

"""Tests for common layers."""

import logging
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import hetero_gnn
import jax
import jax.numpy as jnp


class HeteroGNNTest(parameterized.TestCase):

  def test_sort_plan(self):
    schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1"),
            "e2": schema_lib.EdgeSchema(source="n1", target="n2"),
        },
    )
    plan = [
        ("e1", False),  # n1 -> n1
        ("e1", True),  # n1 <- n1
        ("e2", False),  # n1 -> n2
    ]
    sorted_plan = hetero_gnn.sort_plan(plan, schema)
    expected_sorted_plan = {
        "n1": [("e1", "n1", False), ("e1", "n1", True)],
        "n2": [("e2", "n1", False)],
    }
    self.assertEqual(sorted_plan, expected_sorted_plan)

  def test_message_passing(self):
    schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1"),
            "e2": schema_lib.EdgeSchema(source="n1", target="n2"),
        },
    )
    gnn = hetero_gnn.HeterogeneousGraphConvolutionConfig(
        plan=[
            ("e1", False),  # n1 -> n1
            ("e1", True),  # n1 <- n1
            ("e2", False),  # n1 -> n2
        ],
    ).make(schema)
    input_graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.array([[1.0], [2.0]])},
                num_nodes=2,
            ),
            "n2": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.array([[3.0], [4.0]])},
                num_nodes=2,
            ),
        },
        edge_sets={
            "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0], [1]]),
            ),
            "e2": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0], [0]]),
            ),
        },
    )
    variables = gnn.init(jax.random.PRNGKey(42), input_graph, schema)
    logging.info("variables:\n%s", variables)
    output = gnn.apply(
        variables,
        input_graph,
        training=True,
        rngs={"dropout": jax.random.PRNGKey(42)},
    )

    self.assertEqual(
        output.node_sets["n1"].features["embedding"].shape,
        (2, 128),
    )
    self.assertEqual(
        output.node_sets["n2"].features["embedding"].shape,
        (2, 128),
    )

  def test_architecture(self):
    config = hetero_gnn.HeterogeneousGraphConvolutionConfig()
    infra_str = config.architecture()
    logging.info("architecture:\n%s", infra_str)
    self.assertIn("X = ...", infra_str)
    self.assertIn("MPNN:", infra_str)
    self.assertIn("  Message:", infra_str)
    self.assertIn("    Dense(128)", infra_str)
    self.assertIn("  Update:", infra_str)
    self.assertIn("Residual(X)", infra_str)
    self.assertIn("# Post MPNN", infra_str)
    self.assertIn("Norm(rms_norm)", infra_str)


if __name__ == "__main__":
  absltest.main()
