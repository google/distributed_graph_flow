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

"""Tests for the deprecated heterogeneous graph attention network wrapper."""

from absl.testing import absltest
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import graph_attention
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import hetero_graph_attention_network
import jax
import jax.numpy as jnp

LegacyConfig = (
    hetero_graph_attention_network.HeterogeneousGraphAttentionNetworkConfig
)


class HeteroGraphAttentionNetworkTest(absltest.TestCase):

  def test_to_graph_convolution_config(self):
    config = LegacyConfig(dims=32, num_heads=8, dropout_rate=0.2)
    converted = config.to_graph_convolution_config()
    expected = (
        hetero_gnn.HeterogeneousGraphConvolutionConfig.dot_product_attention(
            dims=32, num_heads=8, dropout_rate=0.2
        )
    )
    self.assertEqual(converted, expected)
    self.assertEqual(
        converted.attention,
        graph_attention.DotProductAttentionConfig(num_heads=8, dims=32),
    )

  def test_make(self):
    schema = schema_lib.GraphSchema(
        node_sets={},
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1"),
            "e2": schema_lib.EdgeSchema(source="n1", target="n2"),
        },
    )
    gnn = LegacyConfig(dims=16, num_heads=4).make(schema)
    self.assertIsInstance(gnn, hetero_gnn.HeterogeneousGraphConvolution)
    graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.ones((3, 16))}, num_nodes=3
            ),
            "n2": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={"embedding": jnp.ones((2, 16))}, num_nodes=2
            ),
        },
        edge_sets={
            "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 1], [1, 2]])
            ),
            "e2": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array([[0, 2], [0, 1]])
            ),
        },
    )
    variables = gnn.init(jax.random.PRNGKey(0), graph, training=False)
    output = gnn.apply(variables, graph, training=False)
    self.assertEqual(
        output.node_sets["n1"].features["embedding"].shape, (3, 16)
    )
    self.assertEqual(
        output.node_sets["n2"].features["embedding"].shape, (2, 16)
    )

  def test_legacy_json_config_loads(self):
    config = LegacyConfig(dims=32, num_heads=8)
    restored = LegacyConfig.from_json(config.to_json())  # pyrefly: ignore[missing-attribute]
    self.assertEqual(restored, config)

  def test_architecture(self):
    architecture = LegacyConfig().architecture()
    self.assertEqual(
        architecture,
        hetero_gnn.HeterogeneousGraphConvolutionConfig.dot_product_attention().architecture(),
    )
    self.assertIn("DotProductAttention(heads=4, dims=128)", architecture)

  def test_alias(self):
    self.assertIs(
        hetero_graph_attention_network.HeterogeneousGraphAttentionNetwork,
        hetero_gnn.HeterogeneousGraphConvolution,
    )

  def test_invalid_message_aggregation(self):
    with self.assertRaisesRegex(
        ValueError, "message_aggregation must be 'sum'"
    ):
      LegacyConfig(message_aggregation="mean")


if __name__ == "__main__":
  absltest.main()
