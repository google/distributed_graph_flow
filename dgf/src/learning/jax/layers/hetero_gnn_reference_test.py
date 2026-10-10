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

"""Compares the heterogeneous graph convolution with dense references.

Each test re-implements a template (or option) with numpy, using the parameters
of the layer, and checks that the layer computes the same output.
"""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import standard
import jax
import jax.numpy as jnp
import numpy as np

Config = hetero_gnn.HeterogeneousGraphConvolutionConfig

DIMS = 16
NUM_HEADS = 2
NUM_N1_NODES = 6
NUM_N2_NODES = 5
NUM_EDGES = 20

# "e1": n1 -> n2, "e2": n2 -> n2.
SCHEMA = schema_lib.GraphSchema(
    node_sets={},
    edge_sets={
        "e1": schema_lib.EdgeSchema(source="n1", target="n2"),
        "e2": schema_lib.EdgeSchema(source="n2", target="n2"),
    },
)


def make_graph(seed: int = 3) -> jax_in_memory_graph.JaxInMemoryGraph:
  """Random graph following `SCHEMA`."""
  keys = iter(jax.random.split(jax.random.PRNGKey(seed), 6))
  node_sets = {
      name: jax_in_memory_graph.JaxInMemoryNodeSet(
          features={"embedding": jax.random.normal(next(keys), (n, DIMS))},
          num_nodes=n,
      )
      for name, n in [("n1", NUM_N1_NODES), ("n2", NUM_N2_NODES)]
  }
  edge_sets = {
      "e1": jax_in_memory_graph.JaxInMemoryEdgeSet(
          adjacency=jnp.stack([
              jax.random.randint(next(keys), (NUM_EDGES,), 0, NUM_N1_NODES),
              jax.random.randint(next(keys), (NUM_EDGES,), 0, NUM_N2_NODES),
          ])
      ),
      "e2": jax_in_memory_graph.JaxInMemoryEdgeSet(
          adjacency=jax.random.randint(
              next(keys), (2, NUM_EDGES), 0, NUM_N2_NODES
          )
      ),
  }
  return jax_in_memory_graph.JaxInMemoryGraph(
      node_sets=node_sets, edge_sets=edge_sets
  )


def embedding(
    graph: jax_in_memory_graph.JaxInMemoryGraph, nodeset: str
) -> np.ndarray:
  return np.asarray(graph.node_sets[nodeset].features["embedding"])


def edges(
    graph: jax_in_memory_graph.JaxInMemoryGraph, edgeset: str
) -> tuple[np.ndarray, np.ndarray]:
  adjacency = np.asarray(graph.edge_sets[edgeset].adjacency)
  return adjacency[0], adjacency[1]


def dense(params: dict[str, jax.Array], x: np.ndarray) -> np.ndarray:
  """Applies the parameters of a nn.Dense in numpy."""
  out = x @ np.asarray(params["kernel"])
  if "bias" in params:
    out = out + np.asarray(params["bias"])
  return out


def leaky_relu(x: np.ndarray, negative_slope: float = 0.2) -> np.ndarray:
  return np.where(x >= 0, x, negative_slope * x)


def elu(x: np.ndarray) -> np.ndarray:
  return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def segment_sum(values: np.ndarray, idxs: np.ndarray, num: int) -> np.ndarray:
  out = np.zeros((num,) + values.shape[1:])
  np.add.at(out, idxs, values)
  return out


def softmax_aggregate(
    logits: np.ndarray, messages: np.ndarray, dst: np.ndarray, num_dst: int
) -> np.ndarray:
  """Reference of the attention aggregation.

  Args:
    logits: Logits, shape [E, H].
    messages: Messages, shape [E, H * head_dims].
    dst: Target node of each edge.
    num_dst: Number of target nodes.

  Returns:
    The aggregated messages, shape [N_dst, H * head_dims].
  """
  num_edges, num_heads = logits.shape
  messages = messages.reshape(num_edges, num_heads, -1)
  out = np.zeros((num_dst,) + messages.shape[1:])
  for node in range(num_dst):
    node_edges = np.flatnonzero(dst == node)
    if node_edges.size:
      weights = np.exp(logits[node_edges] - logits[node_edges].max(axis=0))
      weights /= weights.sum(axis=0)
      out[node] = np.einsum("kh,khd->hd", weights, messages[node_edges])
  return out.reshape(num_dst, -1)


def update_params(
    params: dict[str, dict[str, jax.Array]],
) -> dict[str, jax.Array]:
  """Parameters of the single dense layer of the update block."""
  (update_key,) = [k for k in params if k.startswith("GenericBlock_")]
  return params[update_key]["dense_0"]


class HeteroGNNReferenceTest(parameterized.TestCase):

  def apply(
      self, config: hetero_gnn.HeterogeneousGraphConvolutionConfig
  ) -> tuple[
      jax_in_memory_graph.JaxInMemoryGraph,
      dict[str, dict[str, jax.Array]],
      jax_in_memory_graph.JaxInMemoryGraph,
  ]:
    """Returns the (input graph, parameters, output graph)."""
    graph = make_graph()
    gnn = config.make(SCHEMA)
    variables = gnn.init(jax.random.PRNGKey(0), graph, training=False)
    return (
        graph,
        variables["params"],
        gnn.apply(variables, graph, training=False),
    )

  def assert_close(self, actual: jax.Array, expected: np.ndarray):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def test_graphsage(self):
    graph, params, out = self.apply(
        Config.graphsage(dims=DIMS, plan=[("e1", False)])
    )
    h1, h2 = embedding(graph, "n1"), embedding(graph, "n2")
    src, dst = edges(graph, "e1")
    neighbor_mean = segment_sum(h1[src], dst, NUM_N2_NODES) / np.maximum(
        segment_sum(np.ones((NUM_EDGES, 1)), dst, NUM_N2_NODES), 1.0
    )
    expected = np.maximum(
        dense(update_params(params), np.concatenate([h2, neighbor_mean], 1)), 0
    )
    self.assert_close(out.node_sets["n2"].features["embedding"], expected)
    # "n1" is not a target in the plan and stays unchanged.
    np.testing.assert_array_equal(out.node_sets["n1"].features["embedding"], h1)

  def test_gcn(self):
    graph, params, out = self.apply(
        Config.gcn(dims=DIMS, plan=[("e2", False), ("e2", True)])
    )
    h = embedding(graph, "n2")
    src, dst = edges(graph, "e2")

    # Dense reference. The plan has two relations, "e2" forward (adjacency
    # `A`) and reversed (`A^T`), each normalized with its own degrees:
    # `D_in^-1/2 A_r D_out^-1/2` with `D = 1 + degree`. The node embedding is
    # added as is (`combine="sum"`).
    adj = np.zeros((NUM_N2_NODES, NUM_N2_NODES))
    np.add.at(adj, (dst, src), 1.0)

    def normalize(adjacency: np.ndarray) -> np.ndarray:
      inv_sqrt_in = 1.0 / np.sqrt(1.0 + adjacency.sum(axis=1))
      inv_sqrt_out = 1.0 / np.sqrt(1.0 + adjacency.sum(axis=0))
      return inv_sqrt_in[:, None] * adjacency * inv_sqrt_out[None, :]

    norm_adj = normalize(adj) + normalize(adj.T) + np.eye(NUM_N2_NODES)
    expected = np.maximum(dense(update_params(params), norm_adj @ h), 0)
    self.assert_close(out.node_sets["n2"].features["embedding"], expected)

  @parameterized.parameters("gat", "gatv2")
  def test_gat_and_gatv2(self, template: str):
    graph, params, out = self.apply(
        getattr(Config, template)(
            dims=DIMS, num_heads=NUM_HEADS, plan=[("e1", False)]
        )
    )
    h1, h2 = embedding(graph, "n1"), embedding(graph, "n2")
    src, dst = edges(graph, "e1")
    attention = params["attention"]

    if template == "gat":
      logits = leaky_relu(
          dense(attention["att_e1_fwd_src"], h1)[src]
          + dense(attention["att_e1_fwd_dst"], h2)[dst]
      )
    else:
      hidden = leaky_relu(
          dense(attention["att_e1_fwd_src"], h1)[src]
          + dense(attention["att_e1_fwd_dst"], h2)[dst]
      ).reshape(NUM_EDGES, NUM_HEADS, -1)
      logits = np.einsum(
          "ehd,hd->eh", hidden, np.asarray(attention["att_e1_fwd"])
      )

    values = dense(params["msg_e1_fwd"]["dense_0"], h1)[src]  # W_r h_j
    messages = softmax_aggregate(logits, values, dst, NUM_N2_NODES)
    expected = elu(
        dense(update_params(params), np.concatenate([h2, messages], 1))
    )
    self.assert_close(out.node_sets["n2"].features["embedding"], expected)

  def test_dot_product_attention(self):
    config = Config.dot_product_attention(
        dims=DIMS, num_heads=NUM_HEADS, plan=[("e1", False)]
    )
    # Simpler update and post blocks to keep the reference short.
    config.update = standard.GenericBlockConfig("L", dims=DIMS)
    config.post = standard.identity()
    graph, params, out = self.apply(config)
    h1, h2 = embedding(graph, "n1"), embedding(graph, "n2")
    src, dst = edges(graph, "e1")
    attention = params["attention"]
    head_dims = DIMS // NUM_HEADS

    queries = dense(attention["q_proj_n2"], h2).reshape(
        -1, NUM_HEADS, head_dims
    )
    keys = dense(attention["k_proj_n1"], h1).reshape(-1, NUM_HEADS, head_dims)
    # logits = q^T W_att^T k, per head.
    logits = np.einsum(
        "ehk,hdk,ehd->eh",
        queries[dst],
        np.asarray(attention["w_att_e1_fwd"]),
        keys[src],
    ) / np.sqrt(head_dims)

    # "LAL" message on concat([source, target]), with the default silu.
    hidden = (
        dense(params["msg_e1_fwd_src"], h1)[src]
        + dense(params["msg_e1_fwd_dst"], h2)[dst]
    )
    hidden = hidden / (1.0 + np.exp(-hidden))  # silu
    edge_messages = dense(params["msg_e1_fwd_out"], hidden)
    messages = softmax_aggregate(logits, edge_messages, dst, NUM_N2_NODES)
    expected = (
        dense(update_params(params), np.concatenate([h2, messages], 1)) + h2
    )
    self.assert_close(out.node_sets["n2"].features["embedding"], expected)

  @parameterized.parameters(*hetero_gnn.RelationAggregation)
  def test_relation_aggregation(
      self, relation_aggregation: hetero_gnn.RelationAggregation
  ):
    # Sum of the raw source embeddings, per relation.
    graph, params, out = self.apply(
        Config(
            plan=[("e1", False), ("e2", False)],
            dims=DIMS,
            message_consumes_target=False,
            relation_aggregation=relation_aggregation,
            residual=False,
            message=standard.identity(),
            update=standard.GenericBlockConfig("L", dims=DIMS),
            post=standard.identity(),
        )
    )
    h1, h2 = embedding(graph, "n1"), embedding(graph, "n2")
    src1, dst1 = edges(graph, "e1")
    src2, dst2 = edges(graph, "e2")
    aggregate_e1 = segment_sum(h1[src1], dst1, NUM_N2_NODES)
    aggregate_e2 = segment_sum(h2[src2], dst2, NUM_N2_NODES)
    combined = {
        hetero_gnn.RelationAggregation.SUM: aggregate_e1 + aggregate_e2,
        hetero_gnn.RelationAggregation.MEAN: (aggregate_e1 + aggregate_e2) / 2,
        hetero_gnn.RelationAggregation.CONCAT: np.concatenate(
            [aggregate_e1, aggregate_e2], 1
        ),
    }[relation_aggregation]
    expected = dense(update_params(params), np.concatenate([h2, combined], 1))
    self.assert_close(out.node_sets["n2"].features["embedding"], expected)


if __name__ == "__main__":
  absltest.main()
