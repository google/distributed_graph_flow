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

"""Graph Attention Network layers for heterogeneous graphs."""

import dataclasses
import textwrap
import dataclasses_json
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from flax import linen as nn
import jax
import jax.numpy as jnp

Plan = hetero_gnn.Plan
sort_plan = hetero_gnn.sort_plan
SortedPlan = hetero_gnn.SortedPlan


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class HeterogeneousGraphAttentionNetworkConfig(common.ArchitectureProvider):
  """Configuration for HeterogeneousGraphAttentionNetwork.

  Attributes:
    plan: Message passing plan. A list of (edge_set_name, is_reversed) tuples.
      If None, messages are passed along all edges in both directions.
    embedding_feature: Name of the node feature to use for embeddings.
    dims: Dimension of the embeddings and hidden layers. Used to build the
      default values of `message`, `update`, and `post`.
    dropout_rate: Dropout rate. Used to build the default values of `update` and
      `post`.
    message_aggregation: Aggregation method (unused in GAT attention aggregation
      but kept for signature compatibility).
    num_heads: Number of attention heads.
    message: Optional module to apply to edge features to generate values.
      Defaults to a single-layer MLP.
    update: Optional module to apply to node embeddings after message passing,
      combining the old embedding and aggregated messages. Defaults to a
      single-layer MLP with layer norm and activation.
    post: Optional module applied after the update step, typically a
      transformer-like MLP. Defaults to a two-layer ResidualMLP.
    force_basic_implementation: If True, always compute messages with the
      generic per-edge implementation (no node-level projection, no edge
      sorting). Mostly useful in tests to validate the optimized paths.
  """

  plan: list[tuple[str, bool]] | None = None
  embedding_feature: str = "embedding"
  dims: int = 128
  dropout_rate: float = 0.1
  message_aggregation: str = "sum"
  num_heads: int = 4
  force_basic_implementation: bool = False

  message: common.GenericLayer | None = layer_registry.field(default=None)
  update: common.GenericLayer | None = layer_registry.field(default=None)
  post: common.GenericLayer | None = layer_registry.field(default=None)

  def __post_init__(self):
    if self.message is None:
      self.message = standard.GenericBlockConfig("LAL", dims=self.dims)
    if self.update is None:
      self.update = standard.GenericBlockConfig(
          "LADL", dims=self.dims, dropout_rate=self.dropout_rate
      )
    if self.post is None:
      self.post = standard.modern_residual_mlp(
          dims=self.dims, dropout_rate=self.dropout_rate
      )

  def make(
      self, schema: schema_lib.GraphSchema, name: str | None = None
  ) -> "HeterogeneousGraphAttentionNetwork":
    return HeterogeneousGraphAttentionNetwork(
        config=self, schema=schema, name=name
    )

  def architecture(self) -> str:
    assert self.message is not None
    assert self.update is not None
    assert self.post is not None
    parts = []
    parts.append("X = ...")
    parts.append(
        f"HeterogeneousGraphAttentionNetwork (heads={self.num_heads}):"
    )
    parts.append("  Message/Value:")
    parts.append(textwrap.indent(self.message.architecture(), prefix="    "))
    parts.append("  Update:")
    parts.append(textwrap.indent(self.update.architecture(), prefix="    "))
    parts.append("Residual(X)")
    parts.append("# Post Attention FFN")
    parts.append(self.post.architecture())
    return "\n".join(parts)


class HeterogeneousGraphAttentionNetwork(nn.Module):
  """A single layer of heterogeneous Graph Attention Network.

  This layer performs multi-head relation-aware self-attention on a
  heterogeneous graph.
  """

  config: HeterogeneousGraphAttentionNetworkConfig
  schema: schema_lib.GraphSchema

  sorted_plan: SortedPlan = dataclasses.field(init=False)

  def __post_init__(self):
    if self.config.plan is None:
      self.config.plan = []
      for edge_name in self.schema.edge_sets:
        self.config.plan.append((edge_name, False))
        self.config.plan.append((edge_name, True))
    self.sorted_plan = sort_plan(self.config.plan, self.schema)
    super().__post_init__()

  @nn.compact
  def __call__(
      self,
      graph: jax_in_memory_graph.JaxInMemoryGraph,
      training: bool,
  ) -> jax_in_memory_graph.JaxInMemoryGraph:
    """Computes the Heterogeneous Graph Attention Network layers."""
    config = self.config
    assert config.message is not None
    assert config.update is not None
    assert config.post is not None

    num_heads = config.num_heads
    dims = config.dims
    assert (
        dims % num_heads == 0
    ), f"dims ({dims}) must be divisible by num_heads ({num_heads})"
    head_dim = dims // num_heads
    scale = 1.0 / jnp.sqrt(head_dim)
    lal = None
    if not config.force_basic_implementation and isinstance(
        config.message, standard.GenericBlockConfig
    ):
      lal = config.message.as_linear_activation_linear()

    # Initialize with all original node sets to avoid dropping any.
    new_node_sets = dict(graph.node_sets)

    # Save the nodeset values for the residual.
    res_node_features = {
        nodeset_name: nodeset.features[config.embedding_feature]
        for nodeset_name, nodeset in graph.node_sets.items()
    }

    # Define Query and Key projections for each nodeset.
    q_projs = {
        ns_name: nn.Dense(dims, name=f"q_proj_{ns_name}")
        for ns_name in self.schema.node_sets
    }
    k_projs = {
        ns_name: nn.Dense(dims, name=f"k_proj_{ns_name}")
        for ns_name in self.schema.node_sets
    }
    k_cache: dict[str, jax.Array] = {}

    # Message passing for each nodeset.
    for dst_nodeset_name in sorted(self.sorted_plan.keys()):
      edges_and_src_nodes = self.sorted_plan[dst_nodeset_name]
      num_dst_nodes = graph.node_sets[dst_nodeset_name].num_nodes
      assert num_dst_nodes is not None
      dst_values = res_node_features[dst_nodeset_name]

      if not edges_and_src_nodes:
        # No incoming messages, keep the original node values (or apply post MLP).
        combined = jnp.concatenate(
            [dst_values, jnp.zeros((num_dst_nodes, dims))], axis=1
        )
        combined = config.update.make(name=f"update_{dst_nodeset_name}")(
            combined, training=training
        )
        node_values = combined + dst_values
        node_values = config.post.make(name=f"post_{dst_nodeset_name}")(
            node_values, training=training
        )
        new_node_sets[dst_nodeset_name] = (
            jax_in_memory_graph.JaxInMemoryNodeSet(
                num_nodes=num_dst_nodes,
                features={self.config.embedding_feature: node_values},
            )
        )
        continue

      all_aggregated_messages = []

      # Pre-compute Q for target nodeset
      q_dst = q_projs[dst_nodeset_name](dst_values)
      q_dst = q_dst.reshape(num_dst_nodes, num_heads, head_dim)

      for edgeset_name, message_src_nodeset, reverse in edges_and_src_nodes:
        source_idxs = graph.edge_sets[edgeset_name].adjacency[0]
        target_idxs = graph.edge_sets[edgeset_name].adjacency[1]
        if reverse:
          source_idxs, target_idxs = target_idxs, source_idxs

        src_values = res_node_features[message_src_nodeset]
        num_src_nodes = src_values.shape[0]
        num_edges = source_idxs.shape[0]
        do_sort = (
            not config.force_basic_implementation
            and hetero_gnn.should_sort_edges(
                num_src_nodes, num_dst_nodes, num_edges
            )
        )
        if do_sort:
          source_idxs, target_idxs = hetero_gnn.sort_edges_by_dst(
              source_idxs, target_idxs, num_src_nodes, num_dst_nodes
          )

        # 1. Compute relation-specific attention logits.
        # Project K for the source nodeset (once per nodeset, shared across
        # relations).
        if message_src_nodeset not in k_cache:
          k_src_proj = k_projs[message_src_nodeset](src_values)
          k_cache[message_src_nodeset] = k_src_proj.reshape(
              num_src_nodes, num_heads, head_dim
          )
        k_src = k_cache[message_src_nodeset]  # [N_src, H, head_dim]

        relation_name = f"{edgeset_name}_{'rev' if reverse else 'fwd'}"
        w_att = self.param(
            f"w_att_{relation_name}",
            nn.initializers.glorot_uniform(),
            (num_heads, head_dim, head_dim),
        )

        # logits[e] = <q[dst(e)], w_att k[src(e)]>. The relation-specific
        # matrix is applied on the smallest of (source nodes, target nodes,
        # edges), then Q and K are gathered on the edges.
        if num_src_nodes <= num_dst_nodes and num_src_nodes <= num_edges:
          k_src_rel = jnp.einsum("nhd,hdk->nhk", k_src, w_att)
          logits = jnp.sum(q_dst[target_idxs] * k_src_rel[source_idxs], -1)
        elif num_dst_nodes <= num_edges:
          q_dst_rel = jnp.einsum("nhk,hdk->nhd", q_dst, w_att)
          logits = jnp.sum(q_dst_rel[target_idxs] * k_src[source_idxs], -1)
        else:
          k_s_rel = jnp.einsum("ehd,hdk->ehk", k_src[source_idxs], w_att)
          logits = jnp.sum(q_dst[target_idxs] * k_s_rel, -1)
        logits = logits * scale  # [E, H]

        # 2. Compute the edge messages (values) using config.message.
        if lal is not None:
          # The message block is LAL (linear, activation, linear). Project the
          # source and target nodes separately and sum on the edges, which is
          # equivalent to the linear on concat([src, dst]) but cheaper when
          # there are fewer nodes than edges.
          dense_src = nn.Dense(
              lal.hidden_dims, use_bias=False, name=f"msg_{relation_name}_src"
          )
          dense_dst = nn.Dense(
              lal.hidden_dims, use_bias=True, name=f"msg_{relation_name}_dst"
          )
          dense_out = nn.Dense(
              lal.output_dims, use_bias=True, name=f"msg_{relation_name}_out"
          )
          ps_edge = (
              dense_src(src_values)[source_idxs]
              if num_src_nodes < num_edges
              else dense_src(src_values[source_idxs])
          )  # [E, hidden_dims]
          pd_edge = (
              dense_dst(dst_values)[target_idxs]
              if num_dst_nodes < num_edges
              else dense_dst(dst_values[target_idxs])
          )  # [E, hidden_dims]
          messages = dense_out(lal.activation(ps_edge + pd_edge))  # [E, dims]
        else:
          # Basic / expensive fallback: apply config.message on the edges.
          edge_values = jnp.concatenate(
              [src_values[source_idxs], dst_values[target_idxs]], axis=-1
          )  # [E, 2 * dims]
          messages = config.message.make(name=f"message_{relation_name}")(
              edge_values, training=training
          )  # [E, dims]
        messages = messages.reshape(num_edges, num_heads, head_dim)

        # 3. Softmax over the incoming edges of each target node + weighted
        # sum of the messages (per relation).
        # Local max logit per target node, for numerical stability.
        local_max = jax.ops.segment_max(
            jax.lax.stop_gradient(logits),
            target_idxs,
            num_dst_nodes,
            indices_are_sorted=do_sort,
        )  # [N_dst, H]
        exp_logits = jnp.exp(logits - local_max[target_idxs])  # [E, H]
        # Unnormalized weighted sum of messages and softmax denominator. The
        # normalization is applied once per node rather than once per edge.
        agg_numer = jax.ops.segment_sum(
            exp_logits[..., None] * messages,
            target_idxs,
            num_dst_nodes,
            indices_are_sorted=do_sort,
        )  # [N_dst, H, head_dim]
        agg_denom = jax.ops.segment_sum(
            exp_logits,
            target_idxs,
            num_dst_nodes,
            indices_are_sorted=do_sort,
        )  # [N_dst, H]
        agg_msg = agg_numer / (agg_denom[..., None] + 1e-9)

        all_aggregated_messages.append(agg_msg)

      # Combine aggregated messages from all relations (Sum aggregation)
      aggregated_messages = all_aggregated_messages[0]
      for msg in all_aggregated_messages[1:]:
        aggregated_messages = aggregated_messages + msg

      aggregated_messages = aggregated_messages.reshape(num_dst_nodes, dims)

      # Join messages + update
      combined = jnp.concatenate([dst_values, aggregated_messages], axis=1)
      combined = config.update.make(name=f"update_{dst_nodeset_name}")(
          combined, training=training
      )

      # Residual
      node_values = combined + dst_values

      # Feed-forward
      node_values = config.post.make(name=f"post_{dst_nodeset_name}")(
          node_values, training=training
      )

      new_node_sets[dst_nodeset_name] = jax_in_memory_graph.JaxInMemoryNodeSet(
          num_nodes=num_dst_nodes,
          features={self.config.embedding_feature: node_values},
      )

    return jax_in_memory_graph.JaxInMemoryGraph(
        node_sets=new_node_sets,
        edge_sets=graph.edge_sets,
    )
