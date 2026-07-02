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
from typing import List, Optional, Tuple
import dataclasses_json
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.hetero_gnn import Plan
from dgf.src.learning.jax.layers.hetero_gnn import sort_plan
from dgf.src.learning.jax.layers.hetero_gnn import SortedPlan
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from flax import linen as nn
import jax
import jax.numpy as jnp


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
    message_pooling: Pooling method (unused in GAT attention aggregation but
      kept for signature compatibility).
    num_heads: Number of attention heads.
    message: Optional module to apply to edge features to generate values.
      Defaults to a single-layer MLP.
    update: Optional module to apply to node embeddings after message passing,
      combining the old embedding and aggregated messages. Defaults to a
      single-layer MLP with layer norm and activation.
    post: Optional module applied after the update step, typically a
      transformer-like MLP. Defaults to a two-layer ResidualMLP.
  """

  plan: Optional[List[Tuple[str, bool]]] = None
  embedding_feature: str = "embedding"
  dims: int = 128
  dropout_rate: float = 0.1
  message_pooling: str = "sum"
  num_heads: int = 4

  message: Optional[common.BuildableModule] = layer_registry.field(default=None)
  update: Optional[common.BuildableModule] = layer_registry.field(default=None)
  post: Optional[common.BuildableModule] = layer_registry.field(default=None)

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
      self, schema: schema_lib.GraphSchema, name: Optional[str] = None
  ) -> "HeterogeneousGraphAttentionNetwork":
    return HeterogeneousGraphAttentionNetwork(
        config=self, schema=schema, name=name
    )

  def architecture(self) -> str:
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
        combined = config.update.make(name=f"update_{dst_nodeset_name}")(  # pyrefly: ignore[unexpected-keyword]
            combined, training=training
        )
        node_values = combined + dst_values
        node_values = config.post.make(name=f"post_{dst_nodeset_name}")(  # pyrefly: ignore[unexpected-keyword]
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

        # 1. Compute relation-specific attention
        # Gather Q and K for edges
        q_t = q_dst[target_idxs]  # [E, H, head_dim]

        k_src = k_projs[message_src_nodeset](src_values)
        k_src = k_src.reshape(src_values.shape[0], num_heads, head_dim)
        k_s = k_src[source_idxs]  # [E, H, head_dim]

        relation_name = f"{edgeset_name}_{'rev' if reverse else 'fwd'}"
        w_att = self.param(
            f"w_att_{relation_name}",
            nn.initializers.glorot_uniform(),
            (num_heads, head_dim, head_dim),
        )

        # Project Key with relation-specific matrix
        k_s_rel = jnp.einsum("ehd,hdk->ehk", k_s, w_att)

        # Compute attention logits
        logits = jnp.einsum("ehd,ehd->eh", q_t, k_s_rel) / jnp.sqrt(head_dim)

        # 2. Compute Edge Messages using config.message
        src_edge_values = src_values[source_idxs]
        dst_edge_values = dst_values[target_idxs]
        edge_values = jnp.concatenate(
            [src_edge_values, dst_edge_values], axis=-1
        )

        # Apply config.message on the edges to get Values
        messages = config.message.make(name=f"message_{relation_name}")(  # pyrefly: ignore[unexpected-keyword]
            edge_values, training=training
        )  # [E, dims]
        messages = messages.reshape(messages.shape[0], num_heads, head_dim)

        # Softmax + aggregation (per edgeset)
        # Local max logit per target node for this relation
        local_max = jax.ops.segment_max(logits, target_idxs, num_dst_nodes)

        # Local exp logits
        exp_logits = jnp.exp(logits - local_max[target_idxs])

        # Local sum of exp
        local_sum_exp = jax.ops.segment_sum(
            exp_logits, target_idxs, num_dst_nodes
        )

        # Local attention weights
        attn_weights = exp_logits / (
            local_sum_exp[target_idxs] + 1e-9
        )  # [E, H]

        # Weighted sum of messages for this relation
        weighted_messages = (
            attn_weights[..., None] * messages
        )  # [E, H, head_dim]
        agg_msg = jax.ops.segment_sum(
            weighted_messages, target_idxs, num_dst_nodes
        )  # [N_dst, H, head_dim]

        all_aggregated_messages.append(agg_msg)

      # Combine aggregated messages from all relations (Sum aggregation)
      aggregated_messages = all_aggregated_messages[0]
      for msg in all_aggregated_messages[1:]:
        aggregated_messages = aggregated_messages + msg

      aggregated_messages = aggregated_messages.reshape(num_dst_nodes, dims)

      # Join messages + update
      combined = jnp.concatenate([dst_values, aggregated_messages], axis=1)
      combined = config.update.make(name=f"update_{dst_nodeset_name}")(  # pyrefly: ignore[unexpected-keyword]
          combined, training=training
      )

      # Residual
      node_values = combined + dst_values

      # Feed-forward
      node_values = config.post.make(name=f"post_{dst_nodeset_name}")(  # pyrefly: ignore[unexpected-keyword]
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
