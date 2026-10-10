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

"""Graph attention layer for attention MPNN."""

import abc
from collections.abc import Sequence
import dataclasses
import math
import dataclasses_json
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import message_passing_ops
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from flax import linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt

# Attention logits of each edge and head, indexed by relation name.
LogitsByRelation = dict[str, jt.Float[jt.Array, "num_edges num_heads"]]


class GraphAttentionConfig(common.ArchitectureProvider):
  """Base config of an attention score."""

  num_heads: int

  @abc.abstractmethod
  def make(self, name: str | None = None) -> nn.Module:
    """Creates the attention score module."""
    raise NotImplementedError


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class DotProductAttentionConfig(GraphAttentionConfig):
  """Relation-aware scaled dot-product attention.

  logits_ij = <W_att^r Q_dst x_i, K_src x_j> / sqrt(head_dims)

  where `Q_dst` (resp. `K_src`) is a query (resp. key) projection specific to
  the target (resp. source) node set, shared by all the relations, and
  `W_att^r` is a per-head matrix specific to the relation `r`.

  Attributes:
    num_heads: Number of attention heads.
    dims: Dimension of the queries and keys. Must be divisible by `num_heads`.
  """

  num_heads: int = 4
  dims: int = 128

  def __post_init__(self):
    if self.dims % self.num_heads != 0:
      raise ValueError(
          f"dims ({self.dims}) must be divisible by num_heads"
          f" ({self.num_heads})."
      )

  def make(self, name: str | None = None) -> "DotProductAttention":
    return DotProductAttention(config=self, name=name)

  def architecture(self) -> str:
    return f"DotProductAttention(heads={self.num_heads}, dims={self.dims})"


class DotProductAttention(nn.Module):
  """Module of `DotProductAttentionConfig`."""

  config: DotProductAttentionConfig

  @nn.compact
  def __call__(
      self, relations: Sequence[message_passing_ops.Relation]
  ) -> LogitsByRelation:
    num_heads = self.config.num_heads
    dims = self.config.dims
    head_dims = dims // num_heads
    scale = 1.0 / math.sqrt(head_dims)

    # The queries (resp. keys) are computed once per target (resp. source) node
    # set, and shared by the relations.
    queries: dict[str, jt.Float[jt.Array, "num_nodes num_heads head_dims"]] = {}
    keys: dict[str, jt.Float[jt.Array, "num_nodes num_heads head_dims"]] = {}

    logits_by_relation = {}
    for relation in relations:
      if relation.dst_nodeset not in queries:
        queries[relation.dst_nodeset] = nn.Dense(
            dims, name=f"q_proj_{relation.dst_nodeset}"
        )(relation.dst_values).reshape(
            relation.num_dst_nodes, num_heads, head_dims
        )
      if relation.src_nodeset not in keys:
        keys[relation.src_nodeset] = nn.Dense(
            dims, name=f"k_proj_{relation.src_nodeset}"
        )(relation.src_values).reshape(
            relation.num_src_nodes, num_heads, head_dims
        )
      q_dst = queries[relation.dst_nodeset]  # [N_dst, H, head_dims]
      k_src = keys[relation.src_nodeset]  # [N_src, H, head_dims]

      w_att = self.param(
          f"w_att_{relation.name}",
          nn.initializers.glorot_uniform(),
          (num_heads, head_dims, head_dims),
      )

      # logits[e] = <w_att q[dst(e)], k[src(e)]>. The relation-specific matrix
      # is applied on the smallest of (source nodes, target nodes, edges), then
      # the queries and keys are gathered on the edges.
      source_idxs = relation.source_idxs
      target_idxs = relation.target_idxs
      if (
          relation.num_src_nodes <= relation.num_dst_nodes
          and relation.num_src_nodes <= relation.num_edges
      ):
        k_src_rel = jnp.einsum("nhd,hdk->nhk", k_src, w_att)
        logits = jnp.sum(q_dst[target_idxs] * k_src_rel[source_idxs], -1)
      elif relation.num_dst_nodes <= relation.num_edges:
        q_dst_rel = jnp.einsum("nhk,hdk->nhd", q_dst, w_att)
        logits = jnp.sum(q_dst_rel[target_idxs] * k_src[source_idxs], -1)
      else:
        k_src_rel = jnp.einsum("ehd,hdk->ehk", k_src[source_idxs], w_att)
        logits = jnp.sum(q_dst[target_idxs] * k_src_rel, -1)
      logits_by_relation[relation.name] = logits * scale  # [E, H]
    return logits_by_relation


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class GatAttentionConfig(GraphAttentionConfig):
  """Graph Attention Network (GAT) attention.

  logits_ij = LeakyReLU(s_src^r(x_j) + s_dst^r(x_i))

  where `s_src^r` and `s_dst^r` are linear projections (one output per head)
  specific to the relation `r`. This is the GAT attention of "Graph Attention
  Networks" (Veličković et al., 2018), with a score projection independent
  from the message projection.

  Attributes:
    num_heads: Number of attention heads.
    negative_slope: Slope of the LeakyReLU for negative values.
  """

  num_heads: int = 8
  negative_slope: float = 0.2

  def make(self, name: str | None = None) -> "GatAttention":
    return GatAttention(config=self, name=name)

  def architecture(self) -> str:
    return (
        f"GatAttention(heads={self.num_heads},"
        f" negative_slope={self.negative_slope})"
    )


class GatAttention(nn.Module):
  """Module of `GatAttentionConfig`."""

  config: GatAttentionConfig

  @nn.compact
  def __call__(
      self, relations: Sequence[message_passing_ops.Relation]
  ) -> LogitsByRelation:
    num_heads = self.config.num_heads
    logits_by_relation = {}
    for relation in relations:
      # The scores are projected on the nodes, then summed on the edges.
      src_scores = nn.Dense(
          num_heads, use_bias=False, name=f"att_{relation.name}_src"
      )(
          relation.src_values
      )  # [N_src, H]
      dst_scores = nn.Dense(
          num_heads, use_bias=False, name=f"att_{relation.name}_dst"
      )(
          relation.dst_values
      )  # [N_dst, H]
      logits_by_relation[relation.name] = jax.nn.leaky_relu(
          src_scores[relation.source_idxs] + dst_scores[relation.target_idxs],
          negative_slope=self.config.negative_slope,
      )  # [E, H]
    return logits_by_relation


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Gatv2AttentionConfig(GraphAttentionConfig):
  """GATv2 attention.

  logits_ij = a^r . LeakyReLU(W_src^r x_j + W_dst^r x_i)

  where `W_src^r` and `W_dst^r` are linear projections and `a^r` is a per-head
  vector, all specific to the relation `r`. This is the attention of "How
  Attentive are Graph Attention Networks?" (Brody et al., 2021), with a score
  projection independent from the message projection.

  Attributes:
    num_heads: Number of attention heads.
    dims: Dimension of the hidden projection. Must be divisible by `num_heads`.
    negative_slope: Slope of the LeakyReLU for negative values.
  """

  num_heads: int = 8
  dims: int = 128
  negative_slope: float = 0.2

  def __post_init__(self):
    if self.dims % self.num_heads != 0:
      raise ValueError(
          f"dims ({self.dims}) must be divisible by num_heads"
          f" ({self.num_heads})."
      )

  def make(self, name: str | None = None) -> "Gatv2Attention":
    return Gatv2Attention(config=self, name=name)

  def architecture(self) -> str:
    return (
        f"Gatv2Attention(heads={self.num_heads}, dims={self.dims},"
        f" negative_slope={self.negative_slope})"
    )


class Gatv2Attention(nn.Module):
  """Module of `Gatv2AttentionConfig`."""

  config: Gatv2AttentionConfig

  @nn.compact
  def __call__(
      self, relations: Sequence[message_passing_ops.Relation]
  ) -> LogitsByRelation:
    num_heads = self.config.num_heads
    dims = self.config.dims
    head_dims = dims // num_heads
    logits_by_relation = {}
    for relation in relations:
      dense_src = nn.Dense(
          dims, use_bias=False, name=f"att_{relation.name}_src"
      )
      dense_dst = nn.Dense(dims, use_bias=True, name=f"att_{relation.name}_dst")
      hidden = jax.nn.leaky_relu(
          message_passing_ops.project_on_edges(
              dense_src, relation.src_values, relation.source_idxs
          )
          + message_passing_ops.project_on_edges(
              dense_dst, relation.dst_values, relation.target_idxs
          ),
          negative_slope=self.config.negative_slope,
      ).reshape(
          relation.num_edges, num_heads, head_dims
      )  # [E, H, head_dims]
      attention_vector = self.param(
          f"att_{relation.name}",
          nn.initializers.glorot_uniform(),
          (num_heads, head_dims),
      )
      logits_by_relation[relation.name] = jnp.einsum(
          "ehd,hd->eh", hidden, attention_vector
      )  # [E, H]
    return logits_by_relation
