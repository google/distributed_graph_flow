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

"""GNN layers for heterogeneous graphs."""

import collections
import dataclasses
import enum
import textwrap
import dataclasses_json
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import graph_attention
from dgf.src.learning.jax.layers import message_passing_ops
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from flax import linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt

Relation = message_passing_ops.Relation

# A plan is a list of (edge name, is_reversed) indicating which edge
# is used to propagate the message.
Plan = list[tuple[str, bool]]

# A sorted plan groups plan items by destination nodesets. More precisely, a
# sorted plan maps for each target nodesets, the list of
# (edgeset name, source nodeset, is_reversed).
SortedPlan = dict[str, list[tuple[str, str, bool]]]


class MessageAggregation(enum.Enum):
  """How the messages received by a node are aggregated, within a relation.

  See `HeterogeneousGraphConvolutionConfig.message_aggregation`.
  """

  SUM = "sum"
  MEAN = "mean"
  SYMMETRIC = "symmetric"
  ATTENTION = "attention"


class RelationAggregation(enum.Enum):
  """How the aggregated messages of the different relations are combined.

  See `HeterogeneousGraphConvolutionConfig.relation_aggregation`.
  """

  SUM = "sum"
  MEAN = "mean"
  CONCAT = "concat"


class Combine(enum.Enum):
  """How the node embedding is combined with the aggregated messages.

  See `HeterogeneousGraphConvolutionConfig.combine`.
  """

  CONCAT = "concat"
  SUM = "sum"


def sort_plan(plan: Plan, schema: schema_lib.GraphSchema) -> SortedPlan:
  """Sorts a message passing plan by destination nodeset.

  Args:
    plan: A list of (edge name, is_reversed) indicating which edge is used to
      propagate the message.
    schema: The graph schema.

  Returns:
    A dictionary where keys are destination nodeset names and values are lists
    of (edgeset name, source nodeset name, is_reversed).
  """
  sorted_plan = collections.defaultdict(list)
  for edgeset_name, reverse in plan:
    orig_src = schema.edge_sets[edgeset_name].source
    orig_dst = schema.edge_sets[edgeset_name].target
    if reverse:
      # Message flows from orig_dst to orig_src
      message_dst = orig_src
      message_src = orig_dst
    else:
      # Message flows from orig_src to orig_dst
      message_dst = orig_dst
      message_src = orig_src
    sorted_plan[message_dst].append((edgeset_name, message_src, reverse))
  return sorted_plan


def mean_weights(relation: Relation) -> jt.Float[jt.Array, "num_dst_nodes 1"]:
  """Weights of the MEAN aggregation: `1 / in_degree(i)` for each target node.

  Nodes without incoming edges get a weight of 1 (their aggregate is zero).

  Args:
    relation: The relation.

  Returns:
    The weight of each target node.
  """
  in_degrees = message_passing_ops.node_degrees(
      relation.target_idxs,
      relation.num_dst_nodes,
      relation.dst_values.dtype,
      relation.indices_are_sorted,
  )
  return 1.0 / jnp.maximum(in_degrees, 1.0)


def symmetric_weights(
    relation: Relation,
) -> tuple[
    jt.Float[jt.Array, "num_src_nodes 1"], jt.Float[jt.Array, "num_dst_nodes 1"]
]:
  """Weights of the SYMMETRIC (GCN) aggregation.

  The weight of the edge (j -> i) is `1 / sqrt((1 + out_deg(j)) * (1 +
  in_deg(i)))`, factored into a source and a target weight. The "+1" follows
  GCN's self-loop convention and keeps isolated nodes finite.

  Args:
    relation: The relation.

  Returns:
    The (source node weights, target node weights).
  """
  in_degrees = message_passing_ops.node_degrees(
      relation.target_idxs,
      relation.num_dst_nodes,
      relation.dst_values.dtype,
      relation.indices_are_sorted,
  )
  out_degrees = message_passing_ops.node_degrees(
      relation.source_idxs, relation.num_src_nodes, relation.src_values.dtype
  )
  return jax.lax.rsqrt(1.0 + out_degrees), jax.lax.rsqrt(1.0 + in_degrees)


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class HeterogeneousGraphConvolutionConfig(common.ArchitectureProvider):
  """Configuration for HeterogeneousGraphConvolution.

  Classic architectures are available as templates, e.g. `graphsage()`,
  `gcn()`, `gat()`, `gatv2()`, and `dot_product_attention()`.

  Usage example:

  ```python
  config = HeterogeneousGraphConvolutionConfig.gatv2(dims=64, num_heads=4)
  layer = config.make(schema)
  ```

  Data flow of one layer, for each target node set (names match the code):

  ```
  x_i                    Embedding of node i (feature `embedding_feature`).

  For each relation r = (edge set, direction) of `plan`:

    message_ij = message(x_j)            if not `message_consumes_target`
               = message([x_j ; x_i])    otherwise
                                         Args: `message`,
                                         `message_consumes_target`.

    aggregate_r(i) = sum_{j in N_r(i)} w_ij * message_ij
        w_ij = 1                                      SUM
             = 1 / in_deg(i)                          MEAN
             = 1 / sqrt((1+out_deg(j))(1+in_deg(i)))  SYMMETRIC (GCN)
             = softmax_j(logits_ij), per head         ATTENTION
                                         Args: `message_aggregation`,
                                         `attention` (computes the logits),
                                         `attention_dropout_rate`.

  messages(i) = sum_r aggregate_r(i)                  SUM
              = mean_r aggregate_r(i)                 MEAN
              = concat_r aggregate_r(i)               CONCAT (plan order)
                                         Arg: `relation_aggregation`.

  combined(i) = [x_i ; messages(i)]                   CONCAT
              = x_i + messages(i)                     SUM
                                         Arg: `combine`.

  y_i  = update(combined(i)) (+ x_i if `residual`)    Args: `update`,
                                                      `residual`.
  x'_i = post(y_i)                                    Arg: `post`.
  ```

  Attributes:
    plan: Message passing plan. A list of (edge_set_name, is_reversed) tuples.
      If None, messages are passed along all edges in both directions.
    embedding_feature: Name of the node feature to use for embeddings.
    dims: Dimension of the embeddings and hidden layers. Used to build the
      default values of `message`, `update`, and `post`.
    dropout_rate: Dropout rate. Used to build the default values of `update` and
      `post`.
    message_aggregation: Aggregation of the messages received by a node, within
      a relation. 'symmetric' is the GCN normalization and requires
      `message_consumes_target=False`. 'attention' weights the messages with a
      softmax of the logits computed by `attention`.
    relation_aggregation: Combination of the aggregated messages of the
      relations targeting a node set: 'sum', 'mean', or 'concat' (in plan
      order). 'concat' requires `combine='concat'`.
    message_consumes_target: If True, the message module sees `concat([source,
      target])`; if False, only `source` (GraphSAGE / GCN / GAT style).
    combine: Input of the update module: 'concat' for `concat([X, messages])`
      (GraphSAGE) or 'sum' for `X + messages` (GCN, requires matching dims).
    residual: Whether to add a residual connection around the update step.
    force_basic_implementation: If True, always compute messages with the
      generic per-edge implementation (no node-level projection, no edge
      sorting). Mostly useful in tests to validate the optimized paths.
    attention_dropout_rate: Dropout rate applied to the attention weights during
      training. Requires `message_aggregation='attention'`.
    message: Optional module to apply to edge features to generate messages.
      Defaults to a single-layer MLP.
    update: Optional module to apply to node embeddings after message passing,
      combining the old embedding and aggregated messages. Defaults to a
      single-layer MLP with layer norm and activation.
    post: Optional module applied after the update step, typically a
      transformer-like MLP. Defaults to a two-layer ResidualMLP.
    attention: Attention score (e.g. `GatAttentionConfig`). Required if and only
      if `message_aggregation='attention'`.
    message_pooling: Deprecated string alias of `message_aggregation`.
  """

  plan: list[tuple[str, bool]] | None = None
  embedding_feature: str = "embedding"
  dims: int = 128
  dropout_rate: float = 0.1
  message_aggregation: MessageAggregation = MessageAggregation.SUM
  relation_aggregation: RelationAggregation = RelationAggregation.SUM
  message_consumes_target: bool = True
  combine: Combine = Combine.CONCAT
  residual: bool = True
  force_basic_implementation: bool = False
  attention_dropout_rate: float = 0.0

  message: common.GenericLayer | None = layer_registry.field(default=None)
  update: common.GenericLayer | None = layer_registry.field(default=None)
  post: common.GenericLayer | None = layer_registry.field(default=None)
  attention: graph_attention.GraphAttentionConfig | None = layer_registry.field(
      default=None
  )

  message_pooling: str | None = None

  def __post_init__(self):
    if self.message_pooling is not None:
      self.message_aggregation = MessageAggregation(self.message_pooling)
      self.message_pooling = None

    self.validate()

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

  def validate(self) -> None:
    """Raises a ValueError if the configuration is inconsistent."""
    if (
        self.message_aggregation == MessageAggregation.SYMMETRIC
        and self.message_consumes_target
    ):
      raise ValueError(
          "message_aggregation='symmetric' requires"
          " message_consumes_target=False."
      )
    is_attention = self.message_aggregation == MessageAggregation.ATTENTION
    if is_attention != (self.attention is not None):
      raise ValueError(
          "`attention` must be set if and only if"
          " message_aggregation='attention'. Got"
          f" message_aggregation={self.message_aggregation.value!r} and"
          f" attention={self.attention!r}."
      )
    if self.attention_dropout_rate > 0.0 and not is_attention:
      raise ValueError(
          "attention_dropout_rate requires message_aggregation='attention'."
      )
    if (
        self.relation_aggregation == RelationAggregation.CONCAT
        and self.combine == Combine.SUM
    ):
      raise ValueError(
          "relation_aggregation='concat' requires combine='concat'."
      )

  @classmethod
  def graphsage(
      cls,
      dims: int = 128,
      activation: str = "relu",
      dropout_rate: float = 0.0,
      plan: Plan | None = None,
  ) -> "HeterogeneousGraphConvolutionConfig":
    """GraphSAGE template: `h_v' = act(W [h_v ; mean_{u in N(v)} h_u] + b)`.

    The messages are averaged within each relation, and then over the
    relations.
    """
    return cls(
        plan=plan,
        dims=dims,
        dropout_rate=dropout_rate,
        message_aggregation=MessageAggregation.MEAN,
        relation_aggregation=RelationAggregation.MEAN,
        message_consumes_target=False,
        combine=Combine.CONCAT,
        residual=False,
        message=standard.identity(),
        update=standard.GenericBlockConfig(
            "LAD" if dropout_rate > 0.0 else "LA",
            dims=dims,
            activation=activation,
            dropout_rate=dropout_rate,
        ),
        post=standard.identity(),
    )

  @classmethod
  def gcn(
      cls,
      dims: int = 128,
      activation: str = "relu",
      dropout_rate: float = 0.0,
      plan: Plan | None = None,
  ) -> "HeterogeneousGraphConvolutionConfig":
    """GCN template: `H' = act((D^-1/2 A D^-1/2 H + H) W + b)`.

    Each relation of the plan is normalized with its own degrees (`D = 1 +
    degree`) and the relations are summed. Unlike the original GCN, the
    self-connection `H` is not part of the normalized adjacency.
    """
    return cls(
        plan=plan,
        dims=dims,
        dropout_rate=dropout_rate,
        message_aggregation=MessageAggregation.SYMMETRIC,
        message_consumes_target=False,
        combine=Combine.SUM,
        residual=False,
        message=standard.identity(),
        update=standard.GenericBlockConfig(
            "LAD" if dropout_rate > 0.0 else "LA",
            dims=dims,
            activation=activation,
            dropout_rate=dropout_rate,
        ),
        post=standard.identity(),
    )

  @classmethod
  def gat(
      cls,
      dims: int = 128,
      num_heads: int = 8,
      activation: str = "elu",
      dropout_rate: float = 0.0,
      attention_dropout_rate: float = 0.0,
      plan: Plan | None = None,
  ) -> "HeterogeneousGraphConvolutionConfig":
    """GAT template ("Graph Attention Networks", Veličković et al., 2018).

    ```
    logits_ij = LeakyReLU(s_src^r(h_j) + s_dst^r(h_i))       (per head)
    m_i = sum_r sum_{j in N_r(i)} softmax_j(logits_ij) W_r h_j
    h_i' = act(W_u [h_i ; m_i] + b)
    ```

    The heads are concatenated. The update sees the node embedding (root
    weight), so nodes keep their own state even for relations without
    self-loops.

    Args:
      dims: Dimension of the messages and of the output embeddings. Must be
        divisible by `num_heads`.
      num_heads: Number of attention heads.
      activation: Activation of the update.
      dropout_rate: Dropout rate of the update.
      attention_dropout_rate: Dropout rate of the attention weights.
      plan: Message passing plan.

    Returns:
      The layer configuration.
    """
    return cls._attention_template(
        attention=graph_attention.GatAttentionConfig(num_heads=num_heads),
        dims=dims,
        activation=activation,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        plan=plan,
    )

  @classmethod
  def gatv2(
      cls,
      dims: int = 128,
      num_heads: int = 8,
      activation: str = "elu",
      dropout_rate: float = 0.0,
      attention_dropout_rate: float = 0.0,
      plan: Plan | None = None,
  ) -> "HeterogeneousGraphConvolutionConfig":
    """GATv2 template ("How Attentive are Graph Attention Networks?", 2021).

    ```
    logits_ij = a^r . LeakyReLU(W_src^r h_j + W_dst^r h_i)   (per head)
    m_i = sum_r sum_{j in N_r(i)} softmax_j(logits_ij) W_r h_j
    h_i' = act(W_u [h_i ; m_i] + b)
    ```

    Same as `gat()`, except for the attention logits.

    Args:
      dims: Dimension of the messages and of the output embeddings. Must be
        divisible by `num_heads`.
      num_heads: Number of attention heads.
      activation: Activation of the update.
      dropout_rate: Dropout rate of the update.
      attention_dropout_rate: Dropout rate of the attention weights.
      plan: Message passing plan.

    Returns:
      The layer configuration.
    """
    return cls._attention_template(
        attention=graph_attention.Gatv2AttentionConfig(
            num_heads=num_heads, dims=dims
        ),
        dims=dims,
        activation=activation,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        plan=plan,
    )

  @classmethod
  def _attention_template(
      cls,
      attention: graph_attention.GraphAttentionConfig,
      dims: int,
      activation: str,
      dropout_rate: float,
      attention_dropout_rate: float,
      plan: Plan | None,
  ) -> "HeterogeneousGraphConvolutionConfig":
    """Common configuration of the `gat()` and `gatv2()` templates."""
    return cls(
        plan=plan,
        dims=dims,
        dropout_rate=dropout_rate,
        message_aggregation=MessageAggregation.ATTENTION,
        relation_aggregation=RelationAggregation.SUM,
        message_consumes_target=False,
        combine=Combine.CONCAT,
        residual=False,
        attention_dropout_rate=attention_dropout_rate,
        message=standard.GenericBlockConfig("L", dims=dims),
        update=standard.GenericBlockConfig(
            "LAD" if dropout_rate > 0.0 else "LA",
            dims=dims,
            activation=activation,
            dropout_rate=dropout_rate,
        ),
        post=standard.identity(),
        attention=attention,
    )

  @classmethod
  def dot_product_attention(
      cls,
      dims: int = 128,
      num_heads: int = 4,
      dropout_rate: float = 0.1,
      plan: Plan | None = None,
  ) -> "HeterogeneousGraphConvolutionConfig":
    """Relation-aware dot-product attention template.

    ```
    logits_ij = <W_att^r Q_dst h_i, K_src h_j> / sqrt(head_dims)  (per head)
    m_i = sum_r sum_{j in N_r(i)} softmax_j(logits_ij) message_r([h_j ; h_i])
    h_i' = post(update([h_i ; m_i]) + h_i)
    ```

    This is the architecture of the former
    `HeterogeneousGraphAttentionNetwork`.

    Args:
      dims: Dimension of the embeddings, queries, and keys. Must be divisible by
        `num_heads`.
      num_heads: Number of attention heads.
      dropout_rate: Dropout rate of the update and post blocks.
      plan: Message passing plan.

    Returns:
      The layer configuration.
    """
    return cls(
        plan=plan,
        dims=dims,
        dropout_rate=dropout_rate,
        message_aggregation=MessageAggregation.ATTENTION,
        attention=graph_attention.DotProductAttentionConfig(
            num_heads=num_heads, dims=dims
        ),
    )

  def make(
      self, schema: schema_lib.GraphSchema, name: str | None = None
  ) -> "HeterogeneousGraphConvolution":
    return HeterogeneousGraphConvolution(config=self, schema=schema, name=name)

  def architecture(self) -> str:
    assert self.message is not None
    assert self.update is not None
    assert self.post is not None
    message_input = (
        "concat(source, target)" if self.message_consumes_target else "source"
    )
    if self.combine == Combine.CONCAT:
      combine_input = "concat(X, messages)"
    elif self.combine == Combine.SUM:
      combine_input = "X + messages"
    else:
      raise ValueError(f"Unsupported combine: {self.combine}")
    parts = []
    parts.append("X = ...")
    parts.append("MPNN:")
    parts.append(f"  Message({message_input}):")
    parts.append(textwrap.indent(self.message.architecture(), prefix="    "))
    if self.attention is not None:
      parts.append(f"  Aggregation({self.message_aggregation.value}):")
      parts.append(
          textwrap.indent(self.attention.architecture(), prefix="    ")
      )
    else:
      parts.append(f"  Aggregation({self.message_aggregation.value})")
    parts.append(f"  RelationAggregation({self.relation_aggregation.value})")
    parts.append(f"  Update({combine_input}):")
    parts.append(textwrap.indent(self.update.architecture(), prefix="    "))
    if self.residual:
      parts.append("Residual(X)")
    parts.append("# Post MPNN")
    parts.append(self.post.architecture())
    return "\n".join(parts)


class HeterogeneousGraphConvolution(nn.Module):
  """A single layer of heterogeneous Graph Neural Network message passing.

  This layer performs message passing on a heterogeneous graph. See
  `HeterogeneousGraphConvolutionConfig` for the data flow and the options.
  Classic architectures are available as templates of the config, e.g.
  `HeterogeneousGraphConvolutionConfig.graphsage(dims=64)`, `.gcn()`, `.gat()`,
  `.gatv2()`, and `.dot_product_attention()`.

  All node sets are assumed to have a feature specified by
  `config.embedding_feature` (defaulting to "embedding"), and these features
  must all have the same dimension. The `EmbedGraph` layer can be used to
  preprocess the graph to meet this requirement.

  Usage Example:

  ```python
  # Assume a graph and schema with node sets 'n1' and 'n2', and edge sets 'e1'
  # (n1->n1) and 'e2' (n1->n2).
  graph: JaxInMemoryGraph = ...
  schema: GraphSchema = ...

  # Embed all features into a single embedding vector per node.
  graph_embedder = EmbedGraphConfig().make(schema)
  embedded_graph = graph_embedder(graph)
  embedded_schema = graph_embedder.output_schema(schema)

  # Configure and apply message passing layers.
  message_passer_config = HeterogeneousGraphConvolutionConfig()
  for _ in range(num_layers):
    # Re-create the layer in each iteration to ensure separate weights.
    message_passer = message_passer_config.make(embedded_schema)
    embedded_graph = message_passer(embedded_graph, training=True)
  ```

  Message Passing Plan:

  By default, messages are passed along all edges in both forward and backward
  directions. This behavior can be customized using the `plan` argument in the
  `HeterogeneousGraphConvolutionConfig`. The plan is a list of tuples, where
  each tuple `(edge_set_name, is_reversed)` specifies an edge set and the
  direction of message flow.

  ```python
  # Example of a custom message passing plan:
  message_plan = [
      ("e1", False),  # Messages flow forward along 'e1' (n1 -> n1)
      ("e1", True),   # Messages flow backward along 'e1' (n1 <- n1)
      ("e2", False),  # Messages flow forward along 'e2' (n1 -> n2)
      # No ("e2", True), so no messages flow backward along 'e2' (n2 <- n1)
  ]
  config = HeterogeneousGraphConvolutionConfig(plan=message_plan)
  ```

  Implementation:

  The computation of each relation is chosen to minimize the cost: dense layers
  are applied on the smallest of (nodes, edges), "LAL" message blocks are
  decomposed into node-level projections, and large relations are sorted by
  target on GPU. `force_basic_implementation=True` disables these optimizations.

  Attributes:
    config: The configuration object for the message passing layer.
    schema: The graph schema. Only the source and target fields of the edge sets
      are essential for this layer.
    sorted_plan: The message passing plan, sorted and grouped by the destination
      node set. Each entry maps a destination node set name to a list of
      `(edge_set_name, source_node_set_name, is_reversed)` tuples.
  """

  config: HeterogeneousGraphConvolutionConfig
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
    """Computes the message passing.

    Args:
      graph: The graph structure.
      training: Is the model in training or serving/evaluation?

    Returns:
      The graph after message passing.
    """
    config = self.config
    features = {
        nodeset_name: nodeset.features[config.embedding_feature]
        for nodeset_name, nodeset in graph.node_sets.items()
    }
    relations_by_dst = {
        dst_nodeset: [
            self.relation(graph, features, dst_nodeset, *edge)
            for edge in self.sorted_plan[dst_nodeset]
        ]
        for dst_nodeset in sorted(self.sorted_plan)
    }

    # Attention logits of all the relations (computed together to share the
    # per node set projections).
    logits_by_relation: graph_attention.LogitsByRelation = {}
    if config.attention is not None:
      logits_by_relation = config.attention.make(name="attention")([
          relation
          for relations in relations_by_dst.values()
          for relation in relations
      ])

    # Initialize with all original node sets to avoid dropping any.
    new_node_sets = dict(graph.node_sets)
    for dst_nodeset, relations in relations_by_dst.items():
      aggregates = [
          self.aggregate(
              relation, logits_by_relation.get(relation.name), training
          )
          for relation in relations
      ]
      messages = self.combine_relations(aggregates)
      node_values = self.update(features[dst_nodeset], messages, training)
      new_node_sets[dst_nodeset] = jax_in_memory_graph.JaxInMemoryNodeSet(
          num_nodes=relations[0].num_dst_nodes,
          features={config.embedding_feature: node_values},
      )

    return jax_in_memory_graph.JaxInMemoryGraph(
        node_sets=new_node_sets,
        edge_sets=graph.edge_sets,
    )

  def relation(
      self,
      graph: jax_in_memory_graph.JaxInMemoryGraph,
      features: dict[str, jt.Float[jt.Array, "num_nodes dims"]],
      dst_nodeset: str,
      edgeset_name: str,
      src_nodeset: str,
      reverse: bool,
  ) -> Relation:
    """Gathers the edges of a relation, sorted by target if it pays off."""
    source_idxs = graph.edge_sets[edgeset_name].adjacency[0]
    target_idxs = graph.edge_sets[edgeset_name].adjacency[1]
    if reverse:
      source_idxs, target_idxs = target_idxs, source_idxs

    src_values = features[src_nodeset]
    num_src_nodes = src_values.shape[0]
    num_dst_nodes = graph.node_sets[dst_nodeset].num_nodes
    assert num_dst_nodes is not None
    num_edges = source_idxs.shape[0]

    indices_are_sorted = (
        not self.config.force_basic_implementation
        and message_passing_ops.should_sort_edges(
            num_src_nodes, num_dst_nodes, num_edges
        )
    )
    if indices_are_sorted:
      source_idxs, target_idxs = message_passing_ops.sort_edges_by_dst(
          source_idxs, target_idxs, num_src_nodes, num_dst_nodes
      )
    return Relation(
        name=f"{edgeset_name}_{'rev' if reverse else 'fwd'}",
        src_nodeset=src_nodeset,
        dst_nodeset=dst_nodeset,
        src_values=src_values,
        dst_values=features[dst_nodeset],
        source_idxs=source_idxs,
        target_idxs=target_idxs,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        num_edges=num_edges,
        indices_are_sorted=indices_are_sorted,
    )

  def aggregate(
      self,
      relation: Relation,
      logits: jt.Float[jt.Array, "num_edges num_heads"] | None,
      training: bool,
  ) -> jt.Float[jt.Array, "num_dst_nodes msg_dims"]:
    """Aggregates the messages of a relation: `sum_j w_ij * message_ij`.

    Args:
      relation: The relation.
      logits: Attention logits of the relation. Only used by the 'attention'
        aggregation.
      training: Is the model in training or serving/evaluation?

    Returns:
      The aggregated messages of each target node.
    """
    aggregation = self.config.message_aggregation
    if aggregation == MessageAggregation.SUM:
      return self.aggregate_with_fixed_weights(relation, None, None, training)
    elif aggregation == MessageAggregation.MEAN:
      return self.aggregate_with_fixed_weights(
          relation, None, mean_weights(relation), training
      )
    elif aggregation == MessageAggregation.SYMMETRIC:
      src_weights, dst_weights = symmetric_weights(relation)
      return self.aggregate_with_fixed_weights(
          relation, src_weights, dst_weights, training
      )
    elif aggregation == MessageAggregation.ATTENTION:
      assert logits is not None
      return self.aggregate_with_attention(relation, logits, training)
    else:
      raise ValueError(f"Unsupported message_aggregation: {aggregation}")

  def aggregate_with_fixed_weights(
      self,
      relation: Relation,
      src_weights: jt.Float[jt.Array, "num_src_nodes 1"] | None,
      dst_weights: jt.Float[jt.Array, "num_dst_nodes 1"] | None,
      training: bool,
  ) -> jt.Float[jt.Array, "num_dst_nodes msg_dims"]:
    """Aggregates messages with weights given by the graph structure.

    The weight of the edge (j -> i) is `src_weights[j] * dst_weights[i]`. A
    missing weight is 1.

    Args:
      relation: The relation.
      src_weights: Weight of each source node, or None.
      dst_weights: Weight of each target node, or None.
      training: Is the model in training or serving/evaluation?

    Returns:
      The aggregated messages of each target node.
    """
    linear_activation_linear = self.linear_activation_linear()
    if not self.config.message_consumes_target:
      aggregate = self.segment_sum(
          relation, self.source_messages(relation, src_weights, training)
      )
    elif linear_activation_linear is not None:
      assert src_weights is None
      aggregate = self.aggregate_lal_messages(
          relation, linear_activation_linear
      )
    else:
      assert src_weights is None
      aggregate = self.segment_sum(
          relation, self.basic_messages(relation, training)
      )
    if dst_weights is not None:
      aggregate = aggregate * dst_weights
    return aggregate

  def aggregate_with_attention(
      self,
      relation: Relation,
      logits: jt.Float[jt.Array, "num_edges num_heads"],
      training: bool,
  ) -> jt.Float[jt.Array, "num_dst_nodes msg_dims"]:
    """Aggregates messages with attention: `sum_j softmax_j(logits_ij) m_ij`.

    The messages are split in `num_heads` chunks, each weighted by the logits of
    one head.

    Args:
      relation: The relation.
      logits: Attention logit of each edge and head.
      training: Is the model in training or serving/evaluation?

    Returns:
      The aggregated messages of each target node.
    """
    messages = self.edge_messages(relation, training)  # [E, msg_dims]
    num_heads = logits.shape[-1]
    msg_dims = messages.shape[-1]
    if msg_dims % num_heads != 0:
      raise ValueError(
          f"The message dimension ({msg_dims}) of relation {relation.name!r}"
          f" must be divisible by the number of attention heads ({num_heads})."
      )

    weight_dropout = None
    if training and self.config.attention_dropout_rate > 0.0:
      weight_dropout = nn.Dropout(
          self.config.attention_dropout_rate,
          deterministic=False,
          name=f"attention_dropout_{relation.name}",
      )

    aggregate = message_passing_ops.segment_softmax_aggregate(
        logits,
        messages.reshape(relation.num_edges, num_heads, msg_dims // num_heads),
        relation.target_idxs,
        relation.num_dst_nodes,
        relation.indices_are_sorted,
        weight_dropout,
    )  # [N_dst, H, msg_dims / H]
    return aggregate.reshape(relation.num_dst_nodes, msg_dims)

  def linear_activation_linear(
      self,
  ) -> standard.LinearActivationLinear | None:
    """Returns the "LAL" decomposition of the message block, if applicable."""
    if self.config.force_basic_implementation or not isinstance(
        self.config.message, standard.GenericBlockConfig
    ):
      return None
    return self.config.message.as_linear_activation_linear()

  def edge_messages(
      self, relation: Relation, training: bool
  ) -> jt.Float[jt.Array, "num_edges msg_dims"]:
    """Computes the message of each edge."""
    linear_activation_linear = self.linear_activation_linear()
    if not self.config.message_consumes_target:
      return self.source_messages(relation, None, training)
    elif linear_activation_linear is not None:
      dense_src, dense_dst, dense_out = self.lal_dense_layers(
          relation, linear_activation_linear, use_out_bias=True
      )
      return dense_out(
          self.lal_edge_activations(
              relation, linear_activation_linear, dense_src, dense_dst
          )
      )
    else:
      return self.basic_messages(relation, training)

  def source_messages(
      self,
      relation: Relation,
      src_weights: jt.Float[jt.Array, "num_src_nodes 1"] | None,
      training: bool,
  ) -> jt.Float[jt.Array, "num_edges msg_dims"]:
    """Computes `message(x_j) * src_weights[j]` for each edge (j -> i)."""
    assert self.config.message is not None
    message_fn = self.config.message.make(name=f"msg_{relation.name}")
    if (
        self.config.force_basic_implementation
        or relation.num_src_nodes > relation.num_edges
    ):
      # Message on the edges.
      messages = message_fn(
          relation.src_values[relation.source_idxs], training=training
      )
      if src_weights is not None:
        messages = messages * src_weights[relation.source_idxs]
      return messages
    # Message on the nodes, then gather on the edges.
    node_messages = message_fn(relation.src_values, training=training)
    if src_weights is not None:
      node_messages = node_messages * src_weights
    return node_messages[relation.source_idxs]

  def basic_messages(
      self, relation: Relation, training: bool
  ) -> jt.Float[jt.Array, "num_edges msg_dims"]:
    """Computes `message([x_j ; x_i])` on each edge (j -> i) (slow fallback)."""
    edge_values = jnp.concatenate(
        [
            relation.src_values[relation.source_idxs],
            relation.dst_values[relation.target_idxs],
        ],
        axis=-1,
    )  # [E, 2 * dims]
    assert self.config.message is not None
    message_fn = self.config.message.make(name=f"msg_{relation.name}")
    return message_fn(edge_values, training=training)

  def lal_dense_layers(
      self,
      relation: Relation,
      linear_activation_linear: standard.LinearActivationLinear,
      use_out_bias: bool,
  ) -> tuple[nn.Dense, nn.Dense, nn.Dense]:
    """Dense layers of the "LAL" message block, applied separately.

    `Dense([x_j ; x_i])` is split into `dense_src(x_j) + dense_dst(x_i)`.

    Args:
      relation: The relation.
      linear_activation_linear: The "LAL" decomposition of the message block.
      use_out_bias: Whether the output dense layer has a bias. Without
        attention, the bias is not used since it would be scaled by the in
        degree of the nodes.

    Returns:
      The (src, dst, out) dense layers.
    """
    dense_src = nn.Dense(
        linear_activation_linear.hidden_dims,
        use_bias=False,
        name=f"msg_{relation.name}_src",
    )
    dense_dst = nn.Dense(
        linear_activation_linear.hidden_dims,
        use_bias=True,
        name=f"msg_{relation.name}_dst",
    )
    dense_out = nn.Dense(
        linear_activation_linear.output_dims,
        use_bias=use_out_bias,
        name=f"msg_{relation.name}_out",
    )
    return dense_src, dense_dst, dense_out

  def lal_edge_activations(
      self,
      relation: Relation,
      linear_activation_linear: standard.LinearActivationLinear,
      dense_src: nn.Dense,
      dense_dst: nn.Dense,
  ) -> jt.Float[jt.Array, "num_edges hidden_dims"]:
    """Computes `activation(dense_src(x_j) + dense_dst(x_i))` on each edge."""
    return linear_activation_linear.activation(
        message_passing_ops.project_on_edges(
            dense_src, relation.src_values, relation.source_idxs
        )
        + message_passing_ops.project_on_edges(
            dense_dst, relation.dst_values, relation.target_idxs
        )
    )

  def aggregate_lal_messages(
      self,
      relation: Relation,
      linear_activation_linear: standard.LinearActivationLinear,
  ) -> jt.Float[jt.Array, "num_dst_nodes msg_dims"]:
    """Sum of the "LAL" messages, computed with the cheapest strategy.

    `dense_out` is linear, so it is applied after the aggregation when there
    are fewer target nodes than edges.

    Args:
      relation: The relation.
      linear_activation_linear: The "LAL" decomposition of the message block.

    Returns:
      The aggregated messages of each target node.
    """
    dense_src, dense_dst, dense_out = self.lal_dense_layers(
        relation, linear_activation_linear, use_out_bias=False
    )
    num_src_nodes = relation.num_src_nodes
    num_dst_nodes = relation.num_dst_nodes
    num_edges = relation.num_edges

    # Project on nodes (N_src + 2 N_dst Dense rows) when cheaper than
    # projecting on edges (3 E Dense rows).
    if num_src_nodes + 2 * num_dst_nodes <= 3 * num_edges:
      # Node-level projections, then fused gather + activation + segment_sum
      # on the edges, then output projection on the nodes.
      aggregated_activations = message_passing_ops.make_fused_activation_reduce(
          linear_activation_linear.activation, relation.indices_are_sorted
      )(
          dense_src(relation.src_values),
          dense_dst(relation.dst_values),
          relation.source_idxs,
          relation.target_idxs,
      )  # [N_dst, hidden_dims]
      return dense_out(aggregated_activations)

    # Edge-level activation; each Dense is applied on the smallest of (nodes,
    # edges).
    edge_activations = self.lal_edge_activations(
        relation, linear_activation_linear, dense_src, dense_dst
    )  # [E, hidden_dims]
    if num_dst_nodes <= num_edges:
      # Group by target node, then output projection on the nodes.
      return dense_out(self.segment_sum(relation, edge_activations))
    # Output projection on the edges, then group by target node.
    return self.segment_sum(relation, dense_out(edge_activations))

  def segment_sum(
      self,
      relation: Relation,
      edge_values: jt.Float[jt.Array, "num_edges dims"],
  ) -> jt.Float[jt.Array, "num_dst_nodes dims"]:
    """Sums the values of the edges incoming to each target node."""
    return jax.ops.segment_sum(
        edge_values,
        relation.target_idxs,
        relation.num_dst_nodes,
        indices_are_sorted=relation.indices_are_sorted,
    )

  def combine_relations(
      self, aggregates: list[jt.Float[jt.Array, "num_dst_nodes msg_dims"]]
  ) -> jt.Float[jt.Array, "num_dst_nodes out_dims"]:
    """Combines the aggregated messages of the relations of a node set."""
    aggregation = self.config.relation_aggregation
    if aggregation == RelationAggregation.SUM:
      return sum(aggregates[1:], start=aggregates[0])
    elif aggregation == RelationAggregation.MEAN:
      return sum(aggregates[1:], start=aggregates[0]) * (1.0 / len(aggregates))
    elif aggregation == RelationAggregation.CONCAT:
      return jnp.concatenate(aggregates, axis=-1)
    else:
      raise ValueError(f"Unsupported relation_aggregation: {aggregation}")

  def update(
      self,
      dst_values: jt.Float[jt.Array, "num_dst_nodes dims"],
      messages: jt.Float[jt.Array, "num_dst_nodes msg_dims"],
      training: bool,
  ) -> jt.Float[jt.Array, "num_dst_nodes dims"]:
    """Computes the new node embeddings from the old ones and the messages."""
    config = self.config
    assert config.update is not None
    assert config.post is not None

    if config.combine == Combine.CONCAT:
      combined = jnp.concatenate([dst_values, messages], axis=1)
    elif config.combine == Combine.SUM:
      if messages.shape[-1] != dst_values.shape[-1]:
        raise ValueError(
            "combine='sum' requires the messages and the node embeddings to"
            f" have the same dimension. Got {messages.shape[-1]} and"
            f" {dst_values.shape[-1]}."
        )
      combined = messages + dst_values
    else:
      raise ValueError(f"Unsupported combine: {config.combine}")
    node_values = config.update.make()(combined, training=training)

    if config.residual:
      node_values = node_values + dst_values

    # Feed-forward
    return config.post.make()(node_values, training=training)
