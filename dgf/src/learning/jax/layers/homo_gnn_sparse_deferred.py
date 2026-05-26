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

"""GNN implementations for homogeneous graphs using sparse deferred.

For simplicity and without loss of generality, at the modeling layer, we expect
the input graphs to be homogeneous with a nodeset name named
`nodes` and an edgeset named `edges` and the schema should be a simple {'edges',
('nodes', 'nodes)}. Node features named `initial_state` are
required and edge features are optional but if provided are named
`initial_state`. Again, for simplicity, we assume message passing is always
bidirectional.

Generality is preserved because the user can encode type information in the
`initial_state` representation. This balances simplicity of the GNN modeling
code efficency and expressivity. GraphFlow provides common learnable modules for
encoding type information.
TODO(bmayer): Reference type encoding implementations and examples.
"""

import dataclasses
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import mlp as dgf_layers
from flax import linen as nn
import jax
import jax.numpy as jnp
import sparse_deferred as sd
import sparse_deferred.jax as sdjnp
from sparse_deferred.nn.edges import concat_features  # pylint: disable=g-importing-member
from sparse_deferred.nn.edges import map_nodes_to_incident_edges  # pylint: disable=g-importing-member
from sparse_deferred.structs import graph_struct as graph_struct_lib

_jnp_dtype_from_string = common.jnp_dtype_from_string
JaxBaseConfig = common.JaxBaseConfig

GraphStruct = graph_struct_lib.GraphStruct


# By default we assume a single node set named `nodes` with node features named
# # `initial_state`. Similarly we assume a single edge set named `edges` with
# an optional `initial_state`.
DEFAULT_NODESET_NAME = common.DEFAULT_NODESET_NAME
DEFAULT_NODE_FEATURE_NAME = common.DEFAULT_NODE_FEATURE_NAME
DEFAULT_EDGESET_NAME = common.DEFAULT_EDGESET_NAME
DEFAULT_EDGE_FEATURE_NAME = common.DEFAULT_EDGE_FEATURE_NAME

# TODO(bmayer): "#hidden_state"?
DEFAULT_HIDDEN_STATE_NAME = common.DEFAULT_HIDDEN_STATE_NAME


def get_node_features(
    graph: GraphStruct,
    nodeset_name: str = DEFAULT_NODESET_NAME,
    feature_name: str = DEFAULT_NODE_FEATURE_NAME,
):
  """Get node features by name.

  Args:
    graph: The input sd GraphStruct graph.
    nodeset_name: Target nodeset name.
    feature_name: Target feature name.

  Returns:
    Tensor of node features

  Raises:
    KeyError: If the nodeset_name or feature_name is not found in the graph.
  """
  return graph.nodes[nodeset_name][feature_name]


def get_node_hidden_state(
    graph: GraphStruct,
    nodeset_name: str = DEFAULT_NODESET_NAME,
    node_feature_name: str = DEFAULT_HIDDEN_STATE_NAME,
) -> jax.Array:
  """Get a hidden state by name.

  Args:
    graph: The input sd GraphStruct graph.
    nodeset_name: Target nodeset name.
    node_feature_name: Target feature name.

  Returns:
    The hidden state tensor.

  Raises KeyError: If the `nodeset_name` or `node_feature_name` are not defined
    on the graph.
  """
  return graph.nodes[nodeset_name][node_feature_name]


def labeling_trick_features(
    num_nodes: int,
    idx: int,
    hidden_dim: int,
) -> jax.Array:
  """The labeling trick for node features a-la NBFNet: https://arxiv.org/abs/2106.06935.

  This function returns all-zero initialization for nodes.
  Except for the one labeled with `idx` which will have the all-ones init.

  Example output:
  labeling_trick_features(num_nodes=4, idx=1, hidden_dim=4)
  [0 0 0 0]
  [1 1 1 1]
  [0 0 0 0]
  [0 0 0 0]

  Args:
    num_nodes: The number of nodes in the graph, number of rows in the output.
    idx: The index of the node to be labeled with ones.
    hidden_dim: The dimension of the output.

  Returns:
    The node features for the graph.
  """
  node_features = jnp.zeros((num_nodes, hidden_dim))
  node_features = node_features.at[idx].set(jnp.ones(hidden_dim))
  return node_features


def incidence_pooling(
    message_pooling_type: str,
    incidence: sd.SparseMatrix,
    edge_features: sd.Tensor,
) -> jax.Array:
  """Implementation of a gather using incidence and edge features.

  Args:
    message_pooling_type: ["sum", "mean"]
    incidence: [E, N] sparse matrix.
    edge_features: An [E, D] matrix of edge features.

  Returns:
    An [N, D] jax array where edge features are pooled over every incident node.
  """
  if message_pooling_type == "sum":
    sum_outgoing_messages = incidence.T @ edge_features
    node_features = sum_outgoing_messages
  elif message_pooling_type == "mean":
    mean_outgoing_messages = incidence.T.normalize_right() @ edge_features
    node_features = mean_outgoing_messages
  else:
    raise ValueError(f"Unknown message_pooling_type: {message_pooling_type}")
  return node_features


@dataclasses.dataclass(frozen=True, kw_only=True)
class ProjectorConfig(JaxBaseConfig):
  """Makeable Projector config class with sensible defaults."""

  # TODO(deniscalin): look into making these optional and set to None, and
  # creating mlp_kwargs dict in make() to improve default values maintainance.
  num_layers: int
  hidden_dim: int
  use_bias: bool = True
  activation_fn: str = "tanh"

  name_prefix: str = "projector"

  def name(self) -> str:
    return "Projector"

  def make(self) -> "Projector":
    return Projector(
        num_layers=self.num_layers,
        hidden_dim=self.hidden_dim,
        use_bias=self.use_bias,
        activation=self.activation_fn,
        # TODO(deniscalin): consider creating a `common_kwargs` helper function
        # to reuse in multiple configs.
        dropout_rate=self.dropout_rate,
        matrix_dtype=_jnp_dtype_from_string(self.matrix_precision),
        norm_dtype=_jnp_dtype_from_string(self.pointwise_norm_precision),
        node_set_name=self.nodeset_name,
        input_node_feature=self.input_node_feature,
        output_node_feature=self.input_node_feature,
    )


class Projector(nn.Module):
  r"""Simple wrapper around the generic MLP layer for graph input/output.

  Purpose is typically to project features \in R^D_{in} -> R^D_{hidden}.
  """

  num_layers: int
  hidden_dim: int
  activation: str = "tanh"
  use_bias: bool = True
  matrix_dtype: jnp.dtype = common.DEFAULT_MATRIX_PRECISION
  norm_dtype: jnp.dtype = common.DEFAULT_POINTWISE_NORM_PRECISION
  dropout_rate: float = common.DEFAULT_DROPOUT_RATE
  name_prefix: str = "projector"

  # TODO(deniscalin): standardize to `nodeset_name` as in other models,
  # vs `node_set_name` used here.
  node_set_name: str = DEFAULT_NODESET_NAME
  input_node_feature: str = DEFAULT_NODE_FEATURE_NAME
  output_node_feature: str = DEFAULT_NODE_FEATURE_NAME

  def setup(self):
    self.projector = dgf_layers.MLP(
        num_layers=self.num_layers,
        hidden_dim=self.hidden_dim,
        use_bias=self.use_bias,
        matrix_dtype=self.matrix_dtype,
        dropout_rate=self.dropout_rate,
        name_prefix=self.name_prefix,
    )

  def __call__(self, graph: GraphStruct, training: bool = False) -> GraphStruct:
    x = get_node_features(graph, self.node_set_name, self.input_node_feature)

    x = self.projector(x)

    graph = graph.update(
        nodes={self.node_set_name: {self.output_node_feature: x}}
    )

    return graph


@dataclasses.dataclass(frozen=True, kw_only=True)
class GCNConfig(JaxBaseConfig):
  """Makeable GCN config class with sensible defaults."""

  num_layers: int
  hidden_dim: int
  use_bias: bool = True
  activation_fn: str = "leaky_relu"
  enable_gnn_plus: bool = False

  name_prefix: str = "gcn"

  def name(self) -> str:
    return "GCN"

  def make(self) -> "GCN":
    return GCN(
        num_layers=self.num_layers,
        hidden_dim=self.hidden_dim,
        use_bias=self.use_bias,
        activation_fn=self.activation_fn,
        dropout_rate=self.dropout_rate,
        matrix_dtype=_jnp_dtype_from_string(self.matrix_precision),
        norm_dtype=_jnp_dtype_from_string(self.pointwise_norm_precision),
        nodeset_name=self.nodeset_name,
        edgeset_name=self.edgeset_name,
        input_node_feature=self.input_node_feature,
        output_node_feature=self.output_node_feature,
        enable_gnn_plus=self.enable_gnn_plus,
    )


class GCN(nn.Module):
  """Graph convolutional network: https://arxiv.org/pdf/1609.02907.pdf."""

  num_layers: int
  hidden_dim: int
  use_bias: bool = True
  activation_fn: str = "leaky_relu"
  # Technically, dropout is not used in the original paper.
  dropout_rate: float = common.DEFAULT_DROPOUT_RATE
  matrix_dtype: jnp.dtype = common.DEFAULT_MATRIX_PRECISION
  norm_dtype: jnp.dtype = common.DEFAULT_POINTWISE_NORM_PRECISION
  name_prefix: str = "gcn"

  nodeset_name: str = DEFAULT_NODESET_NAME
  edgeset_name: str = DEFAULT_EDGESET_NAME
  input_node_feature: str = DEFAULT_NODE_FEATURE_NAME
  output_node_feature: str = DEFAULT_HIDDEN_STATE_NAME

  enable_gnn_plus: bool = False

  def setup(self):
    self.activation = common.get_activation(self.activation_fn)

    self.update_fn = [
        nn.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            dtype=self.matrix_dtype,
            name=f"{self.name_prefix}/update/layer_{i:02d}",
        )
        for i in range(self.num_layers)
    ]

    self.dropout = [
        nn.Dropout(
            rate=self.dropout_rate,
            name=f"{self.name_prefix}/dropout/layer_{i:02d}",
        )
        for i in range(self.num_layers)
    ]

    self.post_graph_conv = None
    if self.enable_gnn_plus:
      self.post_graph_conv = dgf_layers.GnnPlus(
          hidden_dim=self.hidden_dim,
          num_layers=self.num_layers,
          activation_fn=self.activation_fn,
          use_bias=self.use_bias,
          dropout_rate=self.dropout_rate,
          matrix_dtype=self.matrix_dtype,
          name_prefix=f"{self.name_prefix}/gnn_plus",
      )

  def __call__(self, graph: GraphStruct, training: bool = False) -> GraphStruct:
    x = get_node_features(
        graph,
        nodeset_name=self.nodeset_name,
        feature_name=self.input_node_feature,
    )

    adj = graph.adj(sdjnp.engine, self.edgeset_name)
    adj_symnorm = (adj + adj.transpose()).add_eye().normalize_symmetric()

    for layer_index in range(self.num_layers):
      hprev = x
      x = self.update_fn[layer_index](adj_symnorm @ x)

      if self.post_graph_conv is not None:
        x = self.post_graph_conv(hprev, x, layer_index, training)
      else:
        x = self.activation(x)
        x = self.dropout[layer_index](x, deterministic=not training)

    return graph.update(
        nodes={self.nodeset_name: {self.output_node_feature: x}}
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class MPNNConfig(JaxBaseConfig):
  """Makeable MPNN config class with sensible defaults."""

  # TODO(bmayer): These can probably go on the base config?
  num_layers: int
  hidden_dim: int
  use_bias: bool = True
  activation_fn: str = "leaky_relu"
  enable_gnn_plus: bool = False

  message_pooling: str = "sum"
  name_prefix: str = "mpnn"

  def name(self) -> str:
    return "MPNN"

  def make(self) -> "MPNN":
    return MPNN(
        num_layers=self.num_layers,
        hidden_dim=self.hidden_dim,
        use_bias=self.use_bias,
        activation_fn=self.activation_fn,
        message_pooling=self.message_pooling,
        dropout_rate=self.dropout_rate,
        matrix_dtype=_jnp_dtype_from_string(self.matrix_precision),
        norm_dtype=_jnp_dtype_from_string(self.pointwise_norm_precision),
        enable_gnn_plus=self.enable_gnn_plus,
    )


class MPNN(nn.Module):
  """Message-Passing Neural Network: https://arxiv.org/abs/1704.01212."""

  num_layers: int
  hidden_dim: int
  use_bias: bool = True
  activation_fn: str = "leaky_relu"
  message_pooling: str = "sum"
  dropout_rate: float = common.DEFAULT_DROPOUT_RATE
  matrix_dtype: jnp.dtype = common.DEFAULT_MATRIX_PRECISION
  norm_dtype: jnp.dtype = common.DEFAULT_POINTWISE_NORM_PRECISION
  name_prefix: str = "mpnn"

  # TODO(bmayer): There's probably a better way to inject this. Either inject
  # it funtionally or with a nested config object (when we get to configs).
  enable_gnn_plus: bool = False

  def setup(self):

    if self.message_pooling not in ["sum", "mean"]:
      raise ValueError(
          f"Unsupported message_pooling spec: {self.message_pooling}"
      )

    self.activation = common.get_activation(self.activation_fn)

    self.message_fn = [
        nn.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            dtype=self.matrix_dtype,
            name=f"{self.name_prefix}/message/layer_{i:02d}",
        )
        for i in range(self.num_layers)
    ]

    self.update_fn = [
        nn.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            dtype=self.matrix_dtype,
            name=f"{self.name_prefix}/update/layer_{i:02d}",
        )
        for i in range(self.num_layers)
    ]

    self.post_graph_conv = None
    if self.enable_gnn_plus:
      self.post_graph_conv = dgf_layers.GnnPlus(
          hidden_dim=self.hidden_dim,
          num_layers=self.num_layers,
          activation_fn=self.activation_fn,
          use_bias=self.use_bias,
          dropout_rate=self.dropout_rate,
          matrix_dtype=self.matrix_dtype,
          name_prefix=f"{self.name_prefix}/gnn_plus",
      )

  def __call__(self, graph: GraphStruct, training: bool = False) -> GraphStruct:
    x = get_node_features(graph)

    # TODO(bmayer): It may be more re-usable to separate node features, state
    # and graph topology. Chat with team and make changes accordingly.
    graph = graph.update(
        nodes={DEFAULT_NODESET_NAME: {DEFAULT_HIDDEN_STATE_NAME: x}}
    )

    for layer_index in range(self.num_layers):
      h_prev = graph.nodes[DEFAULT_NODESET_NAME][DEFAULT_HIDDEN_STATE_NAME]

      edge_feature_name = (
          DEFAULT_EDGE_FEATURE_NAME
          if DEFAULT_EDGE_FEATURE_NAME in graph.edges[DEFAULT_EDGESET_NAME][1]
          else None
      )

      edge_features = map_nodes_to_incident_edges(
          sdjnp.engine,
          graph,
          DEFAULT_EDGESET_NAME,
          node_feature_names=[
              DEFAULT_HIDDEN_STATE_NAME,
              DEFAULT_HIDDEN_STATE_NAME,
          ],
          edge_feature_name=edge_feature_name,
          edge_layer=concat_features,
      )

      # Apply message function to messages on incident edges, will then
      # aggregate and update.
      messages = self.message_fn[layer_index](edge_features)
      src_incidence = graph.incidence(sdjnp.engine, DEFAULT_EDGESET_NAME, 0)
      dst_incidence = graph.incidence(sdjnp.engine, DEFAULT_EDGESET_NAME, 1)

      src_messages = incidence_pooling(
          self.message_pooling, src_incidence, messages
      )
      dst_messages = incidence_pooling(
          self.message_pooling, dst_incidence, messages
      )

      messages = jnp.concatenate([src_messages, dst_messages], axis=-1)
      h_next = self.update_fn[layer_index](
          jnp.concatenate([h_prev, messages], axis=-1)
      )

      if self.enable_gnn_plus:
        h_next = self.post_graph_conv(h_prev, h_next, layer_index, training)
      else:
        h_next = self.activation(h_next)

      graph = graph.update(
          nodes={DEFAULT_NODESET_NAME: {DEFAULT_HIDDEN_STATE_NAME: h_next}}
      )

    return graph


@dataclasses.dataclass(frozen=True, kw_only=True)
class GINConfig(JaxBaseConfig):
  """Makeable GIN config class with sensible defaults."""

  num_layers: int
  hidden_dim: int
  use_bias: bool = True
  activation_fn: str = "relu"
  enable_gnn_plus: bool = False

  epsilon: float = 0.1
  num_mlp_layers: int = 2
  name_prefix: str = "gin"

  def name(self) -> str:
    return "GIN"

  def make(self) -> "GIN":
    return GIN(
        num_layers=self.num_layers,
        hidden_dim=self.hidden_dim,
        use_bias=self.use_bias,
        activation_fn=self.activation_fn,
        epsilon=self.epsilon,
        num_mlp_layers=self.num_mlp_layers,
        dropout_rate=self.dropout_rate,
        matrix_dtype=_jnp_dtype_from_string(self.matrix_precision),
        norm_dtype=_jnp_dtype_from_string(self.pointwise_norm_precision),
        nodeset_name=self.nodeset_name,
        edgeset_name=self.edgeset_name,
        input_node_feature=self.input_node_feature,
        output_node_feature=self.output_node_feature,
        enable_gnn_plus=self.enable_gnn_plus,
    )


class GIN(nn.Module):
  """Graph isomorphism network: https://arxiv.org/pdf/1810.00826.pdf."""

  num_layers: int
  hidden_dim: int
  use_bias: bool = True
  activation_fn: str = "relu"
  epsilon: float = 0.1
  num_mlp_layers: int = 2
  # Technically, dropout is not used in the original paper.
  dropout_rate: float = common.DEFAULT_DROPOUT_RATE
  matrix_dtype: jnp.dtype = common.DEFAULT_MATRIX_PRECISION
  norm_dtype: jnp.dtype = common.DEFAULT_POINTWISE_NORM_PRECISION
  name_prefix: str = "gin"
  nodeset_name: str = DEFAULT_NODESET_NAME
  edgeset_name: str = DEFAULT_EDGESET_NAME
  input_node_feature: str = DEFAULT_NODE_FEATURE_NAME
  output_node_feature: str = DEFAULT_HIDDEN_STATE_NAME

  enable_gnn_plus: bool = False

  def setup(self):
    self.activation = common.get_activation(self.activation_fn)

    self.update_fn = [
        dgf_layers.MLP(
            num_layers=self.num_mlp_layers,
            hidden_dim=self.hidden_dim,
            activation="relu",
            use_bias=True,
            norm_type=None,
            name=f"{self.name_prefix}/update/layer_{i:02d}",
        )
        for i in range(self.num_layers)
    ]

    self.dropout = [
        nn.Dropout(
            rate=self.dropout_rate,
            name=f"{self.name_prefix}/dropout/layer_{i:02d}",
        )
        for i in range(self.num_layers)
    ]

    self.post_graph_conv = None
    if self.enable_gnn_plus:
      self.post_graph_conv = dgf_layers.GnnPlus(
          hidden_dim=self.hidden_dim,
          num_layers=self.num_layers,
          activation_fn=self.activation_fn,
          use_bias=self.use_bias,
          dropout_rate=self.dropout_rate,
          matrix_dtype=self.matrix_dtype,
          name_prefix=f"{self.name_prefix}/gnn_plus",
      )

  def __call__(self, graph: GraphStruct, training: bool = False) -> GraphStruct:
    x = get_node_features(
        graph,
        nodeset_name=self.nodeset_name,
        feature_name=self.input_node_feature,
    )

    adj = graph.adj(sdjnp.engine, self.edgeset_name)
    adj = (adj + adj.transpose()).add_eye(1 + self.epsilon)

    for layer_index in range(self.num_layers):
      hprev = x
      x = self.update_fn[layer_index](adj @ x)

      if self.post_graph_conv is not None:
        x = self.post_graph_conv(hprev, x, layer_index, training)
      else:
        x = self.activation(x)
        x = self.dropout[layer_index](x, deterministic=not training)

    return graph.update(
        nodes={self.nodeset_name: {self.output_node_feature: x}}
    )


class ConditionalGIN(GIN):
  """Conditional GIN with a labeling trick: https://arxiv.org/abs/2106.06935."""

  def setup(self):
    super().setup()
    self.labeling_feature_projection = dgf_layers.MLP(
        num_layers=2,
        hidden_dim=self.hidden_dim,
        activation="relu",
        dropout_rate=0.0,
        name_prefix=f"{self.name_prefix}/labeling_projection",
    )

  def __call__(self, graph: GraphStruct, idx: int, training: bool = False) -> GraphStruct:  # pytype: disable=signature-mismatch  # overriding-return-type-checks
    x = get_node_features(
        graph,
        nodeset_name=self.nodeset_name,
        feature_name=self.input_node_feature,
    )

    # TODO(mgalkin): We can abstract this away into a separate module and use
    # in any GNN module
    label_features = labeling_trick_features(
        num_nodes=x.shape[0],
        idx=idx,
        hidden_dim=self.hidden_dim,
    )
    x = jnp.concatenate([x, label_features], axis=-1)
    # Project d+d features to d features for a safe residual stream.
    x = self.labeling_feature_projection(x)

    # Standard GIN routine.
    adj = graph.adj(sdjnp.engine, self.edgeset_name)
    adj = (adj + adj.transpose()).add_eye(1 + self.epsilon)

    for layer_index in range(self.num_layers):
      hprev = x
      x = self.update_fn[layer_index](adj @ x)

      if self.post_graph_conv is not None:
        x = self.post_graph_conv(hprev, x, layer_index, training)
      else:
        x = self.activation(x)
        x = self.dropout[layer_index](x, deterministic=not training)

    return graph.update(
        nodes={self.nodeset_name: {self.output_node_feature: x}}
    )
