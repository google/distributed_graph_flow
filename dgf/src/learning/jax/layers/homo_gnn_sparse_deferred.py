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
from typing import Optional

from dgf.src.data import schema as schema_lib
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
  """Config for Projector."""

  # TODO(deniscalin): look into making these optional and set to None, and
  # creating mlp_kwargs dict in make() to improve default values maintainance.
  num_layers: int
  hidden_dim: int
  use_bias: bool = True
  activation_fn: str = "tanh"

  name_prefix: str = "projector"

  def name(self) -> str:
    return "Projector"

  def make(  # pytype: disable=signature-mismatch  # pyrefly: ignore[bad-override]
      self, name: Optional[str] = None
  ) -> "Projector":
    return Projector(config=self, name=name)


class Projector(nn.Module):
  r"""Homogeneous Graph Feature Projector.

  Simple wrapper around the generic MLP layer for graph input/output.
  Purpose is typically to project features \in R^D_{in} -> R^D_{hidden}.
  """

  config: ProjectorConfig

  def setup(self):
    self.projector = dgf_layers.MLP(
        num_layers=self.config.num_layers,
        hidden_dim=self.config.hidden_dim,
        use_bias=self.config.use_bias,
        matrix_dtype=_jnp_dtype_from_string(self.config.matrix_precision),
        dropout_rate=self.config.dropout_rate,
        name_prefix=self.config.name_prefix,
    )

  def __call__(self, graph: GraphStruct, training: bool = False) -> GraphStruct:
    nodeset_name = self.config.nodeset_name or DEFAULT_NODESET_NAME
    input_feature = self.config.input_node_feature or DEFAULT_NODE_FEATURE_NAME

    x = get_node_features(graph, nodeset_name, input_feature)

    x = self.projector(x)

    # Output features are written to the input feature name
    # per `Projector` implementation.
    output_feature = input_feature

    graph = graph.update(nodes={nodeset_name: {output_feature: x}})

    return graph


@dataclasses.dataclass(frozen=True, kw_only=True)
class GCNConfig(JaxBaseConfig):
  """Config for GCN."""

  num_layers: int
  hidden_dim: int
  use_bias: bool = True
  activation_fn: str = "leaky_relu"
  enable_gnn_plus: bool = False

  name_prefix: str = "gcn"

  def name(self) -> str:
    return "GCN"

  def make(  # pytype: disable=signature-mismatch  # pyrefly: ignore[bad-override]
      self, name: Optional[str] = None
  ) -> "GCN":
    return GCN(config=self, name=name)


class GCN(nn.Module):
  """Homogeneous Graph Convolutional Network.

  Paper: https://arxiv.org/pdf/1609.02907.pdf.
  """

  config: GCNConfig

  def setup(self):
    self.activation = common.get_activation(self.config.activation_fn)
    matrix_dtype = _jnp_dtype_from_string(self.config.matrix_precision)

    self.update_fn = [
        nn.Dense(
            self.config.hidden_dim,
            use_bias=self.config.use_bias,
            dtype=matrix_dtype,
            name=f"{self.config.name_prefix}/update/layer_{i:02d}",
        )
        for i in range(self.config.num_layers)
    ]

    self.dropout = [
        nn.Dropout(
            rate=self.config.dropout_rate,
            name=f"{self.config.name_prefix}/dropout/layer_{i:02d}",
        )
        for i in range(self.config.num_layers)
    ]

    self.post_graph_conv = None
    if self.config.enable_gnn_plus:
      self.post_graph_conv = dgf_layers.GnnPlus(
          hidden_dim=self.config.hidden_dim,
          num_layers=self.config.num_layers,
          activation_fn=self.config.activation_fn,
          use_bias=self.config.use_bias,
          dropout_rate=self.config.dropout_rate,
          matrix_dtype=matrix_dtype,
          name_prefix=f"{self.config.name_prefix}/gnn_plus",
      )

  def __call__(self, graph: GraphStruct, training: bool = False) -> GraphStruct:
    x = get_node_features(
        graph,
        nodeset_name=self.config.nodeset_name,
        feature_name=self.config.input_node_feature,
    )

    adj = graph.adj(sdjnp.engine, self.config.edgeset_name)
    adj_symnorm = (adj + adj.transpose()).add_eye().normalize_symmetric()

    for layer_index in range(self.config.num_layers):
      hprev = x
      x = self.update_fn[layer_index](adj_symnorm @ x)

      if self.post_graph_conv is not None:
        x = self.post_graph_conv(hprev, x, layer_index, training)
      else:
        x = self.activation(x)
        x = self.dropout[layer_index](x, deterministic=not training)

    return graph.update(
        nodes={self.config.nodeset_name: {self.config.output_node_feature: x}}
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class MPNNConfig(JaxBaseConfig):
  """Config for MPNN."""

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

  def make(  # pytype: disable=signature-mismatch  # pyrefly: ignore[bad-override]
      self, name: Optional[str] = None
  ) -> "MPNN":
    return MPNN(config=self, name=name)


class MPNN(nn.Module):
  """Homogeneous Message-Passing Neural Network.

  Paper: https://arxiv.org/abs/1704.01212.
  """

  config: MPNNConfig

  def setup(self):

    if self.config.message_pooling not in ["sum", "mean"]:
      raise ValueError(
          f"Unsupported message_pooling spec: {self.config.message_pooling}"
      )

    self.activation = common.get_activation(self.config.activation_fn)
    matrix_dtype = _jnp_dtype_from_string(self.config.matrix_precision)

    self.message_fn = [
        nn.Dense(
            self.config.hidden_dim,
            use_bias=self.config.use_bias,
            dtype=matrix_dtype,
            name=f"{self.config.name_prefix}/message/layer_{i:02d}",
        )
        for i in range(self.config.num_layers)
    ]

    self.update_fn = [
        nn.Dense(
            self.config.hidden_dim,
            use_bias=self.config.use_bias,
            dtype=matrix_dtype,
            name=f"{self.config.name_prefix}/update/layer_{i:02d}",
        )
        for i in range(self.config.num_layers)
    ]

    self.post_graph_conv = None
    if self.config.enable_gnn_plus:
      self.post_graph_conv = dgf_layers.GnnPlus(
          hidden_dim=self.config.hidden_dim,
          num_layers=self.config.num_layers,
          activation_fn=self.config.activation_fn,
          use_bias=self.config.use_bias,
          dropout_rate=self.config.dropout_rate,
          matrix_dtype=matrix_dtype,
          name_prefix=f"{self.config.name_prefix}/gnn_plus",
      )

  def __call__(self, graph: GraphStruct, training: bool = False) -> GraphStruct:
    # Use JaxBaseConfig attributes if provided in MPNN config rather than hardcoded default values.
    # We maintain previous functional logic mapping over defaults where config doesn't have it explicitly bound
    nodeset_name = self.config.nodeset_name or DEFAULT_NODESET_NAME
    edgeset_name = self.config.edgeset_name or DEFAULT_EDGESET_NAME
    input_node_feature = (
        self.config.input_node_feature or DEFAULT_NODE_FEATURE_NAME
    )

    x = get_node_features(
        graph, nodeset_name=nodeset_name, feature_name=input_node_feature
    )

    # TODO(bmayer): It may be more re-usable to separate node features, state
    # and graph topology. Chat with team and make changes accordingly.
    graph = graph.update(nodes={nodeset_name: {DEFAULT_HIDDEN_STATE_NAME: x}})

    for layer_index in range(self.config.num_layers):
      h_prev = graph.nodes[nodeset_name][DEFAULT_HIDDEN_STATE_NAME]

      edge_feature_name = (
          DEFAULT_EDGE_FEATURE_NAME
          if DEFAULT_EDGE_FEATURE_NAME in graph.edges[edgeset_name][1]
          else None
      )

      edge_features = map_nodes_to_incident_edges(
          sdjnp.engine,
          graph,
          edgeset_name,
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
      src_incidence = graph.incidence(sdjnp.engine, edgeset_name, 0)
      dst_incidence = graph.incidence(sdjnp.engine, edgeset_name, 1)

      src_messages = incidence_pooling(
          self.config.message_pooling, src_incidence, messages
      )
      dst_messages = incidence_pooling(
          self.config.message_pooling, dst_incidence, messages
      )

      messages = jnp.concatenate([src_messages, dst_messages], axis=-1)
      h_next = self.update_fn[layer_index](
          jnp.concatenate([h_prev, messages], axis=-1)
      )

      if self.config.enable_gnn_plus:
        h_next = self.post_graph_conv(h_prev, h_next, layer_index, training)  # pyrefly: ignore[not-callable]
      else:
        h_next = self.activation(h_next)

      graph = graph.update(
          nodes={nodeset_name: {DEFAULT_HIDDEN_STATE_NAME: h_next}}
      )

    return graph


@dataclasses.dataclass(frozen=True, kw_only=True)
class GINConfig(JaxBaseConfig):
  """Config for GIN."""

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

  def make(  # pytype: disable=signature-mismatch  # pyrefly: ignore[bad-override]
      self, name: Optional[str] = None
  ) -> "GIN":
    return GIN(config=self, name=name)


class GIN(nn.Module):
  """Homogeneous Graph Isomorphism Network.

  Paper: https://arxiv.org/pdf/1810.00826.pdf.
  """

  config: GINConfig

  def setup(self):
    self.activation = common.get_activation(self.config.activation_fn)
    matrix_dtype = _jnp_dtype_from_string(self.config.matrix_precision)

    self.update_fn = [
        dgf_layers.MLP(
            num_layers=self.config.num_mlp_layers,
            hidden_dim=self.config.hidden_dim,
            activation="relu",
            use_bias=True,
            norm_type=None,
            name=f"{self.config.name_prefix}/update/layer_{i:02d}",
        )
        for i in range(self.config.num_layers)
    ]

    self.dropout = [
        nn.Dropout(
            rate=self.config.dropout_rate,
            name=f"{self.config.name_prefix}/dropout/layer_{i:02d}",
        )
        for i in range(self.config.num_layers)
    ]

    self.post_graph_conv = None
    if self.config.enable_gnn_plus:
      self.post_graph_conv = dgf_layers.GnnPlus(
          hidden_dim=self.config.hidden_dim,
          num_layers=self.config.num_layers,
          activation_fn=self.config.activation_fn,
          use_bias=self.config.use_bias,
          dropout_rate=self.config.dropout_rate,
          matrix_dtype=matrix_dtype,
          name_prefix=f"{self.config.name_prefix}/gnn_plus",
      )

  def __call__(self, graph: GraphStruct, training: bool = False) -> GraphStruct:
    x = get_node_features(
        graph,
        nodeset_name=self.config.nodeset_name,
        feature_name=self.config.input_node_feature,
    )

    adj = graph.adj(sdjnp.engine, self.config.edgeset_name)
    adj = (adj + adj.transpose()).add_eye(1 + self.config.epsilon)

    for layer_index in range(self.config.num_layers):
      hprev = x
      x = self.update_fn[layer_index](adj @ x)

      if self.post_graph_conv is not None:
        x = self.post_graph_conv(hprev, x, layer_index, training)
      else:
        x = self.activation(x)
        x = self.dropout[layer_index](x, deterministic=not training)

    return graph.update(
        nodes={self.config.nodeset_name: {self.config.output_node_feature: x}}
    )


class ConditionalGIN(GIN):
  """Homogeneous Conditional GIN with a labeling trick.

  Paper: https://arxiv.org/abs/2106.06935.
  """

  def setup(self):
    super().setup()
    self.labeling_feature_projection = dgf_layers.MLP(
        num_layers=2,
        hidden_dim=self.config.hidden_dim,
        activation="relu",
        dropout_rate=0.0,
        name_prefix=f"{self.config.name_prefix}/labeling_projection",
    )

  def __call__(self, graph: GraphStruct, idx: int, training: bool = False) -> GraphStruct:  # pytype: disable=signature-mismatch  # overriding-return-type-checks
    x = get_node_features(
        graph,
        nodeset_name=self.config.nodeset_name,
        feature_name=self.config.input_node_feature,
    )

    # TODO(mgalkin): We can abstract this away into a separate module and use
    # in any GNN module
    label_features = labeling_trick_features(
        num_nodes=x.shape[0],
        idx=idx,
        hidden_dim=self.config.hidden_dim,
    )
    x = jnp.concatenate([x, label_features], axis=-1)
    # Project d+d features to d features for a safe residual stream.
    x = self.labeling_feature_projection(x)

    # Standard GIN routine.
    adj = graph.adj(sdjnp.engine, self.config.edgeset_name)
    adj = (adj + adj.transpose()).add_eye(1 + self.config.epsilon)

    for layer_index in range(self.config.num_layers):
      hprev = x
      x = self.update_fn[layer_index](adj @ x)

      if self.post_graph_conv is not None:
        x = self.post_graph_conv(hprev, x, layer_index, training)
      else:
        x = self.activation(x)
        x = self.dropout[layer_index](x, deterministic=not training)

    return graph.update(
        nodes={self.config.nodeset_name: {self.config.output_node_feature: x}}
    )
