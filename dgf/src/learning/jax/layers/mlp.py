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

"""Common layers implemented in jax/flax that are agnostic to GNN backend."""

from typing import Optional
from absl import logging
import chex
from dgf.src.learning.jax import common
import flax.linen as nn
import jax.numpy as jnp
import jaxtyping as jt

JaxBaseConfig = common.JaxBaseConfig


class MLP(nn.Module):
  """A generic MLP followed by a linear layer.

  Layer norm is by default applied to the output but can be turned off. If you
  set `norm = None`, you probably also want to set `use_bias = True`.

  NOTES:
   * If you use LayerNorm, typically the dense layer doesn't need bias too and
     you can set `use_bias = False` to save parameters. Since we default to
     using LayerNorm, `use_bias` defaults to False.
   * If `num_layers = 1` this MLP reduces to a linear layer (e.g., no dropout
     or activation). We recommend not applying a normalization layer after a
     linear layer and you probably want to set `use_bias=True`.
   * Activation and dropout are not applied to the last layer to support
     use-cases such as regression. The caller can stack additional layers
     manually as needed.

  Attributes:
    num_layers: The number of dense layers
    hidden_dim: The dimensionality of the hidden dense layers
    output_dim: Optional dimensionality of the last dense layer. If not
      provided, will be equal to `hidden_dim`.
    activation: String name of the flax.linen activation function
    use_bias: Boolean value to use bias in linear layers.
    dropout_rage: The dropout rate applied to each layer during training.
    matrix_dtype: The precision to use for floating point matrix weights and
      ops.
    norm_dtype: The precision to use for normalization computations.
    name_prefix: The custom name prefix for the layer.
  """

  num_layers: int
  hidden_dim: int
  output_dim: Optional[int] = None
  activation: str = "tanh"
  norm_type: Optional[str] = "layer"
  use_bias: bool = False
  dropout_rate: float = common.DEFAULT_DROPOUT_RATE
  matrix_dtype: jnp.dtype = common.DEFAULT_MATRIX_PRECISION
  norm_dtype: jnp.dtype = common.DEFAULT_POINTWISE_NORM_PRECISION
  name_prefix: str = "MLP"

  def setup(self):
    if self.num_layers <= 0:
      raise ValueError(
          f"Must provide strictly positive number of layers {self.num_layers=}."
      )

    if self.norm_type not in [None, "layer"]:
      raise ValueError(f"Unsupported norm type: {self.norm_type}")

    if self.norm_type is None and not self.use_bias:
      logging.warning(
          "LayerNorm is turned off and `use_bias=False`, you may want to try"
          " `use_bias=True`."
      )

    self.activation_fn = common.get_activation(self.activation)
    layer_dims = [self.hidden_dim] * (self.num_layers - 1) + [
        self.output_dim if self.output_dim else self.hidden_dim
    ]
    self.dense_layers = [
        nn.Dense(
            dim,
            use_bias=self.use_bias,
            dtype=self.matrix_dtype,
            name=f"{self.name_prefix}/dense/layer_{i:02d}",
        )
        for i, dim in enumerate(layer_dims)
    ]

    self.dropout = [
        nn.Dropout(
            rate=self.dropout_rate,
            name=f"{self.name_prefix}/dropout/layer_{i:02d}",
        )
        for i in range(self.num_layers)
    ]

    if self.norm_type == "layer":
      self.norm = nn.LayerNorm(
          dtype=self.norm_dtype,
          name=f"{self.name_prefix}/layer_norm",
      )
    elif self.norm_type is None:
      self.norm = None
    else:
      raise ValueError(f"Unsupported norm type: {self.norm_type}")

  def __call__(
      self, x: jt.Float[jt.Array, "... D_in"], training: bool = False
  ) -> jt.Float[jt.Array, "... D_out"]:
    for i in range(self.num_layers):
      x = self.dense_layers[i](x)
      # Don't apply activation and dropout to the last layer.
      if i < self.num_layers - 1:
        x = self.activation_fn(x)
        x = self.dropout[i](x, deterministic=not training)

    if self.norm is not None:
      x = self.norm(x)

    return x


class GnnPlus(nn.Module):
  """Generic module for GnnPlus https://arxiv.org/pdf/2502.09263.

  This module can be interleaved with any API compatible GNN architecture. It
  doesn't require the topology of the graph as it is applied after message
  passing and therefore can be applied to any set of dense features.

  **NOTE** The `num_layers` must match the number of message passing layers. It
  is up to the caller to ensure this else there will be errors.


  **Note** It is the intention that this "layer" be interleaved after a series
  of `num_layers` blocks and would be difficult to instantiate without an
  interleaving caller. See `layers_test.py` and `sd.py` for examples.

  After performing a message passing layer, execute:
  * Normalization.
  * Activation & Dropout.
  * Residual Connections.
  * MLP.
  * Add and norm.
  """

  hidden_dim: int
  num_layers: int
  activation_fn: str = "tanh"
  ff_layers: int = 2
  dropout_rate: float = common.DEFAULT_DROPOUT_RATE
  use_bias: bool = True
  matrix_dtype: jnp.dtype = common.DEFAULT_MATRIX_PRECISION
  norm_dtype: jnp.dtype = common.DEFAULT_POINTWISE_NORM_PRECISION
  name_prefix: str = "gnn/post_message_passing"

  def setup(self):
    self.activation = common.get_activation(self.activation_fn)

    # Need multiple layers due to PRNG keys.
    self.dropout = [
        nn.Dropout(
            rate=self.dropout_rate,
            name=f"{self.name_prefix}/layer_{i:02d}/dropout/",
        )
        for i in range(self.num_layers)
    ]

    self.pre_norm = [
        nn.LayerNorm(
            use_bias=self.use_bias,
            dtype=self.norm_dtype,
            name=f"{self.name_prefix}/layer_{i:02d}/pre_layer_norm",
        )
        for i in range(self.num_layers)
    ]

    self.post_norm = [
        nn.LayerNorm(
            use_bias=self.use_bias,
            dtype=self.norm_dtype,
            name=f"{self.name_prefix}/layer_{i:02d}/post_layer_norm",
        )
        for i in range(self.num_layers)
    ]

    self.mlp = [
        MLP(
            num_layers=self.ff_layers,
            hidden_dim=self.hidden_dim,
            use_bias=self.use_bias,
            matrix_dtype=self.matrix_dtype,
            norm_dtype=self.norm_dtype,
            dropout_rate=self.dropout_rate,
            name_prefix=f"{self.name_prefix}/layer_{i:02d}/mlp",
        )
        for i in range(self.num_layers)
    ]

  def __call__(
      self,
      h_prev: jt.Float[jt.Array, "... D"],
      h_next: jt.Float[jt.Array, "... D"],
      layer_index: int,
      training: bool = False,
  ) -> jt.Float[jt.Array, "... D"]:
    """Forward pass for a given GNN message passing layer.

    Args:
      h_prev: The hidden state prior to a message passing round (for residual
        sconnection).
      h_next: The hidden state after a message passing round.
      layer_index: Integer index in the sequence of message passing rounds.
      training: Boolean indicating if we are training (add dropout) or
        performing inference (no dropout).

    Returns:
      The hidden state.
    """
    chex.assert_shape(h_prev, (..., self.hidden_dim))
    chex.assert_shape(h_next, (..., self.hidden_dim))
    x = self.pre_norm[layer_index](h_next)

    x = self.activation(x)
    x = self.dropout[layer_index](x, deterministic=not training)

    # Residual
    x = h_prev + x

    # MLP
    mlp_input = x
    x = self.mlp[layer_index](mlp_input)

    # Add and norm
    x = mlp_input + x
    x = self.post_norm[layer_index](x)

    return x
