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

"""Residual MLP."""

import dataclasses
from typing import Optional
import dataclasses_json
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers.registry import registry as layer_registry
import flax.linen as nn
import jax.numpy as jnp
import jaxtyping as jt


class ResidualMLP(nn.Module):
  """A simple Residual MLP."""

  hidden_dim: int
  output_dim: int
  use_bias: bool = True
  activation = "relu"
  dropout_rate: float = common.DEFAULT_DROPOUT_RATE
  matrix_dtype: jnp.dtype = common.DEFAULT_MATRIX_PRECISION
  norm_dtype: jnp.dtype = common.DEFAULT_POINTWISE_NORM_PRECISION
  name_prefix: str = "residual_mlp"

  def setup(self):
    self.activation = common.get_activation(self.activation)  # pyrefly: ignore[bad-assignment]

    self.hidden_layer = nn.Dense(
        features=self.hidden_dim,
        use_bias=self.use_bias,
        name=f"{self.name_prefix}/dense/hidden_layer",
    )
    self.output_layer = nn.Dense(
        features=self.output_dim,
        use_bias=self.use_bias,
        name=f"{self.name_prefix}/dense/output_layer",
    )

    # Projects the input into the hidden state w/o activation for residual.
    self.residual_layer = nn.Dense(
        features=self.output_dim,
        use_bias=self.use_bias,
        name=f"{self.name_prefix}/dense/residual_layer",
    )

  def __call__(
      self, x: jt.Float[jt.Array, "... D"], training: bool = False
  ) -> jt.Float[jt.Array, "... D"]:
    return self.output_layer(
        self.activation(self.hidden_layer(x))  # pyrefly: ignore[not-callable]
    ) + self.residual_layer(x)


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ResidualMLPV2Config:
  """A residual MLP layer.

  Structured as: [dense + norm + activation + drop-out + residual] * num_layers

  If the output dim (dims) is different from the input dim, an extra projection
  is applied to residual.

  Attributes:
    dims: The number of hidden units in each dense layer.
    num_layers: The number of layers in the MLP.
    activation: The activation function to use after the normalization. Supports
      all activations in common.get_activation.
    dropout_rate: The dropout rate to apply after the activation. If None, no
      dropout is applied.
    norm: The normalization layer to use. Supports 'batch_norm', 'layer_norm',
      or None.
    residual: Whether to use a residual connection.

  Usage example:
    ```
    layer = ResidualMLPV2Config(dims=64, num_layers=2).make()
    ```
  """

  dims: int
  num_layers: int = 1
  activation: str = "gelu"
  dropout_rate: Optional[float] = 0.1
  norm: Optional[str] = "batch_norm"
  residual: bool = True

  def make(self, name: Optional[str] = None) -> "ResidualMLPV2":
    return ResidualMLPV2(config=self, name=name)


class ResidualMLPV2(nn.Module):
  """A residual MLP layer. See ResidualMLPV2Config."""

  config: ResidualMLPV2Config

  @nn.compact
  def __call__(
      self, x: jt.Float[jt.Array, "... D"], training: bool = False
  ) -> jt.Float[jt.Array, "... D"]:

    if self.config.residual:
      residual = x

      if x.shape[-1] != self.config.dims:
        # Extra dense to match the residual to the output dim.
        residual = nn.Dense(self.config.dims, name="residual_proj")(residual)
    else:
      residual = None

    for layer_idx in range(self.config.num_layers):
      x = nn.Dense(self.config.dims, name=f"dense_{layer_idx}")(x)

      if self.config.norm is None:
        pass
      elif self.config.norm == "batch_norm":
        x = nn.BatchNorm(not training, name=f"batch_norm_{layer_idx}")(x)
      elif self.config.norm == "layer_norm":
        x = nn.LayerNorm(name=f"layer_norm_{layer_idx}")(x)
      else:
        raise ValueError(f"Unsupported norm type: {self.config.norm}")

      x = common.get_activation(self.config.activation)(x)

      if self.config.dropout_rate is not None and self.config.dropout_rate > 0:
        x = nn.Dropout(self.config.dropout_rate, deterministic=not training)(x)

    if self.config.residual:
      assert residual is not None
      x = x + residual

    return x
