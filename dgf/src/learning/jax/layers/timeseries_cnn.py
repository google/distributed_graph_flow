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

"""1D CNN Timeseries Encoder layer for DGF."""

import dataclasses
from typing import Optional
import dataclasses_json
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry
import flax.linen as nn
import jax.numpy as jnp
import jaxtyping as jt


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class TimeseriesCNNEncoderConfig(common.ArchitectureProvider):
  """Configuration for 1D CNN Timeseries Encoder layer.

  Transforms a 3D timeseries sequence tensor (N, T, C) and an optional sequence
  mask (N, T) into a static 2D embedding (N, out_dim) via 1D convolutions,
  NAD blocks (norm, activation, dropout), explicit per-layer masking, masked
  mean pooling, and a dense projection layer.

  Attributes:
    out_dim: Output dimension of the final dense projection.
    conv_channels: Number of channels for 1D convolution layers.
    kernel_size: 1D convolution kernel size.
    num_layers: Number of convolution + block layers.
    activation: Activation function name.
    dropout_rate: Dropout rate.
    norm: Normalization type ('layer_norm', 'rms_norm', 'batch_norm', or None).
  """

  out_dim: int
  conv_channels: int = 64
  kernel_size: int = 3
  num_layers: int = 2
  activation: str = "relu"
  dropout_rate: float = 0.1
  norm: Optional[str] = "layer_norm"

  def __post_init__(self):
    if self.out_dim <= 0:
      raise ValueError(f"out_dim must be positive, got {self.out_dim}.")
    if self.conv_channels <= 0:
      raise ValueError(
          f"conv_channels must be positive, got {self.conv_channels}."
      )
    if self.kernel_size <= 0:
      raise ValueError(f"kernel_size must be positive, got {self.kernel_size}.")
    if self.num_layers <= 0:
      raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
    if not (0.0 <= self.dropout_rate <= 1.0):
      raise ValueError(
          f"dropout_rate must be in [0, 1], got {self.dropout_rate}."
      )

  def nad_block(self) -> standard.GenericBlockConfig:
    config_parts = []
    if self.norm is not None:
      config_parts.append("N")
    config_parts.append("A")
    if self.dropout_rate > 0.0:
      config_parts.append("D")
    return standard.GenericBlockConfig(
        config="".join(config_parts),
        dims=self.conv_channels,
        norm=self.norm,
        activation=self.activation,
        dropout_rate=self.dropout_rate if self.dropout_rate > 0.0 else None,
    )

  def make(
      self,
      feature_schema: schema_lib.FeatureSchema,
      mask_schema: Optional[schema_lib.FeatureSchema] = None,
      name: Optional[str] = None,
  ) -> "TimeseriesCNNEncoder":
    return TimeseriesCNNEncoder(
        config=self,
        feature_schema=feature_schema,
        mask_schema=mask_schema,
        name=name,
    )

  def architecture(self) -> str:
    return (
        f"TimeseriesCNNEncoder(num_layers={self.num_layers},"
        f" kernel_size={self.kernel_size}, channels={self.conv_channels},"
        f" out_dim={self.out_dim})"
    )

  def output_schema(self) -> schema_lib.FeatureSchema:
    return schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        shape=(self.out_dim,),
        is_timeseries=False,
    )


class TimeseriesCNNEncoder(nn.Module):
  """1D CNN Encoder mapping 3D sequence embeddings (N, T, C) -> 2D (N, out_dim)."""

  config: TimeseriesCNNEncoderConfig
  feature_schema: schema_lib.FeatureSchema
  mask_schema: Optional[schema_lib.FeatureSchema] = None

  @nn.compact
  def __call__(
      self,
      x: jt.Float[jt.Array, "N T C"],
      mask: Optional[jt.Bool[jt.Array, "N T"]] = None,
      training: bool = False,
  ) -> jt.Float[jt.Array, "N out_dim"]:
    """Applies 1D CNN + Masking + Masked Mean Pooling + Dense projection.

    Args:
      x: 3D tensor of shape (batch_size, sequence_length, channels).
      mask: Optional boolean tensor of shape (batch_size, sequence_length) where
        True indicates valid positions and False indicates padded/masked
        positions. Must be provided if and only if mask_schema was configured.
      training: Whether the module is executed in training mode (for dropout and
        batch normalization).

    Returns:
      2D static embedding tensor of shape (batch_size, out_dim).
    """
    assert x.ndim == 3

    mask_expanded = None
    if self.mask_schema is not None:
      assert mask is not None
      assert self.mask_schema.shape is not None
      assert len(self.mask_schema.shape) == 1
      assert mask.ndim == 2 and mask.shape == (x.shape[0], x.shape[1])
      mask_expanded = jnp.expand_dims(mask.astype(x.dtype), axis=-1)
      x = x * mask_expanded
    else:
      assert mask is None

    # 1. Process 1D Convolutions + NAD Block + Mask Zeroing.
    nad_block = self.config.nad_block()
    for i in range(self.config.num_layers):
      # TODO(simonmeierhans): Try pre-activation and residual connection.
      x = nn.Conv(
          features=self.config.conv_channels,
          kernel_size=(self.config.kernel_size,),
          padding="SAME",
          name=f"conv1d_{i}",
      )(x)
      x = nad_block.make(name=f"nad_block_{i}")(x, training=training)

      # Reset bias/norm shifts at masked positions to 0.0
      if mask_expanded is not None:
        x = x * mask_expanded

    # 2. Final Zero-Masking before Pooling
    if mask_expanded is not None:
      x = x * mask_expanded

    # 3. Masked Global Mean Pooling over time (N, T, C) -> (N, C)
    if mask_expanded is not None:
      valid_counts = jnp.maximum(jnp.sum(mask_expanded, axis=1), 1.0)
      x_pooled = jnp.sum(x, axis=1) / valid_counts
    else:
      x_pooled = jnp.mean(x, axis=1)

    # 4. Dense Linear Projection (N, out_dim)
    return nn.Dense(features=self.config.out_dim, name="output_projection")(
        x_pooled
    )
