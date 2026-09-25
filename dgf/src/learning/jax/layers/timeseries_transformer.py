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

"""Transformer Timeseries Encoder layer for DGF."""

import dataclasses
import functools
import textwrap
import dataclasses_json
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt


# Normalizations valid for a padded sequence encoder. `batch_norm` is
# deliberately excluded: its statistics would be pooled over padded timesteps
# and over nodes with different sequence lengths.
_SUPPORTED_NORMS = ("layer_norm", "rms_norm")

_SUPPORTED_POOLINGS = ("last", "mean")


def default_rope_max_wavelength(
    max_timeseries_len: int, head_dim: int
) -> float:
  """Returns a RoPE base wavelength tuned to `max_timeseries_len` and `head_dim`.

  The usual 10000 is tuned for sequences of thousands of tokens. DGF timeseries
  windows are typically much shorter.

  Rule of thumb: pick the base so the slowest frequency pair (at index
  `head_dim // 2 - 1`, with timescale `base**((head_dim - 2) / head_dim)`)
  rotates ~1 rad across a window of length `T = max_timeseries_len`, i.e.
  `base = T**(d / (d - 2))` for `head_dim=d`.

  Args:
    max_timeseries_len: Maximum sequence length `T` of the timeseries window.
    head_dim: Dimension `d` of each attention head (must be positive and even).

  Returns:
    Base wavelength for the RoPE geometric frequency ladder (> 1.0).
  """
  if max_timeseries_len <= 0:
    raise ValueError(
        f"max_timeseries_len must be positive, got {max_timeseries_len}."
    )
  if head_dim <= 0 or head_dim % 2 != 0:
    raise ValueError(
        f"head_dim must be a positive even integer, got {head_dim}."
    )
  effective_len = max(max_timeseries_len, 2)
  if head_dim == 2:
    # 2/0 is undefined. Return the effective length directly.
    return float(effective_len)
  return float(effective_len ** (head_dim / (head_dim - 2)))


def _rope_sin_cos(
    positions: jt.Float[jt.Array, " T"],
    head_dim: int,
    max_wavelength: float,
) -> tuple[jt.Float[jt.Array, "T 1 D2"], jt.Float[jt.Array, "T 1 D2"]]:
  """Returns the RoPE sin/cos tables for `positions`.

  The tables only depend on the sequence length and the config, so they are
  computed once per forward pass and shared by the queries and keys of every
  layer.

  Args:
    positions: Position of each timestep, of shape (seq_len,).
    head_dim: Dimension of each attention head (must be even).
    max_wavelength: Base of the geometric frequency ladder.

  Returns:
    `(sin, cos)`, each of shape (seq_len, 1, head_dim // 2), broadcasting over
    the batch and head dimensions.
  """
  fraction = 2 * jnp.arange(head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = max_wavelength**fraction
  sinusoid = positions[:, jnp.newaxis, jnp.newaxis] / timescale
  return jnp.sin(sinusoid), jnp.cos(sinusoid)


def _apply_rope(
    x: jt.Float[jt.Array, "N T H D"],
    sin: jt.Float[jt.Array, "T 1 D2"],
    cos: jt.Float[jt.Array, "T 1 D2"],
) -> jt.Float[jt.Array, "N T H D"]:
  """Rotates `x` by precomputed RoPE angles.

  Uses the split-half convention: the first and second halves of each head's
  features form the rotation pairs.

  Args:
    x: Queries or keys of shape (batch_size, seq_len, num_heads, head_dim).
    sin: Sine table from `_rope_sin_cos`.
    cos: Cosine table from `_rope_sin_cos`.

  Returns:
    `x` rotated by the RoPE angles, with the same shape and dtype.
  """
  first_half, second_half = jnp.split(x, 2, axis=-1)
  rotated = jnp.concatenate(
      [
          first_half * cos - second_half * sin,
          second_half * cos + first_half * sin,
      ],
      axis=-1,
  )
  return rotated.astype(x.dtype)


class _RoPESelfAttention(nn.Module):
  """Multi-head self-attention with RoPE applied to the queries and keys.

  The attention itself is computed by `jax.nn.dot_product_attention`, which
  can dispatch to fused kernels. The `query`/`key`/`value`/`out` projections
  are named like those of `nn.MultiHeadDotProductAttention`, so the parameter
  tree has the same layout. No dropout is applied to the attention weights.

  Attributes:
    num_heads: Number of attention heads.
    head_dim: Dimension of each attention head (must be even).
  """

  num_heads: int
  head_dim: int

  @nn.compact
  def __call__(
      self,
      x: jt.Float[jt.Array, "N T C"],
      rope_sin: jt.Float[jt.Array, "T 1 D2"],
      rope_cos: jt.Float[jt.Array, "T 1 D2"],
      mask: jt.Bool[jt.Array, "N 1 1 T"] | None = None,
  ) -> jt.Float[jt.Array, "N T C"]:
    """Applies RoPE self-attention.

    Args:
      x: Input of shape (batch_size, seq_len, channels).
      rope_sin: Precomputed RoPE sine table from `_rope_sin_cos`.
      rope_cos: Precomputed RoPE cosine table from `_rope_sin_cos`.
      mask: Optional boolean attention mask, broadcastable to (batch_size,
        num_heads, seq_len, seq_len), where True marks the logits that take
        part in attention.

    Returns:
      The attention output, with the same shape as `x`.
    """
    # Self-attention only: the same positions are applied to queries and keys.
    assert x.shape[-2] == rope_sin.shape[0]
    dense = functools.partial(
        nn.DenseGeneral, features=(self.num_heads, self.head_dim)
    )
    query = _apply_rope(dense(name="query")(x), rope_sin, rope_cos)
    key = _apply_rope(dense(name="key")(x), rope_sin, rope_cos)
    value = dense(name="value")(x)

    h = jax.nn.dot_product_attention(query, key, value, mask=mask)

    return nn.DenseGeneral(features=x.shape[-1], axis=(-2, -1), name="out")(h)


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class TimeseriesTransformerEncoderConfig(common.ArchitectureProvider):
  """Configuration for the Transformer Timeseries Encoder layer.

  Transforms a 3D timeseries sequence tensor (N, T, C) and an optional sequence
  mask (N, T) into a static 2D embedding (N, out_dim) with a stack of pre-norm
  transformer blocks, rotary position embeddings (RoPE), pooling and a dense
  projection.

  Unlike `timeseries_cnn.TimeseriesCNNEncoderConfig`, whose receptive field is
  `1 + num_layers * (kernel_size - 1)` steps, every block here sees the whole
  window.

  Position is encoded only by RoPE, applied to the queries and keys. There is
  no learned or additive positional embedding.

  Attributes:
    out_dim: Output dimension of the final dense projection.
    max_timeseries_len: Maximum sequence length `T` of the timeseries window
      (from `SamplingPlan.max_timeseries_len`), used to derive
      `rope_max_wavelength` when `rope_max_wavelength` is not explicitly set.
    dims: Width of the transformer blocks.
    num_heads: Number of attention heads. Must divide `dims`, and the resulting
      head dimension must be even because RoPE rotates features in pairs.
    num_layers: Number of transformer blocks.
    dropout_rate: Dropout rate of the feed-forward blocks.
    norm: Normalization type, `'layer_norm'` or `'rms_norm'`. Used both for the
      pre-attention norm and, by default, inside `ffn`.
    pooling: How the sequence is reduced to a single embedding. `'last'` takes
      the newest timestep, which is always at index `T - 1` because the
      timeseries pipeline pads on the left. `'mean'` averages the valid steps.
      With either pooling, an entity with no valid steps pools to zeros, so its
      embedding is the `output_projection` bias.
    rope_max_wavelength: Base of the RoPE frequency ladder. Defaults to
      `default_rope_max_wavelength(max_timeseries_len, dims // num_heads)`.
    ffn: Configuration of the feed-forward block applied after each attention
      block. It is expected to carry its own normalization and residual.
      Defaults to `standard.modern_residual_mlp`, i.e. a pre-norm, 4x expanded,
      residual MLP.
  """

  out_dim: int
  max_timeseries_len: int
  dims: int = 64
  num_heads: int = 4
  num_layers: int = 2
  dropout_rate: float = 0.1
  norm: str = "layer_norm"
  pooling: str = "last"
  rope_max_wavelength: float | None = None
  ffn: common.GenericLayer | None = layer_registry.field(default=None)

  def __post_init__(self):
    if self.out_dim <= 0:
      raise ValueError(f"out_dim must be positive, got {self.out_dim}.")
    if self.max_timeseries_len <= 0:
      raise ValueError(
          f"max_timeseries_len must be positive, got {self.max_timeseries_len}."
      )
    if self.dims <= 0:
      raise ValueError(f"dims must be positive, got {self.dims}.")
    if self.num_heads <= 0:
      raise ValueError(f"num_heads must be positive, got {self.num_heads}.")
    if self.num_layers <= 0:
      raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
    if self.dims % self.num_heads != 0:
      raise ValueError(
          f"dims ({self.dims}) must be divisible by num_heads"
          f" ({self.num_heads})."
      )
    head_dim = self.dims // self.num_heads
    if head_dim % 2 != 0:
      raise ValueError(
          f"The head dimension (dims // num_heads = {head_dim}) must be even"
          " because RoPE rotates features in pairs."
      )
    if not 0.0 <= self.dropout_rate <= 1.0:
      raise ValueError(
          f"dropout_rate must be in [0, 1], got {self.dropout_rate}."
      )
    if self.norm not in _SUPPORTED_NORMS:
      raise ValueError(
          f"norm must be one of {_SUPPORTED_NORMS}, got {self.norm!r}. Note"
          " that batch_norm is not supported for sequence encoders: its"
          " statistics would be computed over padded timesteps."
      )
    if self.pooling not in _SUPPORTED_POOLINGS:
      raise ValueError(
          f"pooling must be one of {_SUPPORTED_POOLINGS}, got"
          f" {self.pooling!r}."
      )
    if self.rope_max_wavelength is None:
      self.rope_max_wavelength = default_rope_max_wavelength(
          max_timeseries_len=self.max_timeseries_len,
          head_dim=head_dim,
      )
    elif self.rope_max_wavelength <= 1.0:
      raise ValueError(
          "rope_max_wavelength must be greater than 1, got"
          f" {self.rope_max_wavelength}."
      )

    if self.ffn is None:
      # `standard.modern_residual_mlp`, but sharing the encoder's `norm` rather
      # than keeping `GenericBlockConfig`'s own default.
      self.ffn = dataclasses.replace(
          standard.modern_residual_mlp(
              dims=self.dims, dropout_rate=self.dropout_rate
          ),
          norm=self.norm,
      )

  def make(
      self,
      feature_schema: schema_lib.FeatureSchema,
      mask_schema: schema_lib.FeatureSchema | None = None,
      name: str | None = None,
  ) -> "TimeseriesTransformerEncoder":
    return TimeseriesTransformerEncoder(
        config=self,
        feature_schema=feature_schema,
        mask_schema=mask_schema,
        name=name,
    )

  def architecture(self) -> str:
    assert self.ffn is not None
    assert self.rope_max_wavelength is not None
    parts = []
    parts.append(
        f"TimeseriesTransformerEncoder (dims={self.dims},"
        f" heads={self.num_heads})"
    )
    parts.append(f"Dense({self.dims}) # Input projection")
    parts.append(f"Transformer Block x{self.num_layers}:")
    block = []
    block.append("X = ...")
    block.append(f"Norm({self.norm})")
    block.append(
        f"RoPESelfAttention(heads={self.num_heads},"
        f" max_wavelength={self.rope_max_wavelength:.1f})"
    )
    block.append("Residual(X)")
    block.append("# Post Attention FFN")
    block.append(self.ffn.architecture())
    parts.append(textwrap.indent("\n".join(block), prefix="    "))
    parts.append(f"Pool({self.pooling})")
    parts.append(f"Dense({self.out_dim}) # Output projection")
    return "\n".join(parts)

  def output_schema(self) -> schema_lib.FeatureSchema:
    return schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        shape=(self.out_dim,),
        is_timeseries=False,
    )


class TimeseriesTransformerEncoder(nn.Module):
  """Transformer encoder mapping 3D sequences (N, T, C) -> 2D (N, out_dim)."""

  config: TimeseriesTransformerEncoderConfig
  feature_schema: schema_lib.FeatureSchema
  mask_schema: schema_lib.FeatureSchema | None = None

  @nn.compact
  def __call__(
      self,
      x: jt.Float[jt.Array, "N T C"],
      mask: jt.Bool[jt.Array, "N T"] | None = None,
      training: bool = False,
  ) -> jt.Float[jt.Array, "N out_dim"]:
    """Applies input projection + pre-norm RoPE transformer blocks + pooling.

    Args:
      x: 3D tensor of shape (batch_size, sequence_length, channels).
      mask: Optional boolean tensor of shape (batch_size, sequence_length) where
        True indicates valid positions and False indicates padded/masked
        positions. Must be provided if and only if mask_schema was configured.
      training: Whether the module is executed in training mode (for dropout).

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
      # `jnp.where` rather than `x * mask`: padded slots may hold NaN or inf,
      # and `NaN * 0 = NaN` would leak through every later layer.
      x = jnp.where(mask[..., jnp.newaxis], x, 0.0)
    else:
      assert mask is None

    # 1. Project the channels to the transformer width.
    x = nn.Dense(features=self.config.dims, name="input_projection")(x)
    if mask_expanded is not None:
      x = x * mask_expanded

    # Key-side-only mask of shape (N, 1, 1, T). The outer-product form produced
    # by `nn.make_attention_mask` would make every padded *query* row fully
    # masked; keeping the mask key-side leaves those rows well defined and is
    # safe under fused attention kernels.
    attention_mask = None
    if mask is not None:
      # `jax.nn.dot_product_attention` requires a boolean mask.
      attention_mask = mask.astype(jnp.bool_)[:, jnp.newaxis, jnp.newaxis, :]

    # RoPE tables, computed once and shared by every layer. Integer positions:
    # padded slots take part in the rotation but are removed by the mask, and
    # RoPE is shift invariant, so the left padding of a short sequence does not
    # change the relative geometry of its valid steps.
    assert self.config.rope_max_wavelength is not None
    head_dim = self.config.dims // self.config.num_heads
    rope_sin, rope_cos = _rope_sin_cos(
        positions=jnp.arange(x.shape[1], dtype=jnp.float32),
        head_dim=head_dim,
        max_wavelength=self.config.rope_max_wavelength,
    )

    assert self.config.ffn is not None
    for i in range(self.config.num_layers):
      # 2. Pre-norm self-attention with a residual connection.
      h = standard.norm(
          x, self.config.norm, training, name=f"attention_norm_{i}"
      )
      h = _RoPESelfAttention(
          num_heads=self.config.num_heads,
          head_dim=head_dim,
          name=f"attention_{i}",
      )(h, rope_sin=rope_sin, rope_cos=rope_cos, mask=attention_mask)
      x = x + h

      # 3. Feed-forward block. It carries its own pre-norm and residual.
      x = self.config.ffn.make(name=f"ffn_{i}")(x, training=training)

      # 4. Reset bias/norm shifts at masked positions to 0.0, so that they
      # cannot leak into the pooling.
      if mask_expanded is not None:
        x = x * mask_expanded

    # 5. Pool the sequence into a single embedding.
    if self.config.pooling == "last":
      # The timeseries pipeline pads on the left and keeps the last `seq_len`
      # steps, so index T-1 is the newest observed step for every entity.
      x_pooled = x[:, -1, :]
    elif self.config.pooling == "mean":
      if mask_expanded is not None:
        valid_counts = jnp.maximum(jnp.sum(mask_expanded, axis=1), 1.0)
        x_pooled = jnp.sum(x, axis=1) / valid_counts
      else:
        x_pooled = jnp.mean(x, axis=1)
    else:
      raise ValueError(f"Unsupported pooling: {self.config.pooling}.")

    # 6. Dense linear projection (N, out_dim).
    return nn.Dense(features=self.config.out_dim, name="output_projection")(
        x_pooled
    )
