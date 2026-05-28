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

"""A collection of modern standard layers."""

import dataclasses
import re
from typing import Optional
import dataclasses_json
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
import flax.linen as nn
import jax.numpy as jnp
import jaxtyping as jt


def norm(
    x: jnp.ndarray,
    norm_name: Optional[str],
    training: bool,
    name: Optional[str] = None,
) -> jnp.ndarray:
  """Applies a normalization layer to the input."""
  if norm_name is None:
    return x
  if name is None:
    name = norm_name
  if norm_name == "batch_norm":
    return nn.BatchNorm(name=name)(x, use_running_average=not training)
  elif norm_name == "layer_norm":
    return nn.LayerNorm(name=name)(x)
  elif norm_name == "rms_norm":
    return nn.RMSNorm(name=name)(x)
  else:
    raise ValueError(f"Unsupported norm type: {norm_name}")


def modern_residual_mlp(
    dims: int, dropout_rate: Optional[float] = 0.1, expansion_ratio: int = 4
) -> "GenericBlockConfig":
  """Returns a GenericBlockConfig for a modern residual MLP.

  It uses the config string "NL{expansion_ratio}ALDR" which stands for:
  Norm, Linear (expanded), Activation, Linear (contracted), Dropout, Residual.
  """
  return GenericBlockConfig(
      f"NL{expansion_ratio}ALDR", dims=dims, dropout_rate=dropout_rate
  )


def ingest_feature(dims: int, norm: str = "layer_norm") -> "GenericBlockConfig":
  """Returns a GenericBlockConfig for feature ingestion.

  It uses the config string "LAN" which stands for: Linear, Activation, Norm.
  """
  return GenericBlockConfig("LAN", dims=dims, norm=norm)


def sequential_mlp(
    dims: int, num_layers: int = 2, dropout_rate: Optional[float] = None
) -> "GenericBlockConfig":
  """Returns a GenericBlockConfig for a sequential MLP.

  It builds a config string with `num_layers` of Linear + Activation (+ optional
  Dropout) followed by a final Linear layer.
  """
  config_parts = ["N"]
  for _ in range(num_layers):
    config_parts.append("LA")
    if dropout_rate is not None and dropout_rate > 0:
      config_parts.append("D")
  config_parts.append("L")
  return GenericBlockConfig(
      "".join(config_parts), dims=dims, dropout_rate=dropout_rate
  )


def identity() -> "GenericBlockConfig":
  """Returns a GenericBlockConfig that acts as an identity block."""
  return GenericBlockConfig("", dims=1)


def parse_config(config_str: str) -> list[tuple[str, int]]:
  """Parses the config string into a list of (layer_type, multiplier) tuples.

  Normalizes whitespace and case. Validates character structure and positive
  multipliers.
  """
  config_str = config_str.replace(" ", "").upper()
  if not config_str:
    return []

  # Matches L followed by optional digits, or N, A, D, R
  tokens = re.findall(r"(L\d*|N|A|D|R)", config_str)

  # If sum of match lengths is not equal to the length of the config string,
  # it means there were invalid characters or invalid digit structures.
  if sum(len(t) for t in tokens) != len(config_str):
    raise ValueError(
        f"Invalid config string: '{config_str}' contains unsupported"
        " characters or digits not following 'L'."
    )

  parsed = []
  for token in tokens:
    char = token[0]
    if char == "L":
      multiplier = int(token[1:]) if len(token) > 1 else 1
      if multiplier <= 0:
        raise ValueError(
            "Linear layer multiplier must be strictly positive, got"
            f" {multiplier}."
        )
      parsed.append(("L", multiplier))
    else:
      parsed.append((char, 1))
  return parsed


def validate_config(layers: list[tuple[str, int]]) -> None:
  """Validates the parsed layers for redundant consecutive steps."""
  for i in range(len(layers) - 1):
    curr_type, _ = layers[i]
    next_type, _ = layers[i + 1]
    if curr_type == next_type:
      raise ValueError(
          f"Redundant consecutive layers of type '{curr_type}' in config."
      )


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class GenericBlockConfig(common.ArchitectureProvider):
  """Configuration for a generic block parsed from a string."""

  config: str
  dims: int
  norm: Optional[str] = "rms_norm"
  activation: Optional[str] = "silu"
  dropout_rate: Optional[float] = None

  def __post_init__(self):
    # Validate dims
    if self.dims <= 0:
      raise ValueError(f"dims must be strictly positive, got {self.dims}.")

    # Validate config on initialization
    parsed = parse_config(self.config)
    validate_config(parsed)

    # Check that requested layers have valid parameters to avoid silent bypasses
    layer_types = {t for t, _ in parsed}
    if "N" in layer_types and self.norm is None:
      raise ValueError(
          "Normalization layer 'N' was requested in config, but 'norm'"
          " parameter is None."
      )
    if "A" in layer_types and self.activation is None:
      raise ValueError(
          "Activation layer 'A' was requested in config, but 'activation'"
          " parameter is None."
      )
    if "D" in layer_types and (
        self.dropout_rate is None or self.dropout_rate <= 0.0
    ):
      raise ValueError(
          "Dropout layer 'D' was requested in config, but 'dropout_rate' is"
          " None or <= 0.0."
      )

  def make(self, name: Optional[str] = None) -> "GenericBlock":
    return GenericBlock(config=self, name=name)

  def architecture(self) -> str:
    if not self.config:
      return "Identity"

    parsed = parse_config(self.config)
    parts = []
    has_residual = any(t == "R" for t, _ in parsed)
    if has_residual:
      parts.append("X = ...")

    for layer_type, param in parsed:
      if layer_type == "L":
        parts.append(f"Dense({self.dims * param})")
      elif layer_type == "N":
        parts.append(f"Norm({self.norm})")
      elif layer_type == "A":
        parts.append(f"Activation({self.activation})")
      elif layer_type == "D":
        parts.append(f"Dropout({self.dropout_rate})")
      elif layer_type == "R":
        parts.append("Residual(X)")
      else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    return "\n".join(parts)


class GenericBlock(nn.Module):
  """A generic configurable neural network block."""

  config: GenericBlockConfig

  @nn.compact
  def __call__(
      self, x: jt.Float[jt.Array, "... D"], training: bool = False
  ) -> jt.Float[jt.Array, "... D"]:
    if not self.config.config:
      return x

    parsed = parse_config(self.config.config)
    residual_input = x

    for idx, (layer_type, param) in enumerate(parsed):
      if layer_type == "L":
        out_dims = self.config.dims * param
        x = nn.Dense(out_dims, name=f"dense_{idx}")(x)
      elif layer_type == "N":
        x = norm(x, self.config.norm, training, name=f"norm_{idx}")
      elif layer_type == "A":
        if self.config.activation is not None:
          x = common.get_activation(self.config.activation)(x)
      elif layer_type == "D":
        if (
            self.config.dropout_rate is not None
            and self.config.dropout_rate > 0
        ):
          x = nn.Dropout(
              self.config.dropout_rate,
              deterministic=not training,
              name=f"dropout_{idx}",
          )(x)
      elif layer_type == "R":
        x = x + residual_input
      else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    return x
