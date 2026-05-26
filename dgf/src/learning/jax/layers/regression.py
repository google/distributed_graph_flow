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

"""A simple regression head for a GNN."""

import dataclasses
from typing import Optional
import dataclasses_json
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers.registry import registry as layer_registry
import flax.linen as nn
import jax.numpy as jnp


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True, kw_only=True)
class RegressionHeadConfig(common.ArchitectureProvider):
  """Configuration for a regression head.

  This head consists of a single dense layer, mapping an input embedding to
  logits of shape [batch, 1]. It does not apply any normalization,
  dropout, or activation function.

  To convert the output logits to predictions, use the
  `RegressionHead.logits_to_predictions` static method.

  Usage example:

    ```python
    head = RegressionHeadConfig().make()

    embedding = jnp.ones((2, 16))
    logits = head(embedding, training=True)
    predictions = RegressionHead.logits_to_predictions(logits)
    ```
  """

  def make(self, name: Optional[str] = None) -> "RegressionHead":
    return RegressionHead(config=self, name=name)

  def architecture(self) -> str:
    return "Dense(1) # Regression head"


class RegressionHead(nn.Module):
  """Simple regression head."""

  config: RegressionHeadConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
    logits = nn.Dense(1)(x)
    return logits

  @staticmethod
  def logits_to_predictions(logits: jnp.ndarray) -> jnp.ndarray:
    """Converts logits to predictions (squeezes the last unit dim).

    Args:
      logits: A jnp.Array of shape [batch, 1].

    Returns:
      A jnp.Array of shape [batch] with predictions.
    """
    return jnp.squeeze(logits, axis=-1)
