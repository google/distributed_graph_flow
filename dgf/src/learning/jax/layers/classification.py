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

"""A simple classification head for a GNN."""

import dataclasses
from typing import Optional
import dataclasses_json
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers.registry import registry as layer_registry
import flax.linen as nn
import jax.numpy as jnp

JaxBaseConfig = common.JaxBaseConfig


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True, kw_only=True)
class ClassificationHeadConfig(common.ArchitectureProvider):
  """Configuration for a classification head.

  This head consists of a single dense layer, mapping an input embedding to
  logits of shape [batch, num_classes]. It does not apply any normalization,
  dropout, or activation function.

  To convert the output logits to probabilities, use the
  `ClassificationHead.logits_to_probability` static method.

  Usage example:

    ```python
    head = ClassificationHeadConfig(num_classes=5).make()

    embedding = jnp.ones((2, 16))
    logits = head(embedding, training=True)
    probabilities = ClassificationHead.logits_to_probability(logits)
    ```

  Attributes:
    num_classes: The number of classes to predict.
  """

  num_classes: int

  def make(self, name: Optional[str] = None) -> "ClassificationHead":
    return ClassificationHead(config=self, name=name)

  def architecture(self) -> str:
    return f"Dense({self.num_classes}) # Classification head"


class ClassificationHead(nn.Module):
  """Simple classification head."""

  config: ClassificationHeadConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
    logits = nn.Dense(self.config.num_classes)(x)
    return logits

  @staticmethod
  def logits_to_probability(logits: jnp.ndarray) -> jnp.ndarray:
    """Converts logits to probabilities using softmax.

    Args:
      logits: A jnp.Array of shape [batch, num_classes].

    Returns:
      A jnp.Array of shape [batch, num_classes] with probabilities.
    """
    return nn.softmax(logits, axis=-1)
