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
  logits of shape [batch, 1] (or [batch, num_outputs]). It does not apply any
  normalization, dropout, or activation function.

  The logits are the standardized predictions: The loss should be computed
  against `standardize_labels(labels)`, and `logits_to_predictions` converts the
  logits to predictions.

  Usage example:

    ```python
    head_config = RegressionHeadConfig(label_mean=10.0, label_std=2.0)
    head = head_config.make()

    embedding = jnp.ones((2, 16))
    logits = head(embedding, training=True)
    predictions = head_config.logits_to_predictions(logits)
    ```

  Attributes:
    num_outputs: Number of regression outputs.
    label_mean: Mean of the labels.
    label_std: Standard deviation of the labels.
  """

  num_outputs: int = 1
  label_mean: float = 0.0
  label_std: float = 1.0

  def make(self, name: str | None = None) -> "RegressionHead":
    return RegressionHead(config=self, name=name)

  def architecture(self) -> str:
    if self.label_mean == 0.0 and self.label_std == 1.0:
      return f"Dense({self.num_outputs}) # Regression head"
    return (
        f"Dense({self.num_outputs}) * {self.label_std:g} +"
        f" {self.label_mean:g} # Regression head"
    )

  def standardize_labels(self, labels: jnp.ndarray) -> jnp.ndarray:
    """Standardizes labels to be compared with the logits."""
    return (labels - self.label_mean) / self.label_std

  def logits_to_predictions(self, logits: jnp.ndarray) -> jnp.ndarray:
    """Converts logits to predictions in the scale of the labels."""
    predictions = RegressionHead.logits_to_predictions(logits)
    return predictions * self.label_std + self.label_mean


class RegressionHead(nn.Module):
  """Simple regression head."""

  config: RegressionHeadConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
    # Zero kernel: The initial predictions are the label mean.
    logits = nn.Dense(
        self.config.num_outputs, kernel_init=nn.initializers.zeros
    )(x)
    return logits

  @staticmethod
  def logits_to_predictions(logits: jnp.ndarray) -> jnp.ndarray:
    """Squeezes the last unit dim of the logits if of size 1 (no un-scaling).

    Args:
      logits: A jnp.Array of shape [batch, num_outputs].

    Returns:
      A jnp.Array of shape [batch] (if num_outputs==1) or [batch, num_outputs].
    """
    if logits.shape[-1] == 1:
      return jnp.squeeze(logits, axis=-1)
    return logits
