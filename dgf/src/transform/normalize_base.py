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

"""Base class and registry shared by all feature normalizers."""

from __future__ import annotations

import abc
import dataclasses

from dgf.src.data import schema as schema_lib
from dgf.src.util import dataclass_registry
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
import numpy as np

normalizer_registry = dataclass_registry.create_registry("normalizers")


@dataclasses.dataclass
class AbstractFeatureNormalizer(abc.ABC):
  """Abstract base class for normalizing model input features.

  Usage example:

  ```python
  input_schema: statistics_lib.FeatureSetSchema = ...
  input_stats: statistics_lib.FeatureStatistics = ...

  # Create a normalizer (assume `MyNormalizer` derives
  # `AbstractFeatureNormalizer`).
  normalizer = MyNormalizer("f1", stats.nodesets["n1"].features["f1"])
  # Or, if the normalizer has a factory constructor.
  normalizer = MyNormalizer.create(...)

  # Normalize feature stored as numpy array
  raw_value = np.array(...)
  normalized_value = normalizer.normalize_numpy(raw_value)

  # Get the schema of the normalizer output.
  normalized_schema = normalizer.output_schema()
  ```

  You can apply normalizers (i.e., classes that derive from
  AbstractFeatureNormalizer) manually, or you can call `auto_normalize` to
  automatically apply the feature normalizers on a graph.

  Attributes:
    input_feature: The name of the input feature this normalizer operates on.
    type: The name of the normalizer class.
  """

  input_feature: str
  type: str

  @abc.abstractmethod
  def output_schema(self) -> schema_lib.FeatureSetSchema:
    """Returns the schema of the normalized feature."""
    pass

  @abc.abstractmethod
  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    """Applies the normalization to a numpy array of feature values."""
    pass

  @abc.abstractmethod
  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
    """Applies the normalization to a dictionary of TensorFlow tensors."""
    pass

  def tensorflow_resources(self) -> list[tf.Tensor]:
    """Returns the list of TensorFlow resources used by `normalize_tensorflow`.

    Access to resources is necessary to serialize a `tf.Module`. If the
    normalization operation uses resources, `normalize_tensorflow` is expected
    to call this method. Calling this method might initialize the TensorFlow
    resources if they have not been created already.

    Returns:
      A list of TensorFlow resource tensors.
    """
    return []


def output_timeseries_group(
    input_feature: str, input_schema: schema_lib.FeatureSchema
) -> str | None:
  """Returns the timeseries group a feature's normalized outputs belong to.

  Falls back to the input feature name rather than leaving the group unset:
  normalized outputs are renamed, so grouping them by their own name would
  scatter the outputs of one input across separate groups.

  Args:
    input_feature: Name of the input feature.
    input_schema: Schema of the input feature.

  Returns:
    The timeseries group, or None if the feature is not a timeseries.
  """
  if input_schema.group:
    return input_schema.group
  elif input_schema.is_timeseries:
    return input_feature
  else:
    return None
