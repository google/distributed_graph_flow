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

"""Statistics about graphs."""

import dataclasses
import math
from typing import Dict, List, Optional
from dgf.src.analyse import reservoir_sampling


@dataclasses.dataclass
class DictionaryItem:
  index: int
  count: int


# Dictionary of key->(index, count). Non-utf8 characters are encoded using the
# 'surrogateescape' method.
Dictionary = Dict[str, DictionaryItem]

# Dictionary of key->count. Non-utf8 characters are encoded using the
# 'surrogateescape' method.
AccumulatorDictionary = Dict[str, int]


@dataclasses.dataclass
class FeatureStatistics:
  """Statistics for a feature.

  Attributes:
    count: The number of values for the feature. For multi-dim features, each
      observation counts as 1.
    minimum: Minimum value. For INTERGER and FLOAT only.
    maximum: Maximum value. For INTERGER and FLOAT only.
    dictionary: Dictionary of key->(index, count). For BYTES only. Notes:
      Non-utf8 characters are encoded using the 'surrogateescape' method.
    quantiles: Quantiles over the dataset distribution.
  """

  count: int = 0
  minimum: float = math.nan
  maximum: float = math.nan
  dictionary: Dictionary = dataclasses.field(default_factory=dict)
  quantiles: List[float] = dataclasses.field(default_factory=list)

  def __str__(self) -> str:
    s = f"count={self.count}"
    if self.minimum != math.inf and self.maximum != -math.inf:
      s += f", min={self.minimum:.4f}, max={self.maximum:.4f}"
    if self.dictionary:
      dt = self.dictionary
      # Show the first 5 dictionary values.
      first_5_items = list(dt.items())[:5]
      s += f", dictionary=({len(dt)})["
      s += ", ".join([f"{k!r}: {v.count}" for k, v in first_5_items])
      if len(dt) > 5:
        s += ", ..."
      s += "]"
    if self.quantiles:
      qs = self.quantiles
      # Show the first 3 and last 3 quantile values.
      s += f", quantiles=({len(qs)})["
      if len(qs) > 6:
        s += (
            f"{qs[0]:.4f}, {qs[1]:.4f}, {qs[2]:.4f},"
            f" ..., {qs[-3]:.4f}, {qs[-2]:.4f}, {qs[-1]:.4f}"
        )
      else:
        s += ", ".join([f"{q:.4f}" for q in qs])
      s += "]"
    return s


@dataclasses.dataclass
class FeatureSetStatistics:
  """Statistics for a set of features."""

  features: Dict[str, FeatureStatistics]

  def to_string(self, prefix: str) -> str:
    s = ""
    for feature_name in sorted(self.features.keys()):
      stats = self.features[feature_name]
      s += f"{prefix}{feature_name!r}: {stats}\n"
    return s


@dataclasses.dataclass
class GraphFeatureStatistics:
  """Statistics about the features in a graph."""

  node_sets: Dict[str, FeatureSetStatistics]

  def __str__(self) -> str:
    return self.__repr__()

  def __repr__(self) -> str:
    s = "GraphFeatureStatistics:\n"
    if not self.node_sets:
      s += "  No node sets."
      return s
    s += f"  Node Sets ({len(self.node_sets)}):\n"
    for node_set_name in sorted(self.node_sets.keys()):
      feature_set = self.node_sets[node_set_name]
      s += f"    '{node_set_name}':\n"
      s += feature_set.to_string(prefix="      ")
    return s


@dataclasses.dataclass
class FeatureStatisticsAccumulator:
  """In-computation statistics for a feature.

  Attributes:
    count: The number of values encountered. A multi-dim feature counts as one.
    minimum: The minimum value encountered.
    maximum: The maximum value encountered.
    dictionary: An dictionary mapping keys to their index + count. For
      categorical features.
    quantiles: An quantiles. For numerical and timeseries features.
  """

  count: int
  minimum: float
  maximum: float
  dictionary: Optional[Dict[str, int]]
  quantiles: Optional[reservoir_sampling.BatchReservoirSampling]


@dataclasses.dataclass
class FeatureSetStatisticsAccumulator:
  """In-computation statistics for a set of features."""

  features: Dict[str, FeatureStatisticsAccumulator]
