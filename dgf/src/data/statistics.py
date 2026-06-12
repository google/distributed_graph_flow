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
import dataclasses_json
from dgf.src.analyse import reservoir_sampling
from dgf.src.data import histogram
from dgf.src.util import util


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


@dataclasses.dataclass
class NodeSetTopologyStatistics:
  num_nodes: histogram.Histogram

  def __str__(self) -> str:
    num_nodes_str = util.indent_string(str(self.num_nodes))
    return f"""\
      num_nodes:
        {num_nodes_str}"""


@dataclasses.dataclass
class EdgeSetTopologyStatistics:
  num_edges: histogram.Histogram
  in_degree_distribution: histogram.Histogram
  out_degree_distribution: histogram.Histogram

  def __str__(self) -> str:
    num_edges_str = util.indent_string(str(self.num_edges))
    in_dist_str = util.indent_string(str(self.in_degree_distribution))
    out_dist_str = util.indent_string(str(self.out_degree_distribution))
    return f"""\
      num_edges:
        {num_edges_str}
      in_degree:
        {in_dist_str}
      out_degree:
        {out_dist_str}"""


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class GraphTopologyStatistics:
  """Statistics about the topology of a graph."""

  node_sets: Dict[str, NodeSetTopologyStatistics]
  edge_sets: Dict[str, EdgeSetTopologyStatistics]
  num_graphs: int = 1

  def __str__(self) -> str:
    return self.__repr__()

  def __repr__(self) -> str:
    if not self.node_sets:
      node_sets_str = "    empty"
    else:
      node_sets_str = "\n".join(
          f"    '{name}':\n{self.node_sets[name]}"
          for name in sorted(self.node_sets.keys())
      )

    if not self.edge_sets:
      edge_sets_str = "    empty"
    else:
      edge_sets_str = "\n".join(
          f"    '{name}':\n{self.edge_sets[name]}"
          for name in sorted(self.edge_sets.keys())
      )

    return f"""\
GraphTopologyStatistics (num_graphs={self.num_graphs}):
  Node Sets:
{node_sets_str}
  Edge Sets:
{edge_sets_str}
"""
