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

"""Schema of a graph."""

import dataclasses
import enum
from typing import Callable, Dict, Optional, Tuple
import dataclasses_json

Shape = Optional[Tuple[Optional[int], ...]]


class FeatureFormat(enum.Enum):
  """How a value is represented / stored."""

  INTEGER_32 = "INTEGER_32"
  INTEGER_64 = "INTEGER_64"
  FLOAT_32 = "FLOAT_32"
  FLOAT_64 = "FLOAT_64"
  BYTES = "BYTES"
  BOOL = "BOOL"
  # TODO(gbm): Consolidates those, and the related utilities, in a single place.

  def is_integer(self) -> bool:
    return self in (FeatureFormat.INTEGER_32, FeatureFormat.INTEGER_64)

  def is_float(self) -> bool:
    return self in (FeatureFormat.FLOAT_32, FeatureFormat.FLOAT_64)

  def is_numerical(self) -> bool:
    return self.is_integer() or self.is_float()


class FeatureSemantic(enum.Enum):
  """How a value should be interpreted.

  Possible values:
    NUMERICAL: The feature represents a numerical quantity e.g. count, amount.
    CATEGORICAL: The feature represents a category from a finite set of values
      without ordering.
    EMBEDDING: The feature is a dense and normalized vector embedding that can
      be directly consumed by a neural network.
    TIMESTAMP: The feature represents a timestamp.
    TIMESERIES: The feature represents a time series of values.
    PRIMARY_ID: The feature represents a primary ID.
    MASK: The feature represents a boolean sequence padding mask. Its shape has
      to be broadcastable to the feature it is masking.
  """

  UNKNOWN = "UNKNOWN"
  NUMERICAL = "NUMERICAL"
  CATEGORICAL = "CATEGORICAL"
  EMBEDDING = "EMBEDDING"
  TIMESTAMP = "TIMESTAMP"
  TIMESERIES = "TIMESERIES"
  PRIMARY_ID = "PRIMARY_ID"
  MASK = "MASK"


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class FeatureSchema:
  """Schema for a single feature.

  For GNN training, users are expected to provide the `format` (and
  possibly `semantic` if using auto-normalization) fields. Other fields are
  automatically inferred when computing feature statistics + (auto)
  normalization.

  Attributes:
    format: The physical format of the feature values (e.g., INTEGER, FLOAT).
    semantic: The semantic meaning of the feature (e.g., NUMERICAL,
      CATEGORICAL).
    shape: The shape of the feature tensor. [] or [1] for scalar features.
      `None` if the shape is unknown--but assumed to be () in some locations.
      Shape values None (e.g., [None]) indicates an unknown shape. Unknown
      shapes are not handelled by in-memory-graphs and TF-GNN Graph Samples.
    num_categorical_values: The number of possible categories for CATEGORICAL
      features. `None` for other semantic types or if the numeber of possible
      categories is unknown.
    is_utf8_string: Whether the feature is a UTF-8 string. This is only relevant
      when feature_format is BYTES, to distinguish between Spanner STRING (True)
      and Spanner BYTES (False).
    is_timeseries: Whether the feature represents a temporal series / sequence.
    timestamps: For temporal sequence features, the name of the feature
      containing the corresponding timestamp sequence (e.g., "time"). The
      length of the corresponding timestamps feature must equal the length of
      the timeseries feature along the 0th dimension. Cannot be set for non
      timeseries features.
  """

  format: FeatureFormat
  semantic: FeatureSemantic = FeatureSemantic.UNKNOWN
  shape: Shape = None
  num_categorical_values: Optional[int] = None
  is_utf8_string: Optional[bool] = False
  is_timeseries: Optional[bool] = False
  timestamps: Optional[str] = None

  def is_static_shape(self) -> bool:
    """Returns true if the feature has a fully static shape."""
    if self.shape is None:
      return True
    return all(d is not None for d in self.shape)

  def static_size(self) -> int:
    """Product of all the static dimeneions."""
    if self.shape is None:
      return 1
    size = 1
    for dim in self.shape:
      if dim is not None:
        size *= dim
    return size


FeatureSetSchema = Dict[str, FeatureSchema]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class NodeSchema:
  features: FeatureSetSchema = dataclasses.field(default_factory=dict)


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class EdgeSchema:
  source: str
  target: str
  features: FeatureSetSchema = dataclasses.field(default_factory=dict)


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class GraphSchema:
  node_sets: Dict[str, NodeSchema]
  edge_sets: Dict[str, EdgeSchema]


@dataclasses.dataclass
class GraphSchemaFilter:
  """Filters a GraphSchema to sub-select node sets, edge sets, and features.

  A `GraphSchemaFilter` can be used with IO methods (when reading a graph from
  disk; with the `schema_filter` argument when available), or to operate
  directly on a schema (with `dgf.transform.filter_schema`).

  Attributes:
    nodeset_fn: A callable `(name, schema) -> bool` to filter node sets. If it
      returns True, keep the nodeset. Otherwise, remove it.
    edgeset_fn: A callable `(name, schema) -> bool` to filter edge sets. If it
      returns True, keep the edgeset. Otherwise, remove it.
    feature_fn: A callable `(name, schema) -> bool` to filter node OR edge
      features. If it returns True, keep the feature. Otherwise, remove it.
    node_feature_fn: A callable `(name, schema) -> bool` to filter node
      features. If it returns True, keep the feature. Otherwise, remove it.
    edge_feature_fn: A callable `(name, schema) -> bool` to filter edge
      features. If it returns True, keep the feature. Otherwise, remove it.
  """

  nodeset_fn: Callable[[str, NodeSchema], bool] = lambda key, sch: True
  edgeset_fn: Callable[[str, EdgeSchema], bool] = lambda key, sch: True
  feature_fn: Callable[[str, FeatureSchema], bool] = lambda key, sch: True
  node_feature_fn: Callable[[str, FeatureSchema], bool] = lambda key, sch: True
  edge_feature_fn: Callable[[str, FeatureSchema], bool] = lambda key, sch: True
