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

"""Temporal and timeseries feature extraction for graphs."""

# TODO(simonmeierhans): If a timeseries feature has no group defined, and there
# is a group with the same name as the feature it currently implicitly joins
# that group if it has creation_time=True. This should raise an error instead.

# pytype: disable=module-attr
import dataclasses
import enum
from typing import Any, Optional, Tuple

import dataclasses_json
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.util import temporal as temporal_util
import numpy as np


class CalendarFeature(str, enum.Enum):
  """Supported calendar features to extract from timestamps."""

  SECOND = "second"
  MINUTE = "minute"
  HOUR = "hour"
  DAY_OF_WEEK = "day_of_week"
  MONTH = "month"
  YEAR = "year"


_SUPPORTED_CALENDAR_FEATURES = tuple(CalendarFeature)


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class CalendarFeatureExtractorConfig:
  """Configuration for extracting calendar features from timestamps.

  Attributes:
    features: Tuple of calendar feature enums to extract. Supported values:
      CalendarFeature.SECOND, CalendarFeature.MINUTE, CalendarFeature.HOUR,
      CalendarFeature.DAY_OF_WEEK, CalendarFeature.MONTH, CalendarFeature.YEAR.
  """

  features: Tuple[CalendarFeature, ...] = _SUPPORTED_CALENDAR_FEATURES


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class TimestampFeatureExtractorConfig:
  """Configuration for extracting time delta features.

  Attributes:
    fill_value: Value used for masked time steps and missing boundary deltas.
  """

  fill_value: Any = 0


def _compute_calendar_feature(
    ts_array: np.ndarray, feature: CalendarFeature
) -> np.ndarray:
  """Computes a single vectorized calendar feature from an int64 timestamp array."""

  if feature == CalendarFeature.SECOND:
    return (ts_array % 60).astype(np.float32)
  if feature == CalendarFeature.MINUTE:
    return ((ts_array // 60) % 60).astype(np.float32)
  if feature == CalendarFeature.HOUR:
    return ((ts_array // 3600) % 24).astype(np.float32)
  if feature == CalendarFeature.DAY_OF_WEEK:
    return (((ts_array // 86400) + 3) % 7).astype(np.float32)

  dt = ts_array.astype("datetime64[s]")

  if feature == CalendarFeature.MONTH:
    return (dt.astype("datetime64[M]").astype(int) % 12 + 1).astype(np.float32)
  if feature == CalendarFeature.YEAR:
    return (dt.astype("datetime64[Y]").astype(int) + 1970).astype(np.float32)

  raise ValueError(
      f"Unsupported calendar feature: '{feature}'. Supported features:"
      f" {[f.value for f in _SUPPORTED_CALENDAR_FEATURES]}"
  )


class CalendarFeatureExtractor:
  """Extracts calendar features (e.g. hour, day_of_week) from timestamp features.

  Scans the input `GraphSchema` for all features with semantic `TIMESTAMP`
  and derives specified calendar features (such as second, minute, hour,
  day of the week, month, and year) in UTC.

  For each timestamp feature `feat` and configured calendar event `event`:
  - An output feature named `{feat}_{event.value}` is generated.
  - The output feature has format `FLOAT_32` and semantic `NUMERICAL`.
  - The output feature inherits the shape and `is_timeseries` flag of the parent
    timestamp feature, as well as its sequence `group` (if part of a
    timeseries).

  Attributes:
    schema: The input `GraphSchema`.
    config: Configuration specifying which calendar features to extract.
  """

  def __init__(
      self,
      schema: schema_lib.GraphSchema,
      config: Optional[CalendarFeatureExtractorConfig] = None,
  ):
    self.config = config or CalendarFeatureExtractorConfig()
    self.schema = schema

  def _compute_feature_set_calendar_schema(
      self,
      schemas: schema_lib.FeatureSetSchema,
  ) -> schema_lib.FeatureSetSchema:
    """Computes schema for a feature set after extracting calendar features."""
    new_schemas = dict(schemas)
    for fname, schema in schemas.items():
      if schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP:
        continue
      if schema.group is not None:
        cal_group = schema.group
      elif schema.is_timeseries:
        cal_group = fname
      else:
        cal_group = None

      for cal_feat in self.config.features:
        out_fname = f"{fname}_{cal_feat.value}"
        new_schemas[out_fname] = schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.NUMERICAL,
            shape=schema.shape,
            is_timeseries=schema.is_timeseries,
            group=cal_group,
        )
    return new_schemas

  def _extract_feature_set_calendar_features(
      self,
      values: in_memory_graph.Features,
      schemas: schema_lib.FeatureSetSchema,
  ) -> in_memory_graph.Features:
    """Extracts calendar features from timestamp features of a single feature set."""
    new_values: in_memory_graph.Features = {}

    for fname, schema in schemas.items():
      raw_val = values[fname]
      new_values[fname] = raw_val

      # Skip non-timestamp features.
      if schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP:
        continue

      assert raw_val.dtype != np.object_, (
          "CalendarFeatureExtractor requires fixed-length timestamp tensors,"
          f" but feature '{fname}' is a variable-length object array."
      )

      for cal_feat in self.config.features:
        out_fname = f"{fname}_{cal_feat.value}"
        new_values[out_fname] = _compute_calendar_feature(raw_val, cal_feat)

    return new_values

  def output_schema(self) -> schema_lib.GraphSchema:
    """Returns the transformed GraphSchema."""
    new_ns_schemas = {}
    for ns_name, ns_schema in self.schema.node_sets.items():
      new_ns_schemas[ns_name] = schema_lib.NodeSchema(
          features=self._compute_feature_set_calendar_schema(ns_schema.features)
      )

    new_es_schemas = {}
    for es_name, es_schema in self.schema.edge_sets.items():
      new_es_schemas[es_name] = schema_lib.EdgeSchema(
          source=es_schema.source,
          target=es_schema.target,
          features=self._compute_feature_set_calendar_schema(
              es_schema.features
          ),
      )

    return schema_lib.GraphSchema(
        node_sets=new_ns_schemas, edge_sets=new_es_schemas
    )

  def __call__(
      self, graph: in_memory_graph.InMemoryGraph
  ) -> in_memory_graph.InMemoryGraph:
    """Extracts calendar features from timestamps in the graph."""
    new_node_sets = {}
    for ns_name, ns_schema in self.schema.node_sets.items():
      ns_val = graph.node_sets[ns_name]
      new_vals = self._extract_feature_set_calendar_features(
          values=ns_val.features,
          schemas=ns_schema.features,
      )
      new_node_sets[ns_name] = in_memory_graph.InMemoryNodeSet(
          num_nodes=ns_val.num_nodes, features=new_vals
      )

    new_edge_sets = {}
    for es_name, es_schema in self.schema.edge_sets.items():
      es_val = graph.edge_sets[es_name]
      new_vals = self._extract_feature_set_calendar_features(
          values=es_val.features,
          schemas=es_schema.features,
      )
      new_edge_sets[es_name] = in_memory_graph.InMemoryEdgeSet(
          adjacency=es_val.adjacency, features=new_vals
      )

    return in_memory_graph.InMemoryGraph(
        node_sets=new_node_sets, edge_sets=new_edge_sets
    )


def _compute_seed_deltas(
    raw_val: np.ndarray,
    mask: Optional[np.ndarray],
    seed_timestamp: int,
    fill_value: Any,
) -> np.ndarray:
  """Computes seed_timestamp - t_i."""
  deltas = seed_timestamp - raw_val
  if mask is not None:
    mask_for_where = temporal_util.expand_mask_dims(mask, raw_val)
    deltas = np.where(mask_for_where, deltas, fill_value)
  return deltas


class TimestampFeatureExtractor:
  """Extracts time delta features from timestamp features."""

  def __init__(
      self,
      schema: schema_lib.GraphSchema,
      config: Optional[TimestampFeatureExtractorConfig] = None,
  ):
    self.config = config or TimestampFeatureExtractorConfig()
    self.schema = schema

  def _compute_feature_set_timestamp_schema(
      self,
      schemas: schema_lib.FeatureSetSchema,
  ) -> schema_lib.FeatureSetSchema:
    """Computes schema for a feature set after extracting time delta features."""
    new_schemas = dict(schemas)
    for fname, schema in schemas.items():
      if schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP:
        continue
      if schema.group is not None:
        ts_group = schema.group
      # If the timestamp features is a creation time timeseries, we need to
      # infer a group name to link it to a potential future mask feature.
      elif schema.is_timeseries and schema.is_creation_time:
        ts_group = fname
      else:
        ts_group = None

      out_fname = f"{fname}_seed_delta"
      new_schemas[out_fname] = schema_lib.FeatureSchema(
          format=schema.format,
          semantic=schema_lib.FeatureSemantic.TIMEDELTA,
          shape=schema.shape,
          is_timeseries=schema.is_timeseries,
          group=ts_group,
      )
    return new_schemas

  def _extract_feature_set_timestamp_features(
      self,
      values: in_memory_graph.Features,
      schemas: schema_lib.FeatureSetSchema,
      seed_timestamp: int,
  ) -> in_memory_graph.Features:
    """Extracts time delta features for a single feature set."""
    new_values: in_memory_graph.Features = {}
    assert seed_timestamp is not None, (
        "seed_timestamp must be provided to extract seed deltas."
    )

    for fname, schema in schemas.items():
      raw_val = values[fname]
      new_values[fname] = raw_val

      if schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP:
        continue

      assert schema.is_static_shape() and raw_val.dtype != np.object_, (
          "TimestampFeatureExtractor requires fixed-length timestamp tensors,"
          f" but feature '{fname}' is a variable-length object array or has"
          f" dynamic shape ({schema.shape}). Please pad timeseries features"
          " (e.g. via pad_timeseries_graph) first."
      )

      mask = None
      ts_group = schema.group or (
          fname if schema.is_timeseries and schema.is_creation_time else None
      )
      if ts_group is not None:
        mask_name = temporal_util.get_mask_feature_name(fname, schemas)
        if mask_name is not None and mask_name in values:
          mask = values[mask_name]

      out_fname = f"{fname}_seed_delta"
      new_values[out_fname] = _compute_seed_deltas(
          raw_val, mask, seed_timestamp, self.config.fill_value
      )

    return new_values

  def output_schema(self) -> schema_lib.GraphSchema:
    """Returns the transformed GraphSchema."""
    new_ns_schemas = {}
    for ns_name, ns_schema in self.schema.node_sets.items():
      new_ns_schemas[ns_name] = schema_lib.NodeSchema(
          features=self._compute_feature_set_timestamp_schema(
              ns_schema.features
          )
      )

    new_es_schemas = {}
    for es_name, es_schema in self.schema.edge_sets.items():
      new_es_schemas[es_name] = schema_lib.EdgeSchema(
          source=es_schema.source,
          target=es_schema.target,
          features=self._compute_feature_set_timestamp_schema(
              es_schema.features
          ),
      )

    return schema_lib.GraphSchema(
        node_sets=new_ns_schemas, edge_sets=new_es_schemas
    )

  def __call__(
      self, graph: in_memory_graph.InMemoryGraph, seed_timestamp: int
  ) -> in_memory_graph.InMemoryGraph:
    """Extracts timedelta features from timestamps relative to seed_timestamp."""
    new_node_sets = {}
    for ns_name, ns_schema in self.schema.node_sets.items():
      ns_val = graph.node_sets[ns_name]
      new_vals = self._extract_feature_set_timestamp_features(
          values=ns_val.features,
          schemas=ns_schema.features,
          seed_timestamp=seed_timestamp,
      )
      new_node_sets[ns_name] = in_memory_graph.InMemoryNodeSet(
          num_nodes=ns_val.num_nodes, features=new_vals
      )

    new_edge_sets = {}
    for es_name, es_schema in self.schema.edge_sets.items():
      es_val = graph.edge_sets[es_name]
      new_vals = self._extract_feature_set_timestamp_features(
          values=es_val.features,
          schemas=es_schema.features,
          seed_timestamp=seed_timestamp,
      )
      new_edge_sets[es_name] = in_memory_graph.InMemoryEdgeSet(
          adjacency=es_val.adjacency, features=new_vals
      )

    return in_memory_graph.InMemoryGraph(
        node_sets=new_node_sets, edge_sets=new_edge_sets
    )
