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

"""Padding and capping for timeseries sequence features in graphs."""

# TODO(simonmeierhans): If a timeseries feature has no group defined, and there
# is a group with the same name as the feature it currently implicitly joins
# that group if it has creation_time=True. This should raise an error instead.

# pytype: disable=module-attr
import dataclasses
import enum
from typing import Any, List, Optional, Tuple

import dataclasses_json
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
from dgf.src.util import temporal as temporal_util
import numpy as np


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class PadAndCapTimeseriesConfig:
  """Configuration for padding and capping timeseries features.

  Attributes:
    sequence_length: Fixed target sequence dimension K. Sequences longer than K
      are capped to the most recent K steps (`[-K:]`). Shorter sequences are
      left-padded to length K.
    padding_value: Scalar value used for left-padding shorter sequences.
  """

  sequence_length: int = 30
  padding_value: Any = 0


def _pad_and_cap_single_feature(
    raw_series: np.ndarray,
    seq_len: int,
    feat_shape: Tuple[int, ...],
    padding_value: Any,
    dtype: Any,
    is_static_shape: bool,
) -> Tuple[np.ndarray, np.ndarray]:
  """Pads and caps a single sequence feature into (padded_matrix, mask_matrix)."""
  num_entities = raw_series.shape[0]
  if dtype == np.bytes_ and raw_series.dtype.kind in ("S", "a"):
    dtype = raw_series.dtype
  if num_entities == 0:
    return (
        np.empty((0, seq_len) + feat_shape, dtype=dtype),
        np.empty((0, seq_len), dtype=np.bool_),
    )

  # Fast vectorized path when all entities share a fixed sequence length.
  if is_static_shape and raw_series.ndim >= 2:
    num_steps = raw_series.shape[1]
    if num_steps >= seq_len:
      padded_matrix = raw_series[:, -seq_len:].astype(dtype, copy=True)
      mask_matrix = np.ones((num_entities, seq_len), dtype=np.bool_)
      return padded_matrix, mask_matrix

    pad_width = [(0, 0), (seq_len - num_steps, 0)] + [(0, 0)] * len(feat_shape)
    padded_matrix = np.pad(
        raw_series.astype(dtype, copy=False),
        pad_width=pad_width,
        mode="constant",
        constant_values=padding_value,
    )
    mask_width = [(0, 0), (seq_len - num_steps, 0)]
    mask_matrix = np.pad(
        np.ones((num_entities, num_steps), dtype=np.bool_),
        pad_width=mask_width,
        mode="constant",
        constant_values=False,
    )
    return padded_matrix, mask_matrix

  padded_matrix = np.full(
      (num_entities, seq_len) + feat_shape,
      fill_value=padding_value,
      dtype=dtype,
  )
  # Binary mask matrix matching sequence length shape (num_entities, seq_len)
  # where True indicates valid observed time steps and False indicates
  # left-padded steps.
  mask_matrix = np.zeros((num_entities, seq_len), dtype=np.bool_)

  # TODO(mesimon): Move into C++ for performance.
  for idx in range(num_entities):
    raw_arr = raw_series[idx]
    if not isinstance(raw_arr, np.ndarray):
      raw_arr = np.asarray(raw_arr)

    num_steps = len(raw_arr)
    if num_steps >= seq_len:
      padded_matrix[idx] = raw_arr[-seq_len:]
      mask_matrix[idx] = True
    elif num_steps > 0:
      padded_matrix[idx, -num_steps:] = raw_arr
      mask_matrix[idx, -num_steps:] = True

  return padded_matrix, mask_matrix


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


class PadAndCapTimeseries:
  """Pads and caps timeseries sequence features into fixed-dimension tensors.

  Transforms variable-length sequence features (`is_timeseries=True`) in the
  graph into fixed-length matrices of shape `(num_entities, sequence_length) +
  step_shape`. Sequences longer than `sequence_length` are capped from the right
  (keeping the most recent steps), while shorter sequences are left-padded.

  For each sequence group, a corresponding boolean mask feature (`{group}_mask`)
  is generated with semantic `MASK`, where `True` indicates observed time steps
  and `False` indicates padded steps.

  Attributes:
    schema: The input `GraphSchema`.
    config: Configuration specifying sequence length and padding value.
    schema_cache: Pre-computed `TimeseriesSchemaCache` for fast grouping.
  """

  def __init__(
      self,
      schema: schema_lib.GraphSchema,
      config: Optional[PadAndCapTimeseriesConfig] = None,
      schema_cache: Optional[temporal_util.TimeseriesSchemaCache] = None,
  ):
    self.config = config or PadAndCapTimeseriesConfig()
    self.schema = schema
    if schema_cache is None:
      schema_cache = temporal_util.extract_timeseries_schema_cache(schema)
    self.schema_cache = schema_cache

  def _compute_feature_set_pad_and_cap_schema(
      self,
      schemas: schema_lib.FeatureSetSchema,
      ts_specs: List[temporal_util.TimeseriesGroupSpec],
  ) -> schema_lib.FeatureSetSchema:
    """Computes schema for a feature set after padding/capping."""
    new_schemas = dict(schemas)
    seq_len = self.config.sequence_length

    ts_features = {}
    for group in ts_specs:
      for feature_name in group.feature_names:
        ts_features[feature_name] = group.timestamp_feature_name

    for feature_name in ts_features:
      feature_schema = schemas[feature_name]
      ts_group = feature_schema.group or feature_name
      new_schemas[feature_name] = dataclasses.replace(
          temporal_util.with_sequence_length(feature_schema, seq_len),
          group=ts_group,
      )
      if feature_schema.semantic == schema_lib.FeatureSemantic.MASK:
        continue

      mask_name = temporal_util.get_mask_feature_name(feature_name, schemas)
      if mask_name is None:
        mask_name = f"{ts_group}_mask"
        if mask_name in schemas:
          raise ValueError(
              f"Cannot generate mask for sequence group '{ts_group}'. The"
              f" fallback mask name '{mask_name}' clashes with an existing"
              " feature in the schema that is not a valid mask. Please"
              " explicitly define a mask feature for this group or rename the"
              " clashing feature."
          )

      if mask_name not in new_schemas:
        new_schemas[mask_name] = schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BOOL,
            semantic=schema_lib.FeatureSemantic.MASK,
            shape=(seq_len,),
            is_timeseries=feature_schema.is_timeseries,
            group=ts_group,
        )

    return new_schemas

  def _pad_and_cap_feature_set(
      self,
      values: in_memory_graph.Features,
      schemas: schema_lib.FeatureSetSchema,
      ts_specs: List[temporal_util.TimeseriesGroupSpec],
  ) -> in_memory_graph.Features:
    """Extracts fixed-dimension sequence features for a feature set."""
    new_values: in_memory_graph.Features = {}
    seq_len = self.config.sequence_length

    # Map timeseries feature names to their associated timestamp feature name.
    ts_features = {}
    for group in ts_specs:
      for feature_name in group.feature_names:
        ts_features[feature_name] = group.timestamp_feature_name

    # Copy over non-timeseries features.
    for feature_name in schemas:
      if feature_name not in ts_features:
        new_values[feature_name] = values[feature_name]

    for feature_name in ts_features:
      feature_schema = schemas[feature_name]

      dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[feature_schema.format]
      feat_shape = temporal_util.get_timeseries_step_shape(feature_schema)

      padded_matrix, mask_matrix = _pad_and_cap_single_feature(
          raw_series=values[feature_name],
          seq_len=seq_len,
          feat_shape=feat_shape,
          padding_value=self.config.padding_value,
          dtype=dtype,
          is_static_shape=feature_schema.is_static_shape(),
      )

      new_values[feature_name] = padded_matrix
      if feature_schema.semantic == schema_lib.FeatureSemantic.MASK:
        continue

      ts_group = feature_schema.group or feature_name
      mask_name = temporal_util.get_mask_feature_name(feature_name, schemas)
      if mask_name is None:
        mask_name = f"{ts_group}_mask"

      # Copy over mask matrix if it doesn't exist yet. Only store once per
      # group.
      if mask_name not in new_values:
        new_values[mask_name] = mask_matrix

    return new_values

  def output_schema(self) -> schema_lib.GraphSchema:
    """Returns the transformed GraphSchema."""
    new_ns_schemas = {}
    for ns_name, ns_schema in self.schema.node_sets.items():
      ts_specs = self.schema_cache.node_sets[ns_name]
      if not ts_specs:
        new_ns_schemas[ns_name] = ns_schema
      else:
        new_ns_schemas[ns_name] = schema_lib.NodeSchema(
            features=self._compute_feature_set_pad_and_cap_schema(
                ns_schema.features, ts_specs
            )
        )

    new_es_schemas = {}
    for es_name, es_schema in self.schema.edge_sets.items():
      ts_specs = self.schema_cache.edge_sets[es_name]
      if not ts_specs:
        new_es_schemas[es_name] = es_schema
      else:
        new_es_schemas[es_name] = schema_lib.EdgeSchema(
            source=es_schema.source,
            target=es_schema.target,
            features=self._compute_feature_set_pad_and_cap_schema(
                es_schema.features, ts_specs
            ),
        )

    return schema_lib.GraphSchema(
        node_sets=new_ns_schemas, edge_sets=new_es_schemas
    )

  def __call__(
      self, graph: in_memory_graph.InMemoryGraph
  ) -> in_memory_graph.InMemoryGraph:
    """Transforms timeseries sequence features in the graph."""
    new_node_sets = {}
    for ns_name, ns_schema in self.schema.node_sets.items():
      ns_val = graph.node_sets[ns_name]
      ts_specs = self.schema_cache.node_sets[ns_name]
      if not ts_specs:
        new_node_sets[ns_name] = ns_val
        continue
      new_vals = self._pad_and_cap_feature_set(
          values=ns_val.features,
          schemas=ns_schema.features,
          ts_specs=ts_specs,
      )
      new_node_sets[ns_name] = in_memory_graph.InMemoryNodeSet(
          num_nodes=ns_val.num_nodes, features=new_vals
      )

    new_edge_sets = {}
    for es_name, es_schema in self.schema.edge_sets.items():
      es_val = graph.edge_sets[es_name]
      ts_specs = self.schema_cache.edge_sets[es_name]
      if not ts_specs:
        new_edge_sets[es_name] = es_val
        continue
      new_vals = self._pad_and_cap_feature_set(
          values=es_val.features,
          schemas=es_schema.features,
          ts_specs=ts_specs,
      )
      new_edge_sets[es_name] = in_memory_graph.InMemoryEdgeSet(
          adjacency=es_val.adjacency, features=new_vals
      )

    return in_memory_graph.InMemoryGraph(
        node_sets=new_node_sets, edge_sets=new_edge_sets
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
          f" but feature '{fname}' is a variable-length object array. Please"
          " run PadAndCapTimeseries first."
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
          f" dynamic shape ({schema.shape}). Please run PadAndCapTimeseries"
          " first."
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
