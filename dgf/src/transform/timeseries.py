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
class CalendarFeatureConfig:
  """Configuration for extracting calendar features from timestamps.

  Attributes:
    features: Tuple of calendar feature enums to extract. Supported values:
      CalendarFeature.SECOND, CalendarFeature.MINUTE, CalendarFeature.HOUR,
      CalendarFeature.DAY_OF_WEEK, CalendarFeature.MONTH, CalendarFeature.YEAR.
  """

  features: Tuple[CalendarFeature, ...] = _SUPPORTED_CALENDAR_FEATURES


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


def _process_feature_set(
    values: in_memory_graph.Features,
    schemas: schema_lib.FeatureSetSchema,
    ts_specs: List[temporal_util.TimeseriesGroupSpec],
    config: PadAndCapTimeseriesConfig,
) -> Tuple[in_memory_graph.Features, schema_lib.FeatureSetSchema]:
  """Extracts fixed-dimension sequence features for a feature set."""
  new_values: in_memory_graph.Features = {}
  new_schemas: schema_lib.FeatureSetSchema = {}
  seq_len = config.sequence_length

  # Map timeseries feature names to their associated timestamp feature name.
  ts_features = {}
  for group in ts_specs:
    for feature_name in group.feature_names:
      ts_features[feature_name] = group.timestamp_feature_name

  for feature_name, feature_schema in schemas.items():
    if feature_name not in ts_features:
      new_values[feature_name] = values[feature_name]
      new_schemas[feature_name] = feature_schema

  if not ts_features:
    return new_values, new_schemas

  for feature_name in ts_features:
    feature_schema = schemas[feature_name]

    dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[feature_schema.format]
    feat_shape = temporal_util.get_timeseries_step_shape(feature_schema)

    padded_matrix, mask_matrix = _pad_and_cap_single_feature(
        raw_series=values[feature_name],
        seq_len=seq_len,
        feat_shape=feat_shape,
        padding_value=config.padding_value,
        dtype=dtype,
        is_static_shape=feature_schema.is_static_shape(),
    )

    ts_group = feature_schema.group or feature_name
    new_values[feature_name] = padded_matrix
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

    if mask_name not in new_values:
      new_values[mask_name] = mask_matrix
      new_schemas[mask_name] = schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BOOL,
          semantic=schema_lib.FeatureSemantic.MASK,
          shape=(seq_len,),
          is_timeseries=feature_schema.is_timeseries,
          group=ts_group,
      )

  return new_values, new_schemas


def pad_and_cap_timeseries_features(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    config: PadAndCapTimeseriesConfig,
    schema_cache: Optional[temporal_util.TimeseriesSchemaCache] = None,
) -> Tuple[in_memory_graph.InMemoryGraph, schema_lib.GraphSchema]:
  """Pads and caps timeseries sequence features into fixed-dimension tensors.

  For every feature where `is_timeseries=True`, this function caps long causal
  histories (`[-K:]`) and left-pads short histories to `config.sequence_length`
  ($K$). Generates exactly one binary attention mask per sequence group,
  shared across all co-grouped features.

  All resulting features have fixed `shape=(K,)` and maintain
  `is_timeseries=True`, allowing downstream sequence models to identify
  temporal features and inherit the group mask.

  Usage example:

  ```python
  config = dgf.transform.PadAndCapTimeseriesConfig(sequence_length=30)
  new_graph, new_schema = dgf.transform.pad_and_cap_timeseries_features(
      graph, schema, config
  )
  ```

  Args:
    graph: Input in-memory graph (`InMemoryGraph`), usually a sampled subgraph.
    schema: The graph schema where sequence features have `is_timeseries=True`.
    config: `PadAndCapTimeseriesConfig` specifying sequence length.
    schema_cache: Optional pre-computed `TimeseriesSchemaCache` for reuse.

  Returns:
    Tuple `(new_graph, new_schema)` containing fixed shape 2D or 3D tensors.
  """
  # TODO(mesimon): Pre-compute and reuse schema_cache across dataset examples
  # rather than recomputing cache for every graph sample.
  if schema_cache is None:
    schema_cache = temporal_util.extract_timeseries_schema_cache(schema)

  new_node_sets = {}
  new_ns_schemas = {}

  for ns_name, ns_schema in schema.node_sets.items():
    ns_val = graph.node_sets[ns_name]
    ts_specs = schema_cache.node_sets[ns_name]
    if not ts_specs:
      new_node_sets[ns_name] = ns_val
      new_ns_schemas[ns_name] = ns_schema
      continue

    new_vals, new_schemas = _process_feature_set(
        values=ns_val.features,
        schemas=ns_schema.features,
        ts_specs=ts_specs,
        config=config,
    )
    new_node_sets[ns_name] = in_memory_graph.InMemoryNodeSet(
        num_nodes=ns_val.num_nodes, features=new_vals
    )
    new_ns_schemas[ns_name] = schema_lib.NodeSchema(features=new_schemas)

  new_edge_sets = {}
  new_es_schemas = {}

  for es_name, es_schema in schema.edge_sets.items():
    es_val = graph.edge_sets[es_name]
    ts_specs = schema_cache.edge_sets[es_name]
    if not ts_specs:
      new_edge_sets[es_name] = es_val
      new_es_schemas[es_name] = es_schema
      continue

    new_vals, new_schemas = _process_feature_set(
        values=es_val.features,
        schemas=es_schema.features,
        ts_specs=ts_specs,
        config=config,
    )
    new_edge_sets[es_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=es_val.adjacency, features=new_vals
    )
    new_es_schemas[es_name] = schema_lib.EdgeSchema(
        source=es_schema.source,
        target=es_schema.target,
        features=new_schemas,
    )

  return (
      in_memory_graph.InMemoryGraph(
          node_sets=new_node_sets, edge_sets=new_edge_sets
      ),
      schema_lib.GraphSchema(
          node_sets=new_ns_schemas, edge_sets=new_es_schemas
      ),
  )


def _extract_feature_set_calendar_features(
    values: in_memory_graph.Features,
    schemas: schema_lib.FeatureSetSchema,
    config: CalendarFeatureConfig,
) -> Tuple[in_memory_graph.Features, schema_lib.FeatureSetSchema]:
  """Extracts calendar features from timestamp features of a single feature set."""
  new_values: in_memory_graph.Features = {}
  new_schemas: schema_lib.FeatureSetSchema = {}

  for fname, schema in schemas.items():
    raw_val = values[fname]
    new_values[fname] = raw_val
    new_schemas[fname] = schema

    # Skip non-timestamp features.
    if schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP:
      continue

    if raw_val.dtype == np.object_:
      raise ValueError(
          "extract_calendar_features requires fixed-length timestamp tensors,"
          f" but feature '{fname}' is a variable-length object array."
          " Please run pad_and_cap_timeseries_features first."
      )

    # Determine the group name for the generated calendar feature.
    if schema.group is not None:
      # If the original feature specifies an explicit sequence group, inherit
      # it.
      cal_group = schema.group
    elif schema.is_timeseries:
      # If the original feature is a sequence timestamp without an explicit
      # group, its feature name acts as its implicit sequence group name.
      cal_group = fname
    else:
      # Non-timeseries (scalar) timestamp features do not belong to any sequence
      # group.
      cal_group = None

    for cal_feat in config.features:
      out_fname = f"{fname}_{cal_feat.value}"
      cal_arr = _compute_calendar_feature(raw_val, cal_feat)
      new_values[out_fname] = cal_arr
      new_schemas[out_fname] = schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
          shape=schema.shape,
          is_timeseries=schema.is_timeseries,
          group=cal_group,
      )

  return new_values, new_schemas


def extract_calendar_features(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    config: Optional[CalendarFeatureConfig] = None,
) -> Tuple[in_memory_graph.InMemoryGraph, schema_lib.GraphSchema]:
  """Extracts calendar features (e.g. hour, day_of_week) from timestamp features.

  Requires fixed-length timestamp tensors (e.g., produced after running
  `pad_and_cap_timeseries_features`).

  Usage example:

  ```python
  graph, schema = dgf.transform.pad_and_cap_timeseries_features(
      graph, schema, cap_config
  )
  graph, schema = dgf.transform.extract_calendar_features(graph, schema)
  ```

  Args:
    graph: The input in-memory graph.
    schema: The graph schema containing timestamp features.
    config: Optional `CalendarFeatureConfig`.

  Returns:
    Tuple `(new_graph, new_schema)` containing original and extracted calendar
    features.
  """

  # Default to extracting all calendar features
  if config is None:
    config = CalendarFeatureConfig()

  new_node_sets = {}
  new_ns_schemas = {}

  for ns_name, ns_schema in schema.node_sets.items():
    ns_val = graph.node_sets[ns_name]
    new_vals, new_schemas = _extract_feature_set_calendar_features(
        values=ns_val.features,
        schemas=ns_schema.features,
        config=config,
    )
    new_node_sets[ns_name] = in_memory_graph.InMemoryNodeSet(
        num_nodes=ns_val.num_nodes, features=new_vals
    )
    new_ns_schemas[ns_name] = schema_lib.NodeSchema(features=new_schemas)

  new_edge_sets = {}
  new_es_schemas = {}

  for es_name, es_schema in schema.edge_sets.items():
    es_val = graph.edge_sets[es_name]
    new_vals, new_schemas = _extract_feature_set_calendar_features(
        values=es_val.features,
        schemas=es_schema.features,
        config=config,
    )
    new_edge_sets[es_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=es_val.adjacency, features=new_vals
    )
    new_es_schemas[es_name] = schema_lib.EdgeSchema(
        source=es_schema.source,
        target=es_schema.target,
        features=new_schemas,
    )

  return (
      in_memory_graph.InMemoryGraph(
          node_sets=new_node_sets, edge_sets=new_edge_sets
      ),
      schema_lib.GraphSchema(
          node_sets=new_ns_schemas, edge_sets=new_es_schemas
      ),
  )
