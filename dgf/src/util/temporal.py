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

"""Temporal utilities for timeseries schema metadata extraction."""

import collections
import dataclasses
from typing import Dict, List, Optional, Tuple

from dgf.src.data import schema as schema_lib
import numpy as np


@dataclasses.dataclass(frozen=True)
class TimeseriesGroupSpec:
  """Pre-resolved metadata grouping features that share the same timestamp feature.

  Attributes:
    timestamp_feature_name: Name of the associated timestamp feature. None for
      un-timestamped timeseries.
    feature_names: List of feature names associated with this timestamp.
  """

  timestamp_feature_name: Optional[str]
  feature_names: List[str]


@dataclasses.dataclass(frozen=True)
class TimeseriesSchemaCache:
  """Pre-resolved schema metadata for fast timeseries filtering."""

  node_sets: Dict[str, List[TimeseriesGroupSpec]]
  edge_sets: Dict[str, List[TimeseriesGroupSpec]]
  has_timeseries: bool


def get_creation_time_feature_name(
    schemas: schema_lib.FeatureSetSchema,
) -> Optional[str]:
  """Gets the feature name for entity creation time."""
  for name, sch in schemas.items():
    if sch.is_creation_time and not sch.is_timeseries:
      return name
  return None


def get_edgeset_creation_time_feature_name(
    es_schema: schema_lib.EdgeSchema,
    schema: schema_lib.GraphSchema,
) -> Optional[str]:
  """Gets creation timestamp feature for an edgeset, falling back to connected node creation times."""
  for feat_name, feat_schema in es_schema.features.items():
    if feat_schema.is_creation_time and not feat_schema.is_timeseries:
      return feat_name

  if es_schema.source in schema.node_sets:
    src_ts = get_creation_time_feature_name(
        schema.node_sets[es_schema.source].features
    )
    if src_ts in es_schema.features:
      return src_ts

  if es_schema.target in schema.node_sets:
    tgt_ts = get_creation_time_feature_name(
        schema.node_sets[es_schema.target].features
    )
    if tgt_ts in es_schema.features:
      return tgt_ts

  return None


def get_group_creation_time_feature_name(
    group_name: str, schemas: schema_lib.FeatureSetSchema
) -> Optional[str]:
  """Gets the creation time sequence feature name for a group."""
  for name, sch in schemas.items():
    if sch.is_creation_time and sch.is_timeseries:
      sch_group = sch.group or name
      if sch_group == group_name:
        return name
  return None


def _extract_entity_set_timeseries_specs(
    features: Dict[str, schema_lib.FeatureSchema],
) -> List[TimeseriesGroupSpec]:
  """Extracts and groups timeseries feature specs for a node set or edge set."""
  ts_groups: Dict[Optional[str], List[str]] = collections.defaultdict(list)
  for fname, fschema in features.items():
    if not fschema.is_timeseries:
      continue
    grp = fschema.group or (fname if fschema.is_creation_time else None)
    ts_groups[grp].append(fname)

  specs = []
  for grp_name, fnames in ts_groups.items():
    ts_feat_name = None
    if grp_name is not None:
      ts_feat_name = get_group_creation_time_feature_name(grp_name, features)
    specs.append(
        TimeseriesGroupSpec(
            timestamp_feature_name=ts_feat_name, feature_names=fnames
        )
    )
  return specs


def get_mask_feature_name(
    feature_name: str, schemas: schema_lib.FeatureSetSchema
) -> Optional[str]:
  """Gets the authoritative mask feature name associated with a given feature.

  A mask feature has `semantic=FeatureSemantic.MASK` and explicitly shares the
  same `group` as the feature being masked.

  Args:
    feature_name: Name of the feature to find the mask for.
    schemas: The feature set schema (`Dict[str, FeatureSchema]`).

  Returns:
    The name of the mask feature if found, or None if the feature has no
    associated sequence group or mask in the schema.
  """
  feature_schema = schemas[feature_name]
  ts_group = feature_schema.group
  if ts_group is None:
    return None
  for name, sch in schemas.items():
    if (
        sch.semantic == schema_lib.FeatureSemantic.MASK
        and sch.group == ts_group
    ):
      return name
  return None


def extract_timeseries_schema_cache(
    schema: schema_lib.GraphSchema,
) -> TimeseriesSchemaCache:
  """Extracts and pre-resolves timeseries feature metadata from a schema."""
  node_sets_cache: Dict[str, List[TimeseriesGroupSpec]] = {}
  for ns_name, ns_schema in schema.node_sets.items():
    node_sets_cache[ns_name] = _extract_entity_set_timeseries_specs(
        ns_schema.features
    )

  edge_sets_cache: Dict[str, List[TimeseriesGroupSpec]] = {}
  for es_name, es_schema in schema.edge_sets.items():
    edge_sets_cache[es_name] = _extract_entity_set_timeseries_specs(
        es_schema.features
    )

  has_ts = any(node_sets_cache.values()) or any(edge_sets_cache.values())
  return TimeseriesSchemaCache(
      node_sets=node_sets_cache,
      edge_sets=edge_sets_cache,
      has_timeseries=has_ts,
  )


def get_timeseries_step_shape(
    fschema: schema_lib.FeatureSchema,
) -> Tuple[Optional[int], ...]:
  """Returns per-step feature dimension, excluding leading sequence length shape[0]."""
  if not fschema.is_timeseries:
    raise ValueError("Feature schema must be a timeseries feature.")
  elif fschema.shape is None or not fschema.shape:
    raise ValueError(
        "Timeseries feature schema must have at least 1 dimension (sequence"
        f" length at shape[0]), but got shape={fschema.shape}."
    )
  return fschema.shape[1:]


def with_sequence_length(
    fschema: schema_lib.FeatureSchema,
    seq_len: int,
) -> schema_lib.FeatureSchema:
  """Returns a copy of the schema with shape[0] set to seq_len."""
  step_shape = get_timeseries_step_shape(fschema)
  return dataclasses.replace(
      fschema,
      shape=(seq_len,) + step_shape,
      is_timeseries=True,
  )


def expand_mask_dims(mask: np.ndarray, target: np.ndarray) -> np.ndarray:
  """Expands `mask` dimensions to match `target.ndim` for broadcasting.

  Usage example:
  ```python
    mask_for_where = dgf.src.util.temporal.expand_mask_dims(mask, raw_val)
    result = np.where(mask_for_where, raw_val, fill_value)
  ```

  Args:
    mask: Input boolean mask array.
    target: Target array whose number of dimensions `mask` should match.

  Returns:
    The expanded `mask` array with trailing singleton dimensions added if
    `mask.ndim < target.ndim`, or the original `mask` otherwise.
  """
  if mask.ndim < target.ndim:
    return np.expand_dims(mask, axis=tuple(range(mask.ndim, target.ndim)))
  return mask
