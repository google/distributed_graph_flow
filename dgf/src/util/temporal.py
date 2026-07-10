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
from typing import Dict, List, Optional

from dgf.src.data import schema as schema_lib


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


def _extract_entity_set_timeseries_specs(
    features: Dict[str, schema_lib.FeatureSchema],
) -> List[TimeseriesGroupSpec]:
  """Extracts and groups timeseries feature specs for a node set or edge set."""
  ts_groups: Dict[Optional[str], List[str]] = collections.defaultdict(list)
  for fname, fschema in features.items():
    if not fschema.is_timeseries:
      continue
    # Extract timeseries features associated with a timestamp feature.
    if fschema.timestamps is not None:
      ts_groups[fschema.timestamps].append(fname)
    elif fschema.semantic == schema_lib.FeatureSemantic.TIMESTAMP:
      ts_groups[fname].append(fname)
    else:
      ts_groups[None].append(fname)

  return [
      TimeseriesGroupSpec(timestamp_feature_name=ts_fname, feature_names=fnames)
      for ts_fname, fnames in ts_groups.items()
  ]


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
