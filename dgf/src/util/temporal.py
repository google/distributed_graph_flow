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
from typing import Dict, List

from dgf.src.data import schema as schema_lib


@dataclasses.dataclass(frozen=True)
class TimeseriesGroupSpec:
  """Pre-resolved metadata grouping features that share the same timestamp feature.

  Attributes:
    timestamp_feature_name: Name of the timestamp feature.
    feature_names: List of feature names associated with this timestamp.
  """

  timestamp_feature_name: str
  feature_names: List[str]


@dataclasses.dataclass(frozen=True)
class TimeseriesSchemaCache:
  """Pre-resolved schema metadata for fast timeseries filtering."""

  node_sets: Dict[str, List[TimeseriesGroupSpec]]
  edge_sets: Dict[str, List[TimeseriesGroupSpec]]


def _extract_entity_set_timeseries_specs(
    features: Dict[str, schema_lib.FeatureSchema],
) -> List[TimeseriesGroupSpec]:
  """Extracts and groups timeseries feature specs for a node set or edge set."""
  ts_groups: Dict[str, List[str]] = collections.defaultdict(list)
  for fname, fschema in features.items():
    if not fschema.is_timeseries:
      continue
    # Extract timeseries features associated with a timestamp feature.
    if fschema.timestamps is not None:
      ts_groups[fschema.timestamps].append(fname)
    # Extract timestamp features themselves so they are sliced in sync.
    elif fschema.semantic == schema_lib.FeatureSemantic.TIMESTAMP:
      ts_groups[fname].append(fname)

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
    specs = _extract_entity_set_timeseries_specs(ns_schema.features)
    if specs:
      node_sets_cache[ns_name] = specs

  edge_sets_cache: Dict[str, List[TimeseriesGroupSpec]] = {}
  for es_name, es_schema in schema.edge_sets.items():
    specs = _extract_entity_set_timeseries_specs(es_schema.features)
    if specs:
      edge_sets_cache[es_name] = specs

  return TimeseriesSchemaCache(
      node_sets=node_sets_cache, edge_sets=edge_sets_cache
  )
