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

"""Temporal sampling utilities for filtering and slicing timeseries features."""

from typing import List, Optional, Union

from dgf.src.data import in_memory_graph
from dgf.src.util import temporal as temporal_util
import numpy as np


def _filter_entity_set_timeseries(
    entity_val: Union[
        in_memory_graph.InMemoryNodeSet, in_memory_graph.InMemoryEdgeSet
    ],
    ts_specs: List[temporal_util.TimeseriesGroupSpec],
    num_entities: int,
    target_timestamp: int,
    max_timeseries_len: Optional[int],
) -> None:
  """Causally slices pre-scanned timeseries features for a node set or edge set in place."""
  for group in ts_specs:
    ts_val = entity_val.features[group.timestamp_feature_name]
    feature_names = group.feature_names
    feature_arrays = [entity_val.features[fname] for fname in feature_names]
    sliced_target = {
        fname: np.empty(num_entities, dtype=np.object_)
        for fname in feature_names
    }
    target_lists = [sliced_target[fname] for fname in feature_names]

    # TODO(mesimon): Move into C++ for performance.
    for idx in range(num_entities):
      times = ts_val[idx]
      end_idx = np.searchsorted(times, target_timestamp, side="right")
      start_idx = (
          max(0, end_idx - max_timeseries_len)
          if max_timeseries_len is not None
          else 0
      )
      slc = slice(start_idx, end_idx)

      for feat_arr, target_arr in zip(feature_arrays, target_lists):
        target_arr[idx] = feat_arr[idx][slc]

    for fname, arr in sliced_target.items():
      entity_val.features[fname] = arr


def filter_timeseries_by_timestamp(
    graph: in_memory_graph.InMemoryGraph,
    schema_cache: temporal_util.TimeseriesSchemaCache,
    target_timestamp: Union[int, np.integer],
    max_timeseries_len: Optional[int] = None,
) -> None:
  """In-place filters and caps `is_timeseries=True` features by a causal cutoff.

  For each node or edge with `is_timeseries=True` features (such as `time` and
  `signal`), this function identifies valid causal sequence indices where the
  timestamp is less than or equal to `target_timestamp`. Any future sequence
  entries (`time > target_timestamp`) are stripped in place from all timeseries
  feature arrays. Optionally caps the sequence to the most recent
  `max_timeseries_len` observations.

  Note: This function modifies `graph` in place.

  Usage example:

  ```python
  cache = dgf.util.temporal.extract_timeseries_schema_cache(schema)
  dgf.sampling.temporal.filter_timeseries_by_timestamp(
      graph=subgraph,
      schema_cache=cache,
      target_timestamp=1680000000,
      max_timeseries_len=30,
  )
  ```

  Args:
    graph: The input in-memory graph (or sampled subgraph) to be modified in
      place.
    schema_cache: A pre-computed `TimeseriesSchemaCache` derived from the graph
      schema via `extract_timeseries_schema_cache`.
    target_timestamp: The causal cutoff timestamp (`int`).
    max_timeseries_len: Optional integer cap on the number of causal sequence
      steps to retain. If a node or edge has more than `max_timeseries_len`
      causal steps, only the most recent `max_timeseries_len` steps are kept.
  """

  # Process Node Sets
  for ns_name, ts_specs in schema_cache.node_sets.items():
    ns_val = graph.node_sets.get(ns_name)
    if ns_val and ns_val.num_nodes:
      _filter_entity_set_timeseries(
          ns_val,
          ts_specs,
          ns_val.num_nodes,
          target_timestamp=target_timestamp,
          max_timeseries_len=max_timeseries_len,
      )

  # Process Edge Sets
  for es_name, ts_specs in schema_cache.edge_sets.items():
    es_val = graph.edge_sets.get(es_name)
    if es_val and es_val.num_edges():
      _filter_entity_set_timeseries(
          es_val,
          ts_specs,
          es_val.num_edges(),
          target_timestamp=target_timestamp,
          max_timeseries_len=max_timeseries_len,
      )
