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

from typing import List, Optional, Tuple, Union

from dgf.src.data import in_memory_graph
from dgf.src.util import temporal as temporal_util
import numpy as np

# TODO(simonmeierhans): Improve performance ofhandling of static shaped
# timeseries.


def _compute_group_slices(
    timestamp_values: np.ndarray,
    node_idxs: np.ndarray,
    max_timeseries_len: int,
    target_timestamp: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes (start_indices, end_indices) for a sequence group on selected nodes.

  Args:
    timestamp_values: Timestamp sequence feature array indexed by `node_idxs`
      (array of integer timestamps or object array of 1D arrays).
    node_idxs: 1D array of selected node/entity indices.
    max_timeseries_len: Positive integer cap on sequence steps to retain.
    target_timestamp: Optional causal cutoff timestamp (`int`). If None, slices
      up to the end of each sequence.

  Returns:
    A tuple `(start_indices, end_indices)` of 1D int64 arrays.
  """
  if max_timeseries_len <= 0:
    raise ValueError("max_timeseries_len must be positive")

  num_entities = len(node_idxs)
  if num_entities == 0:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

  start_indices = np.empty(num_entities, dtype=np.int64)
  end_indices = np.empty(num_entities, dtype=np.int64)

  for i, idx in enumerate(node_idxs):
    times = timestamp_values[idx]
    if target_timestamp is not None:
      end_idx = int(np.searchsorted(times, target_timestamp, side="right"))
    else:
      end_idx = len(times)
    start_idx = max(0, end_idx - max_timeseries_len)
    start_indices[i] = start_idx
    end_indices[i] = end_idx

  return start_indices, end_indices


def _crop_timeseries(
    feature_values: np.ndarray,
    node_idxs: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
) -> np.ndarray:
  """Extracts and crops timeseries features directly using start and end indices."""
  num_entities = len(node_idxs)
  if num_entities == 0:
    return np.empty(0, dtype=object)

  target_arr = np.empty(num_entities, dtype=object)
  for i, idx in enumerate(node_idxs):
    target_arr[i] = feature_values[idx][start_indices[i] : end_indices[i]]
  return target_arr


def _process_entity_set_timeseries(
    target_val: Union[
        in_memory_graph.InMemoryNodeSet, in_memory_graph.InMemoryEdgeSet
    ],
    source_val: Union[
        in_memory_graph.InMemoryNodeSet, in_memory_graph.InMemoryEdgeSet
    ],
    node_idxs: np.ndarray,
    ts_specs: List[temporal_util.TimeseriesGroupSpec],
    max_timeseries_len: int,
    target_timestamp: Optional[int] = None,
) -> None:
  """Causally filters and/or clips timeseries features in place or from source into target."""
  timeseries_features = set()
  for group in ts_specs:
    if group.timestamp_feature_name is not None:
      ts_val = source_val.features[group.timestamp_feature_name]
      start_indices, end_indices = _compute_group_slices(
          timestamp_values=ts_val,
          node_idxs=node_idxs,
          max_timeseries_len=max_timeseries_len,
          target_timestamp=target_timestamp,
      )
      for fname in group.feature_names:
        timeseries_features.add(fname)
        target_val.features[fname] = _crop_timeseries(
            feature_values=source_val.features[fname],
            node_idxs=node_idxs,
            start_indices=start_indices,
            end_indices=end_indices,
        )
    else:
      for fname in group.feature_names:
        timeseries_features.add(fname)
        val = source_val.features[fname]
        if val.dtype == object:
          target_val.features[fname] = np.array(
              [elem[-max_timeseries_len:] for elem in val[node_idxs]],
              dtype=object,
          )
        else:
          target_val.features[fname] = val[node_idxs, -max_timeseries_len:]

  for fname, full_val in source_val.features.items():
    if fname not in timeseries_features:
      target_val.features[fname] = full_val[node_idxs]


def extract_features_timeseries(
    graph: in_memory_graph.InMemoryGraph,
    timeseries_schema_cache: temporal_util.TimeseriesSchemaCache,
    max_timeseries_len: int = 32,
    target_timestamp: Optional[int] = None,
    source_graph: Optional[in_memory_graph.InMemoryGraph] = None,
) -> None:
  """In-place filters and/or extracts `is_timeseries=True` features for `graph`.

  - If `source_graph` is provided: extracts features directly from 
    `source_graph` for each node set based on the `"#idx"` row indices in a
    single pass. `graph.node_sets` must contain `"#idx"` with 0-based node row
    indices into `source_graph.node_sets`.
  - If `source_graph` is None: modifies `graph` in place.

  Note: This function modifies `graph` in place.

  Args:
    graph: The in-memory graph to be modified in place. When `source_graph` is
      provided, each node set in `graph` must contain `"#idx"`.
    timeseries_schema_cache: A pre-computed `TimeseriesSchemaCache`.
    max_timeseries_len: Positive integer cap on sequence steps to retain.
    target_timestamp: Optional causal cutoff timestamp (`int`).
    source_graph: Optional source graph to extract features from.
  """
  if max_timeseries_len <= 0:
    raise ValueError("max_timeseries_len must be positive")

  if source_graph is not None:
    for ns_name, target_ns in graph.node_sets.items():
      source_ns = source_graph.node_sets[ns_name]
      if "#idx" not in target_ns.features:
        raise ValueError(
            f"NodeSet '{ns_name}' in sampled graph is missing required '#idx' "
            "feature mapping back to source_graph."
        )
      node_idxs = target_ns.features["#idx"]
      ts_specs = timeseries_schema_cache.node_sets.get(ns_name, [])
      _process_entity_set_timeseries(
          target_val=target_ns,
          source_val=source_ns,
          node_idxs=node_idxs,
          ts_specs=ts_specs,
          max_timeseries_len=max_timeseries_len,
          target_timestamp=target_timestamp,
      )
  else:
    # Process Node Sets
    for ns_name, ts_specs in timeseries_schema_cache.node_sets.items():
      ns_val = graph.node_sets[ns_name]
      if ns_val.num_nodes:
        _process_entity_set_timeseries(
            target_val=ns_val,
            source_val=ns_val,
            node_idxs=np.arange(ns_val.num_nodes),
            ts_specs=ts_specs,
            max_timeseries_len=max_timeseries_len,
            target_timestamp=target_timestamp,
        )

    # Process Edge Sets
    for es_name, ts_specs in timeseries_schema_cache.edge_sets.items():
      es_val = graph.edge_sets[es_name]
      if es_val.num_edges():
        _process_entity_set_timeseries(
            target_val=es_val,
            source_val=es_val,
            node_idxs=np.arange(es_val.num_edges()),
            ts_specs=ts_specs,
            max_timeseries_len=max_timeseries_len,
            target_timestamp=target_timestamp,
        )
