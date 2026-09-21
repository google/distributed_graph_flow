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

"""Combines a sequence of DGF Graph Snapshots into a single Temporal Graph."""

import collections
from collections.abc import Collection, Iterable, Sequence
import dataclasses
from typing import Any

from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import graph_snapshots as graph_snapshots_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
from dgf.src.util import log
from dgf.src.util import temporal as temporal_util
import numpy as np

# Feature names reserved for the temporal presence interval of an entity.
_CREATION_TIME_FEATURE_NAME = "creation_time"
_DELETION_TIME_FEATURE_NAME = "deletion_time"


def _concrete_dtype(
    schema_dtype: Any,
    source_arrays: Iterable[np.ndarray],
) -> np.dtype:
  """Resolves an unsized dtype (e.g. `np.bytes_`) to fit `source_arrays`.

  `FeatureFormat.BYTES` maps to `dtype("S")`, which has itemsize 0. Passing it
  to `np.empty` produces an `S1` array that silently truncates values to one
  byte on assignment, so pre-allocated bytes features must take their width
  from the arrays being copied. Fixed-width dtypes are returned unchanged.
  """
  dtype = np.dtype(schema_dtype)
  if dtype.itemsize:
    return dtype
  source_dtypes = [array.dtype for array in source_arrays]
  return np.result_type(*source_dtypes) if source_dtypes else dtype


def _add_interval_schemas(
    features: dict[str, schema_lib.FeatureSchema],
) -> None:
  """Adds creation and deletion time schemas to a feature dictionary."""
  existing_creation_time = temporal_util.creation_time_feature_name(features)
  if existing_creation_time is not None:
    raise ValueError(
        f"Feature '{existing_creation_time}' already has is_creation_time=True."
        f" Cannot add reserved '{_CREATION_TIME_FEATURE_NAME}' feature."
    )
  # TODO(simonmeierhans): Rather than failing, fall back to a non-conflicting
  # feature name (e.g. by suffixing) and log a warning.
  for reserved in (_CREATION_TIME_FEATURE_NAME, _DELETION_TIME_FEATURE_NAME):
    if reserved in features:
      raise ValueError(
          f"Feature name '{reserved}' is reserved for temporal interval"
          " tracking and conflicts with an existing feature in the schema."
      )
  features[_CREATION_TIME_FEATURE_NAME] = schema_lib.FeatureSchema(
      format=schema_lib.FeatureFormat.INTEGER_64,
      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
      is_creation_time=True,
  )
  # Deletion time is realized as a timeseries of length 0 or 1. Registering it
  # as its own sequence timestamp (is_creation_time=True, with a group named
  # after the feature) ensures future deletion times are causally masked when
  # sampling.
  features[_DELETION_TIME_FEATURE_NAME] = schema_lib.FeatureSchema(
      format=schema_lib.FeatureFormat.INTEGER_64,
      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
      is_timeseries=True,
      shape=(None,),
      is_creation_time=True,
      group=_DELETION_TIME_FEATURE_NAME,
  )


# ==============================================================================
# Standalone primitives
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class _EntityOccurrences:
  """Snapshot locations where a single entity (node or edge interval) appears.

  Attributes:
    snapshot_indices: Chronologically ordered snapshot indices `t` where the
      entity is present.
    row_indices: Row index of the entity within snapshot `t`'s feature arrays,
      aligned 1:1 with `snapshot_indices`.
  """

  snapshot_indices: Sequence[int]
  row_indices: Sequence[int]

  def __post_init__(self) -> None:
    if len(self.snapshot_indices) != len(self.row_indices):
      raise ValueError(
          "snapshot_indices and row_indices must have the same length, got"
          f" {len(self.snapshot_indices)} vs {len(self.row_indices)}."
      )

  def __len__(self) -> int:
    return len(self.snapshot_indices)

  @classmethod
  def from_pairs(
      cls, pairs: Sequence[tuple[int, int]]
  ) -> "_EntityOccurrences":
    """Constructs an `_EntityOccurrences` from `(snapshot, row)` pairs."""
    return cls(
        snapshot_indices=[t for t, _ in pairs],
        row_indices=[r for _, r in pairs],
    )


def _build_occurrence_transitions(
    entity_occurrences: Sequence[_EntityOccurrences],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
  """Groups consecutive entity appearances by snapshot pair.

  For each entity, inspects adjacent appearances `(t1, r1) -> (t2, r2)` and
  buckets the row indices by `(t1, t2)`. This lets `_is_feature_static` compare
  all entities sharing a transition in a single slice per snapshot pair.

  Example:
    Entity 0 appears at snapshot 0 (row 5) and snapshot 1 (row 2).
    Entity 1 appears at snapshot 0 (row 3) and snapshot 1 (row 7).

    Input::

      [
          _EntityOccurrences(snapshot_indices=[0, 1], row_indices=[5, 2]),
          _EntityOccurrences(snapshot_indices=[0, 1], row_indices=[3, 7]),
      ]

    Output::

      {(0, 1): (np.array([5, 3]), np.array([2, 7]))}

  Args:
    entity_occurrences: Per-entity snapshot and row index sequences.

  Returns:
    Mapping from `(t1, t2)` snapshot index pairs to `(rows_at_t1, rows_at_t2)`
    1-D `int64` index arrays of equal length.
  """
  transitions_dict = collections.defaultdict(lambda: ([], []))
  for occ in entity_occurrences:
    snaps = occ.snapshot_indices
    rows = occ.row_indices
    for i in range(len(snaps) - 1):
      transitions_dict[(snaps[i], snaps[i + 1])][0].append(rows[i])
      transitions_dict[(snaps[i], snaps[i + 1])][1].append(rows[i + 1])

  return {
      pair: (
          np.array(r1s, dtype=np.int64),
          np.array(r2s, dtype=np.int64),
      )
      for pair, (r1s, r2s) in transitions_dict.items()
  }


def _is_feature_static(
    feature_name: str,
    feature_dtype: Any,
    snapshot_features: Sequence[dict[str, np.ndarray]],
    entity_occurrences: Sequence[_EntityOccurrences],
    transitions: (
        dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None
    ) = None,
) -> bool:
  """Checks whether a feature is constant across each entity's occurrences.

  Compares values pairwise across consecutive appearances of every entity. A
  single differing pair makes the feature dynamic, so it is materialized as a
  timeseries rather than a static column. Floating point values compare with
  `equal_nan=True`, so NaN counts as equal to itself.

  Args:
    feature_name: Name of the feature to check.
    feature_dtype: NumPy dtype of the feature, used to select exact or
      NaN-tolerant comparison.
    snapshot_features: Per-snapshot feature arrays, indexed by snapshot.
    entity_occurrences: Per-entity snapshot and row index sequences.
    transitions: Precomputed output of `_build_occurrence_transitions`. Pass
      it when checking several features over the same entity set to avoid
      rebuilding it per feature; computed on demand when `None`.

  Returns:
    `True` if the feature never changes across any entity's occurrences, or
    if no entity appears in more than one snapshot.
  """
  if transitions is None:
    transitions = _build_occurrence_transitions(entity_occurrences)

  if not transitions:
    return True

  is_inexact = np.issubdtype(feature_dtype, np.inexact)
  for (t1, t2), (rows_t1, rows_t2) in transitions.items():
    vals_t1 = snapshot_features[t1][feature_name][rows_t1]
    vals_t2 = snapshot_features[t2][feature_name][rows_t2]

    if is_inexact:
      if not np.array_equal(vals_t1, vals_t2, equal_nan=True):
        return False
    else:
      if not np.array_equal(vals_t1, vals_t2):
        return False

  return True


def _build_static_features(
    num_entities: int,
    features_schema: dict[str, schema_lib.FeatureSchema],
    snapshot_features: Sequence[dict[str, np.ndarray]],
    entity_occurrences: Sequence[_EntityOccurrences],
) -> tuple[dict[str, np.ndarray], dict[str, schema_lib.FeatureSchema]]:
  """Extracts static features for an entity set.

  Takes each entity's value from the first snapshot in which it appears,
  which is well defined because the caller has already established via
  `_is_feature_static` that the value never changes across occurrences.

  Args:
    num_entities: Number of merged entities in the output entity set.
    features_schema: Static features to materialize, keyed by feature name.
    snapshot_features: Per-snapshot feature arrays, indexed by snapshot.
    entity_occurrences: Per-entity snapshot and row index sequences, each
      ordered by ascending snapshot index.

  Returns:
    A `(features, schema)` pair. Each feature is a dense array of shape
    `(num_entities, *original_shape)`, with `is_timeseries=False` and no
    group in the schema. Both are empty if `features_schema` is empty.
  """
  if not features_schema:
    return {}, {}

  first_seen_dict = collections.defaultdict(lambda: ([], []))
  for entity_idx, occ in enumerate(entity_occurrences):
    first_t = occ.snapshot_indices[0]
    first_r = occ.row_indices[0]
    first_seen_dict[first_t][0].append(entity_idx)
    first_seen_dict[first_t][1].append(first_r)
  first_seen = {
      t: (np.array(e_idxs, dtype=np.int64), np.array(r_idxs, dtype=np.int64))
      for t, (e_idxs, r_idxs) in first_seen_dict.items()
  }

  generated_features: dict[str, np.ndarray] = {}
  generated_schema: dict[str, schema_lib.FeatureSchema] = {}

  for feature_name, original_feature in features_schema.items():
    feature_dtype = _concrete_dtype(
        feature_format.FEATURE_FORMAT_TO_NP_DTYPE[original_feature.format],
        (snapshot_features[t][feature_name] for t in first_seen),
    )

    shape = (
        (num_entities, *original_feature.shape)
        if original_feature.shape
        else (num_entities,)
    )
    # pyrefly: ignore[no-matching-overload]
    feature_array = np.empty(shape, dtype=feature_dtype)
    for t, (entity_idxs, row_idxs) in first_seen.items():
      feature_array[entity_idxs] = snapshot_features[t][feature_name][row_idxs]

    generated_features[feature_name] = feature_array
    generated_schema[feature_name] = dataclasses.replace(
        original_feature,
        is_timeseries=False,
        is_creation_time=False,
        group=None,
    )

  return generated_features, generated_schema


def _build_timeseries_features(
    entity_name: str,
    num_entities: int,
    features_schema: dict[str, schema_lib.FeatureSchema],
    snapshot_features: Sequence[dict[str, np.ndarray]],
    entity_occurrences: Sequence[_EntityOccurrences],
    timestamps: np.ndarray,
    reserved_feature_names: Collection[str],
) -> tuple[dict[str, np.ndarray], dict[str, schema_lib.FeatureSchema]]:
  """Extracts dynamic timeseries feature sequences for an entity set.

  Each feature becomes a ragged `object` array with one variable-length
  subarray per entity, holding that entity's values in snapshot order. A
  companion `{entity_name}_group_timestamp` feature carries the matching
  timestamps so the sampler can causally mask the sequence.

  Args:
    entity_name: Node or edge set name, used to derive the group and
      timestamp feature names.
    num_entities: Number of merged entities in the output entity set.
    features_schema: Dynamic features to materialize, keyed by feature name.
    snapshot_features: Per-snapshot feature arrays, indexed by snapshot.
    entity_occurrences: Per-entity snapshot and row index sequences.
    timestamps: 1-D `int64` array of snapshot timestamps.
    reserved_feature_names: Names the generated timestamp feature must not
      collide with.

  Returns:
    A `(features, schema)` pair. Every entry is an `object` array of
    per-entity subarrays, with `shape=(None, *original_shape)` and
    `group={entity_name}_group` in the schema. Both are empty if
    `features_schema` is empty.

  Raises:
    ValueError: If the generated timestamp feature name is already reserved.
  """
  if not features_schema:
    return {}, {}

  group_name = f"{entity_name}_group"
  timestamp_feature_name = f"{entity_name}_group_timestamp"

  if timestamp_feature_name in reserved_feature_names:
    raise ValueError(
        f"Timestamp feature name '{timestamp_feature_name}' conflicts with an"
        " existing or reserved feature in the schema."
    )

  time_arr = np.empty(num_entities, dtype=object)
  for group_idx, occ in enumerate(entity_occurrences):
    time_arr[group_idx] = timestamps[
        np.asarray(occ.snapshot_indices, dtype=np.int64)
    ]

  generated_features: dict[str, np.ndarray] = {
      timestamp_feature_name: time_arr
  }
  generated_schema: dict[str, schema_lib.FeatureSchema] = {
      timestamp_feature_name: schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.TIMESTAMP,
          shape=(None,),
          is_timeseries=True,
          is_creation_time=True,
          group=group_name,
      )
  }

  for feature_name, original_feature in features_schema.items():
    feature_dtype = _concrete_dtype(
        feature_format.FEATURE_FORMAT_TO_NP_DTYPE[original_feature.format],
        (
            snapshot_feature[feature_name]
            for snapshot_feature in snapshot_features
            if feature_name in snapshot_feature
        ),
    )
    feature_array = np.empty(num_entities, dtype=object)

    elem_shape = original_feature.shape or ()
    for group_idx, occ in enumerate(entity_occurrences):
      # Allocate from the declared element shape rather than letting np.array()
      # infer it from the values: inference silently merges elements whose
      # shapes happen to be compatible, whereas assigning row by row raises.
      # pyrefly: ignore[no-matching-overload]
      arr = np.empty((len(occ), *elem_shape), dtype=feature_dtype)
      for i, (t, row_idx) in enumerate(
          zip(occ.snapshot_indices, occ.row_indices)
      ):
        arr[i] = snapshot_features[t][feature_name][row_idx]
      feature_array[group_idx] = arr

    new_shape = (
        (None, *original_feature.shape) if original_feature.shape else (None,)
    )
    generated_features[feature_name] = feature_array
    generated_schema[feature_name] = dataclasses.replace(
        original_feature,
        shape=new_shape,
        is_timeseries=True,
        is_creation_time=False,
        group=group_name,
    )

  return generated_features, generated_schema


def _process_features(
    entity_name: str,
    num_entities: int,
    candidate_features_schema: dict[str, schema_lib.FeatureSchema],
    snapshot_features: Sequence[dict[str, np.ndarray]],
    entity_occurrences: Sequence[_EntityOccurrences],
    timestamps: np.ndarray,
    reserved_feature_names: Collection[str],
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, schema_lib.FeatureSchema]]:
  """Partitions and builds static and timeseries features for an entity set.

  Classifies each candidate feature as static or dynamic via
  `_is_feature_static`, then delegates to `_build_static_features` and
  `_build_timeseries_features` and merges their outputs. The occurrence
  transitions are computed once here and reused across every feature.

  Args:
    entity_name: Node or edge set name, used to derive the timeseries group
      name and for logging.
    num_entities: Number of merged entities in the output entity set.
    candidate_features_schema: Features to classify, keyed by feature name.
    snapshot_features: Per-snapshot feature arrays, indexed by snapshot.
    entity_occurrences: Per-entity snapshot and row index sequences.
    timestamps: 1-D `int64` array of snapshot timestamps.
    reserved_feature_names: Names that the generated timeseries timestamp
      feature must not collide with.
    verbose: Whether to log each feature's static/dynamic classification.

  Returns:
    A `(features, schema)` pair combining the static and timeseries features.
  """
  static_features_schema: dict[str, schema_lib.FeatureSchema] = {}
  dynamic_features_schema: dict[str, schema_lib.FeatureSchema] = {}

  transitions = _build_occurrence_transitions(entity_occurrences)

  for feature_name, feat_schema in candidate_features_schema.items():
    feature_dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[
        feat_schema.format
    ]
    if _is_feature_static(
        feature_name,
        feature_dtype,
        snapshot_features,
        entity_occurrences,
        transitions=transitions,
    ):
      static_features_schema[feature_name] = feat_schema
      if verbose:
        log.info(
            "Feature '%s' in entity set '%s' is static (not a timeseries).",
            feature_name,
            entity_name,
        )
    else:
      dynamic_features_schema[feature_name] = feat_schema
      if verbose:
        log.info(
            "Feature '%s' in entity set '%s' is dynamic (detected as a"
            " timeseries).",
            feature_name,
            entity_name,
        )

  processed_features, processed_schema = _build_static_features(
      num_entities=num_entities,
      features_schema=static_features_schema,
      snapshot_features=snapshot_features,
      entity_occurrences=entity_occurrences,
  )

  all_reserved_names = set(reserved_feature_names) | set(
      processed_schema.keys()
  )
  ts_features, ts_schema = _build_timeseries_features(
      entity_name=entity_name,
      num_entities=num_entities,
      features_schema=dynamic_features_schema,
      snapshot_features=snapshot_features,
      entity_occurrences=entity_occurrences,
      timestamps=timestamps,
      reserved_feature_names=all_reserved_names,
  )
  processed_features.update(ts_features)
  processed_schema.update(ts_schema)

  return processed_features, processed_schema


def _build_edge_pair_timeline(
    edge_set_name: str,
    edge_set_schema: schema_lib.EdgeSchema,
    snapshot_graphs: Sequence[in_memory_graph_lib.InMemoryGraph],
    src_primary_key: str,
    dst_primary_key: str,
    src_id_to_group_idx: dict[object, int],
    dst_id_to_group_idx: dict[object, int],
    timestamps: np.ndarray,
) -> dict[tuple[int, int], dict[int, int]]:
  """Builds a timeline mapping (u, v) node pairs to {snapshot_idx: row_idx}."""
  src_node_set = edge_set_schema.source
  dst_node_set = edge_set_schema.target
  # Maps each (source, destination) node index pair to the snapshots in which
  # that edge is present: {snapshot index: row index within that snapshot's
  # edge set}. For example, {(0, 1): {0: 5, 1: 2}} means the edge 0 -> 1 exists
  # in snapshots 0 and 1, stored at row 5 and row 2 respectively.
  pair_timeline: dict[tuple[int, int], dict[int, int]] = (
      collections.defaultdict(dict)
  )
  for t, snapshot_graph in enumerate(snapshot_graphs):
    snapshot_edge_set = snapshot_graph.edge_sets[edge_set_name]
    snapshot_src_ids = (
        snapshot_graph.node_sets[src_node_set].features[src_primary_key]
    )
    snapshot_dst_ids = (
        snapshot_graph.node_sets[dst_node_set].features[dst_primary_key]
    )
    adj = snapshot_edge_set.adjacency

    if adj.shape[1] == 0:
      continue

    src_map = np.array(
        [src_id_to_group_idx[node_id] for node_id in snapshot_src_ids],
        dtype=np.int64,
    )
    if src_node_set == dst_node_set:
      dst_map = src_map
    else:
      dst_map = np.array(
          [dst_id_to_group_idx[node_id] for node_id in snapshot_dst_ids],
          dtype=np.int64,
      )

    u_arr = src_map[adj[0]]
    v_arr = dst_map[adj[1]]

    for i, (u, v) in enumerate(zip(u_arr, v_arr)):
      pair = (int(u), int(v))
      if t in pair_timeline[pair]:
        raise ValueError(
            f"Multigraph detected in edge set '{edge_set_name}' at snapshot"
            f" timestamp {timestamps[t]} between node indices ({u}, {v})."
            " Multigraphs are not yet supported."
        )
      pair_timeline[pair][t] = i

  return pair_timeline


def _extract_edge_intervals(
    pair_timeline: dict[tuple[int, int], dict[int, int]],
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[_EntityOccurrences]]:
  """Extracts contiguous edge intervals and occurrences from pair timelines.

  Returns:
    (adj_arr, creation_times, deletion_times, occurrences)
  """
  num_timesteps = len(timestamps)
  src_indices: list[int] = []
  dst_indices: list[int] = []
  edge_creation_times: list[int] = []
  edge_deletion_objs: list[np.ndarray] = []
  raw_edge_occurrences: list[list[tuple[int, int]]] = []

  for (u, v), snapshot_edges in sorted(pair_timeline.items()):
    prev_t: int | None = None
    for t, row_idx in snapshot_edges.items():
      if prev_t is None or t > prev_t + 1:
        if prev_t is not None:
          edge_deletion_objs.append(
              np.array([timestamps[prev_t + 1]], dtype=np.int64)
          )
        src_indices.append(u)
        dst_indices.append(v)
        edge_creation_times.append(int(timestamps[t]))
        raw_edge_occurrences.append([])
      raw_edge_occurrences[-1].append((t, row_idx))
      prev_t = t

    if prev_t is not None:
      del_ts = [timestamps[prev_t + 1]] if prev_t < num_timesteps - 1 else []
      edge_deletion_objs.append(np.array(del_ts, dtype=np.int64))

  num_intervals = len(src_indices)
  adj_arr = np.empty((2, num_intervals), dtype=np.int64)
  adj_arr[0] = src_indices
  adj_arr[1] = dst_indices

  deletion_arr = np.empty(num_intervals, dtype=object)
  for i, arr in enumerate(edge_deletion_objs):
    deletion_arr[i] = arr

  creation_arr = np.array(edge_creation_times, dtype=np.int64)
  edge_occurrences = [
      _EntityOccurrences.from_pairs(pairs) for pairs in raw_edge_occurrences
  ]

  return adj_arr, creation_arr, deletion_arr, edge_occurrences


# ==============================================================================
# Graph-level orchestration
# ==============================================================================


def _process_node_set(
    ns_name: str,
    ns_schema: schema_lib.NodeSchema,
    snapshot_graphs: Sequence[in_memory_graph_lib.InMemoryGraph],
    timestamps: np.ndarray,
    verbose: bool = False,
) -> tuple[in_memory_graph_lib.InMemoryNodeSet, schema_lib.NodeSchema]:
  """Processes node IDs, intervals, and dynamic features for a node set."""
  num_timesteps = len(timestamps)
  primary_key_name = analyse_schema_lib.primary_feature(ns_name, ns_schema)

  node_to_idx: dict[object, int] = {}
  global_node_ids: list[object] = []
  raw_node_occurrences: list[list[tuple[int, int]]] = []

  for t, snapshot_graph in enumerate(snapshot_graphs):
    snapshot_node_ids = (
        snapshot_graph.node_sets[ns_name].features[primary_key_name]
    )
    for row_idx, node_id in enumerate(snapshot_node_ids):
      if node_id not in node_to_idx:
        node_to_idx[node_id] = len(global_node_ids)
        global_node_ids.append(node_id)
        raw_node_occurrences.append([(t, row_idx)])
      else:
        raw_node_occurrences[node_to_idx[node_id]].append((t, row_idx))

  entity_occurrences = [
      _EntityOccurrences.from_pairs(pairs) for pairs in raw_node_occurrences
  ]

  num_nodes = len(global_node_ids)
  creation_time = np.empty(num_nodes, dtype=np.int64)
  deletion_time = np.empty(num_nodes, dtype=object)

  for group_idx, occ in enumerate(entity_occurrences):
    first_t = occ.snapshot_indices[0]
    last_t = occ.snapshot_indices[-1]
    creation_time[group_idx] = timestamps[first_t]
    if last_t < num_timesteps - 1:
      del_ts = [timestamps[last_t + 1]]
    else:
      del_ts = []
    deletion_time[group_idx] = np.array(del_ts, dtype=np.int64)

  primary_key_dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[
      ns_schema.features[primary_key_name].format
  ]
  node_features = {
      primary_key_name: np.array(global_node_ids, dtype=primary_key_dtype),
      _CREATION_TIME_FEATURE_NAME: creation_time,
      _DELETION_TIME_FEATURE_NAME: deletion_time,
  }

  updated_ns_features = dict(ns_schema.features)
  _add_interval_schemas(updated_ns_features)

  candidate_features_schema = {
      feature_name: ns_schema.features[feature_name]
      for feature_name in ns_schema.features
      if feature_name
      not in (
          primary_key_name,
          _CREATION_TIME_FEATURE_NAME,
          _DELETION_TIME_FEATURE_NAME,
      )
  }
  snapshot_features = [
      snapshot_graph.node_sets[ns_name].features
      for snapshot_graph in snapshot_graphs
  ]

  entity_features, entity_feature_schemas = _process_features(
      entity_name=ns_name,
      num_entities=num_nodes,
      candidate_features_schema=candidate_features_schema,
      snapshot_features=snapshot_features,
      entity_occurrences=entity_occurrences,
      timestamps=timestamps,
      reserved_feature_names=updated_ns_features.keys(),
      verbose=verbose,
  )
  node_features.update(entity_features)
  updated_ns_features.update(entity_feature_schemas)

  return (
      in_memory_graph_lib.InMemoryNodeSet(
          num_nodes=num_nodes, features=node_features
      ),
      schema_lib.NodeSchema(features=updated_ns_features),
  )


def _process_edge_set(
    edge_set_name: str,
    edge_set_schema: schema_lib.EdgeSchema,
    snapshot_graphs: Sequence[in_memory_graph_lib.InMemoryGraph],
    temporal_node_sets: dict[str, in_memory_graph_lib.InMemoryNodeSet],
    timestamps: np.ndarray,
    node_schemas: dict[str, schema_lib.NodeSchema],
    verbose: bool = False,
) -> tuple[in_memory_graph_lib.InMemoryEdgeSet, schema_lib.EdgeSchema]:
  """Processes edge intervals, creation/deletion times, and dynamic features."""
  src_node_set = edge_set_schema.source
  dst_node_set = edge_set_schema.target
  src_primary_key = analyse_schema_lib.primary_feature(
      src_node_set, node_schemas[src_node_set]
  )
  dst_primary_key = analyse_schema_lib.primary_feature(
      dst_node_set, node_schemas[dst_node_set]
  )

  src_id_to_group_idx = {
      node_id: i
      for i, node_id in enumerate(
          temporal_node_sets[src_node_set].features[src_primary_key]
      )
  }
  if src_node_set == dst_node_set:
    dst_id_to_group_idx = src_id_to_group_idx
  else:
    dst_id_to_group_idx = {
        node_id: i
        for i, node_id in enumerate(
            temporal_node_sets[dst_node_set].features[dst_primary_key]
        )
    }

  # 1. Map snapshot edges to global node index pairs across all snapshots.
  pair_timeline = _build_edge_pair_timeline(
      edge_set_name=edge_set_name,
      edge_set_schema=edge_set_schema,
      snapshot_graphs=snapshot_graphs,
      src_primary_key=src_primary_key,
      dst_primary_key=dst_primary_key,
      src_id_to_group_idx=src_id_to_group_idx,
      dst_id_to_group_idx=dst_id_to_group_idx,
      timestamps=timestamps,
  )

  # 2. Extract contiguous presence intervals and presence occurrences.
  adj_arr, creation_time, deletion_time, edge_occurrences = (
      _extract_edge_intervals(pair_timeline, timestamps)
  )
  num_intervals = adj_arr.shape[1]

  edge_features = {
      _CREATION_TIME_FEATURE_NAME: creation_time,
      _DELETION_TIME_FEATURE_NAME: deletion_time,
  }

  # 3. Process static and timeseries features.
  updated_es_features = dict(edge_set_schema.features)
  _add_interval_schemas(updated_es_features)

  candidate_edge_schema = {
      feature_name: edge_set_schema.features[feature_name]
      for feature_name in edge_set_schema.features
      if feature_name
      not in (_CREATION_TIME_FEATURE_NAME, _DELETION_TIME_FEATURE_NAME)
  }
  snapshot_edge_features = [
      snapshot_graph.edge_sets[edge_set_name].features
      for snapshot_graph in snapshot_graphs
  ]

  entity_features, entity_feature_schemas = _process_features(
      entity_name=edge_set_name,
      num_entities=num_intervals,
      candidate_features_schema=candidate_edge_schema,
      snapshot_features=snapshot_edge_features,
      entity_occurrences=edge_occurrences,
      timestamps=timestamps,
      reserved_feature_names=updated_es_features.keys(),
      verbose=verbose,
  )
  edge_features.update(entity_features)
  updated_es_features.update(entity_feature_schemas)

  return (
      in_memory_graph_lib.InMemoryEdgeSet(
          adjacency=adj_arr, features=edge_features
      ),
      schema_lib.EdgeSchema(
          source=edge_set_schema.source,
          target=edge_set_schema.target,
          features=updated_es_features,
      ),
  )


def combine_graph_snapshots(
    snapshots: graph_snapshots_lib.GraphSnapshots,
    verbose: bool = False,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Combines chronologically ordered graph snapshots into a Temporal Graph.

  Aligns node entities across all snapshots by primary ID, tracks entity and
  edge presence intervals with creation and deletion times, and converts
  time-varying node and edge features into structured timeseries.

  Note: Edges that re-appear after a deletion are treated as a new edge,
  whereas nodes that re-appear are merged. This behavior is consistent with
  typical usecases, where edges often lack identifiers.

  Usage example:

  ```python
  snapshots = dgf.io.read_graph_snapshots("/path/to/snapshot_dataset")
  graph, schema = dgf.transform.combine_graph_snapshots(snapshots)
  ```

  Args:
    snapshots: The graph snapshots to combine, sorted by increasing timestamp.
    verbose: If True, print progress information.

  Returns:
    A tuple (temporal_graph, temporal_schema) where temporal_graph is an
    InMemoryGraph and temporal_schema is a GraphSchema.
  """
  base_schema = snapshots.schema
  if temporal_util.schema_has_timeseries_features(base_schema):
    raise ValueError(
        "Snapshot graphs containing a timeseries are not supported yet."
    )

  snapshot_graphs = snapshots.graphs
  timestamps = snapshots.timestamps

  temporal_node_sets: dict[str, in_memory_graph_lib.InMemoryNodeSet] = {}
  temporal_node_schemas: dict[str, schema_lib.NodeSchema] = {}

  for node_set_name, node_set_schema in base_schema.node_sets.items():
    if verbose:
      log.info("Processing node set '%s'...", node_set_name)
    node_set, node_schema = _process_node_set(
        node_set_name,
        node_set_schema,
        snapshot_graphs,
        timestamps,
        verbose=verbose,
    )
    temporal_node_sets[node_set_name] = node_set
    temporal_node_schemas[node_set_name] = node_schema

  temporal_edge_sets: dict[str, in_memory_graph_lib.InMemoryEdgeSet] = {}
  temporal_edge_schemas: dict[str, schema_lib.EdgeSchema] = {}

  for edge_set_name, edge_set_schema in base_schema.edge_sets.items():
    if verbose:
      log.info("Processing edge set '%s'...", edge_set_name)
    edge_set, edge_schema = _process_edge_set(
        edge_set_name=edge_set_name,
        edge_set_schema=edge_set_schema,
        snapshot_graphs=snapshot_graphs,
        temporal_node_sets=temporal_node_sets,
        timestamps=timestamps,
        node_schemas=base_schema.node_sets,
        verbose=verbose,
    )
    temporal_edge_sets[edge_set_name] = edge_set
    temporal_edge_schemas[edge_set_name] = edge_schema

  if verbose:
    log.info(
        "Successfully converted %d snapshot(s) to temporal graph.",
        len(snapshots),
    )

  temporal_graph = in_memory_graph_lib.InMemoryGraph(
      node_sets=temporal_node_sets,
      edge_sets=temporal_edge_sets,
  )
  temporal_schema = schema_lib.GraphSchema(
      node_sets=temporal_node_schemas,
      edge_sets=temporal_edge_schemas,
  )
  return temporal_graph, temporal_schema

