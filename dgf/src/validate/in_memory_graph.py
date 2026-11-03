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

"""Validate an in-memory graph object."""

import collections
from collections.abc import Iterator, Sequence
from dgf.src.analyse import schema as analyse_schema
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
from dgf.src.validate import validate as validate_lib
import numpy as np

Issue = validate_lib.Issue

KEY_ID = "#id"


def _object_feature_data_issues(
    feature_name: str,
    feature_data: np.ndarray,
    feature_schema: schema_lib.FeatureSchema,
    expected_dtype: type[np.generic],
    expected_shape: Sequence[int | None],
    source: str,
) -> Iterator[validate_lib.Issue]:
  """Yields data issues for a `dtype=object` feature array."""
  if len(expected_shape) < 1 or expected_shape[0] is not None:
    yield Issue.error(
        f"The feature {feature_name!r} in {source} is stored as an object"
        " (`dtype=object`), which is only permitted for variable-length"
        " features. However, the schema defines a fixed shape of"
        f" {expected_shape}. If your feature is a string, make sure to cast"
        " it as a np.bytes_."
    )
    return
  for row_idx, row in enumerate(feature_data):
    if not isinstance(row, np.ndarray):
      yield Issue.error(
          f"The variable-length feature {feature_name!r} in {source} has a"
          f" non-ndarray row at index {row_idx} (found"
          f" {type(row).__name__!r}). Use `dgf.validate.conform_graph` to"
          " convert features to their schema dtypes."
      )
      return
    if not feature_format.np_dtype_matches(row.dtype, expected_dtype):
      yield Issue.error(
          f"The variable-length feature {feature_name!r} in {source} has row"
          f" {row_idx} with dtype {row.dtype} (i.e., {row.dtype.type}), but"
          f" the schema expects format {feature_schema.format!r}, which"
          f" corresponds to dtype {expected_dtype}. Use"
          " `dgf.validate.conform_graph` to convert features to their schema"
          " dtypes."
      )
      return


def _feature_schema_issues(
    feature_name: str,
    feature_schema: schema_lib.FeatureSchema,
    source: str,
) -> Iterator[validate_lib.Issue]:
  """Yields single-feature schema issues."""
  if feature_name != KEY_ID:
    if (
        feature_schema.semantic is None
        or feature_schema.semantic == schema_lib.FeatureSemantic.UNKNOWN
    ):
      yield Issue.warning(
          f"The feature {feature_name!r} in {source} has no semantic type"
          " defined in the schema. This will make the feature harder to"
          " consume by some tools."
      )
    elif (
        feature_schema.semantic == schema_lib.FeatureSemantic.MASK
        and feature_schema.format != schema_lib.FeatureFormat.BOOL
    ):
      yield Issue.error(
          f"The mask feature {feature_name!r} in {source} must have"
          f" format BOOL, but has format {feature_schema.format!r}."
      )
  elif feature_schema.semantic != schema_lib.FeatureSemantic.PRIMARY_ID:
    yield Issue.warning(
        f"The semantic of feature {feature_name!r} in {source} is"
        f" not PRIMARY_ID. Instead, it is {feature_schema.semantic!r}"
    )

  if (
      feature_schema.is_creation_time
      and feature_schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP
  ):
    yield Issue.error(
        f"The feature {feature_name!r} in {source} has"
        " is_creation_time=True, but its semantic is"
        f" {feature_schema.semantic!r}. Features with"
        " is_creation_time=True must have semantic=TIMESTAMP."
    )

  if (
      feature_schema.semantic == schema_lib.FeatureSemantic.TIMESTAMP
      and feature_schema.format != schema_lib.FeatureFormat.INTEGER_64
  ):
    yield Issue.error(
        f"The feature {feature_name!r} in {source} has"
        " semantic=TIMESTAMP, but its format is"
        f" {feature_schema.format!r}. Features with semantic=TIMESTAMP"
        " must have format=INTEGER_64."
    )

  if feature_schema.group is not None and not feature_schema.is_timeseries:
    yield Issue.error(
        f"The feature {feature_name!r} in {source} has"
        f" group={feature_schema.group!r}, but is_timeseries=False."
        " Non-timeseries features cannot have a group."
    )


def _single_feature_issues(
    feature_name: str,
    featureset_data: in_memory_graph.Features,
    feature_schema: schema_lib.FeatureSchema,
    source: str,
) -> Iterator[validate_lib.Issue]:
  """Yields data and schema issues for a single feature."""
  if feature_name not in featureset_data:
    yield Issue.error(f"Missing feature {feature_name!r} in {source}.")
    return
  feature_data = featureset_data[feature_name]
  if not isinstance(feature_data, np.ndarray):
    yield Issue.error(
        f"The feature {feature_name!r} in {source} is"
        f" not a numpy array, but {type(feature_data).__name__!r}"
    )
    return
  expected_dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[
      feature_schema.format
  ]
  expected_shape = feature_schema.shape or ()
  if feature_data.dtype == np.object_:
    yield from _object_feature_data_issues(
        feature_name,
        feature_data,
        feature_schema,
        expected_dtype,
        expected_shape,
        source,
    )
  else:
    if not feature_format.np_dtype_matches(feature_data.dtype, expected_dtype):
      yield Issue.error(
          f"The feature {feature_name!r} in {source} has dtype"
          f" {feature_data.dtype} (i.e., {feature_data.dtype.type}),"
          f" but the schema expects format {feature_schema.format!r},"
          f" which corresponds to dtype {expected_dtype}"
      )
    if len(expected_shape) + 1 != len(feature_data.shape):
      yield Issue.error(
          f"The feature {feature_name!r} in {source} has shape"
          f" {feature_data.shape}, but the schema expects a shape"
          f" compatible with {expected_shape} (i.e., one more dimension)."
      )
      return
    for dim_idx, expected_dim in enumerate(expected_shape):
      observed_dim = feature_data.shape[dim_idx + 1]
      if expected_dim is not None and expected_dim != observed_dim:
        yield Issue.error(
            f"The feature {feature_name!r} in {source} has shape"
            f" {feature_data.shape}, but the schema expects dimension"
            f" {dim_idx} to be {expected_dim}."
        )
  yield from _feature_schema_issues(feature_name, feature_schema, source)


def _varlen_timeseries_length_issues(
    feature_name: str,
    feat_shape: Sequence[int | None],
    ts_name: str,
    featureset_data: in_memory_graph.Features,
    source: str,
) -> Iterator[validate_lib.Issue]:
  """Yields per-row length mismatch issues for a variable-length timeseries."""
  if feature_name not in featureset_data or ts_name not in featureset_data:
    return
  if not (feat_shape and feat_shape[0] is None):
    return
  feature_data = featureset_data[feature_name]
  ts_data = featureset_data[ts_name]
  if len(feature_data) != len(ts_data):
    return
  for i in range(len(feature_data)):
    f_val = feature_data[i]
    t_val = ts_data[i]
    if f_val is not None and t_val is not None:
      f_len = len(f_val) if hasattr(f_val, "__len__") else 1
      t_len = len(t_val) if hasattr(t_val, "__len__") else 1
      if f_len != t_len:
        yield Issue.error(
            f"The feature {feature_name!r} in {source} has a variable-length"
            f" timeseries at index {i} of length {f_len}, which does not match"
            f" the creation time sequence {ts_name!r} of length {t_len}."
        )


def _creation_time_group_issues(
    ts_group: str,
    fnames: Sequence[str],
    ct_feats: Sequence[str],
    featureset_data: in_memory_graph.Features,
    featureset_schema: schema_lib.FeatureSetSchema,
    source: str,
) -> Iterator[validate_lib.Issue]:
  """Yields creation-time sequence alignment issues for one timeseries group."""
  if len(ct_feats) > 1:
    yield Issue.error(
        "Multiple creation time sequence features found for group"
        f" {ts_group!r} in {source}: {ct_feats}. At most one creation"
        " time feature is allowed per sequence group."
    )
    return
  if not ct_feats:
    return
  ts_name = ct_feats[0]
  ts_schema = featureset_schema[ts_name]
  ts_shape = ts_schema.shape or ()
  if len(ts_shape) != 1:
    yield Issue.error(
        f"The creation time feature {ts_name!r} in {source} must have"
        " exactly 1 sequence dimension in schema shape."
    )
  for feature_name in fnames:
    if feature_name == ts_name:
      continue
    feat_schema = featureset_schema[feature_name]
    feat_shape = feat_schema.shape or ()
    if len(feat_shape) < 1:
      yield Issue.error(
          f"The feature {feature_name!r} in {source} belongs to group"
          f" {ts_group!r}, but must have at least 1 sequence dimension"
          " in schema shape."
      )
    if (
        len(ts_shape) == 1
        and len(feat_shape) >= 1
        and ts_shape[0] != feat_shape[0]
    ):
      yield Issue.error(
          f"The feature {feature_name!r} in {source} has schema shape"
          f" {feat_shape} whose 0th dimension ({feat_shape[0]}) does"
          f" not match creation time feature {ts_name!r} schema shape"
          f" 0th dimension ({ts_shape[0]})."
      )
    yield from _varlen_timeseries_length_issues(
        feature_name, feat_shape, ts_name, featureset_data, source
    )


def _mask_group_issues(
    ts_group: str,
    masks_in_group: Sequence[str],
    featureset_data: in_memory_graph.Features,
    source: str,
) -> Iterator[validate_lib.Issue]:
  """Yields duplicate-mask issues for one timeseries group."""
  if len(masks_in_group) <= 1:
    return
  all_identical = True
  first_data = featureset_data.get(masks_in_group[0])
  for m_name in masks_in_group[1:]:
    m_data = featureset_data.get(m_name)
    if (
        first_data is None
        or m_data is None
        or not np.array_equal(first_data, m_data)
    ):
      all_identical = False
      break
  if all_identical:
    yield Issue.warning(
        "Multiple features with semantic=MASK found for timeseries"
        f" group {ts_group!r} in {source}: {masks_in_group}. Since they"
        " are identical, consider consolidating them into a single"
        " mask."
    )
  else:
    yield Issue.error(
        "Multiple features with semantic=MASK found for timeseries"
        f" group {ts_group!r} in {source} with differing or unavailable"
        f" values: {masks_in_group}."
    )


def _timeseries_group_issues(
    featureset_data: in_memory_graph.Features,
    featureset_schema: schema_lib.FeatureSetSchema,
    source: str,
) -> Iterator[validate_lib.Issue]:
  """Yields cross-feature timeseries and mask group issues."""
  entity_creation_time_feats = []
  group_to_creation_time = collections.defaultdict(list)
  group_to_masks = collections.defaultdict(list)
  group_to_features = collections.defaultdict(list)

  for feature_name, feature_schema in featureset_schema.items():
    if feature_schema.is_creation_time and not feature_schema.is_timeseries:
      entity_creation_time_feats.append(feature_name)

    ts_group = feature_schema.group or (
        feature_name
        if feature_schema.is_timeseries and feature_schema.is_creation_time
        else None
    )

    if feature_schema.is_timeseries and ts_group is not None:
      group_to_features[ts_group].append(feature_name)
      if feature_schema.is_creation_time:
        group_to_creation_time[ts_group].append(feature_name)

    if feature_schema.semantic == schema_lib.FeatureSemantic.MASK:
      if feature_schema.group is None:
        yield Issue.error(
            f"The mask feature {feature_name!r} in {source} must have a"
            " group to associate it with the features it masks."
        )
      else:
        group_to_masks[feature_schema.group].append(feature_name)

  if len(entity_creation_time_feats) > 1:
    yield Issue.error(
        f"Multiple entity creation time features found in {source}:"
        f" {entity_creation_time_feats}. At most one feature can have"
        " is_creation_time=True for a node or edge set."
    )

  for ts_group, fnames in group_to_features.items():
    ct_feats = group_to_creation_time.get(ts_group, [])
    yield from _creation_time_group_issues(
        ts_group, fnames, ct_feats, featureset_data, featureset_schema, source
    )

  for ts_group, masks_in_group in group_to_masks.items():
    yield from _mask_group_issues(
        ts_group, masks_in_group, featureset_data, source
    )


def feature_set_issues(
    featureset_data: in_memory_graph.Features,
    featureset_schema: schema_lib.FeatureSetSchema,
    source: str,
) -> Sequence[validate_lib.Issue]:
  """Lists the issues of a feature."""
  items: list[validate_lib.Issue] = []
  for feature_name, feature_schema in featureset_schema.items():
    items.extend(
        _single_feature_issues(
            feature_name, featureset_data, feature_schema, source
        )
    )
  items.extend(
      _timeseries_group_issues(featureset_data, featureset_schema, source)
  )
  return items


def issues(
    graph: in_memory_graph.InMemoryGraph, schema: schema_lib.GraphSchema
) -> Sequence[validate_lib.Issue]:
  """Lists potential issues with the graph object."""
  items = []
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    if nodeset_name not in graph.node_sets:
      items.append(
          Issue.error(f"The graph is missing the nodeset {nodeset_name!r}")
      )
      continue
    nodeset_data = graph.node_sets[nodeset_name]
    items.extend(
        feature_set_issues(
            nodeset_data.features,
            nodeset_schema.features,
            f"nodeset {nodeset_name!r}",
        )
    )

    primary_key = analyse_schema.primary_feature_or_none(
        nodeset_name, nodeset_schema
    )

    if primary_key is None:
      items.append(
          Issue.warning(
              f"The nodeset {nodeset_name!r} schema has no primary feature,"
              " i.e., a feature with semantic=PRIMARY_ID. You won't be able to"
              " serialize it. Recommended actions: Set it manually (e.g.,"
              f" `schema.node_sets['{nodeset_name}'].features[<some"
              " column>].semantic = dgf.data.FeatureSemantic.PRIMARY_ID`) or"
              " use `schema = dgf.validate.fix_schema(schema)`."
          )
      )

    if nodeset_data.num_nodes is None:
      items.append(
          Issue.warning(
              f"The nodeset {nodeset_name!r} has `num_nodes` set to None."
          )
      )
    else:
      for feature_name, feature_value in nodeset_data.features.items():
        if (
            isinstance(feature_value, np.ndarray)
            and feature_value.shape[0] != nodeset_data.num_nodes
        ):
          items.append(
              Issue.error(
                  f"The nodeset {nodeset_name!r} has feature"
                  f" {feature_name!r} with shape {feature_value.shape}, but"
                  " expected the first dimension to be equal to `num_nodes`"
                  f" ({nodeset_data.num_nodes})."
              )
          )

  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    if edgeset_name not in graph.edge_sets:
      items.append(
          Issue.error(f"The graph is missing the edgeset {edgeset_name!r}")
      )
      continue
    edgeset_data = graph.edge_sets[edgeset_name]
    items.extend(
        feature_set_issues(
            edgeset_data.features,
            edgeset_schema.features,
            f"edgeset {edgeset_name!r}",
        )
    )

    if edgeset_schema.source not in schema.node_sets:
      items.append(
          Issue.error(
              f"The edgeset {edgeset_name!r} refers to a source nodeset "
              f"{edgeset_schema.source!r} which is not defined in the graph "
              "schema's node_sets."
          )
      )
      continue
    if edgeset_schema.target not in schema.node_sets:
      items.append(
          Issue.error(
              f"The edgeset {edgeset_name!r} refers to a target nodeset "
              f"{edgeset_schema.target!r} which is not defined in the graph "
              "schema's node_sets."
          )
      )
      continue
    if edgeset_data.adjacency.dtype not in [np.int64, np.int32]:
      items.append(
          Issue.error(
              f"The edgeset {edgeset_name!r} adjacency has dtype "
              f"{edgeset_data.adjacency.dtype}, but expected one of "
              f"[{np.int64.__name__}, {np.int32.__name__}]."
          )
      )
      continue
    if edgeset_data.adjacency.ndim != 2 or edgeset_data.adjacency.shape[0] != 2:
      items.append(
          Issue.error(
              f"The edgeset {edgeset_name!r} adjacency has shape "
              f"{edgeset_data.adjacency.shape}, but expected shape[0] to be 2."
          )
      )
      continue
    num_edges = edgeset_data.adjacency.shape[1]
    if (
        num_edges > 0
        and edgeset_schema.source in graph.node_sets
        and edgeset_schema.target in graph.node_sets
    ):
      num_source_nodes = graph.node_sets[edgeset_schema.source].num_nodes
      if num_source_nodes is not None:
        min_source = np.min(edgeset_data.adjacency[0])
        max_source = np.max(edgeset_data.adjacency[0])
        if min_source < 0 or max_source >= num_source_nodes:
          items.append(
              Issue.error(
                  f"The edgeset {edgeset_name!r} adjacency contains source "
                  "node indices out of bounds. Expected indices to be in "
                  f"[0, {num_source_nodes}), but found min: {min_source}, "
                  f"max: {max_source}."
              )
          )
      num_target_nodes = graph.node_sets[edgeset_schema.target].num_nodes
      if num_target_nodes is not None:
        min_target = np.min(edgeset_data.adjacency[1])
        max_target = np.max(edgeset_data.adjacency[1])
        if min_target < 0 or max_target >= num_target_nodes:
          items.append(
              Issue.error(
                  f"The edgeset {edgeset_name!r} adjacency contains target "
                  "node indices out of bounds. Expected indices to be in "
                  f"[0, {num_target_nodes}), but found min: {min_target}, "
                  f"max: {max_target}."
              )
          )

  return items


def validate_graph(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    *,
    raise_on_error: bool = True,
    raise_on_warning: bool = True,
):
  """Validates an in memory graph object."""
  validate_lib.print_and_raise(
      issues(graph, schema),
      raise_on_error=raise_on_error,
      raise_on_warning=raise_on_warning,
  )
