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
from typing import Sequence
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
from dgf.src.validate import validate as validate_lib
import numpy as np

Issue = validate_lib.Issue

KEY_ID = "#id"


def feature_set_issues(
    featureset_data: in_memory_graph.Features,
    featureset_schema: schema_lib.FeatureSetSchema,
    source: str,
) -> Sequence[validate_lib.Issue]:
  """Lists the issues of a feature."""
  items = []
  for feature_name, feature_schema in featureset_schema.items():
    if feature_name not in featureset_data:
      items.append(
          Issue.error(f"Missing feature {feature_name!r} in {source}.")
      )
      continue
    feature_data = featureset_data[feature_name]
    if not isinstance(feature_data, np.ndarray):
      items.append(
          Issue.error(
              f"The feature {feature_name!r} in {source} is"
              f" not a numpy array, but {type(feature_data).__name__!r}"
          )
      )
      continue
    expected_dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[
        feature_schema.format
    ]
    expected_shape = feature_schema.shape or ()
    if feature_data.dtype == np.object_:
      if len(expected_shape) < 1 or expected_shape[0] is not None:
        items.append(
            Issue.error(
                f"The feature {feature_name!r} in {source} is stored as an"
                " object (`dtype=object`), which is only permitted for"
                " variable-length features. However, the schema defines a"
                f" fixed shape of {expected_shape}. If your feature is a"
                " string, make sure to cast it as a np.bytes_."
            )
        )
    else:
      if feature_data.dtype.type != expected_dtype:
        if expected_dtype in (
            np.int64,
            np.longlong,
        ) and feature_data.dtype.type in (np.int64, np.longlong):
          pass
        else:
          items.append(
              Issue.error(
                  f"The feature {feature_name!r} in {source} has dtype"
                  f" {feature_data.dtype} (i.e., {feature_data.dtype.type}),"
                  f" but the schema expects format {feature_schema.format!r},"
                  f" which corresponds to dtype {expected_dtype}"
              )
          )
      if len(expected_shape) + 1 != len(feature_data.shape):
        items.append(
            Issue.error(
                f"The feature {feature_name!r} in {source} has shape"
                f" {feature_data.shape}, but the schema expects a shape"
                f" compatible with {expected_shape} (i.e., one more dimension)."
            )
        )
        continue
      for dim_idx in range(len(expected_shape)):
        expected_dim = expected_shape[dim_idx]
        observed_dim = feature_data.shape[dim_idx + 1]
        if expected_dim is None:
          continue
        if expected_dim != observed_dim:
          items.append(
              Issue.error(
                  f"The feature {feature_name!r} in {source} has shape"
                  f" {feature_data.shape}, but the schema expects dimension"
                  f" {dim_idx} to be {expected_dim}."
              )
          )
    if feature_name != KEY_ID:
      if (
          feature_schema.semantic is None
          or feature_schema.semantic == schema_lib.FeatureSemantic.UNKNOWN
      ):
        items.append(
            Issue.warning(
                f"The feature {feature_name!r} in {source} has no semantic type"
                " defined in the schema. This will make the feature harder to"
                " consume by some tools."
            )
        )
      elif feature_schema.semantic == schema_lib.FeatureSemantic.MASK:
        if feature_schema.format != schema_lib.FeatureFormat.BOOL:
          items.append(
              Issue.error(
                  f"The mask feature {feature_name!r} in {source} must have"
                  f" format BOOL, but has format {feature_schema.format!r}."
              )
          )
    else:
      if feature_schema.semantic != schema_lib.FeatureSemantic.PRIMARY_ID:
        items.append(
            Issue.warning(
                f"The semantic of feature {feature_name!r} in {source} is"
                f" not PRIMARY_ID. Instead, it is {feature_schema.semantic!r}"
            )
        )

    if feature_schema.is_creation_time:
      if feature_schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP:
        items.append(
            Issue.error(
                f"The feature {feature_name!r} in {source} has"
                " is_creation_time=True, but its semantic is"
                f" {feature_schema.semantic!r}. Features with"
                " is_creation_time=True must have semantic=TIMESTAMP."
            )
        )

    if feature_schema.semantic == schema_lib.FeatureSemantic.TIMESTAMP:
      if feature_schema.format != schema_lib.FeatureFormat.INTEGER_64:
        items.append(
            Issue.error(
                f"The feature {feature_name!r} in {source} has"
                f" semantic=TIMESTAMP, but its format is"
                f" {feature_schema.format!r}. Features with semantic=TIMESTAMP"
                " must have format=INTEGER_64."
            )
        )

    if feature_schema.group is not None and not feature_schema.is_timeseries:
      items.append(
          Issue.error(
              f"The feature {feature_name!r} in {source} has"
              f" group={feature_schema.group!r}, but is_timeseries=False."
              " Non-timeseries features cannot have a group."
          )
      )

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
        items.append(
            Issue.error(
                f"The mask feature {feature_name!r} in {source} must have a"
                " group to associate it with the features it masks."
            )
        )
      else:
        group_to_masks[feature_schema.group].append(feature_name)

  if len(entity_creation_time_feats) > 1:
    items.append(
        Issue.error(
            f"Multiple entity creation time features found in {source}:"
            f" {entity_creation_time_feats}. At most one feature can have"
            " is_creation_time=True for a node or edge set."
        )
    )

  for ts_group, fnames in group_to_features.items():
    ct_feats = group_to_creation_time.get(ts_group, [])
    if len(ct_feats) > 1:
      items.append(
          Issue.error(
              "Multiple creation time sequence features found for group"
              f" {ts_group!r} in {source}: {ct_feats}. At most one creation"
              " time feature is allowed per sequence group."
          )
      )
    elif len(ct_feats) == 1:
      ts_name = ct_feats[0]
      ts_schema = featureset_schema[ts_name]
      ts_shape = ts_schema.shape or ()
      if len(ts_shape) != 1:
        items.append(
            Issue.error(
                f"The creation time feature {ts_name!r} in {source} must have"
                " exactly 1 sequence dimension in schema shape."
            )
        )
      for feature_name in fnames:
        if feature_name == ts_name:
          continue
        feat_schema = featureset_schema[feature_name]
        feat_shape = feat_schema.shape or ()
        if len(feat_shape) < 1:
          items.append(
              Issue.error(
                  f"The feature {feature_name!r} in {source} belongs to group"
                  f" {ts_group!r}, but must have at least 1 sequence dimension"
                  " in schema shape."
              )
          )
        if (
            len(ts_shape) == 1
            and len(feat_shape) >= 1
            and ts_shape[0] != feat_shape[0]
        ):
          items.append(
              Issue.error(
                  f"The feature {feature_name!r} in {source} has schema shape"
                  f" {feat_shape} whose 0th dimension ({feat_shape[0]}) does"
                  f" not match creation time feature {ts_name!r} schema shape"
                  f" 0th dimension ({ts_shape[0]})."
              )
          )
        if feature_name in featureset_data and ts_name in featureset_data:
          if feat_shape and feat_shape[0] is None:
            feature_data = featureset_data[feature_name]
            ts_data = featureset_data[ts_name]
            if len(feature_data) == len(ts_data):
              for i in range(len(feature_data)):
                f_val = feature_data[i]
                t_val = ts_data[i]
                if f_val is not None and t_val is not None:
                  f_len = len(f_val) if hasattr(f_val, "__len__") else 1
                  t_len = len(t_val) if hasattr(t_val, "__len__") else 1
                  if f_len != t_len:
                    items.append(
                        Issue.error(
                            f"The feature {feature_name!r} in {source} has a"
                            f" variable-length timeseries at index {i} of"
                            f" length {f_len}, which does not match the"
                            f" creation time sequence {ts_name!r} of length"
                            f" {t_len}."
                        )
                    )

  for ts_group, masks_in_group in group_to_masks.items():
    if len(masks_in_group) > 1:
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
        items.append(
            Issue.warning(
                f"Multiple features with semantic=MASK found for timeseries"
                f" group {ts_group!r} in {source}: {masks_in_group}. Since they"
                " are identical, consider consolidating them into a single"
                " mask."
            )
        )
      else:
        items.append(
            Issue.error(
                f"Multiple features with semantic=MASK found for timeseries"
                f" group {ts_group!r} in {source} with differing or unavailable"
                f" values: {masks_in_group}."
            )
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

    if KEY_ID not in nodeset_schema.features:
      items.append(
          Issue.warning(
              f"The nodeset {nodeset_name!r} schema is missing the '#id'"
              " feature. Giving a clearly defined #id column is recommanded."
              " It is also required for non-string node IDs e.g. integer IDs."
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
        if feature_value.shape[0] != nodeset_data.num_nodes:
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
