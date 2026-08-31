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
from typing import Any, Dict, List, Optional, Tuple

from dgf.src.data import in_memory_graph
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format
from dgf.src.util import temporal as temporal_util
import numpy as np


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
  if dtype == np.bytes_:
    if raw_series.dtype.kind in ("S", "a"):
      dtype = raw_series.dtype
    elif raw_series.dtype.kind == "O" and num_entities > 0:
      for elem in raw_series:
        if isinstance(elem, np.ndarray) and elem.dtype.kind in ("S", "a"):
          if dtype == np.bytes_ or elem.dtype.itemsize > dtype.itemsize:
            dtype = elem.dtype
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


def has_timeseries_padding(
    padding: Optional[padding_lib.Padding],
) -> bool:
  """Returns True if padding contains any timeseries feature padding."""
  if padding is None:
    return False
  for ns in padding.node_sets.values():
    for fp in ns.features.values():
      if fp.max_timeseries_len is not None:
        return True
  for es in padding.edge_sets.values():
    for fp in es.features.values():
      if fp.max_timeseries_len is not None:
        return True
  return False


def _validate_group_sequence_lengths(
    group_specs: List[temporal_util.TimeseriesGroupSpec],
    feature_padding: Dict[str, padding_lib.FeaturePadding],
    schemas: schema_lib.FeatureSetSchema,
) -> None:
  """Validates that all features in each timeseries group share the same sequence length."""
  for group in group_specs:
    ts_group = schemas[group.feature_names[0]].group or group.feature_names[0]
    padded_features = {
        feat: feature_padding[feat].max_timeseries_len
        for feat in group.feature_names
        if feat in feature_padding
        and feature_padding[feat].max_timeseries_len is not None
    }
    if 0 < len(padded_features) < len(group.feature_names):
      missing = [f for f in group.feature_names if f not in padded_features]
      raise ValueError(
          f"Features in sequence group '{ts_group}' have inconsistent"
          f" padding configuration. Missing padding for features: {missing}."
          " All features in the same group must share the same sequence length."
      )
    if len(padded_features) == len(group.feature_names):
      lengths = set(padded_features.values())
      if len(lengths) > 1:
        raise ValueError(
            f"Features in sequence group '{ts_group}' have conflicting"
            f" max_timeseries_len: {padded_features}. All features in the same"
            " group must share the same sequence length."
        )


def _pad_timeseries_feature_set_schema(
    schemas: schema_lib.FeatureSetSchema,
    group_specs: List[temporal_util.TimeseriesGroupSpec],
    feature_padding: Dict[str, padding_lib.FeaturePadding],
) -> schema_lib.FeatureSetSchema:
  """Computes schema for a feature set after padding/capping."""
  _validate_group_sequence_lengths(group_specs, feature_padding, schemas)
  new_schemas: schema_lib.FeatureSetSchema = {}

  padded_features = {
      feat: fp.max_timeseries_len
      for feat, fp in feature_padding.items()
      if fp.max_timeseries_len is not None
  }

  for feature_name, feature_schema in schemas.items():
    # Skip features that do not have a timeseries sequence length set.
    if feature_name not in padded_features:
      new_schemas[feature_name] = feature_schema
      continue

    seq_len = padded_features[feature_name]
    ts_group = feature_schema.group or feature_name
    new_schemas[feature_name] = dataclasses.replace(
        temporal_util.with_sequence_length(feature_schema, seq_len),
        group=ts_group,
    )
    # Do not add masks for masks inception.
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


def pad_timeseries_schema(
    schema: schema_lib.GraphSchema,
    padding: padding_lib.Padding,
    schema_cache: Optional[temporal_util.TimeseriesSchemaCache] = None,
) -> schema_lib.GraphSchema:
  """Returns the GraphSchema after padding/capping timeseries and adding mask features."""
  if not temporal_util.schema_has_timeseries_features(schema):
    return schema

  if schema_cache is None:
    schema_cache = temporal_util.extract_timeseries_schema_cache(schema)

  new_node_sets = {}
  for ns_name, ns_schema in schema.node_sets.items():
    ts_specs = schema_cache.node_sets[ns_name]
    if not ts_specs:
      new_node_sets[ns_name] = ns_schema
    else:
      new_node_sets[ns_name] = schema_lib.NodeSchema(
          features=_pad_timeseries_feature_set_schema(
              schemas=ns_schema.features,
              group_specs=ts_specs,
              feature_padding=padding.node_sets[ns_name].features,
          )
      )

  new_edge_sets = {}
  for es_name, es_schema in schema.edge_sets.items():
    ts_specs = schema_cache.edge_sets[es_name]
    if not ts_specs:
      new_edge_sets[es_name] = es_schema
    else:
      new_edge_sets[es_name] = schema_lib.EdgeSchema(
          source=es_schema.source,
          target=es_schema.target,
          features=_pad_timeseries_feature_set_schema(
              schemas=es_schema.features,
              group_specs=ts_specs,
              feature_padding=padding.edge_sets[es_name].features,
          ),
      )

  return schema_lib.GraphSchema(
      node_sets=new_node_sets, edge_sets=new_edge_sets
  )


def pad_timeseries_feature_set(
    values: in_memory_graph.Features,
    schemas: schema_lib.FeatureSetSchema,
    group_specs: List[temporal_util.TimeseriesGroupSpec],
    feature_padding: Dict[str, padding_lib.FeaturePadding],
    padding_value: Any = 0,
) -> in_memory_graph.Features:
  """Pads/caps timeseries features and generates matching mask features for a single entity set."""
  _validate_group_sequence_lengths(group_specs, feature_padding, schemas)
  new_values: in_memory_graph.Features = {}

  padded_features = {
      feat: fp.max_timeseries_len
      for feat, fp in feature_padding.items()
      if fp.max_timeseries_len is not None
  }

  for feature_name, val in values.items():
    # Skip features that do not have a timeseries sequence length set.
    if feature_name not in padded_features:
      new_values[feature_name] = val
      continue

    feature_schema = schemas[feature_name]
    seq_len = padded_features[feature_name]
    dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[feature_schema.format]
    feat_shape = temporal_util.get_timeseries_step_shape(feature_schema)

    feature_padding_val = padding_value
    if (
        padding_value == 0
        and feature_schema.format == schema_lib.FeatureFormat.BYTES
    ):
      feature_padding_val = b""

    padded_matrix, mask_matrix = _pad_and_cap_single_feature(
        raw_series=val,
        seq_len=seq_len,
        feat_shape=feat_shape,
        padding_value=feature_padding_val,
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


def pad_timeseries_graph(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    padding: padding_lib.Padding,
    padding_value: Any = 0,
    schema_cache: Optional[temporal_util.TimeseriesSchemaCache] = None,
) -> in_memory_graph.InMemoryGraph:
  """Pads and caps all timeseries features in a single graph."""
  if not temporal_util.schema_has_timeseries_features(schema):
    return graph

  if schema_cache is None:
    schema_cache = temporal_util.extract_timeseries_schema_cache(schema)

  new_node_sets = {}
  for ns_name, ns_schema in schema.node_sets.items():
    ns_val = graph.node_sets[ns_name]
    ts_specs = schema_cache.node_sets[ns_name]
    if not ts_specs:
      new_node_sets[ns_name] = ns_val
      continue
    new_vals = pad_timeseries_feature_set(
        values=ns_val.features,
        schemas=ns_schema.features,
        group_specs=ts_specs,
        feature_padding=padding.node_sets[ns_name].features,
        padding_value=padding_value,
    )
    new_node_sets[ns_name] = in_memory_graph.InMemoryNodeSet(
        num_nodes=ns_val.num_nodes, features=new_vals
    )

  new_edge_sets = {}
  for es_name, es_schema in schema.edge_sets.items():
    es_val = graph.edge_sets[es_name]
    ts_specs = schema_cache.edge_sets[es_name]
    if not ts_specs:
      new_edge_sets[es_name] = es_val
      continue
    new_vals = pad_timeseries_feature_set(
        values=es_val.features,
        schemas=es_schema.features,
        group_specs=ts_specs,
        feature_padding=padding.edge_sets[es_name].features,
        padding_value=padding_value,
    )
    new_edge_sets[es_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=es_val.adjacency, features=new_vals
    )

  return in_memory_graph.InMemoryGraph(
      node_sets=new_node_sets, edge_sets=new_edge_sets
  )
