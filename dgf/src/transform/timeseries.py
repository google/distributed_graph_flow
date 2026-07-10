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
        np.empty((0, seq_len) + feat_shape, dtype=np.bool_),
    )

  # Fast vectorized path when all entities share a fixed sequence length.
  if is_static_shape and raw_series.ndim >= 2:
    num_steps = raw_series.shape[1]
    if num_steps >= seq_len:
      padded_matrix = raw_series[:, -seq_len:].astype(dtype, copy=True)
      mask_matrix = np.ones(
          (num_entities, seq_len) + feat_shape, dtype=np.bool_
      )
      return padded_matrix, mask_matrix

    pad_width = [(0, 0), (seq_len - num_steps, 0)] + [(0, 0)] * len(feat_shape)
    padded_matrix = np.pad(
        raw_series.astype(dtype, copy=False),
        pad_width=pad_width,
        mode="constant",
        constant_values=padding_value,
    )
    mask_matrix = np.pad(
        np.ones((num_entities, num_steps) + feat_shape, dtype=np.bool_),
        pad_width=pad_width,
        mode="constant",
        constant_values=False,
    )
    return padded_matrix, mask_matrix

  padded_matrix = np.full(
      (num_entities, seq_len) + feat_shape,
      fill_value=padding_value,
      dtype=dtype,
  )
  # Binary mask matrix matching padded_matrix shape ((num_entities, seq_len) +
  # feat_shape) where True indicates valid observed time steps and False
  # indicates left-padded steps.
  mask_matrix = np.zeros((num_entities, seq_len) + feat_shape, dtype=np.bool_)

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


def _process_feature_set(
    features: in_memory_graph.Features,
    feature_schemas: schema_lib.FeatureSetSchema,
    ts_specs: List[temporal_util.TimeseriesGroupSpec],
    config: PadAndCapTimeseriesConfig,
) -> Tuple[in_memory_graph.Features, schema_lib.FeatureSetSchema]:
  """Extracts fixed-dimension sequence features for a feature set."""
  new_features: in_memory_graph.Features = {}
  new_feat_schemas: schema_lib.FeatureSetSchema = {}
  seq_len = config.sequence_length

  # Map timeseries feature names to their associated timestamp feature name.
  ts_features = {}
  for group in ts_specs:
    for fname in group.feature_names:
      ts_features[fname] = group.timestamp_feature_name

  for fname, fschema in feature_schemas.items():
    if fname not in ts_features:
      new_features[fname] = features[fname]
      new_feat_schemas[fname] = fschema

  if not ts_features:
    return new_features, new_feat_schemas

  for fname in ts_features:
    fschema = feature_schemas[fname]

    dtype = feature_format.FEATURE_FORMAT_TO_NP_DTYPE[fschema.format]
    feat_shape = fschema.shape[1:] if fschema.shape is not None else ()

    padded_matrix, mask_matrix = _pad_and_cap_single_feature(
        raw_series=features[fname],
        seq_len=seq_len,
        feat_shape=feat_shape,
        padding_value=config.padding_value,
        dtype=dtype,
        is_static_shape=fschema.is_static_shape(),
    )

    new_features[fname] = padded_matrix
    new_feat_schemas[fname] = dataclasses.replace(
        fschema,
        shape=(seq_len,) + feat_shape,
        is_timeseries=True,
    )

    new_features[f"{fname}_mask"] = mask_matrix
    new_feat_schemas[f"{fname}_mask"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(seq_len,) + feat_shape,
        is_timeseries=fschema.is_timeseries,
        timestamps=fschema.timestamps,
    )

  return new_features, new_feat_schemas


def pad_and_cap_timeseries_features(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    config: PadAndCapTimeseriesConfig,
    schema_cache: Optional[temporal_util.TimeseriesSchemaCache] = None,
) -> Tuple[in_memory_graph.InMemoryGraph, schema_lib.GraphSchema]:
  """Pads and caps timeseries sequence features into fixed-dimension tensors.

  For every feature where `is_timeseries=True`, this function caps long causal
  histories (`[-K:]`) and left-pads short histories to `config.sequence_length`
  ($K$). Generates parallel binary attention masks (`{feature_name}_mask`).

  All resulting features have fixed `shape=(K,)` and maintain
  `is_timeseries=True`, allowing downstream sequence models to identify
  temporal features.

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

    new_feats, new_schemas = _process_feature_set(
        features=ns_val.features,
        feature_schemas=ns_schema.features,
        ts_specs=ts_specs,
        config=config,
    )
    new_node_sets[ns_name] = in_memory_graph.InMemoryNodeSet(
        num_nodes=ns_val.num_nodes, features=new_feats
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

    new_feats, new_schemas = _process_feature_set(
        features=es_val.features,
        feature_schemas=es_schema.features,
        ts_specs=ts_specs,
        config=config,
    )
    new_edge_sets[es_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=es_val.adjacency, features=new_feats
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
