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
    else:
      if feature_schema.semantic != schema_lib.FeatureSemantic.PRIMARY_ID:
        items.append(
            Issue.warning(
                f"The semantic of feature {feature_name!r} in {source} is"
                f" not PRIMARY_ID. Instead, it is {feature_schema.semantic!r}"
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
