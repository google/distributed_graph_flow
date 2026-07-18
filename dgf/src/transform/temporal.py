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

"""Temporal transformations for graphs."""

from typing import Dict, List, Optional, Union
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
import numpy as np


def propagate_timestamp_to_edges(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    target_edgesets: Optional[List[str]] = None,
    node_timestamps: Union[str, Dict[str, str]] = "timestamps",
    target_feature: str = "timestamps",
) -> tuple[in_memory_graph.InMemoryGraph, schema_lib.GraphSchema]:
  """Propagates timestamps from nodes to edges.

  Computes a new edge feature for edgesets that don't have a timestamp,
  as the maximum value of the timestamps of the two connected nodes.

  Usage example:

  ```python
  new_graph, new_schema = dgf.transform.propagate_timestamp_to_edges(
      graph=graph,
      schema=schema,
      target_edgesets=["e1"],
      node_timestamps={"n1": "time", "n2": "timestamp"},
      target_feature="ts",
  )
  ```

  Args:
    graph: The input in-memory graph.
    schema: The graph schema.
    target_edgesets: Edgesets to populate with a timestamp. If None, process all
      edgesets.
    node_timestamps: The feature name for node timestamps. Can be a string if
      it's the same for all nodesets, or a dictionary mapping nodeset name to
      feature name.
    target_feature: The name of the new edge feature. Defaults to "timestamps".

  Returns:
    A tuple containing the new graph and new schema.
  """
  new_edge_sets = dict(graph.edge_sets)
  new_edge_set_schemas = dict(schema.edge_sets)

  def get_node_ts(nodeset_name):
    feat_name = (
        node_timestamps
        if isinstance(node_timestamps, str)
        else node_timestamps.get(nodeset_name)
    )
    if (
        feat_name
        and nodeset_name in schema.node_sets
        and feat_name in schema.node_sets[nodeset_name].features
    ):
      return (
          graph.node_sets[nodeset_name].features[feat_name],
          schema.node_sets[nodeset_name].features[feat_name].format,
      )
    return None, None

  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    if target_edgesets is not None and edgeset_name not in target_edgesets:
      continue

    if target_feature in edgeset_schema.features:
      raise ValueError(
          f"Target feature '{target_feature}' already exists in edgeset"
          f" '{edgeset_name}'."
      )

    src_ts, src_format = get_node_ts(edgeset_schema.source)
    tgt_ts, tgt_format = get_node_ts(edgeset_schema.target)

    if src_ts is None and tgt_ts is None:
      raise ValueError(
          f"Neither source nodeset '{edgeset_schema.source}' nor target nodeset"
          f" '{edgeset_schema.target}' has timestamps for edgeset"
          f" '{edgeset_name}'."
      )

    edgeset_value = graph.edge_sets[edgeset_name]
    src_indices, tgt_indices = edgeset_value.adjacency

    if src_ts is not None and tgt_ts is not None:
      edge_ts = np.maximum(src_ts[src_indices], tgt_ts[tgt_indices])
      ts_format = src_format
    elif src_ts is not None:
      edge_ts = src_ts[src_indices]
      ts_format = src_format
    else:
      edge_ts = tgt_ts[tgt_indices]  # pyrefly: ignore[unsupported-operation]
      ts_format = tgt_format

    new_edge_sets[edgeset_name] = in_memory_graph.InMemoryEdgeSet(
        adjacency=edgeset_value.adjacency,
        features={**edgeset_value.features, target_feature: edge_ts},
    )

    new_edge_set_schemas[edgeset_name] = schema_lib.EdgeSchema(
        source=edgeset_schema.source,
        target=edgeset_schema.target,
        features={
            **edgeset_schema.features,
            target_feature: schema_lib.FeatureSchema(
                format=ts_format, semantic=schema_lib.FeatureSemantic.TIMESTAMP
            ),
        },
    )

  return (
      in_memory_graph.InMemoryGraph(
          node_sets=graph.node_sets, edge_sets=new_edge_sets
      ),
      schema_lib.GraphSchema(
          node_sets=schema.node_sets, edge_sets=new_edge_set_schemas
      ),
  )
