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

"""Beam utilities for working with Google Cloud Native Graphs."""

from typing import Any, Dict

from dgf.src.analyse import schema as schema_analyse_lib
from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.io.gcp import common as gcp_common_lib


def create_distributed_node_set(
    graph_element: Dict[str, Any],
    nodeset_name: str,
    graph_schema: schema_lib.GraphSchema,
    combine_as_json: bool = False,
) -> distributed_graph_lib.Node:
  """Returns a DGF Node from a GCP property graph element.

  Args:
    graph_element: The GCP graph node in the form of DGF Node.
    graph_element_name: The name of the graph element.
    graph_schema: The DGF schema of the graph.
    combine_as_json: Whether to combine the features as JSON.
  """
  features = gcp_common_lib.graph_element_to_features(
      nodeset_name,
      gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE,
      graph_element[gcp_common_lib.GRAPH_ELEMENT_JSON_KEY],
      graph_schema,
      combine_as_json,
  )

  element_id = graph_element[gcp_common_lib.GRAPH_ELEMENT_ID_KEY].encode(
      "utf-8"
  )

  if not combine_as_json:
    nodeset_schema = graph_schema.node_sets[nodeset_name]
    primary_key = schema_analyse_lib.primary_feature(
        nodeset_name, nodeset_schema
    )
    features[primary_key] = element_id

  return distributed_graph_lib.Node(
      id=element_id,
      features=features,
  )


def create_distributed_edge_set(
    graph_element: Dict[str, Any],
    edgeset_name: str,
    graph_schema: schema_lib.GraphSchema,
    combine_as_json: bool = False,
) -> distributed_graph_lib.Edge:
  """Returns a DGF Edge from a GCP property graph element.

  Args:
    graph_element: The GCP graph edge record in the form of DGF Edge.
    graph_element_name: The name of the graph element.
    graph_schema: The DGF schema of the graph.
    combine_as_json: Whether to combine the features as JSON.
  """
  features = gcp_common_lib.graph_element_to_features(
      edgeset_name,
      gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE,
      graph_element[gcp_common_lib.GRAPH_ELEMENT_JSON_KEY],
      graph_schema,
      combine_as_json,
  )

  element_id = graph_element[gcp_common_lib.GRAPH_ELEMENT_ID_KEY].encode(
      "utf-8"
  )

  if not combine_as_json:
    edgeset_schema = graph_schema.edge_sets[edgeset_name]
    primary_key = schema_analyse_lib.primary_feature_or_none(
        edgeset_name, edgeset_schema
    )
    if primary_key is not None:
      features[primary_key] = element_id

  return distributed_graph_lib.Edge(
      id=element_id,
      source=graph_element[gcp_common_lib.GRAPH_ELEMENT_SOURCE_ID_KEY].encode(
          "utf-8"
      ),
      target=graph_element[gcp_common_lib.GRAPH_ELEMENT_TARGET_ID_KEY].encode(
          "utf-8"
      ),
      features=features,
  )
