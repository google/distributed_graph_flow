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

"""Operations on a schema."""

import copy
from typing import List, Tuple
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.transform import schema as schema_transform_lib


def filter_schema(
    src: schema_lib.GraphSchema, selected_features: List[str]
) -> schema_lib.GraphSchema:
  """Creates a new schema with a subset of the features.

  The other parts of the schema are not modified.

  Args:
    src: The source schema to extract.
    selected_features: A list of feature names to include in the new schema.

  Returns:
    The extracted schema.
  """
  extracted_schema = copy.deepcopy(src)

  for node_name in extracted_schema.node_sets:
    node_schema = extracted_schema.node_sets[node_name]
    node_schema.features = {
        k: v for k, v in node_schema.features.items() if k in selected_features
    }

  for edge_name in extracted_schema.edge_sets:
    edge_schema = extracted_schema.edge_sets[edge_name]
    edge_schema.features = {
        k: v for k, v in edge_schema.features.items() if k in selected_features
    }

  return extracted_schema


def filter_graph(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
) -> in_memory_graph_lib.InMemoryGraph:
  """Creates a in memory graph with a subset of nodesets/edgesets/features.

  Args:
    graph: The input InMemoryGraph.
    schema: The schema defining the subset of nodesets, edgesets, and features.

  Returns:
    A new InMemoryGraph containing only the specified nodesets,
    edgesets, and features.
  """
  extracted_node_sets = {}
  for node_name, node_schema in schema.node_sets.items():
    src_nodeset = graph.node_sets[node_name]
    extracted_node_sets[node_name] = in_memory_graph_lib.InMemoryNodeSet(
        features={
            k: src_nodeset.features[k] for k in node_schema.features.keys()
        },
        num_nodes=src_nodeset.num_nodes,
    )

  extracted_edge_sets = {}
  for edge_name, edge_schema in schema.edge_sets.items():
    src_edgeset = graph.edge_sets[edge_name]
    extracted_edge_sets[edge_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=src_edgeset.adjacency,
        features={
            k: src_edgeset.features[k] for k in edge_schema.features.keys()
        },
    )

  return in_memory_graph_lib.InMemoryGraph(
      node_sets=extracted_node_sets, edge_sets=extracted_edge_sets
  )


def drop_edge_features(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Drops all edge features from a graph and its schema."""
  schema = copy.deepcopy(schema)
  schema = schema_transform_lib.drop_edge_features_from_schema(schema)
  return filter_graph(graph, schema), schema
