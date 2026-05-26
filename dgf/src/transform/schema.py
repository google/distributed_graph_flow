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

"""Transformation operations on schemas."""

from dgf.src.data import schema as schema_lib


def filter_schema(
    schema: schema_lib.GraphSchema, filter: schema_lib.GraphSchemaFilter
) -> schema_lib.GraphSchema:
  """Extracts a subset of the nodesets/edgesets/features from a schema.

  Dangling edgesets (those with source or target nodes removed) are
  automatically excluded.

  Args:
    schema: The original GraphSchema to filter.
    filter: A GraphSchemaFilter specifying which parts of the schema to keep.

  Returns:
    A new GraphSchema containing only the filtered elements.
  """

  filtered_node_sets = {}
  for node_name, node_schema in schema.node_sets.items():
    if filter.nodeset_fn(node_name, node_schema):
      filtered_features = {
          feature_name: feature_schema
          for feature_name, feature_schema in node_schema.features.items()
          if filter.feature_fn(feature_name, feature_schema)
          and filter.node_feature_fn(feature_name, feature_schema)
      }
      filtered_node_sets[node_name] = schema_lib.NodeSchema(
          features=filtered_features
      )

  filtered_edge_sets = {}
  for edge_name, edge_schema in schema.edge_sets.items():
    if filter.edgeset_fn(edge_name, edge_schema):
      if edge_schema.source not in filtered_node_sets:
        print(
            f"Warning: Skipping edge '{edge_name}' because its source node "
            f"'{edge_schema.source}' is not in the filtered node sets."
        )
        continue
      if edge_schema.target not in filtered_node_sets:
        print(
            f"Warning: Skipping edge '{edge_name}' because its target node "
            f"'{edge_schema.target}' is not in the filtered node sets."
        )
        continue
      filtered_features = {
          feature_name: feature_schema
          for feature_name, feature_schema in edge_schema.features.items()
          if filter.feature_fn(feature_name, feature_schema)
          and filter.edge_feature_fn(feature_name, feature_schema)
      }
      filtered_edge_sets[edge_name] = schema_lib.EdgeSchema(
          source=edge_schema.source,
          target=edge_schema.target,
          features=filtered_features,
      )

  return schema_lib.GraphSchema(
      node_sets=filtered_node_sets, edge_sets=filtered_edge_sets
  )


def drop_edge_features_from_schema(schema) -> schema_lib.GraphSchema:
  """Drops all edge features from a schema."""
  return filter_schema(
      schema, schema_lib.GraphSchemaFilter(edge_feature_fn=lambda x, _: False)
  )
