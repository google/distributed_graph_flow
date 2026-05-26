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

"""Reverses the direction of edges."""

import apache_beam as beam
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib


# TODO(bmayer): Rename this to reverse_flat_edges. The reverse of the adjacency
# format will require a beam.Transform with distributed explod by target
# followed by a groupby.
def reverse_edges(
    hgraph: distributed_graph.Graph,
) -> distributed_graph.Graph:
  """Reverse the direction of edges in a graph."""
  reversed_edge_sets = {}
  for edge_set_name, edge_set in hgraph.edge_sets.items():
    if hgraph.edge_format == distributed_graph.EdgeFormat.FLAT:
      reversed_edge_sets[edge_set_name] = (
          edge_set
          | f"Reverse edges {edge_set_name}"
          >> beam.Map(
              lambda edge: distributed_graph.Edge(
                  id=edge.id, source=edge.target, target=edge.source
              )
          )
      )
    else:
      raise ValueError(
          f"Unsupported edge set type: {hgraph.edge_format} for edge set"
          f" {edge_set_name}"
      )

  reversed_schema = schema_lib.GraphSchema(
      node_sets=hgraph.schema.node_sets,
      edge_sets={
          name: schema_lib.EdgeSchema(
              source=edge_schema.target,
              target=edge_schema.source,
              features=edge_schema.features,
          )
          for name, edge_schema in hgraph.schema.edge_sets.items()
      },
  )
  return distributed_graph.Graph(
      node_sets=hgraph.node_sets,
      edge_sets=reversed_edge_sets,
      edge_format=hgraph.edge_format,
      edge_features=hgraph.edge_features,
      schema=reversed_schema,
  )
