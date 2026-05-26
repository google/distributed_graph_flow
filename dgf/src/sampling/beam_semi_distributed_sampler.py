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

"""Beam-based sampling using an in-memory sampler for graph topology.

This sampler uses Beam for distributed operations like indexing nodes and edges
or fetching features, while leveraging an in-memory sampler for the core graph
sampling logic. For reference, 1 billion edges require approximately 4GB of RAM
with uint32 indexing or 8GB with uint64 indexing. These memory requirements are
well within the capabilities of typical machines, such as workstations (e.g.,
128GB) and high-end data center machines (up to 2TB).

Design doc:
https://docs.google.com/document/d/1LlWgXaUMuoP4UqDTbMpFgoGxjkX3RGFPEz_6qGfDzV4/edit?tab=t.0#bookmark=id.22c1qdyeeoky
"""

import apache_beam as beam
from dgf.src.data import distributed_graph


def extract_beam_nodes_ids(
    graph: distributed_graph.Graph, target_nodeset: str
) -> beam.PCollection[distributed_graph.NodeId]:
  """Extracts all the node ids of a given nodeset.

  This method can be used to create the seed nodes argument for the sampler.

  See "sample_with_beam_semi_distributed_sampler" for a usage example.

  Args:
    graph: The distributed graph containing the node sets.
    target_nodeset: The name of the node set from which to extract node IDs.

  Returns:
    A `PCollection` of bytes, where each element is a unique node ID
    from the specified `target_nodeset`.
  """
  return graph.node_sets[target_nodeset] | "Extract seeds" >> beam.Map(
      lambda node: node.id
  )
