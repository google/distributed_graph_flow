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

"""GF <-> networkx toolkit."""

from typing import Optional, Sequence
import dgf.src.data.in_memory_graph
import networkx as nx
import numpy as np
import tqdm

InMemoryGraph = dgf.src.data.in_memory_graph.InMemoryGraph


def homogeneous_graph_piece_to_nx(
    graph: InMemoryGraph,
    nodeset_name: str,
    edgeset_name: str,
    *,
    id_feature_name: Optional[str] = None,
    features_to_keep: str | Sequence[str] = (),
    verbose=True
):
  """Convert InMemoryGraph to an nx.Graph object.

  This function is named `homogeneous_` because we assume the domain (source)
  and range (target) of `edgeset_name` is from/onto `nodeset_name`. We do not
  check the schema if this is true.

  Args:
    graph: The dgf.data.InMemoryGraph to convert.
    nodeset_name: The name of the nodeset to convert.
    edgeset_name: The name of the edgeset to convert. We assume domain and range
      of `edgeset_name` is from/onto `nodeset_name`.
    id_feature_name: The name of the feature to use as the node ID. We use
      indices on [0, num_nodes - 1] if not provide
    features_to_keep: A string or list of string of features to add to nx node
      data
    verbose: Print conversion progress.

  Returns:
    A networkx Graph object.
  """
  nodes = []
  edges = []

  if isinstance(features_to_keep, str):
    features_to_keep = [features_to_keep]

  nodeset = graph.node_sets[nodeset_name]
  edgeset = graph.edge_sets[edgeset_name]

  ids = []
  if id_feature_name is not None:
    ids = nodeset.features[id_feature_name].tolist()
  else:
    ids = np.arange(nodeset.num_nodes)  # pyrefly: ignore[no-matching-overload]

  for i, id in tqdm.tqdm(
      enumerate(ids),
      disable=not verbose,
      total=nodeset.num_nodes,
      desc="Converting nodes",
  ):
    attributes = {}

    for fname in features_to_keep:
      if fname in nodeset.features:
        attributes[fname] = nodeset.features[fname][i].tolist()

    if attributes:
      nodes.append((id, attributes))
    else:
      nodes.append(id)

  for source_idx, target_idx in tqdm.tqdm(
      zip(*edgeset.adjacency.tolist()),
      disable=not verbose,
      desc="Converting edges",
  ):
    source_id = ids[source_idx]
    target_id = ids[target_idx]
    edges.append((source_id, target_id))

  g = nx.Graph()
  g.add_nodes_from(nodes)
  g.add_edges_from(edges)

  return g
