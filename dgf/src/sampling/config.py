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

"""Configuration for graph sampling operations.

This file defines the following important objects:

  - SimpleSamplingConfig: A simple sampling configuration to be created by a
    user.
  - PlanNode: A node within a SamplingPlan.
  - PlanEdge: A step within a SamplingPlan.
  - SamplingPlan: A detailed sampling configuration.
  - simple_sampling_config_to_sampling_plan: Create a SamplingPlan from a
    SimpleSamplingConfig.
"""

import collections
import dataclasses
from typing import Dict, List, Union
import dataclasses_json
from dgf.src.data import schema as schema_lib


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class SimpleSamplingConfig:
  """Configuration for simple neighborhood sampling.

  This configuration defines a simple, breadth-first sampling strategy. Starting
  from a specified `seed_nodeset`, it performs a fixed number of `num_hops`.
  At each hop, for every edge type connected to the current nodeset, up to
  `hop_width` neighbors are sampled.

  Attributes:
    seed_nodeset: The name of the nodeset from which to start the sampling.
    num_hops: The maximum number of hops (steps) to perform outwards from the
      `seed_nodeset`. This determines the depth of the sampled neighborhood.
    hop_width: The maximum number of neighbors to sample for each edge type at
      every hop. If more than `hop_width` neighbors are available, a subset is
      chosen.
    reverse: If True, edges can be traversed in both their defined direction and
      the reverse direction. If False, only the defined forward direction of
      edges is used.
    with_replacement: If false, the sampled graph is a sub-graph of the original
      graph with cycles (if the original graph has cycles). Nodes / edges that
      are visited multiple times do not lead to multiple nodes / edges in the
      sampled graph. If true, the sampled graph is a tree where nodes / edges in
      the original graph might lead to multiple nodes / edges in the sampled
      grpah.
    edgeset_timestamp_features: A dictionary mapping edgeset names to the name
      of the feature to use as a timestamp for temporal sampling. Edgesets not
      present in the dictionary are considered atemporal.
  """

  seed_nodeset: str
  num_hops: int = 3
  hop_width: int = 5
  reverse: bool = True
  with_replacement: bool = False
  edgeset_timestamp_features: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )


@dataclasses.dataclass
class PlanNode:
  """Part of a sampling plan."""

  nodeset: str
  children: List["PlanEdge"] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PlanEdge:
  """Represents a single step in a complex sampling plan.

  Defines a transition between `PlanNode`s using a specific `edgeset`.

  Attributes:
    edgeset: The name of the edgeset used for this step.
    reversed: Whether the edge is traversed in the reverse direction.
    node: The `PlanNode` reached by this step.
    hop_width: The number of neighbors to sample using this edgeset.
  """

  edgeset: str
  reversed: bool
  node: PlanNode
  hop_width: int


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class SamplingPlan:
  """Defines a complex sampling config.

  Attributes:
    root: The root node of the sampling plan, specifying the starting nodeset.
    with_replacement: Test if the sampling is done with replacement. See
      documentation for "with_replacement" attribute in SimpleSamplingConfig for
      the full explanation.
    edgeset_timestamp_features: A dictionary mapping edgeset names to the name
      of the feature to use as a timestamp for temporal sampling. Edgesets not
      present in the dictionary are considered atemporal.
  """

  root: PlanNode
  with_replacement: bool = False
  edgeset_timestamp_features: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )



def simple_sampling_config_to_sampling_plan(
    src: SimpleSamplingConfig,
    schema: schema_lib.GraphSchema,
) -> SamplingPlan:
  """Converts a SimpleSamplingConfig to a more general SamplingPlan.

  Args:
    src: The SimpleSamplingConfig to convert.
    schema: The graph schema, used to resolve edge connections.

  Returns:
    A SamplingPlan equivalent to the provided SimpleSamplingConfig.
  """

  if src.seed_nodeset not in schema.node_sets:
    raise ValueError(
        f"Seed nodeset '{src.seed_nodeset}' not found in the graph schema. The"
        f" available nodesets are: {list(schema.node_sets.keys())}"
    )

  # Index edge connections
  # `nodeset_to_edgesets` maps a nodeset to the list of edgeset+nodesets it
  # connects with.
  connections = collections.defaultdict(list)
  for edgeset_name, edgeset in sorted(schema.edge_sets.items()):
    connections[edgeset.source].append((edgeset_name, edgeset.target, False))
    if src.reverse:
      connections[edgeset.target].append((edgeset_name, edgeset.source, True))

  # Build plan recursively.
  def rec_build(nodeset: str, depth: int) -> PlanNode:
    children_list = []
    if depth < src.num_hops and nodeset in connections:
      for edgeset, dst_nodeset, is_reversed in connections[nodeset]:
        children_list.append(
            PlanEdge(
                edgeset=edgeset,
                reversed=is_reversed,
                node=rec_build(dst_nodeset, depth=depth + 1),
                hop_width=src.hop_width,
            )
        )
    return PlanNode(nodeset, children_list)

  return SamplingPlan(
      root=rec_build(src.seed_nodeset, depth=0),
      with_replacement=src.with_replacement,
      edgeset_timestamp_features=src.edgeset_timestamp_features,
  )
