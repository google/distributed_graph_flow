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
from collections.abc import Collection
import dataclasses
import enum
import dataclasses_json
from dgf.src.data import schema as schema_lib
from dgf.src.util import temporal as temporal_util


class TemporalStrategy(enum.Enum):
  """How the valid neighbors are selected during temporal sampling."""

  # Selected uniformly at random.
  UNIFORM = "UNIFORM"
  # The most recent ones are selected.
  LAST = "LAST"


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
    temporal_sampling: If True, temporal sampling is enabled. The creation time
      of the nodes and of the edges is inferred from the schema (see
      `dgf.util.temporal.edgeset_timestamp_features`).
    multi_visit: If True (default), a node reached several times at the same
      step of the plan is expanded (i.e. its own neighbors are sampled) once per
      visit. If False, only the first visit expands it: the other visits still
      create edges, but the traversal does not continue through them. Setting it
      to False bounds the amount of work per sample and matches the default
      behavior of PyG. Note that a node can still be expanded several times
      through different steps of the plan. Ignored if `with_replacement` is
      True.
    max_timeseries_len: The maximum number of historical causal sequence steps
      retained for each timeseries feature.
    propagate_timestamp_to_edges: If true (default) and temporal sampling is
      enabled, the edgesets without a creation time get one derived from their
      connected nodes. Equivalent to calling
      `dgf.transform.propagate_timestamp_to_edges` on the graph, except that the
      propagation is done by the sampler and is not visible in the graph, the
      schema, or the samples.
    temporal_strategy: How the neighbors are selected among the valid ones.
  """

  seed_nodeset: str
  num_hops: int = 3
  hop_width: int = 5
  reverse: bool = True
  with_replacement: bool = False
  temporal_sampling: bool = False
  multi_visit: bool = True
  max_timeseries_len: int = 32
  propagate_timestamp_to_edges: bool = True
  temporal_strategy: TemporalStrategy = TemporalStrategy.UNIFORM


@dataclasses.dataclass
class PlanNode:
  """Part of a sampling plan."""

  nodeset: str
  children: list["PlanEdge"] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PlanEdge:
  """Represents a single step in a complex sampling plan.

  Defines a transition between `PlanNode`s using a specific `edgeset`.

  Attributes:
    edgeset: The name of the edgeset used for this step.
    reversed: Whether the edge is traversed in the reverse direction.
    node: The `PlanNode` reached by this step.
    hop_width: The number of neighbors to sample using this edgeset.
    temporal_strategy: How the neighbors are selected during temporal sampling.
  """

  edgeset: str
  reversed: bool
  node: PlanNode
  hop_width: int
  temporal_strategy: TemporalStrategy = TemporalStrategy.UNIFORM


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class SamplingPlan:
  """Defines a complex sampling config.

  Attributes:
    root: The root node of the sampling plan, specifying the starting nodeset.
    with_replacement: Test if the sampling is done with replacement. See
      documentation for "with_replacement" attribute in SimpleSamplingConfig for
      the full explanation.
    temporal_sampling: If True, temporal sampling is enabled. The creation time
      of the nodes and of the edges is inferred from the schema (see
      `dgf.util.temporal.edgeset_timestamp_features`).
    multi_visit: Test if a node reached several times at the same step of the
      plan is expanded once per visit. See documentation for the "multi_visit"
      attribute in SimpleSamplingConfig for the full explanation.
    max_timeseries_len: The maximum number of historical causal sequence steps
      retained for each timeseries feature.
    propagate_timestamp_to_edges: If true (default) and temporal sampling is
      enabled, the edgesets without a creation time get one derived from their
      connected nodes. Equivalent to calling
      `dgf.transform.propagate_timestamp_to_edges` on the graph, except that the
      propagation is done by the sampler and is not visible in the graph, the
      schema, or the samples.
  """

  root: PlanNode
  with_replacement: bool = False
  temporal_sampling: bool = False
  multi_visit: bool = True
  max_timeseries_len: int = 32
  propagate_timestamp_to_edges: bool = True


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
                temporal_strategy=src.temporal_strategy,
            )
        )
    return PlanNode(nodeset, children_list)

  return SamplingPlan(
      root=rec_build(src.seed_nodeset, depth=0),
      with_replacement=src.with_replacement,
      temporal_sampling=src.temporal_sampling,
      multi_visit=src.multi_visit,
      max_timeseries_len=src.max_timeseries_len,
      propagate_timestamp_to_edges=src.propagate_timestamp_to_edges,
  )


def edgesets_in_plan(plan: SamplingPlan) -> set[str]:
  """Lists the name of the edgesets traversed by a sampling plan."""
  edgesets = set()

  def rec_scan(node: PlanNode):
    for child in node.children:
      edgesets.add(child.edgeset)
      rec_scan(child.node)

  rec_scan(plan.root)
  return edgesets


def _plan_edges(plan: SamplingPlan) -> list[PlanEdge]:
  """Lists the edges of a sampling plan."""
  edges = []

  def rec_scan(node: PlanNode):
    for child in node.children:
      edges.append(child)
      rec_scan(child.node)

  rec_scan(plan.root)
  return edges


def check_temporal_strategies(
    plan: SamplingPlan, timestamped_edgesets: Collection[str]
) -> None:
  """Checks the temporal strategies of the edges of a sampling plan.

  Args:
    plan: The sampling plan.
    timestamped_edgesets: The edgesets with a creation time (possibly propagated
      from their nodes).

  Raises:
    ValueError: If LAST is used without temporal sampling or on an edgeset
      without creation time.
  """
  for edge in _plan_edges(plan):
    if edge.temporal_strategy != TemporalStrategy.LAST:
      continue
    if not plan.temporal_sampling:
      raise ValueError(
          f"temporal_strategy=LAST (edgeset {edge.edgeset!r}) requires"
          " temporal_sampling=True."
      )
    if edge.edgeset not in timestamped_edgesets:
      raise ValueError(
          f"temporal_strategy=LAST is used on edgeset {edge.edgeset!r} which"
          " does not have a creation time. Add a creation time to the edgeset"
          " or to its nodesets (with propagate_timestamp_to_edges=True), or use"
          " temporal_strategy=UNIFORM."
      )


def check_supported_options(
    plan: SamplingPlan,
    schema: schema_lib.GraphSchema | None,
    sampler_name: str,
    *,
    support_with_replacement: bool = False,
    support_non_with_replacement: bool = False,
    support_multi_visit: bool = False,
    support_non_multi_visit: bool = False,
    support_temporal_sampling: bool = False,
    support_last_temporal_strategy: bool = False,
    support_timeseries: bool = False,
) -> None:
  """Raises a NotImplementedError if the plan uses options a sampler lacks.

  Each support_<option> argument tells if the sampler implements an option
  value (e.g., support_non_multi_visit for multi_visit=False, or
  support_timeseries for timeseries features in the schema). By default, the
  options are not supported.
  """
  issues = []
  if plan.with_replacement and not support_with_replacement:
    issues.append("with_replacement=True")
  if not plan.with_replacement:
    if not support_non_with_replacement:
      issues.append("with_replacement=False")
    if plan.multi_visit and not support_multi_visit:
      issues.append(
          "multi_visit=True (note: this is the default value, set"
          " multi_visit=False)"
      )
    if not plan.multi_visit and not support_non_multi_visit:
      issues.append("multi_visit=False")
  if plan.temporal_sampling and not support_temporal_sampling:
    issues.append("temporal_sampling=True")
  supported_strategies = [TemporalStrategy.UNIFORM]
  if support_last_temporal_strategy:
    supported_strategies.append(TemporalStrategy.LAST)
  unsupported_strategies = {
      edge.temporal_strategy.value
      for edge in _plan_edges(plan)
      if edge.temporal_strategy not in supported_strategies
  }
  issues.extend(
      f"temporal_strategy={name}" for name in sorted(unsupported_strategies)
  )
  if (
      not support_timeseries
      and schema is not None
      and temporal_util.schema_has_timeseries_features(schema)
  ):
    issues.append("timeseries features (is_timeseries=True) in the schema")
  if issues:
    raise NotImplementedError(
        f"The {sampler_name} does not support the following options: "
        + ", ".join(issues)
    )
