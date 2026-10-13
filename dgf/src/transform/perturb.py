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

"""Graph perturbation operations for model interpretability.

A perturbation destroys the information carried by one component of a graph
(a feature, an edgeset or a nodeset) while leaving the rest of the graph
untouched. Perturbed graphs are returned as new `InMemoryGraph` objects that
share all the unperturbed arrays with the input graph.

Main entry points:
  - `perturb_node_feature` / `perturb_edge_feature`: Shuffle the values of a
    feature across nodes / edges.
  - `perturb_edgeset`: Rewire or drop the edges of an edgeset.
  - `perturb_nodeset`: Shuffle all the features of a nodeset and rewire its
    incident edgesets.
  - `maslov_sneppen_rewire`: Degree-preserving rewiring of an adjacency matrix.
"""

import dataclasses
import enum
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.util import temporal as temporal_util
import numpy as np


class EdgePerturbationMode(enum.Enum):
  """How the edges of an edgeset are perturbed.

  Attributes:
    MASLOV_SNEPPEN: Degree-preserving rewiring: edges are swapped so that every
      node keeps its exact number of incoming and outgoing edges.
    DROP: All the edges are removed.
  """

  MASLOV_SNEPPEN = "MASLOV_SNEPPEN"
  DROP = "DROP"


def parse_edge_perturbation_mode(
    mode: EdgePerturbationMode | str,
) -> EdgePerturbationMode:
  """Parses a string or `EdgePerturbationMode` into an `EdgePerturbationMode`."""
  if isinstance(mode, EdgePerturbationMode):
    return mode
  if not isinstance(mode, str):
    raise TypeError(
        f"Expected EdgePerturbationMode or str, got {type(mode)}: {mode}"
    )
  try:
    return EdgePerturbationMode(mode)
  except ValueError as exc:
    raise ValueError(
        f"Unknown edge perturbation mode: {mode}. The supported values are:"
        f" {[item.value for item in EdgePerturbationMode]}."
    ) from exc


def timeseries_group(
    feature_name: str, feature_schema: schema_lib.FeatureSchema
) -> str | None:
  """Returns the sequence group of a timeseries feature, or None if static.

  A timeseries feature without an explicit `group` forms its own group.
  """
  if not feature_schema.is_timeseries:
    return None
  return feature_schema.group or feature_name


def shuffled_feature_names(
    features_schema: schema_lib.FeatureSetSchema, feature_name: str
) -> list[str]:
  """Returns the names of the features shuffled together with `feature_name`.

  The sequences of a timeseries group (values, timestamps, mask) are shuffled
  together so they stay aligned. A static feature is shuffled alone.
  """
  group = timeseries_group(feature_name, features_schema[feature_name])
  if group is None:
    return [feature_name]
  return [
      name
      for name, feature_schema in features_schema.items()
      if timeseries_group(name, feature_schema) == group
  ]


def _shuffle_features(
    features: dict[str, np.ndarray],
    features_schema: schema_lib.FeatureSetSchema,
    feature_names: list[str],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
  """Shuffles the given features across the nodes / edges.

  Each feature is shuffled with its own permutation, except the features of a
  timeseries group which share the same permutation to stay aligned.

  Returns:
    A copy of `features` with the shuffled features replaced.
  """
  new_features = dict(features)
  group_perms: dict[str, np.ndarray] = {}
  for name in feature_names:
    group = timeseries_group(name, features_schema[name])
    if group is None:
      perm = rng.permutation(len(features[name]))
    elif group not in group_perms:
      group_perms[group] = perm = rng.permutation(len(features[name]))
    else:
      perm = group_perms[group]
    new_features[name] = features[name][perm]
  return new_features


def perturb_node_feature(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    nodeset_name: str,
    feature_name: str,
    rng: np.random.Generator,
) -> in_memory_graph_lib.InMemoryGraph:
  """Perturbs a node feature by shuffling its values across the nodes.

  The marginal distribution of the feature is preserved while its alignment
  with the nodes (and thus the labels and the topology) is broken. If the
  feature belongs to a timeseries group, the whole group is shuffled together.

  Args:
    graph: The input graph.
    schema: The schema of the graph.
    nodeset_name: Name of the nodeset containing the feature.
    feature_name: Name of the feature to perturb.
    rng: Random generator.

  Returns:
    A new graph sharing all the unperturbed arrays with `graph`.
  """
  features_schema = schema.node_sets[nodeset_name].features
  nodeset = graph.node_sets[nodeset_name]
  new_features = _shuffle_features(
      nodeset.features,
      features_schema,
      shuffled_feature_names(features_schema, feature_name),
      rng,
  )
  return dataclasses.replace(
      graph,
      node_sets={
          **graph.node_sets,
          nodeset_name: dataclasses.replace(nodeset, features=new_features),
      },
  )


def perturb_edge_feature(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    edgeset_name: str,
    feature_name: str,
    rng: np.random.Generator,
) -> in_memory_graph_lib.InMemoryGraph:
  """Perturbs an edge feature by shuffling its values across the edges.

  See `perturb_node_feature` for details.

  Args:
    graph: The input graph.
    schema: The schema of the graph.
    edgeset_name: Name of the edgeset containing the feature.
    feature_name: Name of the feature to perturb.
    rng: Random generator.

  Returns:
    A new graph sharing all the unperturbed arrays with `graph`.
  """
  features_schema = schema.edge_sets[edgeset_name].features
  edgeset = graph.edge_sets[edgeset_name]
  new_features = _shuffle_features(
      edgeset.features,
      features_schema,
      shuffled_feature_names(features_schema, feature_name),
      rng,
  )
  return dataclasses.replace(
      graph,
      edge_sets={
          **graph.edge_sets,
          edgeset_name: dataclasses.replace(edgeset, features=new_features),
      },
  )


def maslov_sneppen_rewire(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    num_swaps_per_edge: int = 4,
    allow_self_loops: bool = True,
) -> np.ndarray:
  """Rewires a directed/bipartite adjacency matrix via Maslov-Sneppen swaps.

  Performs vectorized degree-preserving edge swaps `(u1, v1), (u2, v2) ->
  (u2, v1), (u1, v2)` so that every source node's out-degree and every target
  node's in-degree are strictly preserved.

  Args:
    adjacency: Array of shape `[2, num_edges]` with `[sources, targets]`.
    rng: NumPy random generator.
    num_swaps_per_edge: Number of swap attempts per edge. Each round pairs up
      all the edges and attempts one swap per pair.
    allow_self_loops: Whether self-loops `u == v` are permitted (True for
      bipartite edgesets with distinct source and target nodesets, False for
      homogeneous edgesets).

  Returns:
    A new `np.ndarray` of shape `[2, num_edges]` with identical in-degree and
    out-degree sequences.
  """
  num_edges = adjacency.shape[1]
  if num_edges <= 1:
    # Nothing to swap.
    return adjacency.copy()

  sources = adjacency[0].copy()
  targets = adjacency[1].copy()

  num_pairs = num_edges // 2
  for _ in range(num_swaps_per_edge):
    perm = rng.permutation(num_edges)
    idx1 = perm[:num_pairs]
    idx2 = perm[num_pairs : 2 * num_pairs]

    u1 = sources[idx1]
    u2 = sources[idx2]
    v1 = targets[idx1]
    v2 = targets[idx2]

    valid = (u1 != u2) & (v1 != v2)
    if not allow_self_loops:
      # Reject the swaps that would create a self-loop.
      valid &= (u1 != v2) & (u2 != v1)

    valid_idx1 = idx1[valid]
    valid_idx2 = idx2[valid]
    if len(valid_idx1) > 0:
      # Swap sources between valid_idx1 and valid_idx2 so edge index i remains
      # attached to target v_i (preserving target-side edge timestamp alignment
      # when temporal sampling is active) while preserving exact out-degrees
      # and in-degrees.
      tmp = sources[valid_idx1].copy()
      sources[valid_idx1] = sources[valid_idx2]
      sources[valid_idx2] = tmp

  return np.stack([sources, targets], axis=0)


def perturb_edgeset(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    edgeset_name: str,
    rng: np.random.Generator,
    *,
    mode: EdgePerturbationMode | str = EdgePerturbationMode.MASLOV_SNEPPEN,
    num_swaps_per_edge: int = 4,
    perturb_features: bool = True,
    preserve_timestamps: bool = True,
) -> in_memory_graph_lib.InMemoryGraph:
  """Perturbs an edgeset's topology and optionally its edge features.

  Args:
    graph: The input `InMemoryGraph`.
    schema: The `GraphSchema` of the graph.
    edgeset_name: Name of the edgeset to perturb.
    rng: NumPy random generator.
    mode: How the edges are perturbed. See `EdgePerturbationMode`.
    num_swaps_per_edge: Number of swap attempts per edge for `MASLOV_SNEPPEN`.
      Larger values destroy more of the original topology.
    perturb_features: Whether to also shuffle the edge features of the edgeset.
    preserve_timestamps: If True, creation time features on the edgeset are not
      shuffled when `perturb_features=True` so temporal causal constraints
      remain valid.

  Returns:
    A new `InMemoryGraph` with the perturbed edgeset.
  """
  mode = parse_edge_perturbation_mode(mode)
  edgeset_schema = schema.edge_sets[edgeset_name]
  src_edgeset = graph.edge_sets[edgeset_name]
  num_edges = src_edgeset.num_edges()
  dtype = src_edgeset.adjacency.dtype

  if mode == EdgePerturbationMode.MASLOV_SNEPPEN:
    is_homogeneous = edgeset_schema.source == edgeset_schema.target
    new_adjacency = maslov_sneppen_rewire(
        src_edgeset.adjacency,
        rng,
        num_swaps_per_edge=num_swaps_per_edge,
        allow_self_loops=not is_homogeneous,
    )
  elif mode == EdgePerturbationMode.DROP:
    new_adjacency = np.zeros((2, 0), dtype=dtype)
  else:
    raise ValueError(f"Unsupported edge perturbation mode: {mode}")

  if mode == EdgePerturbationMode.DROP:
    # No more edges: keep empty feature arrays.
    new_features = {
        feat_name: src_edgeset.features[feat_name][:0]
        for feat_name in edgeset_schema.features
    }
  elif perturb_features and edgeset_schema.features:
    # Shuffle the edge features, except the creation timestamps if requested.
    creation_ts_feat = (
        temporal_util.edgeset_creation_time_feature_name(edgeset_schema, schema)
        if preserve_timestamps
        else None
    )
    new_features = _shuffle_features(
        src_edgeset.features,
        edgeset_schema.features,
        [f for f in edgeset_schema.features if f != creation_ts_feat],
        rng,
    )
  else:
    # Keep the edge features aligned with the edge indices.
    new_features = dict(src_edgeset.features)

  return dataclasses.replace(
      graph,
      edge_sets={
          **graph.edge_sets,
          edgeset_name: dataclasses.replace(
              src_edgeset, adjacency=new_adjacency, features=new_features
          ),
      },
  )


def perturb_nodeset(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    nodeset_name: str,
    rng: np.random.Generator,
    *,
    target_nodeset: str | None = None,
    target_column: str | None = None,
    edge_perturbation_mode: EdgePerturbationMode | str = (
        EdgePerturbationMode.MASLOV_SNEPPEN
    ),
    include_incident_edges: bool = True,
    preserve_timestamps: bool = True,
) -> in_memory_graph_lib.InMemoryGraph:
  """Perturbs a nodeset's input features and its incident edgesets.

  For non-target nodesets, all input features of `nodeset_name` and all
  incident edgesets (`source == nodeset_name` or `target == nodeset_name`) are
  perturbed. For `target_nodeset` in a heterogeneous graph (multiple nodesets),
  all input features of `target_nodeset` (excluding `target_column`) and any
  self-edgesets (`source == target_nodeset` and `target == target_nodeset`) are
  perturbed so the target nodeset's own signal is isolated from other
  nodesets.

  Args:
    graph: The input `InMemoryGraph`.
    schema: The `GraphSchema` of the graph.
    nodeset_name: Name of the nodeset to perturb.
    rng: NumPy random generator.
    target_nodeset: Optional target nodeset of the prediction task.
    target_column: Optional target label feature name on `target_nodeset` to
      preserve untouched.
    edge_perturbation_mode: Mode used to perturb incident edgesets.
    include_incident_edges: Whether to perturb incident edgesets in addition to
      the nodeset's features.
    preserve_timestamps: Whether to preserve creation time features when
      perturbing.

  Returns:
    A new `InMemoryGraph` with the specified nodeset perturbed.
  """
  edge_perturbation_mode = parse_edge_perturbation_mode(edge_perturbation_mode)
  nodeset_schema = schema.node_sets[nodeset_name]
  creation_ts_feat = (
      temporal_util.creation_time_feature_name(nodeset_schema.features)
      if preserve_timestamps
      else None
  )

  # Shuffle all the input features. The label is not a model input, and the
  # creation time is kept so the temporal constraints remain valid.
  excluded = {creation_ts_feat}
  if nodeset_name == target_nodeset:
    excluded.add(target_column)
  nodeset = graph.node_sets[nodeset_name]
  new_features = _shuffle_features(
      nodeset.features,
      nodeset_schema.features,
      [f for f in nodeset_schema.features if f not in excluded],
      rng,
  )
  perturbed = dataclasses.replace(
      graph,
      node_sets={
          **graph.node_sets,
          nodeset_name: dataclasses.replace(nodeset, features=new_features),
      },
  )

  if include_incident_edges:
    is_hetero_target = (
        nodeset_name == target_nodeset and len(schema.node_sets) > 1
    )
    for edgeset_name, edgeset_schema in schema.edge_sets.items():
      if is_hetero_target:
        # Only the self-edgesets of the target nodeset, so that the other
        # nodesets keep their own signal.
        is_incident = (
            edgeset_schema.source == nodeset_name
            and edgeset_schema.target == nodeset_name
        )
      else:
        # All the edgesets connected to the nodeset.
        is_incident = (
            edgeset_schema.source == nodeset_name
            or edgeset_schema.target == nodeset_name
        )
      if is_incident:
        perturbed = perturb_edgeset(
            perturbed,
            schema,
            edgeset_name,
            rng,
            mode=edge_perturbation_mode,
            preserve_timestamps=preserve_timestamps,
        )

  return perturbed
