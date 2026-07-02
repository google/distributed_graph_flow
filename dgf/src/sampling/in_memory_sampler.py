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

"""In memory sampler."""

import os
from typing import List, Optional, Union, overload
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.sampling import _in_memory_sampler_ext
from dgf.src.sampling import config as config_lib
import numpy as np


class Sampler:
  """Sampler for generating subgraphs from an in-memory graph."""

  def __init__(
      self,
      cc_sampler,
      full_graph: in_memory_graph_lib.InMemoryGraph,
      return_features: bool,
      return_node_idxs: bool,
  ):
    self._cc_sampler = cc_sampler
    self._full_graph = full_graph
    self._return_features = return_features
    self._return_node_idxs = return_node_idxs

  def set_return_options(self, return_features: bool, return_node_idxs: bool):
    """Sets whether to return features and node indices in sampled graphs.

    Args:
      return_features: Whether to include feature values in the returned graph.
      return_node_idxs: Whether to include node indexes in the returned graph as
        a "#idx" node feature.
    """
    self._return_features = return_features
    self._return_node_idxs = return_node_idxs

  @overload
  def sample(
      self,
      seed_node_idxs: int,
      seed_timestamps: Optional[int] = None,
      masked_edge_idxs: Optional[int] = None,
  ) -> in_memory_graph_lib.InMemoryGraph:
    ...

  @overload
  def sample(
      self,
      seed_node_idxs: Union[np.ndarray, List[int]],
      seed_timestamps: Optional[Union[List[int], np.ndarray]] = None,
      masked_edge_idxs: Optional[Union[List[int], np.ndarray]] = None,
  ) -> List[in_memory_graph_lib.InMemoryGraph]:
    ...

  def sample(
      self,
      seed_node_idxs: Union[int, List[int], np.ndarray],
      seed_timestamps: Optional[Union[int, List[int], np.ndarray]] = None,
      masked_edge_idxs: Optional[Union[int, List[int], np.ndarray]] = None,
  ) -> Union[
      in_memory_graph_lib.InMemoryGraph,
      List[in_memory_graph_lib.InMemoryGraph],
  ]:
    """Samples one (or multiple) subgraphs.

    Grows one or more graph samples starting from the provided seed nodes. Each
    graph sample is constructed by randomly traversing edges and aggregating all
    visited edges and nodes.

    Args:
      seed_node_idxs: The index or indexes of the nodes to start sampling from.
        Sampling multiple nodes at the same time is more efficient that calling
        "sample" multiple times.
      seed_timestamps: Optional timestamps for time-aware sampling. If
        specified, only sample edges with a timestamp anterior (non strict) to
        the provided seed_timestamp. Should have the same length as
        "seed_node_idxs". Requires for the sampler to be initialized with some
        timestamps.
      masked_edge_idxs: Optional edge indices to mask during sampling. If
        specified, masks the specified edge. Should have the same length as
        "seed_node_idxs". Requires for the sampler to be initialized with a
        masked edgeset. If a seed node idx is -1, not edge filtering is done.

    Returns:
      An `InMemoryGraph` or list of `InMemoryGraph`
      representing the sampled subgraph.
    """

    # Check and convert the user input into what the c++ sampler expects.
    return_single_graph = False
    if isinstance(seed_node_idxs, int):
      return_single_graph = True
      seed_node_idxs = np.array([seed_node_idxs], dtype=np.int64)
    elif isinstance(seed_node_idxs, list):
      seed_node_idxs = np.array(seed_node_idxs, dtype=np.int64)
    elif not isinstance(seed_node_idxs, np.ndarray):
      raise ValueError(
          "seed_node_idxs must be an int, a list of ints, or a numpy array,"
          f" but got {type(seed_node_idxs)!r}."
      )

    if seed_timestamps is not None:
      if isinstance(seed_timestamps, int):
        seed_timestamps = np.array([seed_timestamps], dtype=np.int64)
      elif isinstance(seed_timestamps, list):
        seed_timestamps = np.array(seed_timestamps, dtype=np.int64)
      elif not isinstance(seed_timestamps, np.ndarray):
        raise ValueError(
            "seed_timestamps must be an int, a list of ints, or a numpy array,"
            f" but got {type(seed_timestamps)!r}."
        )
      if len(seed_timestamps) != len(seed_node_idxs):
        raise ValueError(
            "seed_timestamps must have the same length as seed_node_idxs"
        )

    if masked_edge_idxs is not None:
      if isinstance(masked_edge_idxs, int):
        masked_edge_idxs = np.array([masked_edge_idxs], dtype=np.int64)
      elif isinstance(masked_edge_idxs, list):
        masked_edge_idxs = np.array(masked_edge_idxs, dtype=np.int64)
      elif not isinstance(masked_edge_idxs, np.ndarray):
        raise ValueError(
            "masked_edge_idxs must be an int, a list of ints, or a numpy array,"
            f" but got {type(masked_edge_idxs)!r}."
        )
      if len(masked_edge_idxs) != len(seed_node_idxs):
        raise ValueError(
            "masked_edge_idxs must have the same length as seed_node_idxs"
        )

    # Sample a graph structure.
    graphs = self._cc_sampler.Sample(
        seed_node_idxs, seed_timestamps, masked_edge_idxs
    )

    self._add_finalize_graphs(graphs)

    if return_single_graph:
      return graphs[0]
    else:
      return graphs

  def subgraph(
      self, seed_node_idxs: List[int]
  ) -> in_memory_graph_lib.InMemoryGraph:
    """Extracts the subgraph around the provided seed nodes.

    This method returns a graph containing all the nodes and edges at a
    distance less than or equal to the configured number of hops from the
    provided `seed_node_idxs`.

    The seed nodes are always the first nodes of their respective node set in
    the returned `InMemoryGraph`. For example, if `seed_node_idxs` contains 3
    elements from a specific node set, these will correspond to the first 3
    nodes of that same node set in the extracted graph.

    Warning: Unlike "sample" that returns a different graph for each seed-node,
    "subgraph" returns a single possibly connected graph. To compute independent
    subgraphs, use `multisubgraph` instead.

    Usage example:
    ```python
    graph, schema = dgf.io.read_graph(<path to graph>)
    config = dgf.sampling.SimpleSamplingConfig(
        seed_nodeset="client",
        num_hops=4,
        hop_width=1, # Not used with "subgraph".
        reverse=True,
    )
    sampler = dgf.sampling.create_sampler(graph, config, schema)

    subgraph = sampler.subgraph([0,1,2])
    print(subgraph)
    ```

    Args:
      seed_node_idxs: The indexes of the nodes to start sampling from.

    Returns:
      The resulting graph.
    """

    graph = self._cc_sampler.SubGraph(seed_node_idxs)
    self._add_finalize_graphs([graph])
    return graph

  def multisubgraph(
      self, seed_node_idxs: List[int]
  ) -> List[in_memory_graph_lib.InMemoryGraph]:
    """Extracts the subgraphs around the provided seed nodes.

    This method returns the graphs containing all the nodes and edges at a
    distance less than or equal to the configured number of hops from the
    provided `seed_node_idxs`. Each seed-node leads to the creation of a
    different sub-graph independently.

    The seed nodes are always the first node of the extracted graph.

    Functionally, `multisubgraph` returns the same results as `sample` with an
    infinite width, but is massively more efficient. Both `multisubgraph` and
    `sample` use a graph traversal algorithm. However, `multisubgraph` doesn't
    re-visit nodes, which can make it more efficient.

    Usage example:
    ```python
    graph, schema = dgf.io.read_graph(<path to graph>)
    config = dgf.sampling.SimpleSamplingConfig(
        seed_nodeset="client",
        num_hops=4,
        hop_width=1, # Not used with "subgraph".
        reverse=True,
    )
    sampler = dgf.sampling.create_sampler(graph, config, schema)

    subgraph = sampler.subgraph([0,1,2])
    print(subgraph)
    ```

    Args:
      seed_node_idxs: The indexes of the nodes to start sampling from.

    Returns:
      A list of graphs. One for each "seed_node_idxs" value.
    """

    graphs = self._cc_sampler.MultiSubGraphs(seed_node_idxs)
    self._add_finalize_graphs(graphs)
    return graphs

  def __str__(self) -> str:
    return str(self._cc_sampler)

  def _add_finalize_graphs(
      self, graphs: List[in_memory_graph_lib.InMemoryGraph]
  ):
    """Adds features and removes temporary node indices based on settings.

    If `_return_features` is True, full feature values are added.
    If `_return_node_idxs` is False, the "#idx" feature is removed.

    Args:
      graphs: A list of `InMemoryGraph` objects to be finalized.
    """
    add_features_to_samples(
        self._full_graph, graphs, self._return_features, self._return_node_idxs
    )


def add_features_to_samples(
    full_graph: in_memory_graph_lib.InMemoryGraph,
    samples: List[in_memory_graph_lib.InMemoryGraph],
    return_features: bool,
    return_node_idxs: bool,
):
  """Adds features and optionally removes temporary node indices from sampled graphs.

  If `return_features` is True, full feature values are copied from the
  `full_graph` to the corresponding nodes in each graph within `graphs`.
  If `return_node_idxs` is False, the "#idx" feature, which contains the
  original node indices, is removed from each node set in the sampled graphs.

  Args:
    full_graph: The complete `InMemoryGraph` from which features are extracted.
    samples: A list of `InMemoryGraph` objects representing the sampled
      subgraphs. These graphs are modified in place.
    return_features: Whether to populate the sampled graphs with full feature
      values from `full_graph`.
    return_node_idxs: Whether to keep the "#idx" feature in the sampled graphs.
      If False, this feature is removed.
  """
  if return_features or not return_node_idxs:
    # Extract feature values.
    # TODO(gbm): Do this in C++.
    for sample in samples:
      for node_set_name, node_set in sample.node_sets.items():
        node_idxs = node_set.features["#idx"]
        if not return_node_idxs:
          del node_set.features["#idx"]
        if return_features:
          features = full_graph.node_sets[node_set_name].features
          for feature_name, full_feature_value in features.items():
            node_set.features[feature_name] = full_feature_value[node_idxs]


def create_sampler(
    graph: in_memory_graph_lib.InMemoryGraph,
    plan: Union[config_lib.SimpleSamplingConfig, config_lib.SamplingPlan],
    schema: schema_lib.GraphSchema,
    *,
    batch_size: Optional[int] = None,
    return_features: bool = True,
    return_node_idxs: bool = False,
    debug_sampling: bool = False,
    num_threads: Optional[int] = None,
    seed: Optional[int] = None,
    edgeset_to_mask: Optional[str] = None,
) -> Sampler:
  """Creates an in-memory sampler.

  Args:
    graph: The in-memory heterogeneous graph to sample from.
    plan: The sampling plan configuration. Can be a `SimpleSamplingConfig` or a
      `SamplingPlan`.
    schema: Graph schema. Required if `plan` is a `SimpleSamplingConfig`.
    batch_size: Number of samples you will sample each time. Sampling more /
      less samples at the same time is possible but possibly less efficient..
    return_features: Whether to include feature values in the returned graph.
    return_node_idxs: Whether to include node indexes in the returned graph in
      as a "#idx" node feature.
    debug_sampling: If true, enables a deterministic debug mode. In this mode,
      sampling always selects the first available edges, making the process
      fully reproducible.
    num_threads: Number of sampling threads. If None, select the number of
      threads automatically. Set num_threads=0 to disable multi-threading.
    seed: A positive integer to use as random seed for the sampler. If not
      provided, the seed is randomly initialized. Note that variation in the
      compilation can lead to variation (e.g., recompiling the binary might lead
      to different results--though this should be rare). Note: For writing unit
      tests,using debug_sampling=True is better.
    edgeset_to_mask: Optional edgeset name to mask during sampling.

  TODO(gbm): Should we remove the compilation variations (e.g., change in random
    number generator, change in hashmaps).

  Returns:
    A `Sampler` instance.
  """

  if plan.edgeset_timestamp_features and edgeset_to_mask is not None:
    raise ValueError(
        "Temporal filtering and edge masking cannot be used at the same time"
        " (yet)."
    )

  if isinstance(plan, config_lib.SimpleSamplingConfig):
    plan = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)

  if seed is None:
    seed = -1

  if batch_size is None and num_threads is None:
    raise ValueError(
        "At least one of 'batch_size' or 'num_threads' must be specified."
    )

  # TODO(gbm): Use batch_size for async sampling.
  if num_threads is None:
    num_threads = min(batch_size, os.cpu_count())  # pyrefly: ignore[bad-specialization]

  cc_sampler = _in_memory_sampler_ext.CreateSampler(
      graph, plan, debug_sampling, num_threads, seed, schema, edgeset_to_mask
  )
  return Sampler(
      cc_sampler,
      full_graph=graph,
      return_features=return_features,
      return_node_idxs=return_node_idxs,
  )
