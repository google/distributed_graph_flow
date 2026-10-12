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

from collections.abc import Callable
import copy
import dataclasses
import math
import os
from typing import overload
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.sampling import _in_memory_sampler_ext
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import temporal as sampling_temporal_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.util import temporal as temporal_util
import numpy as np

# Name of the feature containing the node indices in the sampled graphs.
_IDX_KEY = "#idx"

# Numpy dtype kinds gathered in c++ by `Sampler.sample_merged`: boolean,
# signed/unsigned integers, floats, and fixed-size bytes / unicode strings.
_CC_GATHER_DTYPE_KINDS = "biufSU"


class Sampler:
  """Sampler for generating subgraphs from an in-memory graph."""

  def __init__(
      self,
      cc_sampler,
      full_graph: in_memory_graph_lib.InMemoryGraph,
      schema: schema_lib.GraphSchema,
      return_features: bool,
      return_node_idxs: bool,
      slice_timeseries_by_seed: bool,
      max_timeseries_len: int,
      has_temporal_edgesets: bool,
      padding: padding_lib.Padding | None = None,
  ):
    self._cc_sampler = cc_sampler
    self._full_graph = full_graph
    self._schema = schema
    self._return_features = return_features
    self._return_node_idxs = return_node_idxs
    self._slice_timeseries_by_seed = slice_timeseries_by_seed
    self._max_timeseries_len = max_timeseries_len
    self._has_temporal_edgesets = has_temporal_edgesets
    self._timeseries_schema_cache = (
        temporal_util.extract_timeseries_schema_cache(self._schema)
    )

    if self._max_timeseries_len <= 0:
      raise ValueError("max_timeseries_len must be positive")

    merge_lib.validate_padding(self._schema, padding)
    self._padding = padding
    self._reset_merging()

  def set_return_options(self, return_features: bool, return_node_idxs: bool):
    """Sets whether to return features and node indices in sampled graphs.

    Args:
      return_features: Whether to include feature values in the returned graph.
      return_node_idxs: Whether to include node indexes in the returned graph as
        a "#idx" node feature.
    """
    self._return_features = return_features
    self._return_node_idxs = return_node_idxs
    self._reset_merging()

  @property
  def padding(self) -> padding_lib.Padding | None:
    """The padding of the graphs returned by `sample_merged`."""
    return self._padding

  def with_padding(self, padding: padding_lib.Padding | None) -> "Sampler":
    """Returns a copy of the sampler with a different `sample_merged` padding.

    The copy shares the underlying C++ sampler (including the sampling index and
    random number generator) with this sampler, so the two samplers are not
    thread-safe and must not be called concurrently.

    Args:
      padding: Padding of the graphs returned by `sample_merged`.

    Returns:
      A new sampler.
    """
    merge_lib.validate_padding(self._schema, padding)
    sampler = copy.copy(self)
    sampler._padding = padding  # pylint: disable=protected-access
    sampler._reset_merging()  # pylint: disable=protected-access
    return sampler

  @overload
  def sample(
      self,
      seed_node_idxs: int,
      seed_timestamps: int | None = None,
      masked_edge_idxs: int | None = None,
  ) -> in_memory_graph_lib.InMemoryGraph:
    ...

  @overload
  def sample(
      self,
      seed_node_idxs: np.ndarray | list[int],
      seed_timestamps: list[int] | np.ndarray | None = None,
      masked_edge_idxs: list[int] | np.ndarray | None = None,
  ) -> list[in_memory_graph_lib.InMemoryGraph]:
    ...

  def sample(
      self,
      seed_node_idxs: int | list[int] | np.ndarray,
      seed_timestamps: int | list[int] | np.ndarray | None = None,
      masked_edge_idxs: int | list[int] | np.ndarray | None = None,
  ) -> (
      in_memory_graph_lib.InMemoryGraph
      | list[in_memory_graph_lib.InMemoryGraph]
  ):
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

    return_single_graph = isinstance(seed_node_idxs, int)
    seed_node_idxs, seed_timestamps, masked_edge_idxs = (
        self._convert_sample_inputs(
            seed_node_idxs, seed_timestamps, masked_edge_idxs
        )
    )

    # Sample a graph structure.
    graphs = self._cc_sampler.Sample(
        seed_node_idxs,
        seed_timestamps if self._has_temporal_edgesets else None,
        masked_edge_idxs,
    )

    self._add_finalize_graphs(graphs, seed_timestamps=seed_timestamps)

    if return_single_graph:
      return graphs[0]
    else:
      return graphs

  def sample_merged(
      self,
      seed_node_idxs: int | list[int] | np.ndarray,
      seed_timestamps: int | list[int] | np.ndarray | None = None,
      masked_edge_idxs: int | list[int] | np.ndarray | None = None,
  ) -> tuple[in_memory_graph_lib.InMemoryGraph, dict[str, np.ndarray]]:
    """Samples subgraphs and merges them into a single (padded) graph.

    Functionally equivalent to (but much faster than) merging the graphs
    returned by `sample` with `dgf.transform.GraphMerger(schema, padding)`.

    Usage example:

    ```python
    sampler = dgf.sampling.create_sampler(graph, config, schema, batch_size=64,
                                          padding=padding)
    merged_graph, offsets = sampler.sample_merged(seed_node_idxs)
    # The seed nodes of the samples in the merged graph.
    merged_seed_node_idxs = offsets[config.seed_nodeset][:-1]
    ```

    The returned graph contains the node sets and edge sets of the sampler
    schema. If the sampler is configured with `return_features=True`, the node
    features of the schema are returned. If the sampler is configured with
    `return_node_idxs=True`, the node indices are returned in the "#idx"
    feature.

    Args:
      seed_node_idxs: The index or indexes of the nodes to start sampling from.
      seed_timestamps: Optional timestamps for time-aware sampling. See
        `sample`.
      masked_edge_idxs: Optional edge indices to mask during sampling. See
        `sample`.

    Returns:
      The merged graph, and the node offsets of each sample for each node set
      (with an extra sentinel value). See `GraphMerger` for details.
    """
    ((merged, offsets, _),) = self.sample_merged_sub_batches(
        seed_node_idxs, seed_timestamps, masked_edge_idxs
    )
    return merged, offsets

  def sample_merged_sub_batches(
      self,
      seed_node_idxs: int | list[int] | np.ndarray,
      seed_timestamps: int | list[int] | np.ndarray | None = None,
      masked_edge_idxs: int | list[int] | np.ndarray | None = None,
      *,
      skip_overflow_padding_error: bool = False,
      split_overflow_padding_error: bool = False,
      on_skip_samples: Callable[[int], None] | None = None,
  ) -> list[
      tuple[in_memory_graph_lib.InMemoryGraph, dict[str, np.ndarray], slice]
  ]:
    """Same as `sample_merged`, but splits / skips samples exceeding the padding.

    Equivalent to `GraphMerger.merge_sub_batches` applied on the output of
    `sample`.

    Args:
      seed_node_idxs: See `sample_merged`.
      seed_timestamps: See `sample_merged`.
      masked_edge_idxs: See `sample_merged`.
      skip_overflow_padding_error: See `GraphMerger.merge_sub_batches`.
      split_overflow_padding_error: See `GraphMerger.merge_sub_batches`.
      on_skip_samples: See `GraphMerger.merge_sub_batches`.

    Returns:
      A list of `(merged_graph, offsets, sub_slice)`.
    """
    seed_node_idxs, seed_timestamps, masked_edge_idxs = (
        self._convert_sample_inputs(
            seed_node_idxs, seed_timestamps, masked_edge_idxs
        )
    )
    if not self._merging_configured:
      self._configure_merging()

    if self._python_merger is not None:
      return list(
          self._python_merger.merge_sub_batches(
              self.sample(seed_node_idxs, seed_timestamps, masked_edge_idxs),
              skip_overflow_padding_error=skip_overflow_padding_error,
              split_overflow_padding_error=split_overflow_padding_error,
              on_skip_samples=on_skip_samples,
          )
      )

    res, batch, overflow = self._cc_sampler.SampleMerged(
        seed_node_idxs,
        seed_timestamps if self._has_temporal_edgesets else None,
        masked_edge_idxs,
        self._merge_config,
    )
    if overflow is None:
      merged, offsets = self._finalize_merged(*res)
      return [(merged, offsets, slice(0, len(seed_node_idxs)))]

    results = []
    self._merge_sub_batches(
        batch,
        begin=0,
        end=len(seed_node_idxs),
        overflow=overflow,
        skip_overflow_padding_error=skip_overflow_padding_error,
        split_overflow_padding_error=split_overflow_padding_error,
        on_skip_samples=on_skip_samples,
        results=results,
    )
    return results

  def _merge_sub_batches(
      self,
      batch: _in_memory_sampler_ext.SampledBatch,
      begin: int,
      end: int,
      overflow: tuple[bool, str, int, int] | None,
      skip_overflow_padding_error: bool,
      split_overflow_padding_error: bool,
      on_skip_samples: Callable[[int], None] | None,
      results: list[
          tuple[
              in_memory_graph_lib.InMemoryGraph, dict[str, np.ndarray], slice
          ]
      ],
  ):
    """Merges the samples [begin, end) of `batch` into `results`."""
    if overflow is None:
      res, overflow = batch.Merge(begin, end, self._merge_config)
      if overflow is None:
        merged, offsets = self._finalize_merged(*res)
        results.append((merged, offsets, slice(begin, end)))
        return

    if split_overflow_padding_error and end - begin > 1:
      mid = begin + (end - begin) // 2
      for sub_begin, sub_end in ((begin, mid), (mid, end)):
        self._merge_sub_batches(
            batch,
            begin=sub_begin,
            end=sub_end,
            overflow=None,
            skip_overflow_padding_error=skip_overflow_padding_error,
            split_overflow_padding_error=split_overflow_padding_error,
            on_skip_samples=on_skip_samples,
            results=results,
        )
      return
    if not skip_overflow_padding_error:
      is_node_set, set_name, required, padded = overflow
      if is_node_set:
        # Note: `required` includes the sentinel node.
        message = merge_lib.insufficient_node_padding_message(
            set_name, required - 1, padded
        )
      else:
        message = merge_lib.insufficient_edge_padding_message(
            set_name, required, padded
        )
      raise merge_lib.InsufficientPaddingError(message)
    if on_skip_samples is not None:
      on_skip_samples(end - begin)

  def _finalize_merged(
      self,
      merged: in_memory_graph_lib.InMemoryGraph,
      offsets: dict[str, np.ndarray],
  ) -> tuple[in_memory_graph_lib.InMemoryGraph, dict[str, np.ndarray]]:
    """Finalizes a merged graph exported by C++."""
    # Features not gathered in c++ (e.g. object arrays).
    for node_set_name, feature_name, src in self._python_gathered_features:
      node_set = merged.node_sets[node_set_name]
      assert node_set.num_nodes is not None
      num_real_nodes = int(offsets[node_set_name][-1])
      dst = np.empty((node_set.num_nodes,) + src.shape[1:], dtype=src.dtype)
      merge_lib.gather_rows_numpy(
          src, node_set.features[_IDX_KEY][:num_real_nodes], dst
      )
      node_set.features[feature_name] = dst
    for node_set_name in self._node_idxs_to_remove:
      del merged.node_sets[node_set_name].features[_IDX_KEY]
    return merged, offsets

  def _reset_merging(self):
    """Invalidates the configuration of `sample_merged`."""
    self._merging_configured = False
    self._python_merger: merge_lib.GraphMerger | None = None
    self._merge_config = None
    self._python_gathered_features: list[tuple[str, str, np.ndarray]] = []
    self._node_idxs_to_remove: list[str] = []

  def _configure_merging(self):
    """Pre-computes the configuration of `sample_merged`."""
    self._reset_merging()
    self._merging_configured = True

    if self._return_features and self._has_timeseries_features():
      # TODO(gbm): Support timeseries features (slicing by seed timestamp,
      # clipping to `max_timeseries_len`, and timeseries padding) in the c++
      # merging, and remove `_python_merger`.
      self._python_merger = merge_lib.GraphMerger(
          schema=self._output_schema(), padding=self._padding
      )
      return

    padding_num_nodes = {}
    for node_set_name in self._schema.node_sets:
      num_nodes = merge_lib.padded_num_nodes(self._padding, node_set_name)
      if num_nodes is not None:
        padding_num_nodes[node_set_name] = num_nodes

    padding_num_edges = {}
    for edge_set_name, edge_set_schema in self._schema.edge_sets.items():
      num_edges = merge_lib.padded_num_edges(self._padding, edge_set_name)
      if num_edges is None:
        continue
      if (
          edge_set_schema.source not in padding_num_nodes
          or edge_set_schema.target not in padding_num_nodes
      ):
        raise ValueError(
            merge_lib.missing_sentinel_message(edge_set_name, edge_set_schema)
        )
      padding_num_edges[edge_set_name] = num_edges

    features = {}
    for node_set_name, node_set_schema in self._schema.node_sets.items():
      feature_specs = []
      if self._return_features:
        full_features = self._full_graph.node_sets[node_set_name].features
        for feature_name in node_set_schema.features:
          if feature_name == _IDX_KEY:
            continue
          if feature_name not in full_features:
            raise ValueError(
                f"Feature '{feature_name}' of node set '{node_set_name}' is"
                " defined in the schema but not in the graph."
            )
          src = full_features[feature_name]
          cc_src = _cc_gather_source(src)
          if cc_src is None:
            self._python_gathered_features.append(
                (node_set_name, feature_name, src)
            )
          feature_specs.append(
              (feature_name, cc_src, src.dtype, (-1,) + src.shape[1:])
          )
      features[node_set_name] = feature_specs
      if not self._return_node_idxs and any(
          name == node_set_name for name, _, _ in self._python_gathered_features
      ):
        self._node_idxs_to_remove.append(node_set_name)

    self._merge_config = self._cc_sampler.CreateMergeConfig(
        padding_num_nodes,
        padding_num_edges,
        features,
        self._return_node_idxs,
    )

  def _convert_sample_inputs(
      self,
      seed_node_idxs: int | list[int] | np.ndarray,
      seed_timestamps: int | list[int] | np.ndarray | None,
      masked_edge_idxs: int | list[int] | np.ndarray | None,
  ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Checks and converts the user input into what the c++ sampler expects."""
    assert (
        not (
            self._return_features
            and self._slice_timeseries_by_seed
            and self._has_timeseries_features()
        )
        or seed_timestamps is not None
    ), (
        "`seed_timestamps` must be provided when"
        " `slice_timeseries_by_seed=True` and the schema contains"
        " `is_timeseries=True` features."
    )

    if isinstance(seed_node_idxs, int):
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
    return seed_node_idxs, seed_timestamps, masked_edge_idxs

  def _output_schema(self) -> schema_lib.GraphSchema:
    """Schema of the sampled graphs given the `return_*` options."""
    node_sets = {}
    for node_set_name, node_set_schema in self._schema.node_sets.items():
      features = {}
      if self._return_features:
        features.update({
            k: v for k, v in node_set_schema.features.items() if k != _IDX_KEY
        })
      if self._return_node_idxs:
        features[_IDX_KEY] = schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64
        )
      node_sets[node_set_name] = dataclasses.replace(
          node_set_schema, features=features
      )
    return dataclasses.replace(self._schema, node_sets=node_sets)

  def subgraph(
      self, seed_node_idxs: list[int]
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
      self, seed_node_idxs: list[int]
  ) -> list[in_memory_graph_lib.InMemoryGraph]:
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

  def _has_timeseries_features(self) -> bool:
    if self._timeseries_schema_cache is None:
      return False
    return self._timeseries_schema_cache.has_timeseries

  def _add_finalize_graphs(
      self,
      graphs: list[in_memory_graph_lib.InMemoryGraph],
      seed_timestamps: np.ndarray | None = None,
  ):
    """Adds features and removes temporary node indices based on settings.

    If `_return_features` is True, full feature values are added.
    If `_return_node_idxs` is False, the "#idx" feature is removed.
    If `_return_features` is True and the schema has timeseries features:
    - If `_slice_timeseries_by_seed` is True, timeseries features are causally
      filtered by the seed node timestamp and clipped to `max_timeseries_len`.
    - Otherwise, timeseries features are clipped to `max_timeseries_len`.

    Args:
      graphs: A list of `InMemoryGraph` objects to be finalized.
      seed_timestamps: Optional timestamps for causal timeseries filtering.
    """
    add_features_to_samples(
        self._full_graph, graphs, self._return_features, self._return_node_idxs
    )
    if self._has_timeseries_features() and self._return_features:
      if self._slice_timeseries_by_seed and seed_timestamps is None:
        raise ValueError(
            "`seed_timestamps` must be provided when"
            " `slice_timeseries_by_seed=True` and the schema contains"
            " `is_timeseries=True` features."
        )
      for i, sample in enumerate(graphs):
        target_timestamp = (
            int(seed_timestamps[i])
            if (self._slice_timeseries_by_seed and seed_timestamps is not None)
            else None
        )
        sampling_temporal_lib.extract_features_timeseries(
            graph=sample,
            timeseries_schema_cache=self._timeseries_schema_cache,
            target_timestamp=target_timestamp,
            max_timeseries_len=self._max_timeseries_len,
        )


def _cc_gather_source(src: np.ndarray) -> np.ndarray | None:
  """Returns the array to gather in c++ instead of `src`, or None.

  Fixed-size bytes / unicode arrays are viewed as uint8 arrays since DLPack does
  not support them. Other unsupported arrays (e.g. object arrays) return None.
  """
  if (
      not isinstance(src, np.ndarray)
      or src.ndim < 1
      or not src.flags.c_contiguous
      or src.dtype.kind not in _CC_GATHER_DTYPE_KINDS
  ):
    return None
  row_bytes = src.itemsize * math.prod(src.shape[1:])
  if row_bytes == 0:
    return None
  if src.dtype.kind in "SU":
    return src.view(np.uint8).reshape(src.shape[0], row_bytes)
  return src


def add_features_to_samples(
    full_graph: in_memory_graph_lib.InMemoryGraph,
    samples: list[in_memory_graph_lib.InMemoryGraph],
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
    plan: config_lib.SimpleSamplingConfig | config_lib.SamplingPlan,
    schema: schema_lib.GraphSchema,
    *,
    batch_size: int | None = None,
    return_features: bool = True,
    return_node_idxs: bool = False,
    debug_sampling: bool = False,
    num_threads: int | None = None,
    seed: int | None = None,
    edgeset_to_mask: str | None = None,
    slice_timeseries_by_seed: bool | None = None,
    padding: padding_lib.Padding | None = None,
) -> Sampler:
  """Creates an in-memory sampler.

  If temporal sampling is enabled and `plan.propagate_timestamp_to_edges` is
  true, the edgesets without a creation time get one derived from their
  connected nodes. `graph` and `schema` are not modified.

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
    slice_timeseries_by_seed: Whether to causally slice `is_timeseries=True`
      sequence features by the seed node timestamp. Defaults to
      `plan.temporal_sampling`.
    padding: Padding of the graphs returned by `Sampler.sample_merged`.

  TODO(gbm): Should we remove the compilation variations (e.g., change in random
    number generator, change in hashmaps).

  Returns:
    A `Sampler` instance.
  """

  if isinstance(plan, config_lib.SimpleSamplingConfig):
    plan = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)

  if slice_timeseries_by_seed is None:
    slice_timeseries_by_seed = plan.temporal_sampling

  # The creation time features are inferred from the schema: they are never
  # provided by the user.
  edgeset_timestamp_features = {}
  if plan.temporal_sampling:
    edgeset_timestamp_features = temporal_util.edgeset_timestamp_features(
        schema
    )

  if plan.temporal_sampling and edgeset_to_mask is not None:
    raise ValueError(
        "Temporal filtering and edge masking cannot be used at the same time"
        " (yet)."
    )

  if plan.temporal_sampling and plan.propagate_timestamp_to_edges:
    sampling_graph, edgeset_timestamp_features = (
        sampling_temporal_lib.propagate_timestamps_to_edges(
            graph=graph,
            schema=schema,
            edgeset_timestamp_features=edgeset_timestamp_features,
            edgesets=config_lib.edgesets_in_plan(plan),
        )
    )
  else:
    sampling_graph = graph

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
      sampling_graph,
      plan,
      debug_sampling,
      num_threads,
      seed,
      schema,
      edgeset_to_mask,
      edgeset_timestamp_features,
  )
  return Sampler(
      cc_sampler,
      full_graph=graph,  # The user graph, without the propagated timestamps.
      return_features=return_features,
      return_node_idxs=return_node_idxs,
      schema=schema,
      slice_timeseries_by_seed=slice_timeseries_by_seed,
      max_timeseries_len=plan.max_timeseries_len,
      has_temporal_edgesets=bool(edgeset_timestamp_features),
      padding=padding,
  )
