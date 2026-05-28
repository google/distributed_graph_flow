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

"""Prepare training dataset for the 10-line link prediction model.

The main utility GNNLinkDatasetPreparator generates normalized, padded and
batched graph samples for link prediction.

Mirrors the in_memory_gnn_dataset_preparator.py utility.
"""

import copy
import dataclasses
import itertools
from typing import Dict, Iterator, Literal, Optional, Tuple, Union
from dgf.src.analyse import in_process_feature_statistics as in_process_feature_statistics_lib
from dgf.src.analyse import padding as padding_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import padding as padding_data_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.generate import edge_neighbor_generator as edge_neighbor_generator_lib
from dgf.src.io import jax as jax_lib
from dgf.src.learning.ten_lines import node_prediction_dataset as node_prediction_dataset_lib
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import log
from dgf.src.util import util
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class LiveData:
  """Data computed during the "preparation" stage of the dataset preparator."""

  source_feature_stats: statistics_lib.GraphFeatureStatistics
  target_feature_stats: statistics_lib.GraphFeatureStatistics

  source_normalizer: normalize_lib.GraphNormalizer
  target_normalizer: normalize_lib.GraphNormalizer

  positive_source_padding: padding_data_lib.Padding
  positive_target_padding: padding_data_lib.Padding
  negative_target_padding: padding_data_lib.Padding

  source_sampling_plan: sampling_config_lib.SamplingPlan
  target_sampling_plan: sampling_config_lib.SamplingPlan

  source_sampler: in_memory_sampler_lib.Sampler
  target_sampler: in_memory_sampler_lib.Sampler

  num_edges_in_seed_edgeset: Optional[int]

  normalized_source_graph: Optional[in_memory_graph_lib.InMemoryGraph] = None
  normalized_target_graph: Optional[in_memory_graph_lib.InMemoryGraph] = None
  normalized_jax_source_graph: Optional[
      jax_in_memory_graph_lib.JaxInMemoryGraph
  ] = None
  normalized_jax_target_graph: Optional[
      jax_in_memory_graph_lib.JaxInMemoryGraph
  ] = None

  sampling_schema: Optional[schema_lib.GraphSchema] = None
  merge_schema: Optional[schema_lib.GraphSchema] = None


@dataclasses.dataclass(kw_only=True)
class GNNLinkDatasetPreparatorSample:
  positive_source_graph: in_memory_graph_lib.InMemoryGraph
  positive_target_graph: in_memory_graph_lib.InMemoryGraph
  negative_target_graph: in_memory_graph_lib.InMemoryGraph

  positive_source_offsets: Dict[str, np.ndarray]
  positive_target_offsets: Dict[str, np.ndarray]
  negative_target_offsets: Dict[str, np.ndarray]


@dataclasses.dataclass(kw_only=True)
class GNNLinkDatasetPreparatorJaxSample:
  positive_source_graph: jax_in_memory_graph_lib.JaxInMemoryGraph
  positive_target_graph: jax_in_memory_graph_lib.JaxInMemoryGraph
  negative_target_graph: jax_in_memory_graph_lib.JaxInMemoryGraph

  positive_source_offsets: Dict[str, jnp.ndarray]
  positive_target_offsets: Dict[str, jnp.ndarray]
  negative_target_offsets: Dict[str, jnp.ndarray]


@dataclasses.dataclass
class NodeIdsBatch:
  pos_src_node_idxs: np.ndarray
  pos_trg_node_idxs: np.ndarray
  neg_trg_node_idxs: np.ndarray
  edge_idxs: np.ndarray


@dataclasses.dataclass(kw_only=True)
class GNNLinkDatasetPreparator:
  """Generates graph samples to train link prediction models.

  Before batching, each sample is composed of a positive source node, a positive
  target node, and "num_negative_nodes" negative target nodes.

  The available negative sampling strategies are:
  - Uniform negative.

  TODO(gbm): Add other negative sampling strategies.

  The following transformations are applied:
  - Feature normalization (e.g., soft quantile, indexing) using
    `dgf.analyse.feature_statistics_from_graphs` and
    `dgf.transform.AutoNormalizer`.
  - Padding graphs using `dgf.analyse.padding_from_graph_generator`.
  - Merging of batches of graphs using `dgf.transform.merge_graphs`.

  Attributes:
    graph: One of the graph format defined in data.Graph e.g. in-memory graph,
      generator of graph samples, path to graph samples.
    schema: The schema of the graph. The semantics of the features should be
      configured.
    batch_size: The desired size of each batch of seed edges. Set batch_size=1
      to avoid batching / merging.
    sampling_config: Configuration for sampling subgraphs around the seed nodes.
    source_sampling_plan: Configuration for sampling subgraphs around the seed
      nodes. If specified, replaces sampling_config for the source node.
    target_sampling_plan: Configuration for sampling subgraphs around the seed
      nodes. If specified, replaces sampling_config for the target node.
    drop_remainder: If `True`, the last batch of seed edges will be dropped if
      it contains fewer than `batch_size` elements.
    shuffle: If `True`, the seed edges are shuffled before batching.
    target_edgeset: The name of the edgeset in the schema that represents the
      links for which predictions are made.
    num_negative_nodes: The number of negative target nodes to sample for each
      positive target node.
    edge_neighbor_generator: Configuration for generating edge neighbors,
      including negative sampling strategies.
    seed_edge_idxs: Optional array of edge indices from `target_edgeset` to use
      as seeds. If None, all edges in `target_edgeset` are used.
    mask_seed_edge: Mask the seed edge from the sampling.
    mask_target_edgeset: Mask the seed egdgeset from the sampling.
    num_samples_for_stats: Number of graph samples used to compute feature and
      graph statistics, which are required for feature normalization and
      padding. Larger values lead to more precise statistics but increase the
      time taken during the initial preparation phase. If set to `None`,
      statistics will be computed using one sample for each edge in the seed
      edgeset.
    auto_normalize_config: Configuration for automatic feature normalization.
      Defaults to `normalize_lib.AutoNormalizeConfig()`.
    verbose_preparation: If true, display a progress bar during the preparation
      stage that appears only the first time `generate` is called.
    skip_overflow_padding_error: If padding is set, the merging stage can fail
      if the "padding" is not large enought. If
      skip_overflow_padding_error=True, such batch is skipped. If
      skip_overflow_padding_error=False, and error is raised.
  """

  # Required arguments
  graph: in_memory_graph_lib.InMemoryGraph
  schema: schema_lib.GraphSchema
  batch_size: int
  sampling_config: sampling_config_lib.SimpleSamplingConfig
  source_sampling_plan: Optional[sampling_config_lib.SamplingPlan] = None
  target_sampling_plan: Optional[sampling_config_lib.SamplingPlan] = None
  drop_remainder: bool
  shuffle: bool
  target_edgeset: str
  num_negative_nodes: int
  edge_neighbor_generator: (
      edge_neighbor_generator_lib.EdgeNeighborGeneratorConfig
  )
  seed_edge_idxs: Optional[np.ndarray] = None
  mask_seed_edge: bool = False
  mask_target_edgeset: bool = False

  cache_normalized_features: bool = False
  cache_normalized_features_device: Literal["host", "device"] = "device"

  # Optional arguments
  num_samples_for_stats: Optional[int] = 10000
  auto_normalize_config: normalize_lib.AutoNormalizeConfig = dataclasses.field(
      default_factory=normalize_lib.AutoNormalizeConfig
  )
  verbose_preparation: bool = True
  skip_overflow_padding_error: bool = False

  # The is_prepared data computed by the `prepare()` method.
  live: Optional[LiveData] = dataclasses.field(init=False, default=None)

  def __post_init__(self):
    if self.batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {self.batch_size}")
    if self.mask_seed_edge and self.mask_target_edgeset:
      raise ValueError(
          "`mask_seed_edge` and `mask_target_edgeset` cannot both be True. If"
          " `mask_target_edgeset` is True, the entire target edgeset is removed"
          " from the sampling schema, making `mask_seed_edge` redundant."
      )

  def get_live(self) -> LiveData:
    if self.live is None:
      raise RuntimeError(
          "The dataset preparator has not been is_prepared yet. Call"
          " `prepare()` or `generate()` at least once before calling"
          " `generated_schema()`."
      )
    return self.live

  def is_prepared(self) -> bool:
    return self.live is not None

  def num_edge_in_seed_edgeset(self) -> Optional[int]:
    """Number of nodes in the seed nodeset."""
    return self.get_live().num_edges_in_seed_edgeset

  def prepare_from_existing_one(self, other: "GNNLinkDatasetPreparator"):
    """Pre-compute data (same as "prepare") but with an already computed cache.

    Instead of being recomputed, the following are grabbed from "other":
      - Feature statistics
      - Normalizer
      - Padding
      - Sampling plan

    Args:
      other: Another dataset preparator from which to copy the data.
    """
    other_live = other.get_live()

    sampling_schema = self._sampling_schema()
    edgeset_to_mask = self._edgeset_to_mask()

    # TODO(gbm): Don't instantiate a new sampler if this is using the same data
    # as the "other" preparator.

    return_features = not self.cache_normalized_features
    return_node_idxs = self.cache_normalized_features

    source_sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=other_live.source_sampling_plan,
        schema=sampling_schema,
        batch_size=self.batch_size,
        edgeset_to_mask=edgeset_to_mask,
        return_features=return_features,
        return_node_idxs=return_node_idxs,
    )
    target_sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=other_live.target_sampling_plan,
        schema=sampling_schema,
        batch_size=self.batch_size * self.num_negative_nodes,
        edgeset_to_mask=edgeset_to_mask,
        return_features=return_features,
        return_node_idxs=return_node_idxs,
    )

    self.live = LiveData(
        source_feature_stats=other_live.source_feature_stats,
        target_feature_stats=other_live.target_feature_stats,
        source_normalizer=other_live.source_normalizer,
        target_normalizer=other_live.target_normalizer,
        positive_source_padding=other_live.positive_source_padding,
        positive_target_padding=other_live.positive_target_padding,
        negative_target_padding=other_live.negative_target_padding,
        source_sampling_plan=other_live.source_sampling_plan,
        target_sampling_plan=other_live.target_sampling_plan,
        source_sampler=source_sampler,
        target_sampler=target_sampler,
        num_edges_in_seed_edgeset=len(self.seed_edge_idxs)
        if self.seed_edge_idxs is not None
        else self.graph.edge_sets[self.target_edgeset].num_edges(),
        sampling_schema=sampling_schema,
        merge_schema=self._get_merge_schema(sampling_schema),
    )

    if self.cache_normalized_features:
      if self.graph is other.graph:
        self.live.normalized_source_graph = other_live.normalized_source_graph
        self.live.normalized_target_graph = other_live.normalized_target_graph
        self.live.normalized_jax_source_graph = (
            other_live.normalized_jax_source_graph
        )
        self.live.normalized_jax_target_graph = (
            other_live.normalized_jax_target_graph
        )
      else:
        if self.cache_normalized_features_device == "host":
          self.live.normalized_source_graph = (
              other_live.source_normalizer.normalize_numpy(self.graph)
          )
          self.live.normalized_target_graph = (
              other_live.target_normalizer.normalize_numpy(self.graph)
          )
        elif self.cache_normalized_features_device == "device":
          self.live.normalized_jax_source_graph = (
              other_live.source_normalizer.normalize_numpy_to_jax(
                  self.graph, include_adjacencies=False
              )
          )
          self.live.normalized_jax_target_graph = (
              other_live.target_normalizer.normalize_numpy_to_jax(
                  self.graph, include_adjacencies=False
              )
          )
        else:
          raise ValueError(
              "Unsupported `cache_normalized_features_device`: "
              f"{self.cache_normalized_features_device}"
          )

  def _build_neighbor_generator(
      self, sampler: Optional[in_memory_sampler_lib.Sampler]
  ) -> edge_neighbor_generator_lib.EdgeNeighborGenerator:
    return edge_neighbor_generator_lib.EdgeNeighborGenerator(
        self.graph,
        self.schema,
        self.target_edgeset,
        self.num_negative_nodes,
        config=self.edge_neighbor_generator,
        sampler=sampler,
    )

  def _sampling_schema(self) -> schema_lib.GraphSchema:
    if self.mask_target_edgeset:
      sampling_schema = copy.deepcopy(self.schema)
      del sampling_schema.edge_sets[self.target_edgeset]
      return sampling_schema
    return self.schema

  def _edgeset_to_mask(self) -> Optional[str]:
    if self.mask_seed_edge:
      return self.target_edgeset
    else:
      return None

  def _get_merge_schema(
      self, base_schema: schema_lib.GraphSchema
  ) -> schema_lib.GraphSchema:
    if not self.cache_normalized_features:
      return base_schema
    node_sets = {}
    for name, _ in base_schema.node_sets.items():
      node_sets[name] = schema_lib.NodeSchema(
          features={
              "#idx": schema_lib.FeatureSchema(
                  format=schema_lib.FeatureFormat.INTEGER_64
              )
          }
      )

    edge_sets = {}
    for name, edge_schema in base_schema.edge_sets.items():
      edge_sets[name] = schema_lib.EdgeSchema(
          source=edge_schema.source, target=edge_schema.target
      )

    return schema_lib.GraphSchema(node_sets=node_sets, edge_sets=edge_sets)

  def prepare(self):
    """Pre-compute and prepare what is necessary for the generation.

    Can only be called once. Called automatically the first time "generate" is
    called.
    """
    if self.live is not None:
      raise ValueError("prepare() can only be called once.")

    # TODO(gbm): Add support for pre-generated graph samples.
    self._prepare_on_in_memory_graph()

  def _in_memory_generate_node_ids(
      self,
      nei_generator: edge_neighbor_generator_lib.EdgeNeighborGenerator,
  ) -> Iterator[NodeIdsBatch]:
    """Generates batches of pos src nodes, pos trg nodes, and neg trg nodes."""

    target_edgeset_data = self.graph.edge_sets[self.target_edgeset]

    for batch_edge_idxs in util.batch_indices_generator(
        self.seed_edge_idxs
        if self.seed_edge_idxs is not None
        else target_edgeset_data.num_edges(),
        batch_size=self.batch_size,
        drop_remainder=self.drop_remainder,
        shuffle=self.shuffle,
    ):
      nei = nei_generator.generate(batch_edge_idxs)
      yield NodeIdsBatch(
          pos_src_node_idxs=nei.pos_src_node_idxs,
          pos_trg_node_idxs=nei.pos_trg_node_idxs,
          neg_trg_node_idxs=nei.neg_trg_node_idxs,
          edge_idxs=batch_edge_idxs,
      )

  def _prepare_on_in_memory_graph(self):
    """Prepares the dataset from an in-memory graph."""

    # Create sampler
    if self.verbose_preparation:
      log.info("Create graph sampler")

    if (
        self.sampling_config.seed_nodeset
        != self.schema.edge_sets[self.target_edgeset].source
    ):
      raise ValueError(
          "The `sampling_config.seed_nodeset` must match the source nodeset of "
          "the `target_edgeset` when preparing for link prediction. "
          f"Expected '{self.schema.edge_sets[self.target_edgeset].source}', "
          f"but got '{self.sampling_config.seed_nodeset}'."
      )

    # TODO(gbm): Require to either exclude the target edgeset from the sampling,
    # enable temporal aware sampling on the target edgeset, or mask the target
    # edge in the sampling.

    sampling_schema = self._sampling_schema()
    edgeset_to_mask = self._edgeset_to_mask()

    source_sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=self.source_sampling_plan
        if self.source_sampling_plan is not None
        else self.sampling_config,
        schema=sampling_schema,
        batch_size=self.batch_size,
        edgeset_to_mask=edgeset_to_mask,
    )
    target_plan = dataclasses.replace(
        self.sampling_config,
        seed_nodeset=self.schema.edge_sets[self.target_edgeset].target,
    )
    target_sampler = in_memory_sampler_lib.create_sampler(
        graph=self.graph,
        plan=self.target_sampling_plan
        if self.target_sampling_plan is not None
        else target_plan,
        schema=sampling_schema,
        batch_size=self.batch_size * self.num_negative_nodes,
        edgeset_to_mask=edgeset_to_mask,
    )

    nei_generator = self._build_neighbor_generator(source_sampler)

    # Generate raw samples to compute feature statistics and data normalization.

    def gen_raw_source_samples() -> Iterator[in_memory_graph_lib.InMemoryGraph]:
      """Generate graph samples for the source nodeset."""
      while True:
        for batch_seed in self._in_memory_generate_node_ids(nei_generator):
          samples = source_sampler.sample(
              batch_seed.pos_src_node_idxs,
              masked_edge_idxs=batch_seed.edge_idxs
              if self.mask_seed_edge
              else None,
          )
          for sample in samples:
            yield sample

    def gen_raw_target_samples() -> Iterator[in_memory_graph_lib.InMemoryGraph]:
      """Generate graph samples for the target nodeset."""
      while True:
        for batch_seed in self._in_memory_generate_node_ids(nei_generator):
          # We generate both positive and negative samples.
          pos_trg_samples = target_sampler.sample(
              batch_seed.pos_trg_node_idxs,
              masked_edge_idxs=batch_seed.edge_idxs
              if self.mask_seed_edge
              else None,
          )
          neg_trg_samples = target_sampler.sample(
              batch_seed.neg_trg_node_idxs.flatten(),
              masked_edge_idxs=np.repeat(
                  batch_seed.edge_idxs, self.num_negative_nodes, axis=0
              )
              if self.mask_seed_edge
              else None,
          )
          for sample in pos_trg_samples:
            yield sample
          for sample in neg_trg_samples:
            yield sample

    gen_raw_source_samples_iter = gen_raw_source_samples()
    gen_raw_target_samples_iter = gen_raw_target_samples()
    if self.num_samples_for_stats is not None:
      # Limit the number of samples used to compute the feature stats.
      gen_raw_source_samples_iter = itertools.islice(
          gen_raw_source_samples_iter,
          self.num_samples_for_stats // self.batch_size,
      )
      gen_raw_target_samples_iter = itertools.islice(
          gen_raw_target_samples_iter,
          self.num_samples_for_stats
          // (self.batch_size * (1 + self.num_negative_nodes)),
      )

    # Compute the feature statistics (for the feature normalization)
    if self.verbose_preparation:
      log.info("Compute feature statistics")
    source_feature_stats = (
        in_process_feature_statistics_lib.feature_statistics_from_graphs(
            gen_raw_source_samples_iter, sampling_schema
        )
    )
    target_feature_stats = (
        in_process_feature_statistics_lib.feature_statistics_from_graphs(
            gen_raw_target_samples_iter, sampling_schema
        )
    )

    if self.verbose_preparation:
      log.info("  Source stats:\n%s", source_feature_stats)
      log.info("  Target stats:\n%s", target_feature_stats)

    source_normalizer = normalize_lib.auto_normalize(
        schema=sampling_schema,
        stats=source_feature_stats,
        config=self.auto_normalize_config,
    )
    target_normalizer = normalize_lib.auto_normalize(
        schema=sampling_schema,
        stats=target_feature_stats,
        config=self.auto_normalize_config,
    )

    # Generate samples to estimate padding.
    # Note: Unlike feature statistics, positive and negative target samples are
    # processed independently for padding, requiring separate padding
    # calculations.

    def gen_normalized_merged_positive_source_samples() -> (
        Iterator[in_memory_graph_lib.InMemoryGraph]
    ):
      while True:
        for batch_seed in self._in_memory_generate_node_ids(nei_generator):
          samples = source_sampler.sample(
              batch_seed.pos_src_node_idxs,
              masked_edge_idxs=batch_seed.edge_idxs
              if self.mask_seed_edge
              else None,
          )
          merged_samples, _ = merge_lib.merge_graphs(
              samples, sampling_schema, padding=None, sentinel_offset=True
          )
          yield source_normalizer.normalize_numpy(merged_samples)

    def gen_normalized_merged_positive_target_samples() -> (
        Iterator[in_memory_graph_lib.InMemoryGraph]
    ):
      while True:
        for batch_seed in self._in_memory_generate_node_ids(nei_generator):
          samples = target_sampler.sample(
              batch_seed.pos_trg_node_idxs,
              masked_edge_idxs=batch_seed.edge_idxs
              if self.mask_seed_edge
              else None,
          )
          merged_samples, _ = merge_lib.merge_graphs(
              samples, sampling_schema, padding=None, sentinel_offset=True
          )
          yield target_normalizer.normalize_numpy(merged_samples)

    def gen_normalized_merged_negative_target_samples() -> (
        Iterator[in_memory_graph_lib.InMemoryGraph]
    ):
      while True:
        for batch_seed in self._in_memory_generate_node_ids(nei_generator):
          samples = target_sampler.sample(
              batch_seed.neg_trg_node_idxs.flatten(),
              masked_edge_idxs=np.repeat(
                  batch_seed.edge_idxs, self.num_negative_nodes, axis=0
              )
              if self.mask_seed_edge
              else None,
          )
          merged_samples, _ = merge_lib.merge_graphs(
              samples, sampling_schema, padding=None, sentinel_offset=True
          )
          yield target_normalizer.normalize_numpy(merged_samples)

    gen_normalized_merged_positive_source_samples_iter = (
        gen_normalized_merged_positive_source_samples()
    )
    gen_normalized_merged_positive_target_samples_iter = (
        gen_normalized_merged_positive_target_samples()
    )
    gen_normalized_merged_negative_target_samples_iter = (
        gen_normalized_merged_negative_target_samples()
    )
    if self.num_samples_for_stats is not None:
      # Limit the number of samples used to compute the paddings
      gen_normalized_merged_positive_source_samples_iter = itertools.islice(
          gen_normalized_merged_positive_source_samples_iter,
          self.num_samples_for_stats // self.batch_size,
      )
      gen_normalized_merged_positive_target_samples_iter = itertools.islice(
          gen_normalized_merged_positive_target_samples_iter,
          self.num_samples_for_stats // self.batch_size,
      )
      gen_normalized_merged_negative_target_samples_iter = itertools.islice(
          gen_normalized_merged_negative_target_samples_iter,
          self.num_samples_for_stats
          // (self.batch_size * self.num_negative_nodes),
      )

    # Compute the batched graph statistics (for the padding)
    if self.verbose_preparation:
      log.info("Compute graph statistics for padding")

    positive_source_padding = padding_lib.padding_from_graph_generator(
        sampling_schema, gen_normalized_merged_positive_source_samples_iter
    )
    positive_target_padding = padding_lib.padding_from_graph_generator(
        sampling_schema, gen_normalized_merged_positive_target_samples_iter
    )
    negative_target_padding = padding_lib.padding_from_graph_generator(
        sampling_schema, gen_normalized_merged_negative_target_samples_iter
    )

    if self.verbose_preparation:
      log.info(
          "  positive source padding: %s",
          padding_lib.print_padding(
              positive_source_padding, return_output=True, header=False
          ),
      )
      log.info(
          "  positive target padding: %s",
          padding_lib.print_padding(
              positive_target_padding, return_output=True, header=False
          ),
      )
      log.info(
          "  negative target padding: %s",
          padding_lib.print_padding(
              negative_target_padding, return_output=True, header=False
          ),
      )

    self.live = LiveData(
        source_feature_stats=source_feature_stats,
        target_feature_stats=target_feature_stats,
        source_normalizer=source_normalizer,
        target_normalizer=target_normalizer,
        positive_source_padding=positive_source_padding,
        positive_target_padding=positive_target_padding,
        negative_target_padding=negative_target_padding,
        source_sampling_plan=sampling_config_lib.simple_sampling_config_to_sampling_plan(
            self.sampling_config, self.schema
        )
        if isinstance(
            self.sampling_config, sampling_config_lib.SimpleSamplingConfig
        )
        else self.sampling_config,
        target_sampling_plan=sampling_config_lib.simple_sampling_config_to_sampling_plan(
            target_plan, self.schema
        )
        if isinstance(target_plan, sampling_config_lib.SimpleSamplingConfig)
        else target_plan,
        source_sampler=source_sampler,
        target_sampler=target_sampler,
        num_edges_in_seed_edgeset=len(self.seed_edge_idxs)
        if self.seed_edge_idxs is not None
        else self.graph.edge_sets[self.target_edgeset].num_edges(),
        sampling_schema=sampling_schema,
        merge_schema=self._get_merge_schema(sampling_schema),
    )

    if self.cache_normalized_features:
      # Configure samplers to return node idxs only.
      source_sampler.set_return_options(
          return_features=False, return_node_idxs=True
      )
      target_sampler.set_return_options(
          return_features=False, return_node_idxs=True
      )

      if self.cache_normalized_features_device == "host":
        self.live.normalized_source_graph = source_normalizer.normalize_numpy(
            self.graph
        )
        self.live.normalized_target_graph = target_normalizer.normalize_numpy(
            self.graph
        )
      elif self.cache_normalized_features_device == "device":
        self.live.normalized_jax_source_graph = (
            source_normalizer.normalize_numpy_to_jax(
                self.graph, include_adjacencies=False
            )
        )
        self.live.normalized_jax_target_graph = (
            target_normalizer.normalize_numpy_to_jax(
                self.graph, include_adjacencies=False
            )
        )
      else:
        raise ValueError(
            "Unsupported `cache_normalized_features_device`: "
            f"{self.cache_normalized_features_device}"
        )

  def _sample_and_merge(
      self,
      live: LiveData,
      batch_seed: NodeIdsBatch,
      merge_schema: schema_lib.GraphSchema,
      padding: bool = True,
  ) -> GNNLinkDatasetPreparatorSample:
    """Samples subgraphs and merges them into batched graphs.

    Args:
      live: The live data computed during preparation.
      batch_seed: The seed nodes and edge indices for sampling.
      merge_schema: The schema to use for merging graphs.
      padding: Whether to pad the merged graphs.

    Returns:
      A tuple containing:
        - A tuple of merged graphs: positive source, positive target, negative
          target.
        - A tuple of merge offsets dictionaries for each merged graph.
    """
    # Positive source
    pos_src_samples = live.source_sampler.sample(
        batch_seed.pos_src_node_idxs,
        masked_edge_idxs=batch_seed.edge_idxs if self.mask_seed_edge else None,
    )
    pos_src_merged, pos_src_offsets = merge_lib.merge_graphs(
        pos_src_samples,
        merge_schema,
        padding=live.positive_source_padding if padding else None,
        sentinel_offset=True,
    )

    # Positive target
    pos_trg_samples = live.target_sampler.sample(
        batch_seed.pos_trg_node_idxs,
        masked_edge_idxs=batch_seed.edge_idxs if self.mask_seed_edge else None,
    )
    pos_trg_merged, pos_trg_offsets = merge_lib.merge_graphs(
        pos_trg_samples,
        merge_schema,
        padding=live.positive_target_padding if padding else None,
        sentinel_offset=True,
    )

    # Negative target
    neg_trg_samples = live.target_sampler.sample(
        batch_seed.neg_trg_node_idxs.flatten(),
        masked_edge_idxs=np.repeat(
            batch_seed.edge_idxs, self.num_negative_nodes, axis=0
        )
        if self.mask_seed_edge
        else None,
    )
    neg_trg_merged, neg_trg_offsets = merge_lib.merge_graphs(
        neg_trg_samples,
        merge_schema,
        padding=live.negative_target_padding if padding else None,
        sentinel_offset=True,
    )

    return GNNLinkDatasetPreparatorSample(
        positive_source_graph=pos_src_merged,
        positive_target_graph=pos_trg_merged,
        negative_target_graph=neg_trg_merged,
        positive_source_offsets=pos_src_offsets,
        positive_target_offsets=pos_trg_offsets,
        negative_target_offsets=neg_trg_offsets,
    )

  def _generate_one(
      self,
      live: LiveData,
      batch_seed: NodeIdsBatch,
      padding: bool = True,
  ) -> GNNLinkDatasetPreparatorSample:
    raw = self._sample_and_merge(live, batch_seed, live.merge_schema, padding)

    if self.cache_normalized_features:
      if (
          live.normalized_source_graph is None
          or live.normalized_target_graph is None
      ):
        raise RuntimeError(
            "Cached normalized graph is not available in numpy format. If using"
            " device cache, use generate_jax()."
        )
      pos_src_graph = (
          node_prediction_dataset_lib.attach_features_from_numpy_graph(
              live.normalized_source_graph, raw.positive_source_graph
          )
      )
      pos_trg_graph = (
          node_prediction_dataset_lib.attach_features_from_numpy_graph(
              live.normalized_target_graph, raw.positive_target_graph
          )
      )
      neg_trg_graph = (
          node_prediction_dataset_lib.attach_features_from_numpy_graph(
              live.normalized_target_graph, raw.negative_target_graph
          )
      )
    else:
      pos_src_graph = live.source_normalizer.normalize_numpy(
          raw.positive_source_graph
      )
      pos_trg_graph = live.target_normalizer.normalize_numpy(
          raw.positive_target_graph
      )
      neg_trg_graph = live.target_normalizer.normalize_numpy(
          raw.negative_target_graph
      )

    return GNNLinkDatasetPreparatorSample(
        positive_source_graph=pos_src_graph,
        positive_target_graph=pos_trg_graph,
        negative_target_graph=neg_trg_graph,
        positive_source_offsets=raw.positive_source_offsets,
        positive_target_offsets=raw.positive_target_offsets,
        negative_target_offsets=raw.negative_target_offsets,
    )

  def generate(
      self,
  ) -> Iterator[GNNLinkDatasetPreparatorSample]:
    """Generates graph samples and the merging node offsets.

    Yields:
      A tuple containing the normalized, merged, and padded graph sample, and
      the merging node offsets.
    """

    live = self.get_live()
    nei_generator = self._build_neighbor_generator(live.source_sampler)
    for batch_seed in self._in_memory_generate_node_ids(nei_generator):

      try:
        yield self._generate_one(live, batch_seed)

      except ValueError as e:
        if self.skip_overflow_padding_error:
          log.warning(
              (
                  "Skipping batch due to padding overflow. Consider increasing"
                  " num_samples_for_stats. Error: %s"
              ),
              e,
          )
          continue
        raise e

  def _generate_one_jax(
      self,
      live: LiveData,
      batch_seed: NodeIdsBatch,
      padding: bool = True,
  ) -> GNNLinkDatasetPreparatorJaxSample:
    raw = self._sample_and_merge(live, batch_seed, live.merge_schema, padding)

    if self.cache_normalized_features:
      norm_src = (
          live.normalized_jax_source_graph or live.normalized_source_graph
      )
      norm_trg = (
          live.normalized_jax_target_graph or live.normalized_target_graph
      )
      pos_src_jax = node_prediction_dataset_lib.attach_features_from_jax_graph_and_cast_to_jax(
          norm_src, raw.positive_source_graph
      )
      pos_trg_jax = node_prediction_dataset_lib.attach_features_from_jax_graph_and_cast_to_jax(
          norm_trg, raw.positive_target_graph
      )
      neg_trg_jax = node_prediction_dataset_lib.attach_features_from_jax_graph_and_cast_to_jax(
          norm_trg, raw.negative_target_graph
      )
    else:
      pos_src_norm = live.source_normalizer.normalize_numpy(
          raw.positive_source_graph
      )
      pos_trg_norm = live.target_normalizer.normalize_numpy(
          raw.positive_target_graph
      )
      neg_trg_norm = live.target_normalizer.normalize_numpy(
          raw.negative_target_graph
      )

      pos_src_jax = jax_lib.graph_to_jax_graph(pos_src_norm)
      pos_trg_jax = jax_lib.graph_to_jax_graph(pos_trg_norm)
      neg_trg_jax = jax_lib.graph_to_jax_graph(neg_trg_norm)

    return GNNLinkDatasetPreparatorJaxSample(
        positive_source_graph=pos_src_jax,
        positive_target_graph=pos_trg_jax,
        negative_target_graph=neg_trg_jax,
        positive_source_offsets={
            k: jnp.asarray(v) for k, v in raw.positive_source_offsets.items()
        },
        positive_target_offsets={
            k: jnp.asarray(v) for k, v in raw.positive_target_offsets.items()
        },
        negative_target_offsets={
            k: jnp.asarray(v) for k, v in raw.negative_target_offsets.items()
        },
    )

  def generate_jax(
      self,
  ) -> Iterator[GNNLinkDatasetPreparatorJaxSample]:
    """Generates jax batched + normalized graph samples."""

    live = self.get_live()
    nei_generator = self._build_neighbor_generator(live.source_sampler)
    for batch_seed in self._in_memory_generate_node_ids(nei_generator):
      try:
        yield self._generate_one_jax(live, batch_seed)

      except ValueError as e:
        if self.skip_overflow_padding_error:
          log.warning(
              (
                  "Skipping batch due to padding overflow. Consider increasing"
                  " num_samples_for_stats. Error: %s"
              ),
              e,
          )
          continue
        raise e
