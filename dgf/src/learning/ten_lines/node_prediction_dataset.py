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

"""Prepare training datast for the 10-line node prediction model.

The main utility GNNDatasetPreparator generates normalized, padded and batched
graph samples for node prediction.
"""

import dataclasses
import itertools
from typing import Dict, Iterator, Literal, Optional, Tuple, Union
from dgf.src.analyse import in_process_feature_statistics as in_process_feature_statistics_lib
from dgf.src.analyse import padding as padding_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import padding as padding_data_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.io import jax as jax_lib
from dgf.src.learning.jax import common as jax_common_lib
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import dataset
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import log
from dgf.src.util import util
import jax
import jax.numpy as jnp
import numpy as np


# TODO(gbm): Rename to better class name.
@dataclasses.dataclass
class LiveData:
  """Data computed during the "preparation" stage of the dataset preparator."""

  feature_stats: statistics_lib.GraphFeatureStatistics
  normalizer: normalize_lib.GraphNormalizer
  padding: padding_data_lib.Padding
  sampling_plan: sampling_config_lib.SamplingPlan
  num_nodes_in_seed_nodeset: Optional[int]
  sample_generator: dataset.SampleGeneratorFromAnything
  normalized_graph: Optional[in_memory_graph_lib.InMemoryGraph] = None
  normalized_jax_graph: Optional[jax_in_memory_graph.JaxInMemoryGraph] = None


@dataclasses.dataclass(kw_only=True)
class GNNDatasetPreparator:
  """Generates graph samples to train node prediction models.

  The following transformations are applied:
  - Feature normalization (e.g., soft quantile, indexing) using
    `dgf.analyse.feature_statistics_from_graphs` and
    `dgf.transform.AutoNormalizer`.
  - Padding graphs using `dgf.analyse.padding_from_graph_generator`.
  - Merging of batches of graphs using `dgf.transform.merge_graphs`.

  This class is intended for basic GNN pipelines. For advanced GNN pipelines,
  users should apply those transformations manually.

  Usage example:

  ```python
  graph, schema = dgf.io.read_graph(DATASET_DIR)

  train_dataset = dgf.transform.GNNDatasetPreparator(
      graph=raw_graph,
      schema=raw_schema,
      sampling_config=dgf.sampling.SimpleSamplingConfig(
          seed_nodeset="nodes",
          num_hops=2,
          hop_width=3,
          reverse=True,
      ),
      batch_size=10,
      drop_remainder=True,
      shuffle=True,
  )

  # Generate batched, normalized, padded graphs.
  for graph_sample, merge_offsets in train_dataset.generate():
    print(graph_sample)
    print(merge_offsets)
    break
  ```

  Attributes:
    graph: One of the graph format defined in data.Graph e.g. in-memory graph,
      generator of graph samples, path to graph samples.
    schema: The schema of the graph. The semantics of the features should be
      configured.
    batch_size: The desired size of each batch of seed nodes. Set batch_size=1
      to avoid batching / merging.
    sampling_plan: Configuration for sampling subgraphs around the seed nodes.
    drop_remainder: If `True`, the last batch of seed nodes will be dropped if
      it contains fewer than `batch_size` elements.
    shuffle: If `True`, the seed nodes are shuffled before batching.
    num_samples_for_stats: Number of graph samples used to compute feature and
      graph statistics, which are required for feature normalization and
      padding. Larger values lead to more precise statistics but increase the
      time taken during the initial preparation phase. If set to `None`,
      statistics will be computed using one sample for each node in the seed
      nodeset.
    auto_normalize_config: Configuration for automatic feature normalization.
      Defaults to `normalize_lib.AutoNormalizeConfig()`.
    verbose_preparation: If true, display a progress bar during the preparation
      stage that appears only the first time `generate` is called.
    skip_overflow_padding_error: If padding is set, the merging stage can fail
      if the "padding" is not large enought. If
      skip_overflow_padding_error=True, such batch is skipped. If
      skip_overflow_padding_error=False, and error is raised.
    format: The format of the graph. If set to AUTO, the format will be inferred
      from the graph object.
    seed_node_idxs: Optional array of node indices to use as seeds for sampling.
      If None, all nodes in the seed nodeset are used.
    temporal_sampling: True if the sampling relies on timestamp features to
      condition the sampling.
    edgeset_timestamp_features: Optional dictionary mapping edgeset names to the
      feature name containing timestamp information.
    nodeset_timestamp_features: Optional dictionary mapping nodeset names to the
      feature name containing timestamp information.
    cache_normalized_features: If True, pre-compute the normalized features
      during the preparation stage instead of computing them on the fly in the
      generator. This optioncan speeds up data generation/training but increases
      memory consumption. This option is only available for in memory graph
      inputs.
    cache_normalized_features_device: Specifies the device ("host" for RAM or
      "device" for GPU/TPU) to store the cached normalized features. Caching
      features on the same device used for training reduces host-device
      communication, potentially speeding up training, but increases memory
      consumption on the device.
  """

  # Required arguments
  graph: dataset.Graph
  schema: schema_lib.GraphSchema
  batch_size: int
  sampling_plan: sampling_config_lib.SamplingPlan
  drop_remainder: bool
  shuffle: bool
  format: Union[dataset.GraphFormat, str] = dataset.GraphFormat.AUTO
  seed_node_idxs: Optional[np.ndarray] = None

  # Optional arguments
  num_samples_for_stats: Optional[int] = 10000
  auto_normalize_config: normalize_lib.AutoNormalizeConfig = dataclasses.field(
      default_factory=normalize_lib.AutoNormalizeConfig
  )
  verbose_preparation: bool = True
  skip_overflow_padding_error: bool = False
  temporal_sampling: bool = False
  edgeset_timestamp_features: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  nodeset_timestamp_features: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  cache_normalized_features: bool = True
  cache_normalized_features_device: Literal["host", "device"] = "device"

  # The is_prepared data computed by the `prepare()` method.
  live: Optional[LiveData] = dataclasses.field(init=False, default=None)

  def __post_init__(self):
    if self.batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {self.batch_size}")
    if not self.temporal_sampling:
      if self.edgeset_timestamp_features:
        raise ValueError(
            "`edgeset_timestamp_features` must be empty when "
            "`temporal_sampling` is False."
        )
      if self.nodeset_timestamp_features:
        raise ValueError(
            "`nodeset_timestamp_features` must be empty when "
            "`temporal_sampling` is False."
        )
    elif (
        not self.edgeset_timestamp_features
        and not self.nodeset_timestamp_features
    ):
      raise ValueError(
          "At least one of `edgeset_timestamp_features` or "
          "`nodeset_timestamp_features` must be provided when "
          "`temporal_sampling` is True."
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

  def num_nodes_in_seed_nodeset(self) -> Optional[int]:
    """Number of nodes in the seed nodeset."""
    return self.get_live().num_nodes_in_seed_nodeset

  def generated_schema(self) -> schema_lib.GraphSchema:
    """Returns the schema of the generated samples."""
    return self.get_live().normalizer.output_schema()

  def prepare_from_existing_one(self, other: "GNNDatasetPreparator"):
    """Pre-compute data (same as "prepare") but with an already computed cache.

    Instead of beeing recomputed, the following are grabbed from "other":
      - Feature statistics
      - Normalier
      - Padding
      - Sampling plan

    Args:
      other: Another dataset preparator from which to copy the data.
    """
    other_live = other.get_live()

    sample_generator = dataset.SampleGeneratorFromAnything(
        graph=self.graph,
        schema=self.schema,
        batch_size=self.batch_size,
        seed_node_idxs=self.seed_node_idxs,  # pyrefly: ignore[bad-argument-type]
        sampling_config=other_live.sampling_plan,  # from other
        drop_remainder=self.drop_remainder,
        shuffle=self.shuffle,
        format=self.format,
        skip_overflow_padding_error=self.skip_overflow_padding_error,
        padding=other_live.padding,
        temporal=self.temporal_sampling,
        edgeset_timestamp_features=self.edgeset_timestamp_features,
        nodeset_timestamp_features=self.nodeset_timestamp_features,
    )

    self.live = LiveData(
        feature_stats=other_live.feature_stats,  # from other
        normalizer=other_live.normalizer,  # from other
        padding=other_live.padding,  # from other
        sampling_plan=sample_generator.sampling_config,  # pyrefly: ignore[bad-argument-type]
        num_nodes_in_seed_nodeset=sample_generator.num_seed_nodes,
        sample_generator=sample_generator,
    )

    if self.cache_normalized_features:
      if isinstance(self.graph, in_memory_graph_lib.InMemoryGraph):
        sample_generator.set_sampler_returns_node_idxs_only(True)
        if self.graph is other.graph:
          self.live.normalized_graph = other_live.normalized_graph
          self.live.normalized_jax_graph = other_live.normalized_jax_graph
        else:
          if self.cache_normalized_features_device == "host":
            self.live.normalized_graph = other_live.normalizer.normalize_numpy(
                self.graph
            )
          elif self.cache_normalized_features_device == "device":
            self.live.normalized_jax_graph = (
                other_live.normalizer.normalize_numpy_to_jax(self.graph)
            )
          else:
            raise ValueError(
                "Unsupported `cache_normalized_features_device`: "
                f"{self.cache_normalized_features_device}"
            )
      else:
        self.cache_normalized_features = False

  def prepare(self):
    """Pre-compute and pre-pare what is necessary for the generation.

    Can only be called once. Called automatically the first time "generate" is
    called.
    """
    if self.live is not None:
      raise ValueError("prepare() can only be called once.")

    # Create sampler
    if self.verbose_preparation:
      log.info("Create graph sampler")

    # Convert the user input graph into a generator of batched graph samples.
    # Note: Those samples are neither normalized not padded.
    sample_generator = dataset.SampleGeneratorFromAnything(
        graph=self.graph,
        schema=self.schema,
        batch_size=self.batch_size,
        seed_node_idxs=self.seed_node_idxs,  # pyrefly: ignore[bad-argument-type]
        sampling_config=self.sampling_plan,
        drop_remainder=self.drop_remainder,
        shuffle=self.shuffle,
        format=self.format,
        skip_overflow_padding_error=self.skip_overflow_padding_error,
        temporal=self.temporal_sampling,
        edgeset_timestamp_features=self.edgeset_timestamp_features,
        nodeset_timestamp_features=self.nodeset_timestamp_features,
    )

    # A generator of non-normalized graph samples.
    def gen_raw_samples():
      for sample, _ in sample_generator.iterator():
        yield sample

    gen_raw_samples_iter = gen_raw_samples()
    if self.num_samples_for_stats is not None:
      gen_raw_samples_iter = itertools.islice(
          gen_raw_samples_iter,
          self.num_samples_for_stats // self.batch_size,
      )

    # Compute the feature statistics (for the feature normalization)
    if self.verbose_preparation:
      log.info("Compute feature statistics")
    feature_stats = (
        in_process_feature_statistics_lib.feature_statistics_from_graphs(
            gen_raw_samples_iter, self.schema
        )
    )
    if self.verbose_preparation:
      log.info("  %s", feature_stats)
    normalizer = normalize_lib.auto_normalize(
        schema=self.schema,
        stats=feature_stats,
        config=self.auto_normalize_config,
    )

    # A generator of normalized, non-padded graph samples.
    def gen_normalized_samples():
      for sample, _ in sample_generator.iterator():
        normalized_merged = normalizer.normalize_numpy(sample)
        yield normalized_merged

    gen_normalized_samples_iter = gen_normalized_samples()
    if self.num_samples_for_stats is not None:
      gen_normalized_samples_iter = itertools.islice(
          gen_normalized_samples_iter,
          self.num_samples_for_stats // self.batch_size,
      )

    # Compute the batched graph statistics (for the padding)
    if self.verbose_preparation:
      log.info("Compute graph statistics for padding")
    padding = padding_lib.padding_from_graph_generator(
        self.schema, gen_normalized_samples_iter
    )
    if self.verbose_preparation:
      log.info(
          "  padding: %s",
          padding_lib.print_padding(padding, return_output=True, header=False),
      )
    sample_generator.padding = padding

    self.live = LiveData(
        feature_stats=feature_stats,
        normalizer=normalizer,
        padding=padding,
        sampling_plan=sample_generator.sampling_config,  # pyrefly: ignore[bad-argument-type]
        num_nodes_in_seed_nodeset=sample_generator.num_seed_nodes,
        sample_generator=sample_generator,
    )

    if self.cache_normalized_features:
      if isinstance(self.graph, in_memory_graph_lib.InMemoryGraph):
        # Cache the normalized data.
        sample_generator.set_sampler_returns_node_idxs_only(True)
        if self.cache_normalized_features_device == "host":
          self.live.normalized_graph = normalizer.normalize_numpy(self.graph)
        elif self.cache_normalized_features_device == "device":
          self.live.normalized_jax_graph = normalizer.normalize_numpy_to_jax(
              self.graph
          )
        else:
          raise ValueError(
              "Unsupported `cache_normalized_features_device`: "
              f"{self.cache_normalized_features_device}"
          )
      else:
        self.cache_normalized_features = False

  def generate(
      self,
  ) -> Iterator[
      Tuple[in_memory_graph_lib.InMemoryGraph, Dict[str, np.ndarray]]
  ]:
    """Generates batched + normalized graph samples.

    Yields:
      A tuple containing the normalized, merged, and padded graph sample, and
      the merging node offsets.
    """

    live = self.get_live()
    for sample, merge_offsets in live.sample_generator.iterator():
      if self.cache_normalized_features:
        if live.normalized_graph is None:
          raise RuntimeError(
              "Cached normalized graph is not available. This should have been "
              "prepared in `prepare()`."
          )
        normalized_sample = attach_features_from_numpy_graph(
            live.normalized_graph, sample
        )
      else:
        normalized_sample = live.normalizer.normalize_numpy(sample)
      yield normalized_sample, merge_offsets

  def generate_jax(
      self,
  ) -> Iterator[
      Tuple[jax_in_memory_graph.JaxInMemoryGraph, Dict[str, jnp.ndarray]]
  ]:
    """Generates jax batched + normalized graph samples.

    Yields:
      A tuple containing the normalized, merged, and padded graph sample, and
      the merging node offsets.
    """

    live = self.get_live()
    for sample, merge_offsets in live.sample_generator.iterator():

      if self.cache_normalized_features:
        if live.normalized_jax_graph:
          # The normalized features are already in jax format.
          jax_normalized_sample = (
              attach_features_from_jax_graph_and_cast_to_jax(
                  live.normalized_jax_graph, sample
              )
          )
        else:
          # The normalized features in numpy format, and should be casted.
          jax_normalized_sample = attach_features_from_jax_graph_and_cast_to_jax(
              live.normalized_graph, sample  # pyrefly: ignore[bad-argument-type]
          )
      else:
        normalized_sample = live.normalizer.normalize_numpy(sample)
        jax_normalized_sample = jax_lib.graph_to_jax_graph(normalized_sample)

      jax_merge_offsets = {k: jnp.asarray(v) for k, v in merge_offsets.items()}
      yield jax_normalized_sample, jax_merge_offsets


def attach_features_from_numpy_graph(
    graph: in_memory_graph_lib.InMemoryGraph,
    sample: in_memory_graph_lib.InMemoryGraph,
) -> in_memory_graph_lib.InMemoryGraph:
  """Attaches the numpy features from `graph` to the `sample`."""
  in_memory_sampler_lib.add_features_to_samples(
      graph, [sample], return_features=True, return_node_idxs=False
  )
  return sample


def attach_features_from_jax_graph_and_cast_to_jax(
    graph: Union[
        in_memory_graph_lib.InMemoryGraph, jax_in_memory_graph.JaxInMemoryGraph
    ],
    sample: in_memory_graph_lib.InMemoryGraph,
) -> jax_in_memory_graph.JaxInMemoryGraph:
  """Similar but more efficient than attach_features_from_numpy_graph + optional graph_to_jax_graph."""

  jax_node_sets = {}
  for node_set_name, node_set in sample.node_sets.items():
    node_idxs = node_set.features["#idx"]
    features = graph.node_sets[node_set_name].features

    if isinstance(graph, jax_in_memory_graph.JaxInMemoryGraph):
      node_idxs = jax.numpy.asarray(node_idxs)
      gathered_features = jax_common_lib.jit_gather_features(
          features, node_idxs
      )
    else:
      gathered_features = {
          feature_name: jax.numpy.asarray(feature_value[node_idxs])
          for feature_name, feature_value in features.items()
      }

    jax_node_sets[node_set_name] = jax_in_memory_graph.JaxInMemoryNodeSet(
        features=gathered_features,
        num_nodes=node_set.num_nodes,
    )

  jax_edge_sets = {}
  for edge_set_name, edge_set in sample.edge_sets.items():
    jax_adjacency = jax.numpy.asarray(edge_set.adjacency)
    jax_edge_sets[edge_set_name] = jax_in_memory_graph.JaxInMemoryEdgeSet(
        adjacency=jax_adjacency
    )

  return jax_in_memory_graph.JaxInMemoryGraph(
      node_sets=jax_node_sets, edge_sets=jax_edge_sets
  )


def compute_train_and_valid_node_idxs(
    graph: common.Graph,
    valid_graph: Optional[common.Graph],
    graph_format: Union[dataset.GraphFormat, str],
    target_nodeset: str,
    random_seed: int,
    validation_ratio: float,
    train_seed_nodes: Optional[common.SeedNodeIdxs],
    valid_seed_nodes: Optional[common.SeedNodeIdxs],
    max_num_valid_examples: Optional[int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
  """Computes the training and validation seed node indices."""
  if not isinstance(graph, in_memory_graph_lib.InMemoryGraph) or (
      valid_graph is not None
      and not isinstance(valid_graph, in_memory_graph_lib.InMemoryGraph)
  ):
    if train_seed_nodes is not None or valid_seed_nodes is not None:
      raise ValueError(
          "Specifying 'train_seed_nodes' or 'valid_seed_nodes' is not supported"
          f" for the current graph format ({graph_format}). Currently, only"
          " the InMemoryGraph format is supported."
      )
    return None, None

  num_graph_seed_nodes = graph.node_sets[target_nodeset].num_nodes
  if valid_graph is None:
    num_valid_graph_seed_nodes = num_graph_seed_nodes
  else:
    assert isinstance(valid_graph, in_memory_graph_lib.InMemoryGraph)
    num_valid_graph_seed_nodes = valid_graph.node_sets[target_nodeset].num_nodes
  assert num_graph_seed_nodes is not None
  assert num_valid_graph_seed_nodes is not None

  if train_seed_nodes is not None:
    return np.array(train_seed_nodes), (
        np.array(valid_seed_nodes) if valid_seed_nodes else None
    )

  if valid_seed_nodes is not None:
    if valid_graph is None:
      raise ValueError(
          "`valid_seed_nodes` can only be specified when `train_seed_nodes` is"
          " also specified if not validation graph (valid_graph) is provided."
      )
    return None, np.array(valid_seed_nodes)

  if validation_ratio == 0 or valid_graph is not None:
    log.info(
        "Train model on the full provided graphs. Num training seed nodes:"
        " %d. Num validation seed nodes: %d",
        num_graph_seed_nodes,
        num_valid_graph_seed_nodes,
    )
    return None, None

  train_seed_node_idxs, valid_seed_node_idxs = util.split_train_valid(
      num_graph_seed_nodes,
      validation_ratio,
      random_seed,
      max_num_valid_examples=max_num_valid_examples,
  )
  log.info(
      "Num. training seed nodes: %d, Num. validation seed nodes: %d",
      len(train_seed_node_idxs),
      len(valid_seed_node_idxs),
  )
  return train_seed_node_idxs, valid_seed_node_idxs


def prepare_datasets(
    graph: common.Graph,
    valid_graph: common.Graph,
    schema: schema_lib.GraphSchema,
    target_nodeset: str,
    random_seed: int,
    batch_size: int,
    num_sampling_hops: int,
    sampling_width: int,
    verbose: int,
    graph_format: Union[dataset.GraphFormat, str],
    validation_ratio: float,
    train_seed_nodes: Optional[common.SeedNodeIdxs],
    valid_seed_nodes: Optional[common.SeedNodeIdxs],
    temporal_sampling: bool,
    nodeset_timestamp_features: dict[str, str],
    edgeset_timestamp_features: dict[str, str],
    num_valid_steps: Optional[int],
    cache_valid_dataset: bool,
    cache_normalized_features: bool,
    cache_normalized_features_device: Literal["host", "device"],
    sampling_plan: Optional[sampling_config_lib.SamplingPlan],
    auto_normalize_config: Optional[normalize_lib.AutoNormalizeConfig] = None,
    keep_raw_features: Optional[set[str]] = None,
) -> Tuple["GNNDatasetPreparator", Optional["GNNDatasetPreparator"]]:
  """Prepares the training dataset by sampling, normalizing, and padding."""
  if not cache_valid_dataset or num_valid_steps is None:
    max_num_valid_examples = None
  else:
    max_num_valid_examples = num_valid_steps * batch_size

  train_seed_node_idxs, valid_seed_node_idxs = (
      compute_train_and_valid_node_idxs(
          graph,
          valid_graph,
          graph_format,
          target_nodeset=target_nodeset,
          random_seed=random_seed,
          validation_ratio=validation_ratio,
          train_seed_nodes=train_seed_nodes,
          valid_seed_nodes=valid_seed_nodes,
          max_num_valid_examples=max_num_valid_examples,
      )
  )

  if sampling_plan is None:
    sampling_config = sampling_config_lib.SimpleSamplingConfig(
        seed_nodeset=target_nodeset,
        num_hops=num_sampling_hops,
        hop_width=sampling_width,
        reverse=True,
        edgeset_timestamp_features=edgeset_timestamp_features
        if temporal_sampling
        else {},
    )
    sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, schema
    )

  if auto_normalize_config is None:
    auto_normalize_config = normalize_lib.AutoNormalizeConfig(
        keep_raw_features=keep_raw_features or set(),
        ignore_features_without_stats=True,
    )
  elif keep_raw_features is not None:
    auto_normalize_config.keep_raw_features.update(keep_raw_features)

  common_kwargs = {
      "format": graph_format,
      "schema": schema,
      "sampling_plan": sampling_plan,
      "batch_size": batch_size,
      "drop_remainder": True,
      "verbose_preparation": verbose >= 2,
      "auto_normalize_config": auto_normalize_config,
      "skip_overflow_padding_error": True,
      "temporal_sampling": temporal_sampling,
      "nodeset_timestamp_features": nodeset_timestamp_features,
      "edgeset_timestamp_features": edgeset_timestamp_features,
      "cache_normalized_features": cache_normalized_features,
      "cache_normalized_features_device": cache_normalized_features_device,
  }

  train_dataset = GNNDatasetPreparator(
      graph=graph,
      seed_node_idxs=train_seed_node_idxs,
      shuffle=True,
      **common_kwargs,
  )
  train_dataset.prepare()

  valid_dataset = GNNDatasetPreparator(
      graph=valid_graph if valid_graph is not None else graph,
      seed_node_idxs=valid_seed_node_idxs,
      shuffle=not cache_valid_dataset,
      **common_kwargs,
  )
  valid_dataset.prepare_from_existing_one(train_dataset)

  common.check_number_of_seeds(
      batch_size=batch_size,
      num_training=train_dataset.num_nodes_in_seed_nodeset(),
      num_validation=valid_dataset.num_nodes_in_seed_nodeset(),
      key="node",
  )

  return train_dataset, valid_dataset
