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

"""Utilities to convert user input data into batched graph samples."""

import dataclasses
import enum
from typing import Callable, Dict, Iterator, Optional, Tuple, TypeAlias, Union
from dgf.src.data import in_memory_graph
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import tf_graph_sample
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import timeseries as timeseries_transform
from dgf.src.util import temporal as temporal_util
from dgf.src.util import util
import numpy as np

# The types of graphs supported.
Graph: TypeAlias = Union[
    in_memory_graph.InMemoryGraph,
    str,
]


class GraphFormat(enum.Enum):
  """The format of the input graph.

  Attributes:
    AUTO: Automatically infer the format from the input graph.
    IN_MEMORY_GRAPH: The input graph is an InMemoryGraph e.g. the result of
      dgf.io.read_graph.
    GRAPH_SAMPLE_GENERATOR: The input graph is a function that yields graph
      samples, e.g. the result of dgf.io.read_tfgnn_graphs.
    PATH_TFGNN_SAMPLE_BAGZ:The input is a path to a bagz file (or sharded set of
      files) containing graph samples in the tfgnn format.
    PATH_TF_SAMPLE_TF_RECORD: The input is a path to a tfrecord file (or sharded
      set of files) containing graph samples in the tfgnn format.
  """

  AUTO = "AUTO"
  IN_MEMORY_GRAPH = "IN_MEMORY_GRAPH"
  PATH_TF_SAMPLE_BAGZ = "PATH_TF_SAMPLE_BAGZ"
  PATH_TF_SAMPLE_TF_RECORD = "PATH_TF_SAMPLE_TF_RECORD"
  # TODO(gbm): Add support for pygrain dataset / iter-dataset.


# Generator of batched graph samples.
BatchSampleGeneratorIteratorFn = Callable[
    [], Iterator[Tuple[in_memory_graph.InMemoryGraph, Dict[str, np.ndarray]]]
]

# Generator of individual graph samples.
SingleSampleGeneratorIteratorFn = Callable[
    [], Iterator[in_memory_graph.InMemoryGraph]
]


@dataclasses.dataclass
class SampleGeneratorFromAnything:
  """Converts a user input graph into a generator of batched graph samples.

  If the user input object already represent graph samples, no sampling is
  performed. Not all the arguments are used in all situations. For example,
  the sampling config and seed_node_idxs if the input graph is a generator of
  graph
  samples.

  The format is inferred automatically if format="AUTO". The possible formats
  are:
    - AUTO: Automatically infer the format from the input graph.
    - IN_MEMORY_GRAPH: The input graph is an InMemoryGraph e.g. the result of
      dgf.io.read_graph.
    - PATH_TF_SAMPLE_BAGZ: The input is a path to a bagz file (or sharded set of
      files) containing graph samples in the tfgnn format.
    - PATH_TF_SAMPLE_TF_RECORD: The input is a path to a tfrecord file (or
    sharded set
      of files) containing graph samples in the tfgnn format.

  Attributes:
    graph: The input graph.
    schema: The schema of the input graph.
    batch_size: The batch size of the output graph samples.
    seed_node_idxs: The seed nodes to use for sampling.
    sampling_config: The sampling config to use for sampling.
    drop_remainder: Whether to drop the last batch if it is smaller than
      batch_size.
    shuffle: Whether to shuffle the seed nodes before sampling.
    format: The format of the input graph.
    padding: The padding strategy to use for the output graph samples. To change
      the padding after initialization, use "set_padding".
    skip_overflow_padding_error: If padding is set, the merging stage can fail
      if the "padding" is not large enough. If skip_overflow_padding_error=True,
      such batch is skipped. If skip_overflow_padding_error=False, and error is
      raised. This has not effect if "padding" is None.
    temporal: True if the data sample should be temporally aware.
    edgeset_timestamp_features: A mapping from edge set name to the feature name
      containing timestamps. Only used if temporal=true.
    nodeset_timestamp_features: A mapping from node set name to the feature name
      containing timestamps. Only used if temporal=true.
    num_seed_nodes: The number of seed nodes in the graph.
    batch_iterator: An iterator yielding batches of graph samples.
    in_memory_sampler: The in-memory sampler used when sampling from an
      `InMemoryGraph`.
    sampler_returns_node_idxs_only: If `True`, the sampler returns only node
      indices without feature values. If `False` (default), the sampler returns
      feature values.
    timeseries_pad_and_cap: Configuration for padding and capping timeseries
      sequence features per sample before merging.
    timedelta_extraction: Configuration for time delta extraction per sample
      before merging.
  """

  graph: Graph
  schema: schema_lib.GraphSchema
  batch_size: int
  seed_node_idxs: Optional[np.ndarray]
  sampling_config: Union[
      sampling_config_lib.SimpleSamplingConfig, sampling_config_lib.SamplingPlan
  ]
  drop_remainder: bool
  shuffle: bool
  format: Union[GraphFormat, str] = GraphFormat.AUTO
  padding: Optional[padding_lib.Padding] = None
  skip_overflow_padding_error: bool = False
  temporal: bool = False
  edgeset_timestamp_features: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  nodeset_timestamp_features: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  sampler_returns_node_idxs_only: bool = False
  timeseries_pad_and_cap: Optional[
      timeseries_transform.PadAndCapTimeseriesConfig
  ] = None
  timedelta_extraction: Optional[
      timeseries_transform.TimestampFeatureExtractorConfig
  ] = None

  num_seed_nodes: Optional[int] = dataclasses.field(init=False)
  batch_iterator: BatchSampleGeneratorIteratorFn = dataclasses.field(init=False)
  single_iterator: SingleSampleGeneratorIteratorFn = dataclasses.field(
      init=False
  )
  in_memory_sampler: Optional[in_memory_sampler_lib.Sampler] = None
  _output_schema: schema_lib.GraphSchema = dataclasses.field(init=False)
  _target_nodeset: Optional[str] = dataclasses.field(init=False, default=None)
  _ts_feature: Optional[str] = dataclasses.field(init=False, default=None)
  _seed_timestamps_all: Optional[np.ndarray] = dataclasses.field(
      init=False, default=None
  )
  _pad_and_cap_transformer: Optional[
      timeseries_transform.PadAndCapTimeseries
  ] = dataclasses.field(init=False, default=None)
  _timestamp_extractor: Optional[
      timeseries_transform.TimestampFeatureExtractor
  ] = dataclasses.field(init=False, default=None)
  _has_per_sample_transforms: bool = dataclasses.field(
      init=False, default=False
  )

  def __post_init__(self):
    if self.seed_node_idxs is not None:
      self.seed_node_idxs = np.asarray(self.seed_node_idxs, dtype=np.int64)

    if isinstance(self.format, str):
      self.format = GraphFormat[self.format.upper()]

    if self.format == GraphFormat.AUTO:
      self.format = self._infer_format()

    if isinstance(
        self.sampling_config, sampling_config_lib.SimpleSamplingConfig
    ):
      self.sampling_config = (
          sampling_config_lib.simple_sampling_config_to_sampling_plan(
              self.sampling_config, self.schema
          )
      )

    self._target_nodeset = self.sampling_config.root.nodeset  # pyrefly: ignore[missing-attribute]
    self._ts_feature = temporal_util.creation_time_feature_name(
        self.schema.node_sets[self._target_nodeset].features
    )
    if self.temporal and self._ts_feature is None:
      raise ValueError(
          "Temporal sampling (`temporal=True`) is configured, but no creation"
          " timestamp feature was found in node set"
          f" '{self._target_nodeset}'."
      )

    if self.format == GraphFormat.IN_MEMORY_GRAPH:
      assert isinstance(self.graph, in_memory_graph.InMemoryGraph)
      if self.seed_node_idxs is not None:
        self.num_seed_nodes = len(self.seed_node_idxs)
      else:
        self.num_seed_nodes = self.graph.node_sets[
            self._target_nodeset
        ].num_nodes

      if self._ts_feature is not None:
        self._seed_timestamps_all = self.graph.node_sets[
            self._target_nodeset
        ].features[self._ts_feature]

      # Creating the sampler is expensive.
      self.in_memory_sampler = in_memory_sampler_lib.create_sampler(
          graph=self.graph,
          plan=self.sampling_config,
          schema=self.schema,
          batch_size=self.batch_size,
          return_features=not self.sampler_returns_node_idxs_only,
          return_node_idxs=self.sampler_returns_node_idxs_only,
      )
    else:
      self.num_seed_nodes = (
          len(self.seed_node_idxs) if self.seed_node_idxs is not None else None
      )

    # Initializing per-sample transforms and precomputing the output schema.
    current_schema = self.schema
    self._has_per_sample_transforms = (
        self.timeseries_pad_and_cap is not None
        or self.timedelta_extraction is not None
    )

    if (
        self.temporal or self._has_per_sample_transforms
    ) and self.format != GraphFormat.IN_MEMORY_GRAPH:
      raise ValueError(
          "Temporal sampling (`temporal=True`) and per-sample transformations"
          " (`timeseries_pad_and_cap`, `timedelta_extraction`) are"
          " only supported for GraphFormat.IN_MEMORY_GRAPH, but got format:"
          f" {self.format}"
      )

    if (
        self.timeseries_pad_and_cap is not None
        and temporal_util.schema_has_timeseries_features(current_schema)
    ):
      self._pad_and_cap_transformer = timeseries_transform.PadAndCapTimeseries(
          current_schema, self.timeseries_pad_and_cap
      )
      current_schema = self._pad_and_cap_transformer.output_schema()

    if temporal_util.schema_has_dynamic_timeseries_features(current_schema):
      raise ValueError(
          "Dynamic shape timeseries features were detected in the schema;"
          " please configure `timeseries_pad_and_cap` to pad/cap the sequences"
          " first."
      )

    if self.timedelta_extraction is not None:
      if self._ts_feature is None:
        raise ValueError(
            "Timedelta extraction is configured, but no creation timestamp"
            f" feature was found in node set '{self._target_nodeset}'."
        )
      self._timestamp_extractor = (
          timeseries_transform.TimestampFeatureExtractor(
              current_schema, self.timedelta_extraction
          )
      )
      current_schema = self._timestamp_extractor.output_schema()

    self._output_schema = current_schema

    if self.sampler_returns_node_idxs_only and self._has_per_sample_transforms:
      raise ValueError(
          "Per-sample transformations (`timeseries_pad_and_cap`,"
          " `timedelta_extraction`) cannot be used when"
          " `sampler_returns_node_idxs_only=True`."
      )

    self.batch_iterator, self.single_iterator = self.iterator_builder()

  def output_schema(self) -> schema_lib.GraphSchema:
    """Returns the schema of the generated graph samples."""
    return self._output_schema

  def set_sampler_returns_node_idxs_only(
      self, sampler_returns_node_idxs_only: bool
  ):
    """Changes whether the sampler returns only node indices."""
    if sampler_returns_node_idxs_only and self._has_per_sample_transforms:
      raise ValueError(
          "Per-sample transformations (`timeseries_pad_and_cap`,"
          " `timedelta_extraction`) cannot be used when"
          " `sampler_returns_node_idxs_only=True`."
      )
    self.sampler_returns_node_idxs_only = sampler_returns_node_idxs_only
    if self.in_memory_sampler is not None:
      self.in_memory_sampler.set_return_options(
          return_features=not self.sampler_returns_node_idxs_only,
          return_node_idxs=self.sampler_returns_node_idxs_only,
      )
    self.batch_iterator, self.single_iterator = self.iterator_builder()

  def _infer_format(self) -> GraphFormat:
    if isinstance(self.graph, in_memory_graph.InMemoryGraph):
      return GraphFormat.IN_MEMORY_GRAPH
    if isinstance(self.graph, str):
      if ".bagz" in self.graph:
        return GraphFormat.PATH_TF_SAMPLE_BAGZ
      if ".tfrecord" in self.graph:
        return GraphFormat.PATH_TF_SAMPLE_TF_RECORD
      return GraphFormat.PATH_TF_SAMPLE_TF_RECORD

    options = [f.name for f in GraphFormat if f != GraphFormat.AUTO]
    raise ValueError(
        "Could not infer format from graph. Specify it manually with 'format' ="
        f" one of: {options}"
    )

  def _get_merge_schema(self) -> schema_lib.GraphSchema:
    """Creates schema for merging samples with only node indices."""
    node_sets = {}
    for name, _ in self.schema.node_sets.items():
      node_sets[name] = schema_lib.NodeSchema(
          features={
              "#idx": schema_lib.FeatureSchema(
                  format=schema_lib.FeatureFormat.INTEGER_64
              )
          }
      )

    edge_sets = {}
    for name, edge_schema in self.schema.edge_sets.items():
      edge_sets[name] = schema_lib.EdgeSchema(
          source=edge_schema.source, target=edge_schema.target
      )

    return schema_lib.GraphSchema(node_sets=node_sets, edge_sets=edge_sets)

  def _generator_from_in_memory_graph(
      self,
  ) -> Tuple[BatchSampleGeneratorIteratorFn, SingleSampleGeneratorIteratorFn]:
    """Creates a SampleGenerator from an InMemoryGraph."""
    assert isinstance(self.graph, in_memory_graph.InMemoryGraph)
    assert self.num_seed_nodes is not None

    merge_schema = (
        self._get_merge_schema()
        if self.sampler_returns_node_idxs_only
        else self.output_schema()
    )

    def batch_generator():
      assert self.in_memory_sampler is not None
      for node_idxs in util.batch_indices_generator(
          self.seed_node_idxs  # pyrefly: ignore[bad-argumet-type]
          if self.seed_node_idxs is not None
          else self.num_seed_nodes,
          batch_size=self.batch_size,
          drop_remainder=self.drop_remainder,
          shuffle=self.shuffle,
      ):
        seed_timestamps = None
        if self._seed_timestamps_all is not None:
          seed_timestamps = self._seed_timestamps_all[node_idxs]

        if self.temporal:
          graph_samples = self.in_memory_sampler.sample(
              node_idxs, seed_timestamps=seed_timestamps
          )
        else:
          graph_samples = self.in_memory_sampler.sample(node_idxs)

        # Check if per-sample transforms are configured for efficiency.
        if self._has_per_sample_transforms:
          transformed_samples = []
          for i, sample in enumerate(graph_samples):
            curr_sg = sample
            if self._pad_and_cap_transformer is not None:
              curr_sg = self._pad_and_cap_transformer(curr_sg)
            if self._timestamp_extractor is not None:
              assert seed_timestamps is not None
              st = int(seed_timestamps[i])
              curr_sg = self._timestamp_extractor(curr_sg, seed_timestamp=st)
            transformed_samples.append(curr_sg)
          graph_samples = transformed_samples

        try:
          yield merge_lib.merge_graphs(
              graph_samples, merge_schema, padding=self.padding
          )
        except merge_lib.InsufficientPaddingError as e:
          if not self.skip_overflow_padding_error:
            raise e

    def single_generator():
      assert self.in_memory_sampler is not None
      for node_idxs in util.batch_indices_generator(
          self.seed_node_idxs  # pyrefly: ignore[bad-argument-type]
          if self.seed_node_idxs is not None
          else self.num_seed_nodes,
          batch_size=1,
          drop_remainder=False,
          shuffle=self.shuffle,
      ):
        node_idx = int(node_idxs[0])
        if self.temporal:
          # Get the seed node timestamp
          seed_timestamp = None
          if self._seed_timestamps_all is not None:
            seed_timestamp = int(self._seed_timestamps_all[node_idx])

          yield self.in_memory_sampler.sample(
              node_idx, seed_timestamps=seed_timestamp
          )
        else:
          yield self.in_memory_sampler.sample(node_idx)

    return batch_generator, single_generator

  def _generator_from_path_tf_sample(
      self, container_type
  ) -> Tuple[BatchSampleGeneratorIteratorFn, SingleSampleGeneratorIteratorFn]:
    """Creates a SampleGenerator from a path to a bagz file."""
    assert isinstance(self.graph, str)

    def batch_generator():
      # TODO(gbm): Use pygrain and add shuffling?
      it = tf_graph_sample.read_tfgnn_graphs(
          self.graph, self.schema, container_type=container_type
      )
      while True:
        batch = []
        try:
          for _ in range(self.batch_size):
            batch.append(next(it))
        except StopIteration:
          if batch and not self.drop_remainder:
            try:
              yield merge_lib.merge_graphs(
                  batch, self.schema, padding=self.padding
              )
            except merge_lib.InsufficientPaddingError as e:
              if not self.skip_overflow_padding_error:
                raise e
          return
        try:
          yield merge_lib.merge_graphs(batch, self.schema, padding=self.padding)
        except merge_lib.InsufficientPaddingError as e:
          if not self.skip_overflow_padding_error:
            raise e

    def single_generator():
      return tf_graph_sample.read_tfgnn_graphs(
          self.graph, self.schema, container_type=container_type
      )

    return batch_generator, single_generator

  def iterator_builder(
      self,
  ) -> Tuple[BatchSampleGeneratorIteratorFn, SingleSampleGeneratorIteratorFn]:
    if self.temporal and self.format != GraphFormat.IN_MEMORY_GRAPH:
      raise ValueError(
          "Temporal sampling is only supported for GraphFormat.IN_MEMORY_GRAPH,"
          f" but got format: {self.format}"
      )

    if self.format == GraphFormat.IN_MEMORY_GRAPH:
      return self._generator_from_in_memory_graph()

    elif self.format == GraphFormat.PATH_TF_SAMPLE_BAGZ:
      return self._generator_from_path_tf_sample("BAGZ")

    elif self.format == GraphFormat.PATH_TF_SAMPLE_TF_RECORD:
      return self._generator_from_path_tf_sample("TF_RECORD")

    else:
      raise ValueError(f"Unsupported format: {self.format}")
