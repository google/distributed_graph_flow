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

"""Compute feature stats, in process, on Graph."""

import dataclasses
import math
from typing import Tuple
import apache_beam as beam
from dgf.src.analyse import reservoir_sampling
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
import numpy as np


@dataclasses.dataclass
class Config:
  """Configuration for the feature statistics stage."""

  max_num_dictionary_items: int
  min_dictionary_item_frequency: int
  dictionary_buffer_size: int
  reservoir_sampling_buffer_size: int
  num_quantiles: int


def _compute_dictionary(schema: schema_lib.FeatureSchema) -> bool:
  return (
      schema.semantic
      in [
          schema_lib.FeatureSemantic.CATEGORICAL,
      ]
      and schema.format == schema_lib.FeatureFormat.BYTES
  )


def _compute_numerical_minmax(schema: schema_lib.FeatureSchema) -> bool:
  return (
      schema.semantic
      in [
          schema_lib.FeatureSemantic.NUMERICAL,
          schema_lib.FeatureSemantic.TIMESERIES,
          schema_lib.FeatureSemantic.CATEGORICAL,
          schema_lib.FeatureSemantic.TIMESTAMP,
      ]
      and schema.format.is_numerical()
  )


def _compute_numerical_quantiles(schema: schema_lib.FeatureSchema) -> bool:
  return (
      schema.semantic
      in [
          schema_lib.FeatureSemantic.NUMERICAL,
          schema_lib.FeatureSemantic.TIMESERIES,
      ]
      and schema.format.is_numerical()
  )


def filter_feature_schema(
    schema: schema_lib.FeatureSetSchema,
) -> schema_lib.FeatureSetSchema:
  """Filters out features that should not be used for statistics computation."""

  return {
      k: v
      for k, v in schema.items()
      if v.semantic
      not in [
          schema_lib.FeatureSemantic.UNKNOWN,
      ]
  }


def feature_statistics(
    graph: distributed_graph.Graph,
    max_num_dictionary_items: int = 10_000,
    min_dictionary_item_frequency: int = 1,
    num_quantiles: int = 100,
    dictionary_buffer_size: int | None = None,
    reservoir_sampling_buffer_size: int = 10_000,
) -> beam.PCollection[statistics_lib.GraphFeatureStatistics]:
  """Computes the feature statistics for a distributed Graph.

  To write stats to disk, see "dgf.beam.io.write_feature_statistics".

  Args:
    graph: Input graph.
    max_num_dictionary_items: Maximum number of dictionary items.
    min_dictionary_item_frequency: Minimum frequency of the dictionary items.
    num_quantiles: Number of quantiles computed for the numerical features.
    dictionary_buffer_size: Maximum number of dictionary items accumulated per
      worker. If not set, defaults to 10 * max_num_dictionary_items.
    reservoir_sampling_buffer_size: Maximum buffer size used to estimate
      numerical value distributions using reservoir sampling.

  Returns:
    A PCollection with a single statistics object.
  """

  config = Config(
      max_num_dictionary_items=max_num_dictionary_items,
      min_dictionary_item_frequency=min_dictionary_item_frequency,
      num_quantiles=num_quantiles,
      dictionary_buffer_size=10 * max_num_dictionary_items
      if dictionary_buffer_size is None
      else dictionary_buffer_size,
      reservoir_sampling_buffer_size=reservoir_sampling_buffer_size,
  )

  nodeset_stats_list = []
  for nodeset_name, nodeset in graph.node_sets.items():
    if nodeset_name not in graph.schema.node_sets:
      raise ValueError(f"Nodeset {nodeset_name} not found in schema.")
    feature_schema = filter_feature_schema(
        graph.schema.node_sets[nodeset_name].features
    )
    stats = (
        nodeset
        | f"Extract features for nodeset {nodeset_name}"
        >> beam.Map(lambda x: x.features)
        | f"Compute feature stats for nodeset {nodeset_name}"
        >> beam.CombineGlobally(
            CombineFeatureSetStatistics(
                feature_schema, config, batch_values=False
            )
        )
        | f"Add nodeset name {nodeset_name}" >> beam.Map(add_name, nodeset_name)
    )
    nodeset_stats_list.append(stats)

  nodeset_stats = (
      nodeset_stats_list
      | "Merge all nodesets" >> beam.Flatten()
      | "Combine all nodesets" >> beam.combiners.ToList()
  )

  # TODO(gbm): Compute edge features

  return nodeset_stats | "Into graph statistics" >> beam.Map(
      lambda x: statistics_lib.GraphFeatureStatistics(node_sets=dict(x))
  )


def create_accumulator(
    schema: schema_lib.FeatureSetSchema, config: Config
) -> statistics_lib.FeatureSetStatisticsAccumulator:
  """Creates a new feature stats accumulator."""
  # Initialize the accumulator for a set of features.

  acc_features = {}
  for feature_name, feature_schema in schema.items():
    compute_dict = _compute_dictionary(feature_schema)
    compute_num_quant = _compute_numerical_quantiles(feature_schema)
    compute_num_minmax = _compute_numerical_minmax(feature_schema)
    acc_features[feature_name] = statistics_lib.FeatureStatisticsAccumulator(
        count=0,
        minimum=math.inf if compute_num_minmax else math.nan,
        maximum=-math.inf if compute_num_minmax else math.nan,
        dictionary={} if compute_dict else None,
        quantiles=reservoir_sampling.BatchReservoirSampling(
            config.reservoir_sampling_buffer_size
        )
        if compute_num_quant
        else None,
    )

  return statistics_lib.FeatureSetStatisticsAccumulator(features=acc_features)


def merge_accumulators(
    schema: schema_lib.FeatureSetSchema,
    config: Config,
    accumulators: list[statistics_lib.FeatureSetStatisticsAccumulator],
) -> statistics_lib.FeatureSetStatisticsAccumulator:
  """Merges multiple accumulators."""
  merged_accumulator = create_accumulator(schema, config)
  for accumulator in accumulators:
    for feature_name, feature_accumulator in accumulator.features.items():
      feature_schema = schema[feature_name]
      merged_feature_accumulator = merged_accumulator.features[feature_name]

      merged_feature_accumulator.count += feature_accumulator.count

      if _compute_numerical_minmax(feature_schema):
        merged_feature_accumulator.minimum = min(
            merged_feature_accumulator.minimum, feature_accumulator.minimum
        )
        merged_feature_accumulator.maximum = max(
            merged_feature_accumulator.maximum, feature_accumulator.maximum
        )

      if _compute_numerical_quantiles(feature_schema):
        assert merged_feature_accumulator.quantiles is not None
        merged_feature_accumulator.quantiles.add_reservoir(
            feature_accumulator.quantiles  # pyrefly: ignore[bad-argument-type]
        )

      if feature_accumulator.dictionary is not None:
        merged_dictionary = merged_feature_accumulator.dictionary
        assert merged_dictionary is not None
        for key, count in feature_accumulator.dictionary.items():
          merged_dictionary[key] = merged_dictionary.get(key, 0) + count

  return merged_accumulator


def accumulator_to_feature_stats(
    schema: schema_lib.FeatureSetSchema,
    config: Config,
    accumulator: statistics_lib.FeatureSetStatisticsAccumulator,
):
  """Converts a feature stats accumulator to a simple feature stats."""
  return statistics_lib.FeatureSetStatistics(
      features={
          key: remove_feature_statistics_accumulator(value, config)
          for key, value in accumulator.features.items()
      }
  )


def prune_accumulator(
    schema: schema_lib.FeatureSetSchema,
    config: Config,
    accumulator: statistics_lib.FeatureSetStatisticsAccumulator,
) -> statistics_lib.FeatureSetStatisticsAccumulator:
  """Compress the accumulator before network transmition."""
  for feature_name, feature_schema in schema.items():
    if _compute_dictionary(feature_schema):
      prune_dictionary_before_wiring(
          accumulator.features[feature_name].dictionary, config  # pyrefly: ignore[bad-argument-type]
      )
  return accumulator


def add_to_accumulator(
    schema: schema_lib.FeatureSetSchema,
    config: Config,
    accumulator: statistics_lib.FeatureSetStatisticsAccumulator,
    features: distributed_graph.Features,
    batch_values: bool,
) -> statistics_lib.FeatureSetStatisticsAccumulator:
  """Adds feature values to a feature stats accumulator.

  Args:
    schema: The feature set schema.
    config: The feature statistics configuration.
    accumulator: The accumulator to add the features to.
    features: The features to add.
    batch_values: Whether the input features are batched (i.e., this represent
      one or multiple observations e.g. nodes).

  Returns:
    The accumulator with the features added.
  """

  def injest_flat_value(
      flat_value: np.ndarray, feature_schema: schema_lib.FeatureSchema
  ):
    assert flat_value.dtype != np.object_
    if _compute_numerical_minmax(feature_schema):
      if flat_value.size > 0:
        feature_accumulator.minimum = min(
            feature_accumulator.minimum, flat_value.min().item()
        )
        feature_accumulator.maximum = max(
            feature_accumulator.maximum, flat_value.max().item()
        )
    if _compute_numerical_quantiles(feature_schema):
      assert feature_accumulator.quantiles is not None
      feature_accumulator.quantiles.add(flat_value)

    if _compute_dictionary(feature_schema):
      assert feature_accumulator.dictionary is not None
      for value in flat_value.flat:
        value = value.item().decode(
            "utf-8", errors="surrogateescape"
        )  # From numpy bytes to python string
        if value in feature_accumulator.dictionary:
          feature_accumulator.dictionary[value] += 1
        else:
          feature_accumulator.dictionary[value] = 1

  for feature_name, feature_schema in schema.items():
    feature_values = features.get(feature_name)
    if feature_values is None:
      continue
    feature_accumulator = accumulator.features[feature_name]
    if batch_values:
      feature_accumulator.count += feature_values.shape[0]
    else:
      feature_accumulator.count += 1
    if feature_values.size == 0:
      # No value
      continue

    if feature_schema.is_static_shape():
      injest_flat_value(feature_values, feature_schema)
    else:
      # If the value is variable size, we need to unfold it first.
      for flat_value in feature_values:
        injest_flat_value(flat_value, feature_schema)

  return accumulator


class CombineFeatureSetStatistics(beam.CombineFn):
  """Combines statistics for a set of features."""

  schema: schema_lib.FeatureSetSchema
  config: Config
  batch_values: bool

  def __init__(
      self,
      schema: schema_lib.FeatureSetSchema,
      config: Config,
      batch_values: bool,
  ):
    self.schema = schema
    self.config = config
    self.batch_values = batch_values

  def create_accumulator(
      self,
  ) -> statistics_lib.FeatureSetStatisticsAccumulator:
    return create_accumulator(self.schema, self.config)

  def add_input(
      self,
      accumulator: statistics_lib.FeatureSetStatisticsAccumulator,
      features: distributed_graph.Features,
  ):
    """Add an item (node, edge) to the accumulator."""
    return add_to_accumulator(
        self.schema,
        self.config,
        accumulator,
        features,
        batch_values=self.batch_values,
    )

  def merge_accumulators(
      self,
      accumulators: list[statistics_lib.FeatureSetStatisticsAccumulator],
  ) -> statistics_lib.FeatureSetStatisticsAccumulator:
    return merge_accumulators(self.schema, self.config, accumulators)

  def compact(
      self, accumulator: statistics_lib.FeatureSetStatisticsAccumulator
  ):
    return prune_accumulator(self.schema, self.config, accumulator)

  def extract_output(
      self, accumulator: statistics_lib.FeatureSetStatisticsAccumulator
  ) -> statistics_lib.FeatureSetStatistics:
    return accumulator_to_feature_stats(self.schema, self.config, accumulator)


def finalize_dictionary(
    value: statistics_lib.AccumulatorDictionary,
    config: Config,
) -> statistics_lib.Dictionary:
  """Finalizes an accumulator dictionary into a dictionary.

  Args:
    value: The accumulator dictionary.
    config: The configuration; the fields `config.max_num_dictionary_items` and
      `config.min_dictionary_item_frequency` are used.

  Returns:
    The finalized dictionary. Indices are assigned to the items in the order of
    decreasing frequency, with ties broken lexicographically by the category
    name.
  """
  filtered_items = [
      (key, count)
      for key, count in value.items()
      if count >= config.min_dictionary_item_frequency
  ]
  sorted_items = sorted(filtered_items, key=lambda item: (-item[1], item[0]))
  truncated_items = sorted_items[: config.max_num_dictionary_items]
  return {
      key: statistics_lib.DictionaryItem(index=index, count=count)
      for index, (key, count) in enumerate(truncated_items)
  }


def prune_dictionary_before_wiring(
    value: statistics_lib.AccumulatorDictionary,
    config: Config,
):
  """Prunes a dictionary before wiring.

  This function reduces the size of the dictionary by keeping only the most
  frequent items, and by removing items with low frequency. This is done to
  reduce the amount of data that needs to be transmitted between workers.

  Args:
    value: The input dictionary.
    config: The configuration.

  Returns:
    The pruned dictionary.
  """
  if len(value) <= config.dictionary_buffer_size:
    return value

  items = [(key, count) for key, count in value.items()]
  sorted_items = sorted(items, key=lambda item: item[1], reverse=True)
  truncated_items = sorted_items[: config.dictionary_buffer_size]
  return {key: count for key, count in truncated_items}


def remove_feature_statistics_accumulator(
    value: statistics_lib.FeatureStatisticsAccumulator,
    config: Config,
) -> statistics_lib.FeatureStatistics:
  """Removes the accumulator information from a feature statistics."""

  if value.quantiles is None or value.count == 0:
    quantiles = []
  else:
    quantiles = value.quantiles.get_quantiles(config.num_quantiles)[0]

  return statistics_lib.FeatureStatistics(
      count=value.count,
      minimum=float(value.minimum),
      maximum=float(value.maximum),
      dictionary=finalize_dictionary(value.dictionary, config)
      if value.dictionary
      else {},
      quantiles=quantiles,
  )


def add_name(
    stats: statistics_lib.FeatureSetStatistics, name: str
) -> Tuple[str, statistics_lib.FeatureSetStatistics]:
  return name, stats
