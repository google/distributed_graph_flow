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

"""Compute feature stats, in process, on InMemoryGraph."""

from typing import Iterator
from dgf.src.analyse import feature_statistics as feature_statistics_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib


def feature_statistics(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    max_num_dictionary_items: int = 10_000,
    min_dictionary_item_frequency: int = 1,
    num_quantiles: int = 100,
    dictionary_buffer_size: int | None = None,
    reservoir_sampling_buffer_size: int = 10_000,
) -> statistics_lib.GraphFeatureStatistics:
  """Computes the feature stats from a single graph.

  The following statistics are computed:
  - For numerical (e.g., format=float* or format=integer*) with
  semantic=NUMERICAL or TIMESERIES, computes min/max and histogram.
  - For format=bytes with semantic=CATEGORICAL, computes a dictionary.

  Warning: If the feature's semantic is not set, not statsitics will be
  computed.

  This method computes the feature statistics on an iterator of in-memory
  graphs.

  For large sets of graphs, use the Beam distributed version instead:
  dgf.analyse.feature_in_memory_statistics.

  Usage example:

  ```python
  # Read a graph reader
  graph, schema = dgf.io.read_graph("/tmp/my_graph")

  # Compute the statistics
  feature_statistics = dgf.analyse.feature_statistics(graph,schema)

  # Save the statistics
  dgf.io.write_feature_statistics(feature_statistics,"/my/stats.json")
  ```

  Args:
    graph: A graph.
    schema: Schema of the graph.
    max_num_dictionary_items: Maximum number of dictionary items.
    min_dictionary_item_frequency: Minimum frequency of the dictionary items.
    num_quantiles: Number of quantiles computed for the numerical features.
    dictionary_buffer_size: Maximum number of dictionary items accumulated per
      worker. If not set, defaults to 10 * max_num_dictionary_items.
    reservoir_sampling_buffer_size: Maximum buffer size used to estimate
      numerical value distributions using reservoir sampling.

  Returns:
    Feature statistics object.
  """
  return feature_statistics_from_graphs(
      graphs=iter([graph]),
      schema=schema,
      max_num_dictionary_items=max_num_dictionary_items,
      min_dictionary_item_frequency=min_dictionary_item_frequency,
      num_quantiles=num_quantiles,
      dictionary_buffer_size=dictionary_buffer_size,
      reservoir_sampling_buffer_size=reservoir_sampling_buffer_size,
  )


def feature_statistics_from_graphs(
    graphs: Iterator[in_memory_graph_lib.InMemoryGraph],
    schema: schema_lib.GraphSchema,
    max_num_dictionary_items: int = 10_000,
    min_dictionary_item_frequency: int = 1,
    num_quantiles: int = 100,
    dictionary_buffer_size: int | None = None,
    reservoir_sampling_buffer_size: int = 10_000,
) -> statistics_lib.GraphFeatureStatistics:
  """Computes the feature stats from multiple graphs.

  The following statistics are computed:
  - For numerical (e.g., format=float* or format=integer*) with
  semantic=NUMERICAL or TIMESERIES, computes min/max and histogram.
  - For format=bytes with semantic=CATEGORICAL, computes a dictionary.

  Warning: If the feature's semantic is not set, not statsitics will be
  computed.

  This method computes the feature statistics on an iterator of in-memory
  graphs.

  For large sets of graphs, use the Beam distributed version instead:
  dgf.analyse.feature_in_memory_statistics.

  Usage example:

  ```python
  # Read a graph reader
  graphs, schema =
  dgf.io.read_tfgnn_graphs("/my/data@10")

  # Compute the statistics
  feature_statistics =
  dgf.analyse.feature_statistics_from_graphs(graphs,schema)

  # Save the statistics
  dgf.io.write_feature_statistics(feature_statistics,"/my/stats.json")
  ```

  Args:
    graphs: An iterator over in-memory heterogeneous graphs.
    schema: Schema of the graph.
    max_num_dictionary_items: Maximum number of dictionary items.
    min_dictionary_item_frequency: Minimum frequency of the dictionary items.
    num_quantiles: Number of quantiles computed for the numerical features.
    dictionary_buffer_size: Maximum number of dictionary items accumulated per
      worker. If not set, defaults to 10 * max_num_dictionary_items.
    reservoir_sampling_buffer_size: Maximum buffer size used to estimate
      numerical value distributions using reservoir sampling.

  Returns:
    Feature statistics object.
  """

  filtered_nodeset_feature_schemas = {
      k: feature_statistics_lib.filter_feature_schema(v.features)
      for k, v in schema.node_sets.items()
  }

  config = feature_statistics_lib.Config(
      max_num_dictionary_items=max_num_dictionary_items,
      min_dictionary_item_frequency=min_dictionary_item_frequency,
      num_quantiles=num_quantiles,
      dictionary_buffer_size=10 * max_num_dictionary_items
      if dictionary_buffer_size is None
      else dictionary_buffer_size,
      reservoir_sampling_buffer_size=reservoir_sampling_buffer_size,
  )

  # Initialize accumulators
  nodeset_accumulators = {}
  for nodeset_name in schema.node_sets:
    nodeset_accumulators[nodeset_name] = (
        feature_statistics_lib.create_accumulator(
            filtered_nodeset_feature_schemas[nodeset_name], config
        )
    )

  # Scan data
  num_graphs = 0
  for graph in graphs:
    for nodeset_name in schema.node_sets:
      nodeset_accumulator = nodeset_accumulators[nodeset_name]
      feature_values = graph.node_sets[nodeset_name].features
      feature_statistics_lib.add_to_accumulator(
          filtered_nodeset_feature_schemas[nodeset_name],
          config,
          nodeset_accumulator,
          feature_values,
          batch_values=True,
      )
    num_graphs += 1
  if num_graphs == 0:
    raise ValueError("The input 'graphs' iterator was empty.")

  # Finalize accumulator
  nodeset_features = {}
  for nodeset_name in schema.node_sets:
    nodeset_accumulator = nodeset_accumulators[nodeset_name]
    accumulator = feature_statistics_lib.prune_accumulator(
        filtered_nodeset_feature_schemas[nodeset_name],
        config,
        nodeset_accumulator,
    )
    nodeset_features[nodeset_name] = (
        feature_statistics_lib.accumulator_to_feature_stats(
            filtered_nodeset_feature_schemas[nodeset_name], config, accumulator
        )
    )

  return statistics_lib.GraphFeatureStatistics(node_sets=nodeset_features)
