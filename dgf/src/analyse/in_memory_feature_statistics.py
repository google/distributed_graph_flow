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

"""Compute feature stats, in beam, on InMemoryGraph."""

import apache_beam as beam
from dgf.src.analyse import feature_statistics as feature_statistics_lib
from dgf.src.data import distributed_graph
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib


def _extract_nodeset_features(
    g: distributed_graph.KeyedInMemoryGraph,
    nodeset_name: str,
) -> in_memory_graph.Features:
  return g.graph.node_sets[nodeset_name].features


def feature_statistics_from_graphs(
    graph: distributed_graph.PKeyedInMemoryGraph,
    schema: schema_lib.GraphSchema,
    max_num_dictionary_items: int = 10_000,
    min_dictionary_item_frequency: int = 1,
    num_quantiles: int = 100,
    dictionary_buffer_size: int | None = None,
    reservoir_sampling_buffer_size: int = 10_000,
) -> beam.PCollection[statistics_lib.GraphFeatureStatistics]:
  """Computes the feature statistics for a set of InMemoryGraphs.

  To write stats to disk, see "write_feature_statistics".

  For a small set of graphs, use the in-process version instead:
  dgf.analyse.feature_statistics_from_graphs.

  Usage example:

  ```python
  # Read a graph reader
  graphs = dgf.beam.io.read_tfgnn_graphs(pipe, "/my/data@10")

  # Compute the statistics
  feature_statistics = (
      dgf.beam.analyse.feature_statistics_from_graphs(graphs, schema)
  )

  # Save the statistics
  dgf.beam.io.write_feature_statistics(feature_statistics, "/my/stats.json")
  ```

  Args:
    graph: Input collection of graphs.
    schema: Schema of the graph.
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

  config = feature_statistics_lib.Config(
      max_num_dictionary_items=max_num_dictionary_items,
      min_dictionary_item_frequency=min_dictionary_item_frequency,
      num_quantiles=num_quantiles,
      dictionary_buffer_size=10 * max_num_dictionary_items
      if dictionary_buffer_size is None
      else dictionary_buffer_size,
      reservoir_sampling_buffer_size=reservoir_sampling_buffer_size,
  )

  nodeset_stats_list = []
  for nodeset_name, nodeset_def in schema.node_sets.items():

    stats = (
        graph
        | f"Extract features for nodeset {nodeset_name}"
        >> beam.Map(_extract_nodeset_features, nodeset_name)
        | f"Compute feature stats for nodeset {nodeset_name}"
        >> beam.CombineGlobally(
            feature_statistics_lib.CombineFeatureSetStatistics(
                feature_statistics_lib.filter_feature_schema(
                    nodeset_def.features
                ),
                config,
                batch_values=True,
            )
        )
        | f"Add nodeset name {nodeset_name}"
        >> beam.Map(feature_statistics_lib.add_name, nodeset_name)
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
