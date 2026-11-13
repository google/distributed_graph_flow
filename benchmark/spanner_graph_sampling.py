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

"""Benchmarking of Spanner Graph sampling (Cloud Spanner and Borg Spanner)."""

from collections.abc import Callable, Sequence

import dgf
from dgf.benchmark import utils as benchmark_utils
from dgf.src.data import schema as schema_lib
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.sampling import spanner_graph_sampler as spanner_graph_sampler_lib

# Creates a sampler from (sampling config, schema, query engine).
CreateSamplerFn = Callable[
    [dgf.sampling.SimpleSamplingConfig, dgf.data.GraphSchema, str],
    spanner_graph_sampler_lib.SpannerGraphSampler,
]


def _feature_set_schema(
    element_table: (
        spanner_graph_metadata_lib.NodeTable
        | spanner_graph_metadata_lib.EdgeTable
    ),
    property_types: dict[str, str],
    skip_primary_keys: bool,
) -> schema_lib.FeatureSetSchema:
  """Infers the DGF features of an element table from its properties."""
  definitions = element_table.property_definitions or []
  return gcp_common_lib.infer_feature_set_schema(
      graph_element_table={
          definition.property_declaration_name: property_types[
              definition.property_declaration_name
          ]
          for definition in definitions
      },
      # The keys are identified by their property (not column) name.
      key_columns=[
          definition.property_declaration_name
          for definition in definitions
          if definition.value_expression_sql in element_table.key_columns
      ],
      combine_as_json=False,
      skip_primary_keys=skip_primary_keys,
  )


def schema_from_metadata(
    metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
) -> dgf.data.GraphSchema:
  """Infers a DGF schema reading all the properties of a Spanner Graph.

  The nodesets and edgesets are named after the element tables.

  Args:
    metadata: Definition of the property graph.

  Returns:
    The DGF schema.
  """
  property_types = metadata.property_types or {}
  return dgf.data.GraphSchema(
      node_sets={
          node_table.name: dgf.data.NodeSchema(
              features=_feature_set_schema(
                  node_table, property_types, skip_primary_keys=False
              )
          )
          for node_table in metadata.node_tables
      },
      edge_sets={
          edge_table.name: dgf.data.EdgeSchema(
              source=edge_table.source_node_table.node_table_name,
              target=edge_table.destination_node_table.node_table_name,
              features=_feature_set_schema(
                  edge_table, property_types, skip_primary_keys=True
              ),
          )
          for edge_table in metadata.edge_tables
      },
  )


class SpannerGraphSamplingBenchmark(benchmark_utils.Benchmark):
  """Benchmarks subgraph sampling from a Spanner Graph."""

  def __init__(
      self,
      *,
      create_sampler: CreateSamplerFn,
      sampler_name: str,
      schema: dgf.data.GraphSchema,
      seed_nodeset: str,
      seed_ids: Sequence[bytes],
      query_engine: str,
      num_hops: int,
      hop_width: int,
      batch_size: int,
      reverse: bool,
      print_query: bool,
  ):
    self._create_sampler = create_sampler
    self._sampler_name = sampler_name
    self._schema = schema
    self._seed_nodeset = seed_nodeset
    self._seed_ids = seed_ids
    self._query_engine = query_engine
    self._num_hops = num_hops
    self._hop_width = hop_width
    self._batch_size = batch_size
    self._reverse = reverse
    self._print_query = print_query
    self._sampler: spanner_graph_sampler_lib.SpannerGraphSampler | None = None
    self._next_seed = 0
    self._sum_sampled_nodes = 0
    self._sum_sampled_edges = 0
    self._num_samples = 0

  def impl_name(self) -> str:
    return f"{self._sampler_name} {self._query_engine}"

  def setup(self):
    self._sampler = self._create_sampler(
        dgf.sampling.SimpleSamplingConfig(
            seed_nodeset=self._seed_nodeset,
            num_hops=self._num_hops,
            hop_width=self._hop_width,
            reverse=self._reverse,
            multi_visit=False,
        ),
        self._schema,
        self._query_engine,
    )
    if self._print_query:
      print(self._sampler.query)

  def run(self):
    """Samples one batch of seeds, different at each call."""
    assert self._sampler is not None
    batch_seeds = [
        self._seed_ids[(self._next_seed + i) % len(self._seed_ids)]
        for i in range(self._batch_size)
    ]
    self._next_seed += self._batch_size
    samples = self._sampler.sample(batch_seeds)
    for sample in samples:
      for nodeset in sample.node_sets.values():
        self._sum_sampled_nodes += nodeset.num_nodes or 0
      for edgeset in sample.edge_sets.values():
        self._sum_sampled_edges += edgeset.num_edges()
    self._num_samples += len(samples)

  def num_units(self) -> int:
    return self._batch_size

  def details(self) -> str:
    num_samples = max(1, self._num_samples)
    return (
        f"hops={self._num_hops}"
        f" width={self._hop_width}"
        f" batch={self._batch_size}"
        f" reverse={int(self._reverse)}"
        f" nodes/spl.={self._sum_sampled_nodes / num_samples:.1f}"
        f" edges/spl.={self._sum_sampled_edges / num_samples:.1f}"
    )


def run_spanner_graph_sampling_benchmark(
    *,
    create_sampler: CreateSamplerFn,
    sampler_name: str,
    schema: dgf.data.GraphSchema,
    seed_nodeset: str,
    seed_ids: Sequence[bytes],
    list_query_engines: Sequence[str] = ("SQL", "GQL"),
    list_num_hops: Sequence[int] = (0, 1, 2),
    list_batch_sizes: Sequence[int] = (1, 32),
    hop_width: int = 10,
    reverse: bool = True,
    repetitions: int = 5,
    warmup_repetitions: int = 1,
    print_query: bool = False,
) -> None:
  """Runs the Spanner Graph sampling benchmark suite.

  A plan with 0 hops only reads the seed nodes i.e. it measures the latency of
  a round trip to Spanner.

  Args:
    create_sampler: Creates the sampler to benchmark.
    sampler_name: Name of the sampler in the results.
    schema: The DGF schema of the graph.
    seed_nodeset: The seed nodeset.
    seed_ids: Ids of existing nodes of the seed nodeset. Each run samples the
      next batch_size ids.
    list_query_engines: Query engines to benchmark.
    list_num_hops: Number of hops to benchmark.
    list_batch_sizes: Number of seeds per query to benchmark.
    hop_width: Number of neighbors sampled per hop.
    reverse: Whether the edges are also traversed backward.
    repetitions: Number of timed queries per setup.
    warmup_repetitions: Number of non-timed queries per setup.
    print_query: Whether to print the sampling query of each setup.
  """
  benchmarker = benchmark_utils.Benchmarker()
  for num_hops in list_num_hops:
    for batch_size in list_batch_sizes:
      for query_engine in list_query_engines:
        benchmarker.run(
            SpannerGraphSamplingBenchmark(
                create_sampler=create_sampler,
                sampler_name=sampler_name,
                schema=schema,
                seed_nodeset=seed_nodeset,
                seed_ids=seed_ids,
                query_engine=query_engine,
                num_hops=num_hops,
                hop_width=hop_width,
                batch_size=batch_size,
                reverse=reverse,
                print_query=print_query,
            ),
            repetitions=repetitions,
            warmup_repetitions=warmup_repetitions,
        )
    benchmarker.add_separator()
  benchmarker.print_results()
