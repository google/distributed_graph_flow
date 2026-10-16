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

"""Benchmarking of BigQuery Graph sampling."""

from collections.abc import Sequence
import dataclasses
import dgf
from dgf.benchmark import utils as benchmark_utils
from dgf.src.io.gcp import bigquery_graph as bigquery_graph_lib
from dgf.src.io.gcp import bigquery_graph_metadata as bigquery_graph_metadata_lib


@dataclasses.dataclass
class BigQuerySamplingBenchmarkConfig:
  """Configuration for the BigQuery Graph sampling benchmark."""

  project_id: str = "biggraphs-poc"
  dataset_id: str = "mag"
  graph_id: str = "mag_graph"
  seed_nodeset: str = "paper"
  seed_ids: Sequence[str] = tuple(f"p{i}" for i in range(64))


class BigQueryGraphSamplingBenchmark(benchmark_utils.Benchmark):
  """Benchmarks subgraph sampling from a BigQuery Property Graph."""

  def __init__(
      self,
      *,
      config: BigQuerySamplingBenchmarkConfig,
      schema: dgf.data.GraphSchema,
      metadata: bigquery_graph_metadata_lib.BigQueryGraphMetadata,
      num_hops: int,
      hop_width: int = 5,
      batch_size: int = 8,
      reverse: bool = True,
      check_optimizations: bool = True,
  ):
    self._config = config
    self._schema = schema
    self._metadata = metadata
    self._num_hops = num_hops
    self._hop_width = hop_width
    self._batch_size = batch_size
    self._reverse = reverse
    self._check_optimizations = check_optimizations
    self._sampler: dgf.sampling.BigQueryGraphSampler | None = None
    self._sum_sampled_nodes = 0
    self._num_samples = 0

  def impl_name(self) -> str:
    """Returns the benchmark implementation name."""
    return "BigQueryGraphSampler"

  def setup(self):
    """Initializes the BigQueryGraphSampler before timing."""
    sampling_config = dgf.sampling.SimpleSamplingConfig(
        seed_nodeset=self._config.seed_nodeset,
        num_hops=self._num_hops,
        hop_width=self._hop_width,
        reverse=self._reverse,
    )
    self._sampler = dgf.sampling.create_bigquery_graph_sampler(
        project=self._config.project_id,
        dataset=self._config.dataset_id,
        graph=self._config.graph_id,
        plan=sampling_config,
        schema=self._schema,
        metadata=self._metadata,
        check_optimizations=self._check_optimizations,
    )

  def run(self):
    """Executes a single batch sampling query against BigQuery."""
    assert self._sampler is not None
    base_seeds = list(self._config.seed_ids)
    batch_seeds = [
        base_seeds[i % len(base_seeds)] for i in range(self._batch_size)
    ]
    samples = self._sampler.sample(batch_seeds)
    for sample in samples:
      for ns in sample.node_sets.values():
        self._sum_sampled_nodes += ns.num_nodes or 0
    self._num_samples += len(samples)

  def num_units(self) -> int:
    """Returns the number of seed nodes sampled per run."""
    return self._batch_size

  def details(self) -> str:
    """Returns benchmark configuration and average node count details."""
    avg_nodes = self._sum_sampled_nodes / max(1, self._num_samples)
    return (
        f"hops={self._num_hops}"
        f" width={self._hop_width}"
        f" batch={self._batch_size}"
        f" reverse={int(self._reverse)}"
        f" nodes/spl.={avg_nodes:.1f}"
    )


def run_bq_sampling_benchmark(
    config: BigQuerySamplingBenchmarkConfig,
    list_num_hops: Sequence[int] = (1, 2),
    list_batch_sizes: Sequence[int] = (1, 8),
    hop_width: int = 5,
    repetitions: int = 2,
    warmup_repetitions: int = 1,
) -> None:
  """Runs the BigQuery Graph sampling benchmark suite."""
  metadata = bigquery_graph_lib.get_metadata(
      config.project_id, config.dataset_id, config.graph_id
  )
  schema = bigquery_graph_lib.metadata_to_schema(
      metadata, combine_as_json=False
  )

  benchmarker = benchmark_utils.Benchmarker()
  first = True
  for num_hops in list_num_hops:
    for batch_size in list_batch_sizes:
      benchmarker.run(
          BigQueryGraphSamplingBenchmark(
              config=config,
              schema=schema,
              metadata=metadata,
              num_hops=num_hops,
              hop_width=hop_width,
              batch_size=batch_size,
              check_optimizations=first,
          ),
          repetitions=repetitions,
          warmup_repetitions=warmup_repetitions,
      )
      first = False
  benchmarker.print_results()
