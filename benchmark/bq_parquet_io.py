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

"""Library for BigQuery Parquet IO benchmarking."""

import dataclasses
from typing import override

import dgf
from dgf.benchmark import utils as benchmark_utils


@dataclasses.dataclass(frozen=True)
class BigQueryGraphConfig:
  """Configuration for BigQuery Graph benchmarking."""

  project_id: str
  dataset_id: str
  graph_id: str
  gcs_prefix: str


class ReadBigQueryGraphInMemoryParquetExport(benchmark_utils.Benchmark):
  """Read a BigQuery Graph using Parquet Export in memory."""

  def __init__(self, config: BigQueryGraphConfig):
    self.config = config
    self._num_nodes = -1
    self._num_edges = -1

  @override
  def name(self) -> str:
    return "Read a BigQuery Graph using Parquet Export in memory"

  @override
  def run(self):
    graph, schema = dgf.io.read_bigquery_graph(
        project=self.config.project_id,
        dataset=self.config.dataset_id,
        graph=self.config.graph_id,
        work_dir=self.config.gcs_prefix,
    )
    self._num_nodes = sum(x.num_nodes for x in graph.node_sets.values())  # pyrefly: ignore[no-matching-overload]
    self._num_edges = sum(
        x.adjacency.shape[1] for x in graph.edge_sets.values()
    )
    del graph
    del schema

  @override
  def num_units(self) -> int:
    return self._num_nodes

  @override
  def details(self) -> str:
    return f"num_nodes={self._num_nodes} num_edges={self._num_edges}"


def run_bq_parquet_benchmark(config: BigQueryGraphConfig):
  """Runs the BigQuery Parquet IO benchmark."""
  benchmarker = benchmark_utils.Benchmarker()
  benchmarker.run(
      ReadBigQueryGraphInMemoryParquetExport(config=config),
      repetitions=1,
      warmup_repetitions=0,
  )
  benchmarker.print_results()
