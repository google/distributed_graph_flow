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

"""Benchmarking of IO operations on in memory graphs."""

import enum
import os
import random
from typing import Any, Callable, List, Optional
import dgf
from dgf.benchmark import utils as benchmark_utils
from dgf.src.util import log


class OutputFormat(enum.Enum):
  """The format of the sampler output.

  Attributes:
    NUMPY: Generate a "InMemoryGraph".
    JAX: Generate a "JAXInMemoryGraph".
    JAX_SD: Generate a Sparse Deferred Struct with JAX engine.
  """

  NUMPY = "NUMPY_IN_MEMORY"
  JAX = "JAX_IN_MEMORY"
  JAX_SD = "JAX_SD"


class GenGraphSamples(benchmark_utils.Benchmark):
  """Generate independent graph samples in memory."""

  num_nodes: int
  num_samples: int
  sampler: dgf.sampling.Sampler
  sampling_config: dgf.sampling.SimpleSamplingConfig
  output_fn: Callable[[List[dgf.data.InMemoryGraph]], Any]
  edgeset_to_mask: Optional[str]

  def __init__(
      self,
      *,
      graph: dgf.data.InMemoryGraph,
      schema: dgf.data.GraphSchema,
      seed_nodeset: str,
      num_hops: int,
      extract_features: bool = True,
      output_format: OutputFormat = OutputFormat.NUMPY,
      edgeset_to_mask: Optional[str] = None,
      with_replacement: bool = False,
  ):
    self.seed_nodeset = seed_nodeset
    self.extract_features = extract_features
    self.output_format = output_format
    self.graph = graph
    self.schema = schema
    self.num_hops = num_hops
    self.with_replacement = with_replacement
    self.hop_width = 5
    self.batch_size = 12
    self.edgeset_to_mask = edgeset_to_mask
    self.set_unit_multiplicator(self.batch_size)

    self.sum_sampled_nodes = 0
    self.num_samples = 0

  def name(self) -> str:
    return "GraphSAGE"

  def setup(self):

    self.sampling_config = dgf.sampling.SimpleSamplingConfig(
        seed_nodeset=self.seed_nodeset,
        num_hops=self.num_hops,
        hop_width=self.hop_width,
        with_replacement=self.with_replacement,
    )
    sampling_plan = dgf.sampling.simple_sampling_config_to_sampling_plan(
        self.sampling_config,
        self.schema,
    )
    self.sampler = dgf.sampling.create_sampler(
        self.graph,
        sampling_plan,
        self.schema,
        return_features=self.extract_features,
        return_node_idxs=not self.extract_features,
        batch_size=self.batch_size,
        edgeset_to_mask=self.edgeset_to_mask,
    )
    num_nodes = self.graph.node_sets[
        self.sampling_config.seed_nodeset
    ].num_nodes
    assert num_nodes is not None
    self.num_nodes = num_nodes

    if self.output_format == OutputFormat.NUMPY:

      def output_fn(
          graphs: List[dgf.data.InMemoryGraph],
      ):
        # Nothing to do
        return graphs

    elif self.output_format == OutputFormat.JAX:

      def output_fn(
          graphs: List[dgf.data.InMemoryGraph],
      ):
        return [dgf.convert.graph_to_jax_graph(g) for g in graphs]

    elif self.output_format == OutputFormat.JAX_SD:

      def output_fn(
          graphs: List[dgf.data.InMemoryGraph],
      ):
        return [
            dgf.convert.graph_to_sparse_deferred_struct(g, schema=self.schema)
            for g in graphs
        ]

    else:
      assert False
    self.output_fn = output_fn

  def run_unit(self):
    seed_node_idxs = [
        random.randrange(0, self.num_nodes) for _ in range(self.batch_size)
    ]
    if self.edgeset_to_mask is not None:
      # Pass dummy masked edge indices (e.g. all 0).
      masked_edge_idxs = [0 for _ in range(self.batch_size)]
      samples = self.sampler.sample(
          seed_node_idxs, masked_edge_idxs=masked_edge_idxs
      )
    else:
      samples = self.sampler.sample(seed_node_idxs)

    for sample in samples:
      for ns in sample.node_sets.values():
        self.sum_sampled_nodes += ns.num_nodes
    _ = self.output_fn(samples)
    self.num_samples += 1

  def details(self) -> str:
    return (
        f"num_hops={self.sampling_config.num_hops}"
        f" hop_width={self.sampling_config.hop_width}"
        f" extract_features={self.extract_features}"
        f" output_format={self.output_format.value}"
        f" with_replacement={self.with_replacement}"
        f" batch_size={self.batch_size}"
        f" edgeset_to_mask={self.edgeset_to_mask}"
        f" nodes_per_sample={self.sum_sampled_nodes / self.num_samples}"
    )


class GenGraphSubsets(benchmark_utils.Benchmark):
  """Generate a single graph subset (from multiple seeds) in memory."""

  num_nodes: int
  num_samples: int
  sampler: dgf.sampling.Sampler
  sampling_config: dgf.sampling.SimpleSamplingConfig

  def __init__(
      self,
      graph: dgf.data.InMemoryGraph,
      schema: dgf.data.GraphSchema,
      seed_nodeset: str,
      num_hops: int,
      extract_features: bool = True,
  ):
    self.graph = graph
    self.schema = schema
    self.seed_nodeset = seed_nodeset
    self.num_hops = num_hops
    self.batch_size = 12
    self.extract_features = extract_features

    self.sum_sampled_nodes = 0
    self.num_samples = 0

    self.set_unit_multiplicator(self.batch_size)

  def name(self) -> str:
    return "Subgraph"

  def setup(self):
    self.sampling_config = dgf.sampling.SimpleSamplingConfig(
        seed_nodeset=self.seed_nodeset,
        num_hops=self.num_hops,
        hop_width=1,  # Not used
    )

    self.sampler = dgf.sampling.create_sampler(
        self.graph,
        self.sampling_config,
        schema=self.schema,
        return_features=self.extract_features,
        return_node_idxs=not self.extract_features,
        batch_size=self.batch_size,
    )
    num_nodes = self.graph.node_sets[
        self.sampling_config.seed_nodeset
    ].num_nodes
    assert num_nodes is not None
    self.num_nodes = num_nodes

  def run_unit(self):
    seed_node_idxs = [
        random.randrange(0, self.num_nodes) for _ in range(self.batch_size)
    ]
    subgraph = self.sampler.subgraph(seed_node_idxs)
    for ns in subgraph.node_sets.values():
      self.sum_sampled_nodes += ns.num_nodes
    self.num_samples += 1

  def details(self) -> str:
    return (
        f"num_hops={self.sampling_config.num_hops}"
        f" extract_features={self.extract_features}"
        f" batch_size={self.batch_size}"
        f" nodes_per_sample={self.sum_sampled_nodes / self.num_samples}"
    )


def in_process_sampling(
    work_dir: Optional[str],
    gf_graph_path: str,
    seed_nodeset: str,
    list_num_hops: List[int],
    benchmark_output_formats: bool = True,
):
  """Benchmarks the IO of in-memory graphs."""

  log.info("Loading graph")
  load_fn = lambda: dgf.io.read_graph(gf_graph_path, verbose=True)
  if work_dir is not None:
    dgf.filesystem.makedirs(work_dir)
    graph, schema = dgf.io.cache(
        os.path.join(work_dir, "in_process_sampling.pickle"), load_fn
    )
  else:
    graph, schema = load_fn()

  benchmarker = benchmark_utils.Benchmarker()

  # Dynamically select the first edgeset in the schema to mask, if available.
  edgeset_names = list(schema.edge_sets.keys())
  edgeset_to_mask = edgeset_names[0] if edgeset_names else None

  for num_hops in list_num_hops:
    for extract_features in [False, True]:
      benchmarker.run(
          GenGraphSubsets(
              graph=graph,
              schema=schema,
              seed_nodeset=seed_nodeset,
              extract_features=extract_features,
              num_hops=num_hops,
          ),
          repetitions=1,
          warmup_repetitions=1,
      )

      benchmarker.run(
          GenGraphSamples(
              num_hops=num_hops,
              graph=graph,
              schema=schema,
              seed_nodeset=seed_nodeset,
              extract_features=extract_features,
              output_format=OutputFormat.NUMPY,
          ),
          repetitions=1,
          warmup_repetitions=1,
      )

      if edgeset_to_mask is not None:
        benchmarker.run(
            GenGraphSamples(
                num_hops=num_hops,
                graph=graph,
                schema=schema,
                seed_nodeset=seed_nodeset,
                extract_features=extract_features,
                output_format=OutputFormat.NUMPY,
                edgeset_to_mask=edgeset_to_mask,
            ),
            repetitions=1,
            warmup_repetitions=1,
        )

  if benchmark_output_formats:
    for output_format in [
        OutputFormat.NUMPY,
        OutputFormat.JAX,
        OutputFormat.JAX_SD,
    ]:
      benchmarker.run(
          GenGraphSamples(
              graph=graph,
              schema=schema,
              seed_nodeset=seed_nodeset,
              extract_features=False,
              output_format=output_format,
              num_hops=list_num_hops[0],
              with_replacement=False,
          ),
          repetitions=1,
          warmup_repetitions=1,
      )

  benchmarker.print_results()
