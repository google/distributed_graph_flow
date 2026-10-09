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

from collections.abc import Callable
import dataclasses
import enum
import os
import random
from typing import Any
import dgf
from dgf.benchmark import utils as benchmark_utils
from dgf.src.analyse import padding as padding_analysis_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.util import log
import numpy as np


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


class MergeMode(enum.Enum):
  """How the graph samples of a batch are merged.

  Attributes:
    NONE: The samples are not merged.
    GRAPH_MERGER: `Sampler.sample` + `GraphMerger` with padding.
    SAMPLE_MERGED: `Sampler.sample_merged` with padding (c++ merging and
      concurrent feature gathering).
  """

  NONE = "NONE"
  GRAPH_MERGER = "GRAPH_MERGER"
  SAMPLE_MERGED = "SAMPLE_MERGED"


class GenGraphSamples(benchmark_utils.Benchmark):
  """Generate independent graph samples in memory."""

  num_nodes: int
  num_samples: int
  sampler: dgf.sampling.Sampler
  sampling_config: dgf.sampling.SimpleSamplingConfig
  output_fn: Callable[[list[dgf.data.InMemoryGraph]], Any]
  edgeset_to_mask: str | None

  def __init__(
      self,
      *,
      graph: dgf.data.InMemoryGraph,
      schema: dgf.data.GraphSchema,
      seed_nodeset: str,
      num_hops: int,
      extract_features: bool = True,
      output_format: OutputFormat = OutputFormat.NUMPY,
      edgeset_to_mask: str | None = None,
      with_replacement: bool = False,
      multi_visit: bool = True,
      merge_mode: MergeMode = MergeMode.NONE,
      batch_size: int = 12,
  ):
    self.seed_nodeset = seed_nodeset
    self.extract_features = extract_features
    self.output_format = output_format
    self.graph = graph
    self.schema = schema
    self.num_hops = num_hops
    self.with_replacement = with_replacement
    self.hop_width = 5
    self.batch_size = batch_size
    self.merge_mode = merge_mode
    self.edgeset_to_mask = edgeset_to_mask
    self.set_unit_multiplicator(self.batch_size)
    self.multi_visit = multi_visit

    self.sum_sampled_nodes = 0
    self.num_samples = 0

  def impl_name(self) -> str:
    return "GraphSAGE"

  def setup(self):

    self.sampling_config = dgf.sampling.SimpleSamplingConfig(
        seed_nodeset=self.seed_nodeset,
        num_hops=self.num_hops,
        hop_width=self.hop_width,
        with_replacement=self.with_replacement,
        multi_visit=self.multi_visit,
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
          graphs: list[dgf.data.InMemoryGraph],
      ):
        # Nothing to do
        return graphs

    elif self.output_format == OutputFormat.JAX:

      def output_fn(
          graphs: list[dgf.data.InMemoryGraph],
      ):
        return [dgf.convert.graph_to_jax_graph(g) for g in graphs]

    elif self.output_format == OutputFormat.JAX_SD:

      def output_fn(
          graphs: list[dgf.data.InMemoryGraph],
      ):
        return [
            dgf.convert.graph_to_sparse_deferred_struct(g, schema=self.schema)
            for g in graphs
        ]

    else:
      assert False
    self.output_fn = output_fn

    if self.merge_mode != MergeMode.NONE:
      # Output schema of the sampler.
      node_sets = {}
      for name, node_set in self.schema.node_sets.items():
        features = dict(node_set.features) if self.extract_features else {}
        if not self.extract_features:
          features["#idx"] = dgf.data.FeatureSchema(
              format=dgf.data.FeatureFormat.INTEGER_64
          )
        node_sets[name] = dataclasses.replace(node_set, features=features)
      self.merge_schema = dataclasses.replace(self.schema, node_sets=node_sets)

      def gen_merged_samples():
        for _ in range(10):
          yield self.sampler.sample_merged(
              np.random.randint(
                  0, self.num_nodes, size=self.batch_size, dtype=np.int64
              )
          )[0]

      padding = padding_analysis_lib.padding_from_graph_generator(
          self.merge_schema, gen_merged_samples(), relative_margin=0.3
      )
      self.graph_merger = merge_lib.GraphMerger(
          self.merge_schema, padding=padding
      )
      self.padded_sampler = self.sampler.with_padding(padding)

  def run_unit(self):
    seed_node_idxs = np.random.randint(
        0, self.num_nodes, size=self.batch_size, dtype=np.int64
    )
    if self.merge_mode != MergeMode.NONE:
      if self.merge_mode == MergeMode.GRAPH_MERGER:
        _, offsets = self.graph_merger(self.sampler.sample(seed_node_idxs))
      elif self.merge_mode == MergeMode.SAMPLE_MERGED:
        _, offsets = self.padded_sampler.sample_merged(seed_node_idxs)
      else:
        raise ValueError(f"Unsupported merge mode: {self.merge_mode}")
      for node_set_offsets in offsets.values():
        self.sum_sampled_nodes += int(node_set_offsets[-1])
      self.num_samples += 1
      return
    if self.edgeset_to_mask is not None:
      # Pass dummy masked edge indices (e.g. all 0).
      masked_edge_idxs = np.zeros(self.batch_size, dtype=np.int64)
      samples = self.sampler.sample(
          seed_node_idxs, masked_edge_idxs=masked_edge_idxs
      )
    else:
      samples = self.sampler.sample(seed_node_idxs)

    for sample in samples:
      for ns in sample.node_sets.values():
        self.sum_sampled_nodes += ns.num_nodes  # pyrefly: ignore[unsupported-operation]
    _ = self.output_fn(samples)
    self.num_samples += 1

  def details(self) -> str:
    return (
        f"hops={self.sampling_config.num_hops}"
        f" width={self.sampling_config.hop_width}"
        f" feat.={self.extract_features}"
        f" format={self.output_format.value}"
        f" with_rep.={int(self.with_replacement)}"
        f" batch={self.batch_size}"
        f" mask={self.edgeset_to_mask}"
        f" nodes/spl.={int(self.sum_sampled_nodes / self.num_samples)}"
        f" multi_visit={self.multi_visit}"
        f" merge={self.merge_mode.value}"
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

  def impl_name(self) -> str:
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
      self.sum_sampled_nodes += ns.num_nodes  # pyrefly: ignore[unsupported-operation]
    self.num_samples += 1

  def details(self) -> str:
    return (
        f"num_hops={self.sampling_config.num_hops}"
        f" extract_features={self.extract_features}"
        f" batch_size={self.batch_size}"
        f" nodes_per_sample={self.sum_sampled_nodes / self.num_samples}"
    )


def in_process_sampling(
    work_dir: str | None,
    gf_graph_path: str,
    seed_nodeset: str,
    list_num_hops: list[int],
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
    for extract_features in [True, False]:
      for with_replacement in [True, False]:
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
                with_replacement=with_replacement,
                multi_visit=True,
            ),
            repetitions=1,
            warmup_repetitions=1,
        )

        if not with_replacement:
          benchmarker.run(
              GenGraphSamples(
                  num_hops=num_hops,
                  graph=graph,
                  schema=schema,
                  seed_nodeset=seed_nodeset,
                  extract_features=extract_features,
                  output_format=OutputFormat.NUMPY,
                  with_replacement=with_replacement,
                  multi_visit=False,
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

    # Merging + padding of batches of samples.
    for extract_features in [True, False]:
      for merge_mode in [MergeMode.GRAPH_MERGER, MergeMode.SAMPLE_MERGED]:
        benchmarker.run(
            GenGraphSamples(
                num_hops=num_hops,
                graph=graph,
                schema=schema,
                seed_nodeset=seed_nodeset,
                extract_features=extract_features,
                with_replacement=False,
                merge_mode=merge_mode,
                batch_size=128,
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
