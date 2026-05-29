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

"""Semi-distributed sampler where data is loaded with in-mem IO instead of beam."""

import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import apache_beam as beam
from apache_beam.utils import shared as beam_shared
from dgf.src.data import distributed_graph
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.io import graph_in_memory as gf_graph_in_memory
from dgf.src.io import schema as schema_io_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import schema as schema_filter_lib
import numpy as np

NodeId = distributed_graph.NodeId
SampleId = NodeId
NodesetName = str
NodeIdx = int


# A unique key used to represent samples within Beam pipeline stages.
# This key must not conflict with any nodeset names.
SAMPLE_KEY = "__gfsample__"

# TODO(gbm): Customize primary key.
KEY_ID = "#id"


def sample_with_beam_semi_distributed_sampler_v2(
    graph_path: str,
    plan: Union[config_lib.SimpleSamplingConfig, config_lib.SamplingPlan],
    seeds: beam.PCollection[NodeId],
    debug_sampling: bool = False,
    num_threads: int = 20,
    beam_feature_collection: bool = False,
    beam_namespace: str = "",
) -> Tuple[distributed_graph.PKeyedInMemoryGraph, schema_lib.GraphSchema]:
  """Samples subgraphs from a distributed graph using a semi-distributed algo.

  This beam sampler generates samples by running the in-process sampler multiple
  times in parallel on different workers. Only the final feature gathering is
  distributed. This sampler is suited for graph where the topology fits in
  memory. For reference, a graph with 1B edges and 100M nodes takes (1B + 100M)
  * 8 = ~9GB of RAM (assuming uint64 indexing, no compression).

  Usage example:
  ```python
  graph_path = ...

  # Use all the nodes as seed nodes.
  graph = dgf.beam.io.read_graph(root, graph_path)
  seeds = dgf.sampler.extract_beam_nodes_ids(graph, "paper")
  seeds = seeds | "Reshuffle seeds" >> beam.Reshuffle()

  # Create the sampling config
  sampling_config = dgf.sampling.SimpleSamplingConfig(
      seed_nodeset="paper", num_hops=2, hop_width=3)

  # Generate samples
  samples = dgf.sampler.sample_with_beam_semi_distributed_sampler_v2(
      graph_path, sampling_config, seeds)

  # Save the samples to disk
  dgf.io.write_to_tf_graph_sample(samples, "/cns/.../samples@*")
  ```

  Args:
    graph_path: Path to graph on disk.
    plan: The sampling plan..
    seeds: PCollection of node IDs to use as seed. These IDs must belong to the
      nodeset specified in `plan.root.nodeset`.
    debug_sampling: If true, enables debug mode in the sampler. Used for unit
      testing.
    num_threads: The number of threads used by each worker.
    beam_feature_collection: If False (default), the feature values are gathered
      by the in-memory sampler. This option is fast but requires more RAM. If
      false, the feature values are gathered by a Beam join after the sampling.
    beam_namespace: Prefix added to all the beam ptransforms.

  Returns:
    A `PCollection` of `InMemoryHeterogeneousGraph` instances, where each
    instance represents a sampled subgraph.
  """

  # TODO(gbm): Add option to read graph data with any graph.

  # TODO(gbm): Use read any graph utility.
  schema = schema_io_lib.read_schema(
      os.path.join(graph_path, gf_graph_in_memory.FILENAME_SCHEMA)
  )

  for nodeset_name, nodeset_schema in schema.node_sets.items():
    if KEY_ID not in nodeset_schema.features:
      raise ValueError(
          f"The nodeset '{nodeset_name}' is missing the required feature "
          f"'{KEY_ID}'. This feature is needed for mapping node IDs."
      )

  # Sample the graph topology.
  samples = (
      seeds
      | f"{beam_namespace}Batch seeds"
      >> beam.BatchElements(max_batch_size=num_threads)
      | f"{beam_namespace}Sample"
      >> beam.ParDo(
          RawSamplerV2(
              graph_path,
              schema,
              plan,
              num_threads,
              debug_sampling,
              beam_feature_collection=beam_feature_collection,
          ),
      )
  )

  if beam_feature_collection:
    # Gather the feature values.
    # Note: Only load the node features (not the edge features).
    schema_filter = schema_lib.GraphSchemaFilter(
        edgeset_fn=lambda key, sch: False
    )
    feature_graph = gf_graph_in_beam_lib.read_graph(
        seeds.pipeline,
        graph_path,
        schema_filter=schema_filter,
        beam_namespace=f"{beam_namespace}semi_distributed_sampler/",
    )
    # Add the feature values to the graph.
    samples = add_features_to_graph_samples(
        samples,
        feature_graph,
        beam_namespace=f"{beam_namespace}AddFeatureToGraph/",
    )

  return samples, schema


class SharedSampler:
  """A wrapper to allow weak references to the shared sampler and mapper."""

  def __init__(self, sampler, mapper):
    self.sampler = sampler
    self.mapper = mapper


class RawSamplerV2(beam.DoFn):

  # All the sampler in this process.
  shared_in_memory_samplers = beam_shared.Shared()

  def __init__(
      self,
      graph_path: str,
      schema: schema_lib.GraphSchema,
      plan: Union[config_lib.SimpleSamplingConfig, config_lib.SamplingPlan],
      num_threads: int,
      debug_sampling: bool,
      beam_feature_collection: bool,
  ):
    self.graph_path = graph_path
    self.plan = plan
    self.num_threads = num_threads
    self.sampler = None
    self.schema = schema
    self.debug_sampling = debug_sampling
    self.beam_feature_collection = beam_feature_collection

  def setup(self):
    def initializer():
      start_time = time.time()
      logging.info(
          "Thread %s: Start loading shared graph",
          threading.current_thread().name,
      )

      read_schema = self.schema

      if self.beam_feature_collection:
        # Only load the ids (not the other features).
        schema_filter = schema_lib.GraphSchemaFilter(
            feature_fn=lambda key, sch: key == KEY_ID
        )
        read_schema = schema_filter_lib.filter_schema(
            read_schema, schema_filter
        )
      in_memory_graph, schema = gf_graph_in_memory.read_graph(
          self.graph_path, override_schema=read_schema, verbose=True
      )

      logging.info("Index data sampler")
      effective_plan = self.plan
      if isinstance(effective_plan, config_lib.SimpleSamplingConfig):
        effective_plan = config_lib.simple_sampling_config_to_sampling_plan(
            effective_plan, schema
        )
      sampler = in_memory_sampler_lib.create_sampler(
          in_memory_graph,
          effective_plan,
          self.schema,
          num_threads=self.num_threads,
          return_features=True,
          return_node_idxs=False,
          debug_sampling=self.debug_sampling,
      )
      end_time = time.time()
      logging.info("Sampler built in %.4f seconds", end_time - start_time)
      seed_node_ids = in_memory_graph.node_sets[
          effective_plan.root.nodeset
      ].features[KEY_ID]
      mapper = {id.item(): idx for idx, id in enumerate(seed_node_ids)}

      return SharedSampler(sampler, mapper)

    shared_sampler = RawSamplerV2.shared_in_memory_samplers.acquire(
        initializer, tag=self.graph_path
    )
    self.sampler = shared_sampler.sampler
    self.mapper = shared_sampler.mapper

  def process(
      self, seeds: Sequence[distributed_graph.NodeId]
  ) -> Iterator[distributed_graph.KeyedInMemoryGraph]:

    if self.sampler is None:
      raise ValueError("Sampler was not initialized in setup.")

    # Get seed node idx
    # TODO(gbm): Implement in c++; use the same as in the GF reader.
    seed_idxs = [self.mapper[node_id] for node_id in seeds]

    # Create the samples
    samples = self.sampler.sample(list(seed_idxs))

    # Emit the samples
    for seed, sample in zip(seeds, samples):
      yield distributed_graph.KeyedInMemoryGraph(seed, sample)


def add_features_to_graph_samples(
    raw_samples: distributed_graph.PKeyedInMemoryGraph,
    feature_graph: distributed_graph.Graph,
    probe_stages: Optional[Dict[str, Any]] = None,
    beam_namespace: str = "",
) -> distributed_graph.PKeyedInMemoryGraph:
  """Adds feature values from "feature_graph" to "raw_samples".

  The steps of this method are:

  Inputs
    1. raw_samples: (sample id, graph sample)
    2. feature_graph (node features) & (edge features; later)

  Stages:
    3. From 1 => (node id, (sample id, node idx) )
    4. From 2 => (node id, Features)
    5. CoGroupByKey 3 and 4 =>
        (node id, {
            _: [(sample id, node idx)]
          , _: [Features] # Should be only one.
          })
    6. From 5 => (sample id, node idx, Features)
    7. CoGroupByKey 1 and 5 =>
        (sample id,{
          _: [graph sample], # Should only be one
          _: [ (node idx, Features) ]
          })
    8. Merge the data in 7 =>
      (sample id, graph sample)
    9. return 8

  Args:
    raw_samples: `PCollection` of `(seed_id, InMemoryHeterogeneousGraph)`
      containing the sampled topology (potentially without full features).
    feature_graph: `distributed_graph.Graph` with full node and edge feature
      values.
    probe_stages: If specified, stage the intermediate results of each stage.
    beam_namespace: Prefix added to all the beam ptransforms.

  Returns:
    A `PCollection` of `(seed_id, InMemoryHeterogeneousGraph)` tuples, where
    each `InMemoryHeterogeneousGraph` has its feature values populated from
    `feature_graph`.
  """

  # Stage 3
  expanded_samples = {
      nodeset_name: (
          raw_samples
          | f"{beam_namespace}Stage3/ExtractNodeIds/{nodeset_name}"
          >> beam.ParDo(Stage3ExpandRawSamples(nodeset_name))
      )
      for nodeset_name in feature_graph.schema.node_sets.keys()
  }

  # Stage 4
  index_feature_graph_nodes = {
      nodeset_name: (
          nodeset
          | f"{beam_namespace}Stage4/KeyFeatureGraphNodes/{nodeset_name}"
          >> beam.Map(Stage4IndexFeatureGraphNodes)
      )
      for nodeset_name, nodeset in feature_graph.node_sets.items()
  }

  # Stage 5
  join_by_node_id = {}
  for nodeset_name in index_feature_graph_nodes.keys():
    join_by_node_id[nodeset_name] = (
        {
            "s": expanded_samples[nodeset_name],
            "f": index_feature_graph_nodes[nodeset_name],
        }
        | f"{beam_namespace}Stage5/CoGroupByKeySamplesAndFeatures/{nodeset_name}"
        >> beam.CoGroupByKey(pipeline=expanded_samples[nodeset_name].pipeline)
        | f"{beam_namespace}Stage5/FilterMissingSamples/{nodeset_name}"
        >> beam.Filter(lambda e: e[1]["s"])
    )

  # Stage 6
  by_sample_id = {}
  for nodeset_name in index_feature_graph_nodes.keys():
    by_sample_id[nodeset_name] = join_by_node_id[
        nodeset_name
    ] | f"{beam_namespace}Stage6/KeyFeaturesBySampleId/{nodeset_name}" >> beam.FlatMap(
        Stage6IndexBySampleId
    )

  # Stage 7
  by_sample_id_with_sample = {**by_sample_id}
  by_sample_id_with_sample[SAMPLE_KEY] = (
      raw_samples
      | f"{beam_namespace}Stage7/KeyRawSamplesBySampleId"
      >> beam.Map(Stage8IndexSample)
  )
  raw_samples_and_features = (
      by_sample_id_with_sample
      | f"{beam_namespace}Stage7/CoGroupByKeySamplesAndFeatures"
      >> beam.CoGroupByKey(pipeline=raw_samples.pipeline)
  )

  # Stage 8
  augmented_samples = (
      raw_samples_and_features
      | f"{beam_namespace}Stage8/AddFeaturesToSamples"
      >> beam.Map(Stage8AddFeatureValueToSample, feature_graph.schema)
  )

  if probe_stages is not None:
    probe_stages["stage_3"] = expanded_samples
    probe_stages["stage_4"] = index_feature_graph_nodes
    probe_stages["stage_5"] = join_by_node_id
    probe_stages["stage_6"] = by_sample_id
    probe_stages["stage_7"] = raw_samples_and_features

  return augmented_samples


class Stage3ExpandRawSamples(beam.DoFn):
  """Expands individual nodes in a raw sample.

  See "add_features_to_graph_samples"'s documentation for the definition.
  """

  def __init__(self, nodeset_name: str):
    self._nodeset_name = nodeset_name

  def process(
      self, element: distributed_graph.KeyedInMemoryGraph
  ) -> Iterator[tuple[NodeId, tuple[SampleId, NodeIdx]]]:

    if element.key is None:
      raise ValueError("Samples should have ids")
    nodeset = element.graph.node_sets[self._nodeset_name]
    node_ids = nodeset.features.get(KEY_ID)
    if node_ids is None:
      raise ValueError(
          f"Node ID feature ({KEY_ID}) not found in nodeset:"
          f" {self._nodeset_name}. The available features are"
          f" {list(nodeset.features.keys())}"
      )
    for node_idx, node_id in enumerate(node_ids):
      # TODO(gbm): Would it be more efficient to store feature_idxs instead of
      # feature names?
      yield (node_id.item(), (element.key, node_idx))


def Stage4IndexFeatureGraphNodes(
    element: distributed_graph.Node,
) -> Tuple[NodeId, distributed_graph.Features]:
  return element.id, element.features or {}


def Stage6IndexBySampleId(
    element: Tuple[
        NodeId,
        Dict[
            str,  # Note: Beam would not support: Literal["s", "f"],
            Union[
                Iterable[Tuple[SampleId, NodeIdx]],
                Iterable[distributed_graph.Features],
            ],
        ],
    ],
) -> Iterator[Tuple[SampleId, Tuple[NodeIdx, distributed_graph.Features]]]:
  _, d = element

  # Get the node feature values.
  f_list = list(d["f"])
  assert len(f_list) == 1, f"Got: {f_list!r}"
  features = f_list[0]

  # Note: The left join ensures this is not empty, and the program struct
  # ensures there is only one element.
  d_list = d["s"]
  for sample_id, node_idx in d_list:
    yield sample_id, (node_idx, features)


def Stage8IndexSample(
    raw_sample: distributed_graph.KeyedInMemoryGraph,
) -> Tuple[SampleId, in_memory_graph_lib.InMemoryGraph]:
  assert raw_sample.key is not None
  return raw_sample.key, raw_sample.graph


def Stage8AddFeatureValueToSample(
    element: Tuple[
        SampleId,
        Dict[
            str,  # Nodeset name or special SAMPLE_KEY value
            Union[
                Iterable[Tuple[NodeIdx, distributed_graph.Features]],
                Iterable[in_memory_graph_lib.InMemoryGraph],
            ],
        ],
    ],
    schema: schema_lib.GraphSchema,
) -> distributed_graph.KeyedInMemoryGraph:
  sample_id, super_dict = element
  raw_graph_list = super_dict[SAMPLE_KEY]
  raw_graph = next(iter(raw_graph_list))
  assert isinstance(raw_graph, in_memory_graph_lib.InMemoryGraph)

  augmented_nodesets = {}
  for nodeset_name, raw_nodeset in raw_graph.node_sets.items():
    src_features = super_dict[nodeset_name]
    num_nodes = raw_nodeset.num_nodes

    dst_features = {}
    for feature_name, feature_schema in schema.node_sets[
        nodeset_name
    ].features.items():
      if feature_name == KEY_ID:
        continue
      values = [None] * num_nodes
      num_values = 0
      for node_idx, row_features in src_features:  # pytype: disable=attribute-error
        values[node_idx] = row_features[feature_name]
        num_values += 1
      assert num_nodes == num_values
      # TODO(gbm): Handle variable length features.
      dst_features[feature_name] = safe_stack(values, feature_schema)

    dst_features[KEY_ID] = raw_nodeset.features[KEY_ID]
    augmented_nodesets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=num_nodes, features=dst_features
    )

  return distributed_graph.KeyedInMemoryGraph(
      sample_id,
      in_memory_graph_lib.InMemoryGraph(
          node_sets=augmented_nodesets,
          edge_sets=raw_graph.edge_sets,
      ),
  )


def safe_stack(
    values: List[np.ndarray], schema: schema_lib.FeatureSchema
) -> np.ndarray:
  """Stacks feature value arrays, handling static and variable shapes."""
  try:
    if not values:
      return np.empty(
          dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[schema.format],
          shape=(0,) + (schema.shape or ()),
      )
    if schema.is_static_shape():
      return np.stack(
          values,
          axis=0,
          dtype=feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[schema.format],
      )
    else:
      array = np.empty(len(values), dtype=np.object_)
      array[:] = values
      return array

  except ValueError as e:
    raise ValueError(f"Values: {values}") from e
