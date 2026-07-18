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

"""Beam-based sampling using an in-memory sampler for graph topology.

You likely want to use the v2.

In this Beam semi-distributed sampler v1, the graph topology is first aggregated
on each worker from the input Beam graph and then used for in-memory sampling.
This contrasts with sampler v2, which loads graph data using in-memory IO
primitives for faster and more memory-efficient sampling.
"""

import dataclasses
import logging
import time
from typing import Dict, Sequence, Tuple
import apache_beam as beam
from dgf.src.data import distributed_graph
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
import numpy as np


def sample_with_beam_semi_distributed_sampler(
    graph: distributed_graph.Graph,
    plan: config_lib.SamplingPlan,
    seeds: beam.PCollection[distributed_graph.NodeId],
    debug_sampling: bool = False,
) -> beam.PCollection[distributed_graph.PKeyedInMemoryGraph]:
  """Samples subgraphs from a distributed graph using a semi-distributed algo.

  WARNING: This sampler is not operational. Use the
  "sample_with_beam_semi_distributed_sampler_v2".

  This beam sampler generates samples by running the in-process sampler multiple
  times in parallel on different workers. Only the final feature gathering is
  distributed. This sampler is suited for graph where the topologie fits in
  memory. For reference, a graph with 1B edges and 100M nodes takes (1B + 100M)
  * 8 = ~9GB of RAM (assuming uint64 indexing, no compression).

  Usage example:
  ```python
  # Read HGraph
  graph = dgf.io.ReadFromHGraph("/cns/.../my_hgraph")

  # Create the sampling config / plan
  sampling_config = dgf.sampling.SimpleSamplingConfig(
      seed_nodeset="paper", num_hops=2, hop_width=3)
  sampling_plan = dgf.sampling.simple_sampling_config_to_sampling_plan(
      sampling_config,
      graph.schema,
  )

  # Get all the nodes as seeds
  seeds = dgf.sampler.extract_beam_nodes_ids(graph, "paper")

  # Generate samples
  samples = dgf.sampler.sample_with_beam_semi_distributed_sampler(
      graph, plan seeds)

  # Save the samples to disk
  dgf.io.write_to_tf_graph_sample(samples, "/cns/.../samples@*")
  ```

  Args:
    graph: The distributed graph to sample from..
    plan: The sampling plan..
    seeds: PCollection of node IDs to use as seed. These IDs must belong to the
      nodeset specified in `plan.root.nodeset`.
    debug_sampling: If true, enables debug mode in the sampler. Used for unit
      testing.

  Returns:
    A `PCollection` of `InMemoryHeterogeneousGraph` instances, where each
    instance represents a sampled subgraph.
  """
  # TODO(gbm): Extract feature values.
  logging.warning(
      "WIP: The sample_with_beam_semi_distributed_sampler method does not YET"
      " extract the feature values."
  )

  # Shuffle the edges. This allows for the edge reading (input hgraph) and
  # adjacencies computation ("compute_dense_adjacencies") to be separated.
  graph = dataclasses.replace(
      graph,
      edge_sets={
          k: v | f"Shuffle {k}" >> beam.Reshuffle()
          for k, v in graph.edge_sets.items()
      },
  )

  # Compute a dense integer index mapping for the node ids.
  node_id_to_idx, num_nodes = compute_dense_node_idx_as_side_input(
      graph, debug_sampling=debug_sampling
  )

  # Apply dense node mapping to seed node ids
  # TODO(gbm): Add option to feed pre-dense-indexed seeds.
  dense_seeds = compute_dense_seeds(seeds, node_id_to_idx[plan.root.nodeset])

  # Compute the dense adjacency matrices.
  dense_node_adjs = compute_dense_adjacencies(graph, node_id_to_idx)

  # Create samples
  raw_samples = create_raw_samples(
      graph.schema,
      plan,
      dense_node_adjs,
      dense_seeds,
      num_nodes,
      debug_sampling=debug_sampling,
  )

  # Final shuffle of the samples
  raw_samples = raw_samples | "Shuffle the samples" >> beam.Reshuffle()

  return raw_samples


def compute_dense_node_idx_as_side_input(
    graph: distributed_graph.Graph,
    debug_sampling: bool,
) -> Tuple[Dict[str, Dict[bytes, int]], Dict[str, int]]:
  """Build a dense integer indexing map for the node ids.

  Example of output mapping:
    "n1":
      "nodeX" => 0
      "nodeY" => 1
    "n2":
      "nodeZ" => 0
      "nodeW" => 1

  Note: This operation is currently only semi-distributed because the index is
  loaded into memory on a single worker. This is acceptable because the
  semi-distributed sampler already assumes the graph topology fits within a
  single machine's memory, and the node index is smaller than the full
  topology. This operation could be made fully distributed in the future.

  Args:
    graph: The input distributed graph.
    debug_sampling: If true, enables debug mode in the sampler. Used for unit
      testing.

  Returns:
    A tuple with:
      - A dictionary mapping each node set name to a beam side input dictionary,
        itself mapping the original node IDs (bytes) to their new dense integer
        index (int).
      - A beam side input dictionary mapping each node set name to the total
        number of nodes in that set.
  """

  # Note: We have to create those sub-ptransform just to avoid collisions in
  # stage names.
  # TODO(gbm): Find a way to create stage names with a context bases system.
  @beam.ptransform_fn
  def _process(
      nodes: distributed_graph.PNode,
  ) -> beam.PCollection[Tuple[bytes, int]]:

    if debug_sampling:

      def my_enumerate(ids):
        # In debug mode, we sort the ids to facilitate the unit test checks.
        ids.sort()
        return [(id, idx) for idx, id in enumerate(ids)]

    else:

      def my_enumerate(ids):
        return [(id, idx) for idx, id in enumerate(ids)]

    return (
        nodes
        | "Extract id" >> beam.Map(lambda n: n.id)
        # This is the non-distributed part that could be distributed.
        | "To list" >> beam.combiners.ToList()
        | "Enumerate" >> beam.FlatMap(my_enumerate)
    )

  mappings = {}
  num_nodes = []
  for nodeset_name in graph.schema.node_sets.keys():
    nodes = graph.node_sets[nodeset_name]
    id_to_idx = (
        nodes | f"Compute dense node index for {nodeset_name}" >> _process()
    )

    num_nodes_in_set = (
        id_to_idx
        | f"Count nodes for {nodeset_name}" >> beam.combiners.Count.Globally()
        | f"Key num nodes for {nodeset_name}"
        >> beam.Map(
            lambda count, ns_name: (ns_name, count), ns_name=nodeset_name
        )
    )
    num_nodes.append(num_nodes_in_set)
    mappings[nodeset_name] = beam.pvalue.AsDict(id_to_idx)

  return mappings, beam.pvalue.AsDict(
      num_nodes | "Flatten num nodes dict" >> beam.Flatten()
  )


class BatchedNumpyCombineFn(beam.CombineFn):
  """Combines pairs of integers into a numpy array of shape [2, num pairs]."""

  def create_accumulator(self):
    return []

  def add_input(self, acc, pair):
    assert isinstance(acc, list)
    acc.append(pair)
    return acc

  def merge_accumulators(self, accs):
    arrays = []
    for acc in accs:
      if isinstance(acc, list):
        arrays.append(np.array(acc, dtype=np.int64))
      else:
        arrays.append(acc)
    return [np.concatenate(arrays)]

  def extract_output(self, acc):
    assert len(acc) == 1
    value = acc[0]
    if isinstance(value, tuple):
      value = np.array([value], dtype=np.int64)
    # Return an array of shape (2,num_items).
    return value.T


def compute_dense_adjacencies(
    graph: distributed_graph.Graph,
    node_id_to_idx: Dict[str, Dict[bytes, int]],
) -> Dict[str, np.ndarray]:
  """Computes the dense adjacency matrices of the edgesets.

  Example of input:

    Edges:
      "e1": "A"->"B", "A->A"

  Example of output:
    "e1":
      [[1 1]
        [0 1]]

  Args:
    graph: The input distributed graph.
    node_id_to_idx: Dense index mapping for the node ids.

  Returns:
    A beam side input dictionary with the dense adjacency matric (e.g., [2,
    num_edges] shape).
  """

  @beam.ptransform_fn
  def _process(
      edges: distributed_graph.PEdge,
      source_node_id_to_idx: Dict[bytes, int],
      target_node_id_to_idx: Dict[bytes, int],
      edgeset_name: str,
  ) -> beam.PCollection[Tuple[str, np.ndarray]]:

    def _edge_to_np_array(
        edge: distributed_graph.Edge,
        source_node_id_to_idx: Dict[bytes, int],
        target_node_id_to_idx: Dict[bytes, int],
    ) -> Tuple[int, int]:
      source_node_idx = source_node_id_to_idx[edge.source]  # pyrefly: ignore[bad-index]
      target_node_idx = target_node_id_to_idx[edge.target]  # pyrefly: ignore[bad-index]
      return (source_node_idx, target_node_idx)

    return (
        edges
        | "Edge to np array"
        >> beam.Map(
            _edge_to_np_array,
            source_node_id_to_idx=source_node_id_to_idx,
            target_node_id_to_idx=target_node_id_to_idx,
        )
        | "Merge arrays" >> beam.CombineGlobally(BatchedNumpyCombineFn())
        | "Add key" >> beam.Map(lambda x, k: (k, x), k=edgeset_name)
    )

  dense_adjs = []
  for edgeset_name, edgeset_schema in graph.schema.edge_sets.items():
    edges = graph.edge_sets[edgeset_name]
    dense_adjs.append(
        edges
        | f"Extract dense idx for edgeset {edgeset_name}"
        >> _process(
            source_node_id_to_idx=node_id_to_idx[edgeset_schema.source],
            target_node_id_to_idx=node_id_to_idx[edgeset_schema.target],
            edgeset_name=edgeset_name,
        )
    )

  return beam.pvalue.AsDict(
      dense_adjs | "Flatten dense adjacency matrices dict" >> beam.Flatten()
  )


def compute_dense_seeds(
    seeds: beam.PCollection[distributed_graph.NodeId],
    node_id_to_idx: Dict[bytes, int],  # beam.pvalue.AsDict,
) -> beam.PCollection[Tuple[int, distributed_graph.NodeId]]:
  """Computes the dense node idx of seeds.

  Example of input:
    seeds = ["node1", "node3", "node2"]
    node_id_to_idx = {"node1":0, "node2":2, "node3":3}

  Example of output:
    [(0,"node1"), (3,"node3"), (2,"node2")]

  Args:
    seeds: A PCollection of node IDs.
    node_id_to_idx: A beam side input dictionary mapping original node IDs to
      their dense integer index.

  Returns:
    A pcollection of tuple (node idx, original node id).
  """

  return seeds | "Index seeds" >> beam.Map(
      lambda x, index: (index[x], x), index=node_id_to_idx
  )


class RawSampler(beam.DoFn):
  """Generates graph samples without the feature values."""

  def __init__(
      self,
      schema: schema_lib.GraphSchema,
      plan: config_lib.SamplingPlan,
      batch_size: int,
      debug_sampling: bool,
  ):
    self.schema = schema
    self.plan = plan
    self.batch_size = batch_size
    self.sampler = None
    self.debug_sampling = debug_sampling

  def process(
      self,
      seeds: Sequence[Tuple[int, distributed_graph.NodeId]],
      num_nodes: Dict[str, int],
      dense_node_adjs: Dict[str, np.ndarray],
  ):

    if self.sampler is None:
      logging.info("Build sampler")
      start_time = time.time()
      # Create the graph without any features.
      in_memory_graph = in_memory_graph_lib.InMemoryGraph(
          node_sets={
              n: in_memory_graph_lib.InMemoryNodeSet(
                  features={},
                  num_nodes=num_nodes[n],
              )
              for n in self.schema.node_sets.keys()
          },
          edge_sets={
              e: in_memory_graph_lib.InMemoryEdgeSet(
                  adjacency=dense_node_adjs[e],
                  features={},
              )
              for e in self.schema.edge_sets.keys()
          },
      )

      # Create the sampler
      self.sampler = in_memory_sampler_lib.create_sampler(
          in_memory_graph,
          self.plan,
          self.schema,
          batch_size=self.batch_size,
          return_features=False,
          return_node_idxs=True,
          debug_sampling=self.debug_sampling,
      )
      end_time = time.time()
      logging.info("Sampler built in %.4f seconds", end_time - start_time)

    # Create the samples
    seed_idxs, seed_ids = zip(*list(seeds))
    samples = self.sampler.sample(list(seed_idxs))

    # Emit the samples
    for seed_id, sample in zip(seed_ids, samples):
      yield (seed_id, sample)


def create_raw_samples(
    schema: schema_lib.GraphSchema,
    plan: config_lib.SamplingPlan,
    dense_node_adjs: Dict[str, np.ndarray],
    dense_seeds: beam.PCollection[Tuple[int, distributed_graph.NodeId]],
    num_nodes: Dict[str, int],
    debug_sampling: bool,
) -> beam.PCollection[distributed_graph.PKeyedInMemoryGraph]:
  """Create graph samples."""
  # TODO(gbm): Increase batch size when more optimized in the in-memory sampler.
  batch_size = 8

  return (
      dense_seeds
      | "Batch seeds" >> beam.BatchElements(max_batch_size=batch_size)
      | "Sample"
      >> beam.ParDo(
          RawSampler(schema, plan, batch_size, debug_sampling),
          num_nodes,
          dense_node_adjs,
      )
  )
