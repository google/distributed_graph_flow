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

r"""Integration test.
"""

import time
from typing import List
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import cache as cache_lib
from dgf.src.io.gcp import spanner_graph as spanner_graph_io_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.sampling.gcp import spanner_graph_sampler as spanner_graph_sampler_lib
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np

InMemoryGraph = in_memory_graph_lib.InMemoryGraph
InMemoryNodeSet = in_memory_graph_lib.InMemoryNodeSet
InMemoryEdgeSet = in_memory_graph_lib.InMemoryEdgeSet

DEBUG_SAMPLING = True  # If true, usses deterministic sampling.


def normalize_graph(
    graph: in_memory_graph_lib.InMemoryGraph, schema: schema_lib.GraphSchema
) -> in_memory_graph_lib.InMemoryGraph:
  node_mappings = {}
  new_node_sets = {}

  for nodeset_name, nodeset in graph.node_sets.items():
    primary_key_feature = analyse_schema_lib.primary_feature(
        nodeset_name, schema.node_sets[nodeset_name]
    )
    ids = nodeset.features[primary_key_feature]
    sorted_idxs = np.argsort(ids)

    old_to_new = np.zeros_like(sorted_idxs)
    for new_idx, old_idx in enumerate(sorted_idxs):
      old_to_new[old_idx] = new_idx
    node_mappings[nodeset_name] = old_to_new

    new_features = {}
    for feat_name, feat_val in nodeset.features.items():
      new_features[feat_name] = feat_val[sorted_idxs]

    new_node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=nodeset.num_nodes,
        features=new_features,
    )

  new_edge_sets = {}
  for edgeset_name, edgeset in graph.edge_sets.items():
    edgeset_schema = schema.edge_sets[edgeset_name]
    source_nodeset = edgeset_schema.source
    target_nodeset = edgeset_schema.target

    src_map = node_mappings[source_nodeset]
    trg_map = node_mappings[target_nodeset]

    adj = edgeset.adjacency
    new_adj = np.zeros_like(adj)
    new_adj[0] = src_map[adj[0]]
    new_adj[1] = trg_map[adj[1]]

    sorted_edge_idxs = np.lexsort((new_adj[1], new_adj[0]))
    new_adj = new_adj[:, sorted_edge_idxs]

    new_features = {}
    for feat_name, feat_val in edgeset.features.items():
      new_features[feat_name] = feat_val[sorted_edge_idxs]

    new_edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=new_adj,
        features=new_features,
    )

  return in_memory_graph_lib.InMemoryGraph(
      node_sets=new_node_sets,
      edge_sets=new_edge_sets,
  )


def print_graph_stats(
    name: str,
    samples: List[in_memory_graph_lib.InMemoryGraph],
    seed_ids: List[bytes],
):
  for sample, seed_id in zip(samples, seed_ids):
    num_nodes = sum(nodeset.num_nodes for nodeset in sample.node_sets.values())
    num_edges = sum(
        edgeset.num_edges() for edgeset in sample.edge_sets.values()
    )
    seed_id_str = (
        seed_id.decode("utf-8") if isinstance(seed_id, bytes) else str(seed_id)
    )
    print(
        f"\t{name} sample for seed {seed_id_str}: {num_nodes} nodes,"
        f" {num_edges} edges"
    )


class E2EArxivTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.spanner_graph_config = {
        "project": "biggraphs-poc",
        "instance": "gcp-gnns",
        "database": "ogbn_arxiv",
        "graph": "ogbn_arxiv",
    }

    print("Downloading graph from Spanner (or loading from cache)...")
    start_time = time.time()
    cls.graph, cls.schema = cache_lib.cache(
        "/tmp/spanner_graph_sampler_integration_test_arxiv_cache_v2.pkl",
        lambda: spanner_graph_io_lib.read_spanner_graph(
            **cls.spanner_graph_config, verbose=2
        ),
    )
    print(f"Graph loaded in {time.time() - start_time:.2f} seconds.")

  def test_benchmark_and_equivalence(self):
    seed_ids = [b"10", b"11"]

    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="nodes",
        num_hops=3,
        hop_width=10,
        reverse=True,
    )

    # CTE Sampler
    cte_sampler = spanner_graph_sampler_lib.create_graph_spanner_sampler(
        schema=self.schema,
        plan=plan,
        **self.spanner_graph_config,
        debug_sampling=DEBUG_SAMPLING,
    )
    start_time = time.time()
    cte_samples = cte_sampler.sample(seed_ids)
    cte_time = time.time() - start_time
    in_memory_graph_validate_lib.validate_graph(
        cte_samples[0], self.schema, raise_on_warning=False
    )
    print(f"CTE sampling took {cte_time:.4f} seconds")
    print_graph_stats("CTE", cte_samples, seed_ids)

    # In-Memory Sampler
    plan_tree = config_lib.simple_sampling_config_to_sampling_plan(
        plan, self.schema
    )
    in_memory_sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan_tree,
        self.schema,
        batch_size=len(seed_ids),
        num_threads=0,
        debug_sampling=DEBUG_SAMPLING,
    )

    # Map seed IDs to indices for in-memory sampler
    node_ids = self.graph.node_sets["nodes"].features["id"]
    seed_bytes = seed_ids
    seed_indices = []
    for sb in seed_bytes:
      idx = np.where(node_ids == sb)[0]
      if len(idx) == 0:
        raise ValueError(f"Seed ID {sb} not found in graph")
      seed_indices.append(idx[0])

    start_time = time.time()
    in_mem_samples = in_memory_sampler.sample(seed_indices)
    in_mem_time = time.time() - start_time
    in_memory_graph_validate_lib.validate_graph(
        in_mem_samples[0], self.schema, raise_on_warning=False
    )
    print(f"In-Memory sampling took {in_mem_time:.4f} seconds\n")
    print_graph_stats("In-Memory", in_mem_samples, seed_ids)

    self.assertEqual(len(cte_samples), len(seed_ids))
    self.assertEqual(len(in_mem_samples), len(seed_ids))

    if DEBUG_SAMPLING:
      for i in range(len(seed_ids)):
        norm_cte = normalize_graph(cte_samples[i], self.schema)
        norm_in_mem = normalize_graph(in_mem_samples[i], self.schema)

        # Compare In-Memory vs CTE
        test_util.assert_are_equal(self, norm_in_mem, norm_cte)


class E2EMagTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.spanner_graph_config = {
        "project": "biggraphs-poc",
        "instance": "gcp-gnns",
        "database": "ogbn_mag",
        "graph": "ogbn_mag",
    }

    cls.test_in_process = False

    if cls.test_in_process:
      print("Downloading graph from Spanner (or loading from cache)...")
      start_time = time.time()
      cls.graph, cls.schema = cache_lib.cache(
          "/tmp/spanner_graph_sampler_integration_test_mag_cache_v2.pkl",
          lambda: spanner_graph_io_lib.read_spanner_graph(
              **cls.spanner_graph_config, verbose=2, max_workers=4
          ),
      )
      print(f"Graph loaded in {time.time() - start_time:.2f} seconds.")

  def test_benchmark_and_equivalence(self):
    seed_ids = [b"paper0", b"paper1"]

    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="paper",
        num_hops=2,
        hop_width=5,
        reverse=True,
    )

    # CTE Sampler
    spanner_schema = spanner_graph_io_lib.read_spanner_graph_schema(
        **self.spanner_graph_config
    )
    cte_sampler = spanner_graph_sampler_lib.create_graph_spanner_sampler(
        schema=spanner_schema,
        plan=plan,
        **self.spanner_graph_config,
        debug_sampling=DEBUG_SAMPLING,
    )
    start_time = time.time()
    cte_samples = cte_sampler.sample(seed_ids)
    cte_time = time.time() - start_time
    in_memory_graph_validate_lib.validate_graph(
        cte_samples[0], spanner_schema, raise_on_warning=False
    )
    print(f"CTE sampling took {cte_time:.4f} seconds")
    print_graph_stats("CTE", cte_samples, seed_ids)

    if self.test_in_process:
      # In-Memory Sampler
      plan_tree = config_lib.simple_sampling_config_to_sampling_plan(
          plan, self.schema
      )
      in_memory_sampler = in_memory_sampler_lib.create_sampler(
          self.graph,
          plan_tree,
          self.schema,
          batch_size=len(seed_ids),
          num_threads=0,
          debug_sampling=DEBUG_SAMPLING,
      )

      # Map seed IDs to indices for in-memory sampler
      node_ids = self.graph.node_sets["paper"].features["id"]
      seed_bytes = seed_ids
      seed_indices = []
      for sb in seed_bytes:
        idx = np.where(node_ids == sb)[0]
        if len(idx) == 0:
          raise ValueError(f"Seed ID {sb} not found in graph")
        seed_indices.append(idx[0])

      start_time = time.time()
      in_mem_samples = in_memory_sampler.sample(seed_indices)
      in_mem_time = time.time() - start_time
      in_memory_graph_validate_lib.validate_graph(
          in_mem_samples[0], self.schema, raise_on_warning=False
      )
      print(f"In-Memory sampling took {in_mem_time:.4f} seconds\n")
      print_graph_stats("In-Memory", in_mem_samples, seed_ids)

      self.assertEqual(len(cte_samples), len(seed_ids))
      self.assertEqual(len(in_mem_samples), len(seed_ids))

      if DEBUG_SAMPLING:
        for i in range(len(seed_ids)):
          norm_cte = normalize_graph(cte_samples[i], self.schema)
          norm_in_mem = normalize_graph(in_mem_samples[i], self.schema)

          # Compare In-Memory vs CTE
          test_util.assert_are_equal(self, norm_in_mem, norm_cte)


if __name__ == "__main__":
  absltest.main()
