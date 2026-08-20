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

import dataclasses
import os
import pickle
from typing import override

import dgf
from dgf.benchmark import utils as benchmark_utils


def _compute_num_nodes_edges(graph) -> tuple[int, int]:
  num_nodes = sum([x.num_nodes for x in graph.node_sets.values()])  # pyrefly: ignore[no-matching-overload]
  num_edges = sum([x.adjacency.shape[1] for x in graph.edge_sets.values()])
  return num_nodes, num_edges


@dataclasses.dataclass(frozen=True)
class SpannerGraphConfig:
  """Configuration for Spanner Graph benchmarking."""

  project_id: str
  instance_id: str
  database_id: str
  graph_id: str


class ReadGraphAIGraphInMemory(benchmark_utils.Benchmark):
  """Read a GraphAI Graph from disk in memory."""

  def __init__(self, hgraph_path: str):
    self.hgraph_path = hgraph_path
    self.num_nodes = -1
    self.num_edges = -1

  def impl_name(self) -> str:
    return "Read GraphAI Graph"

  def run(self):
    graph, schema = dgf.io.read_graphai_hgraph(self.hgraph_path, verbose=True)
    del schema
    self.num_nodes, self.num_edges = _compute_num_nodes_edges(graph)

  def num_units(self) -> int:
    return self.num_nodes

  def details(self) -> str:
    return f"num_nodes={self.num_nodes} num_edges={self.num_edges}"


class ReadGFGraphInMemory(benchmark_utils.Benchmark):
  """Read a GF Graph from disk in memory."""

  def __init__(self, gf_graph_path: str):
    """Initializes the benchmark.

    Args:
      gf_graph_path: Path to the GF graph on disk.
    """
    self._gf_graph_path = gf_graph_path
    self._num_nodes = -1
    self._num_edges = -1

  @override
  def impl_name(self) -> str:
    return "Read DGF Graph"

  @override
  def run(self):
    graph, schema = dgf.io.read_graph(self._gf_graph_path, verbose=True)
    del schema
    self._num_nodes, self._num_edges = _compute_num_nodes_edges(graph)

  @override
  def num_units(self) -> int:
    return self._num_nodes

  @override
  def details(self) -> str:
    return f"num_nodes={self._num_nodes} num_edges={self._num_edges}"


class WriteGFGraphInMemory(benchmark_utils.Benchmark):
  """Writes a GF Graph from disk in memory."""

  def __init__(self, work_dir: str, gf_graph_path: str):
    """Initializes the benchmark."""
    self.work_dir = work_dir
    self._gf_graph_path = gf_graph_path
    self._num_nodes = -1
    self._num_edges = -1
    self._schema = None
    self._graph = None

  @override
  def impl_name(self) -> str:
    return "Write DGF Graph"

  def setup(self):
    self._graph, self._schema = dgf.io.read_graph(
        self._gf_graph_path, verbose=True
    )

    self._num_nodes, self._num_edges = _compute_num_nodes_edges(self._graph)

  @override
  def run(self):
    assert self._schema is not None
    assert self._graph is not None
    dgf.io.write_graph(
        graph=self._graph,
        schema=self._schema,
        path=os.path.join(self.work_dir, f"WriteGFGraphInMemory"),
        max_num_shards=20,
    )

  @override
  def num_units(self) -> int:
    return self._num_nodes

  @override
  def details(self) -> str:
    return f"num_nodes={self._num_nodes} num_edges={self._num_edges}"


class ReadPickleInMemoryGraph(benchmark_utils.Benchmark):
  """Read a pickled in-memory graph from disk."""

  def __init__(self, work_dir: str, hgraph_path: str):
    self.work_dir = work_dir
    self.hgraph_path = hgraph_path
    self.num_nodes = -1
    self.num_edges = -1
    self.pickle_path = None

  def impl_name(self) -> str:
    return "Read Pickled DGF Graph"

  def setup(self):
    graph, schema = dgf.io.cache(
        os.path.join(self.work_dir, "cache_ReadPickleInMemoryGraph.pickle"),
        lambda: dgf.io.read_graphai_hgraph(self.hgraph_path, verbose=True),
    )
    self.num_nodes, self.num_edges = _compute_num_nodes_edges(graph)
    self.pickle_path = os.path.join(self.work_dir, "graph.pickle")
    with open(self.pickle_path, "wb") as f:
      pickle.dump((graph, schema), f)

  def run(self):
    assert self.pickle_path is not None
    with open(self.pickle_path, "rb") as f:
      graph, schema = pickle.load(f)
      del graph
      del schema

  def num_units(self) -> int:
    return self.num_nodes

  def details(self) -> str:
    return f"num_nodes={self.num_nodes} num_edges={self.num_edges}"


class ReadTFGraphSamplesInMemory(benchmark_utils.Benchmark):
  """Read TF Graph samples in memory."""

  def __init__(self, tf_graph_samples_path: str, max_samples: int = 10000):
    self.num_samples = -1
    self.tf_graph_samples_path = tf_graph_samples_path
    self.schema = None
    self.max_samples = max_samples

  def setup(self):
    self.schema = dgf.io.read_schema(
        os.path.join(self.tf_graph_samples_path, "schema.json")
    )

  def impl_name(self) -> str:
    return "Read TF-GNN Graph samples"

  def run(self):
    assert self.schema is not None
    generator = dgf.io.read_tfgnn_graphs(
        os.path.join(self.tf_graph_samples_path, "data@*.tfrecord.gz"),
        self.schema,
    )
    self.num_samples = 0
    for sample in generator:
      self.num_samples += 1
      del sample
      if self.num_samples >= self.max_samples:
        break

  def num_units(self) -> int:
    return self.num_samples

  def details(self) -> str:
    return f"num_samples={self.num_samples}"


class WriteTFGraphSamplesInMemory(benchmark_utils.Benchmark):
  """Write TF Graph samples from in memory."""

  def __init__(
      self, work_dir: str, tf_graph_samples_path: str, max_samples: int = 10000
  ):
    self.num_samples = -1
    self.tf_graph_samples_path = tf_graph_samples_path
    self.schema = None
    self.graphs = []
    self.work_dir = work_dir
    self.max_samples = max_samples

  def setup(self):
    self.schema = dgf.io.read_schema(
        os.path.join(self.tf_graph_samples_path, "schema.json")
    )

    def build_graphs():
      graphs = []
      generator = dgf.io.read_tfgnn_graphs(
          os.path.join(self.tf_graph_samples_path, "data@*.tfrecord.gz"), self.schema  # pyrefly: ignore[bad-argument-type]
      )
      for graph in generator:
        graphs.append(graph)
        if len(graphs) >= self.max_samples:
          break
      return graphs

    self.graphs = dgf.io.cache(
        os.path.join(self.work_dir, "cache_WriteTFGraphSamplesInMemory.pickle"),
        build_graphs,
    )

  def impl_name(self) -> str:
    return "Write TF-GNN Graph Samples"

  def run(self):
    assert self.schema is not None
    dgf.io.write_tfgnn_graphs(
        iter(self.graphs),  # pyrefly: ignore[bad-argument-type]
        os.path.join(self.work_dir, "samples@20.rio"),
        schema=self.schema,
    )

  def num_units(self) -> int:
    return len(self.graphs)

  def details(self) -> str:
    return f"num_samples={len(self.graphs)}"
