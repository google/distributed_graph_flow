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

import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import distributed_graph
from dgf.src.data import gf_metadata as gf_metadata_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_memory as gf_graph_in_memory
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np

test_util.disable_diff_truncation()
Edge = distributed_graph.Edge


class ReadGfGraphTest(parameterized.TestCase):

  def test_gf_graph_in_memory(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "gf_graph")
      gen_test_graph.generate_gf_graph(path, edge_ids=True)

      graph, schema = gf_graph_in_memory.read_graph(path)

      self.assertEqual(
          schema,
          gen_test_graph.generate_schema(
              node_ids=True, edge_ids=True, semantic=True
          ),
      )
      expected_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=True
      )
      test_util.assert_are_equal(self, graph, expected_graph)

  def test_gf_graph_in_memory_with_filter(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "gf_graph")
      gen_test_graph.generate_gf_graph(path, edge_ids=True)

      graph, schema = gf_graph_in_memory.read_graph(
          path,
          schema_filter=schema_lib.GraphSchemaFilter(
              # Remove all the edges
              edgeset_fn=lambda key, sch: False
          ),
      )

      expected_schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=True, semantic=True
      )
      expected_schema.edge_sets = {}
      self.assertEqual(
          schema,
          expected_schema,
      )
      expected_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=True
      )
      expected_graph = in_memory_graph_lib.InMemoryGraph(
          node_sets=expected_graph.node_sets,
          edge_sets={},
      )
      test_util.assert_are_equal(self, graph, expected_graph)

  def test_gf_graph_in_memory_fail_on_dangeling_edge(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "gf_graph")
      gen_test_graph.generate_gf_graph(
          path, edge_ids=True, insert_dangling_edges=True
      )
      with self.assertRaisesRegex(
          ValueError, "Node ID 'missing' not found in nodeset 'n1'"
      ):
        _, _ = gf_graph_in_memory.read_graph(path)

  def test_gf_graph_in_memory_skip_dangeling_edge(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "gf_graph")
      gen_test_graph.generate_gf_graph(
          path, edge_ids=True, insert_dangling_edges=True
      )
      graph, schema = gf_graph_in_memory.read_graph(
          path, remove_dangling_edges=True
      )
      in_memory_graph_validate_lib.validate_graph(graph, schema)
      self.assertEqual(graph.edge_sets["e2"].num_edges(), 1)

  @parameterized.product(
      edge_ids=[True, False], container=["PARQUET", "TF_RECORD", "RECORDIO"]
  )
  def test_write_graph(self, edge_ids: bool, container: str):
    with tempfile.TemporaryDirectory() as tmpdir:

      # Generate a toy in-memory graph
      output_path = os.path.join(tmpdir, "output_gf_graph")
      in_memory_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=edge_ids
      )
      schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=edge_ids, semantic=True
      )

      # Write and read back the graph
      gf_graph_in_memory.write_graph(
          in_memory_graph, schema, output_path, container=container
      )
      output_in_memory_graph, output_schema = gf_graph_in_memory.read_graph(
          output_path
      )

      # Test equality
      test_util.assert_are_equal(self, output_schema, schema)
      test_util.assert_are_equal(self, output_in_memory_graph, in_memory_graph)

      # Check files

      extension = gf_graph_in_memory.get_extension(
          gf_metadata_lib.Container(container)
      )
      expected_files = [
          "/schema.json",
          "/metadata.json",
          f"/nodesets/n1-00000-of-00001{extension}",
          f"/nodesets/n2-00000-of-00001{extension}",
          f"/edgesets/e1-00000-of-00001{extension}",
          f"/edgesets/e2-00000-of-00001{extension}",
      ]
      actual_files = []
      for dirpath, _, filenames in os.walk(output_path):
        for filename in filenames:
          actual_files.append(
              os.path.join(dirpath, filename).removeprefix(output_path)
          )
      self.assertSameElements(sorted(actual_files), sorted(expected_files))

  def test_write_and_read_graph_timestamp(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = os.path.join(tmpdir, "timestamped_gf_graph")
      in_memory_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=True
      )
      schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=True, semantic=True
      )

      # 1. Write graph with timestamp set on InMemoryGraph object
      timestamped_graph = in_memory_graph_lib.InMemoryGraph(
          node_sets=in_memory_graph.node_sets,
          edge_sets=in_memory_graph.edge_sets,
          timestamp=123456,
      )
      gf_graph_in_memory.write_graph(timestamped_graph, schema, output_path)
      loaded_graph, _ = gf_graph_in_memory.read_graph(output_path)
      self.assertEqual(loaded_graph.timestamp, 123456)

      # 2. Write un-timestamped graph (graph.timestamp is None)
      output_path2 = os.path.join(tmpdir, "untimestamped_gf_graph")
      gf_graph_in_memory.write_graph(in_memory_graph, schema, output_path2)
      loaded_graph2, _ = gf_graph_in_memory.read_graph(output_path2)
      self.assertIsNone(loaded_graph2.timestamp)

  @parameterized.product(
      max_num_shards=[1, 2, 3],
      container=["PARQUET", "TF_RECORD", "RECORDIO"],
  )
  def test_write_and_read_sharded_graph(
      self, max_num_shards: int, container: str
  ):
    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = os.path.join(tmpdir, "sharded_gf_graph")
      num_nodes = 2500
      node_ids = np.array([f"n_{i}".encode("utf-8") for i in range(num_nodes)])
      node_features = {
          "#id": node_ids,
          "val": np.arange(num_nodes, dtype=np.int64),
      }
      edge_sources = np.arange(num_nodes, dtype=np.int64)
      edge_targets = (np.arange(num_nodes, dtype=np.int64) + 1) % num_nodes
      edge_ids = np.array([f"e_{i}".encode("utf-8") for i in range(num_nodes)])
      edge_features = {
          "#id": edge_ids,
          "weight": np.linspace(0.0, 1.0, num_nodes, dtype=np.float32),
      }
      graph = in_memory_graph_lib.InMemoryGraph(
          node_sets={
              "n1": in_memory_graph_lib.InMemoryNodeSet(
                  num_nodes=num_nodes, features=node_features
              )
          },
          edge_sets={
              "e1": in_memory_graph_lib.InMemoryEdgeSet(
                  adjacency=np.stack([edge_sources, edge_targets]),
                  features=edge_features,
              )
          },
      )
      schema = schema_lib.GraphSchema(
          node_sets={
              "n1": schema_lib.NodeSchema(
                  features={
                      "#id": schema_lib.FeatureSchema(
                          format=schema_lib.FeatureFormat.BYTES,
                          shape=(),
                          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                      ),
                      "val": schema_lib.FeatureSchema(
                          format=schema_lib.FeatureFormat.INTEGER_64,
                          shape=(),
                      ),
                  }
              )
          },
          edge_sets={
              "e1": schema_lib.EdgeSchema(
                  source="n1",
                  target="n1",
                  features={
                      "#id": schema_lib.FeatureSchema(
                          format=schema_lib.FeatureFormat.BYTES,
                          shape=(),
                          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                      ),
                      "weight": schema_lib.FeatureSchema(
                          format=schema_lib.FeatureFormat.FLOAT_32,
                          shape=(),
                      ),
                  },
              )
          },
      )

      gf_graph_in_memory.write_graph(
          graph,
          schema,
          output_path,
          max_num_shards=max_num_shards,
          container=container,
      )
      loaded_graph, loaded_schema = gf_graph_in_memory.read_graph(output_path)

      test_util.assert_are_equal(self, loaded_schema, schema)
      test_util.assert_are_equal(
          self,
          _canonicalize_graph(loaded_graph, schema),
          _canonicalize_graph(graph, schema),
      )

      extension = gf_graph_in_memory.get_extension(
          gf_metadata_lib.Container(container)
      )
      expected_num_shards = min(3, max_num_shards)
      expected_files = ["/schema.json", "/metadata.json"]
      for s in range(expected_num_shards):
        expected_files.append(
            f"/nodesets/n1-{s:05d}-of-{expected_num_shards:05d}{extension}"
        )
        expected_files.append(
            f"/edgesets/e1-{s:05d}-of-{expected_num_shards:05d}{extension}"
        )

      actual_files = []
      for dirpath, _, filenames in os.walk(output_path):
        for filename in filenames:
          actual_files.append(
              os.path.join(dirpath, filename).removeprefix(output_path)
          )
      self.assertSameElements(sorted(actual_files), sorted(expected_files))


def _canonicalize_graph(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
) -> in_memory_graph_lib.InMemoryGraph:
  new_node_sets = {}
  node_perm = {}
  for nodeset_name, nodeset in graph.node_sets.items():
    if (
        "#id" in nodeset.features
        and nodeset.num_nodes
        and nodeset.num_nodes > 1
    ):
      order = np.argsort(nodeset.features["#id"])
      inv_order = np.empty_like(order)
      inv_order[order] = np.arange(len(order))
      node_perm[nodeset_name] = inv_order
      new_features = {k: v[order] for k, v in nodeset.features.items()}
      new_node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
          num_nodes=nodeset.num_nodes, features=new_features
      )
    else:
      node_perm[nodeset_name] = np.arange(nodeset.num_nodes or 0)
      new_node_sets[nodeset_name] = nodeset

  new_edge_sets = {}
  for edgeset_name, edgeset in graph.edge_sets.items():
    edge_schema = schema.edge_sets[edgeset_name]
    src_perm = node_perm[edge_schema.source]
    dst_perm = node_perm[edge_schema.target]
    new_src = src_perm[edgeset.adjacency[0]]
    new_dst = dst_perm[edgeset.adjacency[1]]
    if "#id" in edgeset.features:
      edge_order = np.argsort(edgeset.features["#id"])
    else:
      edge_order = np.lexsort((new_dst, new_src))
    new_adjacency = np.stack([new_src[edge_order], new_dst[edge_order]])
    new_features = {k: v[edge_order] for k, v in edgeset.features.items()}
    new_edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=new_adjacency, features=new_features
    )

  return in_memory_graph_lib.InMemoryGraph(
      node_sets=new_node_sets,
      edge_sets=new_edge_sets,
      timestamp=graph.timestamp,
  )


if __name__ == "__main__":
  absltest.main()
