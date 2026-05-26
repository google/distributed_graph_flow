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
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_memory as gf_graph_in_memory
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib

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

  @parameterized.product(edge_ids=[True, False])
  def test_write_graph(self, edge_ids: bool):
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
      gf_graph_in_memory.write_graph(in_memory_graph, schema, output_path)
      output_in_memory_graph, output_schema = gf_graph_in_memory.read_graph(
          output_path
      )

      # Test equality
      test_util.assert_are_equal(self, output_schema, schema)
      test_util.assert_are_equal(self, output_in_memory_graph, in_memory_graph)

      # Check files
      expected_files = [
          "/schema.json",
          "/metadata.json",
          "/nodesets/n1-00000-of-00001.parquet",
          "/nodesets/n2-00000-of-00001.parquet",
          "/edgesets/e1-00000-of-00001.parquet",
          "/edgesets/e2-00000-of-00001.parquet",
      ]
      actual_files = []
      for dirpath, _, filenames in os.walk(output_path):
        for filename in filenames:
          actual_files.append(
              os.path.join(dirpath, filename).removeprefix(output_path)
          )
      self.assertSameElements(sorted(actual_files), sorted(expected_files))


if __name__ == "__main__":
  absltest.main()
