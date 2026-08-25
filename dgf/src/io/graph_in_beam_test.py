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
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_test_util
from dgf.src.beam import runners
from dgf.src.data import distributed_graph
from dgf.src.data import gf_metadata as gf_metadata_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.io import graph_in_memory as gf_graph_in_memory
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np

test_util.disable_diff_truncation()
Edge = distributed_graph.Edge


def _create_pipeline():
  return test_pipeline.TestPipeline(
      runner=runners.runner_from_name("FlumePython")
  )


class ReadGFGGraphTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_read_graph(self, edge_ids: bool):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "gf_graph")
      gen_test_graph.generate_gf_graph(path, edge_ids=edge_ids)

      with _create_pipeline() as root:
        graph = gf_graph_in_beam_lib.read_graph(root, path)
        _check_graph(self, graph, edge_ids=edge_ids)

  @parameterized.parameters("PARQUET", "TF_RECORD", "RECORDIO")
  def test_read_graph_with_filter(self, container: str):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "gf_graph")
      in_memory_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=False
      )
      schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=False, semantic=True
      )
      gf_graph_in_memory.write_graph(
          in_memory_graph, schema, path, container=container
      )

      with _create_pipeline() as root:
        graph = gf_graph_in_beam_lib.read_graph(
            root,
            path,
            schema_filter=schema_lib.GraphSchemaFilter(
                # Remove all the edges
                edgeset_fn=lambda key, sch: False
            ),
        )
        _check_graph(self, graph, edge_ids=False, has_edges=False)

  @parameterized.product(
      edge_ids=[True, False], container=["PARQUET", "TF_RECORD", "RECORDIO"]
  )
  def test_write_graph(self, edge_ids: bool, container: str):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      old_path = os.path.join(tmpdir, "old_gf_graph")
      new_path = os.path.join(tmpdir, "new_gf_graph")

      with _create_pipeline() as root:
        gen_test_graph.generate_gf_graph(old_path, edge_ids=edge_ids)
        graph = gf_graph_in_beam_lib.read_graph(root, old_path)
        gf_graph_in_beam_lib.write_graph(
            graph,
            new_path,
            num_node_shards=1,
            num_edge_shards=1,
            container_type=container,
        )

      with _create_pipeline() as root:
        reloaded_graph = gf_graph_in_beam_lib.read_graph(root, new_path)
        _check_graph(self, reloaded_graph, edge_ids=edge_ids)

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
      for dirpath, _, filenames in os.walk(new_path):
        for filename in filenames:
          actual_files.append(
              os.path.join(dirpath, filename).removeprefix(new_path)
          )
      self.assertSameElements(sorted(actual_files), sorted(expected_files))

  @parameterized.product(
      edge_ids=[True, False], container=["PARQUET", "TF_RECORD", "RECORDIO"]
  )
  def test_in_memory_write_and_beam_read(self, edge_ids: bool, container: str):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "gf_graph")
      in_memory_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=edge_ids
      )
      schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=edge_ids, semantic=True
      )
      gf_graph_in_memory.write_graph(
          in_memory_graph, schema, path, container=container
      )

      with _create_pipeline() as root:
        graph = gf_graph_in_beam_lib.read_graph(root, path)
        _check_graph(self, graph, edge_ids=edge_ids)

  @parameterized.product(
      edge_ids=[True, False], container=["PARQUET", "TF_RECORD", "RECORDIO"]
  )
  def test_beam_write_and_in_memory_read(self, edge_ids: bool, container: str):
    with tempfile.TemporaryDirectory() as tmpdir:
      old_path = os.path.join(tmpdir, "old_gf_graph")
      new_path = os.path.join(tmpdir, "new_gf_graph")

      with _create_pipeline() as root:
        gen_test_graph.generate_gf_graph(old_path, edge_ids=edge_ids)
        graph = gf_graph_in_beam_lib.read_graph(root, old_path)
        gf_graph_in_beam_lib.write_graph(
            graph,
            new_path,
            num_node_shards=1,
            num_edge_shards=1,
            container_type=container,
        )

      output_in_memory_graph, output_schema = gf_graph_in_memory.read_graph(
          new_path
      )
      expected_schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=edge_ids, semantic=True
      )
      expected_in_memory_graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=edge_ids
      )
      test_util.assert_are_equal(self, output_schema, expected_schema)
      test_util.assert_are_equal(
          self,
          _canonicalize_graph(output_in_memory_graph, expected_schema),
          _canonicalize_graph(expected_in_memory_graph, expected_schema),
      )


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


def _check_graph(self, graph, edge_ids: bool, has_edges: bool = True):
  expected_schema = gen_test_graph.generate_schema(
      node_ids=True, edge_ids=edge_ids, semantic=True
  )
  if not has_edges:
    expected_schema.edge_sets = {}
  self.assertEqual(
      graph.schema,
      expected_schema,
  )

  beam_test_util.assert_that(
      graph.node_sets["n1"],
      beam_test_util.equal_to(
          [
              distributed_graph.Node(
                  id=b"1",
                  features={
                      "f2": np.array([0.0, 1.0], dtype=np.float32),
                      "f1": np.array([b"blue"]),
                      "#id": np.array(b"1"),
                  },
              ),
              distributed_graph.Node(
                  id=b"2",
                  features={
                      "f2": np.array([2.0, 3.0], dtype=np.float32),
                      "f1": np.array([b"red"]),
                      "#id": np.array(b"2"),
                  },
              ),
          ],
          equals_fn=test_util.are_equal,
      ),
  )
  beam_test_util.assert_that(
      graph.node_sets["n2"],
      beam_test_util.equal_to(
          [
              distributed_graph.Node(
                  id=1,
                  features={
                      "f3": np.array(4, dtype=np.int64),
                      "f4": np.array(10, dtype=np.int64),
                      "f5": np.array([11, 12], dtype=np.int64),
                      "f6": np.array([[11, 12], [13, 14]], dtype=np.int64),
                      "#id": np.array(1, dtype=np.int64),
                  },
              ),
              distributed_graph.Node(
                  id=2,
                  features={
                      "f3": np.array(5, dtype=np.int64),
                      "f4": np.array(11, dtype=np.int64),
                      "f5": np.array([12, 13, 14], dtype=np.int64),
                      "f6": np.array(
                          [[15, 16], [17, 18], [19, 20]], dtype=np.int64
                      ),
                      "#id": np.array(2, dtype=np.int64),
                  },
              ),
          ],
          equals_fn=test_util.are_equal,
      ),
  )

  if has_edges:
    e1_features_1 = {"#id": np.array(b"a")} if edge_ids else None
    e1_features_2 = {"#id": np.array(b"b")} if edge_ids else None
    beam_test_util.assert_that(
        graph.edge_sets["e1"],
        beam_test_util.equal_to(
            [
                Edge(
                    id=b"a" if edge_ids else None,
                    source=b"1",
                    target=b"1",
                    features=e1_features_1,
                ),
                Edge(
                    id=b"b" if edge_ids else None,
                    source=b"1",
                    target=b"2",
                    features=e1_features_2,
                ),
            ],
            equals_fn=test_util.are_equal,
        ),
    )
    e2_features_1 = {"#id": np.array(b"A")} if edge_ids else None
    e2_features_2 = {"#id": np.array(b"B")} if edge_ids else None
    beam_test_util.assert_that(
        graph.edge_sets["e2"],
        beam_test_util.equal_to(
            [
                Edge(
                    id=b"A" if edge_ids else None,
                    source=b"1",
                    target=1,
                    features=e2_features_1,
                ),
                Edge(
                    id=b"B" if edge_ids else None,
                    source=b"1",
                    target=2,
                    features=e2_features_2,
                ),
            ],
            equals_fn=test_util.are_equal,
        ),
    )
  else:
    self.assertEmpty(graph.edge_sets)


if __name__ == "__main__":
  absltest.main()
