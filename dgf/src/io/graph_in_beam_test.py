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
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_test_util
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np

test_util.disable_diff_truncation()
Edge = distributed_graph.Edge


class ReadGFGGraphTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_read_graph(self, edge_ids: bool):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "gf_graph")
      gen_test_graph.generate_gf_graph(path, edge_ids=edge_ids)

      with test_pipeline.TestPipeline() as root:
        graph = gf_graph_in_beam_lib.read_graph(root, path)
        _check_graph(self, graph, edge_ids=edge_ids)

  def test_read_graph_with_filter(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "gf_graph")
      gen_test_graph.generate_gf_graph(path, edge_ids=False)

      with test_pipeline.TestPipeline() as root:
        graph = gf_graph_in_beam_lib.read_graph(
            root,
            path,
            schema_filter=schema_lib.GraphSchemaFilter(
                # Remove all the edges
                edgeset_fn=lambda key, sch: False
            ),
        )
        _check_graph(self, graph, edge_ids=False, has_edges=False)

  @parameterized.parameters(True, False)
  def test_write_graph(self, edge_ids: bool):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      old_path = os.path.join(tmpdir, "old_gf_graph")
      new_path = os.path.join(tmpdir, "new_gf_graph")

      with test_pipeline.TestPipeline() as root:
        gen_test_graph.generate_gf_graph(old_path, edge_ids=edge_ids)
        graph = gf_graph_in_beam_lib.read_graph(root, old_path)
        gf_graph_in_beam_lib.write_graph(graph, new_path)

      with test_pipeline.TestPipeline() as root:
        reloaded_graph = gf_graph_in_beam_lib.read_graph(root, new_path)
        _check_graph(self, reloaded_graph, edge_ids=edge_ids)


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
                      "#id": np.array(1, dtype=np.int64),
                  },
              ),
              distributed_graph.Node(
                  id=2,
                  features={
                      "f3": np.array(5, dtype=np.int64),
                      "f4": np.array(11, dtype=np.int64),
                      "f5": np.array([12, 13, 14], dtype=np.int64),
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
