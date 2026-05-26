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
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_test_util
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import hgraph_in_beam
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np

test_util.disable_diff_truncation()
Edge = distributed_graph.Edge


class ReadHGraphTest(absltest.TestCase):

  def test_read_graphai_hgraph(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "hgraph")
      gen_test_graph.generate_hgraph(path, node_id=True)

      with test_pipeline.TestPipeline() as p:
        hgraph = hgraph_in_beam.read_graphai_hgraph(p, path)

        expected_schema = gen_test_graph.generate_schema(
            node_ids=True, semantic=False
        )
        for nodeset_def in expected_schema.node_sets.values():
          nodeset_def.features["#id"].semantic = (
              schema_lib.FeatureSemantic.PRIMARY_ID
          )
        test_util.assert_are_equal(self, hgraph.schema, expected_schema)

        beam_test_util.assert_that(
            hgraph.node_sets["n1"],
            beam_test_util.equal_to(
                [
                    distributed_graph.Node(
                        id=b"1",
                        features={
                            "f1": np.array([b"blue"]),
                            "f2": np.array([0.0, 1.0], dtype=np.float32),
                        },
                    ),
                    distributed_graph.Node(
                        id=b"2",
                        features={
                            "f1": np.array([b"red"]),
                            "f2": np.array([2.0, 3.0], dtype=np.float32),
                        },
                    ),
                ],
                equals_fn=test_util.are_equal,
            ),
        )

        beam_test_util.assert_that(
            hgraph.node_sets["n2"],
            beam_test_util.equal_to(
                [
                    distributed_graph.Node(
                        id=1,
                        features={
                            "f3": np.array(4),
                            "f4": np.array(10),
                            "f5": np.array([11, 12]),
                        },
                    ),
                    distributed_graph.Node(
                        id=2,
                        features={
                            "f3": np.array(5),
                            "f4": np.array(11),
                            "f5": np.array([12, 13, 14]),
                        },
                    ),
                ],
                equals_fn=test_util.are_equal,
            ),
        )

        beam_test_util.assert_that(
            hgraph.edge_sets["e1"],
            beam_test_util.equal_to(
                [
                    Edge(id=b"a", source=b"1", target=b"1"),
                    Edge(id=b"b", source=b"1", target=b"2"),
                ],
                equals_fn=test_util.are_equal,
            ),
        )
        beam_test_util.assert_that(
            hgraph.edge_sets["e2"],
            beam_test_util.equal_to(
                [
                    Edge(source=b"1", target=1),
                    Edge(source=b"1", target=2),
                ],
                equals_fn=test_util.are_equal,
            ),
        )


class WriteHGraphTest(absltest.TestCase):

  def test_write_and_read(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      input_path = os.path.join(tmpdir, "input_hgraph")
      gen_test_graph.generate_hgraph(input_path, node_id=True)

      output_path = os.path.join(tmpdir, "output_hgraph")

      with test_pipeline.TestPipeline() as p_read:
        # Read the graph
        input_hgraph = hgraph_in_beam.read_graphai_hgraph(p_read, input_path)
        # Write the graph
        hgraph_in_beam.write_graphai_hgraph(input_hgraph, output_path)

      # Read the written graph
      with test_pipeline.TestPipeline() as p_write:
        output_hgraph = hgraph_in_beam.read_graphai_hgraph(p_write, output_path)

        expected_schema = gen_test_graph.generate_schema(
            node_ids=True, semantic=False
        )
        for nodeset_def in expected_schema.node_sets.values():
          nodeset_def.features["#id"].semantic = (
              schema_lib.FeatureSemantic.PRIMARY_ID
          )
        self.assertEqual(output_hgraph.schema, expected_schema)
        array = np.array

        beam_test_util.assert_that(
            output_hgraph.node_sets["n1"],
            beam_test_util.equal_to(
                [
                    distributed_graph.Node(
                        id=b"2",
                        features={
                            "f2": array([2.0, 3.0], dtype=np.float32),
                            "f1": array([b"red"]),
                        },
                    ),
                    distributed_graph.Node(
                        id=b"1",
                        features={
                            "f2": array([0.0, 1.0], dtype=np.float32),
                            "f1": array([b"blue"]),
                        },
                    ),
                ],
                equals_fn=test_util.are_equal,
            ),
        )
        beam_test_util.assert_that(
            output_hgraph.edge_sets["e1"],
            beam_test_util.equal_to(
                [
                    Edge(id=b"a", source=b"1", target=b"1"),
                    Edge(id=b"b", source=b"1", target=b"2"),
                ],
                equals_fn=test_util.are_equal,
            ),
        )
        beam_test_util.assert_that(
            output_hgraph.edge_sets["e2"],
            beam_test_util.equal_to(
                [
                    Edge(source=b"1", target=1),
                    Edge(source=b"1", target=2),
                ],
                equals_fn=test_util.are_equal,
            ),
        )


if __name__ == "__main__":
  absltest.main()
