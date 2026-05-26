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

from typing import Tuple
from absl.testing import absltest
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
from dgf.src.validate import validate as validate_lib

Issue = validate_lib.Issue

test_util.disable_diff_truncation()


def good_graph() -> (
    Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
):
  graph = gen_test_graph.generate_in_memory_graph(
      node_ids=True, edge_ids=True, variable_length=True
  )
  schema = gen_test_graph.generate_schema(
      node_ids=True,
      edge_ids=True,
      variable_length=True,
      semantic=True,
  )
  return graph, schema


class InMemoryGraphTest(absltest.TestCase):

  def test_good(self):
    graph, schema = good_graph()
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(issues, [])

  def test_warning(self):
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids=False, edge_ids=False, variable_length=True
    )
    schema = gen_test_graph.generate_schema(
        node_ids=False,
        edge_ids=False,
        variable_length=True,
        semantic=True,
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.warning(
                "The nodeset 'n1' schema is missing the '#id' feature."
                " Giving a clearly defined #id column is recommanded. It is"
                " also required for non-string node IDs e.g. integer IDs."
            ),
            Issue.warning(
                "The nodeset 'n2' schema is missing the '#id' feature."
                " Giving a clearly defined #id column is recommanded. It is"
                " also required for non-string node IDs e.g. integer IDs."
            ),
        ],
    )

  def test_missing_nodeset(self):
    graph, schema = good_graph()
    del graph.node_sets["n1"]
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues, [Issue.error("The graph is missing the nodeset 'n1'")]
    )

  def test_missing_feature(self):
    graph, schema = good_graph()
    del graph.node_sets["n1"].features["f1"]
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues, [Issue.error("Missing feature 'f1' in nodeset 'n1'.")]
    )

  def test_wrong_feature_type(self):
    graph, schema = good_graph()
    schema.node_sets["n1"].features[
        "f1"
    ].format = schema_lib.FeatureFormat.INTEGER_32
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'f1' in nodeset 'n1' has dtype |S4 (i.e., <class"
                " 'numpy.bytes_'>), but the schema expects format"
                " <FeatureFormat.INTEGER_32: 'INTEGER_32'>, which corresponds"
                " to dtype <class 'numpy.int32'>"
            )
        ],
    )

  def test_wrong_shape(self):
    graph, schema = good_graph()
    schema.node_sets["n1"].features["f1"].shape = (2, 3, 4)
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'f1' in nodeset 'n1' has shape (2, 1), but the"
                " schema expects a shape compatible with (2, 3, 4) (i.e., one"
                " more dimension)."
            )
        ],
    )

  def test_non_existing_source(self):
    graph, schema = good_graph()
    schema.edge_sets["e1"].source = "non_existing"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The edgeset 'e1' refers to a source nodeset 'non_existing'"
                " which is not defined in the graph schema's node_sets."
            )
        ],
    )

  def test_non_existing_target(self):
    graph, schema = good_graph()
    schema.edge_sets["e1"].target = "non_existing"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The edgeset 'e1' refers to a target nodeset 'non_existing'"
                " which is not defined in the graph schema's node_sets."
            )
        ],
    )

  def test_out_bound_source(self):
    graph, schema = good_graph()
    graph.edge_sets["e1"].adjacency[:] += 1
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The edgeset 'e1' adjacency contains target node indices out of"
                " bounds. Expected indices to be in [0, 2), but found min: 1,"
                " max: 2."
            ),
        ],
    )


if __name__ == "__main__":
  absltest.main()
