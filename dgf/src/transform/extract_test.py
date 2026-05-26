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

"""Test converting between heterogeneous graph edge types."""

import copy
from absl.testing import absltest
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.transform import extract as extract_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util


class FilterTest(absltest.TestCase):

  def test_filter_schema(self):
    schema = gen_test_graph.generate_schema()
    selected_features = ["f1", "f3"]
    extracted_schema = extract_lib.filter_schema(schema, selected_features)
    expected_extracted_schema = copy.deepcopy(schema)
    del expected_extracted_schema.node_sets["n1"].features["f2"]
    del expected_extracted_schema.node_sets["n2"].features["f4"]
    del expected_extracted_schema.node_sets["n2"].features["f5"]
    test_util.assert_are_equal(
        self,
        extracted_schema,
        expected_extracted_schema,
    )

  def test_filter_graph(self):
    graph = gen_test_graph.generate_in_memory_graph(variable_length=False)
    full_schema = gen_test_graph.generate_schema(variable_length=False)
    extracted_schema = extract_lib.filter_schema(full_schema, ["f1", "f3"])
    extracted_graph = extract_lib.filter_graph(graph, extracted_schema)
    expected_extracted_graph = copy.deepcopy(graph)
    del expected_extracted_graph.node_sets["n1"].features["f2"]
    del expected_extracted_graph.node_sets["n2"].features["f4"]
    test_util.assert_are_equal(
        self,
        extracted_graph,
        expected_extracted_graph,
    )

  def test_drop_edge_features(self):
    graph = gen_test_graph.generate_in_memory_graph(variable_length=False)
    schema = gen_test_graph.generate_schema(variable_length=False)

    extracted_graph, extracted_schema = extract_lib.drop_edge_features(
        graph, schema
    )

    expected_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets=graph.node_sets,
        edge_sets={
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=graph.edge_sets["e1"].adjacency, features={}
            ),
            "e2": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=graph.edge_sets["e2"].adjacency, features={}
            ),
        },
    )

    expected_schema = schema_lib.GraphSchema(
        node_sets=schema.node_sets,
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1", features={}),
            "e2": schema_lib.EdgeSchema(source="n1", target="n2", features={}),
        },
    )

    test_util.assert_are_equal(self, extracted_schema, expected_schema)
    test_util.assert_are_equal(self, extracted_graph, expected_graph)


if __name__ == "__main__":
  absltest.main()
