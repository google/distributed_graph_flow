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

"""Tests for table_2_graph."""

from absl.testing import absltest
from dgf.src.data import schema as schema_lib
from dgf.src.transform import table_2_graph as table_2_graph_lib
import numpy as np
import pandas as pd


class Table2GraphTest(absltest.TestCase):

  def test_dict_input(self):
    table = {
        "feature_a": np.array([1, 2, 3]),
        "feature_b": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    }
    graph, schema = table_2_graph_lib.table2graph(
        table, nodeset_name="my_nodes"
    )

    # Assert Graph
    self.assertIn("my_nodes", graph.node_sets)
    self.assertEqual(graph.node_sets["my_nodes"].num_nodes, 3)
    self.assertEmpty(graph.edge_sets)

    np.testing.assert_array_equal(
        graph.node_sets["my_nodes"].features["feature_a"], table["feature_a"]
    )
    np.testing.assert_array_equal(
        graph.node_sets["my_nodes"].features["feature_b"], table["feature_b"]
    )

    # Assert Schema
    self.assertIn("my_nodes", schema.node_sets)
    self.assertEmpty(schema.edge_sets)

    nodeset_schema = schema.node_sets["my_nodes"]
    self.assertIn("feature_a", nodeset_schema.features)
    self.assertIn("feature_b", nodeset_schema.features)

    self.assertEqual(
        nodeset_schema.features["feature_a"].format,
        schema_lib.FeatureFormat.INTEGER_64,
    )
    self.assertEqual(nodeset_schema.features["feature_a"].shape, ())
    self.assertEqual(
        nodeset_schema.features["feature_a"].semantic,
        schema_lib.FeatureSemantic.NUMERICAL,
    )

    self.assertEqual(
        nodeset_schema.features["feature_b"].format,
        schema_lib.FeatureFormat.FLOAT_64,
    )
    self.assertEqual(nodeset_schema.features["feature_b"].shape, (2,))
    self.assertEqual(
        nodeset_schema.features["feature_b"].semantic,
        schema_lib.FeatureSemantic.EMBEDDING,
    )

  def test_dataframe_input(self):
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": [4.0, 5.0, 6.0],
    })
    graph, schema = table_2_graph_lib.table2graph(df)

    # Assert Graph
    self.assertIn("nodes", graph.node_sets)
    self.assertEqual(graph.node_sets["nodes"].num_nodes, 3)
    self.assertEmpty(graph.edge_sets)

    np.testing.assert_array_equal(
        graph.node_sets["nodes"].features["feature_a"],
        df["feature_a"].to_numpy(),
    )
    np.testing.assert_array_equal(
        graph.node_sets["nodes"].features["feature_b"],
        df["feature_b"].to_numpy(),
    )

    # Assert Schema
    self.assertIn("nodes", schema.node_sets)
    nodeset_schema = schema.node_sets["nodes"]
    self.assertEqual(
        nodeset_schema.features["feature_a"].format,
        schema_lib.FeatureFormat.INTEGER_64,
    )
    self.assertEqual(
        nodeset_schema.features["feature_a"].semantic,
        schema_lib.FeatureSemantic.NUMERICAL,
    )
    self.assertEqual(
        nodeset_schema.features["feature_b"].format,
        schema_lib.FeatureFormat.FLOAT_64,
    )
    self.assertEqual(
        nodeset_schema.features["feature_b"].semantic,
        schema_lib.FeatureSemantic.NUMERICAL,
    )

  def test_detect_semantic_false(self):
    table = {
        "feature_a": np.array([1, 2, 3]),
        "feature_b": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    }
    _, schema = table_2_graph_lib.table2graph(table, detect_semantic=False)
    nodeset_schema = schema.node_sets["nodes"]
    self.assertEqual(
        nodeset_schema.features["feature_a"].semantic,
        schema_lib.FeatureSemantic.UNKNOWN,
    )
    self.assertEqual(
        nodeset_schema.features["feature_b"].semantic,
        schema_lib.FeatureSemantic.UNKNOWN,
    )

  def test_bytes_semantic(self):
    table = {
        "feature_bytes": np.array([b"a", b"b", b"c"]),
    }
    _, schema = table_2_graph_lib.table2graph(table)
    nodeset_schema = schema.node_sets["nodes"]
    self.assertEqual(
        nodeset_schema.features["feature_bytes"].format,
        schema_lib.FeatureFormat.BYTES,
    )
    self.assertEqual(
        nodeset_schema.features["feature_bytes"].semantic,
        schema_lib.FeatureSemantic.CATEGORICAL,
    )

  def test_primary_id_semantic(self):
    table = {
        "id": np.array([1, 2, 3]),
        "#id": np.array([4, 5, 6]),
        "ID": np.array([7, 8, 9]),
        "other_id": np.array([10, 11, 12]),
    }
    _, schema = table_2_graph_lib.table2graph(table)
    nodeset_schema = schema.node_sets["nodes"]

    self.assertEqual(
        nodeset_schema.features["id"].semantic,
        schema_lib.FeatureSemantic.PRIMARY_ID,
    )
    self.assertEqual(
        nodeset_schema.features["#id"].semantic,
        schema_lib.FeatureSemantic.PRIMARY_ID,
    )
    self.assertEqual(
        nodeset_schema.features["ID"].semantic,
        schema_lib.FeatureSemantic.PRIMARY_ID,
    )
    self.assertEqual(
        nodeset_schema.features["other_id"].semantic,
        schema_lib.FeatureSemantic.NUMERICAL,
    )

  def test_invalid_input_type(self):
    with self.assertRaises(TypeError):
      table_2_graph_lib.table2graph([1, 2, 3])  # pytype: disable=wrong-arg-types

  def test_empty_input(self):
    with self.assertRaises(ValueError):
      table_2_graph_lib.table2graph({})

    with self.assertRaises(ValueError):
      table_2_graph_lib.table2graph(pd.DataFrame())

  def test_mismatched_lengths(self):
    table = {
        "feature_a": np.array([1, 2, 3]),
        "feature_b": np.array([4, 5]),
    }
    with self.assertRaisesRegex(
        ValueError, "All columns must have the same length"
    ):
      table_2_graph_lib.table2graph(table)


if __name__ == "__main__":
  absltest.main()
