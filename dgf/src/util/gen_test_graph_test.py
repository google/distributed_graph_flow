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
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np


def _list_all_files(root_dir: str) -> list[str]:
  all_files = []
  for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
      # Get the relative path from the root directory
      rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
      all_files.append(rel_path)
  return all_files


class GenHGraphTest(parameterized.TestCase):

  def test_generate_hgraph(self):
    work_dir = self.create_tempdir().full_path
    gen_test_graph.generate_hgraph(work_dir)
    self.assertSameElements(
        _list_all_files(work_dir),
        [
            "graph_schema.pbtxt",
            "node_features/n1-00000-of-00002.tfrecord.gz",
            "node_features/n1-00001-of-00002.tfrecord.gz",
            "node_features/n2-00000-of-00001.tfrecord.gz",
            "edges/e1-00000-of-00001.tfrecord.gz",
            "edges/e2-00000-of-00001.tfrecord.gz",
        ],
    )

  @parameterized.parameters(True, False)
  def test_generate_gf_graph(self, edge_ids):
    # Note: Actual content is tested in "gf_graph_in_beam_test.py".
    work_dir = self.create_tempdir().full_path
    gen_test_graph.generate_gf_graph(work_dir, edge_ids=edge_ids)
    self.assertSameElements(
        _list_all_files(work_dir),
        [
            "metadata.json",
            "schema.json",
            "nodesets/n1-00000-of-00001.parquet",
            "nodesets/n2-00000-of-00002.parquet",
            "nodesets/n2-00001-of-00002.parquet",
            "edgesets/e1-00000-of-00001.parquet",
            "edgesets/e2-00000-of-00001.parquet",
        ],
    )

  @parameterized.product(
      node_ids=[True, False],
      edge_ids=[True, False],
      variable_length=[True, False],
  )
  def test_generate_in_memory_graph_combinations(
      self, node_ids, edge_ids, variable_length
  ):
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids=node_ids,
        edge_ids=edge_ids,
        variable_length=variable_length,
    )
    self.assertIsInstance(graph, in_memory_graph_lib.InMemoryGraph)

    # Check node_ids
    for node_set_name, node_set in graph.node_sets.items():
      has_id = "#id" in node_set.features
      self.assertEqual(
          has_id,
          node_ids,
          f"Node set '{node_set_name}' feature '#id' presence mismatch."
          f" Expected: {node_ids}, Got: {has_id}",
      )

    # Check edge_ids
    for edge_set_name, edge_set in graph.edge_sets.items():
      has_id = "#id" in edge_set.features
      self.assertEqual(
          has_id,
          edge_ids,
          f"Edge set '{edge_set_name}' feature '#id' presence mismatch."
          f" Expected: {edge_ids}, Got: {has_id}",
      )

    # Check variable_length feature 'f5' in node set 'n2'
    n2_has_f5 = "f5" in graph.node_sets["n2"].features
    self.assertEqual(
        n2_has_f5,
        variable_length,
        "Node set 'n2' feature 'f5' presence mismatch."
        f" Expected: {variable_length}, Got: {n2_has_f5}",
    )
    if variable_length:
      self.assertEqual(graph.node_sets["n2"].features["f5"].dtype, object)
      self.assertTrue(
          test_util.are_equal(
              graph.node_sets["n2"].features["f5"],
              np.array(
                  [np.array([11, 12]), np.array([12, 13, 14])], dtype=object
              ),
          )
      )

    # Basic structure checks
    self.assertIn("n1", graph.node_sets)
    self.assertIn("n2", graph.node_sets)
    self.assertIn("e1", graph.edge_sets)
    self.assertIn("e2", graph.edge_sets)
    self.assertIsInstance(graph.node_sets["n1"].features["f1"], np.ndarray)


def test_generate_in_memory_graph_default_values(self):
  """Tests generate_in_memory_graph with default values."""
  graph = gen_test_graph.generate_in_memory_graph()
  self.assertIsInstance(graph, in_memory_graph_lib.InMemoryGraph)
  # Default is node_ids=False
  self.assertNotIn("#id", graph.node_sets["n1"].features)
  # Default is edge_ids=False
  self.assertNotIn("#id", graph.edge_sets["e1"].features)
  # Default is variable_length=True
  self.assertIn("f5", graph.node_sets["n2"].features)


class GenToyDatasetTest(absltest.TestCase):

  def test_generate_toy_dataset(self):
    num_n1_nodes = 20
    num_n2_nodes = 50
    graph, schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=num_n1_nodes, num_n2_nodes=num_n2_nodes, random_seed=42
    )

    in_memory_graph_validate_lib.validate_graph(graph, schema)

    labels = graph.node_sets["N1"].features["label"]
    self.assertAlmostEqual(np.mean(labels), 0.4, delta=0.1)

  def test_generate_toy_regression_dataset(self):
    num_n1_nodes = 20
    num_n2_nodes = 50
    graph, schema = gen_test_graph.gen_toy_regression_dataset(
        num_n1_nodes=num_n1_nodes,
        num_n2_nodes=num_n2_nodes,
        random_seed=42,
        label_dtype="float32",
    )

    in_memory_graph_validate_lib.validate_graph(graph, schema)

    labels = graph.node_sets["N1"].features["label"]
    self.assertEqual(labels.dtype, np.float32)


if __name__ == "__main__":
  absltest.main()
