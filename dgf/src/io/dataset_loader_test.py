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

"""Unit tests for load_dataset.py."""

import logging
import os
import time
from typing import Any, Tuple
import unittest
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.analyse import print_schema
from dgf.src.io import dataset_loader
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np

test_util.disable_diff_truncation()


class LoadDatasetTest(parameterized.TestCase):

  def test_build_split_idx(self):
    num_nodes = 10
    ogb_splits = {
        "train": np.array([0, 1, 2]),
        "valid": np.array([3, 4, 5]),
        "test": np.array([6, 7, 8]),
    }
    splits = dataset_loader.build_split_idx(num_nodes, ogb_splits)
    expected_splits = np.array([
        b"train",
        b"train",
        b"train",
        b"valid",
        b"valid",
        b"valid",
        b"test",
        b"test",
        b"test",
        b"n/a",
    ])
    np.testing.assert_array_equal(splits, expected_splits)

  @parameterized.parameters(("arxiv",))
  @mock.patch.object(dataset_loader, "download_ogb_graph", autospec=True)
  def test_ogb(self, graph_name, mock_download_ogb_graph):

    # Mock the OGB downloader.
    def download_ogb_graph_mock(name: str) -> Tuple[Any, Any, Any]:
      del name
      label = np.random.randint(0, 10, size=(3, 1))
      ogb_graph = {
          "edge_index": np.array([[0, 0, 1], [1, 2, 2]]),
          "node_feat": np.random.rand(3, 128).astype(np.float32),
          "node_year": np.random.randint(0, 10, size=(3, 1)),
          "num_nodes": 3,
      }
      splits = {
          "train": np.array([0, 1]),
          "test": np.array([2]),
      }
      return ogb_graph, label, splits

    mock_download_ogb_graph.side_effect = download_ogb_graph_mock
    tmpdir = self.create_tempdir().full_path
    graph, schema = dataset_loader.fetch_ogb_graph(
        graph_name, cache_dir=tmpdir, repo="OGB"
    )
    logging.info(
        "Schema:\n%s", print_schema.print_schema(schema, return_output=True)
    )
    in_memory_graph_validate_lib.validate_graph(graph, schema)

  @unittest.skipIf(
      os.environ.get("TEST_STRATEGY") != "local",
      "Manual test that requires internet access and only runs on a"
      " workstation with --test_strategy=local",
  )
  @parameterized.parameters((
      "arxiv",
      "mag",
      "products",
  ))
  def test_real_ogb(self, graph_name):
    r"""Download a graph from the net and check it.

    This test actually download the graph from the net. Therefore, it can only
    be run manually on a workstation.

    Usage example:

    ```shell
    blaze test -c opt --test_strategy=local --test_output=streamed \
      --test_arg=--alsologtostderr \
      --test_filter=LoadDatasetTest.test_real_ogb \
      //third_party/py/dgf/src/io:dataset_loader_test
    ```
    """
    tmpdir = self.create_tempdir().full_path
    start_time = time.time()
    graph_1, schema_1 = dataset_loader.fetch_ogb_graph(
        graph_name, cache_dir=tmpdir, repo="OGB"
    )
    end_time = time.time()
    first_fetch_time = end_time - start_time
    logging.info("First fetching time: %s seconds", first_fetch_time)
    logging.info(
        "Schema:\n%s", print_schema.print_schema(schema_1, return_output=True)
    )
    in_memory_graph_validate_lib.validate_graph(graph_1, schema_1)
    start_time = time.time()
    graph_2, schema_2 = dataset_loader.fetch_ogb_graph(
        graph_name, cache_dir=tmpdir
    )
    end_time = time.time()
    second_fetch_time = end_time - start_time
    logging.info("Second fetching time: %s seconds", second_fetch_time)
    self.assertGreater(first_fetch_time, 2 * second_fetch_time)
    test_util.assert_are_equal(self, graph_1, graph_2)
    test_util.assert_are_equal(self, schema_1, schema_2)

  @parameterized.parameters(("tolokers-2",))
  @mock.patch.object(dataset_loader, "download_graphland_graph", autospec=True)
  def test_graphland_mocked(self, graph_name, mock_download_graphland_graph):

    def download_graphland_mock(
        name: str, mask_name: str, repo: dataset_loader.Repo
    ):
      del name
      del mask_name
      del repo
      num_nodes = 5
      edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
      features = {
          "feat_1": np.random.rand(num_nodes).astype(np.float32),
          "feat_2": (
              np.random.randint(0, 10, size=(num_nodes,)).astype(np.float32)
          ),
      }
      targets = np.random.randint(0, 2, size=(num_nodes,)).astype(np.int64)
      splits = np.array([b"train", b"train", b"valid", b"test", b"test"])
      info = {
          "task": "binary_classification",
          "target_name": "target",
          "num_classes": 2,
          "numerical_features_names": ["feat_1"],
          "categorical_features_names": ["feat_2"],
      }
      return edges, features, targets, splits, info

    mock_download_graphland_graph.side_effect = download_graphland_mock
    tmpdir = self.create_tempdir().full_path
    graph, schema = dataset_loader.fetch_graphland_graph(
        graph_name, cache_dir=tmpdir
    )

    logging.info(
        "Graphland Schema:\n%s",
        print_schema.print_schema(schema, return_output=True),
    )
    in_memory_graph_validate_lib.validate_graph(graph, schema)

  @unittest.skipIf(
      os.environ.get("TEST_STRATEGY") != "local",
      "Manual test that requires internet access and only runs on a"
      " workstation with --test_strategy=local",
  )
  @parameterized.parameters(
      ("pokec-regions",),
      ("tolokers-2",),
      ("hm-prices",),
      ("city-reviews",),
  )
  def test_real_graphland(self, graph_name):
    r"""Download a Graphland graph and check it.

    This test actually downloads the graph from Zenodo. Therefore, it can only
    be run manually on a workstation.

    Usage example:

    ```shell
    blaze test -c opt --test_strategy=local --test_output=streamed \
      --test_arg=--alsologtostderr \
      --test_filter=LoadDatasetTest.test_real_graphland \
      //third_party/py/dgf/src/io:dataset_loader_test
    ```
    """
    tmpdir = self.create_tempdir().full_path
    start_time = time.time()
    graph, schema = dataset_loader.fetch_graphland_graph(
        graph_name, cache_dir=tmpdir
    )
    end_time = time.time()
    fetch_time = end_time - start_time
    logging.info("Fetching time: %s seconds", fetch_time)
    logging.info(
        "Schema:\n%s", print_schema.print_schema(schema, return_output=True)
    )
    in_memory_graph_validate_lib.validate_graph(graph, schema)


if __name__ == "__main__":
  absltest.main()
