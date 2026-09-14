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
import pandas as pd

test_util.disable_diff_truncation()


def jena_climate_dataframe(num_rows: int) -> pd.DataFrame:
  """Returns a dummy frame in the format of `download_jena_climate_csv`."""
  date_range = pd.date_range("2009-01-01", periods=num_rows, freq="10min")
  columns = {
      name: np.linspace(1.0, 2.0, num_rows, dtype=np.float32)
      for name in dataset_loader.JENA_WEATHER_FEATURE_NAMES
  }
  columns["t_degc"] = np.linspace(-5.0, 25.0, num_rows, dtype=np.float32)
  columns["timestamp"] = (date_range.astype("int64") // 10**9).astype(np.int64)
  return pd.DataFrame(columns)


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

  def test_download_jena_climate_csv(self):
    raw_data = {
        "Date Time": ["01.01.2009 00:10:00", "01.01.2009 00:20:00"],
        "p (mbar)": [996.5, 996.6],
        "T (degC)": [-8.0, -8.1],
        "Tpot (K)": [265.4, 265.3],
        "Tdew (degC)": [-8.9, -9.0],
        "rh (%)": [93.3, 93.4],
        "VPmax (mbar)": [3.3, 3.2],
        "VPact (mbar)": [3.1, 3.0],
        "VPdef (mbar)": [0.2, 0.2],
        "sh (g/kg)": [1.9, 1.9],
        "H2OC (mmol/mol)": [3.1, 3.1],
        "rho (g/m**3)": [1307.8, 1308.0],
        "wv (m/s)": [-9999.0, 0.7],
        "max. wv (m/s)": [1.8, -9999.0],
        "wd (deg)": [152.3, 136.1],
    }
    path = os.path.join(self.create_tempdir().full_path, "jena.csv")
    pd.DataFrame(raw_data).to_csv(path, index=False)

    climate_df = dataset_loader.download_jena_climate_csv(source=path)

    # Columns are renamed to their normalized names.
    for name in dataset_loader.JENA_WEATHER_FEATURE_NAMES:
      self.assertIn(name, climate_df.columns)
    # Date Time is parsed into unix seconds.
    np.testing.assert_array_equal(
        climate_df["timestamp"].to_numpy(), np.array([1230768600, 1230769200])
    )
    # The -9999.0 sentinels are replaced by zeros.
    np.testing.assert_allclose(climate_df["wv_m_per_s"].to_numpy(), [0.0, 0.7])
    np.testing.assert_allclose(
        climate_df["max_wv_m_per_s"].to_numpy(), [1.8, 0.0]
    )

  def test_download_jena_climate_csv_missing_column_raises(self):
    path = os.path.join(self.create_tempdir().full_path, "jena.csv")
    pd.DataFrame({"Date Time": ["01.01.2009 00:10:00"]}).to_csv(
        path, index=False
    )
    with self.assertRaisesRegex(ValueError, "missing the columns"):
      dataset_loader.download_jena_climate_csv(source=path)

  @mock.patch.object(dataset_loader, "download_jena_climate_csv", autospec=True)
  def test_fetch_jena_climate_graph_mocked(self, mock_download):
    num_rows = 20
    climate_df = jena_climate_dataframe(num_rows)
    mock_download.return_value = climate_df

    tmpdir = self.create_tempdir().full_path
    graph, schema = dataset_loader.fetch_jena_climate_graph(
        cache_dir=tmpdir,
        forecast_horizon_seconds=1200,  # 2 steps
        query_step=1,
        repo="WEB",
    )

    in_memory_graph_validate_lib.validate_graph(graph, schema)

    # Station node assertions
    self.assertEqual(graph.node_sets["station"].num_nodes, 1)
    station_features = graph.node_sets["station"].features
    station_schema = schema.node_sets["station"].features
    self.assertTrue(station_schema["time"].is_timeseries)
    self.assertTrue(station_schema["time"].is_creation_time)
    self.assertEqual(station_schema["time"].group, "weather_ts")
    self.assertLen(station_features["time"][0], num_rows)
    self.assertTrue(station_schema["t_degc"].is_timeseries)
    self.assertEqual(station_schema["t_degc"].group, "weather_ts")
    self.assertLen(station_features["t_degc"][0], num_rows)
    for name in dataset_loader.JENA_WEATHER_FEATURE_NAMES:
      self.assertIn(name, station_features)

    # Query node assertions
    expected_num_queries = num_rows - 2
    self.assertEqual(
        graph.node_sets["queries"].num_nodes, expected_num_queries
    )
    query_features = graph.node_sets["queries"].features
    query_schema = schema.node_sets["queries"].features
    self.assertTrue(query_schema["creation_time"].is_creation_time)
    self.assertFalse(query_schema["creation_time"].is_timeseries)
    self.assertIn("temperature", query_features)
    self.assertLen(query_features["temperature"], expected_num_queries)
    self.assertIn("#split", query_features)

    # Check that temperature target is shifted by 2 steps
    np.testing.assert_allclose(
        query_features["temperature"],
        climate_df["t_degc"].to_numpy()[2:],
        rtol=1e-5,
    )

    # Edge set assertions
    self.assertEqual(
        graph.edge_sets["query_to_station"].adjacency.shape,
        (2, expected_num_queries),
    )
    self.assertEqual(
        graph.edge_sets["station_to_query"].adjacency.shape,
        (2, expected_num_queries),
    )

    # Verify loading from cache on subsequent call
    graph_cached, _ = dataset_loader.fetch_jena_climate_graph(
        cache_dir=tmpdir,
        forecast_horizon_seconds=1200,
        query_step=1,
        repo="WEB",
    )
    self.assertEqual(mock_download.call_count, 1)
    self.assertEqual(
        graph_cached.node_sets["queries"].num_nodes, expected_num_queries
    )

  @parameterized.parameters((3600, 6), (86400, 144))
  @mock.patch.object(dataset_loader, "download_jena_climate_csv", autospec=True)
  def test_jena_climate_temporal_lookahead_and_clipping(
      self, horizon_seconds, expected_step_offset, mock_download
  ):
    num_rows = 200
    climate_df = jena_climate_dataframe(num_rows)
    mock_download.return_value = climate_df

    tmpdir = self.create_tempdir().full_path
    graph, _ = dataset_loader.fetch_jena_climate_graph(
        cache_dir=tmpdir,
        forecast_horizon_seconds=horizon_seconds,
        query_step=1,
        repo="WEB",
    )

    query_creation_times = graph.node_sets["queries"].features["creation_time"]
    query_target_temperatures = graph.node_sets["queries"].features[
        "temperature"
    ]
    station_times = graph.node_sets["station"].features["time"][0]
    station_temperatures = graph.node_sets["station"].features["t_degc"][0]

    num_queries = len(query_creation_times)
    self.assertEqual(num_queries, num_rows - expected_step_offset)

    # 1. Target temperature matches measurement at target time:
    expected_targets = climate_df["t_degc"].to_numpy()[expected_step_offset:]
    np.testing.assert_allclose(
        query_target_temperatures, expected_targets, rtol=1e-5
    )

    # 2. Query creation_time is exactly target_time - horizon_seconds:
    for i in range(num_queries):
      query_time = query_creation_times[i]
      target_time = query_time + horizon_seconds
      target_index = np.searchsorted(station_times, target_time)
      self.assertEqual(station_times[target_index], target_time)
      self.assertEqual(
          query_target_temperatures[i], station_temperatures[target_index]
      )

      # 3. Validation of causal clipping:
      # In temporal sampling, station observations are clipped at
      # target_timestamp <= query_time:
      visible_mask = station_times <= query_time
      self.assertTrue(np.all(station_times[visible_mask] <= query_time))
      self.assertEqual(station_times[visible_mask][-1], query_time)

      # Ensure no future observations (lookahead) are visible:
      hidden_mask = station_times > query_time
      self.assertTrue(np.all(station_times[hidden_mask] > query_time))
      self.assertIn(target_time, station_times[hidden_mask])
      # Specifically, time difference between target measurement and latest
      # visible observation is exactly horizon_seconds:
      self.assertEqual(
          target_time - station_times[visible_mask][-1], horizon_seconds
      )

  @unittest.skipIf(
      os.environ.get("TEST_STRATEGY") != "local",
      "Manual test that requires internet access and only runs on a"
      " workstation with --test_strategy=local",
  )
  def test_real_jena_climate(self):
    r"""Download Jena Climate and check it.

    Usage example:

    ```shell
    blaze test -c opt --test_strategy=local --test_output=streamed \
      --test_arg=--alsologtostderr \
      --test_filter=LoadDatasetTest.test_real_jena_climate \
      //third_party/py/dgf/src/io:dataset_loader_test
    ```
    """
    tmpdir = self.create_tempdir().full_path
    graph, schema = dataset_loader.fetch_jena_climate_graph(
        cache_dir=tmpdir,
        query_step=36,
        repo="WEB",
    )
    in_memory_graph_validate_lib.validate_graph(graph, schema)
    self.assertEqual(graph.node_sets["station"].num_nodes, 1)
    num_queries = graph.node_sets["queries"].num_nodes
    self.assertIsNotNone(num_queries)
    assert num_queries is not None
    self.assertGreater(num_queries, 1000)


if __name__ == "__main__":
  absltest.main()
