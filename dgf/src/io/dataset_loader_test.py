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
from dgf.src.util import log
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


def traffic_speeds_dataframe(num_times: int) -> pd.DataFrame:
  """Returns a dummy frame in the format of `download_traffic_speeds`.

  Args:
    num_times: Number of time steps. The recordings are sampled every five
      minutes, like the published ones.
  """
  # No recording is zero, since a zero encodes a missing recording.
  timestamps = 1330560000 + 300 * np.arange(num_times, dtype=np.int64)
  values = 1.0 + np.arange(num_times * 3, dtype=np.float32).reshape(num_times, 3)
  return pd.DataFrame(values, index=timestamps, columns=["s0", "s1", "s2"])


def traffic_distances_dataframe() -> pd.DataFrame:
  """Returns dummy distances where s0 and s1 are close and s2 is far away."""
  return pd.DataFrame({
      "from": ["s0", "s1", "s2", "s0", "s1", "s1"],
      "to": ["s0", "s1", "s2", "s1", "s0", "s2"],
      "cost": np.array(
          [0.0, 0.0, 0.0, 100.0, 100.0, 10000.0], dtype=np.float32
      ),
  })


def traffic_locations_dataframe() -> pd.DataFrame:
  """Returns the dummy coordinates of the three sensors."""
  return pd.DataFrame({
      "sensor_id": ["s0", "s1", "s2"],
      "latitude": np.array([34.0, 34.1, 34.2], dtype=np.float32),
      "longitude": np.array([-118.0, -118.1, -118.2], dtype=np.float32),
  })


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

  def test_download_traffic_speeds_from_file(self):
    path = os.path.join(self.create_tempdir().full_path, "speeds.csv")
    # The published csvs are indexed by a datetime string, and the sensor ids
    # are the header of the other columns.
    with open(path, "w") as speeds_file:
      speeds_file.write(",773869,767541\n")
      speeds_file.write("2012-03-01 00:00:00,64.0,65.0\n")
      speeds_file.write("2012-03-01 00:05:00,66.0,67.0\n")

    speeds = dataset_loader.download_traffic_speeds("metr_la", source=path)

    # Timestamps are converted to seconds and sensor ids to strings.
    np.testing.assert_array_equal(
        speeds.index.to_numpy(), [1330560000, 1330560300]
    )
    self.assertEqual(list(speeds.columns), ["773869", "767541"])
    self.assertEqual(speeds.to_numpy().dtype, np.float32)
    np.testing.assert_allclose(
        speeds.to_numpy(), [[64.0, 65.0], [66.0, 67.0]]
    )

  def test_download_traffic_speeds_unknown_dataset_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown traffic dataset"):
      dataset_loader.download_traffic_speeds("metr_sf")

  @parameterized.parameters(("metr_la",), ("pems_bay",))
  def test_download_traffic_sensor_graph_from_files(self, dataset):
    # METR-LA ships csvs with a header row, PEMS-BAY without.
    tmpdir = self.create_tempdir().full_path
    distances_path = os.path.join(tmpdir, "distances.csv")
    locations_path = os.path.join(tmpdir, "locations.csv")
    has_header = dataset_loader.traffic_dataset_spec(
        dataset
    ).distances_have_header
    with open(distances_path, "w") as distances_file:
      if has_header:
        distances_file.write("from,to,cost\n")
      distances_file.write("773869,767541,100.5\n")
    with open(locations_path, "w") as locations_file:
      if has_header:
        locations_file.write("index,sensor_id,latitude,longitude\n0,")
      locations_file.write("773869,34.15497,-118.31829\n")

    distances, locations = dataset_loader.download_traffic_sensor_graph(
        dataset,
        distances_source=distances_path,
        locations_source=locations_path,
    )

    self.assertEqual(list(distances.columns), ["from", "to", "cost"])
    self.assertEqual(distances["from"].tolist(), ["773869"])
    self.assertEqual(distances["to"].tolist(), ["767541"])
    np.testing.assert_allclose(distances["cost"].to_numpy(), [100.5])
    self.assertEqual(locations["sensor_id"].tolist(), ["773869"])
    np.testing.assert_allclose(
        locations["latitude"].to_numpy(), [34.15497], rtol=1e-5
    )

  def test_build_sensor_adjacency(self):
    # s0 and s1 are close to each other, s2 is far away from both.
    distances = traffic_distances_dataframe()

    sources, targets, weights = dataset_loader.build_sensor_adjacency(
        distances, ["s0", "s1", "s2"]
    )

    # The far away sensor falls below the threshold, and self loops are not
    # part of the sensor graph.
    np.testing.assert_array_equal(sources, [0, 1])
    np.testing.assert_array_equal(targets, [1, 0])
    self.assertTrue(np.all(weights > 0.99))

  def test_build_sensor_adjacency_without_known_pair_raises(self):
    distances = traffic_distances_dataframe()
    with self.assertRaisesRegex(ValueError, "None of the distances"):
      dataset_loader.build_sensor_adjacency(distances, ["other"])

  def test_build_traffic_graph(self):
    num_times = 20
    speeds = traffic_speeds_dataframe(num_times)
    # A missing recording, encoded as a zero, on the 6th time step of s1.
    speeds.iloc[5, 1] = 0.0

    graph, schema = dataset_loader.build_traffic_graph(
        speeds=speeds,
        distances=traffic_distances_dataframe(),
        locations=traffic_locations_dataframe(),
        forecast_horizon_seconds=600,  # 2 steps
        query_step=1,
    )

    in_memory_graph_validate_lib.validate_graph(graph, schema)

    # Sensor node assertions.
    self.assertEqual(graph.node_sets["sensors"].num_nodes, 3)
    sensor_features = graph.node_sets["sensors"].features
    sensor_schema = schema.node_sets["sensors"].features
    self.assertTrue(sensor_schema["time"].is_timeseries)
    self.assertTrue(sensor_schema["time"].is_creation_time)
    self.assertEqual(sensor_schema["time"].group, "speed_ts")
    self.assertTrue(sensor_schema["speed"].is_timeseries)
    self.assertEqual(sensor_schema["speed"].group, "speed_ts")
    self.assertFalse(sensor_schema["latitude"].is_timeseries)
    for sensor_index in range(3):
      self.assertLen(sensor_features["time"][sensor_index], num_times)
      np.testing.assert_allclose(
          sensor_features["speed"][sensor_index],
          speeds.to_numpy()[:, sensor_index],
      )
    np.testing.assert_allclose(
        sensor_features["latitude"], [34.0, 34.1, 34.2], rtol=1e-5
    )

    # Query node assertions. There is one query per sensor and per query time,
    # minus the single query whose target is missing.
    num_query_times = num_times - 2
    self.assertEqual(
        graph.node_sets["queries"].num_nodes, num_query_times * 3 - 1
    )
    query_features = graph.node_sets["queries"].features
    self.assertTrue(
        schema.node_sets["queries"].features["creation_time"].is_creation_time
    )

    # Queries are time major and the target is the speed two steps ahead.
    timestamps = speeds.index.to_numpy()
    np.testing.assert_array_equal(
        query_features["creation_time"][:3],
        [timestamps[0], timestamps[0], timestamps[0]],
    )
    np.testing.assert_allclose(
        query_features["speed"][:3], speeds.to_numpy()[2, :]
    )
    self.assertNotIn(
        dataset_loader.TRAFFIC_MISSING_SPEED, query_features["speed"]
    )

    # The split is chronological and 70/10/20 over the query times, and all the
    # queries of a time step share the same split.
    splits = query_features["#split"]
    self.assertCountEqual(np.unique(splits), [b"train", b"valid", b"test"])
    train_times = query_features["creation_time"][splits == b"train"]
    valid_times = query_features["creation_time"][splits == b"valid"]
    test_times = query_features["creation_time"][splits == b"test"]
    self.assertLen(np.unique(train_times), int(num_query_times * 0.7))
    self.assertLen(np.unique(valid_times), int(num_query_times * 0.1))
    self.assertLess(train_times.max(), valid_times.min())
    self.assertLess(valid_times.max(), test_times.min())

    # Edge assertions.
    num_queries = graph.node_sets["queries"].num_nodes
    assert num_queries is not None
    self.assertEqual(
        graph.edge_sets["query_to_sensor"].adjacency.shape, (2, num_queries)
    )
    self.assertEqual(
        graph.edge_sets["sensor_to_query"].adjacency.shape, (2, num_queries)
    )
    np.testing.assert_array_equal(
        graph.edge_sets["sensor_to_sensor"].adjacency, [[0, 1], [1, 0]]
    )
    self.assertIn("weight", graph.edge_sets["sensor_to_sensor"].features)
    self.assertIn("weight", schema.edge_sets["sensor_to_sensor"].features)

    # Each query is connected to the sensor holding its own recordings.
    query_indices, sensor_indices = graph.edge_sets[
        "query_to_sensor"
    ].adjacency
    np.testing.assert_array_equal(query_indices, np.arange(num_queries))
    np.testing.assert_array_equal(sensor_indices[:3], [0, 1, 2])

  def test_build_traffic_graph_drops_queries_across_a_gap(self):
    # PEMS-BAY is recorded in local time, so an hour is missing on the day of
    # the daylight saving time change. Here, the 900s recording is missing.
    speeds = traffic_speeds_dataframe(5)
    speeds = speeds.set_axis(
        np.array([0, 300, 600, 1200, 1500], dtype=np.int64), axis="index"
    )

    graph, schema = dataset_loader.build_traffic_graph(
        speeds=speeds,
        distances=traffic_distances_dataframe(),
        locations=traffic_locations_dataframe(),
        forecast_horizon_seconds=600,
        query_step=1,
    )

    in_memory_graph_validate_lib.validate_graph(graph, schema)
    # Only the queries at 0s and 600s have a recording exactly 600s later.
    query_features = graph.node_sets["queries"].features
    self.assertEqual(graph.node_sets["queries"].num_nodes, 2 * 3)
    np.testing.assert_array_equal(
        np.unique(query_features["creation_time"]), [0, 600]
    )
    np.testing.assert_allclose(
        query_features["speed"][:3], speeds.to_numpy()[2, :]
    )

  def test_build_traffic_graph_reports_the_dropped_queries(self):
    # The recording 600s after 600s is missing, so the query at 600s has no
    # target at all, and the target of the query at 300s is a missing recording.
    speeds = traffic_speeds_dataframe(5)
    speeds = speeds.set_axis(
        np.array([0, 300, 600, 1200, 1500], dtype=np.int64), axis="index"
    )
    speeds.iloc[3, :] = 0.0

    with log.capture_logs(log_info=True) as logs:
      dataset_loader.build_traffic_graph(
          speeds=speeds,
          distances=traffic_distances_dataframe(),
          locations=traffic_locations_dataframe(),
          forecast_horizon_seconds=900,
          query_step=1,
      )

    # Match on the reported counts rather than the full sentence, so that
    # rewording the log messages does not break the test.
    messages = "\n".join(message.text for message in logs)
    self.assertRegex(messages, r"Dropping 3 of the 5 query times")
    self.assertRegex(messages, r"Dropping 3 of the 6 queries")

  def test_build_traffic_graph_horizon_not_a_multiple_raises(self):
    with self.assertRaisesRegex(ValueError, "not a multiple"):
      dataset_loader.build_traffic_graph(
          speeds=traffic_speeds_dataframe(10),
          distances=traffic_distances_dataframe(),
          locations=traffic_locations_dataframe(),
          forecast_horizon_seconds=100,
      )

  def test_build_traffic_graph_missing_location_raises(self):
    locations = traffic_locations_dataframe().iloc[:2]
    with self.assertRaisesRegex(ValueError, "missing from the locations"):
      dataset_loader.build_traffic_graph(
          speeds=traffic_speeds_dataframe(10),
          distances=traffic_distances_dataframe(),
          locations=locations,
      )

  @mock.patch.object(
      dataset_loader, "download_traffic_sensor_graph", autospec=True
  )
  @mock.patch.object(dataset_loader, "download_traffic_speeds", autospec=True)
  def test_fetch_metr_la_graph_mocked(
      self, mock_download_speeds, mock_download_sensor_graph
  ):
    num_times = 20
    mock_download_speeds.return_value = traffic_speeds_dataframe(num_times)
    mock_download_sensor_graph.return_value = (
        traffic_distances_dataframe(),
        traffic_locations_dataframe(),
    )

    tmpdir = self.create_tempdir().full_path
    graph, schema = dataset_loader.fetch_metr_la_graph(
        cache_dir=tmpdir,
        forecast_horizon_seconds=600,
        query_step=1,
        repo="WEB",
    )

    in_memory_graph_validate_lib.validate_graph(graph, schema)
    self.assertEqual(graph.node_sets["sensors"].num_nodes, 3)
    self.assertEqual(graph.node_sets["queries"].num_nodes, (num_times - 2) * 3)
    self.assertEqual(mock_download_speeds.call_args.args[0], "metr_la")

    # The second call is served from the cache.
    graph_cached, _ = dataset_loader.fetch_metr_la_graph(
        cache_dir=tmpdir,
        forecast_horizon_seconds=600,
        query_step=1,
        repo="WEB",
    )
    self.assertEqual(mock_download_speeds.call_count, 1)
    self.assertEqual(
        graph_cached.node_sets["queries"].num_nodes, (num_times - 2) * 3
    )

  @mock.patch.object(
      dataset_loader, "download_traffic_sensor_graph", autospec=True
  )
  @mock.patch.object(dataset_loader, "download_traffic_speeds", autospec=True)
  def test_fetch_pems_bay_graph_mocked(
      self, mock_download_speeds, mock_download_sensor_graph
  ):
    mock_download_speeds.return_value = traffic_speeds_dataframe(20)
    mock_download_sensor_graph.return_value = (
        traffic_distances_dataframe(),
        traffic_locations_dataframe(),
    )

    graph, schema = dataset_loader.fetch_pems_bay_graph(
        cache_dir=None,
        forecast_horizon_seconds=600,
        query_step=1,
        repo="WEB",
    )

    in_memory_graph_validate_lib.validate_graph(graph, schema)
    self.assertEqual(mock_download_speeds.call_args.args[0], "pems_bay")

  @parameterized.named_parameters(
      ("metr_la", "metr_la", 207, 900),
      ("pems_bay", "pems_bay", 325, 3600),
  )
  @unittest.skipIf(
      os.environ.get("TEST_STRATEGY") != "local",
      "Manual test that requires internet access and only runs on a"
      " workstation with --test_strategy=local",
  )
  def test_real_traffic(self, dataset, expected_num_sensors, horizon_seconds):
    r"""Download METR-LA / PEMS-BAY and check them.

    Usage example:

    ```shell
    blaze test -c opt --test_strategy=local --test_output=streamed \
      --test_arg=--alsologtostderr \
      --test_filter=LoadDatasetTest.test_real_traffic \
      //third_party/py/dgf/src/io:dataset_loader_test
    ```

    Args:
      dataset: Name of the traffic dataset.
      expected_num_sensors: Number of loop detectors of the dataset.
      horizon_seconds: Forecast horizon to build the graph with.
    """
    tmpdir = self.create_tempdir().full_path
    graph, schema = dataset_loader.fetch_traffic_graph(
        dataset=dataset,
        name=dataset,
        cache_dir=tmpdir,
        forecast_horizon_seconds=horizon_seconds,
        repo="WEB",
    )
    in_memory_graph_validate_lib.validate_graph(graph, schema)
    logging.info(
        "Schema:\n%s", print_schema.print_schema(schema, return_output=True)
    )
    self.assertEqual(
        graph.node_sets["sensors"].num_nodes, expected_num_sensors
    )
    num_queries = graph.node_sets["queries"].num_nodes
    self.assertIsNotNone(num_queries)
    assert num_queries is not None
    self.assertGreater(num_queries, 1000)
    self.assertGreater(
        graph.edge_sets["sensor_to_sensor"].adjacency.shape[1],
        expected_num_sensors,
    )


if __name__ == "__main__":
  absltest.main()
