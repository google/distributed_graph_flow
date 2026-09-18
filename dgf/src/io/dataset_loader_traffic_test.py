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

"""Unit tests for dataset_loader_traffic.py."""

import logging
import os
import unittest
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.analyse import print_schema
from dgf.src.io import dataset_loader
from dgf.src.io import dataset_loader_traffic
from dgf.src.util import log
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np
import pandas as pd

test_util.disable_diff_truncation()


def traffic_speeds_dataframe(num_times: int) -> pd.DataFrame:
  """Returns a dummy frame in the format of `download_traffic_speeds`.

  Args:
    num_times: Number of time steps. The recordings are sampled every five
      minutes, like the published ones.
  """
  # No recording is zero, since a zero encodes a missing recording.
  timestamps = 1330560000 + 300 * np.arange(num_times, dtype=np.int64)
  values = 1.0 + np.arange(num_times * 3, dtype=np.float32).reshape(
      num_times, 3
  )
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


class LoadTrafficDatasetTest(parameterized.TestCase):

  def test_download_traffic_speeds_from_file(self):
    path = os.path.join(self.create_tempdir().full_path, "speeds.csv")
    # The published csvs are indexed by a datetime string, and the sensor ids
    # are the header of the other columns.
    with open(path, "w") as speeds_file:
      speeds_file.write(",773869,767541\n")
      speeds_file.write("2012-03-01 00:00:00,64.0,65.0\n")
      speeds_file.write("2012-03-01 00:05:00,66.0,67.0\n")

    speeds = dataset_loader_traffic.download_traffic_speeds(
        "metr_la", source=path
    )

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
      dataset_loader_traffic.download_traffic_speeds("metr_sf")

  @parameterized.parameters(("metr_la",), ("pems_bay",))
  def test_download_traffic_sensor_graph_from_files(self, dataset):
    # METR-LA ships csvs with a header row, PEMS-BAY without.
    tmpdir = self.create_tempdir().full_path
    distances_path = os.path.join(tmpdir, "distances.csv")
    locations_path = os.path.join(tmpdir, "locations.csv")
    has_header = dataset_loader_traffic.traffic_dataset_spec(
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

    distances, locations = dataset_loader_traffic.download_traffic_sensor_graph(
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

    sources, targets, weights = dataset_loader_traffic.build_sensor_adjacency(
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
      dataset_loader_traffic.build_sensor_adjacency(distances, ["other"])

  def test_build_traffic_graph(self):
    num_times = 20
    speeds = traffic_speeds_dataframe(num_times)
    # A missing recording, encoded as a zero, on the 6th time step of s1.
    speeds.iloc[5, 1] = 0.0

    graph, schema = dataset_loader_traffic.build_traffic_graph(
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
        dataset_loader_traffic.TRAFFIC_MISSING_SPEED, query_features["speed"]
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

    graph, schema = dataset_loader_traffic.build_traffic_graph(
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
      dataset_loader_traffic.build_traffic_graph(
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
      dataset_loader_traffic.build_traffic_graph(
          speeds=traffic_speeds_dataframe(10),
          distances=traffic_distances_dataframe(),
          locations=traffic_locations_dataframe(),
          forecast_horizon_seconds=100,
      )

  def test_build_traffic_graph_missing_location_raises(self):
    locations = traffic_locations_dataframe().iloc[:2]
    with self.assertRaisesRegex(ValueError, "missing from the locations"):
      dataset_loader_traffic.build_traffic_graph(
          speeds=traffic_speeds_dataframe(10),
          distances=traffic_distances_dataframe(),
          locations=locations,
      )

  @mock.patch.object(
      dataset_loader_traffic, "download_traffic_sensor_graph", autospec=True
  )
  @mock.patch.object(
      dataset_loader_traffic, "download_traffic_speeds", autospec=True
  )
  def test_fetch_traffic_graph_metr_la_mocked(
      self, mock_download_speeds, mock_download_sensor_graph
  ):
    num_times = 20
    mock_download_speeds.return_value = traffic_speeds_dataframe(num_times)
    mock_download_sensor_graph.return_value = (
        traffic_distances_dataframe(),
        traffic_locations_dataframe(),
    )

    tmpdir = self.create_tempdir().full_path
    graph, schema = dataset_loader_traffic.fetch_traffic_graph(
        dataset="metr_la",
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
    graph_cached, _ = dataset_loader_traffic.fetch_traffic_graph(
        dataset="metr_la",
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
      dataset_loader_traffic, "download_traffic_sensor_graph", autospec=True
  )
  @mock.patch.object(
      dataset_loader_traffic, "download_traffic_speeds", autospec=True
  )
  def test_fetch_traffic_graph_pems_bay_mocked(
      self, mock_download_speeds, mock_download_sensor_graph
  ):
    mock_download_speeds.return_value = traffic_speeds_dataframe(20)
    mock_download_sensor_graph.return_value = (
        traffic_distances_dataframe(),
        traffic_locations_dataframe(),
    )

    graph, schema = dataset_loader_traffic.fetch_traffic_graph(
        dataset="pems_bay",
        cache_dir=None,
        forecast_horizon_seconds=600,
        query_step=1,
        repo="WEB",
    )

    in_memory_graph_validate_lib.validate_graph(graph, schema)
    self.assertEqual(mock_download_speeds.call_args.args[0], "pems_bay")

  @parameterized.named_parameters(
      ("metr_la_15m", "metr_la", 900, "metr_la_15m"),
      ("metr_la_30m", "metr_la", 1800, "metr_la_30m"),
      ("metr_la_60m", "metr_la", 3600, "metr_la_60m"),
      ("pems_bay_15m", "pems_bay", 900, "pems_bay_15m"),
      ("pems_bay_30m", "pems_bay", 1800, "pems_bay_30m"),
      ("pems_bay_60m", "pems_bay", 3600, "pems_bay_60m"),
  )
  def test_traffic_cns_name(self, dataset, horizon, expected_name):
    self.assertEqual(
        dataset_loader_traffic.traffic_cns_name(dataset, horizon), expected_name
    )

  def test_traffic_cns_name_unknown_dataset_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown traffic dataset"):
      dataset_loader_traffic.traffic_cns_name("unknown_dataset", 900)

  def test_traffic_cns_name_uncached_horizon_raises(self):
    with self.assertRaisesRegex(ValueError, "not cached on CNS"):
      dataset_loader_traffic.traffic_cns_name("metr_la", 600)

  def test_fetch_traffic_graph_cns_uncached_horizon_raises(self):
    with self.assertRaisesRegex(ValueError, "not cached on CNS"):
      dataset_loader_traffic.fetch_traffic_graph(
          dataset="metr_la",
          forecast_horizon_seconds=600,
          repo="CNS",
      )

  @mock.patch.object(dataset_loader, "load_from_cns", autospec=True)
  def test_fetch_traffic_graph_cns_delegates_to_load_from_cns(
      self, mock_load_from_cns
  ):
    mock_load_from_cns.return_value = (mock.MagicMock(), mock.MagicMock())
    dataset_loader_traffic.fetch_traffic_graph(
        dataset="metr_la",
        forecast_horizon_seconds=900,
        repo="CNS",
        cache_dir=None,
    )
    mock_load_from_cns.assert_called_once_with(name="metr_la_15m")

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
      --test_filter=LoadTrafficDatasetTest.test_real_traffic \
      //third_party/py/dgf/src/io:dataset_loader_traffic_test
    ```

    Args:
      dataset: Name of the traffic dataset.
      expected_num_sensors: Number of loop detectors of the dataset.
      horizon_seconds: Forecast horizon to build the graph with.
    """
    tmpdir = self.create_tempdir().full_path
    graph, schema = dataset_loader_traffic.fetch_traffic_graph(
        dataset=dataset,
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
