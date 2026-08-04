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
from dgf.src.generate import gen_temporal_alert_graph
from dgf.src.io import graph_in_memory
from dgf.src.util import filesystem
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np


def _list_all_files(root_dir: str) -> list[str]:
  all_paths = filesystem.glob(os.path.join(root_dir, "*"))
  return [
      os.path.relpath(path, root_dir)
      for path in all_paths
      if not os.path.isdir(path)
  ]


class GenTemporalAlertGraphTest(absltest.TestCase):

  def test_generate_signal_regression_dataset(self):
    num_hw = 15
    num_alerts = 25
    schema = gen_temporal_alert_graph.generate_signal_regression_schema()
    config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
        num_hardware=num_hw, num_alerts=num_alerts, seed=42
    )
    graph = gen_temporal_alert_graph.generate_signal_regression_in_memory_graph(
        config=config, schema=schema
    )
    self.assertEqual(graph.node_sets["hardware"].num_nodes, num_hw)
    self.assertEqual(graph.node_sets["alerts"].num_nodes, num_alerts)
    self.assertEqual(
        graph.node_sets["alerts"].features["signal_regression"].dtype,
        np.float32,
    )
    creation_times = graph.node_sets["alerts"].features["creation_time"]
    self.assertTrue(np.all(creation_times >= 1700000000 + 3600))
    hw_creation_times = graph.node_sets["hardware"].features["creation_time"]
    self.assertEqual(len(hw_creation_times), num_hw)
    self.assertTrue(np.all(hw_creation_times == 1700000000))

    # Verify edge features
    edge_set = graph.edge_sets["hardware_to_alert"]
    self.assertIn("creation_time", edge_set.features)
    num_edges = edge_set.adjacency.shape[1]
    self.assertEqual(len(edge_set.features["creation_time"]), num_edges)
    for e_idx in range(num_edges):
      tgt_alert_idx = edge_set.adjacency[1, e_idx]
      self.assertEqual(
          edge_set.features["creation_time"][e_idx],
          creation_times[tgt_alert_idx],
      )

    # Verify schema annotations
    hw_features = schema.node_sets["hardware"].features
    self.assertTrue(hw_features["creation_time"].is_creation_time)
    self.assertFalse(hw_features["creation_time"].is_timeseries)
    self.assertTrue(hw_features["time"].is_creation_time)
    self.assertTrue(hw_features["time"].is_timeseries)
    self.assertEqual(hw_features["time"].group, "time")
    self.assertFalse(hw_features["signal"].is_creation_time)
    self.assertTrue(hw_features["signal"].is_timeseries)
    self.assertEqual(hw_features["signal"].group, "time")

    alert_features = schema.node_sets["alerts"].features
    self.assertTrue(alert_features["creation_time"].is_creation_time)
    self.assertFalse(alert_features["creation_time"].is_timeseries)

    edge_features = schema.edge_sets["hardware_to_alert"].features
    self.assertTrue(edge_features["creation_time"].is_creation_time)
    self.assertFalse(edge_features["creation_time"].is_timeseries)

  def test_generate_signal_regression_graph(self):
    work_dir = self.create_tempdir().full_path
    gen_temporal_alert_graph.generate_signal_regression_graph(work_dir)
    self.assertIn("metadata.json", _list_all_files(work_dir))
    self.assertIn("schema.json", _list_all_files(work_dir))

    # Read back the graph and validate it
    graph, schema = graph_in_memory.read_graph(work_dir)
    in_memory_graph_validate_lib.validate_graph(graph, schema)
    self.assertIn("creation_time", graph.node_sets["hardware"].features)
    self.assertIn("creation_time", graph.node_sets["alerts"].features)
    self.assertIn(
        "creation_time", graph.edge_sets["hardware_to_alert"].features
    )

  def test_generate_signal_regression_invalid_window(self):
    config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
        duration=1000, window_duration=2000
    )
    with self.assertRaisesRegex(
        ValueError,
        r"'window_duration' \(2000\) must be less than or equal to 'duration'"
        r" \(1000\)\.",
    ):
      gen_temporal_alert_graph.generate_signal_regression_in_memory_graph(
          config=config
      )

  def test_generate_signal_regression_invalid_neighbor_decay(self):
    config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
        neighbor_decay=1.5
    )
    with self.assertRaisesRegex(
        ValueError,
        r"'neighbor_decay' \(1\.5\) must be in \[0\.0, 1\.0\)\.",
    ):
      gen_temporal_alert_graph.generate_signal_regression_in_memory_graph(
          config=config
      )

  def test_generate_signal_regression_max_hardware_per_alert(self):
    num_hw = 50
    num_alerts = 100
    max_hw = 3
    config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
        num_hardware=num_hw,
        num_alerts=num_alerts,
        neighbor_decay=0.9,
        max_hardware_per_alert=max_hw,
        seed=42,
    )
    graph = gen_temporal_alert_graph.generate_signal_regression_in_memory_graph(
        config=config
    )
    edge_set = graph.edge_sets["hardware_to_alert"]
    target_alerts = edge_set.adjacency[1]
    alert_in_degrees = np.bincount(target_alerts, minlength=num_alerts)
    self.assertTrue(np.all(alert_in_degrees <= max_hw))
    self.assertTrue(np.all(alert_in_degrees >= 1))

  def test_generate_signal_regression_invalid_max_hardware_per_alert(self):
    config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
        max_hardware_per_alert=0
    )
    with self.assertRaisesRegex(
        ValueError,
        r"'max_hardware_per_alert' \(0\) must be positive\.",
    ):
      gen_temporal_alert_graph.generate_signal_regression_in_memory_graph(
          config=config
      )

  def test_generate_signal_regression_invalid_signal_period(self):
    config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
        signal_period=0
    )
    with self.assertRaisesRegex(
        ValueError,
        r"'signal_period' \(0\) must be positive\.",
    ):
      gen_temporal_alert_graph.generate_signal_regression_in_memory_graph(
          config=config
      )

  def test_generate_signal_regression_boundary_conditions(self):
    # Setting jitter=0 forces exact timestamp lookups (`hw_t[idx] == t_val`).
    # Setting large sample_interval_mean relative to duration forces boundary
    # clamping (`t_val <= hw_t[0]` and `t_val >= hw_t[-1]`).
    config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
        num_hardware=3,
        num_alerts=5,
        duration=100,
        window_duration=50,
        signal_period=50,
        sample_interval_mean=200,
        sample_interval_jitter=0,
    )
    graph = gen_temporal_alert_graph.generate_signal_regression_in_memory_graph(
        config=config
    )
    self.assertEqual(graph.node_sets["alerts"].num_nodes, 5)


if __name__ == "__main__":
  absltest.main()
