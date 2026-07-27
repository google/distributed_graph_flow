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
from dgf.src.util import filesystem
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
    self.assertTrue(
        schema.node_sets["hardware"].features["time"].is_creation_time
    )
    self.assertTrue(
        schema.node_sets["alerts"].features["creation_time"].is_creation_time
    )

  def test_generate_signal_regression_graph(self):
    work_dir = self.create_tempdir().full_path
    gen_temporal_alert_graph.generate_signal_regression_graph(work_dir)
    self.assertIn("metadata.json", _list_all_files(work_dir))
    self.assertIn("schema.json", _list_all_files(work_dir))

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

  def test_generate_signal_regression_boundary_conditions(self):
    # Setting jitter=0 forces exact timestamp lookups (`hw_t[idx] == t_val`).
    # Setting large sample_interval_mean relative to duration forces boundary
    # clamping (`t_val <= hw_t[0]` and `t_val >= hw_t[-1]`).
    config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
        num_hardware=3,
        num_alerts=5,
        duration=100,
        window_duration=50,
        sample_interval_mean=200,
        sample_interval_jitter=0,
    )
    graph = gen_temporal_alert_graph.generate_signal_regression_in_memory_graph(
        config=config
    )
    self.assertEqual(graph.node_sets["alerts"].num_nodes, 5)


if __name__ == "__main__":
  absltest.main()
