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

from absl.testing import absltest
from dgf.src.learning import early_stopping_monitor


class EarlyStoppingMonitorTest(absltest.TestCase):

  def test_improving_loss(self):
    config = early_stopping_monitor.EarlyStoppingMonitorConfig(patience=2)
    monitor = early_stopping_monitor.EarlyStoppingMonitor(config)

    monitor.add_loss(1, 1.0)
    self.assertFalse(monitor.should_stop())

    monitor.add_loss(2, 0.9)
    self.assertFalse(monitor.should_stop())

    monitor.add_loss(3, 0.8)
    self.assertFalse(monitor.should_stop())

  def test_early_stopping(self):
    config = early_stopping_monitor.EarlyStoppingMonitorConfig(patience=2)
    monitor = early_stopping_monitor.EarlyStoppingMonitor(config)

    monitor.add_loss(1, 1.0)
    self.assertFalse(monitor.should_stop())

    # No improvement
    monitor.add_loss(2, 1.0)
    self.assertFalse(monitor.should_stop())

    # Still no improvement -> Patience is 2 which triggers should_stop
    monitor.add_loss(3, 1.0)
    self.assertTrue(monitor.should_stop())

  def test_restore_best_params(self):
    config = early_stopping_monitor.EarlyStoppingMonitorConfig(patience=2)
    monitor = early_stopping_monitor.EarlyStoppingMonitor(config)

    monitor.add_loss(1, 1.0, params={"w": 1})
    self.assertEqual(monitor.best_params, {"w": 1})
    self.assertEqual(monitor.best_loss, 1.0)
    self.assertEqual(monitor.best_step, 1)

    # Not improving enough
    monitor.add_loss(2, 1.1, params={"w": 2})
    self.assertEqual(monitor.best_params, {"w": 1})
    self.assertEqual(monitor.best_loss, 1.0)
    self.assertEqual(monitor.best_step, 1)

    # Improving
    monitor.add_loss(3, 0.9, params={"w": 3})
    self.assertEqual(monitor.best_params, {"w": 3})
    self.assertEqual(monitor.best_loss, 0.9)
    self.assertEqual(monitor.best_step, 3)

  def test_min_improvement(self):
    config = early_stopping_monitor.EarlyStoppingMonitorConfig(
        patience=1, min_improvement=0.1
    )
    monitor = early_stopping_monitor.EarlyStoppingMonitor(config)

    monitor.add_loss(1, 1.0)

    # Improvement is 0.05 < 0.1, so it shouldn't count as an improvement.
    monitor.add_loss(2, 0.95)
    self.assertTrue(monitor.should_stop())

  def test_normalize_early_stopping_config(self):
    self.assertIsNone(
        early_stopping_monitor.normalize_early_stopping_config(False)
    )

    config_true = early_stopping_monitor.normalize_early_stopping_config(True)
    self.assertIsInstance(
        config_true, early_stopping_monitor.EarlyStoppingMonitorConfig
    )
    self.assertEqual(config_true.patience, 5)

    config_int = early_stopping_monitor.normalize_early_stopping_config(10)
    self.assertIsInstance(
        config_int, early_stopping_monitor.EarlyStoppingMonitorConfig
    )
    self.assertEqual(config_int.patience, 10)

  def test_get_set_state(self):
    config = early_stopping_monitor.EarlyStoppingMonitorConfig(patience=3)
    monitor1 = early_stopping_monitor.EarlyStoppingMonitor(config)
    monitor1.add_loss(1, 1.0, params={"w": 10})
    monitor1.add_loss(2, 1.05, params={"w": 11})  # patience counter = 1

    state = monitor1.get_state()
    self.assertEqual(state["patience_counter"], 1)
    self.assertFalse(state["should_stop"])
    self.assertEqual(state["best_loss"], 1.0)
    self.assertEqual(state["best_params"], {"w": 10})
    self.assertEqual(state["best_step"], 1)

    monitor2 = early_stopping_monitor.EarlyStoppingMonitor(config)
    monitor2.set_state(state)
    self.assertEqual(monitor2.best_loss, 1.0)
    self.assertEqual(monitor2.best_params, {"w": 10})
    self.assertEqual(monitor2.best_step, 1)

    # Adding non-improving loss should increment patience counter to 2
    monitor2.add_loss(3, 1.02, params={"w": 12})
    self.assertFalse(monitor2.should_stop())

    # Adding another non-improving loss should reach patience 3 and trigger stop
    monitor2.add_loss(4, 1.03, params={"w": 13})
    self.assertTrue(monitor2.should_stop())

  def test_get_template_state(self):
    config = early_stopping_monitor.EarlyStoppingMonitorConfig(patience=3)
    monitor = early_stopping_monitor.EarlyStoppingMonitor(config)
    template = monitor.get_template_state(model_params={"w": 0})
    self.assertEqual(template["patience_counter"], 0)
    self.assertFalse(template["should_stop"])
    self.assertEqual(template["best_loss"], 0.0)
    self.assertEqual(template["best_params"], {"w": 0})
    self.assertEqual(template["best_step"], 0)

  def test_set_state_defaults_and_missing_keys(self):
    config = early_stopping_monitor.EarlyStoppingMonitorConfig(patience=3)
    monitor = early_stopping_monitor.EarlyStoppingMonitor(config)
    monitor.add_loss(1, 1.0, params={"w": 1})
    monitor.add_loss(2, 1.1, params={"w": 2})

    monitor.set_state({})
    self.assertEqual(
        monitor.get_state(),
        {
            "patience_counter": 0,
            "should_stop": False,
            "best_loss": None,
            "best_params": None,
            "best_step": None,
        },
    )
    self.assertFalse(monitor.should_stop())

  def test_set_state_should_stop_true(self):
    config = early_stopping_monitor.EarlyStoppingMonitorConfig(patience=3)
    monitor = early_stopping_monitor.EarlyStoppingMonitor(config)
    monitor.set_state({
        "patience_counter": 3,
        "should_stop": True,
        "best_loss": 0.5,
        "best_params": {"w": 1},
        "best_step": 1,
    })
    self.assertTrue(monitor.should_stop())
    # Further loss additions should be ignored when already stopped
    monitor.add_loss(2, 0.1, params={"w": 2})
    self.assertEqual(monitor.best_loss, 0.5)


if __name__ == "__main__":
  absltest.main()
