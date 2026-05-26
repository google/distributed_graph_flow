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
import numpy as np

from dgf.src.generate import signals


class SignalsTest(absltest.TestCase):

  def test_sinusoidal(self):
    t = np.linspace(0, 1, 10)
    config = signals.Sinusoidal(amplitude=2.0, frequency=1.0)
    y = config.evaluate(t)
    self.assertEqual(y.shape, (10,))
    # Test a specific value
    # sin(0) = 0, so at t=0, y should be offset = 0
    self.assertAlmostEqual(y[0], 0.0)

  def test_linear(self):
    t = np.linspace(0, 1, 10)
    config = signals.Linear(slope=2.0, intercept=1.0)
    y = config.evaluate(t)
    np.testing.assert_array_almost_equal(y, 2.0 * t + 1.0)

  def test_composite_signal(self):
    t = np.linspace(0, 1, 10)
    config = signals.CompositeSignal(
        components=[signals.Linear(slope=1.0), signals.Linear(slope=2.0)],
        weights=[1.0, 1.0],
    )
    y = config.evaluate(t)
    np.testing.assert_array_almost_equal(y, 3.0 * t)

  def test_registry(self):
    config = signals.SignalRegistry.create('CPU_IDLE')
    self.assertIsInstance(config, signals.BetaSeries)

  def test_registry_with_kwargs(self):
    config = signals.SignalRegistry.create('NOISY_SINE', frequency=2.0)
    self.assertIsInstance(config, signals.CompositeSignal)
    component = config.components[0]
    assert isinstance(component, signals.Sinusoidal)
    self.assertEqual(component.frequency, 2.0)


if __name__ == '__main__':
  absltest.main()
