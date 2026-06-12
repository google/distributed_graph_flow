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

"""Tests for histogram."""

from absl.testing import absltest
from dgf.src.data import histogram


class HistogramTest(absltest.TestCase):

  def test_histogram_printing(self):
    h = histogram.Histogram(values=[4.0, 34.0, 8.0], bins=[0.0, 1.0, 2.0, 5.0])
    expected_output = """\
Count:46 Average:1.7608696 StdDev:0.845314 Min:0 Max:5
-----------------------------------
[0, 1)  4  8.70%   8.70% #
[1, 2) 34 73.91%  82.61% ##########
[2, 5]  8 17.39% 100.00% ##
"""
    self.assertEqual(str(h), expected_output)

  def test_histogram_printing_empty(self):
    h = histogram.Histogram()
    self.assertEqual(str(h), "empty")

  def test_histogram_printing_invalid_bins(self):
    h = histogram.Histogram(values=[4.0, 34.0, 8.0], bins=[0.0, 1.0])
    self.assertEqual(str(h), "invalid data")

  def test_histogram_printing_large_values(self):
    h = histogram.Histogram(
        values=[1000000.0, 2000000.0, 3000000.0], bins=[0.0, 1e6, 2e6, 5e6]
    )
    expected_output = """\
Count:6000000 Average:2333333.3 StdDev:1213351.6 Min:0 Max:5000000
----------------------------------------------------
[0, 1000000)       1000000 16.67%  16.67% ###
[1000000, 2000000) 2000000 33.33%  50.00% #######
[2000000, 5000000] 3000000 50.00% 100.00% ##########
"""
    self.assertEqual(str(h), expected_output)

  def test_histogram_printing_single_value(self):
    h = histogram.Histogram(values=[1.0], bins=[10.0, 10.0])
    self.assertEqual(str(h), "10 (a single value)")


if __name__ == "__main__":
  absltest.main()
