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

"""Tests for analyse/histogram.py."""

from absl.testing import absltest
from dgf.src.analyse import histogram as analyse_hist
from dgf.src.data import histogram as histogram_lib
from dgf.src.util import test_util
import numpy as np

test_util.disable_diff_truncation()


class HistogramTest(absltest.TestCase):

  def test_make_single_value_histogram(self):
    h = analyse_hist.make_single_value_histogram(10.0)
    expected = histogram_lib.Histogram(values=[1.0], bins=[10.0, 10.0])
    test_util.assert_are_equal(self, h, expected)

  def test_make_histogram_empty(self):
    h = analyse_hist.make_histogram(np.array([]))
    expected = histogram_lib.Histogram()
    test_util.assert_are_equal(self, h, expected)

  def test_make_histogram_single_value(self):
    h = analyse_hist.make_histogram(np.array([5.0, 5.0, 5.0]))
    expected = histogram_lib.Histogram(values=[3.0], bins=[5.0, 5.0])
    test_util.assert_are_equal(self, h, expected)

  def test_make_histogram_linear_float(self):
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    h = analyse_hist.make_histogram(values, num_bins=5)
    expected = histogram_lib.Histogram(
        values=[1.0, 1.0, 1.0, 1.0, 2.0], bins=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    )
    test_util.assert_are_equal(self, h, expected)

  def test_make_histogram_linear_integer(self):
    values = np.array([0, 1, 2, 3, 4, 5])
    h = analyse_hist.make_histogram(values, num_bins=5, is_integer=True)
    expected = histogram_lib.Histogram(
        values=[1.0, 1.0, 1.0, 1.0, 2.0], bins=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    )
    test_util.assert_are_equal(self, h, expected)

  def test_make_histogram_linear_integer_small_range(self):
    values = np.array([0, 1, 2, 2])
    h = analyse_hist.make_histogram(values, num_bins=5, is_integer=True)
    expected = histogram_lib.Histogram(
        values=[1.0, 1.0, 2.0], bins=[0.0, 1.0, 2.0, 3.0]
    )
    test_util.assert_are_equal(self, h, expected)

  def test_make_histogram_log_float(self):
    values = np.array([1.0, 10.0, 100.0])
    h = analyse_hist.make_histogram(values, num_bins=2, log_scale=True)
    expected = histogram_lib.Histogram(
        values=[1.0, 2.0], bins=[1.0, 10.0, 100.0]
    )
    test_util.assert_are_equal(self, h, expected)

  def test_make_histogram_log_float_with_zero(self):
    values = np.array([0.0, 1.0, 10.0, 100.0])
    h = analyse_hist.make_histogram(values, num_bins=3, log_scale=True)
    expected = histogram_lib.Histogram(
        values=[1.0, 1.0, 2.0], bins=[0.0, 1.0, 10.0, 100.0]
    )
    test_util.assert_are_equal(self, h, expected)

  def test_make_histogram_log_integer_with_zero(self):
    values = np.array([0, 1, 10, 100])
    h = analyse_hist.make_histogram(
        values, num_bins=3, log_scale=True, is_integer=True
    )
    expected = histogram_lib.Histogram(
        values=[1.0, 1.0, 2.0], bins=[0.0, 1.0, 10.0, 100.0]
    )
    test_util.assert_are_equal(self, h, expected)


if __name__ == "__main__":
  absltest.main()
