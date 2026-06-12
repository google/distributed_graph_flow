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

"""Histogram dataclass and ASCII art printing."""

import dataclasses
import math
from typing import List

_DEFAULT_PRECISION = 8


@dataclasses.dataclass
class Histogram:
  """A dataclass representing a histogram.

  Attributes:
    values: List of float values representing the counts/heights of the bins.
    bins: List of float values representing the bin edges. Must have length
      len(values) + 1.
  """

  values: List[float] = dataclasses.field(default_factory=list)
  bins: List[float] = dataclasses.field(default_factory=list)

  def __str__(self) -> str:
    if not self.values:
      return "empty"
    if len(self.bins) != len(self.values) + 1:
      return "invalid data"

    if (
        len(self.values) == 1
        and self.values[0] == 1.0
        and len(self.bins) == 2
        and self.bins[0] == self.bins[1]
    ):
      return f"{self.bins[0]:.{_DEFAULT_PRECISION}g} (a single value)"

    total_count = sum(self.values)
    max_count = max(self.values)

    minimum = self.bins[0]
    maximum = self.bins[-1]

    if total_count > 0:
      centers = [
          (self.bins[i] + self.bins[i + 1]) / 2.0
          for i in range(len(self.values))
      ]
      mean = sum(w * x for w, x in zip(self.values, centers)) / total_count
      variance = (
          sum(w * (x - mean) ** 2 for w, x in zip(self.values, centers))
          / total_count
      )
      stddev = math.sqrt(variance)
    else:
      mean = math.nan
      stddev = math.nan

    count_str = (
        f"{int(total_count)}"
        if total_count.is_integer()
        else f"{total_count:.{_DEFAULT_PRECISION}g}"
    )

    # Format header
    s = (
        f"Count:{count_str} Average:{mean:.{_DEFAULT_PRECISION}g}"
        f" StdDev:{stddev:.{_DEFAULT_PRECISION}g}"
        f" Min:{minimum:.{_DEFAULT_PRECISION}g}"
        f" Max:{maximum:.{_DEFAULT_PRECISION}g}\n"
    )

    # Pre-generate strings to calculate widths for alignment
    intervals = []
    counts = []
    percentages = []
    cum_percentages = []
    bars = []

    cum_percentage = 0.0
    for i in range(len(self.values)):
      lower = self.bins[i]
      upper = self.bins[i + 1]
      count = self.values[i]

      percentage = (count / total_count * 100.0) if total_count > 0 else 0.0
      cum_percentage += percentage

      bar_len = int(round(count / max_count * 10)) if max_count > 0 else 0
      bar = "#" * bar_len

      is_last = i == len(self.values) - 1
      interval_close = "]" if is_last else ")"

      intervals.append(
          f"[{lower:.{_DEFAULT_PRECISION}g},"
          f" {upper:.{_DEFAULT_PRECISION}g}{interval_close}"
      )
      counts.append(
          f"{int(count)}"
          if count.is_integer()
          else f"{count:.{_DEFAULT_PRECISION}g}"
      )
      percentages.append(f"{percentage:.2f}%")
      cum_percentages.append(f"{cum_percentage:.2f}%")
      bars.append(bar)

    max_interval_len = max(len(x) for x in intervals)
    max_count_len = max(len(x) for x in counts)
    max_pct_len = max(len(x) for x in percentages)
    max_cum_pct_len = max(len(x) for x in cum_percentages)

    # We want to draw a separator line that matches the total width
    total_width = (
        max_interval_len + max_count_len + max_pct_len + max_cum_pct_len + 14
    )
    s += "-" * total_width + "\n"

    for i in range(len(self.values)):
      s += (
          f"{intervals[i]:<{max_interval_len}} "
          f"{counts[i]:>{max_count_len}} "
          f"{percentages[i]:>{max_pct_len}} "
          f"{cum_percentages[i]:>{max_cum_pct_len}} "
          f"{bars[i]}\n"
      )

    return s
