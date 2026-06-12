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

"""Histogram utilities for analysis."""

from dgf.src.data import histogram as histogram_lib
import numpy as np


def make_histogram(
    values: np.ndarray,
    num_bins: int = 32,
    log_scale: bool = False,
    is_integer: bool = False,
) -> histogram_lib.Histogram:
  """Helper to create a Histogram from a numpy array of values.

  Args:
    values: A numpy array of values.
    num_bins: Desired number of bins.
    log_scale: If True, use logarithmic bins. Useful for power-law distributions
      like degrees.
    is_integer: If True, force bin edges to be integers.

  Returns:
    A Histogram object.
  """
  if values.size == 0:
    return histogram_lib.Histogram()

  min_val = np.min(values)
  max_val = np.max(values)

  if min_val == max_val:
    return histogram_lib.Histogram(
        values=[float(values.size)], bins=[float(min_val), float(min_val)]
    )

  if is_integer:
    # If range is smaller than num_bins, use unit-width bins
    if max_val - min_val < num_bins:
      bins = np.arange(min_val, max_val + 2, dtype=int)
    else:
      if log_scale:
        if min_val == 0:
          # Handle 0 by prepend it to log-spaced bins starting at 1
          log_bins = np.logspace(0, np.log10(max_val), num_bins)
          bins = np.concatenate(([0], np.round(log_bins).astype(int)))
        else:
          # If min_val > 0, we can just log-space
          bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)
          bins = np.round(bins).astype(int)
      else:
        # Linear integer bins
        bins = np.linspace(min_val, max_val, num_bins + 1)
        bins = np.round(bins).astype(int)

      # Ensure bins are unique (rounding might introduce duplicates)
      bins = np.unique(bins)
  else:
    # Float bins
    if log_scale:
      if min_val == 0:
        # We need a small epsilon to start logspace.
        # Let's find the minimum positive value, or use a default epsilon.
        pos_values = values[values > 0]
        epsilon = np.min(pos_values) if pos_values.size > 0 else 1e-3
        # Logspace from epsilon to max_val
        log_bins = np.logspace(np.log10(epsilon), np.log10(max_val), num_bins)
        bins = np.concatenate(([0.0], log_bins))
      else:
        bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)
    else:
      # Linear float bins
      bins = np.linspace(min_val, max_val, num_bins + 1)

  counts, bin_edges = np.histogram(values, bins=bins)
  return histogram_lib.Histogram(
      values=counts.astype(float).tolist(), bins=bin_edges.tolist()
  )


def make_single_value_histogram(value: float) -> histogram_lib.Histogram:
  """Creates a histogram representing a single value."""
  return histogram_lib.Histogram(
      values=[1.0], bins=[float(value), float(value)]
  )
