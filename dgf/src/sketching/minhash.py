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

"""Utilities for computing minhash sketches.
"""

import apache_beam as beam
import datasketch.minhash as ds_minhash

MinHash = ds_minhash.MinHash


# TODO(bmayer): Allow user to specify the datasketch.LeanMinHash to save mem.
class DatasSketchMinHashCombiner(beam.CombineFn):
  """CombineFn that computes a minhash on a stream of strings.

  Quick/Simple implementation of a distributed minhash combiner using the
  open-source datasketch library.

  The minhash has an update and merge interface which we can use in a simple
  combiner.
  """

  def __init__(self, num_perm=128):
    """Initializes the combiner with the number of permutations.

    Args:
      num_perm: The number of random permutation functions
    """
    self.num_perm = num_perm

  def create_accumulator(self):
    """Initializes an empty MinHash object."""

    # TODO(bmayer): Datasketch uses numpy random number generators. Check if we
    # need to control seeding in the distributed setting.
    return MinHash(num_perm=self.num_perm)

  def add_input(self, accumulator: MinHash, element: bytes):
    """Updates the MinHash object with a new string element."""
    accumulator.update(element)
    return accumulator

  def merge_accumulators(self, accumulators):
    """Merges multiple MinHash objects."""
    # Create a new MinHash object to store the merged result
    merged_minhash = self.create_accumulator()

    # Merge each accumulator into the new object
    for accumulator in accumulators:
      merged_minhash.merge(accumulator)

    return merged_minhash

  def extract_output(self, accumulator):
    """The MinHash object itself is the final output."""
    return accumulator
