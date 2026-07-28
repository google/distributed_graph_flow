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

"""Dataclass representing the contents of a DGF Graph Snapshots dataset.

A DGF Graph Snapshots dataset is a directory of independent graphs, each one
capturing the state of the same graph at a different point in time. This
dataclass holds those graphs together with the metadata that orders them.
"""

import dataclasses
from typing import Sequence

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
import numpy as np


@dataclasses.dataclass(frozen=True)
class GraphSnapshots:
  """A chronologically ordered sequence of graph snapshots.

  The `timestamps`, `snapshot_ids` and `graphs` attributes are parallel
  sequences: index `t` of each refers to the same snapshot. Snapshots are
  sorted by increasing timestamp, and timestamps are unique.

  All snapshots share a single `schema`, read from the dataset root.

  Attributes:
    timestamps: Array of shape [num_snapshots]. Dtype: int64. The timestamp of
      each snapshot, sorted in increasing order.
    snapshot_ids: The directory name of each snapshot, used for error messages
      and debugging.
    schema: The schema shared by all the snapshots.
    graphs: The in-memory graph of each snapshot.
  """

  timestamps: np.ndarray
  snapshot_ids: Sequence[str]
  schema: schema_lib.GraphSchema
  graphs: Sequence[in_memory_graph_lib.InMemoryGraph]

  def __post_init__(self):
    if len(self.timestamps) != len(self.snapshot_ids) or len(
        self.timestamps
    ) != len(self.graphs):
      raise ValueError(
          "timestamps, snapshot_ids and graphs must have the same length. Got"
          f" {len(self.timestamps)}, {len(self.snapshot_ids)} and"
          f" {len(self.graphs)} respectively."
      )
    if len(self.timestamps) > 1 and not np.all(
        self.timestamps[:-1] < self.timestamps[1:]
    ):
      raise ValueError("timestamps must be strictly increasing.")

  def __len__(self) -> int:
    """Returns the number of snapshots."""
    return len(self.graphs)
