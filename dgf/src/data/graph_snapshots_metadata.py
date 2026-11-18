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

"""Dataclasses for representing DGF Graph Snapshots metadata.

Graph Snapshots metadata (`metadata.json`) marks a root directory as a DGF
Graph Snapshots dataset and specifies the format and version.
"""

import dataclasses
import enum

import dataclasses_json
from dgf.src.io import graph_constants


class GraphSnapshotsFormat(str, enum.Enum):
  """Format options for snapshots datasets."""

  GRAPH_SNAPSHOTS = graph_constants.FORMAT_GRAPH_SNAPSHOTS


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class GraphSnapshotsMetadata:
  """Root metadata for a DGF Graph Snapshots dataset.

  Attributes:
    format: The format of the dataset.
    version: The version of the format.
    name: Optional name of the graph captured by the snapshots, e.g. to tell
      apart the datasets of several graphs.
  """

  format: GraphSnapshotsFormat | str
  version: int
  name: str | None = None
