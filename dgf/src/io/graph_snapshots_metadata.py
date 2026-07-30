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

"""IO functions for representing and serializing DGF Graph Snapshots metadata.

Graph Snapshots metadata (`metadata.json`) marks a root directory as a DGF
Graph Snapshots dataset and specifies the format and version.
"""

import json

from dgf.src.data import graph_snapshots_metadata as snapshot_metadata_data
from dgf.src.util import filesystem


def read_metadata(
    metadata_path: str,
) -> snapshot_metadata_data.GraphSnapshotsMetadata:
  """Reads and parses GraphSnapshotsMetadata from a JSON file."""
  with filesystem.open_read(metadata_path) as f:
    data = json.loads(f.read())
  return snapshot_metadata_data.GraphSnapshotsMetadata.from_dict(data)  # pytype: disable=attribute-error  # pyrefly: ignore[missing-attribute]


def write_metadata(
    metadata: snapshot_metadata_data.GraphSnapshotsMetadata, metadata_path: str
) -> None:
  """Serializes and writes GraphSnapshotsMetadata to a JSON file."""
  with filesystem.open_write(metadata_path) as f:
    f.write(
        json.dumps(metadata.to_dict(), indent=2)  # pytype: disable=attribute-error  # pyrefly: ignore[missing-attribute]
    )
