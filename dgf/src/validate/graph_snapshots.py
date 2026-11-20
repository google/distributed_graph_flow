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

"""Validation utilities for DGF Snapshots Datasets."""

from collections.abc import Sequence
import concurrent.futures
import dataclasses
import json
import os
from typing import Any

from dgf.src.data import graph_snapshots_metadata as data_snapshot_metadata
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_memory
from dgf.src.io import graph_snapshots_metadata as io_snapshot_metadata
from dgf.src.io import schema as io_schema
from dgf.src.util import filesystem
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
from dgf.src.validate import validate as validate_lib

Issue = validate_lib.Issue

# Number of threads used to read the metadata of the snapshots.
_NUM_WORKERS = 32


def _get_snapshots_dir(dataset_path: str) -> str:
  return os.path.join(dataset_path, "snapshots")


def _read_snapshot_metadata(snapshot_path: str) -> dict[str, Any] | None:
  """Reads the metadata.json of a snapshot. None if the file is missing."""
  metadata_path = os.path.join(snapshot_path, "metadata.json")
  if not filesystem.exists(metadata_path):
    return None
  with filesystem.open_read(metadata_path) as f:
    return json.loads(f.read())


@dataclasses.dataclass(frozen=True)
class _SnapshotsLayout:
  """Result of the inspection of the layout of a snapshots dataset.

  Attributes:
    issues: Issues found in the layout.
    schema: Root schema. None if it is missing.
    snapshots: (timestamp, snapshot_id) of each snapshot with valid metadata.
  """

  issues: list[validate_lib.Issue]
  schema: schema_lib.GraphSchema | None
  snapshots: list[tuple[int, str]]


def _inspect_snapshots_layout(dataset_path: str) -> _SnapshotsLayout:
  """Inspects dataset layout and snapshot metadata in a single parallel pass."""
  issues: list[validate_lib.Issue] = []

  # 1. Root Metadata Check
  metadata_path = os.path.join(dataset_path, "metadata.json")
  if not filesystem.exists(metadata_path):
    issues.append(Issue.error(f"Missing root metadata.json at {metadata_path}"))
  else:
    meta = io_snapshot_metadata.read_metadata(metadata_path)
    if (
        meta.format
        != data_snapshot_metadata.GraphSnapshotsFormat.GRAPH_SNAPSHOTS
    ):
      issues.append(
          Issue.error(
              "Expected root metadata format='graph_snapshots', found"
              f" {meta.format!r}"
          )
      )

  # 2. Root Schema Check
  global_schema = None
  global_schema_path = os.path.join(dataset_path, "schema.json")
  if not filesystem.exists(global_schema_path):
    issues.append(
        Issue.error(f"Missing root schema.json at {global_schema_path}")
    )
  else:
    global_schema = io_schema.read_schema(global_schema_path)

  # 3. Snapshots Directory & Subdirectory Checks
  snaps_dir = _get_snapshots_dir(dataset_path)
  if not filesystem.exists(snaps_dir):
    issues.append(Issue.error(f"Snapshots directory missing at {snaps_dir}"))
    return _SnapshotsLayout(issues=issues, schema=global_schema, snapshots=[])

  # Each entry of the snapshots directory is a snapshot.
  snapshot_paths = sorted(filesystem.glob(os.path.join(snaps_dir, "*")))
  if snapshot_paths:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(_NUM_WORKERS, len(snapshot_paths))
    ) as executor:
      metadata_dicts = list(
          executor.map(_read_snapshot_metadata, snapshot_paths)
      )
  else:
    metadata_dicts = []
    issues.append(
        Issue.warning(f"No snapshot directories found in {snaps_dir}")
    )

  seen_timestamps: dict[int, str] = {}
  valid_snapshots: list[tuple[int, str]] = []
  for snapshot_path, snap_meta_dict in zip(snapshot_paths, metadata_dicts):
    snapshot_id = os.path.basename(snapshot_path)
    snap_meta_path = os.path.join(snapshot_path, "metadata.json")
    if snap_meta_dict is None:
      issues.append(
          Issue.error(
              f"Snapshot {snapshot_id!r} missing metadata.json at"
              f" {snap_meta_path}"
          )
      )
      continue
    timestamp = snap_meta_dict.get("timestamp")
    if timestamp is None:
      issues.append(
          Issue.error(
              f"Snapshot {snapshot_id!r} metadata.json missing 'timestamp'"
          )
      )
    elif timestamp in seen_timestamps:
      issues.append(
          Issue.error(
              f"Duplicate timestamp {timestamp} found in snapshot"
              f" {snapshot_id!r} (already defined in"
              f" {seen_timestamps[timestamp]!r})"
          )
      )
    else:
      seen_timestamps[timestamp] = snapshot_id
      valid_snapshots.append((int(timestamp), snapshot_id))

  return _SnapshotsLayout(
      issues=issues, schema=global_schema, snapshots=valid_snapshots
  )


def snapshots_issues(
    dataset_path: str, validate_contents: bool = True
) -> Sequence[validate_lib.Issue]:
  """Lists potential storage and content issues in DGF Graph Snapshots.

  Args:
    dataset_path: Path to the DGF Graph Snapshots directory.
    validate_contents: If True, loads each graph snapshot into memory and
      validates node/edge features, shapes, dtypes, and adjacency ranges.

  Returns:
    Sequence of Issue objects describing errors or warnings.
  """
  layout = _inspect_snapshots_layout(dataset_path)
  issues = list(layout.issues)
  if validate_contents and layout.schema is not None:
    snaps_dir = _get_snapshots_dir(dataset_path)
    for _, snapshot_id in layout.snapshots:
      snap_dir = os.path.join(snaps_dir, snapshot_id)
      graph, _ = graph_in_memory.read_graph(
          snap_dir, override_schema=layout.schema
      )
      content_issues = in_memory_graph_validate_lib.issues(graph, layout.schema)
      for issue in content_issues:
        issues.append(
            Issue(issue.severity, f"Snapshot {snapshot_id!r}: {issue.text}")
        )

  return issues


def validate_snapshots(
    dataset_path: str,
    *,
    validate_contents: bool = True,
    raise_on_error: bool = True,
    raise_on_warning: bool = False,
) -> None:
  """Validates the storage layout and contents of DGF Graph Snapshots.

  Usage example:

  ```python
    dgf.validate.validate_snapshots("/tmp/my_snapshots")
  ```

  Args:
    dataset_path: Path to the DGF Graph Snapshots directory.
    validate_contents: If True, reads and validates each individual snapshot
      graph against the schema.
    raise_on_error: If True, raises a ValueError if any error issues are found.
    raise_on_warning: If True, raises a ValueError if any warning issues are
      found.
  """
  validate_lib.print_and_raise(
      snapshots_issues(dataset_path, validate_contents=validate_contents),
      raise_on_error=raise_on_error,
      raise_on_warning=raise_on_warning,
  )
