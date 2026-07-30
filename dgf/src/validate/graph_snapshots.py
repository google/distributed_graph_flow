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

import json
import os
from typing import List, Sequence

from dgf.src.data import graph_snapshots_metadata as data_snapshot_metadata
from dgf.src.io import graph_in_memory
from dgf.src.io import graph_snapshots_metadata as io_snapshot_metadata
from dgf.src.io import schema as io_schema
from dgf.src.util import filesystem
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
from dgf.src.validate import validate as validate_lib
from etils import epath

Issue = validate_lib.Issue


def _get_snapshots_dir(dataset_path: str) -> str:
  return os.path.join(dataset_path, "snapshots")


def _list_snapshot_ids(dataset_path: str) -> List[str]:
  snaps_dir = _get_snapshots_dir(dataset_path)
  if not filesystem.exists(snaps_dir):
    return []
  paths = filesystem.glob(os.path.join(snaps_dir, "*"))
  return sorted([
      os.path.basename(p) for p in paths if epath.Path(p).is_dir()
  ])


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
  issues: List[validate_lib.Issue] = []

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
  global_schema_path = os.path.join(dataset_path, "schema.json")
  global_schema = None
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
    return issues

  snapshot_ids = _list_snapshot_ids(dataset_path)
  if not snapshot_ids:
    issues.append(
        Issue.warning(f"No snapshot directories found in {snaps_dir}")
    )

  seen_timestamps = {}
  for snap_id in snapshot_ids:
    snap_dir = os.path.join(snaps_dir, snap_id)

    # Validate per-snapshot metadata.json
    snap_meta_path = os.path.join(snap_dir, "metadata.json")
    snap_meta_valid = True
    if not filesystem.exists(snap_meta_path):
      snap_meta_valid = False
      issues.append(
          Issue.error(
              f"Snapshot {snap_id!r} missing metadata.json at {snap_meta_path}"
          )
      )
    else:
      with filesystem.open_read(snap_meta_path) as f:
        snap_meta_dict = json.loads(f.read())
      timestamp = snap_meta_dict.get("timestamp")
      if timestamp is None:
        snap_meta_valid = False
        issues.append(
            Issue.error(
                f"Snapshot {snap_id!r} metadata.json missing 'timestamp'"
            )
        )
      elif timestamp in seen_timestamps:
        issues.append(
            Issue.error(
                f"Duplicate timestamp {timestamp} found in snapshot"
                f" {snap_id!r} (already defined in"
                f" {seen_timestamps[timestamp]!r})"
            )
        )
      else:
        seen_timestamps[timestamp] = snap_id

    # Read and validate in-memory graph content if requested
    if validate_contents and global_schema is not None and snap_meta_valid:
      graph, _ = graph_in_memory.read_graph(
          snap_dir, override_schema=global_schema
      )
      content_issues = in_memory_graph_validate_lib.issues(
          graph, global_schema
      )
      for issue in content_issues:
        issues.append(
            Issue(issue.severity, f"Snapshot {snap_id!r}: {issue.text}")
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
      snapshots_issues(
          dataset_path, validate_contents=validate_contents
      ),
      raise_on_error=raise_on_error,
      raise_on_warning=raise_on_warning,
  )
