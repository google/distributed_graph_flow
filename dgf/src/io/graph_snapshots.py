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

"""IO functions to read and write a DGF Graph Snapshots dataset."""

from collections.abc import Callable, Sequence
import concurrent.futures
import dataclasses
import os
from typing import TypeVar, overload

from dgf.src.data import gf_metadata as gf_metadata_lib
from dgf.src.data import graph_snapshots as graph_snapshots_lib
from dgf.src.data import graph_snapshots_metadata as snapshots_metadata_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_constants
from dgf.src.io import graph_in_memory
from dgf.src.io import graph_snapshots_metadata as io_snapshots_metadata
from dgf.src.io import schema as io_schema
from dgf.src.transform import combine_graph_snapshots as combine_lib
from dgf.src.util import filesystem
from dgf.src.util import log
from dgf.src.validate import graph_snapshots as snapshots_validate_lib
import numpy as np

_DEFAULT_NUM_WORKERS = 32

# Version of the Graph Snapshots format written by `write_graph_snapshots`.
_GRAPH_SNAPSHOTS_VERSION = 0

_T = TypeVar("_T")
_R = TypeVar("_R")


def _parallel_map(
    fn: Callable[[_T], _R], items: Sequence[_T], num_workers: int
) -> list[_R]:
  """Applies `fn` to each item on a thread pool, preserving the input order."""
  if not items:
    return []
  max_workers = max(1, min(num_workers, len(items)))
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    return list(executor.map(fn, items))


def _read_snapshot_timestamp(snapshot_path: str) -> int:
  """Reads the timestamp of a single snapshot from its metadata."""
  meta_path = os.path.join(snapshot_path, graph_constants.FILENAME_METADATA)
  with filesystem.open_read(meta_path) as f:
    # pyrefly: ignore[missing-attribute]
    meta = gf_metadata_lib.GFGraphMetadata.from_json(f.read())
  assert meta.timestamp is not None
  return meta.timestamp


def _discover_and_sort_snapshots(
    dataset_path: str, num_workers: int
) -> list[tuple[int, str]]:
  """Lists the snapshots of a dataset, sorted by increasing timestamp.

  Only the snapshot metadata is read; the snapshot graphs are left on disk.

  Args:
    dataset_path: Path to the DGF Graph Snapshots directory.
    num_workers: Number of threads used to read the snapshot metadata.

  Returns:
    A list of (timestamp, snapshot_id) pairs sorted by increasing timestamp.
  """
  snapshots_dir = os.path.join(
      dataset_path, graph_constants.DIRNAME_SNAPSHOTS
  )
  snapshot_paths = sorted(
      path
      for path in filesystem.glob(os.path.join(snapshots_dir, "*"))
      if filesystem.is_dir(path)
  )
  if not snapshot_paths:
    raise ValueError(f"No snapshot directories found in {snapshots_dir}")

  timestamps = _parallel_map(
      _read_snapshot_timestamp, snapshot_paths, num_workers
  )
  snapshot_info = [
      (timestamp, os.path.basename(path))
      for timestamp, path in zip(timestamps, snapshot_paths)
  ]
  snapshot_info.sort()
  return snapshot_info


def _read_snapshot_graph(
    dataset_path: str, snapshot_id: str, schema: schema_lib.GraphSchema
) -> in_memory_graph_lib.InMemoryGraph:
  """Reads the in-memory graph of a single snapshot."""
  snapshot_dir = os.path.join(
      dataset_path, graph_constants.DIRNAME_SNAPSHOTS, snapshot_id
  )
  return graph_in_memory.read_graph(
      snapshot_dir,
      override_schema=schema,
      verbose=False,
  )[0]


def _load_snapshot_graphs(
    dataset_path: str,
    snapshot_ids: Sequence[str],
    schema: schema_lib.GraphSchema,
    num_workers: int,
) -> list[in_memory_graph_lib.InMemoryGraph]:
  """Loads the in-memory graph of each snapshot in parallel."""

  def _load_single(snapshot_id: str) -> in_memory_graph_lib.InMemoryGraph:
    return _read_snapshot_graph(dataset_path, snapshot_id, schema)

  return _parallel_map(_load_single, snapshot_ids, num_workers)


class _SnapshotGraphsOnDisk(Sequence[in_memory_graph_lib.InMemoryGraph]):
  """The graphs of the snapshots of a dataset, read from disk when accessed.

  A graph is read each time it is accessed and it is not kept: Only the graphs
  held by the caller are in memory.
  """

  def __init__(
      self,
      dataset_path: str,
      snapshot_ids: Sequence[str],
      schema: schema_lib.GraphSchema,
  ):
    self._dataset_path = dataset_path
    self._snapshot_ids = list(snapshot_ids)
    self._schema = schema

  def __len__(self) -> int:
    return len(self._snapshot_ids)

  @overload
  def __getitem__(self, index: int) -> in_memory_graph_lib.InMemoryGraph: ...

  @overload
  def __getitem__(
      self, index: slice
  ) -> list[in_memory_graph_lib.InMemoryGraph]: ...

  def __getitem__(self, index):
    if isinstance(index, slice):
      return [
          _read_snapshot_graph(self._dataset_path, snapshot_id, self._schema)
          for snapshot_id in self._snapshot_ids[index]
      ]
    return _read_snapshot_graph(
        self._dataset_path, self._snapshot_ids[index], self._schema
    )


def read_graph_snapshots(
    path: str,
    verbose: bool = False,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    lazy: bool = False,
) -> graph_snapshots_lib.GraphSnapshots:
  """Reads a DGF Graph Snapshots dataset from disk.

  A Graph Snapshots dataset is a directory of independent graphs, each one
  capturing the state of the same graph at a different point in time. The
  snapshots are returned sorted by increasing timestamp.

  By default, all the snapshot graphs are loaded into memory. With `lazy=True`,
  only the schema and the timestamps are read: each graph is read from disk
  when it is accessed, e.g. with `snapshots.graphs[i]`, and it is not kept. Use
  it for snapshots that do not fit in memory together.

  Use `dgf.transform.combine_graph_snapshots` to convert the snapshots into a
  single Temporal Graph.

  This function validates the dataset layout, but not the contents of the
  snapshot graphs. Call `dgf.validate.validate_snapshots` to also check the
  graphs against the schema.

  Usage example:

  ```python
  dgf.validate.validate_snapshots("/path/to/snapshot_dataset")

  snapshots = dgf.io.read_graph_snapshots("/path/to/snapshot_dataset")
  graph, schema = dgf.transform.combine_graph_snapshots(snapshots)

  # One snapshot in memory at a time.
  snapshots = dgf.io.read_graph_snapshots(
      "/path/to/snapshot_dataset", lazy=True
  )
  for timestamp, graph in zip(snapshots.timestamps, snapshots.graphs):
    ...
  ```

  Args:
    path: Path to the dataset directory containing 'schema.json' and a
      'snapshots/' sub-directory.
    verbose: If True, print progress information.
    num_workers: Number of threads used to read the snapshots.
    lazy: If True, the graphs are not loaded: each one is read from disk each
      time it is accessed.

  Returns:
    The graph snapshots, sorted by increasing timestamp.
  """
  schema_path = os.path.join(path, graph_constants.FILENAME_SCHEMA)
  if not filesystem.exists(schema_path):
    raise ValueError(f"Root schema.json missing at {schema_path}")
  if verbose:
    log.info("Reading global schema from %s", schema_path)
  schema = io_schema.read_schema(schema_path)

  if verbose:
    log.info("Validating snapshot dataset at %s", path)
  snapshots_validate_lib.validate_snapshots(path, validate_contents=False)

  if verbose:
    log.info("Discovering snapshots in %s", path)
  snapshot_info = _discover_and_sort_snapshots(path, num_workers=num_workers)
  timestamps = np.array([ts for ts, _ in snapshot_info], dtype=np.int64)
  snapshot_ids = [snapshot_id for _, snapshot_id in snapshot_info]
  if verbose:
    log.info(
        "Discovered %d snapshot(s) with timestamps: %s",
        len(snapshot_ids),
        timestamps,
    )

  if lazy:
    graphs = _SnapshotGraphsOnDisk(path, snapshot_ids, schema)
  else:
    if verbose:
      log.info("Loading %d snapshot graphs...", len(snapshot_ids))
    graphs = _load_snapshot_graphs(
        path, snapshot_ids, schema=schema, num_workers=num_workers
    )
    if verbose:
      log.info("Loaded %d snapshot graphs.", len(graphs))

  return graph_snapshots_lib.GraphSnapshots(
      timestamps=timestamps,
      snapshot_ids=snapshot_ids,
      schema=schema,
      graphs=graphs,
  )


def _check_snapshot_id(snapshot_id: str) -> None:
  """Raises a ValueError if `snapshot_id` is not a single directory name."""
  if (
      not snapshot_id
      or snapshot_id in (".", "..")
      or "/" in snapshot_id
      or "\\" in snapshot_id
      or os.path.basename(snapshot_id) != snapshot_id
  ):
    raise ValueError(
        "A snapshot id should be the name of a directory, without path"
        f" separators. Got {snapshot_id!r}."
    )


def write_graph_snapshots(
    snapshots: graph_snapshots_lib.GraphSnapshots,
    path: str,
    verbose: bool = False,
) -> None:
  """Writes graph snapshots to disk as a DGF Graph Snapshots dataset.

  The graphs are read from `snapshots.graphs` and written one at a time, such
  that snapshots that do not fit in memory together can be written from a
  sequence that creates each graph when it is accessed.

  Usage example:

  ```python
  snapshots = dgf.data.GraphSnapshots(
      timestamps=np.array([100, 200], dtype=np.int64),
      snapshot_ids=["first", "second"],
      schema=schema,
      graphs=[first_graph, second_graph],
  )
  dgf.io.write_graph_snapshots(snapshots, "/path/to/snapshot_dataset")

  snapshots = dgf.io.read_graph_snapshots("/path/to/snapshot_dataset")
  ```

  Args:
    snapshots: The graph snapshots. The timestamp of a graph, if it has one,
      should be the timestamp of its snapshot.
    path: Path to the dataset directory to create.
    verbose: If True, print progress information.

  Raises:
    ValueError: If a snapshot id is not a valid directory name, or if the
      timestamp of a graph is not the timestamp of its snapshot.
  """
  # The snapshot ids are directory names: They are checked before anything is
  # written, so that a dataset is never written outside of `path`.
  for snapshot_id in snapshots.snapshot_ids:
    _check_snapshot_id(snapshot_id)
  filesystem.makedirs(path)
  io_snapshots_metadata.write_metadata(
      snapshots_metadata_lib.GraphSnapshotsMetadata(
          format=snapshots_metadata_lib.GraphSnapshotsFormat.GRAPH_SNAPSHOTS,
          version=_GRAPH_SNAPSHOTS_VERSION,
      ),
      os.path.join(path, graph_constants.FILENAME_METADATA),
  )
  io_schema.write_schema(
      snapshots.schema, os.path.join(path, graph_constants.FILENAME_SCHEMA)
  )
  for index, snapshot_id in enumerate(snapshots.snapshot_ids):
    timestamp = int(snapshots.timestamps[index])
    graph = snapshots.graphs[index]
    if graph.timestamp is not None and graph.timestamp != timestamp:
      raise ValueError(
          f"The graph of snapshot {snapshot_id!r} has timestamp"
          f" {graph.timestamp}, but the timestamp of the snapshot is"
          f" {timestamp}."
      )
    if verbose:
      log.info(
          "Writing snapshot %d / %d (%s)",
          index + 1,
          len(snapshots),
          snapshot_id,
      )
    graph_in_memory.write_graph(
        dataclasses.replace(graph, timestamp=timestamp),
        snapshots.schema,
        path=os.path.join(path, graph_constants.DIRNAME_SNAPSHOTS, snapshot_id),
        verbose=False,
    )


def read_graph_snapshots_as_temporal_graph(
    path: str,
    verbose: bool = False,
    num_workers: int = _DEFAULT_NUM_WORKERS,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a DGF Graph Snapshots dataset and combines it into a Temporal Graph.

  Convenience wrapper that reads the snapshots with
  `dgf.io.read_graph_snapshots` and combines them with
  `dgf.transform.combine_graph_snapshots`. Call those two directly if you need
  to inspect or modify the snapshots in between.

  This function validates the dataset layout, but not the contents of the
  snapshot graphs. Call `dgf.validate.validate_snapshots` to also check the
  graphs against the schema.

  Usage example:

  ```python
  dgf.validate.validate_snapshots("/path/to/snapshot_dataset")

  graph, schema = dgf.io.read_graph_snapshots_as_temporal_graph(
      "/path/to/snapshot_dataset"
  )
  ```

  Args:
    path: Path to the dataset directory containing 'schema.json' and a
      'snapshots/' sub-directory.
    verbose: If True, print progress information.
    num_workers: Number of threads used to read the snapshots.

  Returns:
    A tuple (temporal_graph, temporal_schema) where temporal_graph is an
    InMemoryGraph and temporal_schema is a GraphSchema.
  """
  snapshots = read_graph_snapshots(
      path, verbose=verbose, num_workers=num_workers
  )
  return combine_lib.combine_graph_snapshots(snapshots, verbose=verbose)
