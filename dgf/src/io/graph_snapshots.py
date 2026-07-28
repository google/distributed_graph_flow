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

"""IO functions to read a DGF Graph Snapshots dataset into memory."""

import concurrent.futures
import os
from typing import Callable, List, Sequence, Tuple, TypeVar

from dgf.src.data import gf_metadata as gf_metadata_lib
from dgf.src.data import graph_snapshots as graph_snapshots_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_constants
from dgf.src.io import graph_in_memory
from dgf.src.io import schema as io_schema
from dgf.src.transform import combine_graph_snapshots as combine_lib
from dgf.src.util import filesystem
from dgf.src.util import log
from dgf.src.validate import graph_snapshots as snapshots_validate_lib
import numpy as np

_DEFAULT_NUM_WORKERS = 32

_T = TypeVar("_T")
_R = TypeVar("_R")


def _parallel_map(
    fn: Callable[[_T], _R], items: Sequence[_T], num_workers: int
) -> List[_R]:
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
) -> List[Tuple[int, str]]:
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


def _load_snapshot_graphs(
    dataset_path: str,
    snapshot_ids: Sequence[str],
    schema: schema_lib.GraphSchema,
    num_workers: int,
) -> List[in_memory_graph_lib.InMemoryGraph]:
  """Loads the in-memory graph of each snapshot in parallel."""

  def _load_single(snapshot_id: str) -> in_memory_graph_lib.InMemoryGraph:
    snapshot_dir = os.path.join(
        dataset_path, graph_constants.DIRNAME_SNAPSHOTS, snapshot_id
    )
    return graph_in_memory.read_graph(
        snapshot_dir,
        override_schema=schema,
        verbose=False,
    )[0]

  return _parallel_map(_load_single, snapshot_ids, num_workers)


def read_graph_snapshots(
    path: str,
    verbose: bool = False,
    num_workers: int = _DEFAULT_NUM_WORKERS,
) -> graph_snapshots_lib.GraphSnapshots:
  """Reads a DGF Graph Snapshots dataset from disk into memory.

  A Graph Snapshots dataset is a directory of independent graphs, each one
  capturing the state of the same graph at a different point in time. The
  snapshots are returned sorted by increasing timestamp.

  All the snapshot graphs are loaded into memory. Use
  `dgf.transform.combine_graph_snapshots` to convert them into a single
  Temporal Graph.

  This function validates the dataset layout, but not the contents of the
  snapshot graphs. Call `dgf.validate.validate_snapshots` to also check the
  graphs against the schema.

  Usage example:

  ```python
  dgf.validate.validate_snapshots("/path/to/snapshot_dataset")

  snapshots = dgf.io.read_graph_snapshots("/path/to/snapshot_dataset")
  graph, schema = dgf.transform.combine_graph_snapshots(snapshots)
  ```

  Args:
    path: Path to the dataset directory containing 'schema.json' and a
      'snapshots/' sub-directory.
    verbose: If True, print progress information.
    num_workers: Number of threads used to read the snapshots.

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


def read_graph_snapshots_as_temporal_graph(
    path: str,
    verbose: bool = False,
    num_workers: int = _DEFAULT_NUM_WORKERS,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a DGF Graph Snapshots dataset and combines it into a Temporal Graph.

  Convenience wrapper that reads the snapshots with `dgf.io.read_graph_snapshots`
  and combines them with `dgf.transform.combine_graph_snapshots`. Call those two
  directly if you need to inspect or modify the snapshots in between.

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
