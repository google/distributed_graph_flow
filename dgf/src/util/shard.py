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

"""Utilities to handle shards."""

import re
from typing import List, Optional, Tuple
import tensorflow as tf

NUM_NODES_PER_SHARD = 1000
NUM_EDGES_PER_SHARD = 1000


def expand_output_paths(path: str, num_shards: Optional[int]) -> List[str]:
  """Generates a list of concrete filenames from a path expression.

  The input path can be a sharded path or a concrete path. If sharded, but with
  a non defined number of shards (e.g. data@*.rio), use "num_shards" shards. If
  the number of shards is defined (e.g., data@5.rio), "num_shards" is ignored.

  This function is only applicable for output path. For output path extension,
  use "expand_input_paths".

  Examples:
    "/a/b" => ["/a/b"]
    "/a/b@2" => ["/a/b-00000-of-00002", "/a/b-00001-of-00002"]
    "/a/b@2.ext" => ["/a/b-00000-of-00002.ext", "/a/b-00001-of-00002.ext"]
    "/a/b@*.ext" with num_shards=2 => ["/a/b-00000-of-00002.ext",
    "/a/b-00001-of-00002.ext"]

  Args:
    path: Path, possibly sharded.
    num_shards: Number of shards.

  Returns:
    Returns the list of paths.
  """
  sharding_spec = _match_sharded_path(path)
  if sharding_spec is not None:
    # This is a sharded path
    basename, num_shards_or_star, extension = sharding_spec
    extension = extension if extension else ""
    if num_shards_or_star == "*":
      if num_shards is None:
        raise ValueError(
            f"Must specify `num_shards` if filespece contains a * spec: {path}."
        )
      return [
          sharded_filename(basename, i, num_shards, extension)
          for i in range(num_shards)
      ]
    else:
      num_shards = int(num_shards_or_star)
      return [
          sharded_filename(basename, i, num_shards, extension)
          for i in range(num_shards)
      ]
  else:
    if any(c in path for c in "*?[]"):
      raise ValueError(
          f"Wildcards are not allowed in output paths, but got: {path}"
      )
    else:
      # This is a concrete path.
      return [path]


def expand_input_paths(path: str) -> List[str]:
  """Generates a list of concrete filenames from a path expression.

  The input path can be a sharded path, a glob path, or a concrete path.

  This function is only applicable for input path i.e. where then files already
  exist. For output path extension, use "expand_output_paths".

  It's the most general function in this module for expanding a path expression.

  Examples:
    "/a/b" => ["/a/b"]
    "/a/b@2" => ["/a/b-00000-of-00002", "/a/b-00001-of-00002"]
    "/a/b@2.ext" => ["/a/b-00000-of-00002.ext", "/a/b-00001-of-00002.ext"]
    "/a/b@*.ext" => ["/a/b-00000-of-00002.ext", "/a/b-00001-of-00002.ext"]
    "/a/b-00000-of-00002.ext" => ["/a/b-00000-of-00002.ext"]
    "/a/b-0000?-of-00002.ext" => ["/a/b-00000-of-00002", "/a/b-00001-of-00002"]
    "/a/*.ext" => ["/a/b-00000-of-00002.ext", "/a/b-00001-of-00002.ext"]

  If the expression contains @*, * or, ?, this function will scan the directory.

  Args:
    path: Path, possibly sharded.

  Returns:
    Returns the list of paths.
  """

  sharding_spec = _match_sharded_path(path)
  if sharding_spec is not None:
    # This is a sharded path
    basename, num_shards_or_star, extension = sharding_spec
    extension = extension if extension else ""
    if num_shards_or_star == "*":
      # Scan the directory to find the actual files.
      shard_glob = shard_pattern_to_glob(basename, extension)
      return sorted(tf.io.gfile.glob(shard_glob))
    else:
      num_shards = int(num_shards_or_star)
      return [
          sharded_filename(basename, i, num_shards, extension)
          for i in range(num_shards)
      ]
  else:
    if any(c in path for c in "*?[]"):
      # This is a glob
      return sorted(tf.io.gfile.glob(path))
    else:
      # This is a concrete path.
      return [path]


def shard_pattern_to_glob(basename: str, extension: str) -> str:
  """Creates a glob matching a sharded set of files.

  Args:
    basename: The base name of the sharded files, e.g., "my_file".
    extension: The extension of the sharded files, e.g., ".sst".

  Returns:
    A glob pattern that matches all shards of the file, e.g.,
    "my_file-?????-of-?????.sst".
  """
  ## shard pattern like basename-?????-of-????? for Beam Dataflow.
  ## shard pattern like basename-* for BigQuery export.
  return f"{basename}-*{extension}"


def sharded_filename(
    filename: str, shard: int, num_shards: int, extension: str
) -> str:
  """Generates a sharded filename.

  Args:
    filename: The base name of the sharded file, e.g., "my_file".
    shard: The index of the shard, e.g., 0.
    num_shards: The total number of shards, e.g., 100.
    extension: The extension of the file, e.g., ".sst".

  Returns:
    A sharded filename, e.g., "my_file-00000-of-00100.sst".
  """
  return f"{filename}-{shard:05d}-of-{num_shards:05d}{extension}"


def _match_sharded_path(
    path: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
  """Matches the parts of a sharded path.

  Examples:
    - "my_file@*.rio" -> ("my_file", "*", ".rio")
    - "my_file@100.rio" -> ("my_file", "100", ".rio")
    - "my_file" -> None

  Args:
    path: The path to match.

  Returns:
    A tuple of (prefix, num_shards_or_star, extension) if the path is sharded,
    None otherwise.
  """
  matches = [
      re.match(r"^(.*)@(\*|\d+)(\..*)?$", path),
  ]
  for match in matches:
    if match:
      prefix = match.group(1)
      num_shards_or_star = match.group(2)
      extension = match.group(3)
      return prefix, num_shards_or_star, extension

    return None


def shard_path_to_glob(path: str) -> str:
  """Creates a glob matching a sharded set of files from a sharded path.

  Args:
    path: A sharded file path, e.g., "my_file@*.rio" or, "my_file@100.rio".

  Returns:
    A glob pattern that matches all shards of the file, e.g.,
    "my_file-?????-of-00100.rio".
  """
  # If @*, use the generic shard pattern.
  # If @<number>, use this number in the shard pattern. Handle extensions.
  match_result = _match_sharded_path(path)
  if match_result:
    prefix, num_shards_or_star, extension = match_result
    extension = extension if extension else ""
    if num_shards_or_star == "*":
      return f"{prefix}-?????-of-?????{extension}"
    elif len(num_shards_or_star) == 5:
      return f"{prefix}-?????-of-{num_shards_or_star}{extension}"
    else:
      num_shards = int(num_shards_or_star)
      return f"{prefix}-?????-of-{num_shards:05d}{extension}"
  else:
    return path


def parse_sharded_filename(path: str) -> Tuple[str, int | None, str]:
  """Parses the basename, number of shards, and extension of a sharded path."""
  match_result = _match_sharded_path(path)
  if match_result:
    prefix, num_shards_or_star, extension = match_result
    extension = extension if extension else ""
    if num_shards_or_star == "*":
      return prefix, None, extension
    else:
      num_shards = int(num_shards_or_star)
      return prefix, num_shards, extension
  else:
    return path, None, ""


def list_paths(
    basename: str, extension: str, allow_bq_fallback: bool = False
) -> List[str]:
  """Lists the files matching a sharded patter.

  Example:
    ("/a/b", ".c") => ["/a/b-00000-of-00002.c", "/a/b-00001-of-00002.c"]

  Args:
    basename: The base name of the sharded files, e.g., "/a/b".
    extension: The extension of the sharded files, e.g., ".c".
    allow_bq_fallback: If True, fall back to BigQuery's EXPORT DATA format which
      uses basename-*.extension if the primary pattern yields no results.

  Returns:
    A list of paths matching the sharded pattern.
  """
  shard_glob = shard_pattern_to_glob(basename, extension)
  results = sorted(tf.io.gfile.glob(shard_glob))
  if not results and allow_bq_fallback:
    results = sorted(tf.io.gfile.glob(f"{basename}-*{extension}"))
  return results


def estimate_num_node_shards(num_nodes: int) -> tuple[int, int]:
  """Estimates the number of shards to use for a given number of nodes.

  This is a heuristic to keep each shard ~10-100 MB in size.

  Args:
    num_nodes: The number of nodes to consider.

  Returns:
    The estimated number of shards, and the number of nodes per shard.
  """
  num_shards = (num_nodes // NUM_NODES_PER_SHARD) + (
      num_nodes % NUM_NODES_PER_SHARD > 0
  )
  if num_shards <= 100:
    return num_shards, NUM_NODES_PER_SHARD
  else:
    return 100, num_nodes // 100 + (num_nodes % 100 > 0)


def estimate_num_edge_shards(num_edges: int) -> tuple[int, int]:
  """Estimates the number of shards to use for a given number of edges.

  This is a heuristic to keep each shard ~10-100 MB in size.

  Args:
    num_edges: Number of edges.

  Returns:
    The estimated number of shards, and the number of edges per shard.
  """
  num_shards = (num_edges // NUM_EDGES_PER_SHARD) + (
      num_edges % NUM_EDGES_PER_SHARD > 0
  )
  if num_shards <= 100:
    return num_shards, NUM_EDGES_PER_SHARD
  else:
    return 100, num_edges // 100 + (num_edges % 100 > 0)
