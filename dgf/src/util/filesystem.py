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

"""A basic filesystem compatible with local and gcp.
"""

from collections.abc import Sequence
import concurrent.futures
import os
import time
from typing import List, Optional, Sequence, Tuple
from absl import logging
from etils import epath
import fsspec
from google.cloud import storage


def is_gcs_path(path: str) -> bool:
  """Returns True if the path is a Google Cloud Storage (GCS) path."""
  return path.startswith("gs://")


def is_url(url_or_path: str) -> bool:
  """Returns True if `url_or_path` is an http(s) URL instead of a path."""
  return url_or_path.startswith("http://") or url_or_path.startswith("https://")


def _parse_gcs_path(path: str) -> Tuple[str, str]:
  """Parses a gs:// path into (bucket_name, blob_path)."""
  gcs_path = path.replace("gs://", "", 1)
  parts = gcs_path.split("/", 1)
  bucket_name = parts[0]
  blob_path = parts[1] if len(parts) > 1 else ""
  return bucket_name, blob_path


def _unnormalize_io_path(path: str) -> str:
  return path


def glob(pattern: str) -> list[str]:
  """Returns a list of files and directories matching a pattern.

  Args:
    pattern: The pattern to match. Can be a local, CNS, or GCS path.

  Returns:
    A list of paths matching the pattern.
  """
  pattern_path = epath.Path(pattern)
  base_dir = pattern_path.parent
  glob_pattern = pattern_path.name
  return [_unnormalize_io_path(str(p)) for p in base_dir.glob(glob_pattern)]


def open_read(path: str, binary: bool = False):
  """Opens a file for reading and return a python file handle.

  Args:
    path: The path to the file to open. Can be a local, CNS, or GCS path.
    binary: If True, the file is opened in binary mode ('rb'). Otherwise, it's
      opened in text mode ('r').

  Returns:
    A file-like object for reading.
  """
  if is_gcs_path(path):
    # TODO(gbm): Check other possible fixes
    # Direct GCS blob open for GCS reads (epath unreachable from GCP Dataflow runner).
    # Note: blob.open() returns a streaming BlobReader which downloads in chunks,
    # so it does NOT load the entire file into memory at once (safe for large files).
    # We also add exponential-backoff retries to handle transient GCS network timeouts.
    client = storage.Client()
    bucket_name, blob_path = _parse_gcs_path(path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Open the GCS blob as a streaming file-like object.
    max_retries = 5
    for attempt in range(max_retries):
      try:
        return blob.open("rb" if binary else "r")
      except Exception as e:  # pylint: disable=broad-except
        if attempt < max_retries - 1:
          wait = 2**attempt  # Exponential backoff: 1s, 2s, 4s, 8s.
          logging.warning(
              "GCS open attempt %d/%d failed for %s: %s. Retrying in %ds.",
              attempt + 1,
              max_retries,
              path,
              e,
              wait,
          )
          time.sleep(wait)
        else:
          raise  # All retries exhausted; propagate the exception.

  # For non-GCS paths (local, CNS, etc.), delegate to etils.epath.
  return epath.Path(path).open("rb" if binary else "r")


def open_write(path: str, binary: bool = False):
  """Opens a file for writing and returns a python file handle.

  Args:
    path: The path to the file to open. Can be a local, CNS, or GCS path.
    binary: If True, the file is opened in binary mode ('wb'). Otherwise, it's
      opened in text mode ('w').

  Returns:
    A file-like object for writing.
  """
  return epath.Path(path).open("wb" if binary else "w")


def write_text(
    file_path: str, content: str, project: Optional[str] = None
) -> None:
  """Writes string content to a file (local, CNS, or GCS).

  Args:
    file_path: Path to the destination file (must include file name).
    content: String content to write.
    project: Optional GCP project ID to use for GCS operations.

  Raises:
    ValueError: If file_path ends with '/' or points to a bare GCS bucket root.
  """
  if file_path.endswith("/"):
    raise ValueError(
        f"file_path must point to a file, got directory path: {file_path}"
    )
  if is_gcs_path(file_path):
    bucket_name, blob_path = _parse_gcs_path(file_path)
    if not blob_path:
      raise ValueError(
          f"file_path must include a blob name, got bucket root: {file_path}"
      )
    client = storage.Client(project=project)
    client.bucket(bucket_name).blob(blob_path).upload_from_string(content)
  else:
    target_path = epath.Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content)


def exists(path: str, project: Optional[str] = None) -> bool:
  """Returns True if the file or directory exists.

  Args:
    path: Path to check (local, CNS, or GCS file/prefix).
    project: Optional GCP project ID to use for GCS operations.

  Returns:
    True if the file or directory/prefix exists, False otherwise.
  """
  if is_gcs_path(path):
    client = storage.Client(project=project)
    bucket_name, blob_path = _parse_gcs_path(path)
    bucket = client.bucket(bucket_name)
    # Check if path points to an exact file blob (skip for bucket root or dir).
    if (
        blob_path
        and not blob_path.endswith("/")
        and bucket.blob(blob_path).exists()
    ):
      return True
    # Otherwise, check if any objects exist under this directory prefix.
    prefix = f"{blob_path.rstrip('/')}/" if blob_path else ""
    return any(bucket.list_blobs(prefix=prefix, max_results=1))
  return epath.Path(path).exists()


def is_dir(path: str) -> bool:
  """Returns True if the path exists and is a directory."""
  return epath.Path(path).is_dir()


def copy_local_dir(
    local_src_dir: str, dst_dir: str, project: Optional[str] = None
) -> None:
  """Recursively copies a local directory to dst_dir (local, CNS, or GCS).

  Args:
    local_src_dir: Path to an existing local directory to copy from.
    dst_dir: Destination directory path (local, CNS, or GCS).
    project: Optional GCP project ID to use for GCS operations.

  Raises:
    ValueError: If local_src_dir is a GCS path or not an existing local
      directory.
  """
  if is_gcs_path(local_src_dir) or not os.path.isdir(local_src_dir):
    raise ValueError(
        "local_src_dir must be an existing local directory, got:"
        f" {local_src_dir}"
    )
  if is_gcs_path(dst_dir):
    client = storage.Client(project=project)
    bucket_name, prefix = _parse_gcs_path(dst_dir)
    prefix = prefix.rstrip("/")
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_src_dir):
      for file in files:
        local_path = os.path.join(root, file)
        rel_path = os.path.relpath(local_path, local_src_dir)
        blob_path = f"{prefix}/{rel_path}" if prefix else rel_path
        bucket.blob(blob_path).upload_from_filename(local_path)
  else:
    for root, _, files in os.walk(local_src_dir):
      for file in files:
        local_path = os.path.join(root, file)
        rel_path = os.path.relpath(local_path, local_src_dir)
        target_path = epath.Path(dst_dir) / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        epath.Path(local_path).copy(target_path)


def create_gcs_bucket(
    bucket_name: str,
    client: storage.Client | None = None,
    project: str | None = None,
):
  """Creates a GCS bucket.

  Args:
    bucket_name: The name of the bucket to create.
    client: Optional GCS client to use. If None, an attempt will be made to
      create a default client from the environment.
    project: Optional GCP project to use. If None, the default project will be
      used.

  Returns:
    The created bucket object.
  """
  logging.info("Attempting to creating GCS bucket: %s", bucket_name)
  try:
    if client is None:
      logging.info("Creating GCS client...")
      client = storage.Client(project=project)
    bucket = client.create_bucket(bucket_name, project=project)

    return bucket
  except Exception as e:  # pylint: disable=broad-except
    print(f"Error creating GCS bucket: {bucket_name}, {e}.")


def makedirs(path: str, parents: bool = True, exist_ok: bool = True):
  """Creates directories if it does not exist.

  Note: If using with GCS, this will create the subpath structure but cannot
  create a new bucket.

  Args:
    path: The path to create.
    parents: Create parent directories if they do not exist. Defaults to True.
    exist_ok: Do not crash if the directory already exists. Defaults to True (no
      crash). Useful in pre-emption tolerance use-cases for ML.

  Returns:
    A file object.
  """

  epath.Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def rmtree(path: str):
  """Recursively removes a directory and its contents."""
  epath.Path(path).rmtree()


# TODO(bmayer,gbm): Figure out if we need these or can move to epath.
def remove_paths(paths: Sequence[str], fail_if_absent: bool = True):
  """Removes all the files in parallel."""

  def _remove_if_exists(path):
    fs, clean_path = fsspec.core.url_to_fs(path)
    if not fail_if_absent and not fs.exists(clean_path):
      logging.info("Path not found, skipping removal: %s", path)
      return
    fs.rm(clean_path)

  with concurrent.futures.ThreadPoolExecutor() as executor:
    list(executor.map(_remove_if_exists, paths))


def rename(src: str, dst: str):
  """Renames (moves) a file or directory from old_path to new_path."""

  fs_src, path_src = fsspec.core.url_to_fs(src)
  fs_dst, path_dst = fsspec.core.url_to_fs(dst)
  if type(fs_src) != type(fs_dst):
    raise ValueError("Source and destination must be on the same filesystem.")
  fs_src.rename(path_src, path_dst)


def replace(src: str, dst: str):
  """Atomically replaces destination with source."""
  epath.Path(src).replace(dst)
