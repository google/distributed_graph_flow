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

import concurrent.futures
from typing import List, Optional, Sequence
from absl import logging
from etils import epath
import fsspec
from google.cloud import storage


def is_gcs_path(path: str) -> bool:
  return path.startswith("gs://")


def _unnormalize_io_path(path: str) -> str:
  return path


def glob(pattern: str) -> List[str]:
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


def exists(path: str) -> bool:
  return epath.Path(path).exists()


def create_gcs_bucket(
    bucket_name: str,
    client: Optional[storage.Client] = None,
    project: Optional[str] = None,
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
