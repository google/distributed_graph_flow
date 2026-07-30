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

import importlib
import types
from typing import Any, Optional


def _error_message(library_name: str, pip: str, bazel_rule: str) -> str:
  return (
      f"This feature requires the {library_name} library to be"
      " installed\nmanually (it is a weak dependency). Install it with\n`pip"
      f" install {pip}` (or `pip install dgf[all]`), or link the Bazel target"
      f" `{bazel_rule}`."
  )


class LazyModule(types.ModuleType):
  """A lazy-import module proxy."""

  def __init__(
      self,
      local_name: str,
      import_path: str,
      library_name: str,
      pip: str,
      bazel_rule: str,
  ):
    self._local_name = local_name
    self._import_path = import_path
    self._library_name = library_name
    self._pip = pip
    self._bazel_rule = bazel_rule
    super().__init__(local_name)

  def _load(self):
    try:
      return importlib.import_module(self._import_path)
    except ImportError as e:
      raise RuntimeError(
          _error_message(self._library_name, self._pip, self._bazel_rule)
      ) from e

  def is_available(self) -> bool:
    try:
      importlib.import_module(self._import_path)
      return True
    except ImportError:
      return False

  def __getattr__(self, item: str) -> Any:
    return getattr(self._load(), item)

  def __dir__(self) -> list[str]:
    return dir(self._load())
