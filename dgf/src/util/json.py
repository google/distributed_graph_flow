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

"""Utilities to handle jsons."""

import json
from typing import Any
from dgf.src.util import filesystem


def read_json(path: str) -> Any:
  """Reads a json from a file.

  Args:
    path: The path to the file.

  Returns:
    The json object.
  """
  with filesystem.open_read(path) as f:
    return json.load(f)


def write_json(path: str, value: Any):
  """Writes a json to a file.

  Args:
    path: The path to the file.
    json: The json object.
  """
  with filesystem.open_write(path) as f:
    json.dump(value, f, indent=2)
