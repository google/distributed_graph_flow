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

"""Base buildable configuration object.
"""

import abc
import dataclasses
import json
from typing import Any, Protocol, TypeVar

from absl import logging
from dgf.src.util import filesystem

_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config(Protocol[_T]):  # pyrefly: ignore[bad-class-definition]
  """Base class for configurations that can instantiate objects.

  Subclasses should, at least, implement `name`, `make`. Other functions such as
  `to_dict` and the load/save variants may be customized as needed but the base
  should provide a decent implementation for most cases. There is
  an expectation that the configuration class should be serializable to JSON.

  It is encouraged that implementations implement validation as appropriate.
  """

  @abc.abstractmethod
  def make(self) -> _T:
    """Creates an instantiation of _T according to the configuration."""

  @abc.abstractmethod
  def name(self) -> str:
    """Returns the name of the object."""

  def to_dict(self) -> dict[str, Any]:
    """Convert the configuration to a dictionary."""
    params = dataclasses.asdict(self)  # pyrefly: ignore[bad-argument-type]
    params["name"] = self.name()
    return params

  def json_save(self, path: str) -> None:
    """Save to a human-readable format."""
    try:
      with filesystem.open_write(path, binary=False) as f:
        json.dump(self.to_dict(), f, indent=2)
    except Exception as e:
      logging.error("Failed to write file %s", path)
      raise e

  @classmethod
  def json_load(cls, path: str) -> None:
    """Load from a human-readable format.

    Args:
      path: The path to the materialized configuration.

    Returns:
      An instantiated configuration object.
    """
    try:
      with filesystem.open_read(path, binary=False) as f:
        cfg_str = f.read()
    except Exception as e:
      logging.error("Failed to read file %s", path)
      raise e

    params = json.loads(cfg_str)
    if "name" in params:
      del params["name"]
    return cls(**params)  # pyrefly: ignore[bad-return]
