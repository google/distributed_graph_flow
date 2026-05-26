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

"""Utilities to handle proto.

Note: Keeping all the proto io and parsing here, will make open-sourcing easier.
"""

from typing import Any, Callable
from dgf.src.util import filesystem
from google.protobuf import text_format


def parse_text_proto(content: str, proto_class: Any) -> Any:
  proto_object = proto_class()
  text_format.Parse(content, proto_object)
  return proto_object


def read_text_proto(
    path: str,
    proto_class: Any,
    process_fn: Callable[[str], str] = lambda x: x,
) -> Any:
  """Read a proto from disk in text format."""
  proto_object = proto_class()
  with filesystem.open_read(path) as f:
    text_format.Parse(process_fn(f.read()), proto_object)
  return proto_object


def write_text_proto(path: str, proto: Any):
  """Writes a proto to disk in text format."""
  with filesystem.open_write(path) as f:
    f.write(text_format.MessageToString(proto))
