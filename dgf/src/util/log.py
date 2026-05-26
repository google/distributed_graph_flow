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

"""Uniform display and controls the logs displayed on the different surfaces.

Allow for nice logging on colab and terminal.
To use instead of "logging" or "print" for user code.
"""

import io
import sys
from typing import Any
from absl import logging

# TODO(gbm): Add support for log levels, also for c++ code.


def info(msg: str, *args: Any) -> None:
  """Print an info message.

  Usage example:
    info("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  print(msg % args, flush=True)
  logging.info(msg, *args)


def warning(msg: str, *args: Any) -> None:
  """Print a warning message.

  Usage example:
    warning("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  # TODO(gbm): Add coloring
  print("[Warning]", msg % args, flush=True, file=sys.stderr)
  logging.warning(msg, *args)


def is_direct_output(stream=sys.stdout):
  """Checks if the output stream redirects to the shell/console directly.

  This function checks if the given stream is a terminal device. It handles
  common stream wrappers to determine if the underlying file descriptor
  corresponds to stdout or stderr.

  Args:
    stream: The output stream to check. Defaults to sys.stdout.

  Returns:
    True if the stream is considered to be direct output to the console,
    False otherwise.
  """

  if stream.isatty():
    return True
  if isinstance(stream, io.TextIOWrapper):
    return is_direct_output(stream.buffer)
  if isinstance(stream, io.BufferedWriter):
    return is_direct_output(stream.raw)
  if isinstance(stream, io.FileIO):
    return stream.fileno() in [1, 2]
  return False
