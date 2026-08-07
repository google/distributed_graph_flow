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

import contextlib
import dataclasses
import enum
import io
import sys
from typing import Any
from absl import logging

# TODO(gbm): Add support for log levels, also for c++ code.


class Severity(enum.Enum):
  """The severity type of a message."""

  INFO = "INFO"
  WARNING = "WARNING"
  ERROR = "ERROR"


@dataclasses.dataclass
class Message:
  """A message shown to the user."""

  severity: Severity
  text: str

  @classmethod
  def info(cls, text: str) -> "Message":
    return Message(severity=Severity.INFO, text=text)

  @classmethod
  def error(cls, text: str) -> "Message":
    return Message(severity=Severity.ERROR, text=text)

  @classmethod
  def warning(cls, text: str) -> "Message":
    return Message(severity=Severity.WARNING, text=text)


@dataclasses.dataclass
class _Capturer:
  logs: list[Message]
  log_info: bool
  log_warning: bool


_ACTIVE_LOGGERS: list[_Capturer] = []


@contextlib.contextmanager
def capture_logs(log_info: bool = False, log_warning: bool = True):
  """Captures logs in a context block.

  Usage example:
    with log.capture_logs() as logs:
      log.info("Hello %s", "world")

    print(f"Captured {len(logs)} logs.")
  """

  if _ACTIVE_LOGGERS:
    logging.warning("Nested capture_logs detected.")

  logs: list[Message] = []
  capturer = _Capturer(logs=logs, log_info=log_info, log_warning=log_warning)
  _ACTIVE_LOGGERS.append(capturer)
  try:
    yield logs
  finally:
    _ACTIVE_LOGGERS.remove(capturer)


def _record_log(message: Message):
  for capturer in _ACTIVE_LOGGERS:
    if message.severity == Severity.INFO and not capturer.log_info:
      continue
    if message.severity == Severity.WARNING and not capturer.log_warning:
      continue
    capturer.logs.append(message)


def info(msg: str, *args: Any) -> None:
  """Print an info message.

  Usage example:
    info("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  text = msg % args if args else msg
  print(text, flush=True)
  logging.info(msg, *args)
  _record_log(Message.info(text))


def warning(msg: str, *args: Any) -> None:
  """Print a warning message.

  Usage example:
    warning("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  # TODO(gbm): Add coloring e.g., colorama.
  text = msg % args if args else msg
  print("[Warning]", text, flush=True, file=sys.stderr)
  logging.warning(msg, *args)
  _record_log(Message.warning(text))


def error(msg: str, *args: Any) -> None:
  """Print an error message.

  Usage example:
    error("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """
  text = msg % args if args else msg
  print("[Error]", text, flush=True, file=sys.stderr)
  logging.error(msg, *args)
  _record_log(Message.error(text))


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
