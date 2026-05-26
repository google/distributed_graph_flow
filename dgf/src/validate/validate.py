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

"""Validation utility."""

import dataclasses
import enum
import logging
from typing import Sequence


class Severity(enum.Enum):
  """The severity type of a message."""

  INFO = "INFO"
  WARNING = "WARNING"
  ERROR = "ERROR"


@dataclasses.dataclass
class Issue:
  """A message shown to the user."""

  severity: Severity
  text: str

  @classmethod
  def error(cls, text: str) -> "Issue":
    return Issue(severity=Severity.ERROR, text=text)

  @classmethod
  def warning(cls, text: str) -> "Issue":
    return Issue(severity=Severity.WARNING, text=text)


def print_and_raise(
    issues: Sequence[Issue],
    *,
    raise_on_error: bool = True,
    raise_on_warning: bool = True,
):
  """Prints the issues, and possibly raises an exception.

  Args:
    issues: Issues to print.
    raise_on_error: If true, raises an exception if there are any error
      messages.
    raise_on_warning: If true, raises an exception if there are any warning
      messages.
  """

  num_warnings = 0
  num_errors = 0
  for issue in issues:
    if issue.severity == Severity.WARNING:
      num_warnings += 1
    if issue.severity == Severity.ERROR:
      num_errors += 1
    print_msg = f"[{issue.severity.value}] {issue.text}"
    print(print_msg)
    logging.info("%s", print_msg)
  if raise_on_error and num_errors:
    raise ValueError(f"{num_errors} errors found")
  if raise_on_warning and num_warnings:
    raise ValueError(f"{num_warnings} warnings found")
