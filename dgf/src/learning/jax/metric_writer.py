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

"""Writers of training metrics (e.g. logs, TensorBoard).

`clu.metric_writers` depends on TensorFlow, which is a weak dependency of DGF.
This module provides TensorFlow-free writers, and only uses CLU writers when
available. CLU writers implement `MetricWriter`, so they can be passed
directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any, Protocol

from absl import logging
from dgf.src.util.weak_dep.weak_dep_clu import metric_writers as clu_metric_writers


class MetricWriter(Protocol):
  """Writes scalar metrics. Compatible with `clu.metric_writers`."""

  def write_scalars(self, step: int, scalars: Mapping[str, Any]) -> None:
    ...

  def flush(self) -> None:
    ...

  def close(self) -> None:
    ...


class LoggingWriter:
  """Logs scalar metrics with absl logging."""

  def write_scalars(self, step: int, scalars: Mapping[str, Any]) -> None:
    values = ", ".join(f"{k}={v}" for k, v in sorted(scalars.items()))
    logging.info("[%d] %s", step, values)

  def flush(self) -> None:
    pass

  def close(self) -> None:
    pass


class MultiWriter:
  """Forwards scalar metrics to multiple writers."""

  def __init__(self, writers: Sequence[MetricWriter]):
    self._writers = list(writers)

  def write_scalars(self, step: int, scalars: Mapping[str, Any]) -> None:
    for writer in self._writers:
      writer.write_scalars(step, scalars)

  def flush(self) -> None:
    for writer in self._writers:
      writer.flush()

  def close(self) -> None:
    for writer in self._writers:
      writer.close()


def create_default_metric_writer(
    working_path: str | None, export_metrics_to_xm: bool
) -> MultiWriter:
  """Creates a writer that logs metrics, and uses CLU writers if available.

  If CLU (and TensorFlow) is available, metrics are also written as TensorBoard
  summaries in `working_path` (if set), and exported to XManager (if
  `export_metrics_to_xm`).

  Args:
    working_path: Directory of the TensorBoard summaries.
    export_metrics_to_xm: Whether to export the metrics to XManager.

  Returns:
    The metric writer.
  """
  writers: list[MetricWriter] = [LoggingWriter()]
  if not clu_metric_writers.is_available():
    return MultiWriter(writers)
  if working_path is not None:
    writers.append(
        clu_metric_writers.SummaryWriter(os.path.join(working_path, "summary"))
    )
  if export_metrics_to_xm:
    try:
      writers.append(
          clu_metric_writers.XmMeasurementsWriter(asynchronous=True)
      )
    except (ImportError, AttributeError):
      # The XManager writer is not available outside of Google.
      pass
  return MultiWriter(writers)
