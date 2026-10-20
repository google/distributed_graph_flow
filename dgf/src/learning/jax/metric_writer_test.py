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

"""Tests for the metric writers."""

import os
from unittest import mock

from absl.testing import absltest
from dgf.src.learning.jax import metric_writer as metric_writer_lib


class FakeWriter:

  def __init__(self):
    self.scalars = []
    self.num_flushes = 0
    self.num_closes = 0

  def write_scalars(self, step, scalars):
    self.scalars.append((step, dict(scalars)))

  def flush(self):
    self.num_flushes += 1

  def close(self):
    self.num_closes += 1


class MetricWriterTest(absltest.TestCase):

  def test_logging_writer(self):
    with self.assertLogs(level="INFO") as logs:
      metric_writer_lib.LoggingWriter().write_scalars(5, {"b": 2.0, "a": 1.0})
    self.assertIn("[5] a=1.0, b=2.0", logs.output[0])

  def test_multi_writer(self):
    writer_1 = FakeWriter()
    writer_2 = FakeWriter()
    writer = metric_writer_lib.MultiWriter([writer_1, writer_2])
    writer.write_scalars(1, {"loss": 0.5})
    writer.flush()
    writer.close()
    for sub_writer in [writer_1, writer_2]:
      self.assertEqual(sub_writer.scalars, [(1, {"loss": 0.5})])
      self.assertEqual(sub_writer.num_flushes, 1)
      self.assertEqual(sub_writer.num_closes, 1)

  def test_default_writer_without_clu(self):
    working_path = self.create_tempdir().full_path
    with mock.patch.object(
        metric_writer_lib.clu_metric_writers,
        "is_available",
        return_value=False,
    ):
      writer = metric_writer_lib.create_default_metric_writer(
          working_path=working_path, export_metrics_to_xm=True
      )
    writer.write_scalars(1, {"loss": 0.5})
    writer.flush()
    self.assertFalse(os.path.exists(os.path.join(working_path, "summary")))

  def test_default_writer_with_clu(self):
    working_path = self.create_tempdir().full_path
    writer = metric_writer_lib.create_default_metric_writer(
        working_path=working_path, export_metrics_to_xm=False
    )
    writer.write_scalars(1, {"loss": 0.5})
    writer.flush()
    writer.close()
    self.assertNotEmpty(os.listdir(os.path.join(working_path, "summary")))


if __name__ == "__main__":
  absltest.main()
