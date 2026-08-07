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

"""Tests for log."""

from absl.testing import absltest
from dgf.src.util import log


class LogTest(absltest.TestCase):

  def test_base(self):
    log.info("hello %s", "world")
    log.warning("hello %s", "world")

  def test_capture_logs(self):
    with log.capture_logs(log_info=True, log_warning=True) as logs:
      log.info("A")
      log.warning("B")
      log.info("C")

    self.assertListEqual(
        logs,
        [
            log.Message.info("A"),
            log.Message.warning("B"),
            log.Message.info("C"),
        ],
    )

  def test_encapsulated_capture_logs(self):
    with log.capture_logs(log_info=True) as logs1:
      log.info("A")
      with log.capture_logs(log_info=True) as logs2:
        log.info("B")
      log.info("C")

    self.assertListEqual(
        [m.text for m in logs1],
        ["A", "B", "C"],
    )
    self.assertListEqual([m.text for m in logs2], ["B"])

  def test_capture_logs_filtering(self):
    # Test that info is filtered out (default) and warning passes.
    with log.capture_logs() as logs1:
      log.info("A")
      log.warning("B")
      log.error("C")
    self.assertListEqual(
        [m.text for m in logs1],
        ["B", "C"],
    )

    # Test that warning is filtered out too.
    with log.capture_logs(log_warning=False) as logs2:
      log.info("D")
      log.warning("E")
      log.error("F")
    self.assertListEqual(
        [m.text for m in logs2],
        ["F"],
    )


if __name__ == "__main__":
  absltest.main()
