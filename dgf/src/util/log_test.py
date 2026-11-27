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

import contextlib
import io
import threading

from absl.testing import absltest
from dgf.src.util import log
from dgf.src.util import test_util


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

  def test_silence_info(self):
    with log.capture_logs(log_info=True) as logs:
      with test_util.assert_no_output(self):
        with log.silence_info():
          log.info("A")
    # The silenced messages are still recorded.
    self.assertListEqual(logs, [log.Message.info("A")])

  def test_silence_info_does_not_silence_warnings_and_errors(self):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), log.silence_info():
      log.warning("A")
      log.error("B")
    self.assertEqual(stderr.getvalue(), "[Warning] A\n[Error] B\n")

  def test_silence_info_disabled(self):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), log.silence_info(enabled=False):
      log.info("A")
    self.assertEqual(stdout.getvalue(), "A\n")

  def test_silence_info_nested(self):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      with log.silence_info():
        with log.silence_info():
          log.info("A")
        log.info("B")
      log.info("C")
    self.assertEqual(stdout.getvalue(), "C\n")

  def test_silence_info_does_not_silence_other_threads(self):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), log.silence_info():
      thread = threading.Thread(target=log.info, args=("A",))
      thread.start()
      thread.join()
      log.info("B")
    self.assertEqual(stdout.getvalue(), "A\n")

  def test_silence_info_is_restored(self):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      with self.assertRaises(ValueError):
        with log.silence_info():
          raise ValueError()
      log.info("A")
    self.assertEqual(stdout.getvalue(), "A\n")


if __name__ == "__main__":
  absltest.main()
