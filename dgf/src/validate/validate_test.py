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

import io
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.validate import validate as validate_lib


class ValidateTest(parameterized.TestCase):

  def test_print_message(self):
    with mock.patch("sys.stdout", io.StringIO()) as mock_stdout:
      validate_lib.print_and_raise(
          [
              validate_lib.Issue(validate_lib.Severity.WARNING, "hello"),
              validate_lib.Issue(validate_lib.Severity.ERROR, "world"),
          ],
          raise_on_error=False,
          raise_on_warning=False,
      )
      self.assertEqual(
          mock_stdout.getvalue(),
          """\
[WARNING] hello
[ERROR] world
""",
      )

  def test_print_and_raise_warning(self):
    with self.assertRaisesRegex(ValueError, "1 warnings found"):
      validate_lib.print_and_raise(
          [validate_lib.Issue(validate_lib.Severity.WARNING, "hello")],
          raise_on_error=False,
          raise_on_warning=True,
      )

  def test_print_and_raise_errors(self):
    with self.assertRaisesRegex(ValueError, "1 errors found"):
      validate_lib.print_and_raise(
          [validate_lib.Issue(validate_lib.Severity.ERROR, "hello")],
          raise_on_error=True,
          raise_on_warning=False,
      )


if __name__ == "__main__":
  absltest.main()
