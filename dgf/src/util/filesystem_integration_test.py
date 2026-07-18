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

r"""Run filesystem integration test on GCS or CNS.

Usage examples:

1. Run on GCS:
   blaze build //third_party/py/dgf/src/util:filesystem_integration_test && \
   blaze-bin/third_party/py/dgf/src/util/filesystem_integration_test \
     --test_dir=gs://graph-flow/integration_test --alsologtostderr

2. Run on CNS:
   blaze build //third_party/py/dgf/src/util:filesystem_integration_test && \
   blaze-bin/third_party/py/dgf/src/util/filesystem_integration_test \
     --test_dir=/cns/is-d/home/gbm/tmp/ttl=365d/integration_test --alsologtostderr
"""

import os
from absl import flags
from absl.testing import absltest
from dgf.src.util import filesystem as fs

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "test_dir",
    None,
    "Directory path for testing (GCS gs://... or CNS /cns/... path).",
    required=True,
)


class FilesystemIntegrationTest(absltest.TestCase):

  def test_base(self):
    dir_path = FLAGS.test_dir
    file_path = os.path.join(dir_path, "test.txt")
    fs.makedirs(dir_path)
    content = "hello world"
    with fs.open_write(file_path) as f:
      f.write(content)
    with fs.open_read(file_path) as f:
      read_content = f.read()
    self.assertEqual(read_content, content)


if __name__ == "__main__":
  absltest.main()
