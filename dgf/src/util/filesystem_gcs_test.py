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

r"""Run filesystem test on gcs.

Usage example:

Create find an existing a GC project and update "kTestBucket" bellow.

gcloud auth application-default login
gcloud config set project <project id>
blaze build //third_party/py/dgf/src/util:filesystem_gcs_test && \
blaze-bin/third_party/py/dgf/src/util/filesystem_gcs_test --alsologtostderr
"""

import os
from absl.testing import absltest
from dgf.src.util import filesystem as fs

TEST_BUCKET = "graph-flow"


class FilesystemGcsTest(absltest.TestCase):

  def test_base(self):
    dir_path = f"gs://{TEST_BUCKET}/test_dir"
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
