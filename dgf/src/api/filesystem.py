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

"""GraphFlow unified filesystem API."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error


from dgf.src.util.filesystem import create_gcs_bucket
from dgf.src.util.filesystem import exists
from dgf.src.util.filesystem import glob
from dgf.src.util.filesystem import is_gcs_path
from dgf.src.util.filesystem import makedirs
from dgf.src.util.filesystem import open_read
from dgf.src.util.filesystem import remove_paths
from dgf.src.util.filesystem import rename
from dgf.src.util.filesystem import rmtree
