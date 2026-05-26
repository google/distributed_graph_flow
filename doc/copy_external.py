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

"""Copy files from YDF repo into the doc."""

import os
import mkdocs_gen_files

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

files_to_copy = {
    "CHANGELOG.md": "changelog.md",
}

for src_path, dest_name in files_to_copy.items():
  full_src_path = os.path.join(REPO_ROOT, src_path)
  try:
    with open(full_src_path, "r", encoding="utf-8") as f_in:
      with mkdocs_gen_files.open(dest_name, "w") as f_out:
        f_out.write(f_in.read())
  except FileNotFoundError:
    print(f"Warning: Could not find source file {full_src_path} to copy.")
  except Exception as e:
    print(f"Error copying {full_src_path}: {e}")
