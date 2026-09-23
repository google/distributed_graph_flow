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

import importlib
from dgf.src.util.weak_dep.base import _error_message
from dgf.src.util.weak_dep.base import LazyModule


class _LazyAiplatform(LazyModule):
  """Lazy module proxy supporting both Google3 and OSS aiplatform imports."""

  def _load(self):
    # In Google3, the SDK is at google.cloud.aiplatform.aiplatform
    try:
      return importlib.import_module("google.cloud.aiplatform.aiplatform")
    except (ImportError, AttributeError):
      pass

    # In OSS, the SDK is at google.cloud.aiplatform
    try:
      return importlib.import_module("google.cloud.aiplatform")
    except ImportError as e:
      raise RuntimeError(
          _error_message(self._library_name, self._pip, self._bazel_rule)
      ) from e

  def is_available(self) -> bool:
    try:
      self._load()
      return True
    except (ImportError, RuntimeError):
      return False


aiplatform = _LazyAiplatform(
    local_name="aiplatform",
    import_path="google.cloud.aiplatform",
    library_name="google-cloud-aiplatform",
    pip="google-cloud-aiplatform",
    bazel_rule="//third_party/py/google/cloud/aiplatform",
)

