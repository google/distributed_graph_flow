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

"""Lazy wrapper for TensorFlow Text."""

from __future__ import annotations

from dgf.src.util.weak_dep.base import LazyModule

tf_text = LazyModule(
    local_name="tf_text",
    import_path="tensorflow_text",
    library_name="TensorFlow Text",
    pip="tensorflow-text",
    bazel_rule="//third_party/py/tensorflow_text",
)
