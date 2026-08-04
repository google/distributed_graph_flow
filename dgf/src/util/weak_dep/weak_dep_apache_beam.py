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

from __future__ import annotations

import typing

from dgf.src.util.weak_dep.base import LazyModule

beam = LazyModule(
    local_name="beam",
    import_path="apache_beam",
    library_name="Apache Beam",
    pip="apache-beam",
    bazel_rule="//third_party/py/apache_beam",
)

if typing.TYPE_CHECKING:
  import apache_beam as _apache_beam

  DoFn = _apache_beam.DoFn
  CombineFn = _apache_beam.CombineFn
  PTransform = _apache_beam.PTransform
  CoderBase = _apache_beam.coders.Coder
  ptransform_fn = _apache_beam.ptransform_fn
else:
  DoFn = beam.DoFn if getattr(beam, "is_available", lambda: True)() else object
  CombineFn = (
      beam.CombineFn
      if getattr(beam, "is_available", lambda: True)()
      else object
  )
  PTransform = (
      beam.PTransform
      if getattr(beam, "is_available", lambda: True)()
      else object
  )
  CoderBase = (
      beam.coders.Coder
      if getattr(beam, "is_available", lambda: True)()
      else object
  )

  def _fallback_ptransform_fn(*args, **kwargs):
    def decorator(func):
      return func

    if len(args) == 1 and callable(args[0]) and not kwargs:
      return args[0]
    return decorator

  ptransform_fn = (
      beam.ptransform_fn
      if getattr(beam, "is_available", lambda: True)()
      else _fallback_ptransform_fn
  )
