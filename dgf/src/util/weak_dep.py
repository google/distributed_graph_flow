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

"""Utilities for handling optional/weak dependencies in DGF.

A weak dependency is a third-party library used by some DGF functions. Unlike
core dependencies, weak dependencies are not installed or linked by default
when importing `dgf`.

Instead, DGF attempts to load these dependencies dynamically when a function
requiring them is called. If a weak dependency is not available, an informative
error message guides the user on how to install or link it.

Example Scenario:

pip install dgf

import dgf

dgf.some_function_using_numpy()
# This works because numpy is a core dependency.

dgf.some_function_using_pytorch()
# Error: The function "some_function_using_pytorch" requires PyTorch, which is
# a weak dependency. Installing PyTorch (e.g., `pip install torch`)
# or linking it manually (e.g., `//third_party/py/torch`).

# Install pytorch manually
pip install torch

# Now this works
dgf.some_function_using_pytorch()
"""

import importlib
from typing import Any, Optional


def _error_message(library_name: str, pip: str, bazel_rule: str) -> str:
  return (
      f"This feature requires the {library_name} library to be installed"
      " manually (it is a weak dependency). Install it with"
      f" `pip install {pip}` or link the Bazel target `{bazel_rule}`."
  )


def _import_weak_dependency(
    import_path: str,
    library_name: str,
    pip: str,
    bazel_rule: str,
    attribute_name: Optional[str] = None,
) -> Any:
  try:
    module = importlib.import_module(import_path)
    if attribute_name is not None:
      return getattr(module, attribute_name)
    return module
  except (ImportError, AttributeError) as e:
    raise RuntimeError(_error_message(library_name, pip, bazel_rule)) from e


def import_ogb_nodeproppred() -> Any:
  return _import_weak_dependency(
      import_path="ogb.nodeproppred",
      library_name="OGB",
      pip="ogb",
      bazel_rule="//third_party/py/ogb",
  )


def has_ogb_nodeproppred() -> bool:
  try:
    import_ogb_nodeproppred()
    return True
  except RuntimeError:
    return False


def import_tfgnn() -> Any:
  """Same as import tensorflow_gnn."""
  return _import_weak_dependency(
      import_path="tensorflow_gnn",
      library_name="TensorFlow GNN",
      pip="tensorflow-gnn",
      bazel_rule="//third_party/py/tensorflow_gnn",
  )


def import_tf_gnn_proto() -> Any:
  """Same as from tensorflow_gnn import proto."""
  return _import_weak_dependency(
      import_path="tensorflow_gnn.proto",
      library_name="TensorFlow GNN",
      pip="tensorflow-gnn",
      bazel_rule="//third_party/py/tensorflow_gnn",
  )
