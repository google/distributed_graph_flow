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

from dgf.src.util.weak_dep.base import LazyModule

tfgnn = LazyModule(
    local_name="tfgnn",
    import_path="tensorflow_gnn",
    library_name="TensorFlow GNN",
    pip="tensorflow-gnn",
    bazel_rule="//third_party/py/tensorflow_gnn",
)

tf_gnn_proto = LazyModule(
    local_name="tf_gnn_proto",
    import_path="tensorflow_gnn.proto.graph_schema_pb2",
    library_name="TensorFlow GNN",
    pip="tensorflow-gnn",
    bazel_rule="//third_party/py/tensorflow_gnn/proto",
)
