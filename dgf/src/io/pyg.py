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

"""Conversion to PyTorch Geometric related graph objects."""

import typing

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.util.weak_dep.base import LazyModule
import numpy as np

torch = LazyModule(
    local_name="torch",
    import_path="torch",
    library_name="PyTorch",
    pip="torch",
    bazel_rule="//third_party/py/torch:pytorch",
)

torch_geometric = LazyModule(
    local_name="torch_geometric",
    import_path="torch_geometric",
    library_name="PyTorch Geometric",
    pip="torch_geometric",
    bazel_rule="//third_party/py/torch_geometric:torch_geometric",
)

if typing.TYPE_CHECKING:
  import torch_geometric as _torch_geometric

  HeteroData = _torch_geometric.data.HeteroData
else:
  HeteroData = (
      torch_geometric.data.HeteroData
      if getattr(torch_geometric, "is_available", lambda: True)()
      else object
  )


def graph_to_pyg_data(
    dgf_graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
) -> HeteroData:
  """Converts a normalized DGF InMemoryGraph to a PyG HeteroData object.

  Usage example:

  ```python
    data = dgf.convert.graph_to_pyg_data(graph, schema)
  ```

  Args:
    dgf_graph: The input graph in memory.
    schema: The schema of the graph.

  Returns:
    A PyG HeteroData object representing the graph.
  """
  data = torch_geometric.data.HeteroData()

  for nodeset_name, nodeset_schema in schema.node_sets.items():
    nodeset_value = dgf_graph.node_sets[nodeset_name]
    data[nodeset_name].num_nodes = nodeset_value.num_nodes

    x_features = []
    # Sort feature names to ensure deterministic order if multiple exist
    for feature_name in sorted(nodeset_schema.features.keys()):
      feature_value = nodeset_value.features[feature_name]

      # PyTorch only supports numerical / boolean arrays.
      feature_tensor = torch.from_numpy(feature_value)
      if feature_tensor.ndim == 1:
        feature_tensor = feature_tensor.unsqueeze(1)
      x_features.append(feature_tensor)

    if x_features:
      data[nodeset_name].x = torch.cat(x_features, dim=-1)

  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    edgeset_value = dgf_graph.edge_sets[edgeset_name]
    src = edgeset_schema.source
    dst = edgeset_schema.target
    edge_type = (src, edgeset_name, dst)

    adjacency = edgeset_value.adjacency
    data[edge_type].edge_index = torch.from_numpy(adjacency).long()

    edge_attrs = []
    for feature_name in sorted(edgeset_schema.features.keys()):
      feature_value = edgeset_value.features[feature_name]

      feature_tensor = torch.from_numpy(feature_value)
      if feature_tensor.ndim == 1:
        feature_tensor = feature_tensor.unsqueeze(1)
      edge_attrs.append(feature_tensor)

    if edge_attrs:
      data[edge_type].edge_attr = torch.cat(edge_attrs, dim=-1)

  return data
