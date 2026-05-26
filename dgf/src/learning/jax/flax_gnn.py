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

"""A simple flax wrapper for GNN modeling.

This is a mid-level API and is not necessary to use with DFG's lower-level APIs.
The flax GNN module provides an interface guide for common GNN patterns:

1) Learnable Initial Node State Layer
2) A series of Graph Convolutions
3) A user-configurable set of heads applied to the graph convolution
representation.
"""

from typing import Mapping, Optional
from dgf.src.learning.jax import common
import flax.linen as nn
import frozendict

BuildableModule = common.BuildableModule


# TODO(bmayer): Change API for from_config to take buildable modules for
# initial_node_state_fn and heads.
# TODO(bmayer): Add this to the public API with appropriate name.
def from_config(
    gnn_config: BuildableModule,
    initial_node_state_fn: Optional[nn.Module] = None,
    heads: Optional[Mapping[str, BuildableModule]] = None,
) -> 'GNN':
  """Build a flax GNN model from configs.

  Args:
    gnn_config: A buildable module that returns a GNN module.
    initial_node_state_fn: An optional learnable model that takes an input graph
      and returns an obejct the `gnn` can consume. If not provided (default) the
      input graph will be passed directly to the `gnn` layer.
    heads: Optional mapping to task-specific (prediction, regression, etc.)
      heads (nn.Modules) that are key by name. The call function of each head
      must accept a boolean `training` named argument, accept the output of the
      `gnn` module and generate a pytree.

  Returns:
    A mid-level GNN flax object.
  """

  return GNN(
      gnn=gnn_config.make(),
      initial_node_state_fn=initial_node_state_fn,
      heads=heads,
  )


class GNN(nn.Module):
  """Mid-level API FLAX GNN encapsulating initial node state, convs and heads.

  **Note** Both the gnn and initial_node_state_fn are expected to consume a
  graph object and have a boolean `training` variable parameter defined in the
  __call__ function. A good default value is `False`.

  **Note** `heads must be passed as an immutable mapping so it can be hashed
  when used in a jitted function. This constraint does not apply to un-jitted
  workflows. For example, pass heads as a flax.core.FrozenDict or
  immutabledict.immutabledict.

  The __call__ function will return a flax.core.FrozenDict of named PyTrees. The
  output of the gnn module is keyed by `gnn`, a reserved keyword (don't name a
  head `gnn`). If provided, the output of each head will likewise be keyed by
  the key of the head.

  Attributes:
    gnn: A graph convolutional model.
    initial_node_state_fn: An optional learnable model that takes an input graph
      and returns an obejct the `gnn` can consume. If not provided (default) the
      input graph will be passed directly to the `gnn` layer.
    heads: Optional mapping to task-specific (prediction, regression, etc.)
      heads (nn.Modules) that are key by name. The call function of each head
      must accept a boolean `training` named argument, accept the output of the
      `gnn` module and generate a pytree.
  """

  gnn: nn.Module
  initial_node_state_fn: Optional[nn.Module] = None

  # TODO(bmayer): We may want to define "parent" heads so we can make a
  # compute DAG.
  # Note these need to be buildable modules due hashing and binding
  # requirements. Buildable means any callable that has a .make() function that
  # returns a nn.Module. The buildable callable itself **must** be hashable,
  # e.g., if it's a dataclass, it needs to be marked/annotated as `frozen`.
  heads: Optional[Mapping[str, BuildableModule]] = None

  def __post_init__(self):
    if self.heads:
      if 'gnn' in self.heads:
        raise ValueError('Cannot have a head named `gnn`.')
    super().__post_init__()

  def setup(self):
    # This is needed to bind the modules in `head` to this nn.Module while
    # jitting.
    _bound_heads_dict = {}
    if self.heads:
      for k, v in self.heads.items():
        head_instance = v.make()
        # Assign dynamically named attribute for Flax to recognize as a
        # submodule
        setattr(self, f'_head_{k}', head_instance)

        # Points to the bound attribute.
        _bound_heads_dict[k] = head_instance
    self._heads = frozendict.frozendict(_bound_heads_dict)

  def __call__(self, graph, training: bool = True):
    if self.initial_node_state_fn is not None:
      graph = self.initial_node_state_fn(graph, training=training)

    graph = self.gnn(graph, training=training)
    outputs = {'gnn': graph}
    if self._heads:
      for k, head in self._heads.items():
        outputs[k] = head(graph, training=training)

    return outputs
