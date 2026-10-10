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

"""Deprecated: Graph Attention Network layers for heterogeneous graphs.

Use `HeterogeneousGraphConvolutionConfig.dot_product_attention()` (in
`hetero_gnn.py`) instead. This module is kept so that existing code and saved
configs keep working.
"""

import dataclasses
import dataclasses_json
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import graph_attention
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from dgf.src.util import log

DEPRECATION_MESSAGE = (
    "HeterogeneousGraphAttentionNetworkConfig is deprecated. Use"
    " HeterogeneousGraphConvolutionConfig.dot_product_attention() instead."
)


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class HeterogeneousGraphAttentionNetworkConfig(common.ArchitectureProvider):
  """Deprecated.

  Use `HeterogeneousGraphConvolutionConfig.dot_product_attention`.

  `make()` returns the equivalent `HeterogeneousGraphConvolution` layer (see
  `to_graph_convolution_config()`).

  Usage example:

  ```python
  # Deprecated:
  config = HeterogeneousGraphAttentionNetworkConfig(dims=64, num_heads=4)
  # Equivalent:
  config = HeterogeneousGraphConvolutionConfig.dot_product_attention(
      dims=64, num_heads=4)
  ```

  Attributes:
    plan: Message passing plan. A list of (edge_set_name, is_reversed) tuples.
      If None, messages are passed along all edges in both directions.
    embedding_feature: Name of the node feature to use for embeddings.
    dims: Dimension of the embeddings and hidden layers. Used to build the
      default values of `message`, `update`, and `post`.
    dropout_rate: Dropout rate. Used to build the default values of `update` and
      `post`.
    message_aggregation: Unused attribute kept for backward compatibility (e.g.
      with serialized configs). Within each relation, messages are always
      aggregated using attention (`MessageAggregation.ATTENTION`), and across
      relations, the aggregated messages are always summed
      (`RelationAggregation.SUM`). Must be `"sum"`.
    num_heads: Number of attention heads.
    force_basic_implementation: If True, always compute messages with the
      generic per-edge implementation (no node-level projection, no edge
      sorting). Mostly useful in tests to validate the optimized paths.
    message: Optional module to apply to edge features to generate values.
      Defaults to a single-layer MLP.
    update: Optional module to apply to node embeddings after message passing,
      combining the old embedding and aggregated messages. Defaults to a
      single-layer MLP with layer norm and activation.
    post: Optional module applied after the update step, typically a
      transformer-like MLP. Defaults to a two-layer ResidualMLP.
  """

  plan: list[tuple[str, bool]] | None = None
  embedding_feature: str = "embedding"
  dims: int = 128
  dropout_rate: float = 0.1
  message_aggregation: str = "sum"
  num_heads: int = 4
  force_basic_implementation: bool = False

  message: common.GenericLayer | None = layer_registry.field(default=None)
  update: common.GenericLayer | None = layer_registry.field(default=None)
  post: common.GenericLayer | None = layer_registry.field(default=None)

  def __post_init__(self):
    if self.message_aggregation != "sum":
      raise ValueError(
          "message_aggregation must be 'sum', got"
          f" {self.message_aggregation!r}."
      )
    if self.message is None:
      self.message = standard.GenericBlockConfig("LAL", dims=self.dims)
    if self.update is None:
      self.update = standard.GenericBlockConfig(
          "LADL", dims=self.dims, dropout_rate=self.dropout_rate
      )
    if self.post is None:
      self.post = standard.modern_residual_mlp(
          dims=self.dims, dropout_rate=self.dropout_rate
      )

  def to_graph_convolution_config(
      self,
  ) -> hetero_gnn.HeterogeneousGraphConvolutionConfig:
    """Returns the equivalent `HeterogeneousGraphConvolutionConfig`."""
    return hetero_gnn.HeterogeneousGraphConvolutionConfig(
        plan=None if self.plan is None else list(self.plan),
        embedding_feature=self.embedding_feature,
        dims=self.dims,
        dropout_rate=self.dropout_rate,
        message_aggregation=hetero_gnn.MessageAggregation.ATTENTION,
        relation_aggregation=hetero_gnn.RelationAggregation.SUM,
        message_consumes_target=True,
        combine=hetero_gnn.Combine.CONCAT,
        residual=True,
        force_basic_implementation=self.force_basic_implementation,
        message=self.message,
        update=self.update,
        post=self.post,
        attention=graph_attention.DotProductAttentionConfig(
            num_heads=self.num_heads, dims=self.dims
        ),
    )

  def make(
      self, schema: schema_lib.GraphSchema, name: str | None = None
  ) -> hetero_gnn.HeterogeneousGraphConvolution:
    log.warning(DEPRECATION_MESSAGE)
    return self.to_graph_convolution_config().make(schema, name=name)

  def architecture(self) -> str:
    return self.to_graph_convolution_config().architecture()


# Deprecated alias. Use `hetero_gnn.HeterogeneousGraphConvolution` instead.
HeterogeneousGraphAttentionNetwork = hetero_gnn.HeterogeneousGraphConvolution
