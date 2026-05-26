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

"""Flax modules implementing low level GNN operations."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error

from dgf.src.learning.jax.layers.mlp import MLP

from dgf.src.learning.jax.layers.preprocess import EmbedAndHomogenizeGraphConfig
from dgf.src.learning.jax.layers.preprocess import EmbedAndHomogenizeGraph

from dgf.src.learning.jax.layers.preprocess import EmbedFeatureSetConfig
from dgf.src.learning.jax.layers.preprocess import EmbedFeatureSet

from dgf.src.learning.jax.layers.preprocess import EmbedGraphConfig
from dgf.src.learning.jax.layers.preprocess import EmbedGraph

from dgf.src.learning.jax.layers.classification import ClassificationHead
from dgf.src.learning.jax.layers.classification import ClassificationHeadConfig

from dgf.src.learning.jax.layers.hetero_gnn import HeterogeneousGraphConvolution
from dgf.src.learning.jax.layers.hetero_gnn import HeterogeneousGraphConvolutionConfig

from dgf.src.learning.jax.layers.hetero_graph_attention_network import HeterogeneousGraphAttentionNetwork
from dgf.src.learning.jax.layers.hetero_graph_attention_network import HeterogeneousGraphAttentionNetworkConfig

from dgf.src.learning.jax.layers.residual_mlp import ResidualMLPV2
from dgf.src.learning.jax.layers.residual_mlp import ResidualMLPV2Config

from dgf.src.learning.jax.layers.standard import GenericBlock
from dgf.src.learning.jax.layers.standard import GenericBlockConfig
from dgf.src.learning.jax.layers.standard import modern_residual_mlp
from dgf.src.learning.jax.layers.standard import ingest_feature
from dgf.src.learning.jax.layers.standard import sequential_mlp
from dgf.src.learning.jax.layers.standard import identity
from dgf.src.learning.jax.layers.homo_gnn_sparse_deferred import ProjectorConfig

from dgf.src.learning.jax.layers.homo_gnn_sparse_deferred import Projector
from dgf.src.learning.jax.layers.homo_gnn_sparse_deferred import GCNConfig
from dgf.src.learning.jax.layers.homo_gnn_sparse_deferred import GCN
from dgf.src.learning.jax.layers.homo_gnn_sparse_deferred import MPNNConfig
from dgf.src.learning.jax.layers.homo_gnn_sparse_deferred import MPNN
from dgf.src.learning.jax.layers.homo_gnn_sparse_deferred import GINConfig
from dgf.src.learning.jax.layers.homo_gnn_sparse_deferred import GIN
from dgf.src.learning.jax.layers.homo_gnn_sparse_deferred import ConditionalGIN
