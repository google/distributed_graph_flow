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

"""GraphMAE-style masked graph auto-encoder layer for self-supervised learning.

Implements a masked graph auto-encoder inspired by GraphMAE (Hou et al., 2022,
https://arxiv.org/abs/2205.10803) operating directly on embedded node features
(see "Input embeddings" below).

This is not a faithful re-implementation of GraphMAE. It keeps the core recipe
of the paper: masking node features with a learnable mask token, encoding the
corrupted graph with a GNN, a bias-free encoder-to-decoder projection,
re-masking the masked nodes with zeros before a GNN decoder, and a Scaled
Cosine Error (SCE) reconstruction loss on the masked nodes. It deviates from
the paper as follows:

- Reconstruction target: the learned, and typically jointly trained, node
  feature embeddings (`"embedding"`) of the target nodeset are reconstructed
  instead of the raw input node features. Raw heterogeneous features can be
  scalars, categoricals, or variable-length sequences, for which cosine
  similarity is ill-defined.
- Heterogeneous graphs: only the target nodeset is masked and reconstructed.
  The encoder and decoder are stacks of `HeterogeneousGraphConvolution` layers
  (instead of GAT or GIN). Like the encoder norms, the encoder-to-decoder
  projection is nodeset-specific, which reduces to the single projection of
  the paper on homogeneous graphs.
- Decoder: the decoder can have several GNN layers, and its output is
  normalized and mapped to the embedding dimension by a linear head. The paper
  uses a single GNN decoder layer that directly outputs the reconstructed
  features.
- Seed nodes: all the target nodes can be masked, but the reconstruction loss
  is only computed on the masked seed nodes of the (sampled) input graph.
- Anti-collapse regularization: reconstructing a learned embedding can collapse
  to a constant or low-rank trivial solution. To prevent this, `compute_losses`
  adds two mechanisms from the self-supervised learning literature, which are
  not part of GraphMAE:
  1. Stop-gradient on the reconstruction target `x_init`
     (`stop_gradient_target`, inspired by BYOL / SimSiam / BGRL), preventing
     the reconstruction loss from pulling target feature embeddings toward a
     constant decoder prediction while still propagating gradients through
     unmasked context nodes in the encoder.
  2. VICReg variance and covariance regularization (`vicreg_loss`, Bardes et
     al., 2022) applied on real seed nodes to the encoder representations
     (`enc_rep`) and, if `regularize_input_embeddings`, to the input feature
     embeddings (`x_init`), enforcing unit standard deviation per dimension
     and decorrelated dimensions.

Input embeddings: each nodeset needs an `"embedding"` feature of dimension
`encoder_conv.dims`, which is also the reconstruction target of the target
nodeset. Compute it with `preprocess.EmbedGraph` followed by a non-linear
projection of each nodeset to `encoder_conv.dims`, e.g., with
`standard.ingest_feature`. `EmbedGraph` alone is not enough: it keeps each
numerical feature as a raw one-dimensional column, so the dimension of the
embedding differs between nodesets. A linear projection without bias is not
enough either: with a single numerical feature, all the embeddings of a
nodeset lie on a line through the origin, and the cosine of the reconstruction
loss only sees the sign of the feature.

IMPORTANT: The last node in each nodeset is required to be a padding node (as
ensured by `GraphMerger`), since `graph_masking.DropEdges` redirects dropped
edges to the last node of each nodeset.
"""

import dataclasses
import textwrap

import dataclasses_json
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import graph_masking
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp

# Node feature read and written by the masking layer and encoder GNN layers.
_EMBEDDING_FEATURE = "embedding"

Batch = tuple[jax_in_memory_graph_lib.JaxInMemoryGraph, jax.Array, jax.Array]


def sce_loss(
    x: jax.Array,
    y: jax.Array,
    alpha: float = 2.0,
    mask: jax.Array | None = None,
) -> jax.Array:
  """Scaled Cosine Error (SCE) reconstruction loss.

  Args:
    x: Predicted representations of shape ``[N, dim]``.
    y: Target representations of shape ``[N, dim]``.
    alpha: Scaling exponent for ``(1 - cos(x, y))^alpha``.
    mask: Optional boolean mask of shape ``[N]``. Loss is computed only where
      ``mask`` is True.

  Returns:
    Scalar mean SCE loss over the masked nodes.
  """
  eps = 1e-6

  def safe_normalize(v: jax.Array) -> jax.Array:
    norm = jnp.sqrt(jnp.sum(v**2, axis=-1, keepdims=True) + eps**2)
    return v / norm

  x = safe_normalize(x)
  y = safe_normalize(y)
  cos_sim = (x * y).sum(axis=-1)
  # Float32 rounding can make `cos_sim` slightly larger than 1. Clamp so that
  # `jnp.power` with a non-integer exponent never gets a negative base (NaN).
  diff = jnp.maximum(1 - cos_sim, 0.0)
  if alpha == 2:
    loss = jnp.square(diff)
  elif alpha == 1:
    loss = diff
  else:
    loss = jnp.power(diff + 1e-8, alpha)

  if mask is not None:
    loss = jnp.where(mask, loss, 0.0)
    return loss.sum() / jnp.maximum(mask.sum(), 1.0)
  return loss.mean()


def vicreg_loss(
    z: jax.Array,
    is_real: jax.Array | None = None,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[jax.Array, jax.Array]:
  """Computes VICReg variance and covariance regularization losses.

  Args:
    z: Representation matrix of shape ``[N, D]``.
    is_real: Optional boolean mask of shape ``[N]`` excluding padded nodes.
    gamma: Target standard deviation threshold (default 1.0).
    eps: Small constant for numerical stability of the standard deviation.

  Returns:
    A tuple ``(var_loss, cov_loss)`` of scalar JAX arrays.
  """
  if is_real is not None:
    num_real = jnp.maximum(is_real.sum(), 1.0)
    is_real_f = is_real[:, None].astype(z.dtype)
    z_mean = (z * is_real_f).sum(axis=0, keepdims=True) / num_real
    z_centered = (z - z_mean) * is_real_f
  else:
    num_real = z.shape[0]
    z_mean = z.mean(axis=0, keepdims=True)
    z_centered = z - z_mean

  z_var = (z_centered**2).sum(axis=0) / num_real
  z_std = jnp.sqrt(z_var + eps)
  var_loss = jnp.maximum(0.0, gamma - z_std).mean()

  cov = (z_centered.T @ z_centered) / num_real
  eye = jnp.eye(z.shape[1], dtype=z.dtype)
  cov_off_diag = cov * (1.0 - eye)
  cov_loss = (cov_off_diag**2).sum() / z.shape[1]

  return var_loss, cov_loss


@struct.dataclass
class GraphMAEOutput:
  """Outputs from the `GraphMAE` forward pass.

  Attributes:
    recon: Reconstructed embedded features of the target nodeset of shape
      ``[N, feat_dim]``.
    x_init: Original (pre-masking) embedded features of the target nodeset of
      shape ``[N, feat_dim]``.
    loss_mask: Boolean mask of shape ``[N]`` indicating which nodes are real
      (non-padding) seed nodes AND masked. The reconstruction loss is computed
      on these nodes. Since any target node can be masked, it can be all False,
      e.g., for a batch with few seed nodes.
    seed_mask: Boolean mask of shape ``[N]`` indicating which nodes are real
      seed nodes. The anti-collapse regularization is computed on these nodes.
    enc_rep: Encoder representations of the target nodeset of shape
      ``[N, hidden_dim]``.
  """

  recon: jax.Array
  x_init: jax.Array
  loss_mask: jax.Array
  seed_mask: jax.Array
  enc_rep: jax.Array


@struct.dataclass
class Losses:
  """Losses of a `GraphMAE` forward pass. See `compute_losses`.

  Attributes:
    reconstruction: Scaled cosine error reconstruction loss on the masked seed
      nodes.
    variance: Weighted VICReg variance loss.
    covariance: Weighted VICReg covariance loss.
  """

  reconstruction: jax.Array
  variance: jax.Array
  covariance: jax.Array

  def total(self) -> jax.Array:
    """The total loss optimized during training."""
    return self.reconstruction + self.variance + self.covariance


def compute_losses(
    output: GraphMAEOutput,
    config: "GraphMAEConfig",
) -> Losses:
  """Computes the reconstruction and anti-collapse regularization losses.

  The reconstruction loss is computed on `output.loss_mask` nodes, and the
  VICReg variance/covariance regularization on `output.seed_mask` nodes. If no
  seed node is masked, the reconstruction loss is 0.

  Args:
    output: The output of the `GraphMAE` forward pass.
    config: The `GraphMAEConfig` defining the loss weights and anti-collapse
      options.

  Returns:
    The computed `Losses`.
  """
  target = (
      jax.lax.stop_gradient(output.x_init)
      if config.stop_gradient_target
      else output.x_init
  )
  reconstruction = sce_loss(
      output.recon,
      target,
      alpha=config.scaled_cosine_error_alpha,
      mask=output.loss_mask,
  )

  if (
      config.variance_regularization_weight > 0.0
      or config.covariance_regularization_weight > 0.0
  ):
    var_loss, cov_loss = vicreg_loss(output.enc_rep, output.seed_mask)
    if config.regularize_input_embeddings:
      init_var_loss, init_cov_loss = vicreg_loss(
          output.x_init, output.seed_mask
      )
      var_loss = var_loss + init_var_loss
      cov_loss = cov_loss + init_cov_loss
  else:
    var_loss = cov_loss = jnp.zeros(())

  return Losses(
      reconstruction=reconstruction,
      variance=config.variance_regularization_weight * var_loss,
      covariance=config.covariance_regularization_weight * cov_loss,
  )


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True, kw_only=True)
class GraphMAEConfig(common.ArchitectureProvider):
  """Configuration for the `GraphMAE` layer.

  `GraphMAE` expects an already-embedded input graph where each nodeset has an
  `"embedding"` feature of dimension `encoder_conv.dims`, e.g., computed with
  `preprocess.EmbedGraph` and a per-nodeset `standard.ingest_feature` (see the
  module docstring). The embeddings of a random subset of the target nodes are
  masked, a GNN encoder computes the latent node representations, and a GNN
  decoder followed by a linear head reconstructs the input `"embedding"`
  feature of the masked target nodes.

  Usage example:

    ```python
    config = GraphMAEConfig(
        encoder_conv=hetero_gnn.HeterogeneousGraphConvolutionConfig(dims=64),
        num_encoder_layers=2,
        decoder_conv=hetero_gnn.HeterogeneousGraphConvolutionConfig(
            dims=64, embedding_feature="latent_embedding"
        ),
        target_nodeset="paper",
    )
    layer = config.make(embedded_schema)
    output = layer(embedded_graph, seed_idxs, is_real_node, training=True)
    losses = compute_losses(output, config)
    ```

  Attributes:
    encoder_conv: Encoder GNN configuration. Its `embedding_feature` must be
      "embedding", and its `dims`, which is also the dimension of the input
      and reconstructed embeddings, must be at least 2.
    num_encoder_layers: Number of encoder GNN layers.
    decoder_conv: Decoder GNN configuration. Its `dims` can differ from
      `encoder_conv.dims` only if `use_encoder_to_decoder` is True.
    num_decoder_layers: Number of decoder GNN layers.
    use_encoder_to_decoder: If True, apply a linear projection (no bias) from
      the encoder output to the decoder input of size `decoder_conv.dims`,
      with separate weights for each nodeset. Matches the GraphMAE paper on
      graphs with a single nodeset.
    norm: Normalization applied after the encoder (to each nodeset) and after
      the decoder. One of 'layer_norm', 'rms_norm', 'batch_norm', or None to
      disable.
    target_nodeset: The name of the target nodeset for masking and
      reconstruction.
    mask_rate: Fraction of nodes in the target nodeset to mask.
    drop_edge_rate: Fraction of edges (of all edge sets) to drop during
      training.
    replace_rate: Fraction of masked nodes replaced with random node features
      instead of the mask token.
    scaled_cosine_error_alpha: Loss scaling exponent for SCE reconstruction
      loss.
    variance_regularization_weight: Weight of the VICReg variance loss on the
      representations.
    covariance_regularization_weight: Weight of the VICReg covariance loss on
      the representations.
    stop_gradient_target: If True, applies `jax.lax.stop_gradient` to `x_init`
      in the SCE reconstruction loss so that a learnable upstream feature
      embedder cannot collapse `x_init` toward the decoder prediction.
    regularize_input_embeddings: If True, applies VICReg variance and covariance
      regularization to `x_init` in addition to `enc_rep`, preventing upstream
      learnable feature embeddings from collapsing to a constant or low-rank
      subspace.
  """

  encoder_conv: hetero_gnn.HeterogeneousGraphConvolutionConfig = (
      layer_registry.field()
  )
  num_encoder_layers: int
  decoder_conv: hetero_gnn.HeterogeneousGraphConvolutionConfig = (
      layer_registry.field()
  )
  num_decoder_layers: int = 1

  use_encoder_to_decoder: bool = True
  norm: str | None = "layer_norm"

  target_nodeset: str
  mask_rate: float = 0.5
  drop_edge_rate: float = 0.5
  replace_rate: float = 0.0
  scaled_cosine_error_alpha: float = 3.0
  variance_regularization_weight: float = 0.5
  covariance_regularization_weight: float = 0.01
  stop_gradient_target: bool = True
  regularize_input_embeddings: bool = True

  def make(
      self, schema: schema_lib.GraphSchema, name: str | None = None
  ) -> "GraphMAE":
    """Checks the config against `schema` and builds the `GraphMAE` layer.

    Args:
      schema: The schema of the embedded graph.
      name: Optional name for the Flax module.

    Returns:
      The `GraphMAE` module.

    Raises:
      ValueError: If the configuration is invalid or incompatible with `schema`.
    """
    if self.encoder_conv.dims < 2:
      raise ValueError(
          "`encoder_conv.dims` must be at least 2, since the reconstruction"
          " loss is a cosine error on embeddings of this dimension. In one"
          " dimension, the cosine only compares the signs. Got"
          f" {self.encoder_conv.dims}."
      )
    if (
        not self.use_encoder_to_decoder
        and self.decoder_conv.dims != self.encoder_conv.dims
    ):
      raise ValueError(
          "With `use_encoder_to_decoder=False`, the decoder input is the"
          f" encoder output: `decoder_conv.dims` ({self.decoder_conv.dims})"
          f" must be equal to `encoder_conv.dims` ({self.encoder_conv.dims})."
      )
    if self.encoder_conv.embedding_feature != _EMBEDDING_FEATURE:
      raise ValueError(
          "`encoder_conv.embedding_feature` must be"
          f" {_EMBEDDING_FEATURE!r}. Got"
          f" {self.encoder_conv.embedding_feature!r}."
      )
    if self.target_nodeset not in schema.node_sets:
      raise ValueError(
          f"Target nodeset {self.target_nodeset!r} not found in the schema."
          f" Available: {sorted(schema.node_sets)}."
      )
    # The encoder reads the embedding of every nodeset, and its residual
    # connections require the embedding dimension to be `encoder_conv.dims`.
    expected_shape = (self.encoder_conv.dims,)
    for nodeset_name, nodeset_schema in schema.node_sets.items():
      feature_schema = nodeset_schema.features.get(_EMBEDDING_FEATURE)
      if feature_schema is None:
        raise ValueError(
            f"Nodeset {nodeset_name!r} must have an {_EMBEDDING_FEATURE!r}"
            " feature in the embedded schema. Available:"
            f" {sorted(nodeset_schema.features)}."
        )
      shape = feature_schema.shape
      if shape is None or tuple(shape) != expected_shape:
        raise ValueError(
            f"The {_EMBEDDING_FEATURE!r} feature of nodeset {nodeset_name!r}"
            f" must have shape {expected_shape} to match `encoder_conv.dims`."
            f" Got {shape}."
        )
    return GraphMAE(config=self, schema=schema, name=name)

  def get_gnn_dims(self) -> int:
    """Returns the hidden dimension of the configured GNN encoder."""
    return self.encoder_conv.dims

  def mask_node_features_config(self) -> graph_masking.MaskNodeFeaturesConfig:
    """Configuration of the layer masking the target nodes."""
    return graph_masking.MaskNodeFeaturesConfig(
        nodeset=self.target_nodeset,
        feature=_EMBEDDING_FEATURE,
        mask_rate=self.mask_rate,
        replace_rate=self.replace_rate,
    )

  def drop_edges_config(self) -> graph_masking.DropEdgesConfig:
    """Configuration of the layer dropping edges during training."""
    return graph_masking.DropEdgesConfig(drop_rate=self.drop_edge_rate)

  def architecture(self) -> str:
    parts = [
        self.mask_node_features_config().architecture(),
        self.drop_edges_config().architecture(),
        f"Encoder GNN Block x{self.num_encoder_layers}:",
        textwrap.indent(self.encoder_conv.architecture(), "    "),
    ]
    if self.norm:
      parts.append(f"Norm({self.norm})")
    if self.use_encoder_to_decoder:
      parts.append(f"Dense({self.decoder_conv.dims}, use_bias=False)")
    parts.append(f"Decoder GNN Block x{self.num_decoder_layers}:")
    parts.append(textwrap.indent(self.decoder_conv.architecture(), "    "))
    if self.norm:
      parts.append(f"Norm({self.norm})")
    parts.append("Reconstruction head: embedding")
    return "\n".join(parts)


class GraphMAE(nn.Module):
  """Masked graph auto-encoder operating on an embedded graph.

  Masks random target node embeddings, encodes the corrupted graph with a GNN
  encoder to obtain latent representations, and applies a GNN decoder to
  reconstruct the original embedded features of the masked target nodes.

  The encoder weights are stored under the `encoder` scope, and are shared by
  `__call__` and `embed`.
  """

  config: GraphMAEConfig
  schema: schema_lib.GraphSchema

  @nn.compact_name_scope
  def encoder(
      self,
      graph: jax_in_memory_graph_lib.JaxInMemoryGraph,
      training: bool,
  ) -> tuple[jax.Array, jax_in_memory_graph_lib.JaxInMemoryGraph]:
    """Shared encoder: GNN layers followed by a norm on each nodeset.

    Used by both `__call__` and `embed`. As a `nn.compact_name_scope` method,
    its submodules are defined inline under the `encoder` scope, while
    `__call__` remains the only `nn.compact` method of the module.

    Args:
      graph: Input graph with `"embedding"` features on each nodeset.
      training: Whether the model is in training mode.

    Returns:
      A tuple of `(enc_rep, enc_graph)` where `enc_graph` contains the
      normalized encoder representations of all the nodesets, and `enc_rep`
      the ones of the target nodeset.
    """
    enc_graph = graph
    for i in range(self.config.num_encoder_layers):
      layer = self.config.encoder_conv.make(
          self.schema, name=f"encoder_layer_{i:02d}"
      )
      enc_graph = layer(enc_graph, training=training)

    # All the nodesets are normalized, since the decoder takes the
    # representations of the target nodes and of their neighbors.
    node_sets = {}
    for nodeset_name in self.schema.node_sets:
      nodeset = enc_graph.node_sets[nodeset_name]
      embedding = standard.norm(
          nodeset.features[_EMBEDDING_FEATURE],
          self.config.norm,
          training,
          name=f"encoder_norm_{nodeset_name}",
      )
      node_sets[nodeset_name] = jax_in_memory_graph_lib.JaxInMemoryNodeSet(
          num_nodes=nodeset.num_nodes, features={_EMBEDDING_FEATURE: embedding}
      )
    enc_graph = jax_in_memory_graph_lib.JaxInMemoryGraph(
        node_sets=node_sets, edge_sets=enc_graph.edge_sets
    )
    enc_rep = node_sets[self.config.target_nodeset].features[_EMBEDDING_FEATURE]
    return enc_rep, enc_graph

  @nn.compact
  def __call__(
      self,
      graph_or_batch: jax_in_memory_graph_lib.JaxInMemoryGraph | Batch,
      seed_idxs: jax.Array | None = None,
      is_real_node: jax.Array | None = None,
      *,
      training: bool,
  ) -> GraphMAEOutput:
    """Full encode-decode forward pass for training and evaluation.

    The target nodes are masked even if `training` is False, so that the
    reconstruction loss can also be computed for evaluation. Therefore, a
    `"masking"` RNG is always required. Edges are only dropped in training.

    Args:
      graph_or_batch: Either an embedded `JaxInMemoryGraph`, or a tuple of
        `(graph, seed_idxs, is_real_node)`.
      seed_idxs: Optional indices of the seed nodes in the target nodeset. If
        None (and `graph_or_batch` is a graph), all real nodes are treated as
        seed nodes. Must be None if `graph_or_batch` is a tuple.
      is_real_node: Optional boolean mask of shape `[num_target_nodes]`
        indicating which nodes in the target nodeset are real (non-padding). If
        None (and `graph_or_batch` is a graph), all nodes except the last
        (sentinel padding) node are treated as real. Must be None if
        `graph_or_batch` is a tuple.
      training: Whether the model is in training mode, e.g., drops edges and
        applies dropout.

    Returns:
      `GraphMAEOutput` containing the reconstructed embeddings, target
      embeddings, masks, and encoder representations.

    Raises:
      ValueError: If `graph_or_batch` is a tuple and `seed_idxs` or
        `is_real_node` is also given, or if the number of target nodes is
        unknown.
    """
    if isinstance(graph_or_batch, tuple):
      if seed_idxs is not None or is_real_node is not None:
        raise ValueError(
            "`seed_idxs` and `is_real_node` cannot be given when"
            " `graph_or_batch` is a `(graph, seed_idxs, is_real_node)` tuple."
        )
      graph, seed_idxs, is_real_node = graph_or_batch
    else:
      graph = graph_or_batch

    target_ns = self.config.target_nodeset
    num_target_nodes = graph.node_sets[target_ns].num_nodes
    if num_target_nodes is None:
      raise ValueError(f"num_nodes for {target_ns} cannot be None")

    if is_real_node is None:
      is_real_node = jnp.arange(num_target_nodes) < max(num_target_nodes - 1, 0)
    if seed_idxs is None:
      is_seed = is_real_node
    else:
      is_seed = (
          jnp.zeros(num_target_nodes, dtype=jnp.bool_)
          .at[seed_idxs]
          .set(True)
          & is_real_node
      )

    # 1. Reconstruction target: the embedded features of the target nodeset.
    x_init = graph.node_sets[target_ns].features[_EMBEDDING_FEATURE]

    # 2. Masking. All the target nodes can be masked, but only the masked seed
    # nodes are reconstructed in the loss.
    mask_layer = self.config.mask_node_features_config().make(
        name="mask_node_features"
    )
    masked_graph, is_masked = mask_layer(graph)
    loss_mask = is_masked & is_seed

    # 3. Drop Edge Augmentation (only during training).
    drop_edges = self.config.drop_edges_config().make(
        self.schema, name="drop_edges"
    )
    masked_graph = drop_edges(masked_graph, training=training)

    # 4. Encode.
    enc_rep, enc_graph = self.encoder(masked_graph, training)

    # 5. Project the representations of all the nodesets to the decoder input,
    # and re-mask the masked target nodes (avoids the identity shortcut). Like
    # the encoder norms, the projections are nodeset-specific: the encoder does
    # not map the different nodesets to a shared latent space.
    decoder_config = self.config.decoder_conv
    dec_feat = decoder_config.embedding_feature
    dec_node_sets = {}
    for nodeset_name in self.schema.node_sets:
      nodeset = enc_graph.node_sets[nodeset_name]
      rep = nodeset.features[_EMBEDDING_FEATURE]
      if self.config.use_encoder_to_decoder:
        rep = nn.Dense(
            decoder_config.dims,
            use_bias=False,
            name=f"encoder_to_decoder_{nodeset_name}",
        )(rep)
      if nodeset_name == target_ns:
        rep = jnp.where(is_masked[:, None], 0.0, rep)
      dec_node_sets[nodeset_name] = jax_in_memory_graph_lib.JaxInMemoryNodeSet(
          num_nodes=nodeset.num_nodes, features={dec_feat: rep}
      )
    remasked_graph = jax_in_memory_graph_lib.JaxInMemoryGraph(
        node_sets=dec_node_sets,
        edge_sets=graph.edge_sets,
    )

    # 6. Decoder.
    recon_graph = remasked_graph
    for i in range(self.config.num_decoder_layers):
      layer = decoder_config.make(self.schema, name=f"decoder_layer_{i:02d}")
      recon_graph = layer(recon_graph, training=training)

    decoder_out = standard.norm(
        recon_graph.node_sets[target_ns].features[dec_feat],
        self.config.norm,
        training,
        name="decoder_norm",
    )

    # 7. Reconstruction head for the embedded features.
    recon = nn.Dense(x_init.shape[-1], name="recon_head")(decoder_out)

    return GraphMAEOutput(
        recon=recon,
        x_init=x_init,
        loss_mask=loss_mask,
        seed_mask=is_seed,
        enc_rep=enc_rep,
    )

  def embed(
      self,
      graph_or_batch: jax_in_memory_graph_lib.JaxInMemoryGraph | Batch,
      training: bool = False,
  ) -> jax.Array:
    """Returns the encoder representations of the target nodeset.

    Unlike `__call__`, no node is masked, so no `"masking"` RNG is needed.

    Args:
      graph_or_batch: Either an embedded `JaxInMemoryGraph`, or a tuple of
        `(graph, seed_idxs, is_real_node)` of which only the graph is used.
      training: Whether the model is in training mode, e.g., applies dropout.

    Returns:
      The encoder representations of all the target nodes, of shape
      `[num_target_nodes, encoder_conv.dims]`.
    """
    graph = (
        graph_or_batch[0]
        if isinstance(graph_or_batch, tuple)
        else graph_or_batch
    )
    enc_rep, _ = self.encoder(graph, training)
    return enc_rep
