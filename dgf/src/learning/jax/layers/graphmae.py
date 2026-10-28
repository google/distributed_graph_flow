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
EMBEDDING_FEATURE = "embedding"

Batch = tuple[jax_in_memory_graph_lib.JaxInMemoryGraph, jax.Array]


def _safe_normalize(v: jax.Array, eps: float = 1e-6) -> jax.Array:
  """L2-normalizes `v` along the last axis; zero vectors stay zero."""
  return v * jax.lax.rsqrt(jnp.sum(v**2, axis=-1, keepdims=True) + eps**2)


def sce_loss(
    x: jax.Array,
    y: jax.Array,
    alpha: float = 2.0,
    mask: jax.Array | None = None,
    eps: float = 1e-6,
) -> jax.Array:
  """Scaled Cosine Error (SCE) reconstruction loss.

  Args:
    x: Predicted representations of shape ``[N, dim]``.
    y: Target representations of shape ``[N, dim]``.
    alpha: Scaling exponent for ``(1 - cos(x, y))^alpha``.
    mask: Optional boolean mask of shape ``[N]``. Loss is computed only where
      ``mask`` is True.
    eps: Small constant for numerical stability of L2 normalization.

  Returns:
    Scalar mean SCE loss over the masked nodes.
  """
  x = _safe_normalize(x, eps)
  y = _safe_normalize(y, eps)
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
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[jax.Array, jax.Array]:
  """Computes VICReg variance and covariance regularization losses.

  Args:
    z: Representation matrix of shape ``[N, D]``.
    gamma: Target standard deviation threshold (default 1.0).
    eps: Small constant for numerical stability of the standard deviation.

  Returns:
    A tuple ``(var_loss, cov_loss)`` of scalar JAX arrays.
  """
  num_nodes = max(z.shape[0], 1)
  z_centered = z - z.mean(axis=0, keepdims=True)

  z_var = (z_centered**2).sum(axis=0) / num_nodes
  z_std = jnp.sqrt(z_var + eps)
  var_loss = jnp.maximum(0.0, gamma - z_std).mean()

  cov = (z_centered.T @ z_centered) / num_nodes
  eye = jnp.eye(z.shape[1], dtype=z.dtype)
  cov_off_diag = cov * (1.0 - eye)
  cov_loss = (cov_off_diag**2).sum() / z.shape[1]

  return var_loss, cov_loss


def effective_rank_ratio(z: jax.Array) -> jax.Array:
  """Effective rank of the rows of `z`, divided by the maximum possible rank.

  The effective rank (Roy & Vetterli, 2007) is the exponential of the entropy
  of the normalized eigenvalues of the covariance matrix of `z`. It is 1 if all
  the variance lies along a single direction (e.g., collapsed embeddings), and
  equal to the number of dimensions if the variance is spread uniformly.

  Since `N` rows span at most `N - 1` dimensions after centering, the effective
  rank is divided by `min(D, N - 1)`, so that 1 is reachable for any batch size.

  Args:
    z: Representation matrix of shape ``[N, D]``.

  Returns:
    A scalar in ``(0, 1]``. Close to 1 means that the representations use all
    the available dimensions, and close to 0 means that they collapsed to a
    low-dimensional subspace.
  """
  num_nodes, dims = z.shape
  z_centered = z - z.mean(axis=0, keepdims=True)
  cov = (z_centered.T @ z_centered) / max(num_nodes, 1)
  eigvals = jnp.maximum(jnp.linalg.eigvalsh(cov), 0.0)
  probs = eigvals / jnp.maximum(eigvals.sum(), 1e-12)
  # 0 * log(0) = 0.
  entropy = -jnp.sum(probs * jnp.log(jnp.where(probs > 0, probs, 1.0)))
  max_rank = max(min(dims, num_nodes - 1), 1)
  return jnp.exp(entropy) / max_rank


@struct.dataclass(frozen=True)
class GraphMAEOutput:
  """Outputs from the `GraphMAE` forward pass on the `S` seed nodes.

  Attributes:
    recon: Reconstructed embedded features of the seed nodes of shape
      ``[S, feat_dim]``.
    x_init: Original (pre-masking) embedded features of the seed nodes of shape
      ``[S, feat_dim]``.
    loss_mask: Boolean mask of shape ``[S]`` indicating which seed nodes were
      masked. The reconstruction loss is computed on these nodes. Since any
      target node can be masked, it can be all False, e.g., for a batch with few
      seed nodes.
    enc_rep: Encoder representations of the seed nodes of shape
      ``[S, hidden_dim]``.
  """

  recon: jax.Array
  x_init: jax.Array
  loss_mask: jax.Array
  enc_rep: jax.Array


@struct.dataclass(frozen=True)
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

  The reconstruction loss is computed on the masked seed nodes
  (`output.loss_mask`), and the VICReg variance/covariance regularization on all
  the seed nodes. If no seed node is masked, the reconstruction loss is 0.

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
    var_loss, cov_loss = vicreg_loss(output.enc_rep)
    if config.regularize_input_embeddings:
      init_var_loss, init_cov_loss = vicreg_loss(output.x_init)
      var_loss = var_loss + init_var_loss
      cov_loss = cov_loss + init_cov_loss
  else:
    var_loss = cov_loss = jnp.zeros(())

  return Losses(
      reconstruction=reconstruction,
      variance=config.variance_regularization_weight * var_loss,
      covariance=config.covariance_regularization_weight * cov_loss,
  )


@struct.dataclass(frozen=True)
class Metrics:
  """Interpretable, scale-free metrics of a `GraphMAE` forward pass.

  Unlike `Losses`, these metrics do not depend on the loss weights nor on
  `scaled_cosine_error_alpha`, and are not optimized. See `compute_metrics`.

  Attributes:
    reconstruction_cosine: Mean cosine similarity in ``[-1, 1]`` between the
      reconstructed and the original embeddings of the masked seed nodes. 1 is
      a perfect reconstruction.
    reconstruction_relative_error: Cosine error of the reconstruction relative
      to the best constant prediction: ``(1 - cos) / (1 - baseline_cos)``,
      where ``baseline_cos`` is the mean cosine similarity of the best constant
      prediction (the mean direction of the targets). 0 is a perfect
      reconstruction, 1 is no better than predicting the same embedding for
      all the nodes, and values above 1 are worse than that. Ill-conditioned
      if the targets collapsed (see `target_rank_ratio`).
    embedding_rank_ratio: `effective_rank_ratio` of the node embeddings, i.e.,
      the encoder representations (`enc_rep`) of the seed nodes.
    target_rank_ratio: `effective_rank_ratio` of the reconstruction targets,
      i.e., the input feature embeddings (`x_init`) of the seed nodes.
  """

  reconstruction_cosine: jax.Array
  reconstruction_relative_error: jax.Array
  embedding_rank_ratio: jax.Array
  target_rank_ratio: jax.Array


def compute_metrics(output: GraphMAEOutput, eps: float = 1e-6) -> Metrics:
  """Computes interpretable metrics for monitoring and evaluation.

  The reconstruction metrics are computed on the masked seed nodes
  (`output.loss_mask`), like the reconstruction loss, and are 0 if no seed node
  is masked. The rank metrics are computed on all the seed nodes, like the
  VICReg regularization. The metrics are computed per batch, and are noisier
  for batches with few seed nodes.

  Args:
    output: The output of the `GraphMAE` forward pass.
    eps: Small constant for numerical stability.

  Returns:
    The computed `Metrics`.
  """
  output = jax.lax.stop_gradient(output)
  mask = output.loss_mask
  num_masked = jnp.maximum(mask.sum(), 1.0)

  def masked_mean(v: jax.Array) -> jax.Array:
    return jnp.where(mask, v, 0.0).sum() / num_masked

  target = _safe_normalize(output.x_init, eps)
  recon = _safe_normalize(output.recon, eps)
  reconstruction_cosine = masked_mean((recon * target).sum(axis=-1))
  # The constant prediction maximizing the mean cosine similarity is the mean
  # direction of the targets, and its mean cosine similarity is the norm of the
  # mean of the normalized targets.
  mean_target = jnp.where(mask[:, None], target, 0.0).sum(axis=0) / num_masked
  baseline_cosine = jnp.linalg.norm(mean_target)
  # With no masked seed node, `masked_mean` is 0 and the relative error is 0.
  reconstruction_relative_error = masked_mean(
      1.0 - (recon * target).sum(axis=-1)
  ) / jnp.maximum(1.0 - baseline_cosine, eps)

  return Metrics(
      reconstruction_cosine=reconstruction_cosine,
      reconstruction_relative_error=reconstruction_relative_error,
      embedding_rank_ratio=effective_rank_ratio(output.enc_rep),
      target_rank_ratio=effective_rank_ratio(output.x_init),
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
    output = layer(embedded_graph, seed_idxs, training=True)
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
    """Checks the config against `schema` and builds the `GraphMAE` layer."""
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
    if self.encoder_conv.embedding_feature != EMBEDDING_FEATURE:
      raise ValueError(
          "`encoder_conv.embedding_feature` must be"
          f" {EMBEDDING_FEATURE!r}. Got"
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
      feature_schema = nodeset_schema.features.get(EMBEDDING_FEATURE)
      if feature_schema is None:
        raise ValueError(
            f"Nodeset {nodeset_name!r} must have an {EMBEDDING_FEATURE!r}"
            " feature in the embedded schema. Available:"
            f" {sorted(nodeset_schema.features)}."
        )
      shape = feature_schema.shape
      if shape is None or tuple(shape) != expected_shape:
        raise ValueError(
            f"The {EMBEDDING_FEATURE!r} feature of nodeset {nodeset_name!r}"
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
        feature=EMBEDDING_FEATURE,
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


class GraphMAEEncoder(nn.Module):
  """Shared GraphMAE encoder: GNN layers followed by a norm on each nodeset."""

  config: GraphMAEConfig
  schema: schema_lib.GraphSchema

  @nn.compact
  def __call__(
      self,
      graph: jax_in_memory_graph_lib.JaxInMemoryGraph,
      training: bool,
  ) -> tuple[jax.Array, jax_in_memory_graph_lib.JaxInMemoryGraph]:
    """Encodes `graph`.

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
          nodeset.features[EMBEDDING_FEATURE],
          self.config.norm,
          training,
          name=f"encoder_norm_{nodeset_name}",
      )
      node_sets[nodeset_name] = jax_in_memory_graph_lib.JaxInMemoryNodeSet(
          num_nodes=nodeset.num_nodes, features={EMBEDDING_FEATURE: embedding}
      )
    enc_graph = jax_in_memory_graph_lib.JaxInMemoryGraph(
        node_sets=node_sets, edge_sets=enc_graph.edge_sets
    )
    enc_rep = node_sets[self.config.target_nodeset].features[EMBEDDING_FEATURE]
    return enc_rep, enc_graph


class GraphMAE(nn.Module):
  """Masked graph auto-encoder operating on an embedded graph.

  Masks random target node embeddings, encodes the corrupted graph with a GNN
  encoder to obtain latent representations, and applies a GNN decoder to
  reconstruct the original embedded features of the masked target nodes.

  The encoder is a `GraphMAEEncoder` submodule named `encoder`, shared by
  `__call__` and `embed`. Both methods go through the single compact method
  `_forward`, so that `GraphMAE` can be nested in any parent module.
  """

  config: GraphMAEConfig
  schema: schema_lib.GraphSchema

  def __call__(
      self,
      graph_or_batch: jax_in_memory_graph_lib.JaxInMemoryGraph | Batch,
      seed_idxs: jax.Array | None = None,
      *,
      training: bool,
  ) -> GraphMAEOutput:
    """Full encode-decode forward pass for training and evaluation.

    The target nodes are masked even if `training` is False, so that the
    reconstruction loss can also be computed for evaluation. Therefore, a
    `"masking"` RNG is always required. Edges are only dropped in training.

    Args:
      graph_or_batch: Either an embedded `JaxInMemoryGraph`, or a tuple of
        `(graph, seed_idxs)`.
      seed_idxs: Optional indices of the seed nodes in the target nodeset (all
        must be real, non-padding nodes). If None (and `graph_or_batch` is a
        graph), all nodes except the last (sentinel padding) node are treated as
        seed nodes. Must be None if `graph_or_batch` is a tuple.
      training: Whether the model is in training mode, e.g., drops edges and
        applies dropout.

    Returns:
      `GraphMAEOutput` containing the reconstructed embeddings, target
      embeddings, loss mask, and encoder representations of the seed nodes.

    Raises:
      ValueError: If `graph_or_batch` is a tuple and `seed_idxs` is also given,
        or if the number of target nodes is unknown.
    """
    if isinstance(graph_or_batch, tuple):
      if seed_idxs is not None:
        raise ValueError(
            "`seed_idxs` cannot be given when `graph_or_batch` is a"
            " `(graph, seed_idxs)` tuple."
        )
      graph, seed_idxs = graph_or_batch
    else:
      graph = graph_or_batch

    output = self._forward(
        graph, seed_idxs, training=training, embed_only=False
    )
    assert isinstance(output, GraphMAEOutput)
    return output

  @nn.compact
  def _forward(
      self,
      graph: jax_in_memory_graph_lib.JaxInMemoryGraph,
      seed_idxs: jax.Array | None,
      *,
      training: bool,
      embed_only: bool,
  ) -> GraphMAEOutput | jax.Array:
    """Single compact method holding all the submodules of `GraphMAE`.

    Args:
      graph: An embedded `JaxInMemoryGraph`.
      seed_idxs: See `__call__`. Unused if `embed_only`.
      training: Whether the model is in training mode.
      embed_only: If True, only runs the encoder on the unmasked graph and
        returns the encoder representations of the target nodeset (see
        `embed`). Otherwise, runs the full forward pass (see `__call__`).

    Returns:
      The encoder representations of the target nodeset if `embed_only`, and
      the `GraphMAEOutput` of the full forward pass otherwise.
    """
    encoder = GraphMAEEncoder(
        config=self.config, schema=self.schema, name="encoder"
    )
    if embed_only:
      enc_rep, _ = encoder(graph, training)
      return enc_rep

    target_ns = self.config.target_nodeset
    num_target_nodes = graph.node_sets[target_ns].num_nodes
    if num_target_nodes is None:
      raise ValueError(f"num_nodes for {target_ns} cannot be None")

    if seed_idxs is None:
      # All nodes except the last (sentinel padding) node are seed nodes.
      seed_idxs = jnp.arange(max(num_target_nodes - 1, 0))

    # 1. Reconstruction target: the embedded features of the seed nodes.
    x_init = graph.node_sets[target_ns].features[EMBEDDING_FEATURE][seed_idxs]

    # 2. Masking. All the target nodes can be masked, but only the masked seed
    # nodes are reconstructed in the loss.
    mask_layer = self.config.mask_node_features_config().make(
        name="mask_node_features"
    )
    masked_graph, is_masked = mask_layer(graph)
    loss_mask = is_masked[seed_idxs]

    # 3. Drop Edge Augmentation (only during training).
    drop_edges = self.config.drop_edges_config().make(
        self.schema, name="drop_edges"
    )
    masked_graph = drop_edges(masked_graph, training=training)

    # 4. Encode.
    enc_rep, enc_graph = encoder(masked_graph, training)

    # 5. Project the representations of all the nodesets to the decoder input,
    # and re-mask the masked target nodes (avoids the identity shortcut). Like
    # the encoder norms, the projections are nodeset-specific: the encoder does
    # not map the different nodesets to a shared latent space.
    decoder_config = self.config.decoder_conv
    dec_feat = decoder_config.embedding_feature
    dec_node_sets = {}
    for nodeset_name in self.schema.node_sets:
      nodeset = enc_graph.node_sets[nodeset_name]
      rep = nodeset.features[EMBEDDING_FEATURE]
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

    # 7. Reconstruction head on the seed nodes.
    recon = nn.Dense(x_init.shape[-1], name="recon_head")(
        decoder_out[seed_idxs]
    )

    return GraphMAEOutput(
        recon=recon,
        x_init=x_init,
        loss_mask=loss_mask,
        enc_rep=enc_rep[seed_idxs],
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
        `(graph, seed_idxs)` of which only the graph is used.
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
    enc_rep = self._forward(graph, None, training=training, embed_only=True)
    assert isinstance(enc_rep, jax.Array)
    return enc_rep
