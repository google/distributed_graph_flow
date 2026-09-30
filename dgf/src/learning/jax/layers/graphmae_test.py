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

"""Tests for the GraphMAE layer and loss utilities."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from absl.testing import absltest
import dataclasses_json
from dgf.src.data import jax_in_memory_graph as jax_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import graphmae
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry  # pylint: disable=g-importing-member
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
from flax import core as flax_core
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


def _make_embedded_hetero_graph(
    num_nodes: int = 6,
    dims: int = 8,
) -> tuple[jax_graph_lib.JaxInMemoryGraph, schema_lib.GraphSchema]:
  """Creates an embedded heterogeneous graph from `gen_test_graph`."""
  raw_schema = gen_test_graph.generate_schema(semantic=True)
  embedded_schema = preprocess.EmbedGraphConfig().output_schema(raw_schema)
  for nodeset_schema in embedded_schema.node_sets.values():
    nodeset_schema.features["embedding"].shape = (dims,)

  rng = np.random.RandomState(42)

  def _make_nodeset_emb() -> jax.Array:
    real_emb = rng.randn(num_nodes - 1, dims).astype(np.float32)
    sentinel = np.zeros((1, dims), dtype=np.float32)
    return jnp.asarray(np.concatenate([real_emb, sentinel], axis=0))

  src = jnp.array([0, 1, 2, 3])
  dst = jnp.array([1, 2, 3, 0])
  graph = jax_graph_lib.JaxInMemoryGraph(
      node_sets={
          name: jax_graph_lib.JaxInMemoryNodeSet(
              features={"embedding": _make_nodeset_emb()},
              num_nodes=num_nodes,
          )
          for name in embedded_schema.node_sets
      },
      edge_sets={
          name: jax_graph_lib.JaxInMemoryEdgeSet(
              adjacency=jnp.stack([src, dst]),
          )
          for name in embedded_schema.edge_sets
      },
  )
  return graph, embedded_schema


def _make_graphmae_config(
    target_nodeset: str = "n1",
    dims: int = 8,
    decoder_dims: int | None = None,
    num_encoder_layers: int = 1,
    **kwargs,
) -> graphmae.GraphMAEConfig:
  """Creates a small `GraphMAEConfig` for unit tests."""
  return graphmae.GraphMAEConfig(
      encoder_conv=hetero_gnn.HeterogeneousGraphConvolutionConfig(
          dims=dims, message_pooling="sum"
      ),
      num_encoder_layers=num_encoder_layers,
      decoder_conv=hetero_gnn.HeterogeneousGraphConvolutionConfig(
          dims=dims if decoder_dims is None else decoder_dims,
          embedding_feature="latent_embedding",
          message_pooling="sum",
      ),
      target_nodeset=target_nodeset,
      **kwargs,
  )


def _init_and_apply(
    model: graphmae.GraphMAE,
    batch: graphmae.Batch | jax_graph_lib.JaxInMemoryGraph,
) -> tuple[Mapping[str, Any], graphmae.GraphMAEOutput]:
  """Initializes `model` and runs a training forward pass on `batch`."""
  params = model.init(
      {
          "params": jax.random.PRNGKey(0),
          "dropout": jax.random.PRNGKey(1),
          "masking": jax.random.PRNGKey(2),
      },
      batch,
      training=True,
  )
  output = model.apply(
      params,
      batch,
      training=True,
      rngs={
          "dropout": jax.random.PRNGKey(3),
          "masking": jax.random.PRNGKey(4),
      },
  )
  assert isinstance(output, graphmae.GraphMAEOutput)
  return params, output


def _projection_param_names(params: Mapping[str, Any]) -> set[str]:
  """Returns the names of the encoder-to-decoder projections in `params`."""
  return {n for n in params["params"] if n.startswith("encoder_to_decoder")}


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class _ParentConfig:
  """A config nesting a registered layer config, like a model config does."""

  layer: Any = layer_registry.field()


class GraphMAETest(absltest.TestCase):

  def test_forward_losses_and_embed(self):
    """End-to-end forward pass, loss calculation, and embed on hetero graph."""
    num_nodes = 6
    dims = 16
    graph, schema = _make_embedded_hetero_graph(num_nodes=num_nodes, dims=dims)

    config = _make_graphmae_config(
        target_nodeset="n1",
        dims=dims,
        num_encoder_layers=2,
        mask_rate=0.5,
        drop_edge_rate=0.5,
        replace_rate=0.0,
    )
    self.assertEqual(config.get_gnn_dims(), dims)
    architecture = config.architecture()
    self.assertIn(
        "MaskNodeFeatures(nodeset='n1', mask_rate=0.5, replace_rate=0.0)",
        architecture,
    )
    self.assertIn("DropEdges(drop_rate=0.5)", architecture)
    self.assertIn("Reconstruction head: embedding", architecture)

    model = config.make(schema)

    seed_idxs = jnp.array([0, 1, 2], dtype=jnp.int32)
    is_real = jnp.array([True] * (num_nodes - 1) + [False])
    batch = (graph, seed_idxs, is_real)

    params, output = _init_and_apply(model, batch)
    # The encoder weights (shared with `embed`) live under the `encoder` scope.
    self.assertContainsSubset(
        {
            "encoder_layer_00",
            "encoder_layer_01",
            "encoder_norm_n1",
            "encoder_norm_n2",
        },
        params["params"]["encoder"],
    )

    self.assertEqual(output.recon.shape, (num_nodes, dims))
    self.assertEqual(output.x_init.shape, (num_nodes, dims))
    self.assertEqual(output.enc_rep.shape, (num_nodes, dims))
    np.testing.assert_array_equal(
        output.seed_mask, [True, True, True, False, False, False]
    )
    self.assertEqual(output.loss_mask.shape, (num_nodes,))
    self.assertFalse(np.any(np.asarray(output.loss_mask & ~output.seed_mask)))

    losses = graphmae.compute_losses(output, config)
    self.assertAlmostEqual(
        float(losses.total()),
        float(losses.reconstruction + losses.variance + losses.covariance),
        places=5,
    )
    self.assertTrue(np.isfinite(float(losses.total())))

    emb = model.apply(params, graph, method=model.embed)
    assert isinstance(emb, jax.Array)
    self.assertEqual(emb.shape, (num_nodes, dims))

  def test_default_seed_and_real_node_masks(self):
    """Passing a graph directly defaults seed_mask to all non-sentinel nodes."""
    num_nodes = 5
    graph, schema = _make_embedded_hetero_graph(num_nodes=num_nodes, dims=8)
    model = _make_graphmae_config(target_nodeset="n1", dims=8).make(schema)

    _, output = _init_and_apply(model, graph)
    np.testing.assert_array_equal(
        output.seed_mask, [True, True, True, True, False]
    )

  def test_decoder_dims_and_encoder_to_decoder_toggle(self):
    """Tests `use_encoder_to_decoder=True` with differing dims vs `False`."""
    graph, schema = _make_embedded_hetero_graph(num_nodes=5, dims=8)

    # 1. With projection to different decoder dimension: exactly one bias-free
    # projection per nodeset.
    config_proj = _make_graphmae_config(dims=8, decoder_dims=6)
    self.assertIn("Dense(6, use_bias=False)", config_proj.architecture())
    params_proj, out_proj = _init_and_apply(config_proj.make(schema), graph)
    self.assertSetEqual(
        _projection_param_names(params_proj),
        {f"encoder_to_decoder_{name}" for name in schema.node_sets},
    )
    for nodeset_name in schema.node_sets:
      projection = params_proj["params"][f"encoder_to_decoder_{nodeset_name}"]
      self.assertEqual(projection["kernel"].shape, (8, 6))
      self.assertNotIn("bias", projection)
    self.assertEqual(out_proj.enc_rep.shape, (5, 8))
    self.assertEqual(out_proj.recon.shape, (5, 8))

    # 2. Without projection (matching dimensions).
    config_no_proj = _make_graphmae_config(
        dims=8, use_encoder_to_decoder=False, norm=None
    )
    self.assertNotIn("use_bias=False", config_no_proj.architecture())
    self.assertNotIn("Norm(layer_norm)", config_no_proj.architecture())
    params_no_proj, out_no_proj = _init_and_apply(
        config_no_proj.make(schema), graph
    )
    self.assertEmpty(_projection_param_names(params_no_proj))
    self.assertEqual(out_no_proj.recon.shape, (5, 8))

  def test_encoder_to_decoder_is_nodeset_specific(self):
    """Each nodeset is projected by its own `encoder_to_decoder_{nodeset}`."""
    graph, schema = _make_embedded_hetero_graph(num_nodes=5, dims=8)
    model = _make_graphmae_config(dims=8, decoder_dims=6).make(schema)
    params, _ = _init_and_apply(model, graph)

    def _capture_filter(module: nn.Module, method_name: str) -> bool:
      del method_name
      return (module.name or "").startswith(
          ("encoder_norm_", "encoder_to_decoder_")
      )

    _, state = model.apply(
        params,
        graph,
        training=True,
        rngs={
            "dropout": jax.random.PRNGKey(3),
            "masking": jax.random.PRNGKey(4),
        },
        capture_intermediates=_capture_filter,
        mutable=["intermediates"],
    )
    intermediates = state["intermediates"]
    for nodeset_name in schema.node_sets:
      # Encoder output of the nodeset, i.e., the input of its projection.
      norm_name = f"encoder_norm_{nodeset_name}"
      (enc_rep,) = intermediates["encoder"][norm_name]["__call__"]
      projection_name = f"encoder_to_decoder_{nodeset_name}"
      (dec_input,) = intermediates[projection_name]["__call__"]
      kernel = params["params"][projection_name]["kernel"]
      np.testing.assert_allclose(
          dec_input, enc_rep @ kernel, rtol=1e-5, atol=1e-5
      )

  def test_serialization_and_registry(self):
    """`GraphMAEConfig` round-trips through JSON, alone and nested."""
    config = _make_graphmae_config(dims=8, decoder_dims=6, num_decoder_layers=2)

    # Serialization, including the nested encoder and decoder configs.
    reconstructed = graphmae.GraphMAEConfig.from_json(config.to_json())
    test_util.assert_are_equal(self, config, reconstructed)

    # Registry: a config nesting a `GraphMAEConfig` can be deserialized.
    self.assertIn("layers.GraphMAEConfig", layer_registry.registered_keys())
    parent = _ParentConfig(layer=config)
    test_util.assert_are_equal(
        self, parent, _ParentConfig.from_json(parent.to_json())
    )


class GraphMAEErrorTest(absltest.TestCase):
  """Configuration validation error tests."""

  def setUp(self):
    super().setUp()
    _, self.schema = _make_embedded_hetero_graph(num_nodes=4, dims=8)

  def test_unknown_target_nodeset_raises(self):
    config = _make_graphmae_config(target_nodeset="unknown")
    with self.assertRaisesRegex(ValueError, "Target nodeset 'unknown'"):
      config.make(self.schema)

  def test_missing_embedding_feature_raises(self):
    raw_schema = gen_test_graph.generate_schema(semantic=True)
    config = _make_graphmae_config(target_nodeset="n1")
    with self.assertRaisesRegex(
        ValueError, "must have an 'embedding' feature"
    ):
      config.make(raw_schema)

  def test_missing_embedding_feature_in_non_target_nodeset_raises(self):
    del self.schema.node_sets["n2"].features["embedding"]
    config = _make_graphmae_config(target_nodeset="n1")
    with self.assertRaisesRegex(
        ValueError, "Nodeset 'n2' must have an 'embedding' feature"
    ):
      config.make(self.schema)

  def test_wrong_embedding_shape_raises(self):
    """Covers the target and non-target nodesets, incl. a broadcastable dim."""
    for nodeset_name, shape in [("n1", (4,)), ("n2", (1,)), ("n2", None)]:
      with self.subTest(nodeset=nodeset_name, shape=shape):
        _, schema = _make_embedded_hetero_graph(num_nodes=4, dims=8)
        schema.node_sets[nodeset_name].features["embedding"].shape = shape
        config = _make_graphmae_config(target_nodeset="n1", dims=8)
        with self.assertRaisesRegex(
            ValueError, rf"nodeset '{nodeset_name}' must have shape \(8,\)"
        ):
          config.make(schema)

  def test_one_dimensional_embedding_raises(self):
    """The cosine reconstruction error is degenerate in one dimension."""
    _, schema = _make_embedded_hetero_graph(num_nodes=4, dims=1)
    config = _make_graphmae_config(dims=1)
    with self.assertRaisesRegex(ValueError, "must be at least 2"):
      config.make(schema)

  def test_different_decoder_dims_without_projection_raises(self):
    config = _make_graphmae_config(
        dims=8, decoder_dims=6, use_encoder_to_decoder=False
    )
    with self.assertRaisesRegex(ValueError, r"decoder_conv\.dims"):
      config.make(self.schema)

  def test_non_default_embedding_feature_raises(self):
    config = dataclasses.replace(
        _make_graphmae_config(),
        encoder_conv=hetero_gnn.HeterogeneousGraphConvolutionConfig(
            dims=8, embedding_feature="other"
        ),
    )
    with self.assertRaisesRegex(ValueError, "embedding_feature"):
      config.make(self.schema)

  def test_none_num_nodes_raises(self):
    graph, schema = _make_embedded_hetero_graph(num_nodes=4, dims=8)
    bad_n1 = dataclasses.replace(graph.node_sets["n1"], num_nodes=None)
    bad_graph = dataclasses.replace(
        graph, node_sets={**graph.node_sets, "n1": bad_n1}
    )
    model = _make_graphmae_config().make(schema)
    with self.assertRaisesRegex(ValueError, "num_nodes for n1 cannot be None"):
      _init_and_apply(model, bad_graph)

  def test_batch_tuple_with_explicit_seed_or_real_nodes_raises(self):
    graph, schema = _make_embedded_hetero_graph(num_nodes=4, dims=8)
    model = _make_graphmae_config().make(schema)
    seed_idxs = jnp.array([0, 1], dtype=jnp.int32)
    is_real = jnp.array([True, True, True, False])
    batch = (graph, seed_idxs, is_real)
    rngs = {
        "params": jax.random.PRNGKey(0),
        "dropout": jax.random.PRNGKey(1),
        "masking": jax.random.PRNGKey(2),
    }
    for kwargs in ({"seed_idxs": seed_idxs}, {"is_real_node": is_real}):
      with self.subTest(argument=next(iter(kwargs))):
        with self.assertRaisesRegex(ValueError, "cannot be given when"):
          model.init(rngs, batch, training=True, **kwargs)


class GraphMAELossTest(absltest.TestCase):
  """Unit tests for `sce_loss`, `vicreg_loss`, and `compute_losses`."""

  def test_sce_loss_identical_opposite_and_alpha_variants(self):
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    self.assertAlmostEqual(
        float(graphmae.sce_loss(x, x, alpha=2.0)), 0.0, places=5
    )
    self.assertAlmostEqual(
        float(graphmae.sce_loss(x, -x, alpha=2.0)), 4.0, places=5
    )

    y = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    loss_a1 = float(graphmae.sce_loss(x[:, :2], y, alpha=1.0))
    loss_a3 = float(graphmae.sce_loss(x[:, :2], y, alpha=3.0))
    self.assertGreater(loss_a1, 0.0)
    self.assertNotAlmostEqual(loss_a1, loss_a3, places=3)

  def test_sce_loss_identical_inputs_non_integer_alpha(self):
    """sce_loss(x, x) with non-integer alpha is finite and ~0 under float32."""
    x = jnp.asarray(np.random.RandomState(0).randn(256, 16).astype(np.float32))
    loss = graphmae.sce_loss(x, x, alpha=2.5)
    self.assertTrue(np.isfinite(float(loss)))
    self.assertAlmostEqual(float(loss), 0.0, places=5)
    grad = jax.grad(lambda v: graphmae.sce_loss(v, x, alpha=2.5))(x)
    self.assertTrue(np.all(np.isfinite(np.asarray(grad))))

  def test_sce_loss_with_mask(self):
    x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    mask = jnp.array([True, False, True])
    self.assertAlmostEqual(
        float(graphmae.sce_loss(x, y, alpha=2.0, mask=mask)), 0.0, places=5
    )
    self.assertGreater(float(graphmae.sce_loss(x, y, alpha=2.0)), 0.1)

  def test_vicreg_loss_and_is_real_mask(self):
    z = jnp.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [999.0, 999.0],  # padded sentinel node
    ])
    is_real = jnp.array([True, True, True, True, False])
    var_masked, cov_masked = graphmae.vicreg_loss(z, is_real=is_real)
    var_real, cov_real = graphmae.vicreg_loss(z[:4])
    self.assertAlmostEqual(float(var_masked), float(var_real), places=5)
    self.assertAlmostEqual(
        float(var_real), 1.0 - np.sqrt(0.5 + 1e-4), places=5
    )
    self.assertAlmostEqual(float(cov_masked), 0.0, places=6)

  def test_compute_losses_empty_mask_and_regularization_options(self):
    output = graphmae.GraphMAEOutput(
        recon=jnp.ones((3, 2)),
        x_init=jnp.array([[1.0, 0.0], [-1.0, 0.0], [5.0, 5.0]]),
        loss_mask=jnp.array([False, False, False]),
        seed_mask=jnp.array([True, True, False]),
        enc_rep=jnp.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]),
    )

    # 1. With empty loss_mask, reconstruction loss is 0.0, not NaN.
    cfg_enc_only = _make_graphmae_config(regularize_input_embeddings=False)
    losses_enc = graphmae.compute_losses(output, cfg_enc_only)
    self.assertEqual(float(losses_enc.reconstruction), 0.0)
    self.assertAlmostEqual(
        float(losses_enc.variance),
        cfg_enc_only.variance_regularization_weight
        * (1.0 - np.sqrt(0.25 + 1e-4)),
        places=5,
    )

    # 2. With regularize_input_embeddings=True, x_init variance/covariance is
    # included.
    cfg_both = _make_graphmae_config(regularize_input_embeddings=True)
    losses_both = graphmae.compute_losses(output, cfg_both)
    self.assertGreater(float(losses_both.variance), float(losses_enc.variance))

    # 3. With zero regularization weights, variance and covariance are 0.
    cfg_no_reg = _make_graphmae_config(
        variance_regularization_weight=0.0,
        covariance_regularization_weight=0.0,
    )
    losses_no_reg = graphmae.compute_losses(output, cfg_no_reg)
    self.assertEqual(float(losses_no_reg.variance), 0.0)
    self.assertEqual(float(losses_no_reg.covariance), 0.0)

  def test_stop_gradient_target_blocks_direct_target_collapse_gradient(self):
    recon = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    x_init = jnp.array([[0.5, 0.5], [-0.5, 0.5]])
    mask = jnp.array([True, True])

    def recon_loss(x_t, stop_grad: bool):
      cfg = _make_graphmae_config(
          stop_gradient_target=stop_grad,
          variance_regularization_weight=0.0,
          covariance_regularization_weight=0.0,
      )
      out = graphmae.GraphMAEOutput(
          recon=recon,
          x_init=x_t,
          loss_mask=mask,
          seed_mask=mask,
          enc_rep=recon,
      )
      return graphmae.compute_losses(out, cfg).reconstruction

    grad_stopped = jax.grad(lambda x: recon_loss(x, stop_grad=True))(x_init)
    grad_unstopped = jax.grad(lambda x: recon_loss(x, stop_grad=False))(x_init)
    np.testing.assert_allclose(grad_stopped, 0.0)
    self.assertGreater(float(jnp.linalg.norm(grad_unstopped)), 1e-2)


# Dimension of the learned input embeddings in the anti-collapse tests.
_TOY_DIMS = 8
# Raw node features of the graphs of the anti-collapse tests.
_RAW_FEATURES = "raw_features"


class _ToyEmbedAndGraphMAE(nn.Module):
  """Embeds the raw node features and applies GraphMAE.

  As recommended for the GraphMAE input embeddings, the raw node features are
  embedded with a per-nodeset `standard.ingest_feature`, trained jointly.
  """

  schema: schema_lib.GraphSchema
  graphmae_config: graphmae.GraphMAEConfig

  @nn.compact
  def __call__(
      self, graph: jax_graph_lib.JaxInMemoryGraph, training: bool
  ) -> graphmae.GraphMAEOutput:
    dims = self.graphmae_config.get_gnn_dims()
    node_sets = {}
    for name in self.schema.node_sets:
      ns = graph.node_sets[name]
      proj = standard.ingest_feature(dims).make(name=f"pre_mlp_{name}")(
          ns.features[_RAW_FEATURES]
      )
      node_sets[name] = jax_graph_lib.JaxInMemoryNodeSet(
          num_nodes=ns.num_nodes, features={"embedding": proj}
      )
    embedded_graph = jax_graph_lib.JaxInMemoryGraph(
        node_sets=node_sets, edge_sets=graph.edge_sets
    )
    return self.graphmae_config.make(self.schema, name="graphmae")(
        embedded_graph, training=training
    )


def _make_toy_classification_graph(
    dims: int,
) -> tuple[jax_graph_lib.JaxInMemoryGraph, schema_lib.GraphSchema]:
  """Creates a padded graph from `gen_toy_classification_dataset`.

  The raw node features are the numerical features of the dataset (excluding
  labels and primary IDs): two for N1 and one for N2. The schema is the one of
  the embedded graph, i.e., declares the `dims`-dimensional `"embedding"`
  computed from the raw features by `_ToyEmbedAndGraphMAE`.

  Args:
    dims: Dimension of the learned node feature embeddings.

  Returns:
    The graph with the raw node features (`_RAW_FEATURES`), and the schema of
    the embedded graph.
  """
  raw_graph, _ = gen_test_graph.gen_toy_classification_dataset(
      num_n1_nodes=32, num_n2_nodes=48, random_seed=0
  )
  n1_feats = np.stack(
      [
          raw_graph.node_sets["N1"].features["f1"].astype(np.float32),
          raw_graph.node_sets["N1"].features["f3"].astype(np.float32),
      ],
      axis=-1,
  )
  n1_feats = np.concatenate(
      [n1_feats, np.zeros((1, 2), dtype=np.float32)], axis=0
  )
  n2_feats = (
      raw_graph.node_sets["N2"].features["f2"].astype(np.float32)[:, None]
  )
  n2_feats = np.concatenate(
      [n2_feats, np.zeros((1, 1), dtype=np.float32)], axis=0
  )

  embedding_schema = schema_lib.FeatureSchema(
      format=schema_lib.FeatureFormat.FLOAT_32,
      semantic=schema_lib.FeatureSemantic.EMBEDDING,
      shape=(dims,),
  )
  schema = schema_lib.GraphSchema(
      node_sets={
          "N1": schema_lib.NodeSchema(features={"embedding": embedding_schema}),
          "N2": schema_lib.NodeSchema(features={"embedding": embedding_schema}),
      },
      edge_sets={
          "N1_to_N1": schema_lib.EdgeSchema(source="N1", target="N1"),
          "N1_to_N2": schema_lib.EdgeSchema(source="N1", target="N2"),
      },
  )
  graph = jax_graph_lib.JaxInMemoryGraph(
      node_sets={
          "N1": jax_graph_lib.JaxInMemoryNodeSet(
              num_nodes=n1_feats.shape[0],
              features={_RAW_FEATURES: jnp.asarray(n1_feats)},
          ),
          "N2": jax_graph_lib.JaxInMemoryNodeSet(
              num_nodes=n2_feats.shape[0],
              features={_RAW_FEATURES: jnp.asarray(n2_feats)},
          ),
      },
      edge_sets={
          name: jax_graph_lib.JaxInMemoryEdgeSet(
              adjacency=jnp.asarray(raw_graph.edge_sets[name].adjacency)
          )
          for name in schema.edge_sets
      },
  )
  return graph, schema


def _collapse_input_embeddings(
    params: Mapping[str, Any], schema: schema_lib.GraphSchema
) -> dict[str, Any]:
  """Returns `params` with nearly collapsed learned input embeddings.

  Sets the final LayerNorm of each `pre_mlp_*` block to a tiny scale and to a
  bias shared by all the nodes, so that all the input embeddings are nearly
  identical. This emulates the state reached by a collapsing training run.

  Args:
    params: Variables of `_ToyEmbedAndGraphMAE`.
    schema: The embedded schema.

  Returns:
    The modified variables.
  """
  params = flax_core.unfreeze(params)
  for nodeset_name in schema.node_sets:
    # `norm_2` is the final LayerNorm of the "LAN" `standard.ingest_feature`.
    norm = params["params"][f"pre_mlp_{nodeset_name}"]["norm_2"]
    norm["scale"] = jnp.full_like(norm["scale"], 1e-2)
    norm["bias"] = jnp.linspace(-1.0, 1.0, norm["bias"].shape[0])
  return params


def _train_toy_model(
    config: graphmae.GraphMAEConfig,
    graph: jax_graph_lib.JaxInMemoryGraph,
    schema: schema_lib.GraphSchema,
    num_steps: int,
    checkpoints: Sequence[int],
    collapse_input_embeddings: bool = False,
) -> dict[int, graphmae.GraphMAEOutput]:
  """Jointly trains the input embeddings and GraphMAE on `graph`.

  Args:
    config: The GraphMAE configuration, including the anti-collapse options.
    graph: Graph with raw node features.
    schema: The embedded schema.
    num_steps: Number of Adam steps.
    checkpoints: Steps at which to return the forward pass output.
    collapse_input_embeddings: If True, starts from nearly collapsed input
      embeddings. See `_collapse_input_embeddings`.

  Returns:
    The forward pass outputs at the `checkpoints` steps, computed before the
    update of the step.
  """
  model = _ToyEmbedAndGraphMAE(schema=schema, graphmae_config=config)
  rng = jax.random.PRNGKey(0)
  params = model.init(
      {
          "params": rng,
          "dropout": jax.random.fold_in(rng, 1),
          "masking": jax.random.fold_in(rng, 2),
      },
      graph,
      training=True,
  )
  if collapse_input_embeddings:
    params = _collapse_input_embeddings(params, schema)
  optimizer = optax.adam(learning_rate=1e-2)
  opt_state = optimizer.init(params)

  @jax.jit
  def step_fn(p, opt_s, step_rng):
    def loss_fn(w):
      out = model.apply(
          w,
          graph,
          training=True,
          rngs={
              "dropout": jax.random.fold_in(step_rng, 1),
              "masking": jax.random.fold_in(step_rng, 2),
          },
      )
      return graphmae.compute_losses(out, config).total(), out

    (_, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
    updates, new_opt_s = optimizer.update(grads, opt_s, p)
    return optax.apply_updates(p, updates), new_opt_s, out

  outputs = {}
  for step in range(num_steps + 1):
    new_params, opt_state, out = step_fn(
        params, opt_state, jax.random.fold_in(rng, 10 + step)
    )
    if step in checkpoints:
      outputs[step] = out
    params = new_params
  return outputs


def _collapse_metrics(z: jax.Array, mask: jax.Array) -> dict[str, float]:
  """Per-dimension std and mean pairwise cosine of `z` on the `mask` rows."""
  z = np.asarray(z)[np.asarray(mask)]
  std = z.std(axis=0)
  z_normalized = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
  num = z.shape[0]
  cosine = z_normalized @ z_normalized.T
  return {
      "std_mean": float(std.mean()),
      "std_min": float(std.min()),
      "cos": float((cosine.sum() - np.trace(cosine)) / (num * (num - 1))),
  }


def _toy_config(anti_collapse: bool) -> graphmae.GraphMAEConfig:
  """Config of the anti-collapse tests, with or without the mechanisms."""
  if anti_collapse:
    anti_collapse_kwargs = dict(
        variance_regularization_weight=1.0,
        covariance_regularization_weight=0.05,
        stop_gradient_target=True,
        regularize_input_embeddings=True,
    )
  else:
    anti_collapse_kwargs = dict(
        variance_regularization_weight=0.0,
        covariance_regularization_weight=0.0,
        stop_gradient_target=False,
    )
  return _make_graphmae_config(
      target_nodeset="N1",
      dims=_TOY_DIMS,
      num_encoder_layers=1,
      **anti_collapse_kwargs,
  )


class GraphMAEAntiCollapseTest(absltest.TestCase):
  """Anti-collapse when jointly training the input embeddings and GraphMAE.

  Without the anti-collapse mechanisms, the input embeddings (`x_init`) and the
  encoder representations (`enc_rep`) collapse progressively, i.e., become
  aligned (high mean pairwise cosine) with a small per-dimension std. A full
  collapse takes long training runs, which are too slow for unit tests.
  Instead, the tests train for a few steps, starting either from a random
  initialization, where the collapse starts, or from nearly collapsed input
  embeddings, i.e., the end state of a long collapsing run. Each test also
  checks that a control run without the mechanisms collapses, so that the test
  is able to detect a collapse.
  """

  def setUp(self):
    super().setUp()
    self.graph, self.schema = _make_toy_classification_graph(_TOY_DIMS)

  def _train(
      self,
      *,
      anti_collapse: bool,
      num_steps: int,
      checkpoints: Sequence[int],
      collapse_input_embeddings: bool = False,
  ) -> dict[int, tuple[dict[str, float], dict[str, float]]]:
    """Returns the `x_init` and `enc_rep` collapse metrics at `checkpoints`."""
    outputs = _train_toy_model(
        _toy_config(anti_collapse),
        self.graph,
        self.schema,
        num_steps,
        checkpoints,
        collapse_input_embeddings,
    )
    return {
        step: (
            _collapse_metrics(out.x_init, out.seed_mask),
            _collapse_metrics(out.enc_rep, out.seed_mask),
        )
        for step, out in outputs.items()
    }

  def test_prevents_collapse_from_random_init(self):
    num_steps = 25
    metrics = self._train(
        anti_collapse=True, num_steps=num_steps, checkpoints=[num_steps]
    )
    x, enc = metrics[num_steps]
    self.assertGreater(x["std_mean"], 0.5)
    self.assertGreater(x["std_min"], 0.2)
    self.assertGreater(enc["std_mean"], 0.5)
    self.assertGreater(enc["std_min"], 0.2)
    self.assertLess(enc["cos"], 0.3)

    # Control: without the mechanisms, the representations start collapsing.
    control_metrics = self._train(
        anti_collapse=False, num_steps=num_steps, checkpoints=[num_steps]
    )
    control_x, control_enc = control_metrics[num_steps]
    self.assertLess(control_x["std_mean"], 0.5)
    self.assertGreater(control_enc["cos"], 0.6)

  def test_recovers_from_collapsed_input_embeddings(self):
    num_steps = 100
    metrics = self._train(
        anti_collapse=True,
        num_steps=num_steps,
        checkpoints=[0, num_steps],
        collapse_input_embeddings=True,
    )
    initial_x, _ = metrics[0]
    self.assertLess(initial_x["std_mean"], 0.01)
    self.assertGreater(initial_x["cos"], 0.99)
    x, _ = metrics[num_steps]
    self.assertGreater(x["std_mean"], 0.3)
    self.assertLess(x["cos"], 0.8)

    # Control: without the mechanisms, the input embeddings stay collapsed.
    control_metrics = self._train(
        anti_collapse=False,
        num_steps=num_steps,
        checkpoints=[num_steps],
        collapse_input_embeddings=True,
    )
    control_x, _ = control_metrics[num_steps]
    self.assertLess(control_x["std_mean"], 0.05)
    self.assertGreater(control_x["cos"], 0.95)


if __name__ == "__main__":
  absltest.main()
