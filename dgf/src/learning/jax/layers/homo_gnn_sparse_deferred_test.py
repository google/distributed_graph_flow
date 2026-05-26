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

"""Tests for SparseDeferred GNNs."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.jax import test_utils
from dgf.src.learning.jax.layers import homo_gnn_sparse_deferred as gnn_lib
import flax.linen as nn
import jax
import sparse_deferred.jax as sdjnp
from sparse_deferred.structs import graph_struct as graph_struct_lib


GraphStruct = graph_struct_lib.GraphStruct


def _forward(model, dummy_input: jax.Array) -> jax.Array:
  params = model.init(jax.random.PRNGKey(42), dummy_input)
  return model.apply(params, dummy_input)


class GnnConfigTest(absltest.TestCase):

  def test_jax_base_config_is_abstract(self):
    with self.assertRaises(TypeError):
      _ = gnn_lib.JaxBaseConfig()

  def test_mpnn_basic(self):
    cfg = gnn_lib.MPNNConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="leaky_relu",
        message_pooling="sum",
        name_prefix="mpnn",
    )
    model = cfg.make()
    self.assertIsInstance(model, gnn_lib.MPNN)
    output = _forward(model, test_utils.generate_test_graph())
    self.assertEqual(gnn_lib.get_node_hidden_state(output).shape, (4, 8))

  def test_mpnn_with_base_overrides(self):
    cfg = gnn_lib.MPNNConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="leaky_relu",
        message_pooling="sum",
        name_prefix="mpnn",
        dropout_rate=0.1,
        nodeset_name="test_nodes",
    )
    self.assertEqual(cfg.dropout_rate, 0.1)
    self.assertEqual(cfg.nodeset_name, "test_nodes")

  def test_mpnn_validation(self):
    with self.assertRaisesRegex(ValueError, "dropout_rate must be in"):
      gnn_lib.MPNNConfig(
          num_layers=2,
          hidden_dim=8,
          dropout_rate=1.1,
      )

    with self.assertRaisesRegex(
        AttributeError, "dtype foo is not found in jax.numpy"
    ):
      gnn_lib.MPNNConfig(
          num_layers=2,
          hidden_dim=8,
          matrix_precision="foo",
      )

  # TODO(deniscalin): Pass `nodeset_name` into the `get_node_hidden_state` and
  # `generate_test_graph` functions.
  def test_mpnn_serialize_and_deserialize_with_plus(self):
    cfg = gnn_lib.MPNNConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="leaky_relu",
        message_pooling="sum",
        name_prefix="mpnn",
        dropout_rate=0.1,
        nodeset_name="test_nodes",
        enable_gnn_plus=True,
    )
    temp_file = self.create_tempfile()
    cfg.json_save(temp_file.full_path)
    loaded_cfg = gnn_lib.MPNNConfig.json_load(temp_file.full_path)
    self.assertEqual(cfg, loaded_cfg)
    model = loaded_cfg.make()
    self.assertIsInstance(model, gnn_lib.MPNN)

    hidden_state = gnn_lib.get_node_hidden_state(
        _forward(model, test_utils.generate_test_graph())
    )
    self.assertEqual(hidden_state.shape, (4, 8))

  def test_gin_basic(self):
    cfg = gnn_lib.GINConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="relu",
        epsilon=0.1,
        num_mlp_layers=2,
        name_prefix="gin",
    )
    model = cfg.make()
    self.assertIsInstance(model, gnn_lib.GIN)
    output = _forward(model, test_utils.generate_test_graph())
    self.assertEqual(gnn_lib.get_node_hidden_state(output).shape, (4, 8))

  def test_gin_with_base_overrides(self):
    cfg = gnn_lib.GINConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="relu",
        epsilon=0.1,
        num_mlp_layers=2,
        name_prefix="gin",
        dropout_rate=0.1,
        matrix_precision="bfloat16",
        pointwise_norm_precision="float32",
        nodeset_name="test_nodes",
        edgeset_name="test_edges",
        input_node_feature="test_input_node_feature",
        output_node_feature="test_output_node_feature",
    )
    self.assertEqual(cfg.dropout_rate, 0.1)
    self.assertEqual(cfg.matrix_precision, "bfloat16")
    self.assertEqual(cfg.pointwise_norm_precision, "float32")
    self.assertEqual(cfg.nodeset_name, "test_nodes")
    self.assertEqual(cfg.edgeset_name, "test_edges")
    self.assertEqual(cfg.input_node_feature, "test_input_node_feature")
    self.assertEqual(cfg.output_node_feature, "test_output_node_feature")

  def test_gin_validation(self):
    with self.assertRaisesRegex(ValueError, "dropout_rate must be in"):
      gnn_lib.GINConfig(
          num_layers=2,
          hidden_dim=8,
          dropout_rate=1.1,
      )

    with self.assertRaisesRegex(
        AttributeError, "dtype foo is not found in jax.numpy"
    ):
      gnn_lib.GINConfig(
          num_layers=2,
          hidden_dim=8,
          matrix_precision="foo",
      )

    with self.assertRaisesRegex(
        AttributeError, "dtype foo is not found in jax.numpy"
    ):
      gnn_lib.GINConfig(
          num_layers=2,
          hidden_dim=8,
          pointwise_norm_precision="foo",
      )

  def test_gin_serialize_and_deserialize_with_plus(self):
    cfg = gnn_lib.GINConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="relu",
        epsilon=0.1,
        num_mlp_layers=2,
        name_prefix="gin",
        dropout_rate=0.1,
        matrix_precision="bfloat16",
        pointwise_norm_precision="float32",
        nodeset_name="test_nodes",
        edgeset_name="test_edges",
        input_node_feature="test_input_node_feature",
        output_node_feature="test_output_node_feature",
        enable_gnn_plus=True,
    )
    temp_file = self.create_tempfile()
    cfg.json_save(temp_file.full_path)
    loaded_cfg = gnn_lib.GINConfig.json_load(temp_file.full_path)
    self.assertEqual(cfg, loaded_cfg)
    model = loaded_cfg.make()
    self.assertIsInstance(model, gnn_lib.GIN)

    hidden_state = gnn_lib.get_node_hidden_state(
        _forward(
            model,
            test_utils.generate_test_graph(
                dim=cfg.hidden_dim,
                nodeset_name=cfg.nodeset_name,
                edgeset_name=cfg.edgeset_name,
                node_feature_name=cfg.input_node_feature,
            ),
        ),
        nodeset_name=cfg.nodeset_name,
        node_feature_name=cfg.output_node_feature,
    )
    self.assertEqual(hidden_state.shape, (4, 8))

  def test_gcn_basic(self):
    cfg = gnn_lib.GCNConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="leaky_relu",
        name_prefix="gcn",
    )
    model = cfg.make()
    self.assertIsInstance(model, gnn_lib.GCN)
    output = _forward(model, test_utils.generate_test_graph())
    self.assertEqual(gnn_lib.get_node_hidden_state(output).shape, (4, 8))

  def test_gcn_with_base_overrides(self):
    cfg = gnn_lib.GCNConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="leaky_relu",
        name_prefix="gcn",
        dropout_rate=0.1,
        matrix_precision="bfloat16",
        pointwise_norm_precision="float32",
        nodeset_name="test_nodes",
        edgeset_name="test_edges",
        input_node_feature="test_input_node_feature",
        output_node_feature="test_output_node_feature",
    )
    self.assertEqual(cfg.dropout_rate, 0.1)
    self.assertEqual(cfg.matrix_precision, "bfloat16")
    self.assertEqual(cfg.pointwise_norm_precision, "float32")
    self.assertEqual(cfg.nodeset_name, "test_nodes")
    self.assertEqual(cfg.edgeset_name, "test_edges")
    self.assertEqual(cfg.input_node_feature, "test_input_node_feature")
    self.assertEqual(cfg.output_node_feature, "test_output_node_feature")

  def test_gcn_validation(self):
    with self.assertRaisesRegex(ValueError, "dropout_rate must be in"):
      gnn_lib.GCNConfig(
          num_layers=2,
          hidden_dim=8,
          dropout_rate=1.1,
      )

    with self.assertRaisesRegex(
        AttributeError, "dtype foo is not found in jax.numpy"
    ):
      gnn_lib.GCNConfig(
          num_layers=2,
          hidden_dim=8,
          matrix_precision="foo",
      )

    with self.assertRaisesRegex(
        AttributeError, "dtype foo is not found in jax.numpy"
    ):
      gnn_lib.GCNConfig(
          num_layers=2,
          hidden_dim=8,
          pointwise_norm_precision="foo",
      )

  def test_gcn_serialize_and_deserialize_with_plus(self):
    cfg = gnn_lib.GCNConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="leaky_relu",
        name_prefix="gcn",
        dropout_rate=0.1,
        matrix_precision="bfloat16",
        pointwise_norm_precision="float32",
        nodeset_name="test_nodes",
        edgeset_name="test_edges",
        input_node_feature="test_input_node_feature",
        output_node_feature="test_output_node_feature",
        enable_gnn_plus=True,
    )
    temp_file = self.create_tempfile()
    cfg.json_save(temp_file.full_path)
    loaded_cfg = gnn_lib.GCNConfig.json_load(temp_file.full_path)
    self.assertEqual(cfg, loaded_cfg)
    model = loaded_cfg.make()
    self.assertIsInstance(model, gnn_lib.GCN)

    hidden_state = gnn_lib.get_node_hidden_state(
        _forward(
            model,
            test_utils.generate_test_graph(
                dim=cfg.hidden_dim,
                nodeset_name=cfg.nodeset_name,
                edgeset_name=cfg.edgeset_name,
                node_feature_name=cfg.input_node_feature,
            ),
        ),
        nodeset_name=cfg.nodeset_name,
        node_feature_name=cfg.output_node_feature,
    )
    self.assertEqual(hidden_state.shape, (4, 8))

  def test_projector_basic(self):
    cfg = gnn_lib.ProjectorConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="tanh",
        name_prefix="projector",
    )
    model = cfg.make()
    self.assertIsInstance(model, gnn_lib.Projector)
    # Input dim different from hidden_dim
    output = _forward(model, test_utils.generate_test_graph(dim=16))
    # Output features are written to the input feature name
    # per `Projector` implementation.
    output_features = gnn_lib.get_node_features(
        output,
        nodeset_name=cfg.nodeset_name,
        feature_name=cfg.input_node_feature,
    )
    self.assertEqual(output_features.shape, (4, 8))

  def test_projector_with_base_overrides(self):
    cfg = gnn_lib.ProjectorConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="tanh",
        name_prefix="projector",
        dropout_rate=0.1,
        matrix_precision="bfloat16",
        pointwise_norm_precision="float32",
        nodeset_name="test_nodes",
        input_node_feature="test_input_node_feature",
        output_node_feature="test_input_node_feature",
    )
    self.assertEqual(cfg.dropout_rate, 0.1)
    self.assertEqual(cfg.matrix_precision, "bfloat16")
    self.assertEqual(cfg.pointwise_norm_precision, "float32")
    self.assertEqual(cfg.nodeset_name, "test_nodes")
    self.assertEqual(cfg.input_node_feature, "test_input_node_feature")
    self.assertEqual(cfg.output_node_feature, "test_input_node_feature")

  def test_projector_validation(self):
    with self.assertRaisesRegex(ValueError, "dropout_rate must be in"):
      gnn_lib.ProjectorConfig(
          num_layers=2,
          hidden_dim=8,
          dropout_rate=1.1,
      )

    with self.assertRaisesRegex(
        AttributeError, "dtype foo is not found in jax.numpy"
    ):
      gnn_lib.ProjectorConfig(
          num_layers=2,
          hidden_dim=8,
          matrix_precision="foo",
      )

    with self.assertRaisesRegex(
        AttributeError, "dtype foo is not found in jax.numpy"
    ):
      gnn_lib.ProjectorConfig(
          num_layers=2,
          hidden_dim=8,
          pointwise_norm_precision="foo",
      )

  def test_projector_serialize_and_deserialize(self):
    cfg = gnn_lib.ProjectorConfig(
        num_layers=2,
        hidden_dim=8,
        use_bias=True,
        activation_fn="tanh",
        name_prefix="projector",
        dropout_rate=0.1,
        matrix_precision="bfloat16",
        pointwise_norm_precision="float32",
        nodeset_name="test_nodes",
        input_node_feature="test_input_node_feature",
        output_node_feature="test_input_node_feature",
    )
    temp_file = self.create_tempfile()
    cfg.json_save(temp_file.full_path)
    loaded_cfg = gnn_lib.ProjectorConfig.json_load(temp_file.full_path)
    self.assertEqual(cfg, loaded_cfg)
    model = loaded_cfg.make()
    self.assertIsInstance(model, gnn_lib.Projector)

    output = _forward(
        model,
        test_utils.generate_test_graph(
            dim=16,  # Input dim different from hidden_dim
            nodeset_name=cfg.nodeset_name,
            node_feature_name=cfg.input_node_feature,
        ),
    )
    # Output features are written to the input feature name
    # per `Projector` implementation.
    output_features = gnn_lib.get_node_features(
        output,
        nodeset_name=cfg.nodeset_name,
        feature_name=cfg.input_node_feature,
    )
    self.assertEqual(output_features.shape, (4, cfg.hidden_dim))


class SDTests(parameterized.TestCase):

  def test_mpnn_basic(self):
    dummy_input = test_utils.generate_test_graph()
    model = gnn_lib.MPNN(num_layers=2, hidden_dim=8)
    params = model.init(jax.random.PRNGKey(42), dummy_input)
    output = model.apply(params, dummy_input)
    hidden_state = gnn_lib.get_node_hidden_state(output)
    self.assertEqual(hidden_state.shape, (4, 8))

  def test_projection_with_mpnn(self):
    dummy_input = test_utils.generate_test_graph()

    class MPNNWithProjector(nn.Module):
      hidden_dim: int

      @nn.compact
      def __call__(self, graph: GraphStruct, training: bool = False):
        graph = gnn_lib.Projector(num_layers=1, hidden_dim=self.hidden_dim)(
            graph, training
        )
        graph = gnn_lib.MPNN(num_layers=2, hidden_dim=self.hidden_dim)(
            graph, training
        )
        return graph

    # Project the input D = 8 -> D = 4 then convolve.
    model = MPNNWithProjector(hidden_dim=4)
    params = model.init(jax.random.PRNGKey(42), dummy_input)
    output = model.apply(params, dummy_input, training=False)
    self.assertEqual(gnn_lib.get_node_hidden_state(output).shape, (4, 4))

  def test_mpnn_plus(self):
    dummy_input = test_utils.generate_test_graph()
    model = gnn_lib.MPNN(num_layers=2, hidden_dim=8, enable_gnn_plus=True)
    params = model.init(jax.random.PRNGKey(42), dummy_input)
    output = model.apply(params, dummy_input)
    hidden_state = gnn_lib.get_node_hidden_state(output)
    self.assertEqual(hidden_state.shape, (4, 8))

  @parameterized.named_parameters(
      dict(
          testcase_name="gcn_basic",
          num_layers=2,
          hidden_dim=4,
          enable_gnn_plus=False,
      ),
      dict(
          testcase_name="gcn_plus",
          num_layers=2,
          hidden_dim=4,
          enable_gnn_plus=True,
      ),
  )
  def test_gcn(self, num_layers, hidden_dim, enable_gnn_plus):

    model = gnn_lib.GCN(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        enable_gnn_plus=enable_gnn_plus,
    )

    dummy_input = test_utils.generate_test_graph(hidden_dim)
    num_nodes = dummy_input.get_num_nodes(
        engine=sdjnp.engine, node_name="nodes"
    )
    params = model.init(jax.random.PRNGKey(42), dummy_input)
    output = model.apply(params, dummy_input)
    hidden_state = gnn_lib.get_node_hidden_state(output)
    self.assertEqual(hidden_state.shape, (num_nodes, hidden_dim))

  @parameterized.named_parameters(
      dict(
          testcase_name="gin_basic",
          num_layers=2,
          hidden_dim=4,
          enable_gnn_plus=False,
      ),
      dict(
          testcase_name="gin_plus",
          num_layers=2,
          hidden_dim=4,
          enable_gnn_plus=True,
      ),
  )
  def test_gin(self, num_layers, hidden_dim, enable_gnn_plus):
    dummy_input = test_utils.generate_test_graph(hidden_dim)
    model = gnn_lib.GIN(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        enable_gnn_plus=enable_gnn_plus,
    )
    num_nodes = dummy_input.get_num_nodes(
        engine=sdjnp.engine, node_name="nodes"
    )
    params = model.init(jax.random.PRNGKey(42), dummy_input)
    output = model.apply(params, dummy_input)
    hidden_state = gnn_lib.get_node_hidden_state(output)
    self.assertEqual(hidden_state.shape, (num_nodes, hidden_dim))

  def test_labeling_trick_features(self):
    node_features = gnn_lib.labeling_trick_features(
        num_nodes=4, idx=1, hidden_dim=4
    )
    # Expected output:
    # [0 0 0 0]
    # [1 1 1 1]
    # [0 0 0 0]
    # [0 0 0 0]

    self.assertEqual(node_features.shape, (4, 4))
    self.assertEqual(node_features[0].sum(), 0)
    self.assertEqual(node_features[1].sum(), 4)
    self.assertEqual(node_features[2].sum(), 0)
    self.assertEqual(node_features[3].sum(), 0)

  @parameterized.named_parameters(
      dict(
          testcase_name="conditional_gin_basic",
          num_layers=2,
          hidden_dim=4,
          enable_gnn_plus=False,
      ),
      dict(
          testcase_name="conditional_gin_plus",
          num_layers=2,
          hidden_dim=4,
          enable_gnn_plus=True,
      ),
  )
  def test_conditional_gin(self, num_layers, hidden_dim, enable_gnn_plus):
    dummy_input = test_utils.generate_test_graph(hidden_dim)
    model = gnn_lib.ConditionalGIN(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        enable_gnn_plus=enable_gnn_plus,
    )
    num_nodes = dummy_input.get_num_nodes(
        engine=sdjnp.engine, node_name="nodes"
    )
    params = model.init(jax.random.PRNGKey(42), dummy_input, idx=0)
    output = model.apply(params, dummy_input, idx=1)
    hidden_state = gnn_lib.get_node_hidden_state(output)
    self.assertEqual(hidden_state.shape, (num_nodes, hidden_dim))


if __name__ == "__main__":
  absltest.main()
