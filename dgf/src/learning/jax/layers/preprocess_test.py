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

"""Tests for common layers."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import jax as jax_io_lib
from dgf.src.learning.jax.layers import preprocess as lib
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import jax
import jax.numpy as jnp


class LayersTest(parameterized.TestCase):

  def test_embed_feature_set(self):
    batch_size = 4
    embedding_dim = 16
    categorical_embed_dim = 64

    input_schema = {
        "embedding_feature": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(embedding_dim,),
        ),
        "categorical_feature": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=10,
        ),
    }

    input = {
        "embedding_feature": jnp.ones(
            (batch_size, embedding_dim), dtype=jnp.float32
        ),
        "categorical_feature": jax.random.randint(
            jax.random.PRNGKey(42),
            (batch_size,),
            minval=0,
            maxval=10,
            dtype=jnp.int32,
        ),
    }
    config = lib.EmbedFeatureSetConfig(
        categorical_feature_embedding_dim=categorical_embed_dim
    )
    embedder = config.make(schema=input_schema)

    variables = embedder.init(jax.random.PRNGKey(0), input, training=False)
    output = embedder.apply(variables, input, training=False)
    expected_output_dim = embedding_dim + categorical_embed_dim
    self.assertEqual(output.shape, (batch_size, expected_output_dim))

    test_util.assert_are_equal(
        self,
        config.output_schema(input_schema),
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(expected_output_dim,),
        ),
    )

  def test_embed_graph(self):
    num_nodes_a = 2
    num_nodes_b = 3
    embedding_dim_a = 2
    categorical_feature_embedding_dim = 4

    # Define a sample GraphSchema
    input_graph_schema = schema_lib.GraphSchema(
        node_sets={
            "nodes_a": schema_lib.NodeSchema(
                features={
                    "embed_a": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(embedding_dim_a,),
                    ),
                    "cat_a": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_32,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        num_categorical_values=5,
                    ),
                }
            ),
            "nodes_b": schema_lib.NodeSchema(
                features={}  # nodes_b has no features
            ),
        },
        edge_sets={},
    )

    # Create dummy input
    input_graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "nodes_a": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={
                    "embed_a": (
                        jnp.arange(
                            num_nodes_a * embedding_dim_a, dtype=jnp.float32
                        ).reshape(num_nodes_a, embedding_dim_a)
                    ),
                    "cat_a": jnp.arange(num_nodes_a, dtype=jnp.int32) % 5,
                },
                num_nodes=num_nodes_a,
            ),
            "nodes_b": jax_in_memory_graph.JaxInMemoryNodeSet(
                num_nodes=num_nodes_b, features={}
            ),
        },
        edge_sets={},
    )
    config = lib.EmbedGraphConfig(
        feature_embedder=lib.EmbedFeatureSetConfig(
            categorical_feature_embedding_dim=categorical_feature_embedding_dim
        )
    )
    embedder = config.make(
        schema=input_graph_schema,
    )
    variables = embedder.init(
        jax.random.PRNGKey(0), input_graph, training=False
    )
    output_graph = embedder.apply(variables, input_graph, training=False)
    in_memory_graph_validate_lib.validate_graph(
        jax_io_lib.jax_graph_to_graph(output_graph),
        config.output_schema(input_graph_schema),
        raise_on_warning=False,
    )
    self.assertEqual(output_graph.node_sets["nodes_a"].num_nodes, num_nodes_a)
    self.assertEqual(output_graph.node_sets["nodes_b"].num_nodes, num_nodes_b)

    # Check output graph
    test_util.assert_are_equal(
        self,
        config.output_schema(input_graph_schema),
        schema_lib.GraphSchema(
            node_sets={
                "nodes_a": schema_lib.NodeSchema(
                    features={
                        "embedding": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32,
                            semantic=schema_lib.FeatureSemantic.EMBEDDING,
                            shape=(
                                embedding_dim_a
                                + categorical_feature_embedding_dim,
                            ),
                        )
                    }
                ),
                "nodes_b": schema_lib.NodeSchema(
                    features={
                        "embedding": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32,
                            semantic=schema_lib.FeatureSemantic.EMBEDDING,
                            shape=(1,),
                        )
                    }
                ),
            },
            edge_sets={},
        ),
    )

  def test_embbed_and_homogenize(self):
    num_nodes_a = 5
    num_nodes_b = 3

    embedding_dim_a = 16
    node_embedding_dim = 64
    target_nodeset = "nodes_a"

    # Define a sample GraphSchema
    graph_schema = schema_lib.GraphSchema(
        node_sets={
            "nodes_a": schema_lib.NodeSchema(
                features={
                    "embed_a": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(embedding_dim_a,),
                    ),
                    "cat_a": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_32,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        num_categorical_values=5,
                    ),
                }
            ),
            "nodes_b": schema_lib.NodeSchema(
                features={}  # nodes_b has no features
            ),
        },
        edge_sets={
            "edges_a_to_b": schema_lib.EdgeSchema(
                source="nodes_a",
                target="nodes_b",
            )
        },
    )

    # Create dummy input
    graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "nodes_a": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={
                    "embed_a": (
                        jnp.arange(
                            num_nodes_a * embedding_dim_a, dtype=jnp.float32
                        ).reshape(num_nodes_a, embedding_dim_a)
                    ),
                    "cat_a": jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32),
                },
                num_nodes=num_nodes_a,
            ),
            "nodes_b": jax_in_memory_graph.JaxInMemoryNodeSet(
                num_nodes=num_nodes_b, features={}
            ),
        },
        edge_sets={
            "edges_a_to_b": jax_in_memory_graph.JaxInMemoryEdgeSet(
                adjacency=jnp.array(
                    [
                        [0, 1, 2, 3, 4, 0, 1],  # sources
                        [0, 0, 1, 1, 2, 2, 0],  # targets
                    ],
                    dtype=jnp.int32,
                )
            ),
        },
    )

    # Create dummy seed_node_idxs
    seed_node_idxs = jnp.array([0, 2], dtype=jnp.int32)
    config = lib.EmbedAndHomogenizeGraphConfig(
        target_nodeset=target_nodeset,
        node_embedding_dim=node_embedding_dim,
    )
    model = config.make(schema=graph_schema)
    variables = model.init(
        jax.random.PRNGKey(0), graph, seed_node_idxs, training=False
    )
    homo_graph, homo_seed_node_idxs = model.apply(
        variables, graph, seed_node_idxs, training=False
    )

    # Check output graph
    total_nodes = num_nodes_a + num_nodes_b
    test_util.assert_are_equal(
        self,
        model.output_schema,
        schema_lib.GraphSchema(
            node_sets={
                "nodes": schema_lib.NodeSchema(
                    features={
                        "initial_state": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32,
                            semantic=schema_lib.FeatureSemantic.EMBEDDING,
                            shape=(node_embedding_dim,),
                        )
                    }
                )
            },
            edge_sets={
                "edges": schema_lib.EdgeSchema(source="nodes", target="nodes")
            },
        ),
    )
    in_memory_graph_validate_lib.validate_graph(
        jax_io_lib.jax_graph_to_graph(homo_graph),
        model.output_schema,  # TODO(gbm) Move the output schema to the config
        raise_on_warning=False,
    )
    self.assertEqual(homo_graph.node_sets["nodes"].num_nodes, total_nodes)

    # Check output seed node indices.
    test_util.assert_are_equal(
        self, homo_seed_node_idxs, jnp.array([0, 2], dtype=jnp.int32)
    )

    # Check node embeddings differ for nodes with different features.
    node_embs = homo_graph.node_sets["nodes"].features["initial_state"]
    self.assertFalse(jnp.allclose(node_embs[0], node_embs[1]))


  def test_embed_feature_set_ignores_timeseries(self):
    batch_size = 4
    seq_len = 10
    embedding_dim = 16
    categorical_embed_dim = 32

    input_schema = {
        "cat_static": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=5,
            is_timeseries=False,
        ),
        "cat_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=10,
            is_timeseries=True,
            group="events",
        ),
        "embed_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, embedding_dim),
            is_timeseries=True,
            group="events",
        ),
    }

    input_data = {
        "cat_static": jax.random.randint(
            jax.random.PRNGKey(1), (batch_size,), 0, 5, dtype=jnp.int32
        ),
        "cat_ts": jax.random.randint(
            jax.random.PRNGKey(2), (batch_size, seq_len), 0, 10, dtype=jnp.int32
        ),
        "embed_ts": jnp.ones(
            (batch_size, seq_len, embedding_dim), dtype=jnp.float32
        ),
    }

    config = lib.EmbedFeatureSetConfig(
        categorical_feature_embedding_dim=categorical_embed_dim
    )
    embedder = config.make(schema=input_schema)

    with mock.patch.object(lib.log, "warning") as mock_warn:
      variables = embedder.init(
          jax.random.PRNGKey(0), input_data, training=False
      )
      self.assertEqual(mock_warn.call_count, 2)
      self.assertEqual(
          list(variables["params"]["EmbedFeatureGroups_0"].keys()),
          ["embed_cat_static"],
      )
      output = embedder.apply(variables, input_data, training=False)
      self.assertEqual(mock_warn.call_count, 4)

    self.assertIsNotNone(output)
    self.assertEqual(output.shape, (batch_size, categorical_embed_dim))

    with mock.patch.object(lib.log, "warning") as mock_warn:
      out_schema = config.output_schema(input_schema)
      self.assertEqual(mock_warn.call_count, 2)

    test_util.assert_are_equal(
        self,
        out_schema,
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(categorical_embed_dim,),
        ),
    )

  def test_embed_feature_groups_static_and_timeseries(self):
    batch_size = 4
    seq_len = 10
    embedding_dim = 16
    categorical_embed_dim = 32

    input_schema = {
        "cat_static": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=5,
            is_timeseries=False,
        ),
        "cat_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=10,
            shape=(seq_len,),
            is_timeseries=True,
            group="group_a",
        ),
        "embed_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, embedding_dim),
            is_timeseries=True,
            group="group_a",
        ),
        "embed_ts_2d": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, 1),
            is_timeseries=True,
            group="group_a",
        ),
        "embed_ts_univariate": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len,),
            is_timeseries=True,
            group="group_a",
        ),
    }

    input_data = {
        "cat_static": jax.random.randint(
            jax.random.PRNGKey(1), (batch_size,), 0, 5, dtype=jnp.int32
        ),
        "cat_ts": jax.random.randint(
            jax.random.PRNGKey(2), (batch_size, seq_len), 0, 10, dtype=jnp.int32
        ),
        "embed_ts": jnp.ones(
            (batch_size, seq_len, embedding_dim), dtype=jnp.float32
        ),
        "embed_ts_2d": jnp.ones((batch_size, seq_len, 1), dtype=jnp.float32),
        "embed_ts_univariate": jnp.ones(
            (batch_size, seq_len), dtype=jnp.float32
        ),
    }

    config = lib.EmbedFeatureGroupsConfig(
        categorical_feature_embedding_dim=categorical_embed_dim
    )
    embedder = config.make(schema=input_schema)

    variables = embedder.init(jax.random.PRNGKey(0), input_data, training=False)
    outputs = embedder.apply(variables, input_data, training=False)

    self.assertIn("embedding", outputs)
    self.assertIn("embedding_group_a", outputs)
    self.assertIn("mask_group_a", outputs)

    expected_ts_dim = categorical_embed_dim + embedding_dim + 1 + 1
    self.assertEqual(
        outputs["embedding"].shape, (batch_size, categorical_embed_dim)
    )
    self.assertEqual(
        outputs["embedding_group_a"].shape,
        (batch_size, seq_len, expected_ts_dim),
    )
    self.assertEqual(outputs["mask_group_a"].shape, (batch_size, seq_len))
    self.assertTrue(jnp.all(outputs["mask_group_a"]))

    out_schema = config.output_schema(input_schema)
    test_util.assert_are_equal(
        self,
        out_schema,
        {
            "embedding": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(categorical_embed_dim,),
            ),
            "embedding_group_a": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(seq_len, expected_ts_dim),
                is_timeseries=True,
                group="group_a",
            ),
            "mask_group_a": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.BOOL,
                semantic=schema_lib.FeatureSemantic.MASK,
                shape=(seq_len,),
                is_timeseries=True,
                group="group_a",
            ),
        },
    )

  def test_embed_feature_groups_with_mask(self):
    batch_size = 2
    seq_len = 4
    dim_a = 3

    input_schema = {
        "feat_a": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, dim_a),
            is_timeseries=True,
            group="group_a",
        ),
        "feat_a_mask": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BOOL,
            semantic=schema_lib.FeatureSemantic.MASK,
            shape=(seq_len,),
            is_timeseries=True,
            group="group_a",
        ),
    }

    # Mask is True for t=0,1 and False for t=2,3 on batch 0
    mask_data = jnp.array(
        [[True, True, False, False], [True, False, True, False]],
        dtype=jnp.bool_,
    )
    feat_data = jnp.ones((batch_size, seq_len, dim_a), dtype=jnp.float32)

    input_data = {
        "feat_a": feat_data,
        "feat_a_mask": mask_data,
    }

    config = lib.EmbedFeatureGroupsConfig()
    embedder = config.make(schema=input_schema)
    variables = embedder.init(jax.random.PRNGKey(0), input_data, training=False)
    outputs = embedder.apply(variables, input_data, training=False)

    self.assertNotIn("embedding", outputs)
    self.assertIn("embedding_group_a", outputs)
    self.assertIn("mask_group_a", outputs)

    ts_out = outputs["embedding_group_a"]
    ts_mask = outputs["mask_group_a"]

    self.assertEqual(ts_out.shape, (batch_size, seq_len, dim_a))
    self.assertEqual(ts_mask.shape, (batch_size, seq_len))

    # Verify invalid steps (False in mask) are zeroed out in ts_out
    self.assertTrue(jnp.all(ts_out[0, 2:, :] == 0.0))
    self.assertTrue(jnp.all(ts_out[0, :2, :] == 1.0))
    self.assertTrue(jnp.all(ts_mask[0, :2]))
    self.assertFalse(jnp.any(ts_mask[0, 2:]))

  def test_embed_feature_groups_multiple_groups(self):
    batch_size = 2
    seq_len_a = 4
    seq_len_b = 6
    dim_a = 3
    dim_b = 5

    input_schema = {
        "feat_a": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len_a, dim_a),
            is_timeseries=True,
            group="group_a",
        ),
        "feat_b": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len_b, dim_b),
            is_timeseries=True,
            group="group_b",
        ),
    }

    input_data = {
        "feat_a": jnp.ones((batch_size, seq_len_a, dim_a), dtype=jnp.float32),
        "feat_b": (
            jnp.ones((batch_size, seq_len_b, dim_b), dtype=jnp.float32) * 2.0
        ),
    }

    config = lib.EmbedFeatureGroupsConfig()
    embedder = config.make(schema=input_schema)
    variables = embedder.init(jax.random.PRNGKey(0), input_data, training=False)
    outputs = embedder.apply(variables, input_data, training=False)

    self.assertIn("embedding_group_a", outputs)
    self.assertIn("mask_group_a", outputs)
    self.assertIn("embedding_group_b", outputs)
    self.assertIn("mask_group_b", outputs)

    self.assertEqual(
        outputs["embedding_group_a"].shape, (batch_size, seq_len_a, dim_a)
    )
    self.assertEqual(outputs["mask_group_a"].shape, (batch_size, seq_len_a))
    self.assertEqual(
        first=outputs["embedding_group_b"].shape,
        second=(batch_size, seq_len_b, dim_b),
    )
    self.assertEqual(outputs["mask_group_b"].shape, (batch_size, seq_len_b))

  def test_embed_feature_groups_mismatched_sequence_length(self):
    input_schema = {
        "cat_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=10,
            shape=(10,),
            is_timeseries=True,
            group="events",
        ),
        "embed_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(12, 8),
            is_timeseries=True,
            group="events",
        ),
    }

    config = lib.EmbedFeatureGroupsConfig()
    with self.assertRaisesRegex(ValueError, "conflicting lengths"):
      config.output_schema(input_schema)

  def test_embed_feature_groups_none_sequence_length(self):
    input_schema_none_shape = {
        "ts_feat": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=None,
            is_timeseries=True,
            group="events",
        ),
    }
    input_schema_none_seq_len = {
        "ts_feat": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(None, 8),
            is_timeseries=True,
            group="events",
        ),
    }

    config = lib.EmbedFeatureGroupsConfig()
    with self.assertRaisesRegex(
        ValueError, "must have a defined sequence length"
    ):
      config.output_schema(input_schema_none_shape)

    with self.assertRaisesRegex(
        ValueError, "must have a defined sequence length"
    ):
      config.output_schema(input_schema_none_seq_len)

  def test_embed_feature_groups_unsupported_shape_dimensions(self):
    input_schema = {
        "ts_3d": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(10, 2, 3),
            is_timeseries=True,
            group="events",
        ),
    }
    config = lib.EmbedFeatureGroupsConfig()
    with self.assertRaisesRegex(ValueError, "unsupported shape"):
      config.output_schema(input_schema)

  def test_embed_feature_groups_invalid_categorical_shape(self):
    input_schema = {
        "cat_ts_2d": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=10,
            shape=(10, 2),
            is_timeseries=True,
            group="events",
        ),
    }
    config = lib.EmbedFeatureGroupsConfig()
    with self.assertRaisesRegex(ValueError, "must be 1D"):
      config.output_schema(input_schema)

  def test_embed_feature_groups_invalid_embedding_feature_dim(self):
    input_schema_none_dim = {
        "ts_embed": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(10, None),
            is_timeseries=True,
            group="events",
        ),
    }
    input_schema_zero_dim = {
        "ts_embed": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(10, 0),
            is_timeseries=True,
            group="events",
        ),
    }
    config = lib.EmbedFeatureGroupsConfig()
    with self.assertRaisesRegex(ValueError, "invalid feature dimension"):
      config.output_schema(input_schema_none_dim)
    with self.assertRaisesRegex(ValueError, "invalid feature dimension"):
      config.output_schema(input_schema_zero_dim)

  def test_embed_feature_groups_invalid_categorical_ndim(self):
    input_schema = {
        "cat_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=10,
            shape=(10,),
            is_timeseries=True,
            group="events",
        ),
    }
    input_data = {
        "cat_ts": jnp.array([1, 2, 3], dtype=jnp.int32),  # 1D instead of 2D
    }
    embedder = lib.EmbedFeatureGroupsConfig().make(schema=input_schema)
    with self.assertRaisesRegex(ValueError, "must have ndim == 2"):
      embedder.init(jax.random.PRNGKey(0), input_data, training=False)

  def test_embed_feature_groups_fallback_group_name(self):
    batch_size = 2
    seq_len = 5
    embedding_dim = 8

    # timeseries features without explicit group specified (group=None)
    input_schema = {
        "ts_feature_one": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, embedding_dim),
            is_timeseries=True,
            group=None,
        ),
        "ts_feature_two": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, embedding_dim),
            is_timeseries=True,
            group=None,
        ),
    }
    input_data = {
        "ts_feature_one": jnp.ones(
            (batch_size, seq_len, embedding_dim), dtype=jnp.float32
        ),
        "ts_feature_two": jnp.ones(
            (batch_size, seq_len, embedding_dim), dtype=jnp.float32
        ) * 3.0,
    }

    config = lib.EmbedFeatureGroupsConfig()
    embedder = config.make(schema=input_schema)
    variables = embedder.init(jax.random.PRNGKey(0), input_data, training=False)
    outputs = embedder.apply(variables, input_data, training=False)

    self.assertIn("embedding_ts_feature_one", outputs)
    self.assertIn("mask_ts_feature_one", outputs)
    self.assertIn("embedding_ts_feature_two", outputs)
    self.assertIn("mask_ts_feature_two", outputs)

    out_schema = config.output_schema(input_schema)
    test_util.assert_are_equal(
        self,
        out_schema,
        {
            "embedding_ts_feature_one": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(seq_len, embedding_dim),
                is_timeseries=True,
                group="ts_feature_one",
            ),
            "mask_ts_feature_one": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.BOOL,
                semantic=schema_lib.FeatureSemantic.MASK,
                shape=(seq_len,),
                is_timeseries=True,
                group="ts_feature_one",
            ),
            "embedding_ts_feature_two": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(seq_len, embedding_dim),
                is_timeseries=True,
                group="ts_feature_two",
            ),
            "mask_ts_feature_two": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.BOOL,
                semantic=schema_lib.FeatureSemantic.MASK,
                shape=(seq_len,),
                is_timeseries=True,
                group="ts_feature_two",
            ),
        },
    )

  def test_is_non_mask_timeseries(self):
    ts_embed = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        is_timeseries=True,
        group="g1",
    )
    ts_mask = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        is_timeseries=True,
        group="g1",
    )
    static_embed = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        is_timeseries=False,
    )
    self.assertTrue(lib._is_non_mask_timeseries(ts_embed))
    self.assertFalse(lib._is_non_mask_timeseries(ts_mask))
    self.assertFalse(lib._is_non_mask_timeseries(static_embed))

  def test_unexpected_feature_warning(self):
    input_schema = {
        "unexpected_static": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.NUMERICAL,
        ),
        "unexpected_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.NUMERICAL,
            shape=(4,),
            is_timeseries=True,
            group="ts_g",
        ),
    }
    input_data = {
        "unexpected_static": jnp.array([1.0, 2.0]),
        "unexpected_ts": jnp.ones((2, 4)),
    }
    config = lib.EmbedFeatureGroupsConfig()
    with mock.patch.object(lib.log, "warning") as mock_warn:
      out_schema = config.output_schema(input_schema)
      self.assertEmpty(out_schema)
      self.assertEqual(mock_warn.call_count, 2)

    embedder = config.make(schema=input_schema)
    with mock.patch.object(lib.log, "warning") as mock_warn:
      variables = embedder.init(
          jax.random.PRNGKey(0), input_data, training=False
      )
      self.assertEqual(mock_warn.call_count, 2)
      outputs = embedder.apply(variables, input_data, training=False)
      self.assertEmpty(outputs)
      self.assertEqual(mock_warn.call_count, 4)


if __name__ == "__main__":
  absltest.main()
