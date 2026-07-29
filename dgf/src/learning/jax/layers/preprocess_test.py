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

  def test_embed_feature_set_timeseries(self):
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
        ),
        "embed_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, embedding_dim),
            is_timeseries=True,
        ),
        "embed_ts_2d": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, 1),
            is_timeseries=True,
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
        "embed_ts_2d": jnp.ones((batch_size, seq_len), dtype=jnp.float32),
    }

    config = lib.EmbedFeatureSetConfig(
        categorical_feature_embedding_dim=categorical_embed_dim
    )
    embedder = config.make(schema=input_schema)

    variables = embedder.init(jax.random.PRNGKey(0), input_data, training=False)
    static_out, ts_out, ts_mask = embedder.apply(
        variables, input_data, training=False
    )

    self.assertIsNotNone(static_out)
    self.assertIsNotNone(ts_out)
    self.assertIsNotNone(ts_mask)

    self.assertEqual(static_out.shape, (batch_size, categorical_embed_dim))
    self.assertEqual(
        ts_out.shape,
        (batch_size, seq_len, categorical_embed_dim + embedding_dim + 1),
    )
    self.assertEqual(ts_mask.shape, ts_out.shape)

    # Ensure concatenation happens on axis=-1 (channel dimension)
    self.assertEqual(ts_out.shape[1], seq_len)
    self.assertEqual(
        ts_out.shape[2], categorical_embed_dim + embedding_dim + 1
    )

    # Verify generated output schemas match expected shapes
    static_schema, ts_schema, ts_mask_schema = config.output_schema(
        input_schema
    )
    test_util.assert_are_equal(
        self,
        static_schema,
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(categorical_embed_dim,),
        ),
    )
    test_util.assert_are_equal(
        self,
        ts_schema,
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, categorical_embed_dim + embedding_dim + 1),
            is_timeseries=True,
        ),
    )
    test_util.assert_are_equal(
        self,
        ts_mask_schema,
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BOOL,
            semantic=schema_lib.FeatureSemantic.MASK,
            shape=(seq_len, categorical_embed_dim + embedding_dim + 1),
            is_timeseries=True,
        ),
    )

  def test_embed_feature_set_with_masks(self):
    batch_size = 2
    seq_len = 4
    dim_a = 3

    # Define schema with a feature and an associated mask
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

    # Mask is True for t=0,1 and False for t=2,3
    mask_data = jnp.array(
        [[True, True, False, False], [True, False, True, False]],
        dtype=jnp.bool_,
    )
    feat_data = jnp.ones((batch_size, seq_len, dim_a), dtype=jnp.float32)

    input_data = {
        "feat_a": feat_data,
        "feat_a_mask": mask_data,
    }

    config = lib.EmbedFeatureSetConfig()
    embedder = config.make(schema=input_schema)
    variables = embedder.init(jax.random.PRNGKey(0), input_data, training=False)
    static_out, ts_out, ts_mask = embedder.apply(
        variables, input_data, training=False
    )

    self.assertIsNone(static_out)
    self.assertEqual(ts_out.shape, (batch_size, seq_len, dim_a))
    self.assertEqual(ts_mask.shape, (batch_size, seq_len, dim_a))

    # Verify invalid steps (False in mask) are zeroed out in ts_out
    self.assertTrue(jnp.all(ts_out[0, 2:, :] == 0.0))
    self.assertTrue(jnp.all(ts_out[0, :2, :] == 1.0))
    self.assertTrue(jnp.all(ts_mask[0, :2, :]))
    self.assertFalse(jnp.any(ts_mask[0, 2:, :]))

  def test_embed_feature_set_multiple_differing_masks(self):
    batch_size = 1
    seq_len = 3
    dim_a = 2
    dim_b = 2

    # Two features in different sequence groups with different masks
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
        "feat_b": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, dim_b),
            is_timeseries=True,
            group="group_b",
        ),
        "feat_b_mask": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BOOL,
            semantic=schema_lib.FeatureSemantic.MASK,
            shape=(seq_len,),
            is_timeseries=True,
            group="group_b",
        ),
    }

    # feat_a valid at t=0,1 (mask: [1, 1, 0])
    # feat_b valid at t=1,2 (mask: [0, 1, 1])
    mask_a = jnp.array([[True, True, False]], dtype=jnp.bool_)
    mask_b = jnp.array([[False, True, True]], dtype=jnp.bool_)

    input_data = {
        "feat_a": jnp.ones((batch_size, seq_len, dim_a), dtype=jnp.float32),
        "feat_a_mask": mask_a,
        "feat_b": (
            jnp.ones((batch_size, seq_len, dim_b), dtype=jnp.float32) * 2.0
        ),
        "feat_b_mask": mask_b,
    }

    config = lib.EmbedFeatureSetConfig()
    embedder = config.make(schema=input_schema)
    variables = embedder.init(jax.random.PRNGKey(0), input_data, training=False)
    _, ts_out, ts_mask = embedder.apply(variables, input_data, training=False)

    self.assertEqual(ts_out.shape, (batch_size, seq_len, dim_a + dim_b))
    self.assertEqual(ts_mask.shape, (batch_size, seq_len, dim_a + dim_b))

    # Channels 0 and 1 are feat_a, channels 2 and 3 are feat_b
    # At t=0: feat_a is valid (1.0), feat_b is masked (0.0)
    self.assertTrue(jnp.all(ts_out[0, 0, :dim_a] == 1.0))
    self.assertTrue(jnp.all(ts_out[0, 0, dim_a:] == 0.0))
    self.assertTrue(jnp.all(ts_mask[0, 0, :dim_a]))
    self.assertFalse(jnp.any(ts_mask[0, 0, dim_a:]))

    # At t=2: feat_a is masked (0.0), feat_b is valid (2.0)
    self.assertTrue(jnp.all(ts_out[0, 2, :dim_a] == 0.0))
    self.assertTrue(jnp.all(ts_out[0, 2, dim_a:] == 2.0))
    self.assertFalse(jnp.any(ts_mask[0, 2, :dim_a]))
    self.assertTrue(jnp.all(ts_mask[0, 2, dim_a:]))

  def test_embed_feature_set_mismatched_timeseries_length(self):
    input_schema = {
        "cat_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=10,
            shape=(10,),
            is_timeseries=True,
        ),
        "embed_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(12, 8),
            is_timeseries=True,
        ),
    }

    config = lib.EmbedFeatureSetConfig()
    # Verify ValueError is raised when sequence lengths conflict across
    # sequence features.
    with self.assertRaisesRegex(
        ValueError, "conflicting lengths"
    ):
      config.output_schema(input_schema)

  def test_embed_feature_set_invalid_timeseries_categorical_ndim(self):
    input_schema = {
        "cat_ts": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=10,
            shape=(10,),
            is_timeseries=True,
        ),
    }
    input_data = {
        "cat_ts": jnp.array([1, 2, 3], dtype=jnp.int32),  # 1D instead of 2D
    }
    embedder = lib.EmbedFeatureSetConfig().make(schema=input_schema)
    # Verify passing a 1D array to a sequence categorical feature raises
    # ValueError (requires 2D)
    with self.assertRaisesRegex(ValueError, "must have ndim == 2"):
      embedder.init(jax.random.PRNGKey(0), input_data, training=False)

  def test_embed_feature_set_invalid_static_categorical_ndim(self):
    input_schema = {
        "cat_static": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_32,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            num_categorical_values=10,
            is_timeseries=False,
        ),
    }
    input_data = {
        "cat_static": jnp.array([[1, 2]], dtype=jnp.int32),  # 2D instead of 1D
    }
    embedder = lib.EmbedFeatureSetConfig().make(schema=input_schema)
    # Verify passing a 2D array to a static categorical feature raises
    # ValueError (requires 1D)
    with self.assertRaisesRegex(ValueError, "must have ndim == 1"):
      embedder.init(jax.random.PRNGKey(0), input_data, training=False)

if __name__ == "__main__":
  absltest.main()

