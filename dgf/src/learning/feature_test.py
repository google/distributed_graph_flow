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

import logging
from absl.testing import absltest
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning import feature as feature_lib
from dgf.src.util import test_util
import jax
import jax.numpy as jnp


class EmbedNodesetFeaturesModuleTest(absltest.TestCase):

  def test_embed_embedding_features(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(2,),
                    ),
                    "f2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                    ),
                    "f3": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                    ),
                }
            )
        },
        edge_sets={},
    )
    graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={
                    "f1": jnp.array([[1, 2], [3, 4]], dtype=jnp.float32),
                    "f2": jnp.array([5, 6], dtype=jnp.float32),
                    "f3": jnp.array([7, 8], dtype=jnp.float32),
                },
                num_nodes=10,
            )
        },
        edge_sets={},
    )
    embed_module = feature_lib.EmbedNodesetFeaturesModule(
        schema=schema, ignore_features=[("n1", "f3")]
    )
    rng = jax.random.PRNGKey(0)
    params = embed_module.init(rng, graph, training=True)
    node_embeddings = embed_module.apply(params, graph, training=True)

    test_util.assert_are_equal(
        self,
        node_embeddings,
        {"n1": jnp.array([[1, 2, 5], [3, 4, 6]], dtype=jnp.float32)},
    )

  def test_embed_categorical_features(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        num_categorical_values=3,
                        shape=(2,),
                    ),
                    "f2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        num_categorical_values=4,
                    ),
                }
            )
        },
        edge_sets={},
    )
    graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={
                    "f1": jnp.array([[1, 2], [3, 4]], dtype=jnp.int64),
                    "f2": jnp.array([5, 6], dtype=jnp.int64),
                },
                num_nodes=10,
            )
        },
        edge_sets={},
    )
    embed_module = feature_lib.EmbedNodesetFeaturesModule(
        schema=schema, categorical_feature_embedding_dim=5
    )
    rng = jax.random.PRNGKey(0)
    params = embed_module.init(rng, graph, training=True)
    node_embeddings = embed_module.apply(params, graph, training=True)
    self.assertEqual(params["params"]["embed_n1_f1"]["embedding"].shape, (3, 5))
    self.assertEqual(params["params"]["embed_n1_f2"]["embedding"].shape, (4, 5))
    self.assertEqual(node_embeddings["n1"].shape, (2, 3 * 5))  # pyrefly: ignore[bad-index]

  def test_embed_unsupported_embedding_format_raises_error(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(2,),
                    ),
                }
            )
        },
        edge_sets={},
    )
    graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={
                    "f1": jnp.array([[1, 2], [3, 4]], dtype=jnp.int64),
                },
                num_nodes=2,
            )
        },
        edge_sets={},
    )
    embed_module = feature_lib.EmbedNodesetFeaturesModule(schema=schema)
    rng = jax.random.PRNGKey(0)
    with self.assertRaisesRegex(
        ValueError,
        "Embedding feature 'f1' in nodeset 'n1' has unexpected dtype"
        " FeatureFormat.INTEGER_64. Embedding is expected to be of dtype"
        " float32.",
    ):
      embed_module.init(rng, graph, training=True)

  def test_embed_unsupported_categorical_format_raises_error(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        num_categorical_values=3,
                    ),
                }
            )
        },
        edge_sets={},
    )
    graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={
                    "f1": jnp.array([1.0, 2.0], dtype=jnp.float32),
                },
                num_nodes=2,
            )
        },
        edge_sets={},
    )
    embed_module = feature_lib.EmbedNodesetFeaturesModule(schema=schema)
    rng = jax.random.PRNGKey(0)
    with self.assertRaisesRegex(
        ValueError,
        "Categorical feature 'f1' in nodeset 'n1' has unexpected dtype "
        "FeatureFormat.FLOAT_32. Categorical is expected to be of dtype int64.",
    ):
      embed_module.init(rng, graph, training=True)

  def test_embed_unsupported_semantic_raises_error(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                }
            )
        },
        edge_sets={},
    )
    graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets={
            "n1": jax_in_memory_graph.JaxInMemoryNodeSet(
                features={
                    "f1": jnp.array([1.0, 2.0], dtype=jnp.float32),
                },
                num_nodes=2,
            )
        },
        edge_sets={},
    )
    embed_module = feature_lib.EmbedNodesetFeaturesModule(schema=schema)
    rng = jax.random.PRNGKey(0)
    with self.assertRaisesRegex(
        NotImplementedError,
        "Unsupported feature semantic <FeatureSemantic.TIMESTAMP: 'TIMESTAMP'>"
        " for feature 'f1' in nodeset 'n1'",
    ):
      embed_module.init(rng, graph, training=True)


if __name__ == "__main__":
  absltest.main()
