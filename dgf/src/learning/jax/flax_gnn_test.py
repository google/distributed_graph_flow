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

"""Test for the GNNModule interface."""

import dataclasses

from absl.testing import absltest
from dgf.src.learning.jax import flax_gnn as dgf_flax_gnn
from dgf.src.learning.jax import test_utils
from dgf.src.learning.jax.layers import homo_gnn_sparse_deferred as homo_gnn
import flax.core as flax_core
import flax.linen as nn
import jax
import numpy as np

from sparse_deferred.structs import graph_struct as graph_struct_lib


class FlaxGnnTest(absltest.TestCase):

  def test_flax_gnn_from_config(self):
    test_graph = test_utils.generate_test_graph()
    model = dgf_flax_gnn.from_config(
        gnn_config=homo_gnn.MPNNConfig(num_layers=2, hidden_dim=8),
        initial_node_state_fn=None,
    )
    params = model.init(jax.random.PRNGKey(42), test_graph)
    output_graph = model.apply(params, test_graph)
    self.assertEqual(
        homo_gnn.get_node_hidden_state(output_graph['gnn']).shape, (4, 8)
    )

  def test_flax_gnn_from_config_with_initial_node_state_fn(self):
    test_graph = graph_struct_lib.GraphStruct.new(
        nodes={'foo': {'bar': np.random.normal(size=(4, 8))}},
        edges={
            'biz': (
                (np.array([0, 0, 0, 1, 3]), np.array([1, 2, 3, 2, 0])),
                {},
            )
        },
    )

    class CustomInitialNodeStateFn(nn.Module):
      hidden_dim: int

      @nn.compact
      def __call__(self, x, training=False):
        xx = x.nodes['foo']['bar']
        xx = nn.Dense(self.hidden_dim)(xx)

        return x.update(
            nodes={'nodes': {'initial_state': xx}},
            edges={'edges': x.edges['biz']},
            schema={'edges': ('nodes', 'nodes')},
        )

    hidden_dim = 16
    model = dgf_flax_gnn.from_config(
        initial_node_state_fn=CustomInitialNodeStateFn(hidden_dim=16),
        gnn_config=homo_gnn.MPNNConfig(num_layers=2, hidden_dim=hidden_dim),
    )

    params = model.init(jax.random.PRNGKey(42), test_graph)
    output_graph = model.apply(params, test_graph)

    self.assertEqual(
        homo_gnn.get_node_hidden_state(output_graph['gnn']).shape,
        (4, hidden_dim),
    )

  def test_flax_gnn_with_classification_head(self):
    class ClassificationHead(nn.Module):
      num_classes: int
      hidden_dim: int = 128
      dropout_rate: float = 0.1

      @nn.compact
      def __call__(self, x, training: bool = False):
        x = x.nodes['nodes']['hidden_state']
        x = nn.Dense(2)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        return nn.Dense(self.num_classes)(x)

    @dataclasses.dataclass(frozen=True)
    class BuildableClassificationHead:
      num_classes: int
      hidden_dim: int
      dropout_rate: float

      def make(self) -> nn.Module:
        return ClassificationHead(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        )

    test_graph = test_utils.generate_test_graph()
    model = dgf_flax_gnn.from_config(
        gnn_config=homo_gnn.MPNNConfig(num_layers=2, hidden_dim=8),
        initial_node_state_fn=None,
        heads=(
            flax_core.FrozenDict({
                'classification_head': BuildableClassificationHead(
                    num_classes=2, hidden_dim=128, dropout_rate=0.1
                )
            })
        ),
    )
    _, rng_init, rng_dropout = jax.random.split(jax.random.PRNGKey(42), 3)
    params = model.init(rng_init, test_graph)

    output = model.apply(params, test_graph, rngs={'dropout': rng_dropout})
    self.assertIsInstance(output, dict)
    self.assertEqual(output['classification_head'].shape, (4, 2))

    # check that jitting works
    @jax.jit(static_argnames=['model'])
    def train_step(model, params, graph, rngs):
      output = model.apply(params, graph, rngs=rngs)
      return output

    output = train_step(
        model, params, test_graph, rngs={'dropout': rng_dropout}
    )
    self.assertIsInstance(output, dict)
    self.assertEqual(output['classification_head'].shape, (4, 2))


if __name__ == '__main__':
  absltest.main()
