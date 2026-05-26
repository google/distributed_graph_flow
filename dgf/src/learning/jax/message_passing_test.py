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

"""Tests for common jax utilities."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.jax import message_passing as lib
from dgf.src.util import test_util
import jax.numpy as jnp


class MessagePassingTest(parameterized.TestCase):

  def test_core_graph_to_sd_sparse_matrix(self):
    edge_list = jnp.array([[0, 1, 0], [1, 1, 0]])
    num_nodes_source = 2
    num_nodes_target = 3
    sd_matrix = lib.core_graph_to_sd_sparse_matrix(
        edge_list, num_nodes_source, num_nodes_target
    )

    # Sum
    vector = jnp.array([10, 20])
    test_util.assert_are_equal(self, sd_matrix @ vector, jnp.array([10, 30, 0]))

    # Normalized
    test_util.assert_are_equal(
        self, sd_matrix.normalize_right() @ vector, jnp.array([10, 30 / 2, 0])
    )


if __name__ == "__main__":
  absltest.main()
