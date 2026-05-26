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

from absl.testing import absltest
from dgf.src.io import jax as jax_io_lib
from dgf.src.io import sparse_deferred as sparse_deferred_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import jax.numpy as jnp
import numpy as np
from sparse_deferred.structs import graph_struct as sd_struct_lib


class SparseDeferredTest(absltest.TestCase):

  def _expected_sd_schema(self):
    return {
        "e1": ("n1", "n1"),
        "e2": ("n1", "n2"),
    }

  def _expected_sd_numpy_struct(self, schema=None, **kwargs):
    # Create a SD struct equivalent to the one created by "gen_test_graph".
    sd_nodes = {
        "n1": {
            "f1": np.array([[b"blue"], [b"red"]]),
            "f2": np.array([[0.0, 1.0], [2.0, 3.0]]),
        },
        "n2": {"f3": np.array([4, 5]), "f4": np.array([10, 11])},
    }
    sd_edges = {
        "e1": ((np.array([0, 0]), np.array([0, 1])), {}),
        "e2": ((np.array([0, 0]), np.array([0, 1])), {}),
    }
    if schema is None:
      sd_schema = {}
    else:
      sd_schema = self._expected_sd_schema()
    return sd_struct_lib.GraphStruct.new(
        nodes=sd_nodes, edges=sd_edges, schema=sd_schema, **kwargs
    )

  def _expected_sd_jax_struct(self, schema=None, **kwargs):
    # Create a SD struct equivalent to the one created by "gen_test_graph".
    sd_nodes = {
        "n1": {
            "f2": jnp.array([[0.0, 1.0], [2.0, 3.0]]),
        },
        "n2": {"f3": jnp.array([4, 5]), "f4": jnp.array([10, 11])},
    }
    sd_edges = {
        "e1": ((jnp.array([0, 0]), jnp.array([0, 1])), {}),
        "e2": ((jnp.array([0, 0]), jnp.array([0, 1])), {}),
    }
    if schema is None:
      sd_schema = {}
    else:
      sd_schema = self._expected_sd_schema()
    return sd_struct_lib.GraphStruct.new(
        nodes=sd_nodes, edges=sd_edges, schema=sd_schema, **kwargs
    )

  def test_graph_to_sparse_deferred_struct_without_schema(self):
    in_memory_graph = gen_test_graph.generate_in_memory_graph(
        variable_length=False
    )
    sd_struct = sparse_deferred_lib.graph_to_sparse_deferred_struct(
        in_memory_graph=in_memory_graph,
        schema=None,
    )
    test_util.assert_are_equal(
        self, sd_struct, self._expected_sd_numpy_struct()
    )

  def test_graph_to_sparse_deferred_struct_with_schema(self):
    in_memory_graph = gen_test_graph.generate_in_memory_graph(
        variable_length=False
    )
    schema = gen_test_graph.generate_schema(variable_length=False)
    sd_struct = sparse_deferred_lib.graph_to_sparse_deferred_struct(
        in_memory_graph=in_memory_graph,
        schema=schema,
    )
    test_util.assert_are_equal(
        self, sd_struct, self._expected_sd_numpy_struct(schema=schema)
    )

  def test_jax_graph_to_sparse_deferred_struct_with_schema(self):
    schema = gen_test_graph.generate_schema(
        variable_length=False, bytes_feature=False
    )
    jax_graph = jax_io_lib.graph_to_jax_graph(
        gen_test_graph.generate_in_memory_graph(
            variable_length=False, bytes_feature=False
        )
    )
    expected_sd_struct = self._expected_sd_jax_struct(schema=schema)
    sd_struct = sparse_deferred_lib.jax_graph_to_sparse_deferred_struct(
        in_memory_graph=jax_graph,
        schema=schema,
    )
    test_util.assert_are_equal(self, sd_struct, expected_sd_struct)

  def test_sparse_deferred_struct_to_graph(self):
    sd_struct = self._expected_sd_numpy_struct()
    in_memory_graph = sparse_deferred_lib.sparse_deferred_struct_to_graph(
        sd_struct
    )
    expected_in_memory_graph = gen_test_graph.generate_in_memory_graph(
        variable_length=False
    )
    test_util.assert_are_equal(self, in_memory_graph, expected_in_memory_graph)

  def test_sparse_deferred_struct_to_jax_graph(self):
    sd_struct = self._expected_sd_jax_struct()
    jax_graph = sparse_deferred_lib.sparse_deferred_struct_to_jax_graph(
        sd_struct
    )
    expected_jax_graph = jax_io_lib.graph_to_jax_graph(
        gen_test_graph.generate_in_memory_graph(
            variable_length=False, bytes_feature=False
        )
    )
    test_util.assert_are_equal(self, jax_graph, expected_jax_graph)

  def test_schema_to_sparse_deferred_schema(self):
    schema = gen_test_graph.generate_schema(variable_length=False)
    sd_schema = sparse_deferred_lib.schema_to_sparse_deferred_schema(schema)
    expected_sd_schema = self._expected_sd_schema()
    self.assertEqual(sd_schema, expected_sd_schema)


if __name__ == "__main__":
  absltest.main()
