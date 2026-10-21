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

"""Tests for O(N)-memory Pallas SpMM kernel."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.jax.layers import pallas_spmm
import jax
import jax.numpy as jnp
import numpy as np


def _ref_spmm(
    x: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    values: jax.Array | None = None,
) -> jax.Array:
  """Reference SpMM using gather + segment_sum."""
  gathered = x[col_ids]
  if values is not None:
    w = values.reshape((-1,) + (1,) * (x.ndim - 1)).astype(x.dtype)
    gathered = gathered * w
  return jax.ops.segment_sum(gathered, row_ids, num_segments=num_rows)


class PallasSpmmTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          num_nodes=32,
          num_edges=100,
          feat_shape=(16,),
          dtype=jnp.float32,
          interpret=False,
      ),
      dict(
          num_nodes=32,
          num_edges=100,
          feat_shape=(16,),
          dtype=jnp.float32,
          interpret=True,
      ),
      dict(
          num_nodes=64,
          num_edges=250,
          feat_shape=(128,),
          dtype=jnp.float32,
          interpret=True,
      ),
      dict(
          num_nodes=64,
          num_edges=250,
          feat_shape=(256,),
          dtype=jnp.float32,
          interpret=True,
      ),
      dict(
          num_nodes=64,
          num_edges=250,
          feat_shape=(2, 128),
          dtype=jnp.float32,
          interpret=True,
      ),
      dict(
          num_nodes=64,
          num_edges=250,
          feat_shape=(256,),
          dtype=jnp.bfloat16,
          interpret=True,
      ),
  )
  def test_forward_and_backward_parity(
      self, num_nodes, num_edges, feat_shape, dtype, interpret
  ):
    rng = np.random.default_rng(42)
    row_ids = jnp.array(
        rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)
    )
    col_ids = jnp.array(
        rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)
    )
    x = jax.random.normal(
        jax.random.PRNGKey(0), (num_nodes, *feat_shape), dtype=dtype
    )
    dy = jax.random.normal(
        jax.random.PRNGKey(1), (num_nodes, *feat_shape), dtype=dtype
    )

    def loss_ref(x_in):
      y = _ref_spmm(x_in, row_ids, col_ids, num_nodes)
      return jnp.sum(y.astype(jnp.float32) * dy.astype(jnp.float32)), y

    def loss_pallas(x_in):
      y = pallas_spmm.pallas_spmm(
          x_in,
          row_ids,
          col_ids,
          num_nodes,
          interpret=interpret,
      )
      return jnp.sum(y.astype(jnp.float32) * dy.astype(jnp.float32)), y

    (_, y_ref), dx_ref = jax.value_and_grad(loss_ref, has_aux=True)(x)
    (_, y_pal), dx_pal = jax.value_and_grad(loss_pallas, has_aux=True)(x)

    atol = 1e-4 if dtype == jnp.float32 else 5e-2
    rtol = 1e-4 if dtype == jnp.float32 else 5e-2
    np.testing.assert_allclose(
        np.asarray(y_pal, dtype=np.float32),
        np.asarray(y_ref, dtype=np.float32),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(dx_pal, dtype=np.float32),
        np.asarray(dx_ref, dtype=np.float32),
        atol=atol,
        rtol=rtol,
    )

  def test_weighted_forward_and_backward_parity(self):
    num_nodes = 48
    num_edges = 150
    feat_dim = 128
    rng = np.random.default_rng(7)
    row_ids = jnp.array(
        rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)
    )
    col_ids = jnp.array(
        rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)
    )
    values = jnp.array(
        rng.uniform(0.1, 2.0, size=(num_edges,)).astype(np.float32)
    )
    x = jax.random.normal(
        jax.random.PRNGKey(2), (num_nodes, feat_dim), dtype=jnp.float32
    )
    dy = jax.random.normal(
        jax.random.PRNGKey(3), (num_nodes, feat_dim), dtype=jnp.float32
    )
    use_interpret = jax.default_backend() != "tpu"

    def loss_ref(x_in):
      y = _ref_spmm(x_in, row_ids, col_ids, num_nodes, values=values)
      return jnp.sum(y * dy), y

    def loss_pallas(x_in):
      y = pallas_spmm.pallas_spmm(
          x_in,
          row_ids,
          col_ids,
          num_nodes,
          values=values,
          interpret=use_interpret,
      )
      return jnp.sum(y * dy), y

    (_, y_ref), dx_ref = jax.value_and_grad(loss_ref, has_aux=True)(x)
    (_, y_pal), dx_pal = jax.value_and_grad(loss_pallas, has_aux=True)(x)

    np.testing.assert_allclose(
        np.asarray(y_pal), np.asarray(y_ref), atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(dx_pal), np.asarray(dx_ref), atol=1e-4, rtol=1e-4
    )

  def test_duplicate_consecutive_destinations_and_padding_edges(self):
    num_nodes = 32
    feat_dim = 128
    row_ids = jnp.array(
        [5, 5, 5, 5, 2, 3, 2, 3, 7, 7, 8, 7] + [num_nodes - 1] * 25,
        dtype=jnp.int32,
    )
    col_ids = jnp.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [num_nodes - 1] * 25,
        dtype=jnp.int32,
    )
    x = jax.random.normal(
        jax.random.PRNGKey(11), (num_nodes, feat_dim), dtype=jnp.float32
    )
    use_interpret = jax.default_backend() != "tpu"

    y_ref = _ref_spmm(x, row_ids, col_ids, num_nodes)
    y_pal = pallas_spmm.pallas_spmm(
        x, row_ids, col_ids, num_nodes, interpret=use_interpret
    )
    np.testing.assert_allclose(
        np.asarray(y_pal), np.asarray(y_ref), atol=1e-5, rtol=1e-5
    )

  def test_node_tiling_spmm_large_parity(self):
    num_nodes = 96
    num_edges = 411
    feat_dim = 256
    rng = np.random.default_rng(777)
    row_rand = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)
    col_rand = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int32)
    row_ids = jnp.array(
        np.concatenate([
            np.array(
                [0, 0, 31, 32, 32, 32, 63, 64, 64, 95, 95, 0], dtype=np.int32
            ),
            row_rand,
        ])
    )
    col_ids = jnp.array(
        np.concatenate([
            np.array(
                [0, 32, 64, 95, 0, 31, 32, 63, 64, 0, 95, 48], dtype=np.int32
            ),
            col_rand,
        ])
    )
    values = jnp.array(
        rng.uniform(0.2, 1.8, size=(row_ids.shape[0],)).astype(np.float32)
    )
    use_interpret = jax.default_backend() != "tpu"

    orig_budget = pallas_spmm._MAX_WORKING_SET_BYTES
    try:
      pallas_spmm._MAX_WORKING_SET_BYTES = 49152
      for dtype in (jnp.float32, jnp.bfloat16):
        for w in (None, values):
          x = jax.random.normal(
              jax.random.PRNGKey(21), (num_nodes, feat_dim), dtype=dtype
          )
          dy = jax.random.normal(
              jax.random.PRNGKey(22), (num_nodes, feat_dim), dtype=dtype
          )

          def loss_ref(x_in, dy_in=dy, w_in=w):
            y = _ref_spmm(x_in, row_ids, col_ids, num_nodes, values=w_in)
            return jnp.sum(y.astype(jnp.float32) * dy_in.astype(jnp.float32)), y

          def loss_pal(x_in, dy_in=dy, w_in=w):
            y = pallas_spmm.spmm_large(
                x_in,
                row_ids,
                col_ids,
                num_nodes,
                values=w_in,
                interpret=use_interpret,
            )
            return jnp.sum(y.astype(jnp.float32) * dy_in.astype(jnp.float32)), y

          (_, y_ref), dx_ref = jax.value_and_grad(loss_ref, has_aux=True)(x)
          (_, y_pal), dx_pal = jax.value_and_grad(loss_pal, has_aux=True)(x)

          atol = 1e-4 if dtype == jnp.float32 else 5e-2
          rtol = 1e-4 if dtype == jnp.float32 else 5e-2
          np.testing.assert_allclose(
              np.asarray(y_pal, dtype=np.float32),
              np.asarray(y_ref, dtype=np.float32),
              atol=atol,
              rtol=rtol,
          )
          np.testing.assert_allclose(
              np.asarray(dx_pal, dtype=np.float32),
              np.asarray(dx_ref, dtype=np.float32),
              atol=atol,
              rtol=rtol,
          )
    finally:
      pallas_spmm._MAX_WORKING_SET_BYTES = orig_budget

  def test_device_profiles_and_cross_device_generalization(self):
    expected_vmem_mib = {
        "glp": 128,
        "vlp": 128,
        "gfl": 64,
        "gf": 64,
        "vf": 64,
        "pf": 16,
    }
    for key, vmem_mib in expected_vmem_mib.items():
      prof = pallas_spmm.get_tpu_device_profile(key)
      self.assertEqual(prof.name, key)
      self.assertEqual(prof.vmem_capacity_bytes, vmem_mib * 1024 * 1024)
      self.assertLess(prof.vmem_limit_bytes, prof.vmem_capacity_bytes)
      self.assertLess(prof.max_working_set_bytes, prof.vmem_limit_bytes)
      self.assertLess(prof.large_avail_vmem_bytes, prof.vmem_limit_bytes)

    self.assertEqual(
        pallas_spmm.get_tpu_device_profile("TPU v6 lite").name, "glp"
    )
    self.assertEqual(pallas_spmm.get_tpu_device_profile("TPU7").name, "gfl")
    self.assertEqual(pallas_spmm.get_tpu_device_profile("TPU7x").name, "gf")


if __name__ == "__main__":
  absltest.main()
