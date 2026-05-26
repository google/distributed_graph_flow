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

"""Tests for the simple flax train loop."""

import os
from typing import Optional
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.jax import flax_train
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class SimpleModel(nn.Module):
  hidden_dim: int

  @nn.compact
  def __call__(self, x, training: bool = False):
    return nn.Dense(self.hidden_dim)(x)


def dataset_iterator(
    num_steps: Optional[int], batch_size: int = 4, dim: int = 8
):
  example = {
      "data": np.random.normal(size=(batch_size, dim)).astype(np.float32),
      "label": np.random.randint(2, size=(batch_size,), dtype=np.int32),
  }
  if num_steps is None:
    while True:
      yield example
  else:
    for _ in range(num_steps):
      yield example


class FlaxTrainTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="no_jit", batch_size=4, jit=False),
      dict(testcase_name="with_jit", batch_size=2, jit=True),
  )
  def test_basic(self, batch_size, jit: bool):
    work_dir = self.create_tempdir()

    def train_step(params, opt_state, batch, rng_key):

      def loss_fn(params, x, y, rng_key):
        logits = model.apply(
            params, x, training=True, rngs={"dropout": rng_key}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.mean(loss)

      loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(
          params, batch["data"], batch["label"], rng_key
      )
      updates, opt_state = opt.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, {"loss": loss, "accuracy": jnp.array(1.0)}

    if jit:
      train_step = jax.jit(train_step)

    @jax.jit
    def valid_step(params, opt_state, batch):
      return {"loss": jnp.array(0.1), "accuracy": jnp.array(0.5)}

    model = SimpleModel(hidden_dim=8)
    opt = optax.adam(1e-3)
    steps = 10
    result = flax_train.train(
        model=model,
        opt=opt,
        train_step=train_step,
        dataset_iterator=dataset_iterator(
            num_steps=None, batch_size=batch_size
        ),
        train_log_every_n_steps=3,
        dummy_data_fn=lambda x: x["data"],
        num_train_steps=steps,
        working_path=work_dir.full_path,
        rng_key=jax.random.PRNGKey(42),
        valid_every_n_steps=5,
        valid_step=valid_step,
        valid_dataset_iterator_fn=lambda: dataset_iterator(
            num_steps=10, batch_size=batch_size
        ),
    )

    self.assertTrue(
        os.path.exists(os.path.join(work_dir.full_path, "checkpoints"))
    )

    self.assertEqual([l.step for l in result.train_logs], [0, 3, 6, 9])
    self.assertEqual(
        set(result.train_logs[-1].metrics.keys()), set(["accuracy", "loss"])
    )

    self.assertEqual([l.step for l in result.valid_logs], [5, 10])
    self.assertEqual(
        set(result.valid_logs[-1].metrics.keys()), set(["accuracy", "loss"])
    )


if __name__ == "__main__":
  absltest.main()
