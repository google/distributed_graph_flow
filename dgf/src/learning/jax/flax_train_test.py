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
from dgf.src.learning import early_stopping_monitor
from dgf.src.learning.jax import flax_train
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp


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

    self.assertEqual([l.step for l in result.train_logs], [3, 6, 9])
    self.assertEqual(
        set(result.train_logs[-1].metrics.keys()), set(["accuracy", "loss"])
    )

    self.assertEqual([l.step for l in result.valid_logs], [5, 10])
    self.assertEqual(
        set(result.valid_logs[-1].metrics.keys()), set(["accuracy", "loss"])
    )

  def test_early_stopping(self):
    def train_step(params, opt_state, batch, rng_key):
      return params, opt_state, {"loss": jnp.array(1.0)}

    def valid_step(params, opt_state, batch):
      return {"loss": jnp.array(0.5)}

    model = SimpleModel(hidden_dim=8)
    opt = optax.adam(1e-3)

    es_config = early_stopping_monitor.EarlyStoppingMonitorConfig(
        patience=1, min_improvement=0.1
    )

    result = flax_train.train(
        model=model,
        opt=opt,
        train_step=train_step,
        dataset_iterator=dataset_iterator(num_steps=None),
        dummy_data_fn=lambda x: x["data"],
        num_train_steps=100,
        rng_key=jax.random.PRNGKey(42),
        valid_every_n_steps=10,
        valid_step=valid_step,
        valid_dataset_iterator_fn=lambda: dataset_iterator(num_steps=2),
        early_stopping=es_config,
    )

    # Should validate at 10, then 20 and break out.
    self.assertEqual([l.step for l in result.valid_logs], [10, 20])

  def test_nan_in_training_metrics(self):
    def train_step(params, opt_state, batch, rng_key):
      return params, opt_state, {"loss": jnp.array(jnp.nan)}

    model = SimpleModel(hidden_dim=8)
    opt = optax.adam(1e-3)

    with self.assertRaisesRegex(
        ValueError, "Training metric 'loss' became NaN"
    ):
      flax_train.train(
          model=model,
          opt=opt,
          train_step=train_step,
          dataset_iterator=dataset_iterator(num_steps=None),
          dummy_data_fn=lambda x: x["data"],
          num_train_steps=10,
          rng_key=jax.random.PRNGKey(42),
          train_log_every_n_steps=1,
      )

  def test_nan_in_valid_metrics(self):
    def train_step(params, opt_state, batch, rng_key):
      return params, opt_state, {"loss": jnp.array(1.0)}

    def valid_step(params, opt_state, batch):
      return {"loss": jnp.array(jnp.nan)}

    model = SimpleModel(hidden_dim=8)
    opt = optax.adam(1e-3)

    with self.assertRaisesRegex(
        ValueError, "Validation metric 'loss' became NaN"
    ):
      flax_train.train(
          model=model,
          opt=opt,
          train_step=train_step,
          dataset_iterator=dataset_iterator(num_steps=None),
          dummy_data_fn=lambda x: x["data"],
          num_train_steps=10,
          rng_key=jax.random.PRNGKey(42),
          valid_every_n_steps=2,
          valid_step=valid_step,
          valid_dataset_iterator_fn=lambda: dataset_iterator(num_steps=2),
      )

  def test_resume_from_checkpoint(self):
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

    train_step = jax.jit(train_step)

    model = SimpleModel(hidden_dim=8)
    opt = optax.adam(1e-3)

    # First run: train 5 steps
    flax_train.train(
        model=model,
        opt=opt,
        train_step=train_step,
        dataset_iterator=dataset_iterator(num_steps=None, batch_size=4),
        dummy_data_fn=lambda x: x["data"],
        num_train_steps=5,
        working_path=work_dir.full_path,
        rng_key=jax.random.PRNGKey(42),
    )

    # Second run: resume from checkpoint and train to step 10
    result_resumed = flax_train.train(
        model=model,
        opt=opt,
        train_step=train_step,
        dataset_iterator=dataset_iterator(num_steps=None, batch_size=4),
        dummy_data_fn=lambda x: x["data"],
        num_train_steps=10,
        working_path=work_dir.full_path,
        rng_key=jax.random.PRNGKey(42),
    )

    self.assertIsNotNone(result_resumed.model_params)
    self.assertIsNotNone(result_resumed.opt_state)

  def test_resume_from_old_checkpoint_format(self):
    """Verifies backward compatibility with old checkpoints (only params and opt_state)."""
    work_dir = self.create_tempdir()
    ckpt_dir = os.path.join(work_dir.full_path, "checkpoints")

    model = SimpleModel(hidden_dim=8)
    opt = optax.adam(1e-3)
    dummy_input = np.random.normal(size=(4, 8)).astype(np.float32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    init_opt_state = opt.init(init_params["params"])

    # Simulate an old checkpoint containing ONLY 'params' and 'opt_state'
    mngr = ocp.CheckpointManager(
        ckpt_dir,
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )
    mngr.save(5, {"params": init_params, "opt_state": init_opt_state})
    mngr.wait_until_finished()

    def train_step(params, opt_state, batch, rng_key):
      return params, opt_state, {"loss": jnp.array(0.5)}

    es_config = early_stopping_monitor.EarlyStoppingMonitorConfig(patience=5)

    result = flax_train.train(
        model=model,
        opt=opt,
        train_step=train_step,
        dataset_iterator=dataset_iterator(num_steps=None, batch_size=4),
        dummy_data_fn=lambda x: x["data"],
        num_train_steps=10,
        working_path=work_dir.full_path,
        rng_key=jax.random.PRNGKey(42),
        early_stopping=es_config,
    )

    self.assertIsNotNone(result.model_params)
    self.assertIsNotNone(result.opt_state)

  def test_resume_with_early_stopping(self):
    work_dir = self.create_tempdir()

    def train_step(params, opt_state, batch, rng_key):
      return params, opt_state, {"loss": jnp.array(1.0)}

    # Valid step returns loss 1.0 (no improvement)
    def valid_step(params, opt_state, batch):
      return {"loss": jnp.array(1.0)}

    model = SimpleModel(hidden_dim=8)
    opt = optax.adam(1e-3)

    es_config = early_stopping_monitor.EarlyStoppingMonitorConfig(
        patience=4, min_improvement=0.1
    )

    # First run: train 4 steps with checkpoint at step 2 and validation every 2 steps
    # At step 2: valid loss = 1.0 (best_loss=1.0, patience=0), checkpoint saved with es_state.
    # At step 4: valid loss = 1.0 (patience=1), checkpoint saved with es_state.
    # At completion of run 1: final validation at step 4 runs (patience=2).
    flax_train.train(
        model=model,
        opt=opt,
        train_step=train_step,
        dataset_iterator=dataset_iterator(num_steps=None),
        dummy_data_fn=lambda x: x["data"],
        num_train_steps=4,
        working_path=work_dir.full_path,
        rng_key=jax.random.PRNGKey(42),
        checkpoint_every_n_steps=2,
        valid_every_n_steps=2,
        valid_step=valid_step,
        valid_dataset_iterator_fn=lambda: dataset_iterator(num_steps=2),
        early_stopping=es_config,
    )

    # Second run: resume from step 4 (resuming patience=2).
    # At step 6: valid loss = 1.0 (patience=3).
    # At step 8: valid loss = 1.0 (patience=4 >= patience limit -> triggers early stopping).
    result_resumed = flax_train.train(
        model=model,
        opt=opt,
        train_step=train_step,
        dataset_iterator=dataset_iterator(num_steps=None),
        dummy_data_fn=lambda x: x["data"],
        num_train_steps=20,
        working_path=work_dir.full_path,
        rng_key=jax.random.PRNGKey(42),
        checkpoint_every_n_steps=2,
        valid_every_n_steps=2,
        valid_step=valid_step,
        valid_dataset_iterator_fn=lambda: dataset_iterator(num_steps=2),
        early_stopping=es_config,
    )

    # Early stopping should have triggered at step 10 with validations at steps 6, 8, 10
    self.assertEqual([l.step for l in result_resumed.valid_logs], [6, 8, 10])

  def test_restore_checkpoint_empty_manager(self):
    work_dir = self.create_tempdir()
    ckpt_dir = os.path.join(work_dir.full_path, "checkpoints")
    mngr = ocp.CheckpointManager(
        ckpt_dir,
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )
    model = SimpleModel(hidden_dim=8)
    opt = optax.adam(1e-3)
    dummy_input = np.random.normal(size=(4, 8)).astype(np.float32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    init_opt_state = opt.init(init_params["params"])
    rng_key = jax.random.PRNGKey(42)

    target_state = flax_train.CheckpointState(
        model_params=init_params,
        opt_state=init_opt_state,
        rng_key=rng_key,
        step=0,
    )
    restored = flax_train._restore_checkpoint(
        checkpoint_manager=mngr,
        target=target_state,
    )
    self.assertIsNone(restored)

  def test_save_and_restore_checkpoint(self):
    work_dir = self.create_tempdir()
    ckpt_dir = os.path.join(work_dir.full_path, "checkpoints")
    mngr = ocp.CheckpointManager(
        ckpt_dir,
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )
    model = SimpleModel(hidden_dim=8)
    opt = optax.adam(1e-3)
    dummy_input = np.random.normal(size=(4, 8)).astype(np.float32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    init_opt_state = opt.init(init_params["params"])
    rng_key = jax.random.PRNGKey(123)

    es_monitor = early_stopping_monitor.EarlyStoppingMonitor(
        early_stopping_monitor.EarlyStoppingMonitorConfig(patience=5)
    )
    es_monitor.add_loss(5, 0.42, init_params)

    save_state = flax_train.CheckpointState(
        model_params=init_params,
        opt_state=init_opt_state,
        rng_key=rng_key,
        step=5,
        es_state=es_monitor.get_state(),
    )
    flax_train._save_checkpoint(
        checkpoint_manager=mngr,
        state=save_state,
    )
    mngr.wait_until_finished()

    target_state = flax_train.CheckpointState(
        model_params=init_params,
        opt_state=init_opt_state,
        rng_key=jax.random.PRNGKey(0),
        step=0,
        es_state=es_monitor.get_template_state(init_params),
    )
    restored = flax_train._restore_checkpoint(
        checkpoint_manager=mngr,
        target=target_state,
    )
    self.assertIsNotNone(restored)
    self.assertEqual(restored.step, 5)
    self.assertTrue(jnp.array_equal(restored.rng_key, rng_key))
    self.assertIsNotNone(restored.es_state)
    self.assertEqual(restored.es_state["best_loss"], 0.42)
    self.assertEqual(restored.es_state["best_step"], 5)


if __name__ == "__main__":
  absltest.main()
