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

"""Test utilities for text feature normalization."""

from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
import numpy as np

DUMMY_EMBEDDER_VARIABLE_NAME = "text_emb_table"
_NUM_BUCKETS = 32


def save_dummy_tf_hub_embedder(export_dir: str, dim: int = 8) -> None:
  """Saves a tiny TF-Hub-like SavedModel hashing strings into `dim`-d vectors."""

  class DummyHubTextModule(tf.Module):
    """Maps each string to a row of a fixed table using a hash."""

    def __init__(self):
      super().__init__()
      init = np.arange(_NUM_BUCKETS * dim, dtype=np.float32).reshape(
          _NUM_BUCKETS, dim
      )
      self.table = tf.Variable(
          init / 100.0, trainable=False, name=DUMMY_EMBEDDER_VARIABLE_NAME
      )

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)]
    )
    def __call__(self, texts: tf.Tensor) -> tf.Tensor:
      buckets = tf.strings.to_hash_bucket_fast(texts, _NUM_BUCKETS)
      return tf.gather(self.table, buckets)

  tf.saved_model.save(DummyHubTextModule(), export_dir)
