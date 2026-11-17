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

"""Tests that DGF works with only the packages of `requirements.txt`.

The packages only listed in `requirements-dev.txt` (and `clu`, which imports
TensorFlow) are made non-importable before importing DGF.
"""

import sys

# Python modules of the packages in `requirements-dev.txt` that are not in
# `requirements.txt`. Keep in sync with `requirements-dev.txt`.
_DEV_ONLY_MODULES = (
    "clu",
    "tensorflow",
    "tensorflow_gnn",
    "tf_keras",
    "apache_beam",
    "ogb",
)

for _module in _DEV_ONLY_MODULES:
  # Setting a module to None makes its import raise an ImportError.
  sys.modules[_module] = None

# pylint: disable=g-import-not-at-top,g-bad-import-order
import os

from absl.testing import absltest
import dgf
from dgf.src.util import gen_test_graph
# pylint: enable=g-import-not-at-top,g-bad-import-order


class MinimalRequirementsTest(absltest.TestCase):

  def test_dev_only_modules_are_blocked(self):
    for module in _DEV_ONLY_MODULES:
      with self.assertRaises(ImportError):
        __import__(module)

  def test_train_node_model(self):
    graph, schema = gen_test_graph.gen_toy_classification_dataset()
    work_dir = self.create_tempdir().full_path
    model = dgf.learning.train_node_model(
        graph=graph,
        schema=schema,
        target_nodeset="N1",
        target_column="label",
        work_dir=os.path.join(work_dir, "train"),
        num_train_steps=20,
        valid_every_n_steps=10,
        batch_size=5,
        num_sampling_hops=1,
        sampling_width=3,
    )
    model.describe()
    model.evaluate(graph)
    predictions = model.predict(graph, seed_node_idxs=[0, 1, 2])
    self.assertLen(predictions, 3)

    model_path = os.path.join(work_dir, "model")
    model.save(model_path)
    loaded_model = dgf.learning.load_model(model_path)
    loaded_model.predict(graph, seed_node_idxs=[0, 1, 2])

  def test_train_node_embedding_model(self):
    graph, schema = gen_test_graph.gen_toy_classification_dataset()
    work_dir = self.create_tempdir().full_path
    model = dgf.learning.train_node_embedding_model(
        graph=graph,
        schema=schema,
        target_nodeset="N1",
        exclude_features=["label"],
        work_dir=os.path.join(work_dir, "train"),
        num_train_steps=10,
        valid_every_n_steps=5,
        batch_size=5,
        num_sampling_hops=1,
        sampling_width=3,
        node_embedding_dim=8,
        fast_compile=True,
    )
    self.assertIsInstance(model, dgf.learning.NodeEmbeddingModel)
    model.describe()
    model.evaluate(graph, seed_node_idxs=[0, 1, 2])
    embeddings = model.predict(graph, seed_node_idxs=[0, 1, 2])
    self.assertEqual(embeddings.shape, (3, 8))

    model_path = os.path.join(work_dir, "model")
    model.save(model_path)
    loaded_model = dgf.learning.load_model(model_path)
    self.assertIsInstance(loaded_model, dgf.learning.NodeEmbeddingModel)
    loaded_model.predict(graph, seed_node_idxs=[0, 1, 2])


if __name__ == "__main__":
  absltest.main()
