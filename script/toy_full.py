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

"""Toy usage of DGF with all the packages of `requirements-dev.txt` installed.

Usage: python script/toy_full.py

This script is run during the release. It runs `toy_minimal.py`, and tests
features using the weak dependencies of DGF (e.g. TensorFlow, TF-GNN, CLU).
"""

import os
import tempfile

import dgf
import tensorflow as tf
import toy_minimal


def main(work_dir: str):
  """Runs the minimal toy examples, and the weak dependency features."""
  model, graph, schema = toy_minimal.main(work_dir)

  # TensorBoard summaries (requires CLU and TensorFlow).
  summary_dir = os.path.join(work_dir, "train", "checkpoint", "summary")
  assert os.listdir(summary_dir), f"No TensorBoard summaries in {summary_dir}"

  # TF-GNN conversion and TensorFlow export of the model.
  sampler = dgf.sampling.create_sampler(
      graph=graph,
      plan=model.data().sampling_plan,
      schema=schema,
      batch_size=2,
  )
  samples = [sampler.sample(0), sampler.sample(1)]
  serialized_samples = tf.constant([
      dgf.convert.graph_to_tfgnn_graph(sample, schema).SerializeToString()
      for sample in samples
  ])
  tf_predict_fn = model.to_tensorflow_function(
      input_format="SERIALIZED_TFGNN_GRAPHS"
  )
  predictions = tf_predict_fn(serialized_samples)
  assert predictions.shape[0] == 2, predictions

  saved_model_path = os.path.join(work_dir, "saved_model")
  tf.saved_model.save(tf_predict_fn, saved_model_path)
  loaded_predictions = tf.saved_model.load(saved_model_path)(serialized_samples)
  assert loaded_predictions.shape == predictions.shape, loaded_predictions


if __name__ == "__main__":
  with tempfile.TemporaryDirectory() as tmp_dir:
    main(tmp_dir)
  print("toy_full.py: Success")
