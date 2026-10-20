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

"""Toy usage of DGF with only the packages of `requirements.txt` installed.

Usage: python script/toy_minimal.py

This script is run during the release to check that DGF works without its weak
dependencies (e.g. TensorFlow). `toy_full.py` tests the weak dependencies.
"""

import os
import tempfile

import dgf
import numpy as np


def main(work_dir: str):
  """Trains a node classification model. Returns the model and its data."""
  rng = np.random.default_rng(0)
  num_nodes = 200
  schema = dgf.data.GraphSchema(
      node_sets={
          "paper": dgf.data.NodeSchema(
              features={
                  "feature": dgf.data.FeatureSchema(
                      format=dgf.data.FeatureFormat.FLOAT_64,
                      semantic=dgf.data.FeatureSemantic.NUMERICAL,
                  ),
                  "label": dgf.data.FeatureSchema(
                      format=dgf.data.FeatureFormat.BYTES,
                      semantic=dgf.data.FeatureSemantic.CATEGORICAL,
                  ),
              }
          )
      },
      edge_sets={
          "cites": dgf.data.EdgeSchema(source="paper", target="paper")
      },
  )
  feature = rng.random(num_nodes)
  graph = dgf.data.InMemoryGraph(
      node_sets={
          "paper": dgf.data.InMemoryNodeSet(
              num_nodes=num_nodes,
              features={
                  "feature": feature,
                  "label": np.where(feature > 0.5, b"A", b"B"),
              },
          )
      },
      edge_sets={
          "cites": dgf.data.InMemoryEdgeSet(
              adjacency=rng.integers(0, num_nodes, size=(2, 1000))
          )
      },
  )
  model = dgf.learning.train_node_model(
      graph=graph,
      schema=schema,
      target_column="label",
      work_dir=os.path.join(work_dir, "train"),
      num_train_steps=20,
      valid_every_n_steps=10,
      batch_size=8,
  )
  return model, graph, schema


if __name__ == "__main__":
  with tempfile.TemporaryDirectory() as tmp_dir:
    main(tmp_dir)
  print("toy_minimal.py: Success")
