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

"""Example showing how to train and evaluate a GNN model using the simple API.

In a few lines, the graph is loaded in memory, normalized, sampled, and a GNN is
trained. After training, the model is evaluated and saved for later reuse.

Usage example:
  blaze run -c opt //third_party/py/dgf/examples/experimental:ten_lines_arxiv

TODO(gbm): Turn into a colab.
"""

from absl import app
import dgf


def main(argv):

  # Load the training dataset.
  graph, schema = dgf.io.read_graph(
      "/cns/iz-d/home/research-graph/public/graphflow_datasets/fetch_repo/ogb_arxiv"
  )

  # You can also use an example graph.
  # graph, schema = dgf.io.fetch_ogb_graph("arxiv")

  # Train the model.
  # TODO(gbm): Use the 10-lines API when available (instead of using internals).
  model = dgf.learning.train_node_model(
      graph=graph,
      schema=schema,
      target_column="labels",
      num_train_steps=2000,
  )

  # Print informations about the model.
  print("Model:\n", model.describe())

  # Evaluate the model.
  # TODO(gbm): Lets use a test dataset instead.
  evaluation = model.evaluate(graph)
  print("Evaluation:\n", evaluation)

  # Save model for later reuse.
  model.save("/tmp/my_model")


if __name__ == "__main__":
  app.run(main)
