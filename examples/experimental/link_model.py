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

r"""Usage example for Edge Prediction in DGF.

blaze run -c opt //third_party/py/dgf/examples/experimental:link_model -- \
--alsologtostderr
"""

from absl import app
from absl import logging
import dgf


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info("Fetching OGB arxiv graph...")
  graph, schema = dgf.io.fetch_ogb_graph("arxiv")

  del schema.node_sets["nodes"].features["#id"]
  del schema.node_sets["nodes"].features["#split"]
  del schema.node_sets["nodes"].features["labels"]
  dgf.analyse.print_schema(schema)

  logging.info("Training link prediction model...")
  model = dgf.learning.train_link_model(
      graph=graph,
      schema=schema,
      # diagnostic_dir="/tmp/gf",
      num_layers=2,
      node_embedding_dim=64,
      num_sampling_hops=1,
      cache_normalized_features=True,
  )

  # Uncomment the following line, and comment the one above to load a
  # pre-trained model instead.
  # model = dgf.learning.load_model("/tmp/link_prediction_on_arxiv")

  assert isinstance(model, dgf.learning.LinkPredictionModel)

  logging.info("Model description:")
  description = model.describe()
  print("Model:\n", model.describe())

  with open("/tmp/gf/link_arxiv.html", "w") as f:
    f.write(description._repr_html_())

  predictions = model.predict(
      graph=graph, source_node_idxs=[0], target_node_idxs=[1, 2, 3, 4, 5]
  )
  print("Some model predictions:\n", predictions)

  evaluation = model.evaluate(graph)
  print("Evaluation on the training dataset:\n", evaluation)

  model.save("/tmp/link_prediction_on_arxiv")


if __name__ == "__main__":
  app.run(main)
