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

r"""Node Classification Example.

This example shows how to train, evaluate, inspect, and export a node
classification model with the high level API (a.k.a. 10-lines API).

This example contains more-of-less the same code as the Getting Started
Notebook, but without detailed explanations:
http://go/graph-flow/link_getting_started_simple_api

Usage example:

# Run locally (you need a GPU)
blaze run -c opt //third_party/py/dgf/examples:node_classification_model
"""

from absl import app
from absl import flags
import dgf
import tensorflow as tf

_OUTPUT_DIR = flags.DEFINE_string("output_dir", "/tmp/dgf_model", "")


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  dgf.filesystem.makedirs(_OUTPUT_DIR.value)

  # Get the graph data
  graph, schema = dgf.io.fetch_ogb_graph("arxiv")

  # Print the graph schema
  dgf.analyse.print_schema(schema)

  # If your graph doesn't have semantic information (e.g., features are
  # numerical, categorical, embedding), training will be poor and you'll get
  # warning messages. You can set the feature semantics manually. For example:
  #   schema.node_sets["paper"].features["labels"].semantic = dgf.data.FeatureSemantic.CATEGORICAL
  # Or use the automatic inference tool. For example:
  #   schema = dgf.analyse.infer_schema_semantic(schema)

  # [Optional] Validate the graph data
  dgf.validate.validate_graph(graph, schema)

  # Train the model
  # With a GPU/TPU, you'll train at >100 steps per seconds.
  model = dgf.learning.train_node_model(
      graph=graph,
      schema=schema,
      target_column="labels",
      verbose=1,
      # Uncomment if you don't have a GPU/TPU
      # num_train_steps=200,
      # valid_every_n_steps=50,
  )

  # Create a HTML file with the model's description:
  description = model.describe()
  with open(_OUTPUT_DIR.value + "/description.html", "w") as f:
    f.write(description.html())

  # Making a prediction
  predictions = model.predict(graph, seed_node_idxs=[0, 1, 2, 3])
  print("Model's prediction:", predictions.argmax(axis=1))

  # Manually evaluate model (in addition to the one in the model description)
  evaluation = model.evaluate(graph)
  with open(_OUTPUT_DIR.value + "/evaluation.html", "w") as f:
    f.write(evaluation.html())

  # Save the model
  model.save(_OUTPUT_DIR.value + "/model")

  # Export the model to TensorFlowSavedModel format
  tf_model = model.to_tensorflow_function()
  tf.saved_model.save(tf_model, _OUTPUT_DIR.value + "/tf_model")

  print("Results are available in :", _OUTPUT_DIR.value)


if __name__ == "__main__":
  app.run(main)
