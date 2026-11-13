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

"""Integration test of the Vertex AI deployment of a model trained on Spanner Graph.

The test does the following:
  1. Reads a graph from Spanner Graph.
  2. Trains a node classification model on it.
  3. Samples subgraphs directly from Spanner Graph with the model's sampling
     plan.
  4. Generates local predictions on the graph samples.
  5. Converts the graph samples into Vertex AI JSON instances, and checks the
     predictions of the exported TF function on those instances.
  6. Deploys the model to Vertex AI with to_vertex_ai.
  7. Generates online Vertex AI predictions on the instances, and checks them
     against the local predictions.

The test requires access to the GCP project _PROJECT.
"""

import os
from typing import Any

from absl.testing import absltest
from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.deploy import vertex
from dgf.src.io import cache as cache_lib
from dgf.src.io import tf as io_tf_lib
from dgf.src.io.gcp import spanner_graph as spanner_graph_io_lib
from dgf.src.learning.ten_lines import common as common_lib
from dgf.src.learning.ten_lines import node_prediction_model
from dgf.src.learning.ten_lines import node_prediction_train
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling.gcp import spanner_graph_sampler as spanner_graph_sampler_lib
import numpy as np
import tensorflow as tf

_PROJECT = "biggraphs-poc"
_LOCATION = "us-central1"
_SPANNER_GRAPH = {
    "project": _PROJECT,
    "instance": "gcp-gnns",
    "database": "ogbn_arxiv",
    "graph": "ogbn_arxiv",
}
_TARGET_NODESET = "nodes"
_TARGET_COLUMN = "labels"
_FEATURE_DIM = 128
_SEED_IDS = [b"10", b"11", b"12"]
_NUM_TRAIN_STEPS = 200
_CACHE_DIR = "/tmp/vertex_integration_2_test"
_MODEL_DIR_ON_GCS = "gs://gf-experiment-gbm-test/vertex_integration_2_test"
_DISPLAY_NAME = "dgf_vertex_integration_2_test"
# Hardware running the model on Vertex AI: "cpu" or "gpu".
_TARGET = "cpu"
# Tolerance between the local (JAX) and the TF / Vertex AI predictions.
_ATOL = 2e-3


def read_graph() -> (
    tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
):
  """Reads the Spanner Graph."""

  graph, schema = cache_lib.cache(
      os.path.join(_CACHE_DIR, "graph.pkl"),
      lambda: spanner_graph_io_lib.read_spanner_graph(**_SPANNER_GRAPH),
  )
  # Spanner arrays have an unknown length. Make "feat" a fixed size vector.
  schema.node_sets[_TARGET_NODESET].features["feat"].shape = (_FEATURE_DIM,)
  features = graph.node_sets[_TARGET_NODESET].features
  if features["feat"].dtype == object:
    features["feat"] = np.stack(list(features["feat"])).astype(np.float64)
  schema.node_sets[_TARGET_NODESET].features[
      _TARGET_COLUMN
  ].semantic = schema_lib.FeatureSemantic.CATEGORICAL
  schema = analyse_schema_lib.infer_schema_semantic(schema)
  return graph, schema


def train_model(
    graph: in_memory_graph_lib.InMemoryGraph, schema: schema_lib.GraphSchema
) -> node_prediction_model.NodePredictionModel:
  """Trains a model."""

  model_path = os.path.join(_CACHE_DIR, "model")
  if os.path.exists(model_path):
    model = common_lib.load_model(model_path)
  else:
    model = node_prediction_train.train_node_model(
        graph=graph,
        schema=schema,
        target_column=_TARGET_COLUMN,
        target_nodeset=_TARGET_NODESET,
        num_train_steps=_NUM_TRAIN_STEPS,
        valid_every_n_steps=100,
        num_valid_steps=20,
        sampling_plan=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset=_TARGET_NODESET,
            num_hops=2,
            hop_width=5,
            multi_visit=False,
        ),
        num_layers=2,
        node_embedding_dim=32,
    )
    model.save(model_path)
  assert isinstance(model, node_prediction_model.NodePredictionModel)
  return model


def graph_sample_to_vertex_instance(
    graph_sample: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    input_names: set[str],
) -> dict[str, Any]:
  """Converts a graph sample into a JSON Vertex AI prediction instance."""

  tf_graph = io_tf_lib.graph_to_tf_graph(graph_sample, schema=schema)
  tf_graph_dict = io_tf_lib.tf_graph_to_tf_graph_dict(tf_graph)
  instance = {}
  for key, value in tf_graph_dict.items():
    if key not in input_names:
      # E.g., the label and the primary IDs.
      continue
    value = value.numpy()
    if value.dtype == object:
      instance[key] = np.vectorize(lambda x: x.decode("utf-8"))(value).tolist()
    else:
      instance[key] = value.tolist()
  # By convention, the seed node is the first node of the target nodeset.
  instance["seed_node_idxs"] = [0]
  return instance


def predict_with_local_tf(
    serving_model: tf.Module, instances: list[dict[str, Any]]
) -> np.ndarray:
  """Runs the batched serving TF model locally on each JSON instance."""

  specs = serving_model.__call__.structured_input_signature[1]
  predictions = []
  for instance in instances:
    kwargs = {
        key: tf.constant([value], dtype=specs[key].dtype)
        for key, value in instance.items()
    }
    predictions.append(
        serving_model.__call__(**kwargs)["predictions"].numpy()[0][0]
    )
  return np.stack(predictions)


class VertexIntegration2Test(absltest.TestCase):

  def test_spanner_graph_to_vertex_ai(self):
    os.makedirs(_CACHE_DIR, exist_ok=True)

    # 1. Read the Spanner Graph.
    graph, schema = read_graph()

    # 2. Train a small model.
    model = train_model(graph, schema)

    # 3. Sample subgraphs directly from Spanner with the model's sampling plan.
    sampling_plan = model.data().sampling_plan
    assert sampling_plan is not None
    spanner_sampler = (
        spanner_graph_sampler_lib.create_spanner_graph_sampler_gcp(
            **_SPANNER_GRAPH, plan=sampling_plan, schema=model.data().schema
        )
    )
    graph_samples = spanner_sampler.sample(_SEED_IDS)
    self.assertLen(graph_samples, len(_SEED_IDS))

    # 4. Local predictions on the graph samples.
    local_predictions = model.predict_on_graph_sample_batch(graph_samples)
    self.assertEqual(
        local_predictions.shape, (len(_SEED_IDS), model.num_label_classes())
    )

    # 5. Convert the samples to Vertex AI instances, and check them locally.
    instance_schema, _ = model._extract_serving_schemata()  # pylint: disable=protected-access
    input_names = set(instance_schema["properties"])
    serving_model = vertex._create_batched_serving_model(  # pylint: disable=protected-access
        model.to_tensorflow_function(input_format="TF_GRAPH_DICT"),
        input_names,
    )
    instances = [
        graph_sample_to_vertex_instance(
            graph_sample, model.data().schema, input_names
        )
        for graph_sample in graph_samples
    ]
    local_tf_predictions = predict_with_local_tf(serving_model, instances)
    np.testing.assert_allclose(
        local_tf_predictions, local_predictions, atol=_ATOL
    )

    # 6. Deploy to Vertex AI.
    endpoint = vertex.to_vertex_ai(
        model,
        model_dir_on_gcs=_MODEL_DIR_ON_GCS,
        display_name=_DISPLAY_NAME,
        location=_LOCATION,
        project=_PROJECT,
        target=_TARGET,
    )

    # 7. Online predictions, with one graph sample per request (graph samples
    # of different sizes cannot be batched in a single request).
    vertex_predictions = []
    for instance in instances:
      response = endpoint.predict(instances=[instance])
      vertex_predictions.append(np.asarray(response.predictions[0])[0])
    vertex_predictions = np.stack(vertex_predictions)
    np.testing.assert_allclose(
        vertex_predictions, local_predictions, atol=_ATOL
    )


if __name__ == "__main__":
  absltest.main()
