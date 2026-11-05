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

"""Checks that the DGF SavedModels have the same interface as the TF-GNN ones.

A small TF-GNN model is trained (like in the quality benchmark) and exported
with the TF-GNN exporter. Both the DGF and TF-GNN SavedModels are then called on
the same serialized TF GNN Graph Samples.
"""

from absl.testing import absltest
from dgf.benchmark.quality import tfgnn_solver
from dgf.src.data import schema as schema_lib
from dgf.src.io import hgraph_in_memory
from dgf.src.io import tf_graph_sample as tf_graph_sample_lib
from dgf.src.learning.ten_lines import node_prediction_train
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.util import gen_test_graph
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner

_TARGET_NODESET = "N1"
_TARGET_COLUMN = "label"
_NUM_CLASSES = 2
_NUM_NODES = 100
_TFGNN_HPARAMS = tfgnn_solver.TFGNNHParams(
    initial_dense_dim=16,
    units=16,
    message_dim=16,
    num_graph_updates=1,
    state_dropout_rate=0.0,
)


def _tfgnn_input_schema(
    schema: schema_lib.GraphSchema,
) -> schema_lib.GraphSchema:
  """Returns the schema of the features consumed by the TF-GNN model."""
  node_sets = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    features = {}
    for feature_name, feature_schema in nodeset_schema.features.items():
      if feature_schema.semantic == schema_lib.FeatureSemantic.PRIMARY_ID or (
          nodeset_name == _TARGET_NODESET and feature_name == _TARGET_COLUMN
      ):
        continue
      if feature_schema.semantic != schema_lib.FeatureSemantic.NUMERICAL:
        raise ValueError(f"Unexpected feature {feature_name!r}")
      # Numerical features are fed as floats to the TF-GNN model.
      features[feature_name] = schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      )
    node_sets[nodeset_name] = schema_lib.NodeSchema(features=features)
  edge_sets = {
      edgeset_name: schema_lib.EdgeSchema(
          source=edgeset_schema.source, target=edgeset_schema.target
      )
      for edgeset_name, edgeset_schema in schema.edge_sets.items()
  }
  return schema_lib.GraphSchema(node_sets=node_sets, edge_sets=edge_sets)


class _ParseGraphSamples(tf.keras.layers.Layer):
  """Parses serialized TF GNN Graph Samples into a TF-GNN model input.

  The graph samples are merged into a single graph (one component per sample),
  and a reverse edge set is added for each edge set (like in the quality
  benchmark). The seed node of each graph sample is the first node of the target
  nodeset.
  """

  def __init__(
      self,
      schema: schema_lib.GraphSchema,
      input_schema: schema_lib.GraphSchema,
  ):
    super().__init__()
    self._graph_spec = tfgnn.create_graph_spec_from_schema_pb(
        hgraph_in_memory.schema_to_tfgnn_schema(schema)
    )
    self._input_schema = input_schema

  def call(self, examples: tf.Tensor) -> tuple[tfgnn.GraphTensor, tf.Tensor]:
    graph = tfgnn.parse_example(
        self._graph_spec, examples
    ).merge_batch_to_components()

    node_sets = {}
    for nodeset_name, nodeset_schema in self._input_schema.node_sets.items():
      nodeset = graph.node_sets[nodeset_name]
      node_sets[nodeset_name] = tfgnn.NodeSet.from_fields(
          sizes=nodeset.sizes,
          features={
              feature_name: tf.cast(nodeset[feature_name], tf.float32)
              for feature_name in nodeset_schema.features
          },
      )
    edge_sets = {}
    for edgeset_name, edgeset_schema in self._input_schema.edge_sets.items():
      edgeset = graph.edge_sets[edgeset_name]
      sources = edgeset.adjacency.source
      targets = edgeset.adjacency.target
      edge_sets[edgeset_name] = tfgnn.EdgeSet.from_fields(
          sizes=edgeset.sizes,
          adjacency=tfgnn.Adjacency.from_indices(
              source=(edgeset_schema.source, sources),
              target=(edgeset_schema.target, targets),
          ),
      )
      edge_sets[tfgnn_solver.reverse_edgeset_name(edgeset_name)] = (
          tfgnn.EdgeSet.from_fields(
              sizes=edgeset.sizes,
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(edgeset_schema.target, targets),
                  target=(edgeset_schema.source, sources),
              ),
          )
      )
    seeds = tf.math.cumsum(
        tf.cast(graph.node_sets[_TARGET_NODESET].sizes, tf.int64),
        exclusive=True,
    )
    input_graph = tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(sizes=graph.context.sizes),
        node_sets=node_sets,
        edge_sets=edge_sets,
    )
    return input_graph, seeds


def _train_and_export_tfgnn_model(
    schema: schema_lib.GraphSchema,
    examples: tf.Tensor,
    labels: np.ndarray,
    path: str,
) -> None:
  """Trains a small TF-GNN model and exports it like the TF-GNN runner."""
  input_schema = _tfgnn_input_schema(schema)
  parse = _ParseGraphSamples(schema, input_schema)
  example_graph, _ = parse(examples[:2])
  graph_spec = example_graph.spec.relax(
      num_components=True, num_nodes=True, num_edges=True
  )
  core_model = tfgnn_solver.build_model(
      graph_spec,
      input_schema=input_schema,
      target_nodeset=_TARGET_NODESET,
      num_outputs=_NUM_CLASSES,
      hparams=_TFGNN_HPARAMS,
  )

  serialized_examples = tf.keras.layers.Input(
      shape=[], dtype=tf.string, name="examples"
  )
  graph, seeds = parse(serialized_examples)
  probabilities = tf.keras.layers.Softmax()(core_model((graph, seeds)))
  model = tf.keras.Model(serialized_examples, probabilities)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-2),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  )
  dataset = tf.data.Dataset.from_tensor_slices((examples, labels)).batch(16)
  model.fit(dataset, epochs=2, verbose=2)

  runner.export_model(model, path, output_names="predictions")


class SavedModelTfGnnLikeTest(absltest.TestCase):

  def test_dgf_and_tfgnn_saved_models_are_aligned(self):
    graph, schema = gen_test_graph.gen_toy_classification_dataset(
        num_n1_nodes=_NUM_NODES
    )
    model = node_prediction_train.train_node_model(
        graph=graph,
        schema=schema,
        target_nodeset=_TARGET_NODESET,
        target_column=_TARGET_COLUMN,
        num_train_steps=20,
        batch_size=8,
    )
    model_schema = model.data().schema
    sampling_plan = model.data().sampling_plan
    assert sampling_plan is not None

    samples = in_memory_sampler_lib.create_sampler(
        graph=graph, plan=sampling_plan, schema=model_schema, batch_size=16
    ).sample(list(range(_NUM_NODES)))
    examples = tf.constant([
        tf_graph_sample_lib.graph_to_tfgnn_graph(
            sample, schema=model_schema
        ).SerializeToString()
        for sample in samples
    ])
    labels = graph.node_sets[_TARGET_NODESET].features[_TARGET_COLUMN]

    dgf_path = self.create_tempdir("dgf").full_path
    model.to_tensorflow_saved_model(dgf_path)
    tfgnn_path = self.create_tempdir("tfgnn").full_path
    _train_and_export_tfgnn_model(model_schema, examples, labels, tfgnn_path)

    dgf_model = tf.saved_model.load(dgf_path)
    tfgnn_model = tf.saved_model.load(tfgnn_path)

    # Same signatures.
    self.assertEqual(
        list(dgf_model.signatures.keys()), list(tfgnn_model.signatures.keys())
    )
    dgf_signature = dgf_model.signatures["serving_default"]
    tfgnn_signature = tfgnn_model.signatures["serving_default"]
    self.assertEqual(
        dgf_signature.structured_input_signature,
        tfgnn_signature.structured_input_signature,
    )
    self.assertEqual(
        dgf_signature.structured_outputs.keys(),
        tfgnn_signature.structured_outputs.keys(),
    )
    for output_name, dgf_output in dgf_signature.structured_outputs.items():
      tfgnn_output = tfgnn_signature.structured_outputs[output_name]
      self.assertEqual(dgf_output.dtype, tfgnn_output.dtype)
      self.assertEqual(dgf_output.shape.rank, tfgnn_output.shape.rank)

    # Both models can be called on the same data.
    dgf_predictions = dgf_signature(examples=examples)["predictions"].numpy()
    tfgnn_predictions = tfgnn_signature(examples=examples)[
        "predictions"
    ].numpy()
    self.assertEqual(dgf_predictions.shape, (_NUM_NODES, _NUM_CLASSES))
    self.assertEqual(tfgnn_predictions.shape, (_NUM_NODES, _NUM_CLASSES))
    np.testing.assert_allclose(dgf_predictions.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(tfgnn_predictions.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
