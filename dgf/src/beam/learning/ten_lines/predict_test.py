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

import functools
import os
from typing import Tuple
from absl.testing import absltest
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_test_util
from dgf.src.beam.learning.ten_lines import predict as predict_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.generate import graphs as synthetic_lib
from dgf.src.io import graph_in_memory as gf_graph_in_memory
from dgf.src.learning.ten_lines import node_prediction_model as node_prediction_lib
from dgf.src.learning.ten_lines import node_prediction_train as node_prediction_train_lib
from dgf.src.util import test_util
import numpy as np

test_util.disable_diff_truncation()


def _gen_graph_real_looking() -> (
    Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
):
  """Generate a toy dataset with real looking features, but without patterns."""
  schema = schema_lib.GraphSchema(
      node_sets={
          "client": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "city": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                      num_categorical_values=10,
                  ),
                  "age": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
                  "categorical_label": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                      num_categorical_values=3,
                  ),
              }
          ),
          "transaction": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "date": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                  ),
              }
          ),
      },
      edge_sets={
          "transation_to_client": schema_lib.EdgeSchema(
              source="transaction", target="client"
          ),
      },
  )
  return (
      synthetic_lib.generate_synthetic_graph(
          schema,
          synthetic_lib.SyntheticGraphConfig(num_nodes=1000, num_edges=1000),
      ),
      schema,
  )


class PredictTest(absltest.TestCase):

  def test_base(self):
    tmpdir = self.create_tempdir().full_path

    # Train a model
    graph, schema = _gen_graph_real_looking()
    model = node_prediction_train_lib.train_node_model(
        graph=graph,
        schema=schema,
        target_nodeset="client",
        target_column="categorical_label",
        num_train_steps=10,
        valid_every_n_steps=50,
    )

    # Generate reference predictions
    seed_node_idxs = np.arange(graph.node_sets["client"].num_nodes)  # pyrefly: ignore[no-matching-overload]
    expected_raw_predictions = model.predict(graph, seed_node_idxs)

    expected_predictions = []
    for node_idx, node_id in enumerate(
        graph.node_sets["client"].features["#id"]
    ):
      expected_predictions.append(
          (node_id.item(), expected_raw_predictions[node_idx])
      )

    # Save model
    model_path = os.path.join(tmpdir, "model")
    model.save(model_path)

    # Save some graph data.
    graph_path = os.path.join(tmpdir, "graph")
    gf_graph_in_memory.write_graph(graph, schema, graph_path)

    # Generate predictions with beam
    with test_pipeline.TestPipeline() as pbegin:
      predictions = predict_lib.predict_node_prediction_on_graph_path(
          pbegin, model_path, graph_path
      )

      beam_test_util.assert_that(
          predictions,
          beam_test_util.equal_to(
              expected_predictions,
              equals_fn=functools.partial(test_util.are_equal, abs_tol=0.0001),
          ),
      )


if __name__ == "__main__":
  absltest.main()
