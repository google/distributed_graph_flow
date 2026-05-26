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

"""Code to compute predictions of a model trained with the 10-lines API."""

import logging
from typing import Iterator, Optional, Sequence, Tuple
import apache_beam as beam
from apache_beam.utils import shared as beam_shared
from dgf.src.analyse import schema as analyse_schema_lib
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.learning.ten_lines import common as ten_lines_common
from dgf.src.learning.ten_lines import node_prediction_model as node_prediction_lib
from dgf.src.sampling import beam_semi_distributed_sampler
from dgf.src.sampling import beam_semi_distributed_sampler_v2
import numpy as np

# An individual model prediction. The semantic depends on the model. For
# example, for a node classification model, Prediction will be the probabilities
# of each label classes available with "model.label_classes()".
Prediction = np.ndarray
KeyedPrediction = Tuple[distributed_graph.NodeId, Prediction]
PKeyedPrediction = beam.PCollection[KeyedPrediction]


class PredictNodePredictionModelCache:
  """Container to share a model between workers on the same machine."""

  def __init__(self):
    self.by_path = {}


class PredictNodePredictionModel(beam.DoFn):
  """Generates prediction of a node prediction model on graph samples.

  Operates on batches of graph samples and return individual predictions.
  """

  # All the sampler in this process.
  shared_model = beam_shared.Shared()

  def __init__(self, model_path: str):
    self.model_path = model_path

  def setup(self):
    # Load the model; make sure it is shard among all the workers on the same
    # machine.
    def initializer():
      return PredictNodePredictionModelCache()

    self.cache = PredictNodePredictionModel.shared_model.acquire(initializer)
    if self.model_path not in self.cache.by_path:
      logging.info("Load model in memory")
      model = ten_lines_common.load_model(self.model_path)
      self.cache.by_path[self.model_path] = model
    self.model = self.cache.by_path[self.model_path]

  def process(
      self,
      graph_samples: Sequence[distributed_graph.KeyedInMemoryGraph],
  ) -> Iterator[KeyedPrediction]:

    if self.model is None:
      raise ValueError("Model not loaded.")
    assert isinstance(self.model, node_prediction_lib.NodePredictionModel)

    samples = []
    keys = []
    for key, sample in graph_samples:
      assert key is not None
      samples.append(sample)
      keys.append(key)

    predictions = self.model.predict_on_graph_sample_batch(samples)

    for key_idx, key in enumerate(keys):
      yield key, predictions[key_idx]


# TODO(gbm): Also have a "predict" method that takes a Beam distributed graph
# when we have a good sampler for this case.
# TODO(gbm): Add it as a model method e.g. "model.predict_beam(...)".
def predict_node_prediction_on_graph_path(
    pbegin: beam.Pipeline,
    model_path: str,
    graph_path: str,
    seed_node_ids: Optional[beam.PCollection[distributed_graph.NodeId]] = None,
    beam_feature_collection: bool = False,
) -> PKeyedPrediction:
  """Computes the predictions of a model on a graph stored on disk.

  Usage example:

  ```python
  import apache_beam as beam

  with beam.Pipeline() as pbegin:
    predictions = predict_node_prediction_on_graph_path(
        pbegin,
        model_path="/path/to/model",
        graph_path="/path/to/graph",
    )
    # predictions is a PCollection of (node_id, prediction) tuples.
  ```

  Args:
    pbegin: Beam pipeline.
    model_path: Path to a trained model.
    graph_path: Path to a GF Graph.
    seed_node_ids: PCollection of node ids to sample from. If None, all nodes in
      the target nodeset of the model are used.
    beam_feature_collection: If False (default), the feature values are gathered
      by the in-memory sampler. This option is fast but requires more RAM. If
      True, the feature values are gathered by a Beam join after the sampling.

  Returns:
    PCollection of (node_id, prediction) tuples.
  """

  model = ten_lines_common.load_model(model_path)
  assert isinstance(model, node_prediction_lib.NodePredictionModel)

  if seed_node_ids is None:
    # List all the nodes as seeds if the user did not provide them.
    target_nodeset = model.data().task.target_nodeset
    primary_key = analyse_schema_lib.primary_feature(
        target_nodeset, model.data().schema.node_sets[target_nodeset]
    )
    graph = gf_graph_in_beam_lib.read_graph(
        pbegin,
        graph_path,
        schema_filter=schema_lib.GraphSchemaFilter(
            nodeset_fn=lambda key, sch: key == target_nodeset,
            edgeset_fn=lambda key, sch: False,
            feature_fn=lambda key, sch: key == primary_key,
        ),
    )
    seed_node_ids = beam_semi_distributed_sampler.extract_beam_nodes_ids(
        graph, target_nodeset
    )
    seed_node_ids = seed_node_ids | "Reshuffle seeds" >> beam.Reshuffle()

  # Generate samples
  graph_samples, _ = (
      beam_semi_distributed_sampler_v2.sample_with_beam_semi_distributed_sampler_v2(
          graph_path,
          model.data().sampling_plan,
          seed_node_ids,
          beam_feature_collection=beam_feature_collection,
      )
  )

  predictions = (
      graph_samples
      | "Batch samples" >> beam.BatchElements(max_batch_size=20)
      | "Predict"
      >> beam.ParDo(
          PredictNodePredictionModel(model_path),
      )
  )

  return predictions
