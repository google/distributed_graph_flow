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

"""The model i.e., the python objects containing everything.

Can be loaded, saved, evaluate, and used to generated predictions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import copy
import dataclasses
import enum
import itertools
import textwrap
from typing import Any

import dataclasses_json
from dgf.src.data import in_memory_graph
from dgf.src.data import padding as padding_data_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.data import tf_in_memory_graph as tf_in_memory_graph_lib
from dgf.src.io import jax as jax_lib
from dgf.src.io import tf as io_tf_lib
from dgf.src.learning.jax.layers import classification as classification_lib
from dgf.src.learning.jax.layers import regression as regression_lib
from dgf.src.learning.ten_lines import analyze as analyze_lib
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import evaluation
from dgf.src.learning.ten_lines import inference
from dgf.src.learning.ten_lines import node_prediction_core_model
from dgf.src.learning.ten_lines import report
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import log
from dgf.src.util import temporal as temporal_util
from dgf.src.util import util
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
import jax

if getattr(tf, "is_available", lambda: True)():
  from jax.experimental import jax2tf
import jax.numpy as jnp
import jaxtyping
import numpy as np

Batch = node_prediction_core_model.Batch
CoreModel = node_prediction_core_model.CoreModel
CoreModelConfig = node_prediction_core_model.CoreModelConfig

# Number of batches sampled and normalized in advance during inference.
_PREDICT_PREFETCH_SIZE = 2


@dataclasses.dataclass
class BatchPrediction:
  """Prediction yielded by "predict_batch" function.

  Attributes:
    batch_seed_node_idxs: The indices of the seed nodes in the input graph, or
      `None` when predicting on pre-sampled subgraphs where input-graph indices
      are unknown.
    normalized_merged_graph: The merged and normalized graph sample used for
      prediction.
    merged_seed_node_idxs: The indices of the seed nodes in the
      `normalized_merged_graph`.
    predictions: The model's output predictions (e.g., probabilities) for the
      batch. A host `np.ndarray` if `predict_batch` is called with
      `to_host=True` (default), and a device `jax.Array` otherwise.
  """

  batch_seed_node_idxs: np.ndarray | None
  normalized_merged_graph: in_memory_graph.InMemoryGraph
  merged_seed_node_idxs: np.ndarray
  predictions: np.ndarray | jax.Array


# TODO(gbm): Populate.
@dataclasses.dataclass
class HParam(common.HParam):
  """Hyperparameters for the NodePredictionModel."""

  pass


class TaskType(enum.Enum):
  NODE_CLASSIFICATION = "NODE_CLASSIFICATION"
  NODE_REGRESSION = "NODE_REGRESSION"


@dataclasses.dataclass
class NodePredictionTask:
  """Internal description of the task to solve.

  Attributes:
    target_nodeset: The name of the nodeset containing the nodes for which a
      prediction is to be made.
    target_column: The name of the feature column within the `target_nodeset`
      that the model is trained to predict.
    normalized_target_column: The name of the normalized version of the
      target_column.
    task_type: The type of prediction task.
  """

  target_nodeset: str
  target_column: str
  normalized_target_column: str | None
  task_type: TaskType


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class TrainingStats:
  """Statistics about the training process.

  Attributes:
    num_train_seed_nodes: The number of seed nodes used for training.
    num_valid_seed_nodes: The number of seed nodes used for validation.
    train_duration_seconds: The duration of the training in seconds.
    num_skipped_train_samples: The number of training samples skipped due to
      padding overflow.
  """

  num_train_seed_nodes: int | None
  num_valid_seed_nodes: int | None
  train_duration_seconds: float
  num_skipped_train_samples: int = 0


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ModelData:
  """All the data of the model. Used for construction and serialization.

  Critically, the model data object should be backward compatible, i.e., loading
  a model data from an old model should still work.
  """

  core_model_config: CoreModelConfig
  task: NodePredictionTask
  hparams: HParam
  schema: schema_lib.GraphSchema
  normalizer_config: normalize_lib.GraphNormalizerConfig
  padding: padding_data_lib.Padding
  # None if the model was trained on already sampled graph samples without an
  # explicit sampling plan: in this case, the sampling plan used to generate
  # the training samples is unknown, and the model can only generate
  # predictions from graph samples.
  sampling_plan: sampling_config_lib.SamplingPlan | None
  feature_stats: statistics_lib.GraphFeatureStatistics
  training_stats: TrainingStats
  temporal_sampling: bool

  nodeset_timestamp_features: dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  edgeset_timestamp_features: dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  final_evaluation: evaluation.Evaluation | None = None

  # This field is serialized / deserialized manually.
  model_params: jaxtyping.PyTree | None = dataclasses.field(
      default_factory=lambda: None,
      metadata=dataclasses_json.config(exclude=dataclasses_json.Exclude.ALWAYS),
      repr=False,
  )


@dataclasses.dataclass
class ModelLiveResource:
  """Resources necessary for inference.

  These live data resources are not saved because they can be recomputed from
  the `ModelData` and may depend on the environment where the model is loaded.

  Unlike the ModelData, ModelLiveResource does not need to be backward
  compatible.
  """

  core_model: CoreModel
  apply_core_model: Callable[[Batch], jnp.ndarray]
  normalized_schema: schema_lib.GraphSchema
  normalizer: normalize_lib.GraphNormalizer


# TODO(gbm): Refactor with common.
# TODO(gbm): Don't use the AutoNormalizeConfig. Store a serializable normalizer
# instead.
class NodePredictionModel(common.Model):
  """The user-visible returned model object."""

  def __init__(self, data: ModelData) -> None:
    super().__init__(data)

    self._data = data
    self._live: ModelLiveResource | None = None

  @classmethod
  def name(cls) -> str:
    return "NodePrediction"

  def data(self) -> ModelData:
    return self._data

  def _internal_save(self, path: str) -> None:
    common.save_params(self._data.model_params, path)

  def _internal_load(self, path: str) -> None:
    self._data.model_params = common.load_params(path)

  def _extract_serving_schemata(
      self,
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extracts (instance_schema, prediction_schema) dicts for Vertex AI serving."""
    if not self.metadata.uuid:
      raise ValueError("Model UUID is not set.")
    model_uuid = self.metadata.uuid
    instance_schema = common.schema_to_serving_signature_dict(
        schema_=self.data().schema,
        target_nodeset=self.data().task.target_nodeset,
        model_name=self.name(),
        model_uuid=model_uuid,
        source_sampling_plan=self.data().sampling_plan,
    )
    task_type = self.data().task.task_type
    if task_type == TaskType.NODE_REGRESSION:
      prediction_schema = {"type": "number"}
    elif task_type == TaskType.NODE_CLASSIFICATION:
      prediction_schema = {"type": "array", "items": {"type": "number"}}
    else:
      raise ValueError(f"Unsupported task_type: {task_type}")
    return instance_schema, prediction_schema

  def describe(self) -> util.RichDisplay:
    # TODO(gbm): Make a good rich report.

    tabs = []

    tabs.append((
        "Objective",
        f"""
<p><b>Node prediction model:</b> Predict the value of a node feature.</p>
<table class="dgf-table">
  <tbody>
    <tr><td>Type</td><td>Node prediction model -  Predict the value of a node feature.</td></tr>
    <tr><td>Target nodeset</td><td>{self._data.task.target_nodeset}</td></tr>
    <tr><td>Target column</td><td>{self._data.task.target_column}</td></tr>
    <tr><td>Number of label classes</td><td>{self.num_label_classes()}</td></tr>
  </tbody>
</table>
""",
    ))

    training_stats_summary = ""
    if self.metadata.trainig_logs is not None:
      training_stats_summary = f"""
    <tr><td>Number of training seed nodes</td><td>{self._data.training_stats.num_train_seed_nodes}</td></tr>
    <tr><td>Number of validation seed nodes</td><td>{self._data.training_stats.num_valid_seed_nodes}</td></tr>
    <tr><td>Number of skipped training samples</td><td>{self._data.training_stats.num_skipped_train_samples}</td></tr>
    <tr><td>Training duration</td><td>{util.format_duration(self._data.training_stats.train_duration_seconds)}</td></tr>
"""

    common_tabs = report.get_common_tabs(
        hparams=self.data().hparams,
        schemas={
            "Raw": self.data().schema,
            "Normalized": self._get_live().normalized_schema,
        },
        feature_stats={
            "Default": self.data().feature_stats,
        },
        sampling_plans={"Default": self.data().sampling_plan},
        training_logs=self.metadata.trainig_logs,
        training_stats_summary=training_stats_summary,
        padding={
            "Default": self.data().padding,
        },
        architecture=self.data().core_model_config.architecture(),
        num_model_weights=common.num_model_weights(self.data().model_params),
        log_messages=self.metadata.captured_logs,
        final_evaluation=self.data().final_evaluation,
    )

    tabs.extend(common_tabs)

    html = report.html_tabs(tabs)
    return util.RichDisplay(html)

  def num_label_classes(self) -> int:
    """Returns the number of classes in the target label column."""
    if self._data.task.task_type == TaskType.NODE_REGRESSION:
      return 1
    assert isinstance(
        self._data.core_model_config.head,
        classification_lib.ClassificationHeadConfig,
    )
    return self._data.core_model_config.head.num_classes

  def label_classes(self) -> list[str]:
    """Returns the string representation of the labels."""
    if self._data.task.task_type != TaskType.NODE_CLASSIFICATION:
      raise ValueError(
          "label_classes is only supported for NODE_CLASSIFICATION tasks."
      )

    target_nodeset = self._data.task.target_nodeset
    target_column = self._data.task.target_column

    try:
      stats = self._data.feature_stats.node_sets[target_nodeset].features[
          target_column
      ]
    except KeyError as e:
      raise ValueError(
          "Could not find statistics for target"
          f" {target_nodeset}.{target_column}"
      ) from e

    if not stats.dictionary:
      raise ValueError(
          f"Target column {target_nodeset}.{target_column} does not have a"
          " string dictionary. It might already be integer-encoded."
      )

    # Sort keys by their index
    sorted_items = sorted(
        stats.dictionary.items(), key=lambda item: item[1].index
    )
    return [key for key, _ in sorted_items]

  def predict(
      self,
      graph: in_memory_graph.InMemoryGraph,
      seed_node_idxs: common.SeedNodeIdxs,
      *,
      verbose: int = 2,
  ) -> np.ndarray:
    """Predicts the target column values for the given seed nodes.

    Args:
      graph: The input graph.
      seed_node_idxs: The indices of the seed nodes in the target nodeset.
      verbose: The verbosity level.

    Returns:
      An array of probabilities for each seed node.
    """
    prediction_list = []
    for batch in self.predict_batch(
        graph, seed_node_idxs, verbose=verbose, input_features_only=True
    ):
      prediction_list.append(batch.predictions)
    return np.concatenate(prediction_list, axis=0)

  # TODO(gbm): Add batch predict to the API.
  def predict_batch(
      self,
      graph: in_memory_graph.InMemoryGraph,
      seed_node_idxs: common.SeedNodeIdxs,
      input_features_only: bool,
      verbose: int = 2,
      *,
      to_host: bool = True,
  ) -> Iterator[BatchPrediction]:
    """Generate batches of predictions.

    Args:
      graph: The input graph.
      seed_node_idxs: The indices of the seed nodes in the target nodeset.
      input_features_only: If true, only the input features (i.e., not the
        label) are sampled.
      verbose: The verbosity level.
      to_host: If true, the predictions are transferred to the host as
        `np.ndarray`. If false, the predictions stay on device as `jax.Array`,
        which avoids a device synchronization at each batch.

    Yields:
      The batches of predictions.
    """
    sampling_plan = self._data.sampling_plan
    if sampling_plan is None:
      raise ValueError(textwrap.dedent("""\
              This model cannot generate predictions from a full graph because
              it does not have a sampling plan: it was trained on already
              sampled graph samples without a `sampling_plan` argument, so the
              sampling plan used to generate those samples is unknown. Instead,
              generate predictions from graph samples (e.g.
              `model.predict_on_graph_sample_batch`, `model.evaluate_generator`,
              `model.to_tensorflow_function`), or re-train the model with the
              `sampling_plan` argument.
              """))

    live = self._get_live()

    if input_features_only:
      sampler_schema = schema_to_input_feature_schema(
          self._data.schema, self._data.task
      )
    else:
      sampler_schema = self._data.schema

    batch_size = self._data.hparams.batch_size
    task = self._data.task

    # If enough nodes are sampled, normalizing the features of the entire graph
    # once is cheaper than normalizing the features of each sampled sub-graph.
    # The normalized features are then directly gathered by the sampler.
    is_normalized = self._should_cache_normalized_features(
        graph, len(seed_node_idxs)
    )
    if is_normalized:
      graph = live.normalizer.normalize_numpy(graph)
      sampler_schema = live.normalizer.output_schema()
      if input_features_only:
        label_features = sampler_schema.node_sets[task.target_nodeset].features
        for name in live.normalizer.get_normalized_feature_names(
            task.target_nodeset, task.target_column
        ):
          del label_features[name]

    normalized_batches = inference.generate_normalized_batches(
        graph,
        seed_node_idxs,
        schema=self._data.schema,
        sampling_plan=sampling_plan,
        padding=self._data.padding,
        normalizer=live.normalizer,
        target_nodeset=task.target_nodeset,
        batch_size=batch_size,
        temporal_sampling=self._data.temporal_sampling,
        sampler_schema=sampler_schema,
        verbose=verbose,
        is_normalized=is_normalized,
    )
    # The next batches are sampled and normalized while the model is applied.
    for normalized_batch in util.prefetch_iterator(
        normalized_batches, prefetch_size=_PREDICT_PREFETCH_SIZE
    ):
      yield self._predict_normalized_batch(
          live, normalized_batch, to_host=to_host
      )

  def _should_cache_normalized_features(
      self, graph: in_memory_graph.InMemoryGraph, num_seed_nodes: int
  ) -> bool:
    """Tests if the features of the entire graph should be normalized at once."""
    live = self._get_live()
    schema = self._data.schema
    if (
        self._data.temporal_sampling
        or "seed_timestamps" in live.normalizer.accepted_kwargs
        or temporal_util.schema_has_timeseries_features(schema)
        or any(not nodeset.num_nodes for nodeset in graph.node_sets.values())
    ):
      return False
    num_graph_nodes = sum(n.num_nodes or 0 for n in graph.node_sets.values())
    # The padding is an upper bound of the number of nodes in a batch.
    num_padded_nodes = sum(
        n.num_nodes or 0 for n in self._data.padding.node_sets.values()
    )
    num_sampled_nodes = (
        num_seed_nodes * num_padded_nodes / self._data.hparams.batch_size
    )
    return num_sampled_nodes >= num_graph_nodes

  def _predict_normalized_batch(
      self,
      live: ModelLiveResource,
      normalized_batch: inference.NormalizedBatch,
      *,
      to_host: bool = True,
  ) -> BatchPrediction:
    """Predicts on an already merged and normalized (sub-)batch.

    Args:
      live: The live model resources.
      normalized_batch: The merged and normalized batch.
      to_host: If true, the predictions are transferred to the host as
        `np.ndarray` (which blocks until the device computation is done). If
        false, the predictions stay on device as `jax.Array`.

    Returns:
      The batch prediction.
    """
    predictions = live.apply_core_model(normalized_batch.jax_batch)
    if to_host:
      predictions = np.asarray(predictions)
    return BatchPrediction(
        batch_seed_node_idxs=normalized_batch.batch_seed_node_idxs,
        normalized_merged_graph=normalized_batch.normalized_merged_graph,
        merged_seed_node_idxs=normalized_batch.merged_seed_node_idxs,
        predictions=predictions,
    )

  def predict_on_graph_sample_batch(
      self,
      graph_samples: list[in_memory_graph.InMemoryGraph],
  ) -> np.ndarray:
    """Predicts the target column values for a batch of graph samples.

    Args:
      graph_samples: A list of graph samples to predict on.

    Returns:
      An array of probabilities for each graph sample.
    """
    live = self._get_live()
    schema = schema_to_input_feature_schema(self._data.schema, self._data.task)

    merged_graph, merge_offsets = merge_lib.GraphMerger(
        schema=schema,
        padding=self._data.padding,
    )(graph_samples)

    normalized_batch = inference.merged_graph_to_normalized_batch(
        merged_graph,
        merge_offsets,
        schema=self._data.schema,
        normalizer=live.normalizer,
        target_nodeset=self._data.task.target_nodeset,
    )
    return np.asarray(
        self._predict_normalized_batch(live, normalized_batch).predictions
    )

  def evaluate(
      self,
      graph: in_memory_graph.InMemoryGraph,
      num_eval_steps: int | None = 10_000,
      *,
      seed_node_idxs: common.SeedNodeIdxs | None = None,
      verbose: int = 2,
      random_seed: int | None = None,
  ) -> evaluation.Evaluation:
    """Evaluates the model on a given graph.

    Usage example:

      ```python
      model = train_node_model(train_graph, ...)
      evaluation = model.evaluate(test_graph)
      # Show the evaluation in a colab
      evaluation
      # Access the evaluation values
      print(evaluation.accuracy)
      ```

    Args:
      graph: The input graph data. This graph should contain the true labels for
        the target column.
      num_eval_steps: Maximum number of evaluation batches to run. If None,
        evaluation is run on all specified `seed_node_idxs`.
      seed_node_idxs: Indices of the seed nodes within the target nodeset to
        evaluate. If None, all nodes in the target nodeset are used.
      verbose: The verbosity level. Higher values provide more detailed logging
        output during the evaluation process.
      random_seed: Random seed to select the seed nodes is num_train_steps is
        less than the number of nodes in the graph. If None, use a new random
        seed each time.

    Returns:
      An `evaluation.Evaluation` object containing the evaluation results,
      specifically the accuracy and the number of examples evaluated.
    """

    target_nodeset = self._data.task.target_nodeset
    num_nodes = graph.node_sets[target_nodeset].num_nodes

    if seed_node_idxs is None:
      # Consider all the seed nodes
      seed_node_idxs = np.arange(num_nodes)  # pyrefly: ignore[no-matching-overload]

    if num_eval_steps is not None and num_eval_steps < num_nodes:  # pyrefly: ignore[unsupported-operation]
      # Sub-select seed nodes.
      rng = np.random.default_rng(random_seed)
      seed_node_idxs = rng.choice(
          seed_node_idxs, size=num_eval_steps, replace=False
      )

    num_examples = len(seed_node_idxs)
    if verbose >= 1:
      log.info("Evaluating model on %d samples", num_examples)

    return self._evaluate_on_batch_iterator(
        self.predict_batch(
            graph,
            seed_node_idxs,
            input_features_only=False,
            verbose=verbose,
            to_host=False,
        )
    )

  def analyze(
      self,
      graph: in_memory_graph.InMemoryGraph,
      *,
      max_duration: float | None = 60.0,
  ) -> analyze_lib.ModelAnalysis:
    """Analyzes the model to help with its interpretation.

    The analysis computes the importance of each component of the graph, i.e.
    how much the model relies on it for its predictions:

    - Importance of the node features and edge features.
    - Importance of the edgesets (i.e., the graph topology).
    - Importance of the nodesets.
    - Gradient saliency of the numerical features.

    Usage example:

      ```python
      model = dgf.learning.train_node_model(train_graph, ...)
      analysis = model.analyze(test_graph)
      analysis  # Interactive report in a notebook.
      print(analysis)  # Text report.
      ```

    Args:
      graph: Graph with the ground truth labels to evaluate the model on.
      max_duration: Maximum duration of the analysis in seconds. Increase to
        improve the precision of the importances. If None, no time limit.

    Returns:
      The analysis of the model.
    """
    options = analyze_lib.AnalyzeOptions(max_duration=max_duration)
    return analyze_lib.analyze_node_model(self, graph, options=options)

  def evaluate_generator(
      self,
      graph_samples: Iterator[in_memory_graph.InMemoryGraph],
      num_eval_steps: int | None = 10_000,
      *,
      verbose: int = 2,
  ) -> evaluation.Evaluation:
    """Evaluates the model on an iterator of graph samples.

    This is useful for running evaluation on datasets where the graph samples
    are generated on the fly, for instance when reading them from a large bagz
    file, avoiding the need to load the whole dataset into memory.

    Usage example:

    ```python
      iterator = dataset.sample_generator.single_iterator()
      evaluation = model.evaluate_generator(iterator, num_eval_steps=1000)
    ```

    Args:
      graph_samples: An iterator of graph samples (e.g. `InMemoryGraph`).
      num_eval_steps: Maximum number of evaluation steps to run. If None,
        evaluate all samples in the iterator.
      verbose: The verbosity level. Higher values provide more detailed logging
        output during the evaluation process.

    Returns:
      An `evaluation.Evaluation` object containing the evaluation results,
      specifically the accuracy/RMSE and the number of examples evaluated.
    """
    if num_eval_steps is not None:
      graph_samples = itertools.islice(graph_samples, num_eval_steps)

    batch_size = max(1, self._data.hparams.batch_size // 2)

    def batch_prediction_generator() -> Iterator[BatchPrediction]:
      live = self._get_live()
      normalized_batches = inference.generate_normalized_batches_from_samples(
          graph_samples,
          schema=self._data.schema,
          padding=self._data.padding,
          normalizer=live.normalizer,
          target_nodeset=self._data.task.target_nodeset,
          batch_size=batch_size,
          num_steps=num_eval_steps,
          verbose=verbose,
      )
      for normalized_batch in normalized_batches:
        yield self._predict_normalized_batch(
            live, normalized_batch, to_host=False
        )

    if verbose >= 1:
      log.info("Evaluating model on generator")
    return self._evaluate_on_batch_iterator(batch_prediction_generator())

  def _evaluate_on_batch_iterator(
      self,
      batch_iterator: Iterator[BatchPrediction],
  ) -> evaluation.Evaluation:
    """Evaluates the model on batches of predictions.

    The metrics are accumulated on device, and only transferred to the host at
    the end, to avoid a device synchronization at each batch. The only
    exception is the classification per-class metrics (e.g., ROC AUC), which
    are accumulated on the host.

    Args:
      batch_iterator: The batches of predictions, ideally with predictions on
        device.

    Returns:
      The evaluation.
    """
    num_good_predictions = jnp.array(0)
    total_squared_error = jnp.array(0.0)
    sum_labels = jnp.array(0.0)
    sum_squared_labels = jnp.array(0.0)
    num_examples = 0
    num_elements = 0

    target_nodeset = self._data.task.target_nodeset
    target_normalized_feature = self._normalized_target_columns()

    accumulator = None
    if self._data.task.task_type == TaskType.NODE_CLASSIFICATION:
      num_classes = self.num_label_classes()
      accumulator = evaluation.ClassificationEvaluationAccumulator(num_classes)

    # Each batch is processed only after the next batch is fetched, i.e., after
    # the model computation of the next batch is dispatched to the device. This
    # way, the host-side work on the current batch (notably the per-class
    # metrics accumulator) overlaps with the device computation of the next
    # batch, instead of leaving the device idle.
    for batch in util.lookahead_iterator(batch_iterator):
      labels = batch.normalized_merged_graph.node_sets[target_nodeset].features[
          target_normalized_feature
      ][batch.merged_seed_node_idxs]

      if self._data.task.task_type == TaskType.NODE_REGRESSION:
        total_squared_error += jnp.sum(jnp.square(labels - batch.predictions))
        sum_labels += jnp.sum(labels)
        sum_squared_labels += jnp.sum(jnp.square(labels))
        num_examples += labels.shape[0]
        num_elements += labels.size
      else:
        predictions = jnp.argmax(batch.predictions, axis=-1)
        num_good_predictions += jnp.sum(labels == predictions)
        assert accumulator is not None
        # Note: The per-class metrics accumulator runs on the host, so this
        # transfers the predictions to the host. Thanks to the lookahead above,
        # the device computes the next batch in the meantime.
        accumulator.add_predictions(
            np.asarray(batch.predictions, dtype=np.float32),
            np.asarray(labels, dtype=np.int32),
        )
        num_examples += labels.shape[0]

    if self._data.task.task_type == TaskType.NODE_REGRESSION:
      mean_labels = sum_labels / num_elements
      total_sum_squares = sum_squared_labels - num_elements * jnp.square(
          mean_labels
      )
      r2 = 1 - total_squared_error / total_sum_squares

      return evaluation.Evaluation(
          rmse=np.sqrt(total_squared_error.item() / num_elements),
          r2=r2.item(),
          num_examples=num_examples,
      )
    else:
      eval_obj = evaluation.Evaluation(
          accuracy=num_good_predictions.item() / num_examples,
          num_examples=num_examples,
      )
      assert accumulator is not None
      accumulator.populate_evaluation(eval_obj)
      return eval_obj

  def _normalized_target_columns(self):
    """Returns the name of the normalized target column."""
    # TODO(gbm): Don't reconstruct the normalizer.
    normalizer = self._data.normalizer_config.make()
    normalized_target_columns = normalizer.get_normalized_feature_names(
        self._data.task.target_nodeset, self._data.task.target_column
    )
    assert len(normalized_target_columns) == 1
    return normalized_target_columns[0]

  def _get_live(self) -> ModelLiveResource:
    """Initializes model resources if they are not already live.

    Returns:
      The ModelLiveResource instance.
    """
    if self._live is None:
      normalizer = self._data.normalizer_config.make()
      normalized_schema = normalizer.output_schema()
      normalized_input_feature_schema = (
          normalized_schema_to_normalized_input_feature_schema(
              schema=normalized_schema, task=self._data.task
          )
      )
      core_model = self._data.core_model_config.make(
          normalized_input_feature_schema
      )

      @jax.jit
      def apply_core_model(batch: Batch):
        # TODO(gbm): Add softmax.
        assert core_model is not None
        logits = core_model.apply(
            self._data.model_params, batch, training=False  # pyrefly: ignore[bad-argument-type]
        )
        if self._data.task.task_type == TaskType.NODE_REGRESSION:
          head_config = self._data.core_model_config.head
          assert isinstance(head_config, regression_lib.RegressionHeadConfig)
          return head_config.logits_to_predictions(logits)  # pyrefly: ignore[bad-argument-type]
        else:
          return classification_lib.ClassificationHead.logits_to_probability(
              logits  # pyrefly: ignore[bad-argument-type]
          )

      self._live = ModelLiveResource(
          core_model=core_model,
          apply_core_model=apply_core_model,
          normalized_schema=normalized_schema,
          normalizer=normalizer,
      )

    return self._live

  def to_tensorflow_function(
      self,
      *,
      input_format: common.TFFunctionInputFormat | str | None = None,
      consume_tf_graph_dict: bool | None = None,
  ) -> tf.Module:
    """Exports the model as a TensorFlow function without the sampling step.

    This method creates a TensorFlow function that encapsulates the model's
    normalization and core prediction logic. It is designed to be used with
    pre-sampled subgraphs.

    Depending on `input_format`, the returned TensorFlow function takes:
      - TF_GRAPH: A `dgf.data.TFInMemoryGraph` named `graph`, and a 1D tensor of
        integers named `seed_node_idxs` specifying the indices of the nodes
        within the target nodeset for which to generate predictions.
      - TF_GRAPH_DICT: The items of a `dgf.data.TFInMemoryGraphDict` as keyword
        arguments, and `seed_node_idxs`.
      - SERIALIZED_TFGNN_GRAPHS: A 1D string tensor named `examples` containing
        serialized TF GNN Graph Samples. One prediction is generated for each
        graph sample, on the first node of the target nodeset (which is the
        seed node of the DGF graph samplers).

    Usage example:

    ```python
    # Consumes serialized TF GNN Graph Samples e.g. for a TFX BulkInferrer.
    tf_predict_fn = model.to_tensorflow_function(
        input_format="SERIALIZED_TFGNN_GRAPHS")
    predictions = tf_predict_fn(tf.constant([serialized_example]))
    ```

    Args:
      input_format: Format of the inputs consumed by the returned TensorFlow
        function. See `dgf.learning.TFFunctionInputFormat`. Defaults to
        `TF_GRAPH`.
      consume_tf_graph_dict: Deprecated. Use `input_format` instead.
        `consume_tf_graph_dict=True` is equivalent to
        `input_format="TF_GRAPH_DICT"`, and `consume_tf_graph_dict=False` is
        equivalent to `input_format="TF_GRAPH"`.

    Returns:
      A `tf.Module` with a `__call__` method returning a tensor of predictions.
    """
    input_format = common.resolve_tf_function_input_format(
        input_format, consume_tf_graph_dict
    )

    live = self._get_live()
    tf_apply_core_model = jax2tf.convert(
        live.apply_core_model,
        polymorphic_shapes=[(None, "(b,)")],
        native_serialization_platforms=["cpu", "cuda"],
    )
    schema = self.data().schema

    graph_spec = io_tf_lib.schema_to_spec(schema)
    graph_dict_spec = io_tf_lib.schema_to_dict_spec(schema)
    seed_node_idxs_spec = tf.TensorSpec(
        shape=[None], dtype=tf.int32, name="seed_node_idxs"
    )

    class TfPredictWrapperBase(tf.Module):

      def __init__(
          self,
          name: str,
          tf_apply_core_model,
          normalizer: normalize_lib.GraphNormalizer,
          schema: schema_lib.GraphSchema,
          padding: padding_data_lib.Padding,
      ):
        super().__init__(name=name)
        self._tf_apply_core_model = tf_apply_core_model
        self._normalizer = normalizer
        self._schema = schema
        self._padding = padding
        self._normalizer_tf_resources = normalizer.tensorflow_resources()

      def _predict(
          self,
          graph: tf_in_memory_graph_lib.TFInMemoryGraph,
          seed_node_idxs: tf.Tensor,
      ) -> tf.Tensor:
        padded_graph = merge_lib.pad_graph_tensorflow(
            graph, self._schema, self._padding
        )
        normalized_graph = self._normalizer.normalize_tensorflow(padded_graph)
        # Note: We don't cast (cast_arrays=False) as to use jax2tf.
        jax_graph = jax_lib.graph_to_jax_graph(
            normalized_graph, cast_arrays=False
        )
        return self._tf_apply_core_model((jax_graph, seed_node_idxs))

    class TfPredictWrapperTFGraph(TfPredictWrapperBase):

      def __init__(self, tf_apply_core_model, normalizer, schema, padding):
        super().__init__(
            "TfPredictWrapperTFGraph",
            tf_apply_core_model,
            normalizer,
            schema,
            padding,
        )

      @tf.function(
          autograph=False,
          input_signature=[graph_spec, seed_node_idxs_spec],
      )
      def __call__(
          self,
          graph: tf_in_memory_graph_lib.TFInMemoryGraph,
          seed_node_idxs: tf.Tensor,
      ) -> tf.Tensor:
        return self._predict(graph, seed_node_idxs)

    class TfPredictWrapperTFGraphDict(TfPredictWrapperBase):

      def __init__(self, tf_apply_core_model, normalizer, schema, padding):
        super().__init__(
            "TfPredictWrapperTFGraphDict",
            tf_apply_core_model,
            normalizer,
            schema,
            padding,
        )

      @tf.function(autograph=False)
      def __call__(
          self,
          **kwargs,
      ) -> tf.Tensor:
        seed_node_idxs = kwargs.pop("seed_node_idxs")
        graph = io_tf_lib.tf_graph_dict_to_tf_graph(kwargs)
        return self._predict(graph, seed_node_idxs)

    class TfPredictWrapperSerializedTFGNNGraphs(TfPredictWrapperBase):
      """Consumes serialized TF GNN Graph Samples."""

      def __init__(self, tf_apply_core_model, normalizer, schema, padding):
        super().__init__(
            "TfPredictWrapperSerializedTFGNNGraphs",
            tf_apply_core_model,
            normalizer,
            schema,
            padding,
        )
        self._parsing_spec = io_tf_lib.schema_to_tfgnn_graph_parsing_spec(
            schema
        )

      @tf.function(
          autograph=False,
          input_signature=[
              tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
          ],
      )
      def __call__(self, examples: tf.Tensor) -> tf.Tensor:
        def predict_one_example(serialized_example: tf.Tensor) -> tf.Tensor:
          graph = io_tf_lib.tfgnn_graph_dict_to_tf_graph(
              tf.io.parse_single_example(
                  serialized_example, self._parsing_spec
              ),
              self._schema,
          )
          # Note: By convention, the seed node is the first node of the target
          # nodeset of the graph sample.
          predictions = self._predict(graph, tf.constant([0], dtype=tf.int32))
          return predictions[0]

        return tf.map_fn(
            predict_one_example,
            examples,
            fn_output_signature=tf.TensorSpec(shape=None, dtype=tf.float32),
        )

    wrapper_class = {
        common.TFFunctionInputFormat.TF_GRAPH: TfPredictWrapperTFGraph,
        common.TFFunctionInputFormat.TF_GRAPH_DICT: TfPredictWrapperTFGraphDict,
        common.TFFunctionInputFormat.SERIALIZED_TFGNN_GRAPHS: (
            TfPredictWrapperSerializedTFGNNGraphs
        ),
    }[input_format]
    wrapper = wrapper_class(
        tf_apply_core_model,
        live.normalizer,
        self.data().schema,
        self.data().padding,
    )

    if input_format == common.TFFunctionInputFormat.TF_GRAPH_DICT:
      spec_kwargs = dict(graph_dict_spec)
      spec_kwargs["seed_node_idxs"] = seed_node_idxs_spec
      wrapper.__call__ = wrapper.__call__.get_concrete_function(**spec_kwargs)

    return wrapper


def normalized_schema_to_normalized_input_feature_schema(
    schema: schema_lib.GraphSchema,
    task: NodePredictionTask,
) -> schema_lib.GraphSchema:
  """Creates an input schema for inference from a training/evaluation schema.

  Removes the target label column specified in `task` from the features of the
  `target_nodeset` in the schema to prepare it for inference.

  Args:
    schema: The normalized graph schema used during training or evaluation.
    task: The NodePredictionTask object containing the target nodeset and
      column.

  Returns:
    A new schema instance suitable for model inference.
  """
  schema = copy.deepcopy(schema)
  del schema.node_sets[task.target_nodeset].features[
      task.normalized_target_column  # pyrefly: ignore[unsupported-operation]
  ]
  return schema


def schema_to_input_feature_schema(
    schema: schema_lib.GraphSchema,
    task: NodePredictionTask,
) -> schema_lib.GraphSchema:
  """Create a schema that does not require the target column."""
  schema = copy.deepcopy(schema)
  del schema.node_sets[task.target_nodeset].features[task.target_column]
  return schema

