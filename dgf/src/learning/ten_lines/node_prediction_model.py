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

import copy
import dataclasses
import enum
import os
from typing import Callable, Dict, Iterator, List, Optional

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
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import evaluation
from dgf.src.learning.ten_lines import node_prediction_core_model
from dgf.src.learning.ten_lines import report
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import log
from dgf.src.util import util
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import jaxtyping
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
import tqdm

Batch = node_prediction_core_model.Batch
CoreModel = node_prediction_core_model.CoreModel
CoreModelConfig = node_prediction_core_model.CoreModelConfig

# Filename to save the model weights as an orbax checkpoint.
FILENAME_PARAMS = "params"


@dataclasses.dataclass
class BatchPrediction:
  """Prediction yielded by "predict_batch" function.

  Attributes:
    batch_seed_node_idxs: The indices of the seed nodes in the input graph.
    normalized_merged_graph: The merged and normalized graph sample used for
      prediction.
    merged_seed_node_idxs: The indices of the seed nodes in the
      `normalized_merged_graph`.
    predictions: The model's output predictions (e.g., probabilities) for the
      batch.
  """

  batch_seed_node_idxs: np.ndarray
  normalized_merged_graph: in_memory_graph.InMemoryGraph
  merged_seed_node_idxs: np.ndarray
  predictions: np.ndarray


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
  normalized_target_column: Optional[str]
  task_type: TaskType


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class TrainingStats:
  """Statistics about the training process.

  Attributes:
    num_train_seed_nodes: The number of seed nodes used for training.
    num_valid_seed_nodes: The number of seed nodes used for validation.
    train_duration_seconds: The duration of the training in seconds.
  """

  num_train_seed_nodes: Optional[int]
  num_valid_seed_nodes: Optional[int]
  train_duration_seconds: float


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
  sampling_plan: sampling_config_lib.SamplingPlan
  feature_stats: statistics_lib.GraphFeatureStatistics
  training_stats: TrainingStats
  temporal_sampling: bool
  nodeset_timestamp_features: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  edgeset_timestamp_features: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )

  # This field is serialized / deserialized manually.
  model_params: Optional[jaxtyping.PyTree] = dataclasses.field(
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
    self._live: Optional[ModelLiveResource] = None

  @classmethod
  def name(cls) -> str:
    return "NodePrediction"

  def data(self) -> ModelData:
    return self._data

  def _internal_save(self, path: str) -> None:
    # TODO(gbm): Have the params saving logic in common.
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        os.path.join(path, FILENAME_PARAMS),
        self._data.model_params,
        #  ocp.args.StandardSave(self._data.model_params),
    )
    checkpointer.wait_until_finished()

  def _internal_load(self, path: str) -> None:
    # TODO(gbm): Have the params saving logic in common.
    checkpointer = ocp.StandardCheckpointer()
    self._data.model_params = checkpointer.restore(
        os.path.join(path, FILENAME_PARAMS)  # , ocp.args.StandardRestore(None)
    )

  def describe(self) -> util.RichDisplay:
    # TODO(gbm): Make a good rich report.

    tabs = []

    tabs.append((
        "Objective",
        f"""
<p><b>Node prediction model:</b> Predict the value of a node feature.</p>
<ul>
  <li>Target nodeset: {self._data.task.target_nodeset}</li>
  <li>Target column: {self._data.task.target_column}</li>
  <li>Number of label classes: {self.num_label_classes()}</li>
</ul>
""",
    ))

    training_stats_summary = ""
    if self.metadata.trainig_logs is not None:
      training_stats_summary = f"""
<ul>
  <li>Number of training seed nodes: {self._data.training_stats.num_train_seed_nodes}</li>
  <li>Number of validation seed nodes: {self._data.training_stats.num_valid_seed_nodes}</li>
  <li>Training duration: {util.format_duration(self._data.training_stats.train_duration_seconds)}</li>
</ul>
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

  def label_classes(self) -> List[str]:
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
          f"Could not find statistics for target {target_nodeset}.{target_column}"
      ) from e

    if not stats.dictionary:
      raise ValueError(
          f"Target column {target_nodeset}.{target_column} does not have a string dictionary. "
          "It might already be integer-encoded."
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
  ) -> Iterator[BatchPrediction]:
    """Generate batches of predictions."""
    live = self._get_live()

    if input_features_only:
      schema = schema_to_input_feature_schema(
          self._data.schema, self._data.task
      )
    else:
      schema = self._data.schema

    # TODO(gbm): Implement a better fall-back option.
    batch_size = max(1, self._data.hparams.batch_size // 2)

    sampler = in_memory_sampler_lib.create_sampler(
        graph=graph,
        plan=self._data.sampling_plan,
        schema=schema,
        batch_size=batch_size,
    )

    np_seed_node_idxs = np.asarray(seed_node_idxs)
    batch_seed_node_idxs_generator = util.batch_indices_generator(
        np_seed_node_idxs,
        batch_size=batch_size,
        drop_remainder=False,
        shuffle=False,
    )

    if verbose >= 2:
      # Create the progress bar.
      batch_seed_node_idxs_generator = tqdm.tqdm(
          batch_seed_node_idxs_generator,
          desc="Inference",
          total=util.num_batches(
              np_seed_node_idxs,
              batch_size=batch_size,
              drop_remainder=False,
          ),
      )

    def predict_sub_batch(
        sub_seed_idxs: np.ndarray,
        sub_samples: List[in_memory_graph.InMemoryGraph],
    ) -> Iterator[BatchPrediction]:
      try:
        merged_graph, merge_offsets = merge_lib.merge_graphs(
            sub_samples,
            schema,
            padding=self._data.padding,
            sentinel_offset=False,
        )
      except merge_lib.InsufficientPaddingError:
        # The graph is too large to fit in the padding. Let's split it in two
        # and try again.
        if len(sub_samples) <= 1:
          raise
        mid = len(sub_samples) // 2
        yield from predict_sub_batch(sub_seed_idxs[:mid], sub_samples[:mid])
        yield from predict_sub_batch(sub_seed_idxs[mid:], sub_samples[mid:])
        return

      normalized_merged = live.normalizer.normalize_numpy(merged_graph)
      normalized_merged_jax = jax_lib.graph_to_jax_graph(normalized_merged)

      seed_node_idxs = merge_offsets[self._data.task.target_nodeset]
      jax_seed_node_idxs = jnp.asarray(seed_node_idxs)
      probabilities = live.apply_core_model(
          (normalized_merged_jax, jax_seed_node_idxs)
      )
      predictions = np.asarray(probabilities)
      yield BatchPrediction(
          batch_seed_node_idxs=sub_seed_idxs,
          normalized_merged_graph=normalized_merged,
          merged_seed_node_idxs=seed_node_idxs,
          predictions=predictions,
      )

    for batch_seed_node_idxs in batch_seed_node_idxs_generator:

      # TODO(gbm): The sampler should consume np array directly.
      if self._data.temporal_sampling:
        target_nodeset = self._data.task.target_nodeset
        ts_feature = self._data.nodeset_timestamp_features[target_nodeset]
        timestamps = graph.node_sets[target_nodeset].features[ts_feature]
        seed_timestamps = timestamps[batch_seed_node_idxs]
        graph_samples = sampler.sample(
            batch_seed_node_idxs, seed_timestamps=seed_timestamps
        )
      else:
        graph_samples = sampler.sample(batch_seed_node_idxs)

      yield from predict_sub_batch(batch_seed_node_idxs, graph_samples)

  # TODO(gbm): Factor with predict_batch above.
  def predict_on_graph_sample_batch(
      self,
      graph_samples: List[in_memory_graph.InMemoryGraph],
  ) -> np.ndarray:
    """Predicts the target column values for a batch of graph samples.

    Args:
      graph_samples: A list of graph samples to predict on.

    Returns:
      An array of probabilities for each graph sample.
    """
    live = self._get_live()
    schema = schema_to_input_feature_schema(self._data.schema, self._data.task)

    merged_graph, merge_offsets = merge_lib.merge_graphs(
        graph_samples,
        schema,
        padding=self._data.padding,
        sentinel_offset=False,
    )
    normalized_merged = live.normalizer.normalize_numpy(merged_graph)
    normalized_merged_jax = jax_lib.graph_to_jax_graph(normalized_merged)
    seed_node_idxs = merge_offsets[self._data.task.target_nodeset]
    jax_seed_node_idxs = jnp.asarray(seed_node_idxs)
    probabilities = live.apply_core_model(
        (normalized_merged_jax, jax_seed_node_idxs)
    )
    predictions = np.asarray(probabilities)
    return predictions

  def evaluate(
      self,
      graph: in_memory_graph.InMemoryGraph,
      num_eval_steps: Optional[int] = 10_000,
      *,
      seed_node_idxs: Optional[common.SeedNodeIdxs] = None,
      verbose: int = 2,
      random_seed: Optional[int] = None,
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
    target_normalized_feature = self._normalized_target_columns()
    num_nodes = graph.node_sets[target_nodeset].num_nodes
    num_good_predictions = jnp.array(0)
    total_squared_error = jnp.array(0.0)
    sum_labels = jnp.array(0.0)
    sum_squared_labels = jnp.array(0.0)

    if seed_node_idxs is None:
      # Consider all the seed nodes
      seed_node_idxs = np.arange(num_nodes)  # pyrefly: ignore[no-matching-overload]

    if num_eval_steps is not None and num_eval_steps < num_nodes:  # pyrefly: ignore[unsupported-operation]
      # Sub-select seed nodes.
      rng = np.random.default_rng(random_seed)
      seed_node_idxs = rng.choice(  # pyrefly: ignore[bad-assignment]
          seed_node_idxs, size=num_eval_steps, replace=False
      )

    num_examples = len(seed_node_idxs)  # pyrefly: ignore[bad-argument-type]
    if verbose >= 1:
      log.info("Evaluating model on %d samples", num_examples)

    accumulator = None
    if self._data.task.task_type == TaskType.NODE_CLASSIFICATION:
      num_classes = self.num_label_classes()
      accumulator = evaluation.ClassificationEvaluationAccumulator(num_classes)

    for batch in self.predict_batch(
        graph, seed_node_idxs, verbose=verbose, input_features_only=False  # pyrefly: ignore[bad-argument-type]
    ):
      labels = batch.normalized_merged_graph.node_sets[target_nodeset].features[
          target_normalized_feature
      ][batch.merged_seed_node_idxs]

      if self._data.task.task_type == TaskType.NODE_REGRESSION:
        total_squared_error += jnp.sum(jnp.square(labels - batch.predictions))
        sum_labels += jnp.sum(labels)
        sum_squared_labels += jnp.sum(jnp.square(labels))
      else:
        predictions = jnp.argmax(batch.predictions, axis=-1)
        num_good_predictions += jnp.sum(labels == predictions)
        assert accumulator is not None
        accumulator.add_predictions(
            np.asarray(batch.predictions, dtype=np.float32),
            np.asarray(labels, dtype=np.int32),
        )

    if self._data.task.task_type == TaskType.NODE_REGRESSION:
      mean_labels = sum_labels / num_examples
      total_sum_squares = sum_squared_labels - num_examples * jnp.square(
          mean_labels
      )
      r2 = 1 - total_squared_error / total_sum_squares

      return evaluation.Evaluation(
          rmse=np.sqrt(total_squared_error.item() / num_examples),
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
          return regression_lib.RegressionHead.logits_to_predictions(logits)  # pyrefly: ignore[bad-argument-type]
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
      self, *, consume_tf_graph_dict: bool = False
  ) -> tf.Module:
    """Exports the model as a TensorFlow function without the sampling step.

    This method creates a TensorFlow function that encapsulates the model's
    normalization and core prediction logic. It is designed to be used with
    pre-sampled subgraphs.

    The returned TensorFlow function takes two arguments:
      - graph: Either a `dgf.data.TFInMemoryGraph` or
      `dgf.data.TFInMemoryGraphDict` graph (depending on consume_tf_graph_dict).
      - seed_node_idxs: A 1D tensor of integers specifying the indices of the
        nodes within the target nodeset for which to generate predictions.

    Args:
      consume_tf_graph_dict: If `True`, the returned TensorFlow function will
        expect a `dgf.data.TFInMemoryGraphDict` as the `graph` argument. This
        format is a flat dictionary, which is often easier to use with TF
        SavedModel signatures. If `False` (default), the function will expect a
        `dgf.data.TFInMemoryGraph` object. While more natural, this can lead to
        more complex manual creation of TF SavedModel signatures.

    Returns:
      A `tf.Module` with a `__call__` method that takes `graph` and
      `seed_node_idxs` and returns a tensor of predictions for the seed nodes.
    """

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

    wrapper_class = (
        TfPredictWrapperTFGraphDict
        if consume_tf_graph_dict
        else TfPredictWrapperTFGraph
    )
    wrapper = wrapper_class(
        tf_apply_core_model,
        live.normalizer,
        self.data().schema,
        self.data().padding,
    )

    if consume_tf_graph_dict:
      spec_kwargs = {}
      for spec in graph_dict_spec:
        spec_kwargs[spec.name] = spec
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
