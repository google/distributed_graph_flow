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

"""The logic to train a node prediction model.

Takes a training dataset and return a model (node_prediction_model).
"""

import itertools
import os
import textwrap
import time
from typing import Callable, Dict, Literal, Optional, Tuple, Union
from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import jax as jax_lib
from dgf.src.learning.jax import flax_train
from dgf.src.learning.jax.layers import classification as classification_lib
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers import regression as regression_lib
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import dataset
from dgf.src.learning.ten_lines import node_prediction_core_model
from dgf.src.learning.ten_lines import node_prediction_dataset
from dgf.src.learning.ten_lines import node_prediction_model
from dgf.src.plot import network as network_lib
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import filesystem as fs
from dgf.src.util import log
from dgf.src.util import util
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import optax
import tqdm

Batch = node_prediction_core_model.Batch
HParam = node_prediction_model.HParam
NodePredictionTask = node_prediction_model.NodePredictionTask
NodePredictionModel = node_prediction_model.NodePredictionModel
ModelData = node_prediction_model.ModelData
TrainingStats = node_prediction_model.TrainingStats
CoreModelConfig = node_prediction_core_model.CoreModelConfig


def create_core_model_config(
    hparams: HParam,
    task: NodePredictionTask,
    label_spec: schema_lib.FeatureSchema,
) -> CoreModelConfig:
  """Creates the FLAX core model config from the hyper-parameters.

  This function is intended to be used during training. The resulting
  `CoreModelConfig` is serialized with the model, so this function is not called
  when the model is reconstructed (e.g., after deserialization).

  Args:
    hparams: Hyperparameters for the model.
    task: The node prediction task.
    label_spec: The schema of the label feature.

  Returns:
    The core model config.
  """

  return CoreModelConfig(
      embbed_graph=preprocess.EmbedGraphConfig(
          feature_embedder=preprocess.EmbedFeatureSetConfig()
      ),
      pre_mlp=standard.ingest_feature(hparams.node_embedding_dim),
      graph_conv=common.build_gnn_config(hparams),  # pyrefly: ignore[bad-argument-type]
      post_mlp=standard.identity(),
      head=classification_lib.ClassificationHeadConfig(
          num_classes=label_spec.num_categorical_values,  # pyrefly: ignore[bad-argument-type]
      )
      if task.task_type == node_prediction_model.TaskType.NODE_CLASSIFICATION
      else regression_lib.RegressionHeadConfig(),
      target_nodeset=task.target_nodeset,
      num_layers=hparams.num_layers,
      dropout=hparams.dropout,
  )


def infer_task_type(
    schema: schema_lib.GraphSchema,
    target_nodeset: str,
    target_column: str,
) -> node_prediction_model.TaskType:
  """Infers the task type from the schema and the target column."""

  nodeset_schema = schema.node_sets[target_nodeset]
  feature_schema = nodeset_schema.features[target_column]
  if feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
    if not (
        feature_schema.format.is_integer()
        or feature_schema.format == schema_lib.FeatureFormat.BYTES
    ):
      raise ValueError(
          f"Categorical feature '{target_column}' in nodeset"
          f" '{target_nodeset}' must have an integer or bytes format, but"
          f" found {feature_schema.format.name}."
      )
    return node_prediction_model.TaskType.NODE_CLASSIFICATION
  elif feature_schema.semantic == schema_lib.FeatureSemantic.NUMERICAL:
    if not feature_schema.format.is_numerical():
      raise ValueError(
          f"Numerical feature '{target_column}' in nodeset"
          f" '{target_nodeset}' must have a float or integer format, but"
          f" found {feature_schema.format.name}."
      )
    return node_prediction_model.TaskType.NODE_REGRESSION
  else:
    raise ValueError(
        f"Cannot infer task type for column '{target_column}' in nodeset"
        f" '{target_nodeset}' with semantic {feature_schema.semantic}. "
        "Please specify the task type explicitly or use CATEGORICAL or "
        "NUMERICAL semantic."
    )


def train_node_model(
    graph: common.Graph,
    schema: schema_lib.GraphSchema,
    target_column: str,
    *,
    target_nodeset: Optional[str] = None,
    max_training_time_seconds: Optional[int] = None,
    work_dir: Optional[str] = None,
    verbose: int = 2,
    validation_ratio: float = 0.1,
    train_seed_nodes: Optional[common.SeedNodeIdxs] = None,
    valid_seed_nodes: Optional[common.SeedNodeIdxs] = None,
    num_train_steps: Optional[int] = 10_000,
    num_valid_steps: Optional[int] = 1_000,
    valid_every_n_steps: int = 1000,
    graph_format: Union[dataset.GraphFormat, str] = dataset.GraphFormat.AUTO,
    valid_graph: Optional[common.Graph] = None,
    num_sampling_hops: int = 2,
    sampling_width: int = 15,
    num_layers: int = 2,
    batch_size: int = 32,
    node_embedding_dim: int = 128,
    learning_rate: float = 1e-3,
    cache_valid_dataset: bool = True,
    time_aware: Union[bool, Dict[str, str]] = False,
    message_pooling: str = "sum",
    experimental_preprocess_core_model_config: Optional[
        Callable[[CoreModelConfig], CoreModelConfig]
    ] = None,
    cache_normalized_features: bool = False,
    cache_normalized_features_device: Literal["host", "device"] = "device",
    export_metrics_to_xm: bool = False,
    architecture: Union[common.Architecture, str] = common.DEFAULT_ARCHITECTURE,
    sampling_plan: Optional[sampling_config_lib.SamplingPlan] = None,
    diagnostic_dir: Optional[str] = None,
) -> NodePredictionModel:
  """Trains a supervised Graph Neural Network model for node-level prediction.

  This function trains a GNN to predict a specific target column within a
  designated nodeset of the provided graph.

  Args:
    graph: The input graph data structure.
    schema: The schema of the graph.
    target_column: The name of the node feature column to be predicted.
    target_nodeset: The name of the nodeset containing the target nodes. If not
      provided, it's inferred if there is only one nodeset.
    max_training_time_seconds: Optional. The maximum duration for training in
      seconds. If None, training runs until convergence or default limits.
    work_dir: Optional. Directory to store model checkpoints. If not provided,
      the training is not checkpointed.
    verbose: The verbosity level. Higher values provide more output.
    validation_ratio: Ratio of the training dataset used to create the
      validation dataset in case no validation dataset is manually provided
      e.g., train_seed_nodes and valid_seed_nodes are provided. If set to 0, the
      entire dataset is used for training, and the tree is not pruned.
    train_seed_nodes: Optional. A np.ndarray or list of integer indices
      specifying the subset of nodes within the `target_nodeset` to be used for
      training. If None, the training nodes are determined based on
      `validation_ratio` and `valid_nodes`.
    valid_seed_nodes: Optional. A np.ndarray or list of integer indices
      specifying the subset of nodes within the `target_nodeset` to be used for
      validation. If None, the validation set is determined based on
      `validation_ratio` and `train_nodes`. If both `train_nodes` and
      `valid_nodes` are None, the data is split according to `validation_ratio`.
    num_train_steps: Optional. The number of training steps to perform.
    num_valid_steps: Optional. Maximum number of validation steps. If validation
      caching is enabled (cache_valid_dataset=True), the same validation batch
      will be used each time. Otherwise, the validation batch will be sampled
      without replacement for each validation.
    valid_every_n_steps: The number of training steps between each validation.
    graph_format: Optional. The format of the input graph. If set to AUTO, the
      format is inferred automatically.
    valid_graph: Optional. The graph to use for validation. If not provided, the
      validation set is split from the training graph.
    num_sampling_hops: The number of hops to sample around the seed nodes.
    sampling_width: The number of nodes to sample at each hop.
    num_layers: The number of message passing layers in the GNN.
    batch_size: The batch size to use for training.
    node_embedding_dim: The dimension of the node embeddings.
    learning_rate: The learning rate to use for training.
    cache_valid_dataset: Whether to cache the validation dataset in memory. This
      can speed up validation if sample generation and normalization are
      time-consuming, but it will increase memory usage.
    time_aware: Enables temporal-aware training. If `False` (default), no
      temporal masking is applied. If `True`, timestamp features are inferred
      from the schema. If a dictionary, it maps nodeset/edgeset names to their
      timestamp feature. Nodesets/edgesets without specified timestamp features
      are treated as atemporal.
    message_pooling: The pooling method to use for aggregating messages.
    experimental_preprocess_core_model_config: Advanced option. An optional
      callable to modify the `CoreModelConfig` before it is used to build the
      core model.
    cache_normalized_features: If True, pre-compute the normalized features
      during the preparation stage instead of computing them on the fly in the
      generator. This option can speed up data generation/training but increases
      memory consumption.
    cache_normalized_features_device: Specifies the device ("host" for RAM or
      "device" for GPU/TPU) to store the cached normalized features. Caching
      features on the same device used for training reduces host-device
      communication, potentially speeding up training, but increases memory
      consumption on the device.
    export_metrics_to_xm: If True, metrics from the training and validation
      steps will be exported to XManager.
    architecture: The architecture of the GNN model.
    sampling_plan: An advanced option to provide a custom plan for the sampler.
      When you use this option, the sampler ignores standard graph sampling
      arguments and validation checks e.g., num_sampling_hops, sampling_width.
    diagnostic_dir: If provided, creates this directory and export to it
      artefacts that can be useful to understand and debug the model training.

  Returns:
    A trained `NodePredictionModel` instance.
  """

  # TODO(gbm): Surface / abstract / determine better default parameters.
  # TODO(gbm): Early stopping.
  # TODO(gbm): Configure the loss, task, and metrics.
  # TODO(gbm): Add support for regression.
  # TODO(gbm): Add an "auto" logic for "cache_normalized_features" and
  # "cache_normalized_features_device".

  architecture = common.parse_architecture(architecture)
  begin_train_time = time.time()

  if diagnostic_dir is not None:
    fs.makedirs(diagnostic_dir)

  if verbose >= 2:
    log.info("Using %s JAX backend", jax.default_backend())

  if target_nodeset is None:
    if len(schema.node_sets) == 1:
      target_nodeset = list(schema.node_sets.keys())[0]
    else:
      raise ValueError(
          "`target_nodeset` must be specified when the schema contains more"
          " than one nodeset. Found nodesets:"
          f" {list(schema.node_sets.keys())}"
      )

  # Note: Maybe one day, the timestamp features will be used for something else.
  temporal_sampling = bool(time_aware)
  if temporal_sampling:
    nodeset_ts_features, edgeset_ts_features = common.parse_temporal_config(
        schema=schema,
        target_nodeset=target_nodeset,
        timestamp_features=None if isinstance(time_aware, bool) else time_aware,
    )
  else:
    nodeset_ts_features = {}
    edgeset_ts_features = {}

  if verbose >= 2:
    log.info(
        "Graph input schema:\n%s",
        print_schema_lib.print_schema(schema, return_output=True, header=False),
    )

  # TODO(gbm): Parametrize.
  hparams = HParam(
      max_training_time_seconds=max_training_time_seconds,
      num_train_steps=num_train_steps,
      num_sampling_hops=num_sampling_hops,
      sampling_width=sampling_width,
      batch_size=batch_size,
      node_embedding_dim=node_embedding_dim,
      learning_rate=learning_rate,
      num_layers=num_layers,
      message_pooling=message_pooling,
      architecture=architecture,
  )

  task = NodePredictionTask(
      target_nodeset=target_nodeset,
      target_column=target_column,
      normalized_target_column=None,
      task_type=infer_task_type(schema, target_nodeset, target_column),
  )

  with util.print_timer("Preparing dataset", verbose >= 1):
    with jax.profiler.TraceAnnotation("prepare dataset"):
      train_dataset, valid_dataset = node_prediction_dataset.prepare_datasets(
          graph=graph,
          valid_graph=valid_graph,  # pyrefly: ignore[bad-argument-type]
          schema=schema,
          target_nodeset=task.target_nodeset,
          random_seed=hparams.random_seed,
          batch_size=hparams.batch_size,
          num_sampling_hops=hparams.num_sampling_hops,
          sampling_width=hparams.sampling_width,
          keep_raw_features={task.target_column}
          if task.task_type == node_prediction_model.TaskType.NODE_REGRESSION
          else set(),
          verbose=verbose,
          graph_format=graph_format,
          validation_ratio=validation_ratio,
          train_seed_nodes=train_seed_nodes,
          valid_seed_nodes=valid_seed_nodes,
          temporal_sampling=temporal_sampling,
          nodeset_timestamp_features=nodeset_ts_features,
          edgeset_timestamp_features=edgeset_ts_features,
          num_valid_steps=num_valid_steps,
          cache_valid_dataset=cache_valid_dataset,
          cache_normalized_features=cache_normalized_features,
          cache_normalized_features_device=cache_normalized_features_device,
          sampling_plan=sampling_plan,
      )
  normalized_schema = train_dataset.generated_schema()

  if verbose >= 2:
    # TODO(gbm): Also print the edge sets normalizer when available.
    log.info(
        "Normalizer:\n%s",
        train_dataset.get_live().normalizer.config.nice_print(
            return_output=True
        ),
    )
    log.info(
        "Normalized graph schema:\n%s",
        print_schema_lib.print_schema(
            normalized_schema, return_output=True, header=False
        ),
    )
  dataset_preparator = train_dataset.get_live()
  normalized_target_columns = (
      dataset_preparator.normalizer.get_normalized_feature_names(
          task.target_nodeset, task.target_column
      )
  )
  if len(normalized_target_columns) != 1:
    raise ValueError(
        "Expected exactly one normalized feature for target column"
        f" '{task.target_column}', but got {len(normalized_target_columns)}:"
        f" {normalized_target_columns}"
    )
  task.normalized_target_column = normalized_target_columns[0]
  label_spec = normalized_schema.node_sets[task.target_nodeset].features[
      task.normalized_target_column
  ]
  if task.task_type == node_prediction_model.TaskType.NODE_CLASSIFICATION:
    if label_spec.semantic != schema_lib.FeatureSemantic.CATEGORICAL:
      raise ValueError(
          f"Target column '{task.target_column}' in nodeset"
          f" '{task.target_nodeset}' must have a CATEGORICAL semantic, but"
          f" found {label_spec.semantic.name}."
      )
    if not label_spec.format.is_integer():
      raise ValueError(
          f"Target column '{task.target_column}' in nodeset"
          f" '{task.target_nodeset}' must have an integer format, but found"
          f" {label_spec.format.name}."
      )
    if label_spec.num_categorical_values is None:
      raise ValueError(
          f"Target column '{task.target_column}' in nodeset"
          f" '{task.target_nodeset}' must have `num_categorical_values` defined"
          " for classification tasks."
      )
  elif task.task_type == node_prediction_model.TaskType.NODE_REGRESSION:
    if label_spec.semantic != schema_lib.FeatureSemantic.NUMERICAL:
      raise ValueError(
          f"Target column '{task.target_column}' in nodeset"
          f" '{task.target_nodeset}' must have a NUMERICAL semantic, but"
          f" found {label_spec.semantic.name}."
      )
    if not label_spec.format.is_numerical():
      raise ValueError(
          f"Target column '{task.target_column}' in nodeset"
          f" '{task.target_nodeset}' must have a numerical format, but found"
          f" {label_spec.format.name}."
      )
  else:
    raise ValueError(f"Unsupported task type: {task.task_type}")

  warmup_steps = min(200, 1 + num_train_steps // 5)  # pyrefly: ignore[unsupported-operation]
  learning_rate_plan = optax.join_schedules(
      schedules=[
          optax.linear_schedule(
              init_value=0.0,
              end_value=hparams.learning_rate,
              transition_steps=warmup_steps,
          ),
          optax.cosine_decay_schedule(
              init_value=hparams.learning_rate,
              decay_steps=int((num_train_steps or 100_000) - warmup_steps),
          ),
      ],
      boundaries=[warmup_steps],
  )

  opt = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(
          learning_rate=learning_rate_plan,
          weight_decay=hparams.opt_weight_decay,
      ),
  )

  def process_batch(
      graph: jax_in_memory_graph.JaxInMemoryGraph,
      merge_offsets: Dict[str, jnp.ndarray],
  ) -> Batch:
    return graph, merge_offsets[task.target_nodeset][:-1]

  core_model_config = create_core_model_config(hparams, task, label_spec)
  if experimental_preprocess_core_model_config is not None:
    core_model_config = experimental_preprocess_core_model_config(
        core_model_config
    )

  if verbose >= 2:
    log.info(
        "Core model config:\n%s",
        textwrap.indent(core_model_config.architecture(), prefix="    "),
    )

  normalized_input_feature_schema = node_prediction_model.normalized_schema_to_normalized_input_feature_schema(
      schema=normalized_schema,
      task=task,
  )
  core_model = core_model_config.make(schema=normalized_input_feature_schema)

  if verbose >= 2:
    log.info(
        "Normalized input features:\n%s",
        print_schema_lib.print_schema(
            normalized_input_feature_schema, return_output=True, header=False
        ),
    )

  def loss_fn(
      core_params: jaxtyping.PyTree,
      batch_stats: jaxtyping.PyTree,
      batch: Batch,
      labels: jax.Array,
      rng_key: Optional[jax.Array],
      training: bool,
  ) -> Tuple[jax.Array, jaxtyping.PyTree]:
    if rng_key is not None:
      rngs = {"dropout": rng_key}
    else:
      rngs = None

    effective_params = {**core_params}
    if batch_stats:
      effective_params["batch_stats"] = batch_stats

    output = core_model.apply(
        effective_params,
        batch,
        training=training,
        rngs=rngs,
        mutable=["batch_stats"] if training and batch_stats else False,
    )

    logits, new_model_state = output if batch_stats else (output, {})

    if task.task_type == node_prediction_model.TaskType.NODE_CLASSIFICATION:
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)  # pyrefly: ignore[bad-argument-type]
      accuracy = jnp.argmax(logits, axis=-1) == labels  # pyrefly: ignore[bad-argument-type]
      metrics = {"accuracy": accuracy.mean()}
    elif task.task_type == node_prediction_model.TaskType.NODE_REGRESSION:
      predictions = regression_lib.RegressionHead.logits_to_predictions(logits)  # pyrefly: ignore[bad-argument-type]
      loss = optax.squared_error(predictions, labels)
      metrics = {"rmse": jnp.sqrt(loss.mean())}
    else:
      raise ValueError(f"Unknown task type: {task.task_type}")

    aux_data = {
        "metrics": metrics,
        "model_state": new_model_state,
    }
    return jnp.mean(loss), aux_data

  def train_step(params, opt_state, batch: Batch, rng_key):
    graph, seed_node_idxs = batch

    labels = graph.node_sets[task.target_nodeset].features[
        task.normalized_target_column  # pyrefly: ignore[bad-index]
    ][seed_node_idxs]

    has_batch_stats = "batch_stats" in params
    batch_stats = params.get("batch_stats", {})
    core_params = {"params": params["params"]}

    (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        core_params, batch_stats, batch, labels, rng_key, True
    )

    updates, opt_state = opt.update(grads, opt_state, core_params)
    core_params = optax.apply_updates(core_params, updates)

    params = {**core_params}  # pyrefly: ignore[invalid-argument]
    if has_batch_stats:
      params["batch_stats"] = batch_stats
    return params, opt_state, {"loss": loss, **aux_data["metrics"]}

  # Use for speed.
  jitted_train_step = jax.jit(train_step)

  def infinite_train_dataset_iterator():
    num_diagnostic_plots = 0
    while True:
      for sample, merge_offset in train_dataset.generate_jax():
        with jax.profiler.TraceAnnotation("process_batch"):

          if diagnostic_dir is not None and num_diagnostic_plots < 5:
            _diagnose_train_batch(
                sample,
                merge_offset,
                diagnostic_dir,
                normalized_schema,
                num_diagnostic_plots,
            )
            num_diagnostic_plots += 1

          yield process_batch(sample, merge_offset)

  train_kwargs = {}
  if valid_dataset is not None:

    def valid_dataset_iterator_fn():
      valid_generator = valid_dataset.generate_jax()
      if num_valid_steps is not None:
        valid_generator = itertools.islice(valid_generator, num_valid_steps)
      for sample, merge_offset in valid_generator:
        # TODO(gbm): Should we store the JAX arrays on device (this version), or
        # the Numpy array on host?
        yield process_batch(sample, merge_offset)

    if cache_valid_dataset:
      num_examples_to_cache = valid_dataset.num_nodes_in_seed_nodeset()
      num_batches_to_cache = (
          util.num_batches(
              num_examples_to_cache,
              batch_size=valid_dataset.batch_size,
              drop_remainder=valid_dataset.drop_remainder,
          )
          if num_examples_to_cache is not None
          else None
      )
      if num_valid_steps is not None:
        if num_batches_to_cache is None:
          num_batches_to_cache = num_valid_steps
        else:
          num_batches_to_cache = min(num_batches_to_cache, num_valid_steps)
      with util.print_timer("Caching validation dataset", verbose >= 1):
        if verbose >= 2:
          valid_dataset_list = list(
              tqdm.tqdm(
                  valid_dataset_iterator_fn(),
                  total=num_batches_to_cache,
                  desc="Caching validation dataset",
              )
          )
        else:
          valid_dataset_list = list(valid_dataset_iterator_fn())

      if verbose >= 1:
        log.info(
            "Number of cache validation batches: %d", len(valid_dataset_list)
        )

      def cached_valid_dataset_iterator_fn():
        yield from valid_dataset_list

      valid_dataset_iterator_fn = cached_valid_dataset_iterator_fn

    def valid_step(params, opt_state, batch: Batch):
      graph, seed_node_idxs = batch
      labels = graph.node_sets[task.target_nodeset].features[
          task.normalized_target_column  # pyrefly: ignore[bad-index]
      ][seed_node_idxs]
      loss, aux = loss_fn(params, {}, batch, labels, None, False)
      return {"loss": loss, **aux["metrics"]}

    jitted_valid_step = jax.jit(valid_step)

    train_kwargs["valid_dataset_iterator_fn"] = valid_dataset_iterator_fn
    train_kwargs["valid_step"] = jitted_valid_step

  with util.print_timer("Training model", verbose >= 1):
    with jax.profiler.TraceAnnotation("train model"):
      checkpoint_dir = (
          os.path.join(work_dir, "checkpoint") if work_dir else None
      )
      train_results = flax_train.train(
          model=core_model,
          opt=opt,
          train_step=jitted_train_step,
          dataset_iterator=infinite_train_dataset_iterator(),
          num_train_steps=num_train_steps,  # pyrefly: ignore[bad-argument-type]
          rng_key=jax.random.PRNGKey(hparams.random_seed),
          working_path=checkpoint_dir,
          disable_progress_bar=verbose == 0,
          display_model_structure=verbose >= 3,
          train_log_every_n_steps=100,
          valid_every_n_steps=valid_every_n_steps,
          print_logs=verbose >= 2,
          max_training_time_seconds=max_training_time_seconds,
          export_metrics_to_xm=export_metrics_to_xm,
          **train_kwargs,
      )

  end_train_time = time.time()
  train_duration = end_train_time - begin_train_time

  model = NodePredictionModel(
      data=ModelData(
          model_params=train_results.model_params,
          core_model_config=core_model_config,
          task=task,
          hparams=hparams,
          schema=schema,
          normalizer_config=dataset_preparator.normalizer.config,
          padding=dataset_preparator.padding,
          sampling_plan=dataset_preparator.sampling_plan,
          feature_stats=dataset_preparator.feature_stats,
          temporal_sampling=temporal_sampling,
          nodeset_timestamp_features=nodeset_ts_features,
          edgeset_timestamp_features=edgeset_ts_features,
          training_stats=TrainingStats(
              num_train_seed_nodes=train_dataset.num_nodes_in_seed_nodeset(),
              num_valid_seed_nodes=valid_dataset.num_nodes_in_seed_nodeset()
              if valid_dataset is not None
              else None,
              train_duration_seconds=train_duration,
          ),
      ),
  )
  model.metadata.trainig_logs = common.TrainingLogs(
      train=train_results.train_logs,
      valid=train_results.valid_logs,
  )

  return model


def _diagnose_train_batch(
    graph: jax_in_memory_graph.JaxInMemoryGraph,
    offsets: Dict[str, jnp.ndarray],
    diagnostic_dir: str,
    schema: schema_lib.GraphSchema,
    batch_idx: int,
):
  """Exports diagnostic information about a training batch."""
  graph_np = jax_lib.jax_graph_to_graph(graph)
  offsets_np = {k: np.asarray(v) for k, v in offsets.items()}
  graph = merge_lib.remove_padding_sentinels(graph_np, schema, offsets_np)  # pyrefly: ignore[bad-assignment]
  network_lib.plot_graph(graph, schema).render(  # pyrefly: ignore[bad-argument-type]
      os.path.join(diagnostic_dir, f"graph_{batch_idx}"),
      format="png",
      cleanup=True,
  )


common.register_model(NodePredictionModel, ModelData)
