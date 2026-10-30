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

"""The logic to train an unsupervised node embedding model.

Takes a training dataset and return a model (node_embedding_model).
"""

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import itertools
import os
import textwrap
import time
from typing import Literal
from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import jax as jax_lib
from dgf.src.learning import early_stopping_monitor
from dgf.src.learning.jax import flax_train
from dgf.src.learning.jax.layers import graphmae
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import dataset
from dgf.src.learning.ten_lines import node_embedding_core_model
from dgf.src.learning.ten_lines import node_embedding_model
from dgf.src.learning.ten_lines import node_prediction_dataset
from dgf.src.learning.ten_lines import node_prediction_model
from dgf.src.learning.ten_lines import training
from dgf.src.plot import network as network_lib
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import normalize_text as normalize_text_lib
from dgf.src.util import filesystem as fs
from dgf.src.util import log
from dgf.src.util import util

import jax
import jax.numpy as jnp
import numpy as np

Batch = node_embedding_core_model.Batch
HParam = node_embedding_model.HParam
NodeEmbeddingModel = node_embedding_model.NodeEmbeddingModel
ModelData = node_embedding_model.ModelData
TrainingStats = node_prediction_model.TrainingStats
CoreModelConfig = node_embedding_core_model.CoreModelConfig

# Features to exclude from training: either a mapping from nodeset name to
# feature names, or feature names of the target nodeset.
ExcludeFeatures = (
    Mapping[str, Sequence[str] | set[str]] | Sequence[str] | set[str]
)

# Range of the automatic batch size. See `common.select_batch_size`. The batch
# size is larger than for supervised node prediction, since the VICReg
# variance and covariance regularization is estimated on each batch.
_AUTO_BATCH_SIZE_RANGE = (128, 256)
# Default learning rate divided by batch size (1e-3 for a batch of 256).
_LEARNING_RATE_PER_EXAMPLE = 1e-3 / 256


def create_core_model_config(
    hparams: HParam,
    target_nodeset: str,
    max_timeseries_len: int,
) -> CoreModelConfig:
  """Creates the FLAX core model config from the hyper-parameters.

  This function is intended to be used during training. The resulting
  `CoreModelConfig` is serialized with the model, so this function is not called
  when the model is reconstructed (e.g., after deserialization).

  Args:
    hparams: Hyperparameters for the model.
    target_nodeset: The nodeset whose node embeddings are learned.
    max_timeseries_len: Maximum sequence length from the sampling plan.

  Returns:
    The core model config.
  """
  encoder_conv = common.build_gnn_config(hparams)
  if not isinstance(
      encoder_conv, hetero_gnn.HeterogeneousGraphConvolutionConfig
  ):
    raise ValueError(
        f"Architecture {hparams.architecture} is not supported by the node"
        " embedding model."
    )
  return CoreModelConfig(
      embed_graph=preprocess.EmbedGraphConfig(
          feature_embedder=preprocess.EmbedFeatureSetConfig(
              timeseries_encoder=common.build_timeseries_encoder_config(
                  hparams,
                  max_timeseries_len=max_timeseries_len,
              ),
          )
      ),
      pre_mlp=standard.ingest_feature(hparams.node_embedding_dim),
      graphmae=graphmae.GraphMAEConfig(
          encoder_conv=encoder_conv,
          num_encoder_layers=hparams.num_layers,
          decoder_conv=dataclasses.replace(
              encoder_conv, embedding_feature="latent_embedding"
          ),
          num_decoder_layers=hparams.num_decoder_layers,
          use_encoder_to_decoder=hparams.use_encoder_to_decoder,
          norm=hparams.norm,
          target_nodeset=target_nodeset,
          mask_rate=hparams.mask_rate,
          drop_edge_rate=hparams.drop_edge_rate,
          replace_rate=hparams.replace_rate,
          scaled_cosine_error_alpha=hparams.scaled_cosine_error_alpha,
          variance_regularization_weight=hparams.variance_regularization_weight,
          covariance_regularization_weight=(
              hparams.covariance_regularization_weight
          ),
          stop_gradient_target=hparams.stop_gradient_target,
          regularize_input_embeddings=hparams.regularize_input_embeddings,
      ),
  )


def _nodeset_has_input_features(nodeset_schema: schema_lib.NodeSchema) -> bool:
  """Returns True if `nodeset_schema` has at least one model input feature."""
  return any(
      feature_schema.semantic
      not in (
          schema_lib.FeatureSemantic.PRIMARY_ID,
          schema_lib.FeatureSemantic.UNKNOWN,
      )
      for feature_schema in nodeset_schema.features.values()
  )


def _normalize_exclude_features(
    exclude_features: ExcludeFeatures | None,
    schema: schema_lib.GraphSchema,
    target_nodeset: str,
) -> dict[str, list[str]]:
  """Normalizes `exclude_features` into a `{nodeset: [feature, ...]}` dict."""
  if exclude_features is None:
    return {}
  if isinstance(exclude_features, str):
    raise ValueError(
        "`exclude_features` must be a mapping from nodeset name to feature"
        " names or a sequence of feature names, not a bare string."
    )
  if not isinstance(exclude_features, Mapping):
    exclude_features = {target_nodeset: exclude_features}
  normalized = {}
  for nodeset_name, feature_names in exclude_features.items():
    if nodeset_name not in schema.node_sets:
      raise ValueError(
          f"Nodeset {nodeset_name!r} in `exclude_features` not found in"
          f" schema. Available nodesets: {list(schema.node_sets.keys())}"
      )
    if isinstance(feature_names, str):
      feature_names = [feature_names]
    available_features = schema.node_sets[nodeset_name].features
    for feature_name in feature_names:
      if feature_name not in available_features:
        raise ValueError(
            f"Feature {feature_name!r} in `exclude_features` not found in"
            f" nodeset {nodeset_name!r}. Available features:"
            f" {list(available_features.keys())}"
        )
    if feature_names:
      normalized[nodeset_name] = sorted(set(feature_names))
  return normalized


def _apply_exclude_features(
    schema: schema_lib.GraphSchema,
    excluded_features: Mapping[str, Sequence[str]],
) -> schema_lib.GraphSchema:
  """Returns a copy of `schema` without `excluded_features`."""
  new_schema = copy.deepcopy(schema)
  for nodeset_name, feature_names in excluded_features.items():
    features = new_schema.node_sets[nodeset_name].features
    for feature_name in feature_names:
      del features[feature_name]
  return new_schema


def train_node_embedding_model(
    graph: common.Graph,
    schema: schema_lib.GraphSchema,
    target_nodeset: str | None = None,
    *,
    exclude_features: ExcludeFeatures | None = None,
    max_training_time_seconds: int | None = None,
    work_dir: str | None = None,
    checkpoint_every_n_minutes: float | None = 10.0,
    verbose: int = 2,
    validation_ratio: float = 0.1,
    train_seed_nodes: common.SeedNodeIdxs | None = None,
    valid_seed_nodes: common.SeedNodeIdxs | None = None,
    num_train_steps: int | None = 10_000,
    num_valid_steps: int | None = 1_000,
    valid_every_n_steps: int = 1000,
    graph_format: dataset.GraphFormat | str = dataset.GraphFormat.AUTO,
    valid_graph: common.Graph | None = None,
    num_sampling_hops: int = 2,
    sampling_width: int = 15,
    num_layers: int = 2,
    num_decoder_layers: int = 1,
    batch_size: int | None = None,
    node_embedding_dim: int = 128,
    learning_rate: float | None = None,
    weight_decay: float = 1e-4,
    warmup_steps: int | None = None,
    gradient_clip_norm: float | None = 1.0,
    mask_rate: float = 0.5,
    drop_edge_rate: float = 0.5,
    replace_rate: float = 0.0,
    scaled_cosine_error_alpha: float = 3.0,
    variance_regularization_weight: float = 0.5,
    covariance_regularization_weight: float = 0.01,
    stop_gradient_target: bool = True,
    regularize_input_embeddings: bool = True,
    use_encoder_to_decoder: bool = True,
    norm: str | None = "layer_norm",
    cache_valid_dataset: bool = True,
    time_aware: bool = False,
    message_aggregation: hetero_gnn.MessageAggregation | str = "sum",
    experimental_preprocess_core_model_config: (
        Callable[[CoreModelConfig], CoreModelConfig] | None
    ) = None,
    cache_normalized_features: bool = False,
    cache_normalized_features_device: Literal["host", "device"] = "device",
    export_metrics_to_xm: bool = False,
    architecture: common.Architecture | str = common.DEFAULT_ARCHITECTURE,
    timeseries_encoder: common.TimeseriesEncoder | str = (
        common.DEFAULT_TIMESERIES_ENCODER
    ),
    timeseries_embedding_dim: int = 64,
    sampling_plan: sampling_config_lib.SamplingPlan | None = None,
    diagnostic_dir: str | None = None,
    early_stopping: bool | int = True,
    evaluate_final_model: bool = True,
    padding_margin: float = 0.1,
    fast_compile: bool = False,
    text_embedder: str | normalize_text_lib.TextEmbedderConfig = (
        "nnlm-en-dim128"
    ),
) -> NodeEmbeddingModel:
  """Trains an unsupervised Graph Neural Network model for node embedding.

  This function trains a self-supervised masked graph auto-encoder (GraphMAE)
  to learn fixed-dimension embeddings for the nodes of a designated nodeset,
  without requiring node labels.

  Usage example:

    ```python
    from dgf.src.learning.ten_lines import node_embedding_train

    model = node_embedding_train.train_node_embedding_model(
        graph=graph,
        schema=schema,
        target_nodeset="paper",
        exclude_features={"paper": ["label"]},
    )
    embeddings = model.predict(graph)
    ```

  Args:
    graph: The input graph data structure.
    schema: The schema of the graph.
    target_nodeset: The name of the nodeset whose node embeddings are learned.
      If not provided, it's inferred if there is only one nodeset.
    exclude_features: Optional. Features not used by the model, e.g., labels of
      a downstream task, to prevent label leakage. Either a mapping from nodeset
      name to feature names (e.g. `{"paper": ["label"]}`), or feature names of
      the target nodeset (e.g. `["label"]`).
    max_training_time_seconds: Optional. The maximum duration for training in
      seconds. If None, training runs until convergence or default limits.
    work_dir: Optional. Directory to store model checkpoints. If not provided,
      the training is not checkpointed.
    checkpoint_every_n_minutes: Interval, in minutes, between periodic
      checkpoints when `work_dir` is set. Periodic checkpoints are written
      asynchronously and skipped if the previous one is still being written.
      If None, only the final checkpoint is written at the end of training.
      Ignored if `work_dir` is not set.
    verbose: The verbosity level. Higher values provide more output.
    validation_ratio: Ratio of the training dataset used to create the
      validation dataset in case no validation dataset is manually provided
      e.g., train_seed_nodes and valid_seed_nodes are provided. If set to 0, the
      entire dataset is used for training.
    train_seed_nodes: Optional. A np.ndarray or list of integer indices
      specifying the subset of nodes within the `target_nodeset` to be used for
      training. If None, the training nodes are determined based on
      `validation_ratio` and `valid_seed_nodes`.
    valid_seed_nodes: Optional. A np.ndarray or list of integer indices
      specifying the subset of nodes within the `target_nodeset` to be used for
      validation. If None, the validation set is determined based on
      `validation_ratio` and `train_seed_nodes`. If both `train_seed_nodes` and
      `valid_seed_nodes` are None, the data is split according to
      `validation_ratio`.
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
    num_layers: The number of message passing layers in the GNN encoder.
    num_decoder_layers: The number of message passing layers in the GNN decoder.
    batch_size: The batch size to use for training. If None, it is selected
      automatically: 256, or 128 for small training sets (see
      `common.select_batch_size`). Large batches make the VICReg regularization,
      estimated on each batch, more accurate.
    node_embedding_dim: The dimension of the node embeddings. Must be at least
      2.
    learning_rate: The learning rate to use for training. If None, it scales
      linearly with the batch size (1e-3 for a batch size of 256).
    weight_decay: The decoupled weight decay of the AdamW optimizer.
    warmup_steps: The number of linear learning rate warmup steps. If None, uses
      `min(200, 1 + num_train_steps // 5)`. If 0, there is no warmup.
    gradient_clip_norm: The maximum global norm of the gradients. If None, the
      gradients are not clipped.
    mask_rate: Fraction of the target nodes masked in each batch.
    drop_edge_rate: Fraction of the edges dropped in each training batch.
    replace_rate: Fraction of the masked nodes replaced with the embedding of a
      random node instead of the mask token.
    scaled_cosine_error_alpha: Scaling exponent of the scaled cosine error
      reconstruction loss.
    variance_regularization_weight: Weight of the VICReg variance
      regularization.
    covariance_regularization_weight: Weight of the VICReg covariance
      regularization.
    stop_gradient_target: Whether to stop the gradient of the reconstruction
      loss through the reconstruction target.
    regularize_input_embeddings: Whether to also apply the VICReg
      regularization to the input feature embeddings (the reconstruction
      targets), in addition to the node embeddings.
    use_encoder_to_decoder: Whether to project the encoder representations
      before the decoder.
    norm: Normalization applied after the encoder and the decoder. One of
      "layer_norm", "rms_norm", "batch_norm", or None.
    cache_valid_dataset: Whether to cache the validation dataset in memory. This
      can speed up validation if sample generation and normalization are
      time-consuming, but it will increase memory usage.
    time_aware: Temporal-aware training is not supported yet. Must be False.
    message_aggregation: How the messages received by a node are aggregated. One
      of `hetero_gnn.MessageAggregation` or its string value ("sum", "mean" or
      "symmetric").
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
    architecture: The architecture of the GNN model. Either an `Architecture`
      value or its name as a string (e.g. "hmpnn", "hgat", "graphsage", "gcn",
      "gat", "gatv2", "mtalbis").
    timeseries_encoder: The encoder used to turn the timeseries features of a
      node into a fixed sized embedding. Either a `TimeseriesEncoder` value or
      its name as a string (e.g. "cnn", "transformer").
    timeseries_embedding_dim: The dimension of the embedding computed by
      `timeseries_encoder` for each timeseries feature group of a node.
    sampling_plan: An advanced option to provide a custom plan for the sampler.
      When you use this option, the sampler ignores standard graph sampling
      arguments and validation checks e.g., num_sampling_hops, sampling_width.
      If `graph` contains already sampled graph samples (instead of a full
      graph) and `sampling_plan` is not provided, the sampling plan used to
      generate those samples is unknown: the model does not contain any sampling
      plan (i.e., `model.data().sampling_plan` is None) and it cannot generate
      embeddings from a full graph (e.g. `model.predict`, `model.evaluate`
      raise an error). In this case, it is recommended to provide the sampling
      plan used to generate the graph samples.
    diagnostic_dir: If provided, creates this directory and export to it
      artefacts that can be useful to understand and debug the model training.
    early_stopping: If True, use early stopping with default parameters
      (patience=5). If an integer, use early stopping with the given patience.
      If False, do not use early stopping.
    evaluate_final_model: If True, performs a full final evaluation of the model
      on the validation dataset after training completes. This is in addition to
      the periodic evaluations done during training.
    padding_margin: Relative margin added to observed maximum node and edge
      counts when estimating static graph padding.
    fast_compile: If True, optimizes compilation speed at the expense of slight
      runtime execution speed. Useful for fast iteration, interactive debugging,
      and unit tests. Specifically, this disables expensive XLA GEMM autotuning
      (--xla_gpu_autotune_level=0).
    text_embedder: Pre-trained frozen text embedding model used to normalize
      `TEXT` features. Can be a built-in name (e.g. `"nnlm-en-dim128"`,
      `"use-en-dim512"`, `"bert-tiny-en-dim128"`, `"bge-base-en-dim768"`), a
      model handle/path, or a `TextEmbedderConfig`.

  Returns:
    A trained `NodeEmbeddingModel` instance.
  """

  with log.capture_logs() as captured_logs:

    if fast_compile:
      common.enable_fast_compile()

    if time_aware:
      raise ValueError(
          "`time_aware=True` is not supported by `train_node_embedding_model`."
      )

    architecture = common.parse_architecture(architecture)
    timeseries_encoder = common.parse_timeseries_encoder(timeseries_encoder)
    begin_train_time = time.time()

    if diagnostic_dir is not None:
      fs.makedirs(diagnostic_dir)

    common.log_jax_backend(verbose)

    if target_nodeset is None:
      if len(schema.node_sets) == 1:
        target_nodeset = list(schema.node_sets.keys())[0]
      else:
        raise ValueError(
            "`target_nodeset` must be specified when the schema contains more"
            " than one nodeset. Found nodesets:"
            f" {list(schema.node_sets.keys())}"
        )
    elif target_nodeset not in schema.node_sets:
      raise ValueError(
          f"Target nodeset {target_nodeset!r} not found in schema. Available"
          f" nodesets: {list(schema.node_sets.keys())}"
      )

    # The model is trained on, and only uses, the non-excluded features.
    excluded_features = _normalize_exclude_features(
        exclude_features, schema, target_nodeset
    )
    schema = _apply_exclude_features(schema, excluded_features)
    if not _nodeset_has_input_features(schema.node_sets[target_nodeset]):
      raise ValueError(
          f"The target nodeset {target_nodeset!r} has no input features (after"
          " applying `exclude_features`). The node embedding model reconstructs"
          " the features of the target nodes."
      )

    # The user provides already sampled graph samples instead of a full graph.
    graph_is_pre_sampled = (
        dataset.resolve_graph_format(graph, graph_format)
        != dataset.GraphFormat.IN_MEMORY_GRAPH
    )
    # The samples were generated by a third party, and the user did not tell
    # how: The sampling plan of those samples cannot be known.
    unknown_sampling_plan = graph_is_pre_sampled and sampling_plan is None
    if unknown_sampling_plan:
      log.warning(
          "The model is trained on graph samples and no `sampling_plan` is"
          " provided. This is fine, but it is recommended to provide the"
          " `sampling_plan` used to generate those samples to make the model"
          " easier to use later (e.g. to generate embeddings from a full"
          " graph)."
      )

    if verbose >= 2:
      log.info(
          "Graph input schema:\n%s",
          print_schema_lib.print_schema(
              schema, return_output=True, header=False
          ),
      )

    batch_size, learning_rate = common.resolve_batch_size_and_learning_rate(
        batch_size,
        learning_rate,
        num_train_examples=common.estimate_num_train_seed_nodes(
            graph,
            valid_graph,
            target_nodeset,
            train_seed_nodes,
            validation_ratio,
        ),
        num_train_steps=num_train_steps,
        batch_size_range=_AUTO_BATCH_SIZE_RANGE,
        learning_rate_per_example=_LEARNING_RATE_PER_EXAMPLE,
    )

    hparams = HParam(
        max_training_time_seconds=max_training_time_seconds,
        num_train_steps=num_train_steps,
        num_sampling_hops=num_sampling_hops,
        sampling_width=sampling_width,
        batch_size=batch_size,
        node_embedding_dim=node_embedding_dim,
        learning_rate=learning_rate,
        opt_weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        gradient_clip_norm=gradient_clip_norm,
        num_layers=num_layers,
        message_aggregation=hetero_gnn.MessageAggregation(message_aggregation),
        architecture=architecture,
        timeseries_encoder=timeseries_encoder,
        timeseries_embedding_dim=timeseries_embedding_dim,
        early_stopping=early_stopping_monitor.normalize_early_stopping_config(
            early_stopping
        ),
        padding_margin=padding_margin,
        num_decoder_layers=num_decoder_layers,
        mask_rate=mask_rate,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        scaled_cosine_error_alpha=scaled_cosine_error_alpha,
        variance_regularization_weight=variance_regularization_weight,
        covariance_regularization_weight=covariance_regularization_weight,
        stop_gradient_target=stop_gradient_target,
        regularize_input_embeddings=regularize_input_embeddings,
        use_encoder_to_decoder=use_encoder_to_decoder,
        norm=norm,
    )

    with util.print_timer("Preparing dataset", verbose >= 1):
      with jax.profiler.TraceAnnotation("prepare dataset"):
        train_dataset, valid_dataset = node_prediction_dataset.prepare_datasets(
            graph=graph,
            valid_graph=valid_graph,  # pyrefly: ignore[bad-argument-type]
            schema=schema,
            target_nodeset=target_nodeset,
            random_seed=hparams.random_seed,
            batch_size=hparams.batch_size,
            num_sampling_hops=hparams.num_sampling_hops,
            sampling_width=hparams.sampling_width,
            keep_raw_features=set(),
            verbose=verbose,
            graph_format=graph_format,
            validation_ratio=validation_ratio,
            train_seed_nodes=train_seed_nodes,
            valid_seed_nodes=valid_seed_nodes,
            temporal_sampling=False,
            nodeset_timestamp_features={},
            edgeset_timestamp_features={},
            num_valid_steps=num_valid_steps,
            cache_valid_dataset=cache_valid_dataset,
            cache_normalized_features=cache_normalized_features,
            cache_normalized_features_device=cache_normalized_features_device,
            sampling_plan=sampling_plan,
            padding_margin=hparams.padding_margin,
            target_column=None,
            text_embedder=text_embedder,
        )
    normalized_schema = train_dataset.generated_schema()

    if verbose >= 2:
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

    opt = training.build_optimizer(hparams, num_train_steps)

    def process_batch(
        graph: jax_in_memory_graph.JaxInMemoryGraph,
        merge_offsets: dict[str, jnp.ndarray],
    ) -> Batch:
      return graph, merge_offsets[target_nodeset][:-1]

    core_model_config = create_core_model_config(
        hparams,
        target_nodeset=target_nodeset,
        max_timeseries_len=train_dataset.sampling_plan.max_timeseries_len,
    )
    if experimental_preprocess_core_model_config is not None:
      core_model_config = experimental_preprocess_core_model_config(
          core_model_config
      )

    if verbose >= 2:
      log.info(
          "Core model config:\n%s",
          textwrap.indent(core_model_config.architecture(), prefix="    "),
      )

    core_model = core_model_config.make(schema=normalized_schema)

    train_step = training.build_train_step(
        core_model,
        opt,
        core_model_config.compute_loss,
        rng_streams=("dropout", "masking"),
    )

    num_steps = 0

    def infinite_train_dataset_iterator():
      nonlocal num_steps
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

            num_steps += 1
            if num_steps == common.SKIPPED_SAMPLES_CHECK_STEP:
              # Fail early (i.e., not at the end of training) if too many
              # samples are skipped due to insufficient padding.
              common.check_skipped_training_samples(
                  num_skipped_samples=train_dataset.num_skipped_samples(),
                  num_generated_samples=num_steps * hparams.batch_size,
              )

            yield process_batch(sample, merge_offset)

    train_kwargs = {}
    if valid_dataset is not None:

      def valid_dataset_iterator_fn():
        valid_generator = valid_dataset.generate_jax()
        if num_valid_steps is not None:
          valid_generator = itertools.islice(valid_generator, num_valid_steps)
        for sample, merge_offset in valid_generator:
          yield process_batch(sample, merge_offset)

      if cache_valid_dataset:
        valid_dataset_iterator_fn = training.cache_valid_dataset(
            valid_dataset_iterator_fn,
            num_examples=valid_dataset.num_nodes_in_seed_nodeset(),
            batch_size=valid_dataset.batch_size,
            drop_remainder=valid_dataset.drop_remainder,
            num_valid_steps=num_valid_steps,
            verbose=verbose,
        )

      train_kwargs["valid_dataset_iterator_fn"] = valid_dataset_iterator_fn
      # The masking is fixed during validation, so that the validation losses
      # are comparable across steps.
      train_kwargs["valid_step"] = training.build_valid_step(
          core_model,
          core_model_config.compute_loss,
          rngs={"masking": jax.random.PRNGKey(hparams.random_seed)},
      )

    with util.print_timer("Training model", verbose >= 1):
      with (
          jax.profiler.TraceAnnotation("train model"),
          common.recommend_smaller_batch_size_on_oom(hparams.batch_size),
      ):
        checkpoint_dir = (
            os.path.join(work_dir, "checkpoint") if work_dir else None
        )
        train_results = flax_train.train(
            model=core_model,
            opt=opt,
            train_step=train_step,
            dataset_iterator=infinite_train_dataset_iterator(),
            num_train_steps=num_train_steps,  # pyrefly: ignore[bad-argument-type]
            rng_key=jax.random.PRNGKey(hparams.random_seed),
            working_path=checkpoint_dir,
            checkpoint_every_n_minutes=checkpoint_every_n_minutes,
            disable_progress_bar=verbose == 0,
            display_model_structure=verbose >= 3,
            train_log_every_n_steps=100,
            valid_every_n_steps=valid_every_n_steps,
            print_logs=verbose >= 2,
            max_training_time_seconds=max_training_time_seconds,
            export_metrics_to_xm=export_metrics_to_xm,
            early_stopping=hparams.early_stopping,
            **train_kwargs,
        )

    num_skipped_train_samples = train_dataset.num_skipped_samples()
    num_generated_train_samples = num_steps * hparams.batch_size
    # Check again at the end, in case training stopped before
    # `SKIPPED_SAMPLES_CHECK_STEP` (e.g., short training or early stopping).
    common.check_skipped_training_samples(
        num_skipped_samples=num_skipped_train_samples,
        num_generated_samples=num_generated_train_samples,
    )
    if num_skipped_train_samples > 0:
      log.warning(
          "Skipped %d out of %d training samples due to insufficient padding."
          " Consider increasing `padding_margin`.",
          num_skipped_train_samples,
          num_skipped_train_samples + num_generated_train_samples,
      )

    end_train_time = time.time()
    train_duration = end_train_time - begin_train_time

    model = NodeEmbeddingModel(
        data=ModelData(
            model_params=train_results.model_params,
            core_model_config=core_model_config,
            hparams=hparams,
            schema=schema,
            normalizer_config=dataset_preparator.normalizer.config,
            padding=dataset_preparator.padding,
            sampling_plan=(
                None
                if unknown_sampling_plan
                else dataset_preparator.sampling_plan
            ),
            feature_stats=dataset_preparator.feature_stats,
            excluded_features=excluded_features,
            training_stats=TrainingStats(
                num_train_seed_nodes=train_dataset.num_nodes_in_seed_nodeset(),
                num_valid_seed_nodes=valid_dataset.num_nodes_in_seed_nodeset()
                if valid_dataset is not None
                else None,
                train_duration_seconds=train_duration,
                num_skipped_train_samples=num_skipped_train_samples,
            ),
        ),
    )
    model.metadata.trainig_logs = common.TrainingLogs(
        train=train_results.train_logs,
        valid=train_results.valid_logs,
        num_train_step=train_results.num_train_step,
        stop_reason=train_results.stop_reason,
    )
    common.warn_if_not_converged(model.metadata.trainig_logs, hparams)

    if evaluate_final_model and valid_dataset is not None:
      if verbose >= 1:
        log.info("Final model evaluation")
      valid_dataset.get_live().sample_generator.set_sampler_returns_node_idxs_only(
          False
      )
      model.data().final_evaluation = model.evaluate_generator(
          valid_dataset.get_live().sample_generator.single_iterator()
      )

  model.metadata.captured_logs = captured_logs
  return model


def _diagnose_train_batch(
    graph: jax_in_memory_graph.JaxInMemoryGraph,
    offsets: dict[str, jnp.ndarray],
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
