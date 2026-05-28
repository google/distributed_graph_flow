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

"""The logic to train a link prediction model."""

import itertools
import os
import time
from typing import Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union
from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.data import in_memory_graph
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.generate import edge_neighbor_generator as edge_neighbor_generator_lib
from dgf.src.io import jax as jax_lib
from dgf.src.learning.jax import flax_train
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import evaluation
from dgf.src.learning.ten_lines import link_prediction_core_model
from dgf.src.learning.ten_lines import link_prediction_dataset
from dgf.src.learning.ten_lines import link_prediction_model
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

Batch = link_prediction_core_model.Batch
EdgeBatch = Tuple[
    in_memory_graph.InMemoryGraph, jax.Array, jax.Array, jax.Array
]
LinkPredictionModel = link_prediction_model.LinkPredictionModel
ModelData = link_prediction_model.ModelData
TrainingStats = link_prediction_model.TrainingStats
CoreModelConfig = link_prediction_core_model.CoreModelConfig
EncoderConfig = link_prediction_core_model.EncoderConfig
LinkPredictionTask = link_prediction_model.LinkPredictionTask
HParam = link_prediction_model.HParam


def compute_train_and_valid_edge_idxs(
    graph: in_memory_graph.InMemoryGraph,
    valid_graph: Optional[in_memory_graph.InMemoryGraph],
    hparams: HParam,
    task: LinkPredictionTask,
    validation_ratio: float,
    train_seed_edges: Optional[common.SeedNodeIdxs],
    valid_seed_edges: Optional[common.SeedNodeIdxs],
    max_num_valid_examples: Optional[int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
  """Computes the training and validation seed edges indices.

  Logics:
    - If `train_seed_edges` is provided, use it as training seed edges.
    - Else if `valid_seed_edges` is provided, use it as validation seed edges.
    - Else if `validation_ratio` is 0 or `valid_graph` is None, use the full
      dataset for training.
    - Else, split the dataset into training and validation seed edges.

  Args:
    graph: The input graph data structure.
    valid_graph: An optional graph for validation. If None, `graph` is used.
    hparams: Hyperparameters for the model.
    task: The link prediction task configuration.
    validation_ratio: The ratio of edges to use for validation when splitting
      from the training graph.
    train_seed_edges: Optional list of edge indices to use for training.
    valid_seed_edges: Optional list of edge indices to use for validation.
    max_num_valid_examples: Optional maximum number of validation examples to
      generate.

  Returns:
    The effective `train_seed_edges` and `valid_seed_edges` to use for
  training.
  """

  # Number of available edges.
  num_graph_seed_edges = graph.edge_sets[task.target_edgeset].num_edges()
  if valid_graph is None:
    num_valid_graph_seed_edges = num_graph_seed_edges
  else:
    num_valid_graph_seed_edges = valid_graph.edge_sets[
        task.target_edgeset
    ].num_edges()

  assert num_graph_seed_edges is not None
  assert num_valid_graph_seed_edges is not None

  if train_seed_edges is not None:
    # The user provided the seed edges. We directly return them.
    return np.array(train_seed_edges), (
        np.array(valid_seed_edges) if valid_seed_edges else None
    )

  if valid_seed_edges is not None:
    # Validation idxs but not training idxs.
    if valid_graph is None:
      raise ValueError(
          "`valid_seed_edges` can only be specified when `train_seed_edges` is"
          " also specified if not validation graph (valid_graph) is provided."
      )
    return None, np.array(valid_seed_edges)

  # The user did not provide any seed edges.

  if validation_ratio == 0 or valid_graph is not None:
    # Use the full dataset for training.
    log.info(
        "Train model on the full provided graphs. Num training seed edges:"
        " %d. Num validation seed edges: %d",
        num_graph_seed_edges,
        num_valid_graph_seed_edges,
    )
    return None, None

  # The user only provided a train graph (no valid graph) and no seed edges.
  train_seed_edge_idxs, valid_seed_edge_idxs = util.split_train_valid(
      num_graph_seed_edges,
      validation_ratio,
      hparams.random_seed,
      batch_size=hparams.batch_size,
      max_num_valid_examples=max_num_valid_examples,
  )
  log.info(
      "Num. training seed edges: %d, Num. validation seed edges: %d",
      len(train_seed_edge_idxs),
      len(valid_seed_edge_idxs),
  )
  return train_seed_edge_idxs, valid_seed_edge_idxs


def prepare_datasets(
    graph: in_memory_graph.InMemoryGraph,
    valid_graph: Optional[in_memory_graph.InMemoryGraph],
    schema: schema_lib.GraphSchema,
    hparams: HParam,
    task: LinkPredictionTask,
    verbose: int,
    validation_ratio: float,
    train_seed_edges: Optional[common.SeedNodeIdxs],
    valid_seed_edges: Optional[common.SeedNodeIdxs],
    num_valid_steps: Optional[int],
    cache_valid_dataset: bool,
    batch_size: int,
    cache_normalized_features: bool,
    cache_normalized_features_device: Literal["host", "device"],
    source_sampling_plan: Optional[sampling_config_lib.SamplingPlan],
    target_sampling_plan: Optional[sampling_config_lib.SamplingPlan],
) -> Tuple[
    link_prediction_dataset.GNNLinkDatasetPreparator,
    Optional[link_prediction_dataset.GNNLinkDatasetPreparator],
]:
  """Builds generators of batches of training and validation graph samples."""

  if not cache_valid_dataset or num_valid_steps is None:
    max_num_valid_examples = None
  else:
    max_num_valid_examples = num_valid_steps * batch_size

  train_seed_edge_idxs, valid_seed_edge_idxs = (
      compute_train_and_valid_edge_idxs(
          graph,
          valid_graph,
          hparams=hparams,
          task=task,
          validation_ratio=validation_ratio,
          train_seed_edges=train_seed_edges,
          valid_seed_edges=valid_seed_edges,
          max_num_valid_examples=max_num_valid_examples,
      )
  )

  sampling_config = sampling_config_lib.SimpleSamplingConfig(
      seed_nodeset=schema.edge_sets[task.target_edgeset].source,
      num_hops=hparams.num_sampling_hops,
      hop_width=hparams.sampling_width,
      reverse=True,
  )

  # TODO(gbm): Parametrize.
  if hparams.negative_edges == "random":
    edge_neighbor_generator = (
        edge_neighbor_generator_lib.RandomEdgeNeighborGeneratorConfig()
    )
  elif hparams.negative_edges == "random-walk":
    edge_neighbor_generator = (
        edge_neighbor_generator_lib.RandomWalkEdgeNeighborGeneratorConfig(
            num_walks_per_negative=hparams.random_walk_num_walks_per_negative
        )
    )
  else:
    raise ValueError(f"Unknown negative_edges type: {hparams.negative_edges}")

  # TODO(gbm): Should we allow for the training and validation graphs to be in
  # different format?
  common_kwargs = {
      "schema": schema,
      "sampling_config": sampling_config,
      "source_sampling_plan": source_sampling_plan,
      "target_sampling_plan": target_sampling_plan,
      "batch_size": hparams.batch_size,
      "drop_remainder": True,
      "target_edgeset": task.target_edgeset,
      "verbose_preparation": verbose >= 2,
      "num_negative_nodes": hparams.num_negative_nodes,
      "auto_normalize_config": normalize_lib.AutoNormalizeConfig(
          keep_raw_features=set(),
          ignore_features_without_stats=True,
      ),
      "skip_overflow_padding_error": True,
      "mask_seed_edge": hparams.message_passing_on_target_edgeset,
      "mask_target_edgeset": not hparams.message_passing_on_target_edgeset,
      "edge_neighbor_generator": edge_neighbor_generator,
      "cache_normalized_features": cache_normalized_features,
      "cache_normalized_features_device": cache_normalized_features_device,
  }

  train_dataset = link_prediction_dataset.GNNLinkDatasetPreparator(
      graph=graph,
      seed_edge_idxs=train_seed_edge_idxs,
      shuffle=True,
      **common_kwargs,
  )
  # Note: This stage computes the feature statistics and the padder.
  train_dataset.prepare()

  valid_dataset = link_prediction_dataset.GNNLinkDatasetPreparator(
      graph=valid_graph if valid_graph is not None else graph,
      seed_edge_idxs=valid_seed_edge_idxs,
      # We shuffle the validation iff. it is not cached.
      shuffle=not cache_valid_dataset,
      **common_kwargs,
  )
  valid_dataset.prepare_from_existing_one(train_dataset)

  common.check_number_of_seeds(
      batch_size=hparams.batch_size,
      num_training=train_dataset.num_edge_in_seed_edgeset(),
      num_validation=valid_dataset.num_edge_in_seed_edgeset(),
      key="edge",
  )

  return train_dataset, valid_dataset


def create_core_model_config(
    hparams: HParam, task: LinkPredictionTask, schema: schema_lib.GraphSchema
) -> CoreModelConfig:
  """Creates the core model architecture from the user inputs."""
  return CoreModelConfig(
      source_nodeset=schema.edge_sets[task.target_edgeset].source,
      target_nodeset=schema.edge_sets[task.target_edgeset].target,
      encoder_config=EncoderConfig(
          embbed_graph=preprocess.EmbedGraphConfig(
              feature_embedder=preprocess.EmbedFeatureSetConfig()
          ),
          pre_mlp=standard.ingest_feature(
              dims=hparams.node_embedding_dim,
          ),
          graph_conv=common.build_gnn_config(hparams),
          post_mlp=standard.identity(),
          num_layers=hparams.num_layers,
          dropout=hparams.dropout,
      ),
  )


def train_link_model(
    graph: in_memory_graph.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    target_edgeset: Optional[str] = None,
    *,
    max_training_time_seconds: Optional[int] = None,
    work_dir: Optional[str] = None,
    verbose: int = 2,
    validation_ratio: float = 0.1,
    train_seed_edges: Optional[List[int]] = None,
    valid_seed_edges: Optional[List[int]] = None,
    num_train_steps: Optional[int] = 10_000,
    num_valid_steps: Optional[int] = 1_000,
    valid_every_n_steps: int = 1000,
    valid_graph: Optional[in_memory_graph.InMemoryGraph] = None,
    num_sampling_hops: int = 2,
    sampling_width: int = 15,
    num_layers: int = 2,
    batch_size: int = 8,
    node_embedding_dim: int = 128,
    learning_rate: float = 1e-3,
    cache_valid_dataset: bool = True,
    num_negative_nodes: int = 8,
    message_passing_on_target_edgeset: bool = True,
    negative_edges: Literal["random", "random-walk"] = "random",
    random_walk_num_walks_per_negative: int = 10,
    diagnostic_dir: Optional[str] = None,
    message_pooling: str = "sum",
    experimental_preprocess_core_model_config: Optional[
        Callable[[CoreModelConfig], CoreModelConfig]
    ] = None,
    cache_normalized_features: bool = False,
    cache_normalized_features_device: Literal["host", "device"] = "device",
    export_metrics_to_xm: bool = False,
    architecture: Union[common.Architecture, str] = common.DEFAULT_ARCHITECTURE,
    source_sampling_plan: Optional[sampling_config_lib.SamplingPlan] = None,
    target_sampling_plan: Optional[sampling_config_lib.SamplingPlan] = None,
) -> LinkPredictionModel:
  """Trains a supervised Graph Neural Network model for edge prediction.

  Usage example:

    ```python
    model = dgf.learning.train_link_model(
        graph=graph,
        schema=schema,
        target_edgeset="cites",
    )
    ```

  The model is trained to predict the existence probability of an edge.

  Args:
    graph: The input graph data structure. Training training data, as well as
      the validation data if `valid_graph` is not specified.
    schema: The schema of the graph.
    target_edgeset: The name of the edgeset to predict. If not specified,
      inferred if only one exists. The model is trained as a recommender system
      where the target edgset contains edges from queries to documents.
    max_training_time_seconds: Maximum training time in seconds. If not
      provided, does not limit training time.
    work_dir: Working directory to store checkpoints.
    verbose: Verbosity level. From 0 (no logs) to 2 (lots of logs).
    validation_ratio: Ratio of edges to use for validation. Only used if the
      "valid_graph" and "valid_seed_edges" arguments are not provided.
    train_seed_edges: List of edge indices to use for training. If not provided,
      uses all the edges (modulo the extraction of validation edges).
    valid_seed_edges: List of edge indices to use for validation.
    num_train_steps: Number of training steps.
    num_valid_steps: Optional. Maximum number of validation steps. If validation
      caching is enabled (cache_valid_dataset=True), the same validation batch
      will be used each time. Otherwise, the validation batch will be sampled
      without replacement for each validation.
    valid_every_n_steps: Validate every n steps.
    valid_graph: An optional graph for validation. If None, `graph` is used.
    num_sampling_hops: Number of sampling hops.
    sampling_width: Sampling width.
    num_layers: Number of GNN layers.
    batch_size: Batch size.
    node_embedding_dim: Node embedding dimension.
    learning_rate: Learning rate.
    cache_valid_dataset: If True, the validation dataset is cached in memory.
    num_negative_nodes: Number of negative target nodes to sample for each edge.
    message_passing_on_target_edgeset: If True, message passing is allowed to
      use edges from the `target_edgeset`. Otherwise, these edges are excluded
      from the GNN message passing. Note: If
      `message_passing_on_target_edgeset=true` the seed edge used to generate
      positive samples is still masked.
    negative_edges: The strategy to use for sampling negative edges. Can be
      "random" or "random-walk".
    random_walk_num_walks_per_negative: Number of random walks to perform per
      negative sample when `negative_edges` is "random-walk".
    diagnostic_dir: If provided, creates this directory and export to it
      artefacts that can be useful to understand and debug the model training.
    experimental_preprocess_core_model_config: An optional function to
      preprocess the `CoreModelConfig` before it is used to create the core
      model.
    cache_normalized_features: If True, pre-compute the normalized features
      during the preparation stage instead of computing them on the fly in the
      generator. This option can speed up data generation/training but increases
      memory consumption.
    cache_normalized_features_device: Specifies the device ("host" for RAM or
      "device" for GPU/TPU) to store the cached normalized features. Caching
      features on the same device used for training reduces host-device
      communication, potentially speeding up training, but increases memory
      consumption on the device.
    export_metrics_to_xm: If True, export training and validation metrics to
      XManager.
    architecture: The architecture of the GNN model to use.
    source_sampling_plan: An advanced option to provide a custom plan for the
      sampler of the source node. When you use this option, the sampler ignores
      standard graph sampling arguments and validation checks e.g.,
      num_sampling_hops, sampling_width.
    target_sampling_plan: An advanced option to provide a custom plan for the
      sampler of the target node. When you use this option, the sampler ignores
      standard graph sampling arguments and validation checks e.g.,
      num_sampling_hops, sampling_width.

  Returns:
    A LinkPredictionModel instance.
  """

  # TODO(gbm): Add support for temporal aware sampling.
  # TODO(gbm): Add support for decomposable and non-decomposable decoders.
  # TODO(gbm): Add support for other type of graph inputs.

  architecture = common.parse_architecture(architecture)
  begin_train_time = time.time()

  if diagnostic_dir is not None:
    fs.makedirs(diagnostic_dir)

  if verbose >= 2:
    log.info("Using %s JAX backend", jax.default_backend())

  if target_edgeset is None:
    if len(schema.edge_sets) == 1:
      target_edgeset = list(schema.edge_sets.keys())[0]
    else:
      raise ValueError(
          "`target_edgeset` must be specified when the schema contains more"
          " than one edgeset."
      )

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
      num_negative_nodes=num_negative_nodes,
      message_passing_on_target_edgeset=message_passing_on_target_edgeset,
      negative_edges=negative_edges,
      random_walk_num_walks_per_negative=random_walk_num_walks_per_negative,
      message_pooling=message_pooling,
      architecture=architecture,
  )

  task = LinkPredictionTask(target_edgeset=target_edgeset)

  with util.print_timer("Preparing dataset", verbose >= 1):
    train_dataset, valid_dataset = prepare_datasets(
        graph=graph,
        valid_graph=valid_graph,
        schema=schema,
        hparams=hparams,
        task=task,
        verbose=verbose,
        validation_ratio=validation_ratio,
        train_seed_edges=train_seed_edges,
        valid_seed_edges=valid_seed_edges,
        num_valid_steps=num_valid_steps,
        batch_size=batch_size,
        cache_valid_dataset=cache_valid_dataset,
        cache_normalized_features=cache_normalized_features,
        cache_normalized_features_device=cache_normalized_features_device,
        source_sampling_plan=source_sampling_plan,
        target_sampling_plan=target_sampling_plan,
    )
  source_normalized_schema = (
      train_dataset.get_live().source_normalizer.output_schema()
  )
  target_normalized_schema = (
      train_dataset.get_live().target_normalizer.output_schema()
  )

  if verbose >= 2:
    # TODO(gbm): Also print the edge sets normalizer when available.
    log.info(
        "Source normalizer:\n%s",
        train_dataset.get_live().source_normalizer.config.nice_print(
            return_output=True
        ),
    )
    log.info(
        "Target normalizer:\n%s",
        train_dataset.get_live().target_normalizer.config.nice_print(
            return_output=True
        ),
    )
    log.info(
        "Source normalized graph schema:\n%s",
        print_schema_lib.print_schema(
            source_normalized_schema, return_output=True, header=False
        ),
    )
    log.info(
        "Target normalized graph schema:\n%s",
        print_schema_lib.print_schema(
            target_normalized_schema, return_output=True, header=False
        ),
    )

  # Optimizer
  warmup_steps = min(200, 1 + num_train_steps // 5)
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

  core_model_config = create_core_model_config(hparams, task, schema)
  if experimental_preprocess_core_model_config is not None:
    core_model_config = experimental_preprocess_core_model_config(
        core_model_config
    )
  core_model = core_model_config.make(
      source_schema=source_normalized_schema,
      target_schema=target_normalized_schema,
  )

  source_nodeset = schema.edge_sets[task.target_edgeset].source
  target_nodeset = schema.edge_sets[task.target_edgeset].target

  def jax_sample_to_batch(
      batch: link_prediction_dataset.GNNLinkDatasetPreparatorJaxSample,
  ) -> Batch:
    return Batch(
        positive_source_graph=batch.positive_source_graph,
        positive_target_graph=batch.positive_target_graph,
        negative_target_graph=batch.negative_target_graph,
        positive_source_offset=batch.positive_source_offsets[source_nodeset][
            :-1
        ],
        positive_target_offset=batch.positive_target_offsets[target_nodeset][
            :-1
        ],
        negative_target_offset=batch.negative_target_offsets[target_nodeset][
            :-1
        ],
    )

  def loss_fn(
      params: jaxtyping.PyTree,
      batch_stats: jaxtyping.PyTree,
      batch: Batch,
      rng_key: Optional[jax.Array],
      training: bool,
  ):

    if rng_key is not None:
      rngs = {"dropout": rng_key}
    else:
      rngs = None

    effective_params = {**params}
    if batch_stats:
      effective_params["batch_stats"] = batch_stats

    output = core_model.apply(
        effective_params,
        batch,
        training=training,
        rngs=rngs,
        mutable=["batch_stats"] if training and batch_stats else False,
    )

    if batch_stats:
      pos_logits, neg_logits, new_model_state = output
    else:
      pos_logits, neg_logits = output
      new_model_state = {}

    # Ranking metrics
    neg_logits_reshaped = neg_logits.reshape(pos_logits.shape[0], -1)
    num_positives = neg_logits_reshaped.shape[0]
    num_negatives = neg_logits_reshaped.shape[1]

    # Binary classification loss.
    loss = (
        jnp.mean(
            optax.sigmoid_binary_cross_entropy(
                pos_logits, jnp.ones_like(pos_logits)
            )
        )
        / num_positives
        + jnp.mean(
            optax.sigmoid_binary_cross_entropy(
                neg_logits, jnp.zeros_like(neg_logits)
            )
        )
        / num_negatives
    )

    # Ranking metrics
    ranking_metrics = evaluation.compute_ranking_metrics(pos_logits, neg_logits)
    aux_data = {
        "metrics": ranking_metrics,
        "model_state": new_model_state,
    }
    return loss, aux_data

  @jax.jit
  def train_step(params, opt_state, batch: Batch, rng_key):
    has_batch_stats = "batch_stats" in params
    batch_stats = params.get("batch_stats", {})
    core_params = {"params": params["params"]}

    (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        core_params, batch_stats, batch, rng_key, training=True
    )
    updates, opt_state = opt.update(grads, opt_state, core_params)
    core_params = optax.apply_updates(core_params, updates)

    params = {**core_params}
    if has_batch_stats:
      params["batch_stats"] = batch_stats
    return params, opt_state, {"loss": loss, **aux_data["metrics"]}

  def infinite_train_iterator() -> Iterator[Batch]:
    num_diagnostic_plots = 0
    while True:
      for batch in train_dataset.generate_jax():

        if diagnostic_dir is not None and num_diagnostic_plots < 5:
          _diagnose_train_batch(
              batch,
              diagnostic_dir,
              source_normalized_schema,
              target_normalized_schema,
              num_diagnostic_plots,
          )
          num_diagnostic_plots += 1

        yield jax_sample_to_batch(batch)

  train_kwargs = {}
  if valid_dataset is not None:

    def valid_dataset_iterator_fn() -> Iterator[Batch]:

      valid_generator = valid_dataset.generate_jax()
      if num_valid_steps is not None:
        valid_generator = itertools.islice(valid_generator, num_valid_steps)
      for batch in valid_generator:
        yield jax_sample_to_batch(batch)

    if cache_valid_dataset:
      with util.print_timer("Caching validation dataset", verbose >= 1):
        if verbose >= 2:
          num_examples_to_cache = valid_dataset.num_edge_in_seed_edgeset()
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

      def cached_valid_dataset_iterator_fn() -> Iterator[Batch]:
        yield from valid_dataset_list

      valid_dataset_iterator_fn = cached_valid_dataset_iterator_fn

    @jax.jit
    def valid_step(params, opt_state, batch):
      del opt_state
      loss, aux = loss_fn(params, {}, batch, None, training=False)
      return {"loss": loss, **aux["metrics"]}

    train_kwargs["valid_dataset_iterator_fn"] = valid_dataset_iterator_fn
    train_kwargs["valid_step"] = valid_step

  with util.print_timer("Training model", verbose >= 1):
    checkpoint_dir = os.path.join(work_dir, "checkpoint") if work_dir else None
    train_results = flax_train.train(
        model=core_model,
        opt=opt,
        train_step=train_step,
        dataset_iterator=infinite_train_iterator(),
        num_train_steps=num_train_steps,
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

  model = LinkPredictionModel(
      data=ModelData(
          model_params=train_results.model_params,
          core_model_config=core_model_config,
          task=task,
          hparams=hparams,
          schema=schema,
          source_normalizer_config=train_dataset.get_live().source_normalizer.config,
          target_normalizer_config=train_dataset.get_live().target_normalizer.config,
          positive_source_padding=train_dataset.get_live().positive_source_padding,
          positive_target_padding=train_dataset.get_live().positive_target_padding,
          negative_target_padding=train_dataset.get_live().negative_target_padding,
          source_feature_stats=train_dataset.get_live().source_feature_stats,
          target_feature_stats=train_dataset.get_live().target_feature_stats,
          source_sampling_plan=train_dataset.get_live().source_sampling_plan,
          target_sampling_plan=train_dataset.get_live().target_sampling_plan,
          training_stats=TrainingStats(
              num_train_seed_edges=train_dataset.num_edge_in_seed_edgeset(),
              num_valid_seed_edges=valid_dataset.num_edge_in_seed_edgeset()
              if valid_dataset is not None
              else None,
              train_duration_seconds=train_duration,
          ),
      )
  )
  model.metadata.trainig_logs = common.TrainingLogs(
      train=train_results.train_logs,
      valid=train_results.valid_logs,
  )
  return model


def _diagnose_train_batch(
    batch: link_prediction_dataset.GNNLinkDatasetPreparatorJaxSample,
    diagnostic_dir: str,
    source_normalized_schema: schema_lib.GraphSchema,
    target_normalized_schema: schema_lib.GraphSchema,
    batch_idx: int,
):
  """Exports diagnostic information about a training batch."""

  def render_graph_diagnostic(
      graph: jax_in_memory_graph.JaxInMemoryGraph,
      offsets: Dict[str, jax.Array],
      schema: schema_lib.GraphSchema,
      name: str,
  ):
    graph_np = jax_lib.jax_graph_to_graph(graph)
    offsets_np = {k: np.asarray(v) for k, v in offsets.items()}
    graph = merge_lib.remove_padding_sentinels(graph_np, schema, offsets_np)
    network_lib.plot_graph(graph, schema).render(
        os.path.join(
            diagnostic_dir,
            f"{name}_{batch_idx}",
        ),
        format="png",
        cleanup=True,
    )

  render_graph_diagnostic(
      batch.positive_source_graph,
      batch.positive_source_offsets,
      source_normalized_schema,
      "positive_source_graph",
  )
  render_graph_diagnostic(
      batch.positive_target_graph,
      batch.positive_target_offsets,
      target_normalized_schema,
      "positive_target_graph",
  )
  render_graph_diagnostic(
      batch.negative_target_graph,
      batch.negative_target_offsets,
      target_normalized_schema,
      "negative_target_graph",
  )


common.register_model(LinkPredictionModel, ModelData)
