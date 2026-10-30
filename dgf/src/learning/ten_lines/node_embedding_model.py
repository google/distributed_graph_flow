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

"""User-facing unsupervised node embedding model (`NodeEmbeddingModel`)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import dataclasses
import html
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
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import evaluation
from dgf.src.learning.ten_lines import inference
from dgf.src.learning.ten_lines import node_embedding_core_model
from dgf.src.learning.ten_lines import node_prediction_model
from dgf.src.learning.ten_lines import report
from dgf.src.learning.ten_lines import training
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

Batch = node_embedding_core_model.Batch
CoreModel = node_embedding_core_model.CoreModel
CoreModelConfig = node_embedding_core_model.CoreModelConfig

# Number of batches sampled and normalized in advance during inference.
_PREDICT_PREFETCH_SIZE = 2

# Interpretable (not optimized) metrics returned by
# `CoreModelConfig.compute_loss` and reported by `NodeEmbeddingModel.evaluate`.
_INTERPRETABLE_METRIC_KEYS = (
    "recon_cosine",
    "recon_relative_error",
    "embedding_rank_ratio",
    "target_rank_ratio",
)


@dataclasses.dataclass
class BatchEmbedding:
  """Embeddings yielded by `NodeEmbeddingModel.predict_batch`.

  Attributes:
    batch_seed_node_idxs: The indices of the seed nodes in the input graph, or
      `None` when predicting on pre-sampled subgraphs where input-graph indices
      are unknown.
    normalized_merged_graph: The merged and normalized graph sample used for
      embedding extraction.
    merged_seed_node_idxs: The indices of the seed nodes in
      `normalized_merged_graph`.
    embeddings: The encoder node embeddings of shape `[B, D]`.
  """

  batch_seed_node_idxs: np.ndarray | None
  normalized_merged_graph: in_memory_graph.InMemoryGraph
  merged_seed_node_idxs: np.ndarray
  embeddings: np.ndarray | jax.Array


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class HParam(common.HParam):
  """Hyperparameters for the `NodeEmbeddingModel`."""

  num_decoder_layers: int = 1
  mask_rate: float = 0.5
  drop_edge_rate: float = 0.5
  replace_rate: float = 0.0
  scaled_cosine_error_alpha: float = 3.0
  variance_regularization_weight: float = 0.5
  covariance_regularization_weight: float = 0.01
  stop_gradient_target: bool = True
  regularize_input_embeddings: bool = True
  use_encoder_to_decoder: bool = True
  norm: str | None = "layer_norm"


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ModelData:
  """Serializable data of `NodeEmbeddingModel`."""

  core_model_config: CoreModelConfig
  hparams: HParam
  schema: schema_lib.GraphSchema
  normalizer_config: normalize_lib.GraphNormalizerConfig
  padding: padding_data_lib.Padding
  sampling_plan: sampling_config_lib.SamplingPlan | None
  feature_stats: statistics_lib.GraphFeatureStatistics
  training_stats: node_prediction_model.TrainingStats
  excluded_features: dict[str, list[str]] = dataclasses.field(
      default_factory=dict
  )
  final_evaluation: evaluation.Evaluation | None = None

  # Serialized / deserialized via Orbax checkpointing in `_internal_save` /
  # `_internal_load`.
  model_params: jaxtyping.PyTree | None = dataclasses.field(
      default_factory=lambda: None,
      metadata=dataclasses_json.config(exclude=dataclasses_json.Exclude.ALWAYS),
      repr=False,
  )


@dataclasses.dataclass
class ModelLiveResource:
  """Live JAX/normalizer resources recomputed from `ModelData` on demand."""

  core_model: CoreModel
  apply_embed: Callable[[Batch], jnp.ndarray]
  valid_step: Callable[[Any, Any, Batch], dict[str, jax.Array]]
  normalized_schema: schema_lib.GraphSchema
  normalizer: normalize_lib.GraphNormalizer


class NodeEmbeddingModel(common.Model):
  """The user-visible returned model object for node embedding."""

  def __init__(self, data: ModelData) -> None:
    super().__init__(data)
    self._data = data
    self._live: ModelLiveResource | None = None

  @classmethod
  def name(cls) -> str:
    return "NodeEmbedding"

  def data(self) -> ModelData:
    return self._data

  @property
  def target_nodeset(self) -> str:
    return self._data.core_model_config.graphmae.target_nodeset

  @property
  def embedding_dim(self) -> int:
    return self._data.core_model_config.graphmae.get_gnn_dims()

  def _internal_save(self, path: str) -> None:
    common.save_params(self._data.model_params, path)

  def _internal_load(self, path: str) -> None:
    self._data.model_params = common.load_params(path)

  def _extract_serving_schemata(self) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns the Vertex AI `(instance, prediction)` serving schemas."""
    if not self.metadata.uuid:
      raise ValueError("Model UUID is not set.")
    instance_schema = common.schema_to_serving_signature_dict(
        schema_=self._data.schema,
        target_nodeset=self.target_nodeset,
        model_name=self.name(),
        model_uuid=self.metadata.uuid,
        source_sampling_plan=self._data.sampling_plan,
    )
    prediction_schema = {"type": "array", "items": {"type": "number"}}
    return instance_schema, prediction_schema

  def describe(self) -> util.RichDisplay:
    """Returns a rich HTML report describing the model."""
    graphmae_cfg = self._data.core_model_config.graphmae
    excluded_features = (
        "; ".join(
            f"{nodeset_name}: {', '.join(feature_names)}"
            for nodeset_name, feature_names in sorted(
                self._data.excluded_features.items()
            )
        )
        or "None"
    )
    tabs = [
        (
            "Objective",
            f"""
<p><b>Node embedding model:</b> Learn unsupervised node representations with GraphMAE.</p>
<table class="dgf-table">
  <tbody>
    <tr><td>Type</td><td>Unsupervised Node Embedding (GraphMAE)</td></tr>
    <tr><td>Target nodeset</td><td>{self.target_nodeset}</td></tr>
    <tr><td>Embedding dimension</td><td>{self.embedding_dim}</td></tr>
    <tr><td>Excluded features</td><td>{html.escape(excluded_features)}</td></tr>
    <tr><td>Mask rate</td><td>{graphmae_cfg.mask_rate}</td></tr>
    <tr><td>Drop edge rate</td><td>{graphmae_cfg.drop_edge_rate}</td></tr>
  </tbody>
</table>
""",
        )
    ]

    training_stats_summary = ""
    if self.metadata.trainig_logs is not None:
      training_stats_summary = f"""
    <tr><td>Number of training seed nodes</td><td>{self._data.training_stats.num_train_seed_nodes}</td></tr>
    <tr><td>Number of validation seed nodes</td><td>{self._data.training_stats.num_valid_seed_nodes}</td></tr>
    <tr><td>Number of skipped training samples</td><td>{self._data.training_stats.num_skipped_train_samples}</td></tr>
    <tr><td>Training duration</td><td>{util.format_duration(self._data.training_stats.train_duration_seconds)}</td></tr>
"""

    common_tabs = report.get_common_tabs(
        hparams=self._data.hparams,
        schemas={
            "Raw": self._data.schema,
            "Normalized": self._get_live().normalized_schema,
        },
        feature_stats={
            "Default": self._data.feature_stats,
        },
        sampling_plans={"Default": self._data.sampling_plan},
        training_logs=self.metadata.trainig_logs,
        training_stats_summary=training_stats_summary,
        padding={
            "Default": self._data.padding,
        },
        architecture=self._data.core_model_config.architecture(),
        num_model_weights=common.num_model_weights(self._data.model_params),
        log_messages=self.metadata.captured_logs,
        final_evaluation=self._data.final_evaluation,
    )
    tabs.extend(common_tabs)
    return util.RichDisplay(report.html_tabs(tabs))

  def predict(
      self,
      graph: in_memory_graph.InMemoryGraph,
      seed_node_idxs: common.SeedNodeIdxs | None = None,
      *,
      verbose: int = 2,
  ) -> np.ndarray:
    """Predicts node embeddings of shape [N, D] for the specified target nodes.

    Args:
      graph: The input graph.
      seed_node_idxs: Indices of the nodes in the target nodeset for which to
        compute embeddings. If None, embeddings are computed for all nodes in
        the target nodeset.
      verbose: Verbosity level (2+ displays a progress bar).

    Returns:
      A float32 array of shape [N, embedding_dim].
    """
    if seed_node_idxs is None:
      num_nodes = graph.node_sets[self.target_nodeset].num_nodes
      assert num_nodes is not None
      effective_idxs = np.arange(num_nodes, dtype=np.int32)
    else:
      effective_idxs = np.asarray(seed_node_idxs, dtype=np.int32)

    if len(effective_idxs) == 0:
      return np.zeros((0, self.embedding_dim), dtype=np.float32)

    embedding_list = []
    for batch in self.predict_batch(
        graph, effective_idxs, verbose=verbose
    ):
      embedding_list.append(batch.embeddings)
    return np.concatenate(embedding_list, axis=0)

  def predict_batch(
      self,
      graph: in_memory_graph.InMemoryGraph,
      seed_node_idxs: common.SeedNodeIdxs | None = None,
      *,
      verbose: int = 2,
      to_host: bool = True,
  ) -> Iterator[BatchEmbedding]:
    """Generates batches of node embeddings.

    Args:
      graph: The input graph.
      seed_node_idxs: The indices of the seed nodes in the target nodeset. If
        None, embeddings are generated for all nodes in the target nodeset.
      verbose: The verbosity level.
      to_host: If true, the embeddings are transferred to the host as
        `np.ndarray`. If false, the embeddings stay on device as `jax.Array`,
        which avoids a device synchronization at each batch.

    Yields:
      The batches of node embeddings.
    """
    self._require_sampling_plan()

    if seed_node_idxs is None:
      num_nodes = graph.node_sets[self.target_nodeset].num_nodes
      assert num_nodes is not None
      seed_node_idxs = np.arange(num_nodes, dtype=np.int32)

    live = self._get_live()

    for normalized_batch in self._generate_normalized_batches(
        graph, seed_node_idxs, verbose=verbose
    ):
      yield self._embed_normalized_batch(
          live, normalized_batch, to_host=to_host
      )

  def _require_sampling_plan(self) -> sampling_config_lib.SamplingPlan:
    """Returns `self._data.sampling_plan`, or raises if it is `None`."""
    sampling_plan = self._data.sampling_plan
    if sampling_plan is None:
      raise ValueError(textwrap.dedent("""\
          This model cannot generate embeddings from a full graph because
          it does not have a sampling plan: it was trained on already
          sampled graph samples without a `sampling_plan` argument, so the
          sampling plan used to generate those samples is unknown. Instead,
          generate embeddings from graph samples (e.g.
          `model.predict_on_graph_sample_batch`, `model.evaluate_generator`,
          `model.to_tensorflow_function`), or re-train the model with the
          `sampling_plan` argument.
          """))
    return sampling_plan

  def _should_cache_normalized_features(
      self, graph: in_memory_graph.InMemoryGraph, num_seed_nodes: int
  ) -> bool:
    """Tests if the features of the entire graph should be normalized at once."""
    live = self._get_live()
    schema = self._data.schema
    if (
        "seed_timestamps" in live.normalizer.accepted_kwargs
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

  def _generate_normalized_batches(
      self,
      graph: in_memory_graph.InMemoryGraph,
      seed_node_idxs: common.SeedNodeIdxs,
      *,
      verbose: int,
      progress_bar_desc: str = "Inference",
  ) -> Iterator[inference.NormalizedBatch]:
    """Samples, normalizes, and prefetches batches from a full graph."""
    sampling_plan = self._require_sampling_plan()
    live = self._get_live()

    # If enough nodes are sampled, normalizing the features of the entire graph
    # once is cheaper than normalizing the features of each sampled sub-graph.
    # The normalized features are then directly gathered by the sampler.
    is_normalized = self._should_cache_normalized_features(
        graph, len(seed_node_idxs)
    )
    if is_normalized:
      graph = live.normalizer.normalize_numpy(graph)
      sampler_schema = live.normalizer.output_schema()
    else:
      sampler_schema = self._data.schema

    normalized_batches = inference.generate_normalized_batches(
        graph,
        seed_node_idxs,
        schema=self._data.schema,
        sampling_plan=sampling_plan,
        padding=self._data.padding,
        normalizer=live.normalizer,
        target_nodeset=self.target_nodeset,
        batch_size=self._data.hparams.batch_size,
        temporal_sampling=False,
        sampler_schema=sampler_schema,
        verbose=verbose,
        progress_bar_desc=progress_bar_desc,
        is_normalized=is_normalized,
    )
    # The next batches are sampled and normalized while the model is applied.
    return util.prefetch_iterator(
        normalized_batches, prefetch_size=_PREDICT_PREFETCH_SIZE
    )

  def _embed_normalized_batch(
      self,
      live: ModelLiveResource,
      normalized_batch: inference.NormalizedBatch,
      *,
      to_host: bool = True,
  ) -> BatchEmbedding:
    """Computes embeddings on an already merged and normalized (sub-)batch.

    Args:
      live: The live model resources.
      normalized_batch: The merged and normalized batch.
      to_host: If true, the embeddings are transferred to the host as
        `np.ndarray` (which blocks until the device computation is done). If
        false, the embeddings stay on device as `jax.Array`.

    Returns:
      The batch embedding.
    """
    emb = live.apply_embed(normalized_batch.jax_batch)
    if to_host:
      emb = np.asarray(jax.device_get(emb), dtype=np.float32)
    return BatchEmbedding(
        batch_seed_node_idxs=normalized_batch.batch_seed_node_idxs,
        normalized_merged_graph=normalized_batch.normalized_merged_graph,
        merged_seed_node_idxs=normalized_batch.merged_seed_node_idxs,
        embeddings=emb,
    )

  def predict_on_graph_sample_batch(
      self,
      graph_samples: list[in_memory_graph.InMemoryGraph],
  ) -> np.ndarray:
    """Predicts embeddings for a batch of pre-sampled graph samples.

    Args:
      graph_samples: A list of graph samples to compute embeddings on.

    Returns:
      A float32 array of shape `[len(graph_samples), embedding_dim]`.
    """
    live = self._get_live()
    merged_graph, merge_offsets = merge_lib.GraphMerger(
        schema=self._data.schema,
        padding=self._data.padding,
    )(graph_samples)
    normalized_batch = inference.merged_graph_to_normalized_batch(
        merged_graph,
        merge_offsets,
        schema=self._data.schema,
        normalizer=live.normalizer,
        target_nodeset=self.target_nodeset,
    )
    return np.asarray(
        self._embed_normalized_batch(live, normalized_batch).embeddings,
        dtype=np.float32,
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
    """Evaluates the quality of the embedding on several embedding related tasks.

    Usage example:

      ```python
      evaluation = model.evaluate(test_graph)
      # Show the evaluation in a notebook
      evaluation
      # Access the evaluation values
      print(evaluation.loss, evaluation.user_metrics["recon_loss"])
      ```

    The evaluation contains (currently; more will be added):
      - recon_loss: Reconstruction loss of the masked seed nodes' input
        features. A value of 0 indicates a perfect reconstruction, and a value
        of ~1 indicates a reconstruction no better than random.
      - var_loss: Penalizes embedding dimensions with a small variance across
        nodes. A value of 0 indicates that no dimension has collapsed, and the
        maximum value indicates that all the nodes have the same embedding.
      - cov_loss: Penalizes correlated embedding dimensions. A value of 0
        indicates that the embedding dimensions are decorrelated (i.e., not
        redundant).
      - total_loss: The sum of the three losses above, i.e. the loss optimized
        during training (also stored in Evaluation.loss).

    The losses depend on the loss hyper-parameters, and are hard to compare
    across models. The evaluation also contains the following interpretable
    metrics, which are not optimized:
      - recon_cosine: Mean cosine similarity in [-1, 1] between the
        reconstructed and the original input features of the masked seed
        nodes. A value of 1 indicates a perfect reconstruction.
      - recon_relative_error: Reconstruction error relative to predicting the
        same (best constant) embedding for all the nodes. A value of 0
        indicates a perfect reconstruction, 1 a reconstruction no better than
        the constant prediction, and a value above 1 a worse one.
      - embedding_rank_ratio: Fraction in (0, 1] of the embedding dimensions
        effectively used by the node embeddings (effective rank). Higher is
        better: a value close to 0 indicates that the embeddings collapsed to a
        few directions.
      - target_rank_ratio: Same as embedding_rank_ratio, for the reconstruction
        targets, i.e., the embedded input features before message passing. If
        it is close to 0, the reconstruction metrics are not meaningful.

    Because this model is trained with self-supervised learning, it should
    ultimately be evaluated on its final application. The evaluation report
    produced by model.evaluate() can only provide a general indication that
    the embedding learning was successful.

    To evaluate on pre-sampled subgraphs (e.g., when the model has no sampling
    plan), use evaluate_generator instead.

    Args:
      graph: Input graph matching the model's training schema. No ground-truth
        label feature is required.
      num_eval_steps: Maximum number of seed nodes to evaluate. If None, all
        specified seed nodes are evaluated. If smaller than the number of seed
        nodes, a random subset of size num_eval_steps is sampled without
        replacement.
      seed_node_idxs: Indices of the seed nodes in the target nodeset to
        evaluate. If None, all nodes in the target nodeset are considered.
      verbose: Verbosity level (1+ logs summary messages; 2+ displays a progress
        bar).
      random_seed: Random seed used to sub-sample the seed nodes when
        num_eval_steps is smaller than the number of seed nodes. If None, a new
        random seed is used each time.

    Returns:
      An evaluation whose loss is the average total loss, num_examples is the
      number of evaluated seed nodes, and user_metrics contains the recon_loss,
      var_loss, cov_loss, total_loss, recon_cosine, recon_relative_error,
      embedding_rank_ratio, and target_rank_ratio metrics.
    """
    self._require_sampling_plan()

    num_nodes = graph.node_sets[self.target_nodeset].num_nodes
    assert num_nodes is not None

    if seed_node_idxs is None:
      seed_node_idxs = np.arange(num_nodes, dtype=np.int32)
    else:
      seed_node_idxs = np.asarray(seed_node_idxs, dtype=np.int32)

    if num_eval_steps is not None and num_eval_steps < len(seed_node_idxs):
      rng = np.random.default_rng(random_seed)
      seed_node_idxs = rng.choice(
          seed_node_idxs, size=num_eval_steps, replace=False
      )

    if verbose >= 1:
      log.info("Evaluating model on %d samples", len(seed_node_idxs))

    live = self._get_live()
    normalized_batches = self._generate_normalized_batches(
        graph,
        seed_node_idxs,
        verbose=verbose,
        progress_bar_desc="Evaluation",
    )
    return self._evaluate_normalized_batches(live, normalized_batches)

  def evaluate_generator(
      self,
      graph_samples: Iterator[in_memory_graph.InMemoryGraph],
      num_eval_steps: int | None = 10_000,
      *,
      verbose: int = 2,
  ) -> evaluation.Evaluation:
    """Evaluates reconstruction and VICReg losses on an iterator of subgraphs.

    Merges up to num_eval_steps pre-sampled subgraphs into padded batches,
    normalizes them, and computes the same GraphMAE reconstruction and VICReg
    losses as evaluate (see evaluate for details).

    Each graph sample must match the model's training schema (no ground-truth
    label feature is required), and the first node of the target nodeset is the
    seed node of the sample. Unlike evaluate, this method does not require the
    model to have a sampling plan.

    Usage example:

      ```python
      iterator = dataset.sample_generator.single_iterator()
      evaluation = model.evaluate_generator(iterator, num_eval_steps=1000)
      ```

    Args:
      graph_samples: Iterator of pre-sampled subgraphs matching the model's
        training schema, with the first node of the target nodeset as the seed
        node.
      num_eval_steps: Maximum number of subgraph samples to evaluate. If None,
        all the graph samples are evaluated.
      verbose: Verbosity level (1+ logs summary messages; 2+ displays a progress
        bar).

    Returns:
      An evaluation whose loss is the average total loss, num_examples is the
      number of evaluated subgraph samples, and user_metrics contains the
      same metrics as evaluate.
    """
    if num_eval_steps is not None:
      graph_samples = itertools.islice(graph_samples, num_eval_steps)

    live = self._get_live()
    batch_size = max(1, self._data.hparams.batch_size // 2)
    if verbose >= 1:
      log.info("Evaluating model on generator")

    normalized_batches = inference.generate_normalized_batches_from_samples(
        graph_samples,
        schema=self._data.schema,
        padding=self._data.padding,
        normalizer=live.normalizer,
        target_nodeset=self.target_nodeset,
        batch_size=batch_size,
        num_steps=num_eval_steps,
        verbose=verbose,
        progress_bar_desc="Evaluation",
    )
    return self._evaluate_normalized_batches(live, normalized_batches)

  def _evaluate_normalized_batches(
      self,
      live: ModelLiveResource,
      normalized_batches: Iterator[inference.NormalizedBatch],
  ) -> evaluation.Evaluation:
    """Aggregates validation step losses over an iterator of `NormalizedBatch`.

    The per-batch losses are accumulated on device (weighted by the number of
    seed nodes in the batch), and only transferred to the host once at the end,
    to avoid a device synchronization at each step.

    Args:
      live: The live model resources.
      normalized_batches: The batches to evaluate.

    Returns:
      The example-weighted average of the validation losses.
    """
    metric_keys = (
        "loss",
        "recon_loss",
        "var_loss",
        "cov_loss",
        *_INTERPRETABLE_METRIC_KEYS,
    )
    total_examples = 0
    weighted_sums: dict[str, jax.Array] | None = None

    for normalized_batch in normalized_batches:
      batch_count = len(normalized_batch.merged_seed_node_idxs)
      metrics = live.valid_step(
          self._data.model_params, None, normalized_batch.jax_batch
      )
      weighted_metrics = {k: metrics[k] * batch_count for k in metric_keys}
      if weighted_sums is None:
        weighted_sums = weighted_metrics
      else:
        weighted_sums = {
            k: weighted_sums[k] + weighted_metrics[k] for k in metric_keys
        }
      total_examples += batch_count

    if weighted_sums is None or total_examples == 0:
      raise ValueError(
          "No examples to evaluate: the evaluation graph samples or seed nodes"
          " are empty."
      )

    # Single device-to-host transfer.
    host_sums = jax.device_get(weighted_sums)
    avg = {k: float(host_sums[k]) / total_examples for k in metric_keys}
    return evaluation.Evaluation(
        loss=avg["loss"],
        num_examples=total_examples,
        user_metrics={
            "recon_loss": avg["recon_loss"],
            "var_loss": avg["var_loss"],
            "cov_loss": avg["cov_loss"],
            "total_loss": avg["loss"],
            **{k: avg[k] for k in _INTERPRETABLE_METRIC_KEYS},
        },
    )

  def _get_live(self) -> ModelLiveResource:
    """Initializes live model resources if not already constructed."""
    if self._live is None:
      normalizer = self._data.normalizer_config.make()
      normalized_schema = normalizer.output_schema()
      core_model = self._data.core_model_config.make(normalized_schema)

      @jax.jit
      def apply_embed(batch: Batch) -> jnp.ndarray:
        assert core_model is not None
        model_params = self._data.model_params
        assert model_params is not None
        out = core_model.apply(
            model_params,
            batch,
            training=False,
            method=CoreModel.embed,
        )
        return out  # pyrefly: ignore[bad-return]

      valid_step = training.build_valid_step(
          core_model,
          self._data.core_model_config.compute_loss,
          rngs={"masking": jax.random.PRNGKey(self._data.hparams.random_seed)},
      )

      self._live = ModelLiveResource(
          core_model=core_model,
          apply_embed=apply_embed,
          valid_step=valid_step,
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
    normalization and core embedding logic. It is designed to be used with
    pre-sampled subgraphs.

    Depending on `input_format`, the returned TensorFlow function takes:
      - TF_GRAPH: A `dgf.data.TFInMemoryGraph` named `graph`, and a 1D tensor of
        integers named `seed_node_idxs` specifying the indices of the nodes
        within the target nodeset for which to generate embeddings.
      - TF_GRAPH_DICT: The items of a `dgf.data.TFInMemoryGraphDict` as keyword
        arguments, and `seed_node_idxs`.
      - SERIALIZED_TFGNN_GRAPHS: A 1D string tensor named `examples` containing
        serialized TF GNN Graph Samples. One embedding is generated for each
        graph sample, on the first node of the target nodeset (which is the
        seed node of the DGF graph samplers).

    Usage example:

      ```python
      # Consumes serialized TF GNN Graph Samples e.g. for a TFX BulkInferrer.
      tf_embed_fn = model.to_tensorflow_function(
          input_format="SERIALIZED_TFGNN_GRAPHS")
      embeddings = tf_embed_fn(tf.constant([serialized_example]))
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
      A `tf.Module` with a `__call__` method returning a float32 tensor of
      embeddings.
    """
    input_format = common.resolve_tf_function_input_format(
        input_format, consume_tf_graph_dict
    )

    live = self._get_live()
    tf_apply_core_model = jax2tf.convert(
        live.apply_embed,
        polymorphic_shapes=[(None, "(b,)")],
        native_serialization_platforms=["cpu", "cuda"],
    )
    schema = self._data.schema

    graph_spec = io_tf_lib.schema_to_spec(schema)
    graph_dict_spec = io_tf_lib.schema_to_dict_spec(schema)
    seed_node_idxs_spec = tf.TensorSpec(
        shape=[None], dtype=tf.int32, name="seed_node_idxs"
    )

    class TfPredictWrapperBase(tf.Module):
      """Base TensorFlow module wrapping the JAX node embedding core model."""

      def __init__(
          self,
          name: str,
          tf_apply_fn,
          normalizer: normalize_lib.GraphNormalizer,
          schema: schema_lib.GraphSchema,
          padding: padding_data_lib.Padding,
      ):
        super().__init__(name=name)
        self._tf_apply_core_model = tf_apply_fn
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
        jax_graph = jax_lib.graph_to_jax_graph(
            normalized_graph, cast_arrays=False
        )
        return self._tf_apply_core_model((jax_graph, seed_node_idxs))

    class TfPredictWrapperTFGraph(TfPredictWrapperBase):
      """Consumes `TFInMemoryGraph` and seed node indices."""

      def __init__(self, tf_apply_fn, normalizer, schema, padding):
        super().__init__(
            "TfPredictWrapperTFGraph",
            tf_apply_fn,
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
      """Consumes flattened TF graph dictionary kwargs."""

      def __init__(self, tf_apply_fn, normalizer, schema, padding):
        super().__init__(
            "TfPredictWrapperTFGraphDict",
            tf_apply_fn,
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

      def __init__(self, tf_apply_fn, normalizer, schema, padding):
        super().__init__(
            "TfPredictWrapperSerializedTFGNNGraphs",
            tf_apply_fn,
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
        self._data.schema,
        self._data.padding,
    )

    if input_format == common.TFFunctionInputFormat.TF_GRAPH_DICT:
      spec_kwargs = dict(graph_dict_spec)
      spec_kwargs["seed_node_idxs"] = seed_node_idxs_spec
      wrapper.__call__ = wrapper.__call__.get_concrete_function(**spec_kwargs)

    return wrapper


common.register_model(NodeEmbeddingModel, ModelData)
