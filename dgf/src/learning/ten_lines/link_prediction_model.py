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

"""The edge prediction model."""

import dataclasses
import os
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple
import dataclasses_json
from dgf.src.data import in_memory_graph
from dgf.src.data import padding as padding_data_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.generate import edge_neighbor_generator as edge_neighbor_generator_lib
from dgf.src.io import jax as jax_lib
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import evaluation
from dgf.src.learning.ten_lines import link_prediction_core_model
from dgf.src.learning.ten_lines import report
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import util
from dgf.src.util import util_ext
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import orbax.checkpoint as ocp
import tqdm

Batch = link_prediction_core_model.Batch
InferenceBatch = link_prediction_core_model.InferenceBatch
CoreModel = link_prediction_core_model.CoreModel
CoreModelConfig = link_prediction_core_model.CoreModelConfig

FILENAME_PARAMS = "params"


# TODO(gbm): Populate.
@dataclasses.dataclass
class HParam(common.HParam):
  """Hyperparameters for the NodePredictionModel."""

  num_negative_nodes: int = 8
  message_passing_on_target_edgeset: bool = True
  negative_edges: Literal["random", "random-walk"] = "random"
  random_walk_num_walks_per_negative: int = 10


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class LinkPredictionTask:
  target_edgeset: str


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class TrainingStats:
  num_train_seed_edges: Optional[int]
  num_valid_seed_edges: Optional[int]
  train_duration_seconds: float


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ModelData:
  """All the data of the model. Used for construction and serialization."""

  core_model_config: CoreModelConfig
  task: LinkPredictionTask
  hparams: HParam
  schema: schema_lib.GraphSchema
  source_normalizer_config: normalize_lib.GraphNormalizerConfig
  target_normalizer_config: normalize_lib.GraphNormalizerConfig
  positive_source_padding: padding_data_lib.Padding
  positive_target_padding: padding_data_lib.Padding
  negative_target_padding: padding_data_lib.Padding
  source_feature_stats: statistics_lib.GraphFeatureStatistics
  target_feature_stats: statistics_lib.GraphFeatureStatistics
  source_sampling_plan: sampling_config_lib.SamplingPlan
  target_sampling_plan: sampling_config_lib.SamplingPlan
  training_stats: TrainingStats
  edge_neighbor_generator: (
      edge_neighbor_generator_lib.EdgeNeighborGeneratorConfig
  ) = edge_neighbor_generator_lib.registry.field(
      default_factory=edge_neighbor_generator_lib.RandomEdgeNeighborGeneratorConfig
  )

  # This field is serialized / deserialized manually.
  model_params: Optional[jaxtyping.PyTree] = dataclasses.field(
      default_factory=lambda: None,
      metadata=dataclasses_json.config(exclude=dataclasses_json.Exclude.ALWAYS),
      repr=False,
  )

  def num_model_weights(self) -> Dict[str, int]:
    """Returns a dictionary of the type and number of weights of the model.

    Example:
      {"float32": 435246, "int16": 345345}
    """
    return common.num_model_weights(self.model_params)


@dataclasses.dataclass
class ModelLiveResource:
  """Resources necessary for inference."""

  core_model: CoreModel
  apply_core_model: Callable[[InferenceBatch], jnp.ndarray]
  apply_encoder_source: Callable[[object, jax.Array], jax.Array]
  apply_encoder_target: Callable[[object, jax.Array], jax.Array]
  source_normalized_schema: schema_lib.GraphSchema
  target_normalized_schema: schema_lib.GraphSchema
  source_normalizer: normalize_lib.GraphNormalizer
  target_normalizer: normalize_lib.GraphNormalizer


@dataclasses.dataclass
class BatchPrediction:
  """Prediction yielded by "predict_batch" function."""

  batch_source_node_idxs: np.ndarray
  batch_target_node_idxs: np.ndarray
  predictions: np.ndarray


def _interleave_positives_and_negatives(
    pos_src: np.ndarray,
    pos_trg: np.ndarray,
    neg_trg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Interleaves positive and negative examples for link prediction evaluation.

  The output arrays are structured such that for each positive edge, its
  positive example comes first, followed by its corresponding negative examples.
  This repeats for all positive edges.

  Structure:
  [pos_0, neg_0_0, neg_0_1, ..., pos_1, neg_1_0, neg_1_1, ...]

  Toy Example:
  Inputs:
    pos_src = [1, 2]
    pos_trg = [10, 20]
    neg_trg = [[11, 12], [21, 22]]

  Output:
    full_src = [1,  1,  1,  2,  2,  2]
    full_trg = [10, 11, 12, 20, 21, 22]
               ^--- Edge 0 ---^--- Edge 1 ---^
               Pos  Neg  Neg  Pos  Neg  Neg

  Args:
    pos_src: 1D array of positive source node indices.
    pos_trg: 1D array of positive target node indices.
    neg_trg: 2D array of negative target node indices, shape (num_pos,
      num_negative_nodes).

  Returns:
    A tuple of (full_src, full_trg) arrays.
  """
  num_negative_nodes = neg_trg.shape[1]
  src_expanded = np.expand_dims(pos_src, 1)
  src_repeated = np.repeat(src_expanded, 1 + num_negative_nodes, axis=1)
  full_src = src_repeated.flatten()

  pos_trg_expanded = np.expand_dims(pos_trg, 1)
  full_trg = np.concatenate([pos_trg_expanded, neg_trg], axis=1).flatten()

  return full_src, full_trg


def _separate_positives_and_negatives(
    probs: np.ndarray, examples_per_seed_edge: int
) -> Tuple[np.ndarray, np.ndarray]:
  """Separates positive and negative probabilities from a combined array.

  Toy Example:
  Inputs:
    probs = [10, 11, 12, 20, 21, 22]
    examples_per_seed_edge = 3

  Output:
    pos_probs = [10, 20]
    neg_probs = [11, 12, 21, 22]

  Args:
    probs: Combined probabilities array, shape (num_groups *
      examples_per_seed_edge,).
    examples_per_seed_edge: Number of examples per seed edge (1 pos + K neg).

  Returns:
    A tuple of (pos_probs, neg_probs) arrays.
  """
  num_groups = len(probs) // examples_per_seed_edge
  reshaped_probs = probs.reshape(num_groups, examples_per_seed_edge)
  pos_probs = reshaped_probs[:, 0]
  neg_probs = reshaped_probs[:, 1:].flatten()
  return pos_probs, neg_probs


class LinkPredictionModel(common.Model):
  """The user-visible returned model object for edge prediction."""

  def __init__(self, data: ModelData) -> None:
    super().__init__(data)
    self._data = data
    self._live = None

  @classmethod
  def name(cls) -> str:
    return "LinkPrediction"

  def data(self) -> ModelData:
    return self._data

  def _internal_save(self, path: str) -> None:
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        os.path.join(path, FILENAME_PARAMS),
        self._data.model_params,
    )
    checkpointer.wait_until_finished()

  def _internal_load(self, path: str) -> None:
    checkpointer = ocp.StandardCheckpointer()
    self._data.model_params = checkpointer.restore(
        os.path.join(path, FILENAME_PARAMS)
    )

  def describe(self) -> util.RichDisplay:
    """Rich display for colab."""
    tabs = []

    tabs.append((
        "Objective",
        f"""
<p><b>Node prediction model:</b> Predict the value of a node feature.</p>
<ul>
  <li>Target edgeset: {self._data.task.target_edgeset}</li>
</ul>
""",
    ))

    training_stats_summary = ""
    if self.metadata.trainig_logs is not None:
      training_stats_summary = f"""
<ul>
  <li>Number of training seed edges: {self._data.training_stats.num_train_seed_edges}</li>
  <li>Number of validation seed edges: {self._data.training_stats.num_valid_seed_edges}</li>
  <li>Training duration: {util.format_duration(self._data.training_stats.train_duration_seconds)}</li>
</ul>
"""

    # Reconstruct normalizers to get normalized schemas.
    source_normalizer = self._data.source_normalizer_config.make()
    source_normalized_schema = source_normalizer.output_schema()
    target_normalizer = self._data.target_normalizer_config.make()
    target_normalized_schema = target_normalizer.output_schema()

    common_tabs = report.get_common_tabs(
        hparams=self.data().hparams,
        schemas={
            "Raw": self.data().schema,
            "Source Normalized": source_normalized_schema,
            "Target Normalized": target_normalized_schema,
        },
        sampling_plans={
            "Source": self._data.source_sampling_plan,
            "Target": self._data.target_sampling_plan,
        },
        feature_stats={
            "Source": self._data.source_feature_stats,
            "Target": self._data.target_feature_stats,
        },
        training_logs=self.metadata.trainig_logs,
        training_stats_summary=training_stats_summary,
        padding={
            "Positive Source": self._data.positive_source_padding,
            "Positive Target": self._data.positive_target_padding,
            "Negative Target": self._data.negative_target_padding,
        },
        architecture=self.data().core_model_config.architecture(),
        num_model_weights=common.num_model_weights(self.data().model_params),
    )

    tabs.extend(common_tabs)

    html = report.html_tabs(tabs)
    return util.RichDisplay(html)

  def _get_live(self) -> ModelLiveResource:
    """Initializes model resources if they are not already live."""
    if self._live is None:
      source_normalizer = self._data.source_normalizer_config.make()
      target_normalizer = self._data.target_normalizer_config.make()

      source_normalized_schema = source_normalizer.output_schema()
      target_normalized_schema = target_normalizer.output_schema()

      core_model = self._data.core_model_config.make(
          source_schema=source_normalized_schema,
          target_schema=target_normalized_schema,
      )

      @jax.jit
      def apply_core_model(batch: InferenceBatch):
        return core_model.apply(
            self._data.model_params, batch, method=core_model.call_inference
        )

      @jax.jit
      def apply_encoder_source(graph: object, offset: jax.Array):
        return core_model.apply(
            self._data.model_params,
            graph,
            offset,
            method=core_model.call_src_encoder,
        )

      @jax.jit
      def apply_encoder_target(graph: object, offset: jax.Array):
        return core_model.apply(
            self._data.model_params,
            graph,
            offset,
            method=core_model.call_trg_encoder,
        )

      self._live = ModelLiveResource(
          core_model=core_model,
          apply_core_model=apply_core_model,
          apply_encoder_source=apply_encoder_source,
          apply_encoder_target=apply_encoder_target,
          source_normalized_schema=source_normalized_schema,
          target_normalized_schema=target_normalized_schema,
          source_normalizer=source_normalizer,
          target_normalizer=target_normalizer,
      )

    return self._live

  def predict(
      self,
      graph: in_memory_graph.InMemoryGraph,
      source_node_idxs: common.SeedNodeIdxs,
      target_node_idxs: common.SeedNodeIdxs,
      *,
      all_combinations: bool = False,
      verbose: int = 2,
  ) -> np.ndarray:
    """Predicts edge probabilities between source and target nodes.

    This function computes scores for edges pointing from source nodes to target
    nodes, and returns them as probabilities in [0, 1].

    Warning: This method creates an in-memory sampling on the graph. This
    operation is costly. When possible, merge different calls to model.predict.

    Usage example:

    ```python
    # Predict probability for edges 0->10 and 1->11
    probs = model.predict(
      graph,
      source_node_idxs=[0, 1],
      target_node_idxs=[10, 11]
    )

    # Predict probability for edges 0->10, 0->11, 0->12
    probs = model.predict(
      graph,
      source_node_idxs=[0],
      target_node_idxs=[10, 11, 12]
    )

    # Predict probability for all pairs of edges between sources [0,1] and
    # targets [10,11,12]: 0->10, 0->11, 0->12, 1->10, 1->11, 1->12
    probs = model.predict(
        graph,
        source_node_idxs=[0, 1],
        target_node_idxs=[10, 11, 12],
        all_combinations=True
    )
    # probs will be shape [2, 3]
    ```

    Args:
      graph: The graph containing nodes and features.
      source_node_idxs: A list of node indices for edge sources.
      target_node_idxs: A list of node indices for edge targets.
      all_combinations: If False (default), `len(target_node_idxs)` must be
        divisible by `len(source_node_idxs)`. If `m = len(target_node_idxs) //
        len(source_node_idxs)`, edge `source_node_idxs[i // m] ->
        target_node_idxs[i]` is predicted for each `i`. This covers pairwise
        prediction (`len(sources)==len(targets)`) and one-to-many prediction
        (`len(sources)==1`). If True, predictions are made for all combinations
        of source and target nodes (cartesian product).
      verbose: Verbosity level.

    Returns:
      If `all_combinations` is False, returns an array `p` of shape
      `[len(target_node_idxs)]` with edge probabilities.
      If `all_combinations` is True, returns an array `p` of shape
      `[len(source_node_idxs), len(target_node_idxs)]` with edge probabilities.
    """
    if not all_combinations:
      if len(target_node_idxs) % len(source_node_idxs) != 0:
        raise ValueError(
            "`len(target_node_idxs)` must be divisible by"
            " `len(source_node_idxs)` when `all_combinations=False`"
        )

    prediction_list = []
    for batch in self.predict_batch(
        graph,
        source_node_idxs,
        target_node_idxs,
        all_combinations,
        verbose=verbose,
    ):
      prediction_list.append(batch.predictions)

    predictions = np.concatenate(prediction_list, axis=0)

    if all_combinations:
      return predictions.reshape(len(source_node_idxs), len(target_node_idxs))
    else:
      return predictions

  def execute_with_split_on_error(
      self, fn: Callable[..., Iterator[Any]], *args
  ) -> Iterator[Any]:
    """Executes `fn`, splitting inputs on `InsufficientPaddingError`.

    Args:
      fn: A callable yielding results from `*args`.
      *args: Lists of equal length, representing batch elements.

    Yields:
      Results from `fn`.

    Raises:
      merge_lib.InsufficientPaddingError: If the error persists with a batch
        size of 1.
    """
    try:
      yield from fn(*args)
    except merge_lib.InsufficientPaddingError:
      if len(args[0]) <= 1:
        raise
      mid = len(args[0]) // 2
      yield from self.execute_with_split_on_error(fn, *[a[:mid] for a in args])
      yield from self.execute_with_split_on_error(fn, *[a[mid:] for a in args])

  def _build_samplers(
      self, graph: in_memory_graph.InMemoryGraph
  ) -> Tuple[in_memory_sampler_lib.Sampler, in_memory_sampler_lib.Sampler]:
    sampler_kwargs = {
        "graph": graph,
        "schema": self._data.schema,
        "batch_size": self._data.hparams.batch_size,
        "edgeset_to_mask": (
            self._data.task.target_edgeset
            if self._data.hparams.message_passing_on_target_edgeset
            else None
        ),
    }
    source_sampler = in_memory_sampler_lib.create_sampler(
        plan=self._data.source_sampling_plan, **sampler_kwargs
    )
    target_sampler = in_memory_sampler_lib.create_sampler(
        plan=self._data.target_sampling_plan, **sampler_kwargs
    )
    return source_sampler, target_sampler

  def predict_batch(
      self,
      graph: in_memory_graph.InMemoryGraph,
      source_node_idxs: common.SeedNodeIdxs,
      target_node_idxs: common.SeedNodeIdxs,
      all_combinations: bool = False,
      verbose: int = 2,
      source_sampler: Optional[in_memory_sampler_lib.Sampler] = None,
      target_sampler: Optional[in_memory_sampler_lib.Sampler] = None,
  ) -> Iterator[BatchPrediction]:
    """Generate batches of predictions."""
    live = self._get_live()

    np_source_node_idxs = np.asarray(source_node_idxs)
    np_target_node_idxs = np.asarray(target_node_idxs)

    mask_edges = self._data.hparams.message_passing_on_target_edgeset
    edge_lookup = None
    target_edgeset = self._data.task.target_edgeset
    if mask_edges:
      edge_set = graph.edge_sets[target_edgeset]
      edge_lookup = util_ext.CreateEdgeIndexer(edge_set.adjacency)

    if all_combinations:
      # TODO(gbm): If all_combinations=true and the model is a two tower model,
      # we can optimize the code below by not computing the mesh, and instead
      # repeat the source / target sampling plan n x m times.
      grid = np.meshgrid(
          np_source_node_idxs, np_target_node_idxs, indexing="ij"
      )
      flat_sources = grid[0].flatten()
      flat_targets = grid[1].flatten()
      num_examples = len(flat_sources)
    else:
      num_examples = len(np_target_node_idxs)
      m = num_examples // len(np_source_node_idxs)
      flat_sources = np.repeat(np_source_node_idxs, m)
      flat_targets = np_target_node_idxs

    source_nodeset = self._data.schema.edge_sets[
        self._data.task.target_edgeset
    ].source
    target_nodeset = self._data.schema.edge_sets[
        self._data.task.target_edgeset
    ].target

    assert (source_sampler is None) == (target_sampler is None)
    if source_sampler is None:
      source_sampler, target_sampler = self._build_samplers(graph)
    assert source_sampler is not None
    assert target_sampler is not None

    generator = util.batch_indices_generator(
        np.arange(num_examples),
        batch_size=self._data.hparams.batch_size,
        drop_remainder=False,
        shuffle=False,
    )
    if verbose >= 2:
      generator = tqdm.tqdm(
          generator,
          desc="Inference",
          total=util.num_batches(
              num_examples,
              batch_size=self._data.hparams.batch_size,
              drop_remainder=False,
          ),
      )

    def merge_and_predict(
        sub_src: np.ndarray,
        sub_trg: np.ndarray,
        sub_src_samples: List[in_memory_graph.InMemoryGraph],
        sub_trg_samples: List[in_memory_graph.InMemoryGraph],
    ):

      source_merged, source_offsets = merge_lib.merge_graphs(
          sub_src_samples,
          self._data.schema,
          padding=self._data.positive_source_padding,
          sentinel_offset=False,
      )
      target_merged, target_offsets = merge_lib.merge_graphs(
          sub_trg_samples,
          self._data.schema,
          padding=self._data.positive_target_padding,
          sentinel_offset=False,
      )

      source_normalized = live.source_normalizer.normalize_numpy(source_merged)
      target_normalized = live.target_normalizer.normalize_numpy(target_merged)

      source_jax = jax_lib.graph_to_jax_graph(source_normalized)
      target_jax = jax_lib.graph_to_jax_graph(target_normalized)

      batch = InferenceBatch(
          source_graph=source_jax,
          target_graph=target_jax,
          source_offset=jnp.asarray(source_offsets[source_nodeset]),
          target_offset=jnp.asarray(target_offsets[target_nodeset]),
      )

      logits = live.apply_core_model(batch)
      predictions = np.asarray(jax.nn.sigmoid(logits))

      yield BatchPrediction(
          batch_source_node_idxs=sub_src,
          batch_target_node_idxs=sub_trg,
          predictions=predictions,
      )

    for batch_indices in generator:
      batch_src = flat_sources[batch_indices]
      batch_trg = flat_targets[batch_indices]

      if mask_edges:
        assert edge_lookup is not None
        queries = np.stack([batch_src, batch_trg], axis=0)
        masked_edge_idxs = edge_lookup.query_array(queries)
        source_samples = source_sampler.sample(
            batch_src, masked_edge_idxs=masked_edge_idxs
        )
        target_samples = target_sampler.sample(
            batch_trg, masked_edge_idxs=masked_edge_idxs
        )
      else:
        source_samples = source_sampler.sample(batch_src)
        target_samples = target_sampler.sample(batch_trg)

      yield from self.execute_with_split_on_error(
          merge_and_predict,
          batch_src,
          batch_trg,
          source_samples,
          target_samples,
      )

  def predict_embedding(
      self,
      graph: in_memory_graph.InMemoryGraph,
      node_idxs: common.SeedNodeIdxs,
      encoder: Literal["source", "target"],
      *,
      verbose: int = 2,
  ) -> np.ndarray:
    """Predicts node embeddings for source or target sides.

    Computes the latent representations (embeddings) of the specified
    `node_idxs`. The `encoder` argument determines whether the embeddings are
    generated using the "source" or "target" encoder of the model.

    Usage example:

    ```python

    model = dgf.learning.train_link_model(..., node_embedding_dim=128)

    # Predict the embeddings of source nodes 0 and 1
    emb_src = model.predict_embedding(graph, [0, 1], encoder="source")
    # Shape: [2, embedding_dim]

    # Predict the embeddings of target nodes 10 and 11
    emb_trg = model.predict_embedding(graph, [10, 11], encoder="target")
    # Shape: [2, embedding_dim]

    # If the model uses a dot product for link prediction, the raw scores
    # for the pairs (0 -> 10) and (1 -> 11) can be computed by:
    proba = jax.nn.sigmoid(jnp.sum(emb_src * emb_trg, axis=-1))
    # Shape: [2]

    # This `proba` contains the predicted probability for the edge
    # source 0 -> target 10 and source 1 -> target 11.
    ```

    Args:
      graph: The graph containing nodes and features.
      node_idxs: A list of node indices for which to predict the embedding.
      encoder: Whether to predict the embedding using the "source" or "target"
        encoder of the model. Must be one of {"source", "target"}.
      verbose: Verbosity level.

    Returns:
      An array `e` of shape `[len(node_idxs), embedding_dim]` with node
      embeddings.
    """
    live = self._get_live()
    np_node_idxs = np.asarray(node_idxs)

    if encoder == "source":
      sampler = in_memory_sampler_lib.create_sampler(
          plan=self._data.source_sampling_plan,
          graph=graph,
          schema=self._data.schema,
          batch_size=self._data.hparams.batch_size,
      )
      padding = self._data.positive_source_padding
      normalizer = live.source_normalizer
      apply_fn = live.apply_encoder_source
      nodeset = self._data.schema.edge_sets[
          self._data.task.target_edgeset
      ].source
    elif encoder == "target":
      sampler = in_memory_sampler_lib.create_sampler(
          plan=self._data.target_sampling_plan,
          graph=graph,
          schema=self._data.schema,
          batch_size=self._data.hparams.batch_size,
      )
      padding = self._data.positive_target_padding
      normalizer = live.target_normalizer
      apply_fn = live.apply_encoder_target
      nodeset = self._data.schema.edge_sets[
          self._data.task.target_edgeset
      ].target
    else:
      raise ValueError(f"Invalid encoder: {encoder}")

    generator = util.batch_indices_generator(
        np.arange(len(np_node_idxs)),
        batch_size=self._data.hparams.batch_size,
        drop_remainder=False,
        shuffle=False,
    )
    if verbose >= 2:
      generator = tqdm.tqdm(
          generator,
          desc=f"Embedding ({encoder})",
          total=util.num_batches(
              len(np_node_idxs),
              batch_size=self._data.hparams.batch_size,
              drop_remainder=False,
          ),
      )

    def merge_and_predict_emb(sub_samples: List[in_memory_graph.InMemoryGraph]):

      merged, offsets = merge_lib.merge_graphs(
          sub_samples,
          self._data.schema,
          padding=padding,
          sentinel_offset=False,
      )

      normalized = normalizer.normalize_numpy(merged)
      jax_graph = jax_lib.graph_to_jax_graph(normalized)

      emb = apply_fn(jax_graph, jnp.asarray(offsets[nodeset]))
      yield np.asarray(emb)

    embeddings_list = []
    for batch_indices in generator:
      nodes = np_node_idxs[batch_indices]
      samples = sampler.sample(nodes)

      for emb in self.execute_with_split_on_error(
          merge_and_predict_emb, samples
      ):
        embeddings_list.append(emb)

    return np.concatenate(embeddings_list, axis=0)

  def evaluate(
      self,
      graph: in_memory_graph.InMemoryGraph,
      num_eval_steps: Optional[int] = 10_000,
      *,
      seed_edge_idxs: Optional[common.SeedNodeIdxs] = None,
      verbose: int = 2,
      random_seed: Optional[int] = None,
      num_negative_nodes: Optional[int] = None,
  ) -> evaluation.Evaluation:
    """Evaluates the model on a given graph.

    Args:
      graph: The input graph data.
      num_eval_steps: Maximum number of evaluation batches to run.
      seed_edge_idxs: Indices of the seed edges within the target edgeset to
        evaluate. If None, all edges in the target edgeset are used.
      verbose: The verbosity level.
      random_seed: Random seed to select the seed edges.
      num_negative_nodes: Number of negative target nodes to sample for each
        edge. If None, use the value from the model's hyperparameters.

    Returns:
      An `evaluation.Evaluation` object containing the evaluation results.
    """

    target_edgeset = self._data.task.target_edgeset
    num_edges = graph.edge_sets[target_edgeset].num_edges()

    if seed_edge_idxs is None:
      seed_edge_idxs = np.arange(num_edges)

    if num_eval_steps is not None and num_eval_steps < len(seed_edge_idxs):
      rng = np.random.default_rng(random_seed)
      seed_edge_idxs = rng.choice(
          seed_edge_idxs, size=num_eval_steps, replace=False
      )

    num_examples = len(seed_edge_idxs)
    if verbose >= 1:
      util.log.info("Evaluating model on %d edges", num_examples)

    num_negative_nodes = (
        num_negative_nodes
        if num_negative_nodes is not None
        else self._data.hparams.num_negative_nodes
    )

    source_sampler, target_sampler = self._build_samplers(graph)

    # Create EdgeNeighborGenerator
    neighbor_generator = edge_neighbor_generator_lib.EdgeNeighborGenerator(
        graph=graph,
        schema=self._data.schema,
        target_edgeset=target_edgeset,
        num_negative_neighbors=num_negative_nodes,
        sampler=source_sampler,
        config=self.data().edge_neighbor_generator,
    )

    # Generate neighbors for all seed edges
    # TODO(gbm): Generate neighbors by batch.
    neighbor_idxs = neighbor_generator.generate(seed_edge_idxs)

    full_src, full_trg = _interleave_positives_and_negatives(
        neighbor_idxs.pos_src_node_idxs,
        neighbor_idxs.pos_trg_node_idxs,
        neighbor_idxs.neg_trg_node_idxs,
    )

    total_mrr = 0.0
    total_hit_at_1 = 0.0
    total_hit_at_5 = 0.0
    total_auc = 0.0

    # Each group contains one positive prediction, followed by
    # num_negative_nodes corresponding predictions.
    examples_per_seed_edge = 1 + num_negative_nodes
    carry_over_probs = np.array([])

    for batch_pred in self.predict_batch(
        graph,
        full_src.tolist(),
        full_trg.tolist(),
        verbose=verbose,
        source_sampler=source_sampler,
        target_sampler=target_sampler,
    ):
      combined_probs = np.concatenate(
          [carry_over_probs, batch_pred.predictions]
      )

      num_total_examples = len(combined_probs)
      num_complete_groups = num_total_examples // examples_per_seed_edge

      if num_complete_groups > 0:
        complete_part_len = num_complete_groups * examples_per_seed_edge
        complete_probs = combined_probs[:complete_part_len]
        carry_over_probs = combined_probs[complete_part_len:]

        pos_probs, neg_probs = _separate_positives_and_negatives(
            complete_probs, examples_per_seed_edge
        )

        ranking_metrics = evaluation.compute_ranking_metrics(
            pos_probs, neg_probs
        )

        total_mrr += float(ranking_metrics["mrr"]) * num_complete_groups
        total_hit_at_1 += (
            float(ranking_metrics["hit_at_1"]) * num_complete_groups
        )
        total_hit_at_5 += (
            float(ranking_metrics["hit_at_5"]) * num_complete_groups
        )
        total_auc += float(ranking_metrics["auc"]) * num_complete_groups
      else:
        carry_over_probs = combined_probs

    if len(carry_over_probs) != 0:
      raise ValueError(
          f"Incomplete group at the end of evaluation: {len(carry_over_probs)}"
          " examples left over. Total number of examples must be a multiple of"
          f" {examples_per_seed_edge}."
      )

    return evaluation.Evaluation(
        num_examples=num_examples,
        mrr=total_mrr / num_examples,
        auc=total_auc / num_examples,
        hit_at={
            1: total_hit_at_1 / num_examples,
            5: total_hit_at_5 / num_examples,
        },
    )
