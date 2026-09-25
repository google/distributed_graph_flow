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

"""Ten-lines-of-code interface."""

import abc
import dataclasses
import enum
import os
from typing import Any, Literal, TypeAlias
import uuid
import dataclasses_json
from dgf.src.learning import early_stopping_monitor
from dgf.src.learning.jax import common as jax_common
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import hetero_graph_attention_network
from dgf.src.learning.jax.layers import preprocess
from dgf.src.learning.jax.layers import timeseries_cnn
from dgf.src.learning.jax.layers import timeseries_transformer
from dgf.src.learning.ten_lines import dataset
from dgf.src.transform import merge as merge_lib
from dgf.src.util import filesystem as fs
from dgf.src.util import log
from dgf.src.util import util
import jax
import numpy as np
import orbax.checkpoint as ocp

# The types of graphs supported.
Graph = dataset.Graph

# The type of seed node idxs supported in high-level APIs.
SeedNodeIdxs: TypeAlias = list[int] | np.ndarray

# Filename in the model saved on disk.
FILENAME_DONE = "DONE"
FILENAME_METADATA = "metadata.json"
FILENAME_DATA = "data.json"
# Directory containing the model weights as an orbax checkpoint.
FILENAME_PARAMS = "params"

# Training step at which `check_skipped_training_samples` is first called. Large
# enough for the skipped ratio to be meaningful, small enough to fail early
# instead of after a long training. The check is run again at the end of
# training.
SKIPPED_SAMPLES_CHECK_STEP = 2000


class Architecture(enum.Enum):
  HETEROGENEOUS_MESSAGE_PASSING = "HETEROGENEOUS_MESSAGE_PASSING"
  HETEROGENEOUS_GRAPH_ATTENTION_NETWORK = (
      "HETEROGENEOUS_GRAPH_ATTENTION_NETWORK"
  )


DEFAULT_ARCHITECTURE = Architecture.HETEROGENEOUS_MESSAGE_PASSING


def parse_architecture(architecture: Architecture | str) -> Architecture:
  """Parses a string or Architecture enum into an Architecture enum."""
  if isinstance(architecture, Architecture):
    return architecture
  if not isinstance(architecture, str):
    raise TypeError(
        f"Expected Architecture or str, got {type(architecture)}:"
        f" {architecture}"
    )
  arch_lower = architecture.lower()
  if arch_lower in ("hmpnn", "heterogeneous_message_passing"):
    return Architecture.HETEROGENEOUS_MESSAGE_PASSING
  elif arch_lower in ("hgat", "han", "heterogeneous_graph_attention_network"):
    return Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK
  else:
    raise ValueError(f"Unknown architecture: {architecture}")


class TimeseriesEncoder(enum.Enum):
  CNN = "CNN"
  TRANSFORMER = "TRANSFORMER"


DEFAULT_TIMESERIES_ENCODER = TimeseriesEncoder.CNN


def parse_timeseries_encoder(
    timeseries_encoder: TimeseriesEncoder | str,
) -> TimeseriesEncoder:
  """Parses a string or TimeseriesEncoder enum into a TimeseriesEncoder enum."""
  if isinstance(timeseries_encoder, TimeseriesEncoder):
    return timeseries_encoder
  if not isinstance(timeseries_encoder, str):
    raise TypeError(
        f"Expected TimeseriesEncoder or str, got {type(timeseries_encoder)}:"
        f" {timeseries_encoder}"
    )
  try:
    return TimeseriesEncoder[timeseries_encoder.upper()]
  except KeyError as exc:
    raise ValueError(
        f"Unknown timeseries encoder: {timeseries_encoder}. The supported"
        f" values are: {[item.value for item in TimeseriesEncoder]}."
    ) from exc


class TFFunctionInputFormat(enum.Enum):
  """Input format of a model exported with `to_tensorflow_function`.

  Possible values:
    TF_GRAPH: A `dgf.data.TFInMemoryGraph` and the indices of the seed nodes.
    TF_GRAPH_DICT: A `dgf.data.TFInMemoryGraphDict` (i.e. a flat dictionary of
      tensors) and the indices of the seed nodes.
    SERIALIZED_TFGNN_GRAPHS: A 1D string tensor of serialized TF GNN Graph
      Samples (i.e. `tf.train.Example` protos), with one graph sample per
      prediction. The seed node is the first node of the target nodeset.
  """

  TF_GRAPH = "TF_GRAPH"
  TF_GRAPH_DICT = "TF_GRAPH_DICT"
  SERIALIZED_TFGNN_GRAPHS = "SERIALIZED_TFGNN_GRAPHS"


DEFAULT_TF_FUNCTION_INPUT_FORMAT = TFFunctionInputFormat.TF_GRAPH


def parse_tf_function_input_format(
    input_format: TFFunctionInputFormat | str,
) -> TFFunctionInputFormat:
  """Parses a string or TFFunctionInputFormat into a TFFunctionInputFormat."""
  if isinstance(input_format, TFFunctionInputFormat):
    return input_format
  if not isinstance(input_format, str):
    raise TypeError(
        f"Expected TFFunctionInputFormat or str, got {type(input_format)}:"
        f" {input_format}"
    )
  try:
    return TFFunctionInputFormat[input_format.upper()]
  except KeyError as exc:
    raise ValueError(
        f"Unknown input format: {input_format}. The supported values are:"
        f" {[item.value for item in TFFunctionInputFormat]}."
    ) from exc


def resolve_tf_function_input_format(
    input_format: TFFunctionInputFormat | str | None,
    consume_tf_graph_dict: bool | None,
) -> TFFunctionInputFormat:
  """Resolves the input format of `to_tensorflow_function`.

  Handles the deprecated `consume_tf_graph_dict` argument, which is superseded
  by `input_format`.

  Args:
    input_format: The `input_format` argument, or None if not set by the user.
    consume_tf_graph_dict: The deprecated `consume_tf_graph_dict` argument, or
      None if not set by the user.

  Returns:
    The input format to use.
  """
  if consume_tf_graph_dict is None:
    if input_format is None:
      return DEFAULT_TF_FUNCTION_INPUT_FORMAT
    return parse_tf_function_input_format(input_format)

  if input_format is not None:
    raise ValueError(
        "The arguments `input_format` and `consume_tf_graph_dict` cannot be"
        " set at the same time. `consume_tf_graph_dict` is deprecated: use"
        f" input_format="
        f'"{TFFunctionInputFormat.TF_GRAPH_DICT.value}" instead of'
        " consume_tf_graph_dict=True."
    )
  log.warning(
      "The argument `consume_tf_graph_dict` of `to_tensorflow_function` is"
      ' deprecated. Use input_format="%s" (instead of'
      ' consume_tf_graph_dict=True) or input_format="%s" (instead of'
      " consume_tf_graph_dict=False).",
      TFFunctionInputFormat.TF_GRAPH_DICT.value,
      TFFunctionInputFormat.TF_GRAPH.value,
  )
  if consume_tf_graph_dict:
    return TFFunctionInputFormat.TF_GRAPH_DICT
  return TFFunctionInputFormat.TF_GRAPH


@dataclasses.dataclass
class LogItem:
  """A single log item.

  Attributes:
    step: The current training step.
    metrics: A dictionary of metrics for the current step.
  """

  step: int
  metrics: dict[str, float]


@dataclasses.dataclass
class TrainingLogs:
  """The logs generated during model training.

  Attributes:
    train: A list of log items for the training dataset.
    valid: A list of log items for the validation dataset.
    num_train_step: The number of training steps of the final model. Note that
      the training logs might contain more steps if early stopping was used and
      the model was reverted to a previous version.
  """

  train: list[LogItem]
  valid: list[LogItem]
  num_train_step: int


class Model(abc.ABC):
  """A generic model from the 10-lines of code API.

  A model is a high-level, user-facing object that "makes predictions".
  Practically, the model can encapsulate core GNN models, sampler
  configurations, normalization settings, padding configurations, and any other
  data required to run the model on raw user data.

  Each `Model` subclass must provide a unique identifier via its `name()` class
  method. This identifier is used for registration, saving, and restoring
  models.

  A model can be saved on disk with the "model.save(path)" method. A saved and
  reloaded model is exactly equivalent to the original model (no lost
  information; this is not an export). The model data is composed of 3
  artifacts:
  - The "metadata.json" file that contains generic information about this Model
    class.
  - The "data.json" file that contains lightweight, JSON-serializable data
    returned by the "model.data()" method.
  - The content written or read by the abstract _internal_save and
    _internal_load methods. This can be any data, and it is generally suited for
    writing large data chunks, e.g., neural network model weights.
  """

  def __init__(self, data: Any) -> None:
    """Initializes the Model.

    Subclasses should implement their specific initialization logic using the
    provided `data`.

    Args:
      data: A dataclass instance containing JSON-serializable data required to
        construct the model. This is typically the output of the `data()`
        method.
    """
    self.metadata = Metadata(name=self.name())

  @abc.abstractmethod
  def describe(self) -> util.RichDisplay:
    """Text or colab augmented text describing the model."""

  def __repr__(self):
    return (
        f"<{self.__class__.__name__} model. Use `model.describe()` to show"
        " details.>"
    )

  def save(self, path: str) -> None:
    """Saves a model to disk. Can be later reloaded with "load_model".

    Usage example:
      ```
      model = dgf.learning.train_node_model(...)
      model.save("/tmp/my_model")
      loaded_model = dgf.learning.load_model("/tmp/my_model")
      ```

    Args:
      path: The directory path where the model should be saved.
    """
    # TODO(gbm): Add usage example with API path.
    save_model(self, path)

  @classmethod
  @abc.abstractmethod
  def name(cls) -> str:
    """The unique name of the model used for serialization and registration."""

  @abc.abstractmethod
  def data(self) -> Any:
    """Returns a JSON-serializable dataclass instance representing the model's data.

    This data is used as the `data` argument in the model's constructor
    when loading the model from disk.
    """

  @abc.abstractmethod
  def _internal_save(self, path: str) -> None:
    """Saves the model data that is not saved in the constructor argument.

    This method can be used to save large or non-convertible to json data.

    Args:
      path: The directory path where the model data should be saved.
    """

  @abc.abstractmethod
  def _internal_load(self, path: str) -> None:
    """Loads the model data that is not saved in the constructor argument.

    This method can be used to load large or non-JSON-serializable data.

    This method is called by `load_model` after the model object has been
    created. It is not expected to be called when initializing the model object
    after training.

    Args:
      path: The directory path from which the model data should be loaded.
    """


@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class Metadata:
  """Generic metadata for all the models.

  All the attributes except for "name" should have a default value.

  Attributes:
    version: The format version of the saved model metadata.
    name: The registered name of the model. Used to identify the model class.
    trainig_logs: The logs generated during model training.
    uuid: A unique identifier for the model, generated at initialization.
    captured_logs: Info and warning message emited during model's training.
  """

  name: str
  version: int = 1  # NOTE: Keep the last version as a default.
  trainig_logs: TrainingLogs | None = None
  uuid: str | None = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
  captured_logs: list[log.Message] | None = None


# TODO(gbm): Structure / organize / populate
@dataclasses.dataclass
class HParam:
  """Internal hyper-parameter of the model.

  Attributes:
    num_sampling_hops: The number of hops to sample neighbors for each node.
    sampling_width: The number of neighbors to sample at each hop.
    num_layers: The number of layers in the GNN model.
    batch_size: The batch size used for training.
    max_training_time_seconds: The maximum training time in seconds.
    num_train_steps: The maximum number of training steps.
    random_seed: The random seed used for reproducibility.
    node_embedding_dim: The dimension of the node embeddings.
    learning_rate: The learning rate used for training.
    opt_weight_decay: The strength of the weight decay regularization of the
      optimizer. See https://optax.readthedocs.io/en/latest/api/optimizers.html.
    dropout: The dropout rate used for training.
    message_pooling: The pooling method used for aggregating messages in GNNs.
      Supported methods are "sum", "mean", and "max".
    architecture: The architecture of the GNN model.
    timeseries_embedding_dim: The dimension of the embedding computed for each
      timeseries feature group.
    timeseries_encoder: The encoder used to turn each timeseries feature group
      into a `timeseries_embedding_dim` sized embedding.
    early_stopping: The configuration for early stopping. If None, early
      stopping is disabled.
    padding_margin: Relative margin added to observed maximum node and edge
      counts when estimating static graph padding. The margin is estimated from
      the training data and decreases with batch size.
  """

  num_sampling_hops: int = 1
  sampling_width: int = 10
  num_layers: int = 2
  batch_size: int = 32
  max_training_time_seconds: int | None = None
  num_train_steps: int | None = None
  random_seed: int = 42
  node_embedding_dim: int = 64
  learning_rate: float = 0.0005
  opt_weight_decay: float = 0.0001
  dropout: float = 0.1
  message_pooling: str = "sum"
  architecture: Architecture = DEFAULT_ARCHITECTURE
  timeseries_embedding_dim: int = 64
  timeseries_encoder: TimeseriesEncoder = DEFAULT_TIMESERIES_ENCODER
  early_stopping: early_stopping_monitor.EarlyStoppingMonitorConfig | None = (
      None
  )
  padding_margin: float = 0.1


def save_model(model: Model, path: str) -> None:
  """Save the model to disk.

  This is an internal helper function called by `Model.save`. Users should
  generally use `model.save(path)` directly.

  Args:
    model: The Model instance to save.
    path: The directory path where the model should be saved.
  """

  if fs.exists(path):
    fs.rmtree(path)
  fs.makedirs(path)

  with fs.open_write(os.path.join(path, FILENAME_METADATA)) as f:
    f.write(model.metadata.to_json(indent=2))  # pyrefly: ignore[missing-attribute]

  data = model.data()
  if dataclasses.is_dataclass(data) and hasattr(data, "model_params"):
    data_copy = dataclasses.replace(data, model_params=None)
  else:
    data_copy = data

  with fs.open_write(os.path.join(path, FILENAME_DATA)) as f:
    f.write(data_copy.to_json(indent=2))
  model._internal_save(path)  # pylint: disable=protected-access
  with fs.open_write(os.path.join(path, FILENAME_DONE)) as f:
    f.write("")


def load_model(path: str) -> Model:
  """Loads a model previously saved with `model.save()`.

  Usage example:

  ```
    model = dgf.learning.train_node_model(...)
    model.save("/tmp/my_model")
    loaded_model = dgf.learning.load_model("/tmp/my_model")
  ```

  Args:
    path: The directory path where the model was saved.

  Returns:
    The loaded Model instance.
  """
  # Make sure the "DONE" has been written.
  if not fs.exists(os.path.join(path, FILENAME_DONE)):
    raise ValueError(
        f"Model save at {path} is not complete. Missing {FILENAME_DONE}."
    )

  with fs.open_read(os.path.join(path, FILENAME_METADATA)) as f:
    metadata = Metadata.from_json(f.read())  # pyrefly: ignore[missing-attribute]

  registered_model = REGISTERED_MODELS.get(metadata.name)
  if registered_model is None:
    raise ValueError(f"Model with name '{metadata.name}' is not registered.")

  with fs.open_read(os.path.join(path, FILENAME_DATA)) as f:
    data = registered_model.data_class.from_json(f.read())

  model = registered_model.model_class(data)  # pytype: disable=not-instantiable
  model.metadata = metadata
  model._internal_load(path)  # pylint: disable=protected-access
  return model


def save_params(params: Any, path: str) -> None:
  """Saves the model weights in the directory of a saved model."""

  checkpointer = ocp.StandardCheckpointer()
  checkpointer.save(os.path.join(path, FILENAME_PARAMS), params)
  checkpointer.wait_until_finished()


def load_params(path: str) -> Any:
  """Loads the model weights saved with "save_params"."""

  # Note: "ocp.StandardCheckpointer" does not support restore arguments.
  checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
  return checkpointer.restore(
      os.path.join(path, FILENAME_PARAMS),
      args=ocp.args.StandardRestore(
          fallback_sharding=jax.sharding.SingleDeviceSharding(
              jax.local_devices()[0]
          )
      ),
  )


def build_gnn_config(hparams: HParam) -> jax_common.GenericLayer:
  """Creates the GNN layer configuration from the hyper-parameters."""

  if hparams.architecture == Architecture.HETEROGENEOUS_MESSAGE_PASSING:
    return hetero_gnn.HeterogeneousGraphConvolutionConfig(  # pyrefly: ignore[bad-return]
        dims=hparams.node_embedding_dim,
        dropout_rate=hparams.dropout,
        message_pooling=hparams.message_pooling,
    )

  elif (
      hparams.architecture == Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK
  ):
    return hetero_graph_attention_network.HeterogeneousGraphAttentionNetworkConfig(  # pyrefly: ignore[bad-return]
        dims=hparams.node_embedding_dim,
        dropout_rate=hparams.dropout,
        message_pooling=hparams.message_pooling,
    )

  else:
    raise NotImplementedError(
        f"Unsupported GNN architecture: {hparams.architecture}"
    )


def build_timeseries_encoder_config(
    hparams: HParam,
    max_timeseries_len: int,
) -> preprocess.TimeseriesEncoderConfig:
  """Creates the timeseries encoder configuration from the hyper-parameters."""

  if hparams.timeseries_encoder == TimeseriesEncoder.CNN:
    return timeseries_cnn.TimeseriesCNNEncoderConfig(
        out_dim=hparams.timeseries_embedding_dim,
        dropout_rate=hparams.dropout,
    )

  elif hparams.timeseries_encoder == TimeseriesEncoder.TRANSFORMER:
    # `dims` tracks `timeseries_embedding_dim` so that the blocks are not
    # bottlenecked relative to the embedding they produce, mirroring how
    # `build_gnn_config` ties the GNN `dims` to `node_embedding_dim`. This means
    # `timeseries_embedding_dim` must be divisible by the encoder's `num_heads`,
    # and the resulting head dimension must be even because RoPE rotates
    # features in pairs, i.e. `timeseries_embedding_dim` must be a multiple of
    # `2 * num_heads` (8 with the default `num_heads=4`).
    num_heads = (
        timeseries_transformer.TimeseriesTransformerEncoderConfig.num_heads
    )
    if hparams.timeseries_embedding_dim % (2 * num_heads) != 0:
      raise ValueError(
          "With timeseries_encoder=TRANSFORMER, timeseries_embedding_dim must"
          f" be a multiple of 2 * num_heads = {2 * num_heads}, got"
          f" {hparams.timeseries_embedding_dim}."
      )
    return timeseries_transformer.TimeseriesTransformerEncoderConfig(
        out_dim=hparams.timeseries_embedding_dim,
        dims=hparams.timeseries_embedding_dim,
        dropout_rate=hparams.dropout,
        max_timeseries_len=max_timeseries_len,
    )

  else:
    raise NotImplementedError(
        f"Unsupported timeseries encoder: {hparams.timeseries_encoder}"
    )


def register_model(
    model_class: type[Model], constructor_argument_class: type[Any]
) -> None:
  """Registers a model class.

  Args:
    model_class: The class of the model, inheriting from Model.
    constructor_argument_class: A dataclass type whose instances are returned by
      "_constructor_argument" and used to construct the model. This class should
      be JSON-serializable.
  """
  name = model_class.name()
  if name in REGISTERED_MODELS:
    raise ValueError(f"Model with name '{name}' is already registered.")
  REGISTERED_MODELS[name] = RegisteredModel(
      model_class, constructor_argument_class
  )


@dataclasses.dataclass
class RegisteredModel:
  model_class: type[Model]
  data_class: type[Any]


REGISTERED_MODELS: dict[str, RegisteredModel] = {}


def num_model_weights(model_params: Any | None) -> dict[str, int]:
  """Returns a dictionary of the type and number of weights of the model.

  Example:
    {"float32": 435246, "int16": 345345}
  """
  if model_params is None:
    return {}

  weights = {}
  for leaf in jax.tree_util.tree_leaves(model_params):
    if hasattr(leaf, "dtype") and hasattr(leaf, "size"):
      dtype_name = str(leaf.dtype)
      weights[dtype_name] = weights.get(dtype_name, 0) + leaf.size
  return weights


def check_number_of_seeds(
    batch_size: int,
    num_training: int | None,
    num_validation: int | None,
    key: Literal["node", "edge"],
):
  """Checks if the number of seed nodes is sufficient for the given batch size.

  Args:
    batch_size: The batch size used for training.
    num_training: The number of seed nodes in the training set, or None if not
      applicable.
    num_validation: The number of seed nodes in the validation set, or None if
      not applicable.
    key: The type of seed being checked, either "node" or "edge".

  Raises:
    ValueError: If the number of seed nodes is smaller than the batch size
      for either the training or validation set.
  """
  if num_training is not None and num_training < batch_size:
    raise ValueError(
        f"The number of training seed nodes ({num_training}) is smaller than"
        f" the batch size ({batch_size}). Increase the number of training seed"
        f" {key}s or decrease the batch size."
    )

  if num_validation is not None and num_validation < batch_size:
    raise ValueError(
        f"The number of validation seed nodes ({num_validation}) is smaller"
        f" than the batch size ({batch_size}). Increase the number of"
        f" validation seed {key}s or decrease the batch size."
    )


def check_skipped_training_samples(
    num_skipped_samples: int,
    num_generated_samples: int,
    max_skipped_ratio: float = 0.1,
) -> None:
  """Fails if too many training samples were skipped.

  Args:
    num_skipped_samples: Number of training samples skipped due to padding
      overflow.
    num_generated_samples: Number of training samples successfully generated.
    max_skipped_ratio: Maximum allowed ratio of skipped samples over total
      attempted samples before raising an error.

  Raises:
    merge_lib.InsufficientPaddingError: If the ratio of skipped training
      samples exceeds `max_skipped_ratio`.
  """
  total_samples = num_skipped_samples + num_generated_samples
  if total_samples == 0:
    return
  skipped_ratio = num_skipped_samples / total_samples
  if skipped_ratio > max_skipped_ratio:
    raise merge_lib.InsufficientPaddingError(
        f"Skipped {num_skipped_samples} out of {total_samples} training"
        f" samples ({skipped_ratio:.1%}) due to insufficient padding, which"
        f" exceeds the maximum allowed threshold of {max_skipped_ratio:.1%}."
        " Consider increasing `padding_margin`."
    )


def log_jax_backend(verbose: int = 2) -> None:
  """Logs the active JAX backend and issues a warning if running on CPU.

  Args:
    verbose: The verbosity level. If >= 2, logs an info message with the
      backend.
  """
  backend = jax.default_backend()
  if verbose >= 2:
    log.info("Using %s JAX backend", backend)

  if backend.lower() == "cpu":
    log.warning(
        "Using CPU JAX backend. Training will be slow. Consider using a GPU"
        " or TPU."
    )
