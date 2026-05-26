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
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
import dataclasses_json
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common as jax_common
from dgf.src.learning.jax.layers import hetero_gnn
from dgf.src.learning.jax.layers import hetero_graph_attention_network
from dgf.src.learning.ten_lines import dataset
from dgf.src.util import filesystem as fs
from dgf.src.util import util
import jax

# The types of graphs supported.
Graph = dataset.Graph

# The type of seed node idxs supported.
SeedNodeIdxs = dataset.SeedNodeIdxs

# Filename in the model saved on disk.
FILENAME_DONE = "DONE"
FILENAME_METADATA = "metadata.json"
FILENAME_DATA = "data.json"


class Architecture(enum.Enum):
  HETEROGENEOUS_MESSAGE_PASSING = "HETEROGENEOUS_MESSAGE_PASSING"
  HETEROGENEOUS_GRAPH_ATTENTION_NETWORK = (
      "HETEROGENEOUS_GRAPH_ATTENTION_NETWORK"
  )


DEFAULT_ARCHITECTURE = Architecture.HETEROGENEOUS_MESSAGE_PASSING


def parse_architecture(architecture: Union[Architecture, str]) -> Architecture:
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


@dataclasses.dataclass
class LogItem:
  """A single log item.

  Attributes:
    step: The current training step.
    metrics: A dictionary of metrics for the current step.
  """

  step: int
  metrics: Dict[str, float]


@dataclasses.dataclass
class TrainingLogs:
  """The logs generated during model training.

  Attributes:
    train: A list of log items for the training dataset.
    valid: A list of log items for the validation dataset.
  """

  train: List[LogItem]
  valid: List[LogItem]


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
  """

  name: str
  version: int = 1  # NOTE: Keep the last version as a default.
  trainig_logs: Optional[TrainingLogs] = None


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
  """

  num_sampling_hops: int = 1
  sampling_width: int = 10
  num_layers: int = 2
  batch_size: int = 32
  max_training_time_seconds: Optional[int] = None
  num_train_steps: Optional[int] = None
  random_seed: int = 42
  node_embedding_dim: int = 64
  learning_rate: float = 0.0005
  opt_weight_decay: float = 0.0001
  dropout: float = 0.1
  message_pooling: str = "sum"
  architecture: Architecture = DEFAULT_ARCHITECTURE


def _auto_detect_timestamp_feature(
    features: schema_lib.FeatureSetSchema, name: str, type_name: str
) -> Optional[str]:
  """Auto-detects a single timestamp feature."""

  ts_features: List[str] = []
  for feat_name, feat in features.items():
    if feat.semantic == schema_lib.FeatureSemantic.TIMESTAMP:
      if feat.format != schema_lib.FeatureFormat.INTEGER_64:
        raise ValueError(
            f"Timestamp feature '{feat_name}' in {type_name} '{name}' must have"
            f" format INTEGER_64. Got {feat.format}."
        )
      ts_features.append(feat_name)
  if len(ts_features) > 1:
    raise ValueError(
        f"Multiple timestamp features found in {type_name} '{name}':"
        f" {ts_features}. Please specify one manually."
    )
  return ts_features[0] if ts_features else None


def _validate_timestamp_feature(
    schema: schema_lib.FeatureSchema,
    feature_name: str,
    container_name: str,
    container_type: Literal["nodeset", "edgeset"],
):
  """Validates that a feature is a valid timestamp feature.

  Args:
    schema: Schema of the feature.
    name: Name of the feature.
    type_name: Name of the nodeset or edgeset.
    type_name: The type of the container, either "nodeset" or "edgeset".
  """

  if (
      schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP
      or schema.format != schema_lib.FeatureFormat.INTEGER_64
  ):
    raise ValueError(
        f"Feature '{feature_name}' in {container_name} '{container_type}' must"
        " have semantic=TIMESTAMP and format=INTEGER_64. Got"
        " semantic={feat.semantic} and format={feat.format}."
    )


def parse_temporal_config(
    schema: schema_lib.GraphSchema,
    timestamp_features: Optional[Dict[str, str]],
    target_nodeset: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
  """Parses the temporal configuration and extracts timestamp features.

  Args:
    schema: The graph schema.
    timestamp_features: Optional; if None, timestamp features are auto-detected.
      Otherwise, a flat dict mapping nodeset/edgeset names to their timestamp
      feature names.
    target_nodeset: The name of the target nodeset.

  Returns:
    A tuple containing (nodeset_timestamp_features, edgeset_timestamp_features)
  """
  nodeset_timestamp_features = {}
  edgeset_timestamp_features = {}

  if timestamp_features is None:
    # Auto-detect TIMESTAMP features
    for name, nodeset in schema.node_sets.items():
      ts_feat = _auto_detect_timestamp_feature(
          nodeset.features, name, "nodeset"
      )
      if ts_feat:
        nodeset_timestamp_features[name] = ts_feat

    for name, edgeset in schema.edge_sets.items():
      ts_feat = _auto_detect_timestamp_feature(
          edgeset.features, name, "edgeset"
      )
      if ts_feat:
        edgeset_timestamp_features[name] = ts_feat

  else:
    # User-provided flat dict
    for name, feat_name in timestamp_features.items():
      is_nodeset = name in schema.node_sets
      is_edgeset = name in schema.edge_sets
      if not is_nodeset and not is_edgeset:
        raise ValueError(
            f"Name '{name}' in temporal configuration is neither a nodeset nor"
            " an edgeset."
        )

      if is_nodeset:
        nodeset = schema.node_sets[name]
        if feat_name not in nodeset.features:
          raise ValueError(
              f"Feature '{feat_name}' not found in nodeset '{name}'"
          )
        feat = nodeset.features[feat_name]
        _validate_timestamp_feature(feat, feat_name, name, "nodeset")
        nodeset_timestamp_features[name] = feat_name

      if is_edgeset:
        edgeset = schema.edge_sets[name]
        if feat_name not in edgeset.features:
          raise ValueError(
              f"Feature '{feat_name}' not found in edgeset '{name}'"
          )
        feat = edgeset.features[feat_name]
        _validate_timestamp_feature(feat, feat_name, name, "edgeset")
        edgeset_timestamp_features[name] = feat_name

  # Constraints validations
  if target_nodeset not in nodeset_timestamp_features:
    raise ValueError(
        f"The target nodeset '{target_nodeset}' must have a timestamp feature."
        " Specify it with the `time_aware` argument (e.g. `time_aware={<target"
        " nodeset>:<timestamp feature>}`), or set one of the feature semantic"
        " as TIMESTAMP (e.g. `schema.node_sets[<target"
        " nodeset>].features[<timestamp feature>].semantic ="
        " dgf.data.FeatureSemantic.TIMESTAMP`)."
    )
  if not edgeset_timestamp_features:
    raise ValueError(
        "At least one edgeset must have a timestamp feature. Specify it with"
        " the `time_aware` argument (e.g. `time_aware={<edgeset>:<timestamp"
        " feature>}`), or set one of the feature semantic as TIMESTAMP (e.g."
        " `schema.edge_sets[<some edgeset>].features[<timestamp"
        " feature>].semantic = dgf.data.FeatureSemantic.TIMESTAMP`)."
    )

  return nodeset_timestamp_features, edgeset_timestamp_features


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
    f.write(model.metadata.to_json(indent=2))

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
    metadata = Metadata.from_json(f.read())

  registered_model = REGISTERED_MODELS.get(metadata.name)
  if registered_model is None:
    raise ValueError(f"Model with name '{metadata.name}' is not registered.")

  with fs.open_read(os.path.join(path, FILENAME_DATA)) as f:
    data = registered_model.data_class.from_json(f.read())

  model = registered_model.model_class(data)  # pytype: disable=not-instantiable
  model.metadata = metadata
  model._internal_load(path)  # pylint: disable=protected-access
  return model


def build_gnn_config(hparams: HParam) -> jax_common.GenericLayer:
  """Creates the GNN layer configuration from the hyper-parameters."""

  if hparams.architecture == Architecture.HETEROGENEOUS_MESSAGE_PASSING:
    return hetero_gnn.HeterogeneousGraphConvolutionConfig(
        dims=hparams.node_embedding_dim,
        dropout_rate=hparams.dropout,
        message_pooling=hparams.message_pooling,
    )

  elif (
      hparams.architecture == Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK
  ):
    return (
        hetero_graph_attention_network.HeterogeneousGraphAttentionNetworkConfig(
            dims=hparams.node_embedding_dim,
            dropout_rate=hparams.dropout,
            message_pooling=hparams.message_pooling,
        )
    )

  else:
    raise NotImplementedError(
        f"Unsupported GNN architecture: {hparams.architecture}"
    )


def register_model(
    model_class: Type[Model], constructor_argument_class: Type[Any]
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
  model_class: Type[Model]
  data_class: Type[Any]


REGISTERED_MODELS: Dict[str, RegisteredModel] = {}


def num_model_weights(model_params: Optional[Any]) -> Dict[str, int]:
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
