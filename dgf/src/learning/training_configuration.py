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

"""Training configuration parameters for the model."""

import dataclasses
import enum
from typing import Optional
import dataclasses_json


class TaskType(enum.Enum):
  """Defines the type of training task."""

  # TODO(goelshreya): Handle in case of not lowercase.
  CLASSIFICATION = "classification"
  REGRESSION = "regression"
  LINK_PREDICTION = "link_prediction"


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ModelTrainingHyperparameters:
  """Defines the model training hyperparameters.

  Attributes:
    nodes_hidden_dim: The number of hidden units in the nodes.
    activation: The activation function to use in the nodes.
    categorical_embedding_dim: The embedding dimension for categorical features.
    message_dim: The dimension of the messages computed transiently on each edge
      in the GraphUpdate layer.
    num_graph_updates: The number of graph updates to apply.
    simple_conv_reduce_type: The type of reduction to use in the simple conv.
    state_dropout_rate: The dropout rate applied to state in GraphUpdate layer.
    l2_regularization: The coefficient of L2 regularization to use in the model.
    normalization_type: The type of normalization of output node states.
    next_state_type: The type of next state to use in the model.
  """

  nodes_hidden_dim: int = 64
  activation: str = "relu"
  categorical_embedding_dim: int = 64
  message_dim: int = 64
  num_graph_updates: int = 1
  simple_conv_reduce_type: str = "mean"
  state_dropout_rate: float = 0.2
  l2_regularization: float = 1e-5
  normalization_type: str = "layer"
  next_state_type: str = "residual"


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class TrainingConfig:
  """Configuration for the model training process.

  Attributes:
    model_name: The name of the model.
    num_epochs: Number of epochs the model was trained.
    num_steps_per_epoch: The number of training steps per epoch. If unspecified,
      epochs are at `tf.data.Dataset` end.
    validation_steps: The number of validation steps per epoch.
    batch_size: The batch size used for training.
    learning_rate: Final learning rate or initial learning rate.
    optimizer: The optimizer used for training.
    hyperparameters: The other relevant hyperparameters used for training.
    task_type: The task type used for training.
    label_node_set: The node set containing the label feature.
    label_feature: The name of the label feature.
  """
  model_name: str = "default"
  num_epochs: int = 10
  num_steps_per_epoch: Optional[int] = None
  validation_steps: Optional[int] = None
  batch_size: int = 32
  learning_rate: float = 0.001
  optimizer: str = "adam"
  hyperparameters: ModelTrainingHyperparameters = dataclasses.field(
      default_factory=ModelTrainingHyperparameters
  )
  task_type: TaskType = TaskType.CLASSIFICATION
  label_node_set: Optional[str] = None
  label_feature: Optional[str] = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class TrainingMonitoring:
  """Monitoring configuration for the model training process.

  Attributes:
    training_config: The training configuration used for the model.
    total_training_time_in_seconds: The total time taken in seconds to train
      the model.
  """
  training_config: TrainingConfig = dataclasses.field(
      default_factory=TrainingConfig
  )
  total_training_time_in_seconds: Optional[float] = None

