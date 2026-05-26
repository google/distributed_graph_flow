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

"""Training model metrics for DGF."""

import dataclasses
from typing import Dict, Optional
import dataclasses_json
from dgf.src.learning import training_configuration

ModelMetadata = training_configuration.TrainingMonitoring


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Metrics:
  """Core model metrics for DGF on each training, validation and test dataset.

  Attributes:
    accuracy: The accuracy of the model.
    loss_name: The name of the loss function.
    loss_value: The loss of the model.
    auc_score: The AUC score of the model.
    precision: The precision of the model.
    recall: The recall of the model.
    f1_score: The F1 score of the model.
    eval_time_in_seconds: The time taken in seconds to evaluate on the dataset.
  """

  accuracy: float
  loss_name: str
  loss_value: float
  auc_score: float
  precision: float
  recall: float
  f1_score: float
  eval_time_in_seconds: Optional[float] = None


DatasetMetricsDict = Dict[str, Metrics]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class TrainedModelResultSet:
  """Container for all metrics and metadata related to a trained model.

  Attributes:
    metadata: Metadata about the training process.
    training_metrics: Metrics evaluated on one or more training datasets.
    validation_metrics: Metrics evaluated on one or more validation datasets.
    test_metrics: Metrics evaluated on one or more test datasets.
  """

  metadata: ModelMetadata = dataclasses.field(default_factory=ModelMetadata)
  training_metrics: DatasetMetricsDict = dataclasses.field(default_factory=dict)
  validation_metrics: DatasetMetricsDict = dataclasses.field(
      default_factory=dict
  )
  test_metrics: DatasetMetricsDict = dataclasses.field(default_factory=dict)
