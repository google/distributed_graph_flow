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

r"""Example of training with TF-GNN.

WARNING: This example does not show GF usage. Instead, it demonstrates the usage 
of the deprecated TF-GNN library with a small wrapper we did in GraphFlow for
comparison purpose.

This is a generic trainer for TF-GNN models, supporting both classification and
regression tasks.

Usage example:

1.  **Run Locally:**
blaze run -c opt //third_party/py/dgf/examples:train_tfgnn \
  -- \
  --train_samples=/cns/iz-d/home/research-graph/public/graphflow_datasets/ogbn_mag_v2/normalized_gnn_samples_d2_w2_sfull/data-000[0-8][0-9]-of-00100.tfrecord.gz \
  --valid_samples=/cns/iz-d/home/research-graph/public/graphflow_datasets/ogbn_mag_v2/normalized_gnn_samples_d2_w2_sfull/data-0009[0-9]-of-00100.tfrecord.gz \
  --graph_schema=/cns/iz-d/home/research-graph/public/graphflow_datasets/ogbn_mag_v2/normalized_gnn_samples_d2_w2_sfull/schema.json \
  --model_dir=gs://goelshreya-ogbn-mag-normalized-samples/model_output \
  --input_training_config=gs://goelshreya-ogbn-mag-normalized-samples/training_config.json \
  --output_metadata_file_path=gs://goelshreya-ogbn-mag-normalized-samples/model_output/output_metadata.json \
  --alsologtostderr

2.  **Run on Borg:**
blaze run -c opt //third_party/py/dgf/examples:train_tfgnn\
    -- \
    --flume_exec_mode=BORG \
    --flume_enable_public_scratch \
    --flume_worker_priority=119 \
    --flume_borg_accounting_charged_user_name=simple-ml-accounting \
    --flume_borg_cells=is \
    --train_samples=/cns/iz-d/home/research-graph/public/graphflow_datasets/ogbn_mag_v2/normalized_gnn_samples_d2_w2_sfull/data-000[0-8][0-9]-of-00100.tfrecord.gz \
    --valid_samples=/cns/iz-d/home/research-graph/public/graphflow_datasets/ogbn_mag_v2/normalized_gnn_samples_d2_w2_sfull/data-0009[0-9]-of-00100.tfrecord.gz \
    --graph_schema=/cns/iz-d/home/research-graph/public/graphflow_datasets/ogbn_mag_v2/normalized_gnn_samples_d2_w2_sfull/schema.json \
    --model_dir=gs://goelshreya-ogbn-mag-normalized-samples/model_output \
    --input_training_config=gs://goelshreya-ogbn-mag-normalized-samples/training_config.json \
    --output_metadata_file_path=gs://goelshreya-ogbn-mag-normalized-samples/model_output/output_metadata.json \
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import dgf
from dgf.src.learning.google.tfgnn import trainer_tfgnn

_SCHEMA = flags.DEFINE_string(
    "graph_schema",
    None,
    "Path to the graph schema file. Expects the GraphSchema format in JSON.",
    required=True,
)

_TRAIN_SAMPLES = flags.DEFINE_string(
    "train_samples",
    None,
    "Path to the training subgraph samples. Accepts a file pattern for sharded"
    " files (e.g., /path/to/data/train-*, /path/to/data.tfrecord). Files must"
    " be in TFRecord format, containing serialized TF-GNN GraphTensor samples.",
    required=True,
)

_VALID_SAMPLES = flags.DEFINE_string(
    "valid_samples",
    None,
    "Path to the validation subgraph samples. Accepts a file pattern for"
    " sharded files (e.g., /path/to/data/valid-*, /path/to/data.tfrecord)."
    " Files must be in TFRecord format, containing serialized TF-GNN"
    " GraphTensor samples.",
    required=False,
)

_MODEL_DIR = flags.DEFINE_string(
    "model_dir",
    None,
    "Directory to save model checkpoints and the final exported model. The"
    " model is exported in TensorFlow SavedModel format, outputting raw"
    " classification logits.",
    required=True,
)

_INPUT_TRAINING_CONFIG = flags.DEFINE_string(
    "input_training_config",
    None,
    "Path to input json file that describes the training config hyper-parameter"
    " dataclass.",
    required=True,
)

_OUTPUT_METADATA_FILE = flags.DEFINE_string(
    "output_metadata_file_path",
    None,
    "Path to output meta-data dataclass stored in json. This file will contain"
    " the training history and other metadata about the training run.",
    required=True,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # TODO(bmayer): Figure out how to get absl.logging.info redirected to stdout,
  # it currently creates ERROR log entries.
  # https://screenshot.googleplex.com/EvP9HF8J5QuyeSN
  logging.info("Model training started!")
  result, output_metadata = trainer_tfgnn.train_tfgnn_model(
      train_samples_path=_TRAIN_SAMPLES.value,
      valid_samples_path=_VALID_SAMPLES.value,
      schema_path=_SCHEMA.value,
      model_dir=_MODEL_DIR.value,
      training_config_path=_INPUT_TRAINING_CONFIG.value,
      output_metadata_path=_OUTPUT_METADATA_FILE.value,
  )
  logging.info("Training finished!")
  logging.info("Final training result: %s", result)
  logging.info("Output Metadata: %s", output_metadata)


if __name__ == "__main__":
  logging.use_python_logging()
  logging.set_verbosity(logging.DEBUG)
  app.run(main)
