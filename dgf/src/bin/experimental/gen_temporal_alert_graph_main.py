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

r"""Script to generate synthetic temporal alert regression graph datasets.
"""

import os
from absl import app
from absl import flags
from absl import logging
from dgf.src.generate import gen_temporal_alert_graph

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "base_output_dir",
    None,
    "Base directory to write the generated synthetic temporal datasets.",
    required=True,
)
flags.DEFINE_integer("seed", 42, "Random seed for graph generation.")
flags.DEFINE_float(
    "neighbor_decay",
    0.8,
    "Probability of adding an additional neighbor in each step (power decay"
    " parameter).",
)
flags.DEFINE_integer(
    "max_hardware_per_alert",
    None,
    "Maximum number of hardware nodes an alert can be connected to (hard"
    " limit).",
)
flags.DEFINE_integer(
    "signal_period",
    7200,
    "Period of the sinusoidal telemetry signal in seconds (default: 7200 for"
    " 2-hour cycle).",
)
flags.DEFINE_string(
    "dataset_name",
    "alert_regression_100k",
    "Name of the dataset subdirectory to generate. If empty, writes directly"
    " to base_output_dir.",
)
flags.DEFINE_integer(
    "num_hardware",
    50000,
    "Number of hardware nodes to generate.",
)
flags.DEFINE_integer(
    "num_alerts",
    50000,
    "Number of alert nodes to generate.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.dataset_name:
    output_path = os.path.join(FLAGS.base_output_dir, FLAGS.dataset_name)
  else:
    output_path = FLAGS.base_output_dir

  logging.info(
      "Generating %s (%d HW, %d Alerts, decay %.2f, max_hw %s, period %ds) to"
      " %s",
      FLAGS.dataset_name or output_path,
      FLAGS.num_hardware,
      FLAGS.num_alerts,
      FLAGS.neighbor_decay,
      FLAGS.max_hardware_per_alert,
      FLAGS.signal_period,
      output_path,
  )
  config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
      num_hardware=FLAGS.num_hardware,
      num_alerts=FLAGS.num_alerts,
      signal_period=FLAGS.signal_period,
      neighbor_decay=FLAGS.neighbor_decay,
      max_hardware_per_alert=FLAGS.max_hardware_per_alert,
      seed=FLAGS.seed,
  )
  gen_temporal_alert_graph.generate_signal_regression_graph(
      path=output_path,
      config=config,
  )
  logging.info("Successfully wrote dataset to %s", output_path)


if __name__ == "__main__":
  app.run(main)
