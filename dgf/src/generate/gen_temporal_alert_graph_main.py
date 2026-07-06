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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  base_dir = FLAGS.base_output_dir

  datasets = [
      ("alert_regression_100k", 50000, 50000),
  ]

  for name, num_hw, num_alerts in datasets:
    output_path = os.path.join(base_dir, name)
    logging.info(
        "Generating %s (%d HW, %d Alerts, decay %.2f) to %s",
        name,
        num_hw,
        num_alerts,
        FLAGS.neighbor_decay,
        output_path,
    )
    config = gen_temporal_alert_graph.TemporalAlertGraphConfig(
        num_hardware=num_hw,
        num_alerts=num_alerts,
        neighbor_decay=FLAGS.neighbor_decay,
        seed=FLAGS.seed,
    )
    gen_temporal_alert_graph.generate_signal_regression_graph(
        path=output_path,
        config=config,
    )
    logging.info("Successfully wrote %s to %s", name, output_path)


if __name__ == "__main__":
  app.run(main)
