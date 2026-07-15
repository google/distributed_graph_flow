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

"""Loads a graph in memory and validates it.

Usage example:

# Pip
dgf-validate-graph --path=...
"""

from absl import app
from absl import flags
import dgf

FLAGS = flags.FLAGS

flags.DEFINE_string("path", None, "Path to the DGF graph.", required=True)
flags.DEFINE_integer("verbose", 1, "Verbose level.")
flags.DEFINE_bool(
    "raise_on_warning", False, "Whether to raise an error on warnings."
)
flags.DEFINE_bool(
    "raise_on_error", True, "Whether to raise an error on errors."
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(f"Loading graph from {FLAGS.path}...")
  graph, schema = dgf.io.read_graph(FLAGS.path, verbose=FLAGS.verbose)

  print("Validating graph...")
  dgf.validate.validate_graph(
      graph,
      schema,
      raise_on_warning=FLAGS.raise_on_warning,
      raise_on_error=FLAGS.raise_on_error,
  )
  print("Graph validation successful!")


if __name__ == "__main__":
  app.run(main)
