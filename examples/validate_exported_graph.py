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

"""Script to validate an exported GF graph."""

from absl import app
from absl import flags
import dgf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "graph_path",
    None,
    "The path to the exported GF graph.",
    required=True,
)

flags.DEFINE_string(
    "schema_path",
    None,
    "The path to the (override) schema file. If not set, the schema will be"
    " read from the graph directory.",
)

flags.DEFINE_bool(
    "verbose",
    False,
    "Whether to print verbose output.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  override_schema = None
  if FLAGS.schema_path:
    print(f"Loading schema from {FLAGS.schema_path}...")
    override_schema = dgf.io.read_schema(FLAGS.schema_path)

  print(f"Loading graph from {FLAGS.graph_path}...")
  graph, schema = dgf.io.read_graph(
      FLAGS.graph_path, override_schema=override_schema, verbose=FLAGS.verbose
  )

  print("Validating graph...")
  dgf.validate.validate_graph(graph, schema, raise_on_warning=False)
  print("Graph validation successful!")


if __name__ == "__main__":
  app.run(main)
