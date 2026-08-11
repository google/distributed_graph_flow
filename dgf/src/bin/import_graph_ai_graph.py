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

r"""Imports a GraphAI Graph into a DGF Graph.

This binary supports both in-process conversion (good for small graphs) and
Apache Beam distributed conversion.

Usage example:

# In-process conversion
blaze run -c opt //third_party/py/dgf/src/bin:import_graph_ai_graph -- \
  --input=/path/to/graphai_graph \
  --output=/path/to/dgf_graph
"""

import dataclasses
from absl import app
from absl import flags
import apache_beam as beam
import dgf

_INPUT = flags.DEFINE_string(
    "input",
    None,
    "Path to the input GraphAI Graph directory.",
    required=True,
)
_INPUT_CONTAINER = flags.DEFINE_enum(
    "input_container",
    "SSTABLE",
    ["TF_RECORD", "SSTABLE"],
    "The container used by the Graph AI Graph. Can be TF_RECORD or SSTABLE.",
)
_INPUT_RESEARCH_NODE = flags.DEFINE_bool(
    "input_research_node",
    True,
    "If true, the containers use the Research Node format.",
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Path to the output Distributed Graph Flow directory.",
    required=True,
)

_VALIDATE_GRAPH = flags.DEFINE_bool(
    "validate_graph",
    True,
    "If true, validates the graph data with dgf.validate.validate_graph()."
    " Currently, only available for in-process.",
)

_INFER_SEMANTIC = flags.DEFINE_bool(
    "infer_semantic",
    True,
    "If true, tries to determine the semantics of all the features in the graph"
    " (e.g., numerical, categorical).",
)


_RUNNER = flags.DEFINE_string(
    "runner",
    None,
    "How to run the conversion. If not specified, the conversion is done"
    " in-process. If set (can be DirectRunner, DataflowRunner, FlumePython),"
    " run the conversion on Apache Beam.",
)


def run_inprocess():
  graph, schema = dgf.io.read_graphai_hgraph(
      path=_INPUT.value,
      container_type=_INPUT_CONTAINER.value,
      research_node_format=_INPUT_RESEARCH_NODE.value,
  )

  if _INFER_SEMANTIC.value:
    schema = dgf.analyse.infer_schema_semantic(schema)

  print("Schema")
  dgf.print.schema(schema)

  if _VALIDATE_GRAPH.value:
    dgf.validate.validate_graph(graph, schema)

  dgf.io.write_graph(graph, schema, _OUTPUT.value)


def run_beam(runner: str):

  def pipeline(root: beam.Pipeline):
    # Read the graph
    graph = dgf.beam.io.read_graphai_hgraph(
        root,
        path=_INPUT.value,
        container_type=_INPUT_CONTAINER.value,
        research_node_format=_INPUT_RESEARCH_NODE.value,
    )

    if _INFER_SEMANTIC.value:
      graph = dataclasses.replace(
          graph, schema=dgf.analyse.infer_schema_semantic(graph.schema)
      )

    # Write the graph
    dgf.beam.io.write_graph(graph, _OUTPUT.value)

    if _VALIDATE_GRAPH.value:
      print("Graph validation not implemented for Apache Beam")

  runner = dgf.beam.runner_from_name(runner)
  runner.run(pipeline)


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _RUNNER.value is not None:
    # Apache Beam execution
    dgf.beam.program_started(_RUNNER.value)
    run_beam(_RUNNER.value)
  else:
    run_inprocess()


if __name__ == "__main__":
  app.run(main)
