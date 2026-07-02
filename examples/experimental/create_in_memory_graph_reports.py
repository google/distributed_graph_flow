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

"""This example is used to generate reports for homogeneous graphs in memory.

example : 

```shell
## Homogeneous Graph Example : 
blaze run -c opt \
  //third_party/py/dgf/examples:create_in_memory_graph_reports -- \
  --input_graph_path="/cns/iz-d/home/research-graph/public/graphflow_datasets/ogbn_arxiv/raw_gf_graph/" \
  --graph_dataset_name="OGBN ARXIV" \
  --task_type="Node Classification" \
  --num_subgraphs=10 \
  --seed_nodeset="nodes" \
  --class_labels_nodeset="nodes" \
  --class_labels_feature="labels" \
  --color_by_attribute="labels" \
  --node_label_attribute="labels" \
  --output_dir="/tmp/graphflow-reports" \
  --alsologtostderr

## Heterogeneous Graph Example : 
blaze run -c opt \
  //third_party/py/dgf/examples:create_in_memory_graph_reports -- \
  --input_graph_path="/cns/iz-d/home/research-graph/public/graphflow_datasets/ogbn_mag_v2/raw_gf_graph/" \
  --graph_dataset_name="OGBN MAG V2" \
  --task_type="Node Classification" \
  --num_subgraphs=10 \
  --seed_nodeset="paper" \
  --sampling_num_hops=2 \
  --sampling_hop_width=2 \
  --class_labels_nodeset="paper" \
  --class_labels_feature="labels" \
  --output_dir="/tmp/graphflow-reports" \
  --alsologtostderr


```

"""

import logging
import os
from typing import Sequence

from absl import app
from absl import flags
from absl.flags import argparse_flags
import dgf
import numpy as np


def main(argv: Sequence[str]) -> None:

  _ = argv[0]
  argv = argv[1:]
  parser = argparse_flags.ArgumentParser(allow_abbrev=False)
  parser.add_argument(
      "--input_graph_path",
      type=str,
      required=True,
      help="Input path for the graph.",
  )
  parser.add_argument(
      "--graph_dataset_name",
      type=str,
      required=True,
      help="Dataset name.",
  )
  parser.add_argument(
      "--task_type",
      type=str,
      required=True,
      help="Task type.",
  )
  parser.add_argument(
      "--num_subgraphs",
      type=int,
      default=4,
      help="Number of subgraphs to generate.",
  )
  parser.add_argument(
      "--seed_nodeset",
      type=str,
      default=None,
      help="Seed nodeset name.",
  )
  parser.add_argument(
      "--sampling_num_hops",
      type=int,
      default=3,
      help="Number of hops to sample.",
  )
  parser.add_argument(
      "--sampling_hop_width",
      type=int,
      default=5,
      help="Hop width to sample.",
  )
  parser.add_argument(
      "--class_labels_nodeset",
      type=str,
      default=None,
      help=(
          "Nodeset name for class labels. If not provided, assume class labels"
          " are not present."
      ),
  )
  parser.add_argument(
      "--class_labels_feature",
      type=str,
      default=None,
      help=(
          "Feature name for class labels. If not provided, assume class labels"
          " are not present."
      ),
  )
  parser.add_argument(
      "--color_by_attribute",
      type=str,
      default=None,
      help=(
          "Attribute name to color the nodes by. If not provided, nodes will"
          " be colored using class labels."
      ),
  )
  parser.add_argument(
      "--node_label_attribute",
      type=str,
      default=None,
      help=(
          "Attribute name to label the nodes by. If not provided, nodes will"
          " be labeled using their ids."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output path for the reports.",
  )

  known_args, _ = parser.parse_known_args(argv)
  logging.info("Known args: %s", known_args)

  graph, schema = dgf.io.read_graph(known_args.input_graph_path)
  ggt = dgf.analyse.global_graph_topology.get_in_memory_graph_topology(
      graph, schema
  )
  num_classes = np.unique(
      graph.node_sets[known_args.class_labels_nodeset].features[
          known_args.class_labels_feature
      ]
  ).shape[0]

  sampler = dgf.sampling.create_sampler(
      graph=graph,
      schema=schema,
      plan=dgf.sampling.SimpleSamplingConfig(
          seed_nodeset=known_args.seed_nodeset,
          num_hops=known_args.sampling_num_hops,
          hop_width=known_args.sampling_hop_width,
      ),
      num_threads=os.cpu_count() * 2,  # pyrefly: ignore[unsupported-operation]
  )
  num_nodes = graph.node_sets[known_args.seed_nodeset].num_nodes
  sub_graphs = sampler.sample(
      np.random.randint(1, num_nodes, (known_args.num_subgraphs,)).tolist()
  )
  # sub_graphs = np.random.choice(sub_graphs, size=10).tolist()

  payload = dgf.analyse.reports_data_model.GraphStatsPayload(
      dataset_name=known_args.graph_dataset_name,
      task_type=known_args.task_type,
      feature_dimensionality=128,  ## TODO(tewariy): calculate from the graph.
      global_graph_topology=ggt,
      subgraphs=sub_graphs,
      graph_schema=schema,
      color_by_attribute=known_args.color_by_attribute,
      node_label_attribute=known_args.node_label_attribute,
      num_classes=num_classes,
  )
  dgf.analyse.reporter.generate_report(
      payload=payload,
      output_dir=known_args.output_dir,
  )

  logging.info(
      "Report generated successfully for graph dataset %s at %s",
      known_args.graph_dataset_name,
      known_args.output_dir,
  )


if __name__ == "__main__":
  flags.FLAGS.mark_as_parsed()
  app.run(main, flags_parser=lambda x: x)
