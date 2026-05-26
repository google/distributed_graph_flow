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

r"""Integration test.
"""

import time
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.plot import network as network_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling.gcp import spanner_graph_sampler as spanner_graph_sampler_lib
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np

InMemoryGraph = in_memory_graph_lib.InMemoryGraph
InMemoryNodeSet = in_memory_graph_lib.InMemoryNodeSet
InMemoryEdgeSet = in_memory_graph_lib.InMemoryEdgeSet


def _toy_schema(has_feat: bool = False):
  features = {
      "id": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
      ),
      "labels": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
      ),
      "year": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      ),
  }
  if has_feat:
    features["feat"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        shape=(128,),
    )
  return schema_lib.GraphSchema(
      node_sets={"nodes": schema_lib.NodeSchema(features=features)},
      edge_sets={
          "edges": schema_lib.EdgeSchema(source="nodes", target="nodes")
      },
  )


class E2ETest(parameterized.TestCase):

  def test_manual_sample_arxiv(self):
    r"""Manual sampling unit test."""
    # Note: The Arxiv spanner graph has a slighly different schema than the
    # one returned by "fetch_ogb_graph".
    #
    # TODO(gbm): Infer the schema from the spanner graph when available:
    # "spanner_graph_in_memory.read_schema_from_spanner_graph"
    schema = _toy_schema()
    spanner_graph_config = {
        "project": "biggraphs-poc",
        "instance": "gcp-gnns",
        "database": "ogbn_arxiv",
        "graph": "ogbn_arxiv",
    }

    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="nodes",
        num_hops=2,
        hop_width=2,
        reverse=True,
    )

    spanner_sampler = spanner_graph_sampler_lib.create_graph_spanner_sampler(
        schema=schema, plan=plan, **spanner_graph_config
    )

    for run_idx, seed_ids in enumerate([["10", "11"]]):
      durations = []
      for _ in range(1):
        start_time = time.time()
        spanner_samples = spanner_sampler.sample(seed_ids)
        end_time = time.time()
        duration = end_time - start_time
        durations.append(duration)
        print(f"Sampling took {duration:.2f} seconds")
      print(f"Average sampling time: {np.mean(durations):.2f} seconds")

      for seed_id, spanner_sample in zip(seed_ids, spanner_samples):
        in_memory_graph_validate_lib.validate_graph(
            spanner_sample, schema, raise_on_warning=False
        )
        self.assertEqual(
            spanner_sample.node_sets["nodes"].features["id"][0].item(),
            seed_id.encode("utf-8"),
        )

        p = network_lib.plot_graph(spanner_sample, schema, features=False)
        p.render(
            f"/tmp/gf/graph_run{run_idx}_seed{seed_id}",
            format="png",
            cleanup=True,
        )


if __name__ == "__main__":
  absltest.main()
