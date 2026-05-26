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

import dgf
import numpy as np

schema = dgf.data.GraphSchema(
    node_sets={
        "nodeset_1": dgf.data.NodeSchema(
            features={
                "f1": dgf.data.FeatureSchema(
                    format=dgf.data.FeatureFormat.BYTES,
                    semantic=dgf.data.FeatureSemantic.CATEGORICAL,
                ),
                "f2": dgf.data.FeatureSchema(
                    format=dgf.data.FeatureFormat.FLOAT_64
                ),
            }
        ),
        "nodeset_2": dgf.data.NodeSchema(),
    },
    edge_sets={
        "edgeset_1": dgf.data.EdgeSchema(
            source="nodeset_1", target="nodeset_1"
        ),
        "edgeset_2": dgf.data.EdgeSchema(
            source="nodeset_1", target="nodeset_2"
        ),
    },
)

graph = dgf.data.InMemoryGraph(
    node_sets={
        "nodeset_1": dgf.data.InMemoryNodeSet(
            num_nodes=3,
            features={
                "f1": np.array([b"A", b"B", b"C"]),
                "f2": np.array([0.1, 0.5, 0.2]),
            },
        ),
        "nodeset_2": dgf.data.InMemoryNodeSet(
            num_nodes=2,
        ),
    },
    edge_sets={
        "edgeset_1": dgf.data.InMemoryEdgeSet(
            adjacency=np.array([
                [0, 0, 1],
                [1, 1, 2],
            ])
        ),
        "edgeset_2": dgf.data.InMemoryEdgeSet(
            adjacency=np.array([
                [0, 1],
                [0, 1],
            ])
        ),
    },
)

dgf.validate.validate_graph(graph, schema, raise_on_warning=False)

sampling_config = dgf.sampling.SimpleSamplingConfig(
    seed_nodeset="nodeset_1",
    num_hops=2,
    hop_width=3,
    reverse=True,
)
sampler = dgf.sampling.create_sampler(
    graph, sampling_config, schema, batch_size=8
)
sample = sampler.sample(seed_node_idxs=0)
dgf.plot.plot_graph(sample, schema, features=False)
