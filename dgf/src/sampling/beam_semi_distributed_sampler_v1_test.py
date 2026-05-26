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

import copy
import functools
import logging
import os
import tempfile
from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
from dgf.src.io import hgraph_in_beam
from dgf.src.io import tf_graph_sample as tf_graph_sample_lib
from dgf.src.sampling import beam_semi_distributed_sampler
from dgf.src.sampling import beam_semi_distributed_sampler_v1
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util


test_util.disable_diff_truncation()


class BeamSemiDistributedSamplerTest(absltest.TestCase):

  def test_sample_with_beam_semi_distributed_sampler(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      path = os.path.join(tmpdir, "hgraph")
      gen_test_graph.generate_hgraph(path)
      schema = gen_test_graph.generate_schema()

      # Sampling configuration
      sampling_config = config_lib.SimpleSamplingConfig(
          seed_nodeset="n1", num_hops=2
      )
      plan = config_lib.simple_sampling_config_to_sampling_plan(
          sampling_config, schema
      )

      with test_pipeline.TestPipeline() as p:
        # Generate some samples (with the beam sampler).
        graph = hgraph_in_beam.read_graphai_hgraph(p, path)
        seeds = beam_semi_distributed_sampler.extract_beam_nodes_ids(
            graph, plan.root.nodeset
        )
        samples = beam_semi_distributed_sampler_v1.sample_with_beam_semi_distributed_sampler(
            graph, plan, seeds=seeds, debug_sampling=True
        )
        _ = samples | "Generated samples" >> beam.Map(print)

        # Write the samples to file (not checked; just to make sure the
        # signature is correct).
        sample_path = os.path.join(tmpdir, "samples@*")

        schema_no_features = copy.deepcopy(schema)
        # TODO(gbm): Don't remove feature when the semi-distributed sampler can
        # generate them.
        # TODO(gbm): Replace with a simple schema filter.
        for nodeset in schema_no_features.node_sets.values():
          to_removes = []
          for feature in nodeset.features:
            if not feature.startswith("#"):
              to_removes.append(feature)
          for to_remove in to_removes:
            del nodeset.features[to_remove]

        tf_graph_sample_lib.write_tfgnn_graphs_beam(
            samples, sample_path, schema_no_features
        )

        # Generate some samples with the in-process sampler.
        in_memory_sampler = in_memory_sampler_lib.create_sampler(
            gen_test_graph.generate_in_memory_graph(node_ids=True),
            plan,
            schema=gen_test_graph.generate_schema(node_ids=True),
            batch_size=1,
            debug_sampling=True,
            return_features=False,
            return_node_idxs=True,
        )

        expected_samples = [
            (b"1", in_memory_sampler.sample(0)),
            (b"2", in_memory_sampler.sample(1)),
        ]
        logging.info("Expected samples:\n%s", expected_samples)
        util.assert_that(
            samples,
            util.equal_to(
                expected_samples,
                equals_fn=functools.partial(test_util.are_equal, abs_tol=0.001),
            ),
        )


if __name__ == "__main__":
  absltest.main()
