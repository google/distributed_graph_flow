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

"""Test the minhash sketching utilities."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_test_util
from dgf.src.sketching import minhash
from dgf.src.util import test_util


DataSketchMinHashCombiner = minhash.DatasSketchMinHashCombiner


def fake_barbell_flat_edges():
  """Undirected barbell edges."""
  return [
      (b"A", b"B"),
      (b"A", b"C"),
      (b"B", b"C"),
      (b"C", b"D"),
      (b"D", b"E"),
      (b"D", b"F"),
      (b"E", b"F"),
  ]


def fake_barbell_adjacency_list():
  return {
      b"A": [b"B", b"C"],
      b"B": [b"C"],
      b"C": [b"D"],
      b"D": [b"E", b"F"],
      b"E": [b"F"],
  }


class DatasSketchMinHashCombiner(absltest.TestCase):

  def test_give_me_a_name(self):
    num_perm = 4
    adjacency_lists = fake_barbell_adjacency_list()
    keyed_hashers = {
        k: minhash.MinHash(num_perm=4) for k in adjacency_lists.keys()
    }

    for k, v in adjacency_lists.items():
      for vv in v:
        keyed_hashers[k].update(vv)
    expected_sketches = [
        (k, keyed_hashers[k].digest().tolist()) for k in adjacency_lists.keys()
    ]

    with test_pipeline.TestPipeline() as p:
      # [N, num_perm]
      sketches = (
          p
          | "CreateEdges" >> beam.Create(fake_barbell_flat_edges())
          | "MinHashSketch"
          >> beam.CombinePerKey(DataSketchMinHashCombiner(num_perm=num_perm))
          | "Digest" >> beam.Map(lambda x: (x[0], x[1].digest().tolist()))
      )

      beam_test_util.assert_that(
          sketches,
          beam_test_util.equal_to(
              expected_sketches,
              equals_fn=lambda x, y: (
                  x[0] == y[0] and test_util.are_equal(x[1], y[1])
              ),
          ),
      )


if __name__ == "__main__":
  absltest.main()
