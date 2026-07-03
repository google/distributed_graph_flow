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

from absl.testing import absltest
from dgf.src.data import schema as schema_lib
from dgf.src.data import schema_ext as lib
from dgf.src.util import gen_test_graph


class SchemaCC(absltest.TestCase):

  def test_parse_schema(self):
    schema = gen_test_graph.generate_schema(node_ids=True)
    self.assertEqual(
        lib.ParseAndDebugPrintSchema(schema),
        """\
GraphSchema(nodesets=[
  Nodeset(name='n1', features=[
    Feature(name='#id', shape=[], format=BYTES),
    Feature(name='f1', shape=[1], format=BYTES),
    Feature(name='f2', shape=[2], format=FLOAT_32)
  ]),
  Nodeset(name='n2', features=[
    Feature(name='#id', shape=[], format=INTEGER_64),
    Feature(name='f3', shape=[], format=INTEGER_64),
    Feature(name='f4', shape=[], format=INTEGER_64),
    Feature(name='f5', shape=[None], format=INTEGER_64)
  ])
], edgesets=[
  Edgeset(name='e1', source_nodeset=0, target_nodeset=0, features=[]),
  Edgeset(name='e2', source_nodeset=0, target_nodeset=1, features=[])
])""",
    )

  def test_parse_temporal_schema(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "#id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    ),
                    "#creation_time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        shape=(None,),
                        is_timeseries=True,
                    ),
                    "f1_seq": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        shape=(None,),
                        is_timeseries=True,
                        timestamps="time",
                    ),
                    "f2_seq": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(None, 4),
                        is_timeseries=True,
                        timestamps="time",
                    ),
                }
            ),
            "n2": schema_lib.NodeSchema(
                features={
                    "#id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    ),
                    "sensor_ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.TIMESERIES,
                        shape=(20, 8),
                        is_timeseries=True,
                    ),
                }
            ),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n2",
                features={
                    "edge_time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        shape=(None,),
                        is_timeseries=True,
                    ),
                    "edge_val": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        shape=(None,),
                        is_timeseries=True,
                        timestamps="edge_time",
                    ),
                },
            )
        },
    )
    self.assertEqual(
        lib.ParseAndDebugPrintSchema(schema),
        """\
GraphSchema(nodesets=[
  Nodeset(name='n1', features=[
    Feature(name='#creation_time', shape=[], format=INTEGER_64),
    Feature(name='#id', shape=[], format=BYTES),
    Feature(name='f1_seq', shape=[None], format=FLOAT_32, is_timeseries=true, timestamps='time'),
    Feature(name='f2_seq', shape=[None, 4], format=FLOAT_32, is_timeseries=true, timestamps='time'),
    Feature(name='time', shape=[None], format=INTEGER_64, is_timeseries=true)
  ]),
  Nodeset(name='n2', features=[
    Feature(name='#id', shape=[], format=INTEGER_64),
    Feature(name='sensor_ts', shape=[20, 8], format=FLOAT_32, is_timeseries=true)
  ])
], edgesets=[
  Edgeset(name='e1', source_nodeset=0, target_nodeset=1, features=[
    Feature(name='edge_time', shape=[None], format=INTEGER_64, is_timeseries=true),
    Feature(name='edge_val', shape=[None], format=INTEGER_32, is_timeseries=true, timestamps='edge_time')
  ])
])""",
    )


if __name__ == "__main__":
  absltest.main()
