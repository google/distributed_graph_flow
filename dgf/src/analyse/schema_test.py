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
from dgf.src.analyse import schema as schema_lib
from dgf.src.data.schema import EdgeSchema
from dgf.src.data.schema import FeatureFormat
from dgf.src.data.schema import FeatureSchema
from dgf.src.data.schema import FeatureSemantic
from dgf.src.data.schema import GraphSchema
from dgf.src.data.schema import NodeSchema


class SchemaTest(absltest.TestCase):

  def test_get_primary_feature_single(self):
    """Schema with one primary feature."""
    schema = NodeSchema(
        features={
            "id": FeatureSchema(
                format=FeatureFormat.BYTES,
                semantic=FeatureSemantic.PRIMARY_ID,
            ),
            "feat1": FeatureSchema(
                format=FeatureFormat.FLOAT_32,
                semantic=FeatureSemantic.NUMERICAL,
            ),
        }
    )
    self.assertEqual(schema_lib.primary_feature("nodes", schema), "id")

  def test_get_primary_feature_none(self):
    """Schema with no primary feature."""
    schema = NodeSchema(
        features={
            "feat1": FeatureSchema(
                format=FeatureFormat.FLOAT_32,
                semantic=FeatureSemantic.NUMERICAL,
            ),
        }
    )
    with self.assertRaisesRegex(ValueError, "No primary feature found"):
      schema_lib.primary_feature("nodes", schema)

  def test_get_primary_feature_multiple(self):
    """Schema with multiple primary features."""
    schema = NodeSchema(
        features={
            "id1": FeatureSchema(
                format=FeatureFormat.BYTES,
                semantic=FeatureSemantic.PRIMARY_ID,
            ),
            "id2": FeatureSchema(
                format=FeatureFormat.BYTES,
                semantic=FeatureSemantic.PRIMARY_ID,
            ),
        }
    )
    with self.assertRaisesRegex(ValueError, "Multiple primary features found"):
      schema_lib.primary_feature("nodes", schema)

  def test_infer_most_likely_primary_key_single(self):
    """Schema with exactly one candidate key."""
    schema = NodeSchema(
        features={
            "id": FeatureSchema(format=FeatureFormat.BYTES),
            "feat1": FeatureSchema(format=FeatureFormat.FLOAT_32),
        }
    )
    self.assertEqual(
        schema_lib.infer_most_likely_primary_key("nodes", schema), "id"
    )

  def test_infer_most_likely_primary_key_multiple(self):
    """Schema with multiple candidate keys."""
    schema = NodeSchema(
        features={
            "id": FeatureSchema(format=FeatureFormat.BYTES),
            "#id": FeatureSchema(format=FeatureFormat.BYTES),
        }
    )
    with self.assertRaisesRegex(
        ValueError, "multiple feature that look like primary key found"
    ):
      schema_lib.infer_most_likely_primary_key("nodes", schema)

  def test_infer_most_likely_primary_key_none(self):
    """Schema with no candidate keys."""
    schema = NodeSchema(
        features={
            "feat1": FeatureSchema(format=FeatureFormat.FLOAT_32),
        }
    )
    with self.assertRaisesRegex(
        ValueError, "no feature look like a primary key"
    ):
      schema_lib.infer_most_likely_primary_key("nodes", schema)

  def test_infer_most_likely_primary_key_or_none_single(self):
    """Schema with exactly one candidate key."""
    schema = NodeSchema(
        features={
            "id": FeatureSchema(
                format=FeatureFormat.BYTES,
            ),
            "feat1": FeatureSchema(
                format=FeatureFormat.FLOAT_32,
            ),
        }
    )
    self.assertEqual(
        schema_lib.infer_most_likely_primary_key_or_none("nodes", schema), "id"
    )

  def test_infer_most_likely_primary_key_or_none_multiple(self):
    """Schema with multiple candidate keys."""
    schema = NodeSchema(
        features={
            "id": FeatureSchema(
                format=FeatureFormat.BYTES,
            ),
            "#id": FeatureSchema(
                format=FeatureFormat.BYTES,
            ),
        }
    )
    with self.assertRaisesRegex(
        ValueError, "multiple feature that look like primary key found"
    ):
      schema_lib.infer_most_likely_primary_key_or_none("nodes", schema)

  def test_infer_most_likely_primary_key_or_none_none(self):
    """Schema with no candidate keys."""
    schema = NodeSchema(
        features={
            "feat1": FeatureSchema(
                format=FeatureFormat.FLOAT_32,
            ),
        }
    )
    self.assertIsNone(
        schema_lib.infer_most_likely_primary_key_or_none("nodes", schema)
    )

  def test_fix_schema_noop(self):
    import copy

    schema = GraphSchema(
        node_sets={
            "nodes": NodeSchema(
                features={
                    "id": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.PRIMARY_ID,
                    )
                }
            )
        },
        edge_sets={
            "edges": EdgeSchema(
                source="nodes",
                target="nodes",
                features={
                    "id": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.PRIMARY_ID,
                    )
                },
            )
        },
    )
    original_schema = copy.deepcopy(schema)
    schema_lib.fix_schema(schema)
    self.assertEqual(schema, original_schema)

  def test_fix_schema_fix_nodeset(self):
    schema = GraphSchema(
        node_sets={
            "nodes": NodeSchema(
                features={
                    "id": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.UNKNOWN,
                    )
                }
            )
        },
        edge_sets={},
    )
    schema_lib.fix_schema(schema)
    self.assertEqual(
        schema.node_sets["nodes"].features["id"].semantic,
        FeatureSemantic.PRIMARY_ID,
    )
    self.assertEqual(
        schema.node_sets["nodes"].features["id"].format,
        FeatureFormat.BYTES,
    )

  def test_fix_schema_fix_edgeset(self):
    schema = GraphSchema(
        node_sets={},
        edge_sets={
            "edges": EdgeSchema(
                source="nodes",
                target="nodes",
                features={
                    "id": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.UNKNOWN,
                    )
                },
            )
        },
    )
    schema_lib.fix_schema(schema)
    self.assertEqual(
        schema.edge_sets["edges"].features["id"].semantic,
        FeatureSemantic.PRIMARY_ID,
    )
    self.assertEqual(
        schema.edge_sets["edges"].features["id"].format,
        FeatureFormat.BYTES,
    )

  def test_fix_schema_nodeset_no_candidate(self):
    schema = GraphSchema(
        node_sets={
            "nodes": NodeSchema(
                features={
                    "feat": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.UNKNOWN,
                    )
                }
            )
        },
        edge_sets={},
    )
    with self.assertRaisesRegex(
        ValueError, "no feature look like a primary key"
    ):
      schema_lib.fix_schema(schema)


if __name__ == "__main__":
  absltest.main()
