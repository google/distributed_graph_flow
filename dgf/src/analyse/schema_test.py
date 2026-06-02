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

from unittest import mock

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

  def test_fix_schema_create_pound_id_fallback(self):
    schema = GraphSchema(
        node_sets={
            "nodes": NodeSchema(
                features={
                    "feat": FeatureSchema(
                        format=FeatureFormat.FLOAT_32,
                        semantic=FeatureSemantic.UNKNOWN,
                    )
                }
            )
        },
        edge_sets={},
    )
    schema_lib.fix_schema(schema, create_pound_id_as_fall_back=True)
    self.assertIn("#id", schema.node_sets["nodes"].features)
    self.assertEqual(
        schema.node_sets["nodes"].features["#id"].semantic,
        FeatureSemantic.PRIMARY_ID,
    )
    self.assertEqual(
        schema.node_sets["nodes"].features["#id"].format,
        FeatureFormat.BYTES,
    )

  def test_fix_schema_fix_suspicious_shape(self):
    schema = GraphSchema(
        node_sets={
            "nodes": NodeSchema(
                features={
                    "id": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.PRIMARY_ID,
                        shape=(None, 1),
                    ),
                    "feat": FeatureSchema(
                        format=FeatureFormat.FLOAT_32,
                        semantic=FeatureSemantic.NUMERICAL,
                        shape=(None, 10),
                    ),
                }
            )
        },
        edge_sets={},
    )
    schema_lib.fix_schema(schema, fix_shape=True)
    self.assertEqual(schema.node_sets["nodes"].features["id"].shape, ())
    self.assertEqual(schema.node_sets["nodes"].features["feat"].shape, (10,))

  def test_infer_schema_semantic(self):

    schema = GraphSchema(
        node_sets={
            "nodes": NodeSchema(
                features={
                    # Bytes -> Categorical
                    "f_bytes": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.UNKNOWN,
                    ),
                    # Bool -> Categorical
                    "f_bool": FeatureSchema(
                        format=FeatureFormat.BOOL,
                        semantic=FeatureSemantic.UNKNOWN,
                    ),
                    # Float -> Numerical
                    "f_float": FeatureSchema(
                        format=FeatureFormat.FLOAT_32,
                        semantic=FeatureSemantic.UNKNOWN,
                    ),
                    # Int -> Numerical
                    "f_int": FeatureSchema(
                        format=FeatureFormat.INTEGER_64,
                        semantic=FeatureSemantic.UNKNOWN,
                    ),
                    # Int with "is_" -> Categorical
                    "is_active": FeatureSchema(
                        format=FeatureFormat.INTEGER_32,
                        semantic=FeatureSemantic.UNKNOWN,
                    ),
                    # Starts with "#" -> Ignored
                    "#id": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.UNKNOWN,
                    ),
                    # Already set -> Ignored
                    "already_set": FeatureSchema(
                        format=FeatureFormat.FLOAT_32,
                        semantic=FeatureSemantic.EMBEDDING,
                    ),
                }
            )
        },
        edge_sets={
            "edges": EdgeSchema(
                source="nodes",
                target="nodes",
                features={
                    "f_bytes_edge": FeatureSchema(
                        format=FeatureFormat.BYTES,
                        semantic=FeatureSemantic.UNKNOWN,
                    ),
                },
            )
        },
    )

    inferred_schema = schema_lib.infer_schema_semantic(schema)

    node_features = inferred_schema.node_sets["nodes"].features
    self.assertEqual(
        node_features["f_bytes"].semantic, FeatureSemantic.CATEGORICAL
    )
    self.assertEqual(
        node_features["f_bool"].semantic, FeatureSemantic.CATEGORICAL
    )
    self.assertEqual(
        node_features["f_float"].semantic, FeatureSemantic.NUMERICAL
    )
    self.assertEqual(node_features["f_int"].semantic, FeatureSemantic.NUMERICAL)
    self.assertEqual(
        node_features["is_active"].semantic, FeatureSemantic.CATEGORICAL
    )
    self.assertEqual(node_features["#id"].semantic, FeatureSemantic.UNKNOWN)
    self.assertEqual(
        node_features["already_set"].semantic, FeatureSemantic.EMBEDDING
    )

    edge_features = inferred_schema.edge_sets["edges"].features
    self.assertEqual(
        edge_features["f_bytes_edge"].semantic, FeatureSemantic.CATEGORICAL
    )

  def test_infer_schema_semantic_cannot_infer_raise(self):
    class FakeFormat:

      def is_numerical(self):
        return False

      def is_integer(self):
        return False

    schema = GraphSchema(
        node_sets={
            "nodes": NodeSchema(
                features={
                    "f_weird": FeatureSchema(
                        format=FakeFormat(),
                        semantic=FeatureSemantic.UNKNOWN,
                    ),
                }
            )
        },
        edge_sets={},
    )
    with self.assertRaisesRegex(
        ValueError, "Could not infer semantic for feature 'f_weird'"
    ):
      schema_lib.infer_schema_semantic(schema, raise_on_error=True)

  @mock.patch("dgf.src.analyse.schema.log.warning")
  def test_infer_schema_semantic_cannot_infer_warn(self, mock_warning):
    class FakeFormat:

      def is_numerical(self):
        return False

      def is_integer(self):
        return False

    schema = GraphSchema(
        node_sets={
            "nodes": NodeSchema(
                features={
                    "f_weird": FeatureSchema(
                        format=FakeFormat(),
                        semantic=FeatureSemantic.UNKNOWN,
                    ),
                }
            )
        },
        edge_sets={},
    )
    schema_lib.infer_schema_semantic(schema, raise_on_error=False)
    mock_warning.assert_called_once()
    self.assertIn("Could not infer semantic", mock_warning.call_args[0][0])


if __name__ == "__main__":
  absltest.main()
