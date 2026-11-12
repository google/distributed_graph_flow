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
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
from dgf.src.validate import validate as validate_lib
import numpy as np

Issue = validate_lib.Issue

test_util.disable_diff_truncation()


def valid_graph() -> (
    tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
):
  graph = gen_test_graph.generate_in_memory_graph(
      node_ids=True, edge_ids=True, variable_length=True
  )
  schema = gen_test_graph.generate_schema(
      node_ids=True,
      edge_ids=True,
      variable_length=True,
      semantic=True,
  )
  return graph, schema


class InMemoryGraphTest(absltest.TestCase):

  def test_valid(self):
    graph, schema = valid_graph()
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(issues, [])

  def test_warning(self):
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids=False, edge_ids=False, variable_length=True
    )
    schema = gen_test_graph.generate_schema(
        node_ids=False,
        edge_ids=False,
        variable_length=True,
        semantic=True,
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.warning(
                "The nodeset 'n1' schema has no primary feature, i.e., a"
                " feature with semantic=PRIMARY_ID. You won't be able to"
                " serialize it. Recommended actions: Set it manually (e.g.,"
                " `schema.node_sets['n1'].features[<some column>].semantic ="
                " dgf.data.FeatureSemantic.PRIMARY_ID`) or use `schema ="
                " dgf.validate.fix_schema(schema)`."
            ),
            Issue.warning(
                "The nodeset 'n2' schema has no primary feature, i.e., a"
                " feature with semantic=PRIMARY_ID. You won't be able to"
                " serialize it. Recommended actions: Set it manually (e.g.,"
                " `schema.node_sets['n2'].features[<some column>].semantic ="
                " dgf.data.FeatureSemantic.PRIMARY_ID`) or use `schema ="
                " dgf.validate.fix_schema(schema)`."
            ),
        ],
    )

  def test_missing_nodeset(self):
    graph, schema = valid_graph()
    del graph.node_sets["n1"]
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues, [Issue.error("The graph is missing the nodeset 'n1'")]
    )

  def test_missing_feature(self):
    graph, schema = valid_graph()
    del graph.node_sets["n1"].features["f1"]
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues, [Issue.error("Missing feature 'f1' in nodeset 'n1'.")]
    )

  def test_wrong_feature_type(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features[
        "f1"
    ].format = schema_lib.FeatureFormat.INTEGER_32
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'f1' in nodeset 'n1' has dtype |S4 (i.e., <class"
                " 'numpy.bytes_'>), but the schema expects format"
                " <FeatureFormat.INTEGER_32: 'INTEGER_32'>, which corresponds"
                " to dtype <class 'numpy.int32'>"
            )
        ],
    )

  def test_wrong_shape(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["f1"].shape = (2, 3, 4)
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'f1' in nodeset 'n1' has shape (2, 1), but the"
                " schema expects a shape compatible with (2, 3, 4) (i.e., one"
                " more dimension)."
            )
        ],
    )

  def test_non_existing_source(self):
    graph, schema = valid_graph()
    schema.edge_sets["e1"].source = "non_existing"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The edgeset 'e1' refers to a source nodeset 'non_existing'"
                " which is not defined in the graph schema's node_sets."
            )
        ],
    )

  def test_non_existing_target(self):
    graph, schema = valid_graph()
    schema.edge_sets["e1"].target = "non_existing"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The edgeset 'e1' refers to a target nodeset 'non_existing'"
                " which is not defined in the graph schema's node_sets."
            )
        ],
    )

  def test_out_bound_source(self):
    graph, schema = valid_graph()
    graph.edge_sets["e1"].adjacency[:] += 1
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The edgeset 'e1' adjacency contains target node indices out of"
                " bounds. Expected indices to be in [0, 2), but found min: 1,"
                " max: 2."
            ),
        ],
    )

  def test_is_creation_time_not_timestamp_semantic(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["f1"].is_creation_time = True
    schema.node_sets["n1"].features[
        "f1"
    ].semantic = schema_lib.FeatureSemantic.NUMERICAL
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertIn(
        Issue.error(
            "The feature 'f1' in nodeset 'n1' has is_creation_time=True, but"
            " its semantic is <FeatureSemantic.NUMERICAL: 'NUMERICAL'>."
            " Features with is_creation_time=True must have semantic=TIMESTAMP."
        ),
        issues,
    )

  def test_timestamp_semantic_not_integer_64(self):
    graph, schema = valid_graph()
    f1 = schema.node_sets["n1"].features["f1"]
    f1.semantic = schema_lib.FeatureSemantic.TIMESTAMP
    f1.format = schema_lib.FeatureFormat.FLOAT_32
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertIn(
        Issue.error(
            "The feature 'f1' in nodeset 'n1' has semantic=TIMESTAMP, but its"
            " format is <FeatureFormat.FLOAT_32: 'FLOAT_32'>. Features with"
            " semantic=TIMESTAMP must have format=INTEGER_64."
        ),
        issues,
    )

  def test_non_timeseries_feature_has_group(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["f1"].group = "g1"
    schema.node_sets["n1"].features["f1"].is_timeseries = False
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertIn(
        Issue.error(
            "The feature 'f1' in nodeset 'n1' has group='g1', but"
            " is_timeseries=False. Non-timeseries features cannot have a group."
        ),
        issues,
    )

  def test_multiple_creation_time_sequences_in_group(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["time1"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_timeseries=True,
        is_creation_time=True,
        group="my_group",
        shape=(None,),
    )
    schema.node_sets["n1"].features["time2"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_timeseries=True,
        is_creation_time=True,
        group="my_group",
        shape=(None,),
    )
    arr = np.array([[10, 20], [10, 20]], dtype=object)
    graph.node_sets["n1"].features["time1"] = arr
    graph.node_sets["n1"].features["time2"] = arr
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertTrue(
        any(
            "Multiple creation time sequence features found for group"
            " 'my_group' in nodeset 'n1'"
            in issue.text
            for issue in issues
        ),
        f"Expected issue not found in: {issues}",
    )

  def test_multiple_entity_creation_time_features(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["ct1"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_timeseries=False,
        is_creation_time=True,
    )
    schema.node_sets["n1"].features["ct2"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_timeseries=False,
        is_creation_time=True,
    )
    graph.node_sets["n1"].features["ct1"] = np.array([10, 20], dtype=np.int64)
    graph.node_sets["n1"].features["ct2"] = np.array([10, 20], dtype=np.int64)
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertTrue(
        any(
            "Multiple entity creation time features found in nodeset 'n1'"
            in issue.text
            for issue in issues
        ),
        f"Expected issue not found in: {issues}",
    )

  def test_timeseries_length_mismatch(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(None,),
        is_timeseries=True,
        is_creation_time=True,
        group="time",
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(None,),
        is_timeseries=True,
        group="time",
    )
    graph.node_sets["n1"].features["time"] = np.array(
        [np.array([10, 20], dtype=np.int64)]
        + [np.array([10], dtype=np.int64)] * (num_nodes - 1),  # pyrefly: ignore[unsupported-operation]
        dtype=object,
    )
    graph.node_sets["n1"].features["val"] = np.fromiter(
        [np.array([1.5], dtype=np.float32)]
        + [np.array([2.5], dtype=np.float32)] * (num_nodes - 1),  # pyrefly: ignore[unsupported-operation]
        dtype=object,
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'val' in nodeset 'n1' has a variable-length"
                " timeseries at index 0 of length 1, which does not match the"
                " creation time sequence 'time' of length 2."
            )
        ],
    )

  def test_valid_temporal_graph(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(None,),
        is_timeseries=True,
        is_creation_time=True,
        group="time",
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(None,),
        is_timeseries=True,
        group="time",
    )
    graph.node_sets["n1"].features["time"] = np.array(
        [np.array([10, 20], dtype=np.int64)]
        + [np.array([30], dtype=np.int64)] * (num_nodes - 1),  # pyrefly: ignore[unsupported-operation]
        dtype=object,
    )
    graph.node_sets["n1"].features["val"] = np.array(
        [np.array([1.5, 2.5], dtype=np.float32)]
        + [np.array([3.5], dtype=np.float32)] * (num_nodes - 1),  # pyrefly: ignore[unsupported-operation]
        dtype=object,
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(issues, [])

  def test_timestamps_target_not_timeseries(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_timeseries=False,
        is_creation_time=True,
    )
    graph.node_sets["n1"].features["time"] = np.zeros(  # pyrefly: ignore[no-matching-overload]
        num_nodes, dtype=np.int64
    )
    schema.node_sets["n1"].features["f1"].is_timeseries = True
    schema.node_sets["n1"].features["f1"].group = "time"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(issues, [])

  def test_timestamps_target_wrong_semantic(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(None,),
        is_timeseries=True,
        is_creation_time=True,
        group="time",
    )
    graph.node_sets["n1"].features["time"] = np.array(
        [[10]] * num_nodes, dtype=object  # pyrefly: ignore[unsupported-operation]
    )
    schema.node_sets["n1"].features["f1"].is_timeseries = True
    schema.node_sets["n1"].features["f1"].group = "time"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertIn(
        Issue.error(
            "The feature 'time' in nodeset 'n1' has is_creation_time=True,"
            " but its semantic is <FeatureSemantic.NUMERICAL: 'NUMERICAL'>."
            " Features with is_creation_time=True must have"
            " semantic=TIMESTAMP."
        ),
        issues,
    )

  def test_timeseries_ndarray_length_mismatch(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(5,),
        is_timeseries=True,
        is_creation_time=True,
        group="time",
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(3,),
        is_timeseries=True,
        group="time",
    )
    graph.node_sets["n1"].features["time"] = np.zeros(  # pyrefly: ignore[no-matching-overload]
        (num_nodes, 5), dtype=np.int64
    )
    graph.node_sets["n1"].features["val"] = np.zeros(  # pyrefly: ignore[no-matching-overload]
        (num_nodes, 3), dtype=np.float32
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'val' in nodeset 'n1' has schema shape (3,) whose"
                " 0th dimension (3) does not match creation time feature 'time'"
                " schema shape 0th dimension (5)."
            )
        ],
    )

  def test_timestamps_schema_shape_incompatible(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(10, 2),
        is_timeseries=True,
        is_creation_time=True,
        group="time",
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
        is_timeseries=True,
        group="time",
    )
    graph.node_sets["n1"].features["time"] = np.zeros(  # pyrefly: ignore[no-matching-overload]
        (num_nodes, 10, 2), dtype=np.int64
    )
    graph.node_sets["n1"].features["val"] = np.zeros(  # pyrefly: ignore[no-matching-overload]
        num_nodes, dtype=np.float32
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The creation time feature 'time' in nodeset 'n1' must have"
                " exactly 1 sequence dimension in schema shape."
            ),
            Issue.error(
                "The feature 'val' in nodeset 'n1' belongs to group 'time',"
                " but must have at least 1 sequence dimension in schema"
                " shape."
            ),
        ],
    )

  def test_static_timeseries_data_length_mismatch(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(5,),
        is_timeseries=True,
        is_creation_time=True,
        group="time",
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(5,),
        is_timeseries=True,
        group="time",
    )
    graph.node_sets["n1"].features["time"] = np.zeros(  # pyrefly: ignore[no-matching-overload]
        (num_nodes, 5), dtype=np.int64
    )
    graph.node_sets["n1"].features["val"] = np.zeros(  # pyrefly: ignore[no-matching-overload]
        (num_nodes, 3), dtype=np.float32
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                f"The feature 'val' in nodeset 'n1' has shape ({num_nodes}, 3),"
                " but the schema expects dimension 0 to be 5."
            )
        ],
    )

  def test_mask_format_invalid(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["bad_mask"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(5,),
        is_timeseries=True,
    )
    graph.node_sets["n1"].features["bad_mask"] = np.zeros(  # pyrefly: ignore[no-matching-overload]
        (num_nodes, 5), dtype=np.float32
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertIn(
        Issue.error(
            "The mask feature 'bad_mask' in nodeset 'n1' must have format"
            " BOOL, but has format <FeatureFormat.FLOAT_32: 'FLOAT_32'>."
        ),
        issues,
    )

  def test_mask_missing_timeseries_group(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["bad_mask"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(5,),
        is_timeseries=True,
    )
    graph.node_sets["n1"].features["bad_mask"] = np.zeros(  # pyrefly: ignore[no-matching-overload]
        (num_nodes, 5), dtype=np.bool_
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertIn(
        Issue.error(
            "The mask feature 'bad_mask' in nodeset 'n1' must have a group to"
            " associate it with the features it masks."
        ),
        issues,
    )

  def test_multiple_masks_in_group(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["mask1"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(5,),
        is_timeseries=True,
        group="my_group",
    )
    schema.node_sets["n1"].features["mask2"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(5,),
        is_timeseries=True,
        group="my_group",
    )

    # Test identical masks yield warning
    mask_data = np.zeros((num_nodes, 5), dtype=np.bool_)  # pyrefly: ignore[no-matching-overload]
    graph.node_sets["n1"].features["mask1"] = mask_data
    graph.node_sets["n1"].features["mask2"] = mask_data
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertTrue(
        any(
            "Since they are identical, consider consolidating them into a"
            " single mask"
            in issue.text
            for issue in issues
        ),
        f"Expected identical masks warning not found in: {issues}",
    )

    # Test differing masks yield error
    graph.node_sets["n1"].features["mask2"] = np.ones(  # pyrefly: ignore[no-matching-overload]
        (num_nodes, 5), dtype=np.bool_
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertTrue(
        any(
            "Multiple features with semantic=MASK found for timeseries group"
            " 'my_group' in nodeset 'n1' with differing or unavailable values"
            in issue.text
            for issue in issues
        ),
        f"Expected differing masks error not found in: {issues}",
    )

  def test_object_dtype_with_non_leading_dynamic_dim(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["var_inner"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(2, None),
    )
    values = np.empty((2, 2), dtype=np.object_)
    values[0, 0] = np.array([b"a", b"bc"], dtype=np.bytes_)
    values[0, 1] = np.array([], dtype=np.bytes_)
    values[1, 0] = np.array([b"d"], dtype=np.bytes_)
    values[1, 1] = np.array([b"e", b"f", b"g"], dtype=np.bytes_)
    graph.node_sets["n1"].features["var_inner"] = values

    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(issues, [])

  def test_object_dtype_with_static_shape(self):
    graph, schema = valid_graph()
    graph.node_sets["n1"].features["f1"] = np.array(
        [[b"v1"], [b"v2"]], dtype=np.object_
    )

    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'f1' in nodeset 'n1' is stored as an object"
                " (`dtype=object`), which is only permitted for variable-length"
                " features. However, the schema defines a fixed shape of (1,)."
                " If your feature is a string, make sure to cast it as a"
                " np.bytes_."
            )
        ],
    )

  def test_varlen_non_ndarray_row(self):
    graph, schema = valid_graph()
    bad_rows = np.empty(2, dtype=object)
    bad_rows[0] = [11, 12]
    bad_rows[1] = np.array([12, 13, 14], dtype=np.int64)
    graph.node_sets["n2"].features["f5"] = bad_rows
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The variable-length feature 'f5' in nodeset 'n2' has a"
                " non-ndarray row at index 0 (found 'list'). Use"
                " `dgf.transform.conform_graph` to convert features to their"
                " schema dtypes."
            )
        ],
    )

  def test_varlen_scalar_row(self):
    graph, schema = valid_graph()
    bad_rows = np.empty(2, dtype=object)
    bad_rows[0] = np.array(11, dtype=np.int64)
    bad_rows[1] = np.array([12, 13, 14], dtype=np.int64)
    graph.node_sets["n2"].features["f5"] = bad_rows
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The variable-length feature 'f5' in nodeset 'n2' has a"
                " scalar (0-D) row at index 0; each row must have at least"
                " one dimension."
            )
        ],
    )

  def test_varlen_object_dtype_row(self):
    graph, schema = valid_graph()
    bad_rows = np.empty(2, dtype=object)
    bad_rows[0] = np.array([11, None], dtype=object)
    bad_rows[1] = np.array([12, 13, 14], dtype=np.int64)
    graph.node_sets["n2"].features["f5"] = bad_rows
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The variable-length feature 'f5' in nodeset 'n2' has row 0"
                " with dtype object (i.e., <class 'numpy.object_'>), but the"
                " schema expects format <FeatureFormat.INTEGER_64:"
                " 'INTEGER_64'>, which corresponds to dtype <class"
                " 'numpy.int64'>. Use `dgf.transform.conform_graph` to convert"
                " features to their schema dtypes."
            )
        ],
    )

  def test_varlen_wrong_dtype_row(self):
    graph, schema = valid_graph()
    bad_rows = np.empty(2, dtype=object)
    bad_rows[0] = np.array([11, 12], dtype=np.int64)
    bad_rows[1] = np.array([12.0, 13.0, 14.0], dtype=np.float32)
    graph.node_sets["n2"].features["f5"] = bad_rows
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The variable-length feature 'f5' in nodeset 'n2' has row 1"
                " with dtype float32 (i.e., <class 'numpy.float32'>), but the"
                " schema expects format <FeatureFormat.INTEGER_64:"
                " 'INTEGER_64'>, which corresponds to dtype <class"
                " 'numpy.int64'>. Use `dgf.transform.conform_graph` to convert"
                " features to their schema dtypes."
            )
        ],
    )

if __name__ == "__main__":
  absltest.main()
