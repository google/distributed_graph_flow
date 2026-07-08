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

from typing import Tuple
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
    Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
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
                "The nodeset 'n1' schema is missing the '#id' feature."
                " Giving a clearly defined #id column is recommanded. It is"
                " also required for non-string node IDs e.g. integer IDs."
            ),
            Issue.warning(
                "The nodeset 'n2' schema is missing the '#id' feature."
                " Giving a clearly defined #id column is recommanded. It is"
                " also required for non-string node IDs e.g. integer IDs."
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

  def test_timestamps_missing_reference(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["f1"].is_timeseries = True
    schema.node_sets["n1"].features["f1"].timestamps = "missing_time"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'f1' in nodeset 'n1' references timestamps feature"
                " 'missing_time' which is not defined in the schema."
            )
        ],
    )

  def test_timestamps_not_timeseries(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["f1"].is_timeseries = False
    schema.node_sets["n1"].features["f1"].timestamps = "time"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'f1' in nodeset 'n1' has timestamps set to 'time',"
                " but is_timeseries is False."
            ),
            Issue.error(
                "The feature 'f1' in nodeset 'n1' references timestamps feature"
                " 'time' which is not defined in the schema."
            ),
        ],
    )

  def test_conflicting_timestamps_in_group(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["f1"].is_timeseries = True
    schema.node_sets["n1"].features["f1"].timestamps = "time1"
    schema.node_sets["n1"].features["f1"].timeseries_group = "my_group"
    schema.node_sets["n1"].features["f1"].shape = (None,)
    schema.node_sets["n1"].features["f2"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        is_timeseries=True,
        timestamps="time2",
        timeseries_group="my_group",
        shape=(None,),
    )
    schema.node_sets["n1"].features["time1"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_timeseries=True,
        timeseries_group="my_group",
        shape=(None,),
    )
    schema.node_sets["n1"].features["time2"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_timeseries=True,
        timeseries_group="my_group",
        shape=(None,),
    )
    arr = np.array([[10, 20], [10, 20]], dtype=object)
    graph.node_sets["n1"].features["f1"] = arr
    graph.node_sets["n1"].features["f2"] = arr
    graph.node_sets["n1"].features["time1"] = arr
    graph.node_sets["n1"].features["time2"] = arr
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertTrue(
        any(
            "Multiple conflicting timestamps features found for timeseries"
            " group 'my_group' in nodeset 'n1': 'time1' and 'time2'"
            in issue.text
            for issue in issues
        ),
        f"Expected conflicting timestamps issue not found in: {issues}",
    )

  def test_timeseries_length_mismatch(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(None,),
        is_timeseries=True,
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(None,),
        is_timeseries=True,
        timestamps="time",
    )
    graph.node_sets["n1"].features["time"] = np.array(
        [[10, 20]] + [[10]] * (num_nodes - 1), dtype=object
    )
    graph.node_sets["n1"].features["val"] = np.array(
        [[1.5]] + [[2.5]] * (num_nodes - 1), dtype=object
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'val' in nodeset 'n1' has a variable-length"
                " timeseries at index 0 of length 1, which does not match the"
                " timestamps sequence 'time' of length 2."
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
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(None,),
        is_timeseries=True,
        timestamps="time",
    )
    graph.node_sets["n1"].features["time"] = np.array(
        [[10, 20]] + [[30]] * (num_nodes - 1), dtype=object
    )
    graph.node_sets["n1"].features["val"] = np.array(
        [[1.5, 2.5]] + [[3.5]] * (num_nodes - 1), dtype=object
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
    )
    graph.node_sets["n1"].features["time"] = np.zeros(
        num_nodes, dtype=np.int64
    )
    schema.node_sets["n1"].features["f1"].is_timeseries = True
    schema.node_sets["n1"].features["f1"].timestamps = "time"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'f1' in nodeset 'n1' references timestamps feature"
                " 'time', but 'time' does not have is_timeseries=True."
            )
        ],
    )

  def test_timestamps_target_wrong_semantic(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(None,),
        is_timeseries=True,
    )
    graph.node_sets["n1"].features["time"] = np.array(
        [[10]] * num_nodes, dtype=object
    )
    schema.node_sets["n1"].features["f1"].is_timeseries = True
    schema.node_sets["n1"].features["f1"].timestamps = "time"
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'f1' in nodeset 'n1' references timestamps feature"
                " 'time', but 'time' does not have semantic=TIMESTAMP."
            )
        ],
    )

  def test_timeseries_ndarray_length_mismatch(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(5,),
        is_timeseries=True,
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(3,),
        is_timeseries=True,
        timestamps="time",
    )
    graph.node_sets["n1"].features["time"] = np.zeros(
        (num_nodes, 5), dtype=np.int64
    )
    graph.node_sets["n1"].features["val"] = np.zeros(
        (num_nodes, 3), dtype=np.float32
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'val' in nodeset 'n1' has schema shape (3,) whose"
                " 0th dimension (3) does not match timestamps feature 'time'"
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
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
        is_timeseries=True,
        timestamps="time",
    )
    graph.node_sets["n1"].features["time"] = np.zeros(
        (num_nodes, 10, 2), dtype=np.int64
    )
    graph.node_sets["n1"].features["val"] = np.zeros(
        num_nodes, dtype=np.float32
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertEqual(
        issues,
        [
            Issue.error(
                "The feature 'val' in nodeset 'n1' references timestamps"
                " feature 'time', but 'time' must have exactly 1 sequence"
                " dimension in schema shape."
            ),
            Issue.error(
                "The feature 'val' in nodeset 'n1' references timestamps"
                " feature 'time', but 'val' must have at least 1 sequence"
                " dimension in schema shape."
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
    )
    schema.node_sets["n1"].features["val"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(5,),
        is_timeseries=True,
        timestamps="time",
    )
    graph.node_sets["n1"].features["time"] = np.zeros(
        (num_nodes, 5), dtype=np.int64
    )
    graph.node_sets["n1"].features["val"] = np.zeros(
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
    graph.node_sets["n1"].features["bad_mask"] = np.zeros(
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
    graph.node_sets["n1"].features["bad_mask"] = np.zeros(
        (num_nodes, 5), dtype=np.bool_
    )
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertIn(
        Issue.error(
            "The mask feature 'bad_mask' in nodeset 'n1' must have a"
            " timeseries_group to associate it with the features it masks."
        ),
        issues,
    )

  def test_conflicting_timeseries_group_with_timestamp(self):
    graph, schema = valid_graph()
    schema.node_sets["n1"].features["f1"].is_timeseries = True
    schema.node_sets["n1"].features["f1"].timestamps = "time1"
    schema.node_sets["n1"].features["f1"].timeseries_group = "groupA"
    schema.node_sets["n1"].features["f1"].shape = (None,)
    schema.node_sets["n1"].features["time1"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_timeseries=True,
        timeseries_group="groupB",
        shape=(None,),
    )
    arr = np.array([[10, 20], [10, 20]], dtype=object)
    graph.node_sets["n1"].features["f1"] = arr
    graph.node_sets["n1"].features["time1"] = arr
    issues = in_memory_graph_validate_lib.issues(graph, schema)
    self.assertTrue(
        any(
            "Features with the same timestamps must be in the same timeseries"
            " group"
            in issue.text
            for issue in issues
        ),
        f"Expected conflicting timeseries group issue not found in: {issues}",
    )

  def test_multiple_masks_in_group(self):
    graph, schema = valid_graph()
    num_nodes = graph.node_sets["n1"].num_nodes
    schema.node_sets["n1"].features["mask1"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(5,),
        is_timeseries=True,
        timeseries_group="my_group",
    )
    schema.node_sets["n1"].features["mask2"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BOOL,
        semantic=schema_lib.FeatureSemantic.MASK,
        shape=(5,),
        is_timeseries=True,
        timeseries_group="my_group",
    )

    # Test identical masks yield warning
    mask_data = np.zeros((num_nodes, 5), dtype=np.bool_)
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
    graph.node_sets["n1"].features["mask2"] = np.ones(
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


if __name__ == "__main__":
  absltest.main()
