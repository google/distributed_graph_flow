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

"""Tests for `dgf.src.transform.conform`."""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
from dgf.src.transform import conform as conform_lib
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np
import pandas as pd

test_util.disable_diff_truncation()


def _single_node_feature_graph(
    values: Any,
    fmt: schema_lib.FeatureFormat,
    shape: tuple[int | None, ...] | None = None,
    num_nodes: int = 4,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "n1": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_nodes,
              features={
                  "#id": np.arange(num_nodes, dtype=np.int64),
                  "feat": values,
              },
          )
      },
      edge_sets={},
  )
  schema = schema_lib.GraphSchema(
      node_sets={
          "n1": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "feat": schema_lib.FeatureSchema(
                      format=fmt,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                      shape=shape,
                  ),
              }
          )
      },
      edge_sets={},
  )
  return graph, schema


class ConformGraphTest(parameterized.TestCase):

  @parameterized.parameters(
      (schema_lib.FeatureFormat.FLOAT_32, np.float32),
      (schema_lib.FeatureFormat.FLOAT_64, np.float64),
  )
  def test_float_static_missing_and_inf(self, fmt, expected_dtype):
    for raw in (
        np.array([1.5, None, pd.NA, np.nan, np.inf, -np.inf], dtype=object),
        [1.5, None, pd.NA, np.nan, np.inf, -np.inf],
        pd.array([1.5, None, pd.NA, np.nan, np.inf, -np.inf], dtype="Float64"),
        pd.Series([1.5, None, pd.NA, np.nan, np.inf, -np.inf], dtype="Float64"),
    ):
      graph, schema = _single_node_feature_graph(raw, fmt, num_nodes=6)
      conformed = conform_lib.conform_graph(graph, schema)
      in_memory_graph_validate_lib.validate_graph(conformed, schema)
      out = conformed.node_sets["n1"].features["feat"]
      self.assertEqual(out.dtype, expected_dtype)
      expected = np.array(
          [1.5, np.nan, np.nan, np.nan, np.inf, -np.inf], dtype=expected_dtype
      )
      np.testing.assert_array_equal(out, expected)

  def test_float_static_from_nullable_int64(self):
    raw = pd.array([1, pd.NA, 3, None], dtype="Int64")
    graph, schema = _single_node_feature_graph(
        raw, schema_lib.FeatureFormat.FLOAT_32, num_nodes=4
    )
    conformed = conform_lib.conform_graph(graph, schema)
    in_memory_graph_validate_lib.validate_graph(conformed, schema)
    out = conformed.node_sets["n1"].features["feat"]
    self.assertEqual(out.dtype, np.float32)
    np.testing.assert_array_equal(
        out, np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
    )

  def test_float_static_nd_reshape(self):
    raw = np.array(
        [[1.0, None, np.inf], [pd.NA, -np.inf, 2.5]], dtype=object
    )
    graph, schema = _single_node_feature_graph(
        raw, schema_lib.FeatureFormat.FLOAT_32, shape=(3,), num_nodes=2
    )
    conformed = conform_lib.conform_graph(graph, schema)
    in_memory_graph_validate_lib.validate_graph(conformed, schema)
    out = conformed.node_sets["n1"].features["feat"]
    self.assertEqual(out.dtype, np.float32)
    self.assertEqual(out.shape, (2, 3))
    np.testing.assert_array_equal(
        out,
        np.array(
            [[1.0, np.nan, np.inf], [np.nan, -np.inf, 2.5]], dtype=np.float32
        ),
    )

  def test_float_varlen_rows(self):
    rows = np.empty(3, dtype=object)
    rows[0] = [1.0, None, np.inf]
    rows[1] = np.array([pd.NA, -np.inf], dtype=object)
    rows[2] = pd.array([2.0, pd.NA, 4.0], dtype="Float64")
    graph, schema = _single_node_feature_graph(
        rows, schema_lib.FeatureFormat.FLOAT_32, shape=(None,), num_nodes=3
    )
    conformed = conform_lib.conform_graph(graph, schema)
    in_memory_graph_validate_lib.validate_graph(conformed, schema)
    out = conformed.node_sets["n1"].features["feat"]
    self.assertEqual(out.dtype, np.object_)
    for row in out:
      self.assertIsInstance(row, np.ndarray)
      self.assertEqual(row.dtype, np.float32)
    np.testing.assert_array_equal(
        out[0], np.array([1.0, np.nan, np.inf], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out[1], np.array([np.nan, -np.inf], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out[2], np.array([2.0, np.nan, 4.0], dtype=np.float32)
    )

  @parameterized.parameters(
      (schema_lib.FeatureFormat.INTEGER_32, np.int32),
      (schema_lib.FeatureFormat.INTEGER_64, np.int64),
  )
  def test_integer_valid_cast_and_missing_raises(self, fmt, expected_dtype):
    graph, schema = _single_node_feature_graph(
        [10, 20, 30, 40], fmt, num_nodes=4
    )
    conformed = conform_lib.conform_graph(graph, schema)
    in_memory_graph_validate_lib.validate_graph(conformed, schema)
    out = conformed.node_sets["n1"].features["feat"]
    self.assertEqual(out.dtype, expected_dtype)
    np.testing.assert_array_equal(
        out, np.array([10, 20, 30, 40], dtype=expected_dtype)
    )

    for bad in (
        [10, None, 30, 40],
        np.array([10, pd.NA, 30, 40], dtype=object),
        pd.array([10, pd.NA, 30, 40], dtype="Int64"),
        np.array([10.0, np.nan, 30.0, 40.0], dtype=np.float64),
    ):
      bad_graph, bad_schema = _single_node_feature_graph(bad, fmt, num_nodes=4)
      with self.assertRaisesRegex(
          ValueError, r"feat.*nodeset 'n1'.*FLOAT_32 or FLOAT_64"
      ):
        conform_lib.conform_graph(bad_graph, bad_schema)

  def test_bytes_static_and_varlen_and_replace_none_bytes(self):
    raw = np.array([b"a", "hello", None, pd.NA, np.nan], dtype=object)
    graph, schema = _single_node_feature_graph(
        raw, schema_lib.FeatureFormat.BYTES, num_nodes=5
    )
    # By default, missing bytes raise.
    with self.assertRaisesRegex(ValueError, "replace_none_bytes"):
      conform_lib.conform_graph(graph, schema)

    conformed = conform_lib.conform_graph(
        graph, schema, replace_none_bytes=b""
    )
    in_memory_graph_validate_lib.validate_graph(conformed, schema)
    out = conformed.node_sets["n1"].features["feat"]
    self.assertEqual(out.dtype.type, np.bytes_)
    np.testing.assert_array_equal(
        out, np.array([b"a", b"hello", b"", b"", b""], dtype=np.bytes_)
    )

    conformed_custom = conform_lib.conform_graph(
        graph, schema, replace_none_bytes=b"<NA>"
    )
    in_memory_graph_validate_lib.validate_graph(conformed_custom, schema)
    np.testing.assert_array_equal(
        conformed_custom.node_sets["n1"].features["feat"],
        np.array([b"a", b"hello", b"<NA>", b"<NA>", b"<NA>"], dtype=np.bytes_),
    )

    varlen_raw = np.empty(2, dtype=object)
    varlen_raw[0] = ["x", None, b"z"]
    varlen_raw[1] = np.array([pd.NA, "w"], dtype=object)
    v_graph, v_schema = _single_node_feature_graph(
        varlen_raw,
        schema_lib.FeatureFormat.BYTES,
        shape=(None,),
        num_nodes=2,
    )
    v_conformed = conform_lib.conform_graph(
        v_graph, v_schema, replace_none_bytes=b"?"
    )
    in_memory_graph_validate_lib.validate_graph(v_conformed, v_schema)
    v_out = v_conformed.node_sets["n1"].features["feat"]
    np.testing.assert_array_equal(
        v_out[0], np.array([b"x", b"?", b"z"], dtype=np.bytes_)
    )
    np.testing.assert_array_equal(
        v_out[1], np.array([b"?", b"w"], dtype=np.bytes_)
    )

  def test_bool_valid_and_missing_raises(self):
    graph, schema = _single_node_feature_graph(
        [True, False, 1, 0], schema_lib.FeatureFormat.BOOL, num_nodes=4
    )
    conformed = conform_lib.conform_graph(graph, schema)
    in_memory_graph_validate_lib.validate_graph(conformed, schema)
    out = conformed.node_sets["n1"].features["feat"]
    self.assertEqual(out.dtype, np.bool_)
    np.testing.assert_array_equal(
        out, np.array([True, False, True, False], dtype=np.bool_)
    )

    for bad in (
        [True, None, False, True],
        np.array([True, pd.NA, False, True], dtype=object),
        pd.array([True, pd.NA, False, True], dtype="boolean"),
    ):
      bad_graph, bad_schema = _single_node_feature_graph(
          bad, schema_lib.FeatureFormat.BOOL, num_nodes=4
      )
      with self.assertRaisesRegex(ValueError, r"feat.*nodeset 'n1'"):
        conform_lib.conform_graph(bad_graph, bad_schema)

  @parameterized.parameters(
      schema_lib.FeatureFormat.FLOAT_32,
      schema_lib.FeatureFormat.FLOAT_64,
      schema_lib.FeatureFormat.INTEGER_32,
      schema_lib.FeatureFormat.INTEGER_64,
  )
  def test_non_numeric_string_in_numeric_feature_raises(self, fmt):
    graph, schema = _single_node_feature_graph(
        np.array(["abc", "def", "ghi", "jkl"], dtype=object), fmt, num_nodes=4
    )
    with self.assertRaisesRegex(ValueError, r"feat.*nodeset 'n1'"):
      conform_lib.conform_graph(graph, schema)

  def test_already_well_formed_arrays_returned_without_copy(self):
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids=True, edge_ids=True, variable_length=True
    )
    schema = gen_test_graph.generate_schema(
        node_ids=True, edge_ids=True, variable_length=True, semantic=True
    )
    conformed = conform_lib.conform_graph(graph, schema)
    in_memory_graph_validate_lib.validate_graph(conformed, schema)
    for ns_name, ns in graph.node_sets.items():
      for feat_name, orig_arr in ns.features.items():
        self.assertIs(conformed.node_sets[ns_name].features[feat_name], orig_arr)
    for es_name, es in graph.edge_sets.items():
      for feat_name, orig_arr in es.features.items():
        self.assertIs(conformed.edge_sets[es_name].features[feat_name], orig_arr)

  def test_mixed_graph_node_and_edge_sets_roundtrip_validation(self):
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids=True, edge_ids=True, variable_length=True
    )
    schema = gen_test_graph.generate_schema(
        node_ids=True, edge_ids=True, variable_length=True, semantic=True
    )
    # Replace node and edge features with unconformed lists/object arrays.
    n1_feats: Any = graph.node_sets["n1"].features
    n2_feats: Any = graph.node_sets["n2"].features
    e1_feats: Any = graph.edge_sets["e1"].features
    n1_feats["f1"] = [["a"], [None]]
    n1_feats["f2"] = [[1.5, None], [pd.NA, np.inf]]
    n2_feats["f5"] = [[11, 12], [12, 13, 14]]
    e1_feats["#id"] = [None, "edge_b"]
    schema.edge_sets["e1"].features["ef_float"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )
    e1_feats["ef_float"] = pd.Series([1.0, pd.NA], dtype="Float64")

    with self.assertRaises(ValueError):
      in_memory_graph_validate_lib.validate_graph(graph, schema)

    conformed = conform_lib.conform_graph(
        graph, schema, replace_none_bytes=b""
    )
    in_memory_graph_validate_lib.validate_graph(conformed, schema)

  @parameterized.parameters(
      schema_lib.FeatureFormat.INTEGER_32, schema_lib.FeatureFormat.INTEGER_64
  )
  def test_integer_fractional_raises(self, fmt):
    graph, schema = _single_node_feature_graph(
        np.array([1.5, 2.0], dtype=object), fmt, num_nodes=2
    )
    with self.assertRaisesRegex(ValueError, "non-integer values"):
      conform_lib.conform_graph(graph, schema)

  @parameterized.parameters(
      (np.array([2**31, 0], dtype=np.int64),),
      (np.array([2**31, 0], dtype=object),),
  )
  def test_integer_out_of_range_raises(self, values):
    graph, schema = _single_node_feature_graph(
        values, schema_lib.FeatureFormat.INTEGER_32, num_nodes=2
    )
    with self.assertRaisesRegex(ValueError, "outside the range"):
      conform_lib.conform_graph(graph, schema)

  @parameterized.parameters(
      (np.array(["", ""]),),
      (np.array([None, None], dtype=object),),
      (np.array([b"", None], dtype=object),),
  )
  def test_bytes_all_empty_validates(self, values):
    graph, schema = _single_node_feature_graph(
        values, schema_lib.FeatureFormat.BYTES, num_nodes=2
    )
    conformed = conform_lib.conform_graph(
        graph, schema, replace_none_bytes=b""
    )
    in_memory_graph_validate_lib.validate_graph(conformed, schema)

  @parameterized.parameters([[10, 20]], [["abc", "d"]])
  def test_varlen_scalar_rows_raise(self, values):
    graph, schema = _single_node_feature_graph(
        values, schema_lib.FeatureFormat.INTEGER_64, shape=(None,), num_nodes=2
    )
    with self.assertRaisesRegex(ValueError, "scalar row"):
      conform_lib.conform_graph(graph, schema)

  def test_non_leading_dynamic_dim_well_formed_is_kept(self):
    values = np.empty((2, 2), dtype=object)
    for idx in np.ndindex(values.shape):
      values[idx] = np.array([1, 2][: idx[1] + 1], dtype=np.int64)
    graph, schema = _single_node_feature_graph(
        values, schema_lib.FeatureFormat.INTEGER_64, shape=(2, None), num_nodes=2
    )
    conformed = conform_lib.conform_graph(graph, schema)
    self.assertIs(conformed.node_sets["n1"].features["feat"], values)

  def test_sets_not_in_schema_are_kept(self):
    graph, schema = _single_node_feature_graph(
        np.array([1.0, 2.0], dtype=np.float32),
        schema_lib.FeatureFormat.FLOAT_32,
        num_nodes=2,
    )
    extra = in_memory_graph_lib.InMemoryNodeSet(num_nodes=1, features={})
    graph.node_sets["extra"] = extra
    conformed = conform_lib.conform_graph(graph, schema)
    self.assertIs(conformed.node_sets["extra"], extra)


if __name__ == "__main__":
  absltest.main()
