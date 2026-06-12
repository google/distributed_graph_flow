load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Binaries
# ========

py_binary(
    name = "bigquery_graph_integration_test",
    srcs = ["bigquery_graph_integration_test.py"],
    deps = [
        ":bigquery_graph",
        # absl/testing:absltest dep,
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
    ],
)

py_binary(
    name = "spanner_graph_integration_test",
    srcs = ["spanner_graph_integration_test.py"],
    deps = [
        ":spanner_graph",
        # absl/testing:absltest dep,
        "//dgf/src/data:schema",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
    ],
)

# Libraries
# =========

py_library(
    name = "bigquery_graph",
    srcs = ["bigquery_graph.py"],
    deps = [
        ":bigquery_graph_metadata",
        ":common",
        ":parquet_export",
        "//dgf/src/analyse:print_schema",
        "//dgf/src/data:gf_metadata",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_constants",
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/io:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util:util_py",
        # google/cloud/bigquery dep,
    ],
)

py_library(
    name = "bigquery_graph_beam",
    srcs = ["bigquery_graph_beam.py"],
    deps = [
        ":bigquery_graph",
        ":common",
        ":common_beam",
        ":parquet_export",
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/io:graph_in_beam",
    ],
)

py_library(
    name = "bigquery_graph_metadata",
    srcs = ["bigquery_graph_metadata.py"],
    deps = [
        # dataclasses_json dep,
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        "//dgf/src/io:io_ext",
        "//dgf/src/util:log",
        # numpy dep,
        # pandas dep,
        # tqdm dep,
    ],
)

py_library(
    name = "common_beam",
    srcs = ["common_beam.py"],
    deps = [
        ":common",
        "//dgf/src/analyse:schema",
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_in_beam",
    ],
)

py_library(
    name = "parquet_export",
    srcs = ["parquet_export.py"],
    deps = [
        ":bigquery_graph_metadata",
        "//dgf/src/io:graph_in_beam",
        "//dgf/src/util:log",
    ],
)

py_library(
    name = "spanner_graph",
    srcs = ["spanner_graph.py"],
    deps = [
        ":common",
        ":spanner_graph_metadata",
        "//dgf/src/analyse:print_schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:io_ext",
        "//dgf/src/util:log",
        "//dgf/src/util:util_py",
        # google/cloud/spanner dep,
        # google/cloud/spanner_v1 dep,
        # numpy dep,
    ],
)

py_library(
    name = "spanner_graph_beam",
    srcs = ["spanner_graph_beam.py"],
    deps = [
        ":common",
        ":common_beam",
        ":spanner_graph",
        ":spanner_graph_metadata",
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/util:log",
        # google/cloud/spanner dep,
    ],
)

py_library(
    name = "spanner_graph_metadata",
    srcs = ["spanner_graph_metadata.py"],
    deps = [
        # dataclasses_json dep,
    ],
)

# Test
# =========

py_test(
    name = "bigquery_graph_beam_test",
    srcs = ["bigquery_graph_beam_test.py"],
    deps = [
        ":bigquery_graph_beam",
        ":bigquery_graph_metadata",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
    ],
)

py_test(
    name = "bigquery_graph_test",
    srcs = ["bigquery_graph_test.py"],
    deps = [
        ":bigquery_graph",
        ":bigquery_graph_metadata",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
    ],
)

py_test(
    name = "common_beam_test",
    srcs = ["common_beam_test.py"],
    deps = [
        ":common_beam",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_in_beam",
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        ":common",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_in_beam",
        "//dgf/src/io:io_ext",
        # numpy dep,
    ],
)

py_test(
    name = "parquet_export_test",
    srcs = ["parquet_export_test.py"],
    deps = [
        ":bigquery_graph_metadata",
        ":common",
        ":parquet_export",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
    ],
)

py_test(
    name = "spanner_graph_beam_test",
    srcs = ["spanner_graph_beam_test.py"],
    deps = [
        ":spanner_graph_beam",
        ":spanner_graph_metadata",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
    ],
)

py_test(
    name = "spanner_graph_test",
    srcs = ["spanner_graph_test.py"],
    data = ["//test_data"],
    deps = [
        ":spanner_graph",
        ":spanner_graph_metadata",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/io:io_ext",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)
