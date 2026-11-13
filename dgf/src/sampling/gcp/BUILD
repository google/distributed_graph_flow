load("@rules_python//python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Binaries
# ========

py_binary(
    name = "bigquery_graph_sampler_integration_test",
    srcs = ["bigquery_graph_sampler_integration_test.py"],
    type_checking = True,
    deps = [
        ":bigquery_graph_sampler",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:cache",
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/io/gcp:bigquery_graph",
        "//dgf/src/learning/ten_lines:node_prediction",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/transform:canonicalize",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)

py_binary(
    name = "spanner_graph_sampler_integration_test",
    srcs = ["spanner_graph_sampler_integration_test.py"],
    type_checking = True,
    deps = [
        ":spanner_graph_sampler",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/io:cache",
        "//dgf/src/io/gcp:spanner_graph",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/transform:canonicalize",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)

# Libraries
# =========

py_library(
    name = "bigquery_graph_sampler",
    srcs = ["bigquery_graph_sampler.py"],
    type_checking = True,
    deps = [
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        "//dgf/src/io/gcp:bigquery_graph",
        "//dgf/src/io/gcp:bigquery_graph_metadata",
        "//dgf/src/io/gcp:common",
        "//dgf/src/sampling:config",
        "//dgf/src/util:log",
        # google/cloud/bigquery dep,
        # numpy dep,
    ],
)

py_library(
    name = "spanner_graph_sampler",
    srcs = ["spanner_graph_sampler.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:schema",
        "//dgf/src/io/gcp:spanner_graph_metadata",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:spanner_graph_sampler",
        "//dgf/src/util:log",
        # google/cloud/spanner dep,
    ],
)

# Test
# =========

py_test(
    name = "bigquery_graph_sampler_test",
    srcs = ["bigquery_graph_sampler_test.py"],
    data = ["//test_data"],
    type_checking = True,
    deps = [
        ":bigquery_graph_sampler",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
        "//dgf/src/io/gcp:bigquery_graph",
        "//dgf/src/io/gcp:bigquery_graph_metadata",
        "//dgf/src/sampling:config",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "spanner_graph_sampler_test",
    srcs = ["spanner_graph_sampler_test.py"],
    type_checking = True,
    deps = [
        ":spanner_graph_sampler",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
        "//dgf/src/sampling:config",
        # google/cloud/spanner dep,
        # numpy dep,
    ],
)
