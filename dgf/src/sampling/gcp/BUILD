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
    name = "spanner_graph_sampler_integration_test",
    srcs = ["spanner_graph_sampler_integration_test.py"],
    deps = [
        ":spanner_graph_sampler",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:cache",
        "//dgf/src/io/gcp:spanner_graph",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)

# Libraries
# =========

py_library(
    name = "spanner_graph_sampler",
    srcs = ["spanner_graph_sampler.py"],
    deps = [
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        "//dgf/src/sampling:config",
        # google/cloud/spanner dep,
        # numpy dep,
    ],
)

# Test
# =========

py_test(
    name = "spanner_graph_sampler_test",
    srcs = ["spanner_graph_sampler_test.py"],
    data = ["//test_data"],
    deps = [
        ":spanner_graph_sampler",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/plot:network",
        "//dgf/src/sampling:config",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)
