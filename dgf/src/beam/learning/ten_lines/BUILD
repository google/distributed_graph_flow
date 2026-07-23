load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_applicable_licenses = ["//dgf:license"])

# Libraries
# =========

py_library(
    name = "predict",
    srcs = ["predict.py"],
    deps = [
        # apache_beam dep,
        # apache_beam/utils:low_level dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_in_beam",
        "//dgf/src/learning/ten_lines:common",
        "//dgf/src/learning/ten_lines:node_prediction",
        "//dgf/src/sampling:beam_semi_distributed_sampler",
        "//dgf/src/sampling:beam_semi_distributed_sampler_v2",
        # numpy dep,
    ],
)

# Test
# =========

py_test(
    name = "predict_test",
    size = "large",
    srcs = ["predict_test.py"],
    deps = [
        ":predict",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/generate:graphs",
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/learning/ten_lines:node_prediction",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)
