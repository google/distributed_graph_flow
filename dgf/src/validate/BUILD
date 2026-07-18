load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Libraries
# =========

py_library(
    name = "validate",
    srcs = ["validate.py"],
    deps = [],
)

py_library(
    name = "in_memory_graph",
    srcs = ["in_memory_graph.py"],
    deps = [
        ":validate",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        # numpy dep,
    ],
)

# Test
# =========

py_test(
    name = "validate_test",
    srcs = ["validate_test.py"],
    deps = [
        ":validate",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
    ],
)

py_test(
    name = "in_memory_graph_test",
    srcs = ["in_memory_graph_test.py"],
    deps = [
        ":in_memory_graph",
        ":validate",
        # absl/testing:absltest dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)
