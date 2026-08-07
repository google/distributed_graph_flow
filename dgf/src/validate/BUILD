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
    deps = ["//dgf/src/util:log"],
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

py_library(
    name = "graph_snapshots",
    srcs = ["graph_snapshots.py"],
    deps = [
        ":in_memory_graph",
        ":validate",
        "//dgf/src/data:graph_snapshots_metadata",
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/io:graph_snapshots_metadata",
        "//dgf/src/io:schema",
        "//dgf/src/util:filesystem",
        # etils/epath dep,
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
    name = "graph_snapshots_test",
    srcs = ["graph_snapshots_test.py"],
    deps = [
        ":graph_snapshots",
        ":validate",
        # absl/testing:absltest dep,
        "//dgf/src/data:graph_snapshots_metadata",
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/io:graph_snapshots_metadata",
        "//dgf/src/io:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
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
