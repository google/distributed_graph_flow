load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = ["//visibility:public"],
)

# Libraries
# =========

py_library(
    name = "runners",
    srcs = ["runners.py"],
    deps = [
        # apache_beam dep,
    ],
)

py_library(
    name = "convert_edge_types",
    srcs = ["convert_edge_types.py"],
    deps = [
        # absl/logging dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
    ],
)

py_library(
    name = "reverse_edges",
    srcs = ["reverse_edges.py"],
    deps = [
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
    ],
)

# Test
# =========

py_test(
    name = "convert_edge_types_test",
    srcs = ["convert_edge_types_test.py"],
    deps = [
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "reverse_edges_test",
    srcs = ["reverse_edges_test.py"],
    deps = [
        ":reverse_edges",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:hgraph_in_beam",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)
