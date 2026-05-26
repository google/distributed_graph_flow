load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Library
# =====

py_library(
    name = "network",
    srcs = ["network.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        # graphviz dep,
    ],
)

py_library(
    name = "nx",
    srcs = ["nx.py"],
    deps = [
        # matplotlib dep,
        # networkx dep,
    ],
)

# Test
# ====

py_test(
    name = "network_test",
    srcs = ["network_test.py"],
    deps = [
        ":network",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/util:gen_test_graph",
    ],
)
