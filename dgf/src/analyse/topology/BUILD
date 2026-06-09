load("@rules_python//python:py_library.bzl", "py_library")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

py_library(
    name = "node_degree",
    srcs = ["node_degree.py"],
    deps = [
        # numpy dep,
    ],
)

py_library(
    name = "betti_defense",
    srcs = ["betti_defense.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        # numpy dep,
    ],
)

py_library(
    name = "global_graph_topology",
    srcs = ["global_graph_topology.py"],
    deps = [
        ":betti_defense",
        ":node_degree",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        # numpy dep,
    ],
)
