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
    name = "graphs",
    srcs = ["graphs.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/sampling:config",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
        # tqdm dep,
    ],
)

py_library(
    name = "signals",
    srcs = ["signals.py"],
    deps = [
        # numpy dep,
    ],
)

# Test
# =========

py_test(
    name = "graphs_test",
    srcs = ["graphs_test.py"],
    deps = [
        ":graphs",
        # absl/testing:absltest dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/sampling:config",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
    ],
)

py_library(
    name = "edge_neighbor_generator",
    srcs = ["edge_neighbor_generator.py"],
    deps = [
        # dataclasses_json dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/util:dataclass_registry",
        # numpy dep,
    ],
)

py_test(
    name = "edge_neighbor_generator_test",
    srcs = ["edge_neighbor_generator_test.py"],
    deps = [
        ":edge_neighbor_generator",
        # absl/testing:absltest dep,
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/util:gen_test_graph",
        # numpy dep,
    ],
)

py_test(
    name = "signals_test",
    srcs = ["signals_test.py"],
    deps = [
        ":signals",
        # absl/testing:absltest dep,
        # numpy dep,
    ],
)
