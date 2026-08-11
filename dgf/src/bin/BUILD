load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    
    default_visibility = ["//dgf/src:__subpackages__"],
)

# Binary
# =========

py_library(
    name = "validate_graph_lib",
    srcs = ["validate_graph.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/validate:in_memory_graph",
    ],
)

py_binary(
    name = "validate_graph",
    srcs = ["validate_graph.py"],
    deps = [
        ":validate_graph_lib",
    ],
)

py_binary(
    name = "create_graph_sampling_plan",
    srcs = ["create_graph_sampling_plan.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf",
    ],
)

py_binary(
    name = "import_graph_ai_graph",
    srcs = ["import_graph_ai_graph.py"],
    deps = [
        "//file/colossus/public:cns",
        # absl:app dep,
        # absl/flags dep,
        # apache_beam dep,
        "//dgf",
    ],
)

# Test
# =========

py_test(
    name = "validate_graph_test",
    srcs = ["validate_graph_test.py"],
    deps = [
        ":validate_graph_lib",
        # absl/flags dep,
        # absl/testing:absltest dep,
        # absl/testing:flagsaver dep,
        "//dgf/src/util:gen_test_graph",
    ],
)
