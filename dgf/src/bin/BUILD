load("//devtools/python/blaze:pytype.bzl", "py_binary", "py_test")

package(
    
    default_visibility = ["//dgf/src:__subpackages__"],
)

# Binary
# =========

py_binary(
    name = "validate_graph",
    srcs = ["validate_graph.py"],
    test_lib = True,
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf",
    ],
)

# Test
# =========

py_test(
    name = "validate_graph_test",
    srcs = ["validate_graph_test.py"],
    deps = [
        ":validate_graph.testonly_lib",
        # absl/flags dep,
        # absl/testing:absltest dep,
        # absl/testing:flagsaver dep,
        "//dgf/src/util:gen_test_graph",
    ],
)
