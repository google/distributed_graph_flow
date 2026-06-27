load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = ["//dgf/src:__subpackages__"],
)

# Libraries
# =========

py_library(
    name = "training_configuration",
    srcs = ["training_configuration.py"],
    deps = [
        # dataclasses_json dep,
    ],
)

py_library(
    name = "feature",
    srcs = ["feature.py"],
    deps = [
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        # absl/logging dep,
        "//dgf/src/util:filesystem",
    ],
)

# Test
# =========

py_test(
    name = "feature_test",
    srcs = ["feature_test.py"],
    deps = [
        ":feature",
        # absl/testing:absltest dep,
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:test_util",
        # jax dep,
    ],
)

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    deps = [
        ":config",
        # absl/testing:absltest dep,
    ],
)
