load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//dgf:__subpackages__",
        "//:internal_users",
    ],
)

# Library
# =====

py_library(
    name = "all",
    deps = [
        ":common",
        "//dgf/src/learning/jax/layers:all",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        # absl/logging dep,
        "//dgf/src/learning:config",
        "//dgf/src/util:log",
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "flax_gnn",
    srcs = ["flax_gnn.py"],
    deps = [
        ":common",
        # flax:core dep,
        # frozendict dep,
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        # sparse_deferred/structs:graph_struct",
        "//dgf/src/learning/jax/layers:homo_gnn_sparse_deferred",
        # jax dep,
    ],
)

py_library(
    name = "flax_train",
    srcs = ["flax_train.py"],
    deps = [
        ":common",
        # clu/metric_writers dep,
        "//dgf/src/learning/ten_lines:common",
        "//dgf/src/util:log",
        "//dgf/src/util:util_py",
        # flax:core dep,
        # jax dep,
        # jaxtyping dep,
        # optax dep,
        # orbax/checkpoint dep,
        # tqdm dep,
    ],
)

py_library(
    name = "message_passing",
    srcs = ["message_passing.py"],
    deps = [
        # sparse_deferred",
        # sparse_deferred/jax",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        # jax dep,
    ],
)

# Test
# ====

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        ":common",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # chex dep,
        # jax dep,
    ],
)

py_test(
    name = "flax_gnn_test",
    srcs = ["flax_gnn_test.py"],
    deps = [
        ":flax_gnn",
        ":test_utils",
        # sparse_deferred/structs:graph_struct",
        # absl/testing:absltest dep,
        # clu dep,
        "//dgf/src/learning/jax/layers:homo_gnn_sparse_deferred",
        # flax:core dep,
        # jax dep,
        # numpy dep,
    ],
)

py_test(
    name = "flax_train_test",
    srcs = ["flax_train_test.py"],
    deps = [
        ":flax_train",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # clu dep,
        # flax:core dep,
        # jax dep,
        # numpy dep,
        # optax dep,
    ],
)

py_test(
    name = "message_passing_test",
    srcs = ["message_passing_test.py"],
    deps = [
        ":message_passing",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/util:test_util",
        # jax dep,
    ],
)
