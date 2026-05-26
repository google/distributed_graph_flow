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
    srcs = ["all.py"],
    deps = [
        ":classification",
        ":hetero_gnn",
        ":hetero_graph_attention_network",
        ":homo_gnn_sparse_deferred",
        ":mlp",
        ":preprocess",
        ":regression",
        ":residual_mlp",
        ":standard",
    ],
)

py_library(
    name = "mlp",
    srcs = ["mlp.py"],
    deps = [
        # absl/logging dep,
        # chex dep,
        "//dgf/src/learning/jax:common",
        # flax:core dep,
        # jax dep,
        # jaxtyping dep,
    ],
)

py_library(
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        ":registry",
        ":standard",
        # dataclasses_json dep,
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/learning/jax:common",
        "//dgf/src/transform:homogenize",
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "residual_mlp",
    srcs = ["residual_mlp.py"],
    deps = [
        ":registry",
        # dataclasses_json dep,
        "//dgf/src/learning/jax:common",
        # flax:core dep,
        # jax dep,
        # jaxtyping dep,
    ],
)

py_library(
    name = "classification",
    srcs = ["classification.py"],
    deps = [
        ":registry",
        # dataclasses_json dep,
        "//dgf/src/learning/jax:common",
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "registry",
    srcs = ["registry.py"],
    deps = ["//dgf/src/util:dataclass_registry"],
)

py_library(
    name = "hetero_gnn",
    srcs = ["hetero_gnn.py"],
    deps = [
        ":registry",
        ":residual_mlp",
        ":standard",
        # dataclasses_json dep,
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/learning/jax:common",
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "hetero_graph_attention_network",
    srcs = ["hetero_graph_attention_network.py"],
    deps = [
        ":hetero_gnn",
        ":registry",
        ":standard",
        # dataclasses_json dep,
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/learning/jax:common",
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "homo_gnn_sparse_deferred",
    srcs = ["homo_gnn_sparse_deferred.py"],
    deps = [
        ":mlp",
        # sparse_deferred",
        # sparse_deferred/jax",
        # sparse_deferred/nn:edges",
        # sparse_deferred/structs:graph_struct",
        "//dgf/src/learning/jax:common",
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "regression",
    srcs = ["regression.py"],
    deps = [
        ":registry",
        # dataclasses_json dep,
        "//dgf/src/learning/jax:common",
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "standard",
    srcs = ["standard.py"],
    deps = [
        ":registry",
        # dataclasses_json dep,
        "//dgf/src/learning/jax:common",
        # flax:core dep,
        # jax dep,
        # jaxtyping dep,
    ],
)

# Test
# ====

py_test(
    name = "preprocess_test",
    srcs = ["preprocess_test.py"],
    deps = [
        ":preprocess",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:jax",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # jax dep,
    ],
)

py_test(
    name = "mlp_test",
    srcs = ["mlp_test.py"],
    deps = [
        ":mlp",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # flax:core dep,
        # jax dep,
    ],
)

py_test(
    name = "residual_mlp_test",
    srcs = ["residual_mlp_test.py"],
    deps = [
        ":residual_mlp",
        # absl/testing:absltest dep,
        # jax dep,
    ],
)

py_test(
    name = "classification_test",
    srcs = ["classification_test.py"],
    deps = [
        ":classification",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
    ],
)

py_test(
    name = "hetero_gnn_test",
    srcs = ["hetero_gnn_test.py"],
    deps = [
        ":hetero_gnn",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        # jax dep,
    ],
)

py_test(
    name = "hetero_graph_attention_network_test",
    srcs = ["hetero_graph_attention_network_test.py"],
    deps = [
        ":hetero_graph_attention_network",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        # jax dep,
    ],
)

py_test(
    name = "homo_gnn_sparse_deferred_test",
    srcs = ["homo_gnn_sparse_deferred_test.py"],
    deps = [
        ":homo_gnn_sparse_deferred",
        # sparse_deferred/jax",
        # sparse_deferred/structs:graph_struct",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/learning/jax:test_utils",
        # flax:core dep,
        # jax dep,
    ],
)

py_test(
    name = "standard_test",
    srcs = ["standard_test.py"],
    deps = [
        ":standard",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
    ],
)

py_test(
    name = "registry_test",
    srcs = ["registry_test.py"],
    deps = [
        ":all",
        ":registry",
        # absl/testing:absltest dep,
    ],
)
