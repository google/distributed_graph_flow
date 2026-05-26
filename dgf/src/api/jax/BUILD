load("@rules_python//python:py_library.bzl", "py_library")

package(
    
    default_visibility = ["//visibility:public"],
)

# Library
# =====

py_library(
    name = "jax",
    srcs = ["__init__.py"],
    deps = [
        ":layers",
        "//dgf/src/learning/jax:common",
        "//dgf/src/learning/jax:flax_gnn",
        "//dgf/src/learning/jax:flax_train",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    deps = [
        "//dgf/src/learning/jax/layers:classification",
        "//dgf/src/learning/jax/layers:hetero_gnn",
        "//dgf/src/learning/jax/layers:hetero_graph_attention_network",
        "//dgf/src/learning/jax/layers:homo_gnn_sparse_deferred",
        "//dgf/src/learning/jax/layers:mlp",
        "//dgf/src/learning/jax/layers:preprocess",
        "//dgf/src/learning/jax/layers:residual_mlp",
        "//dgf/src/learning/jax/layers:standard",
    ],
)
