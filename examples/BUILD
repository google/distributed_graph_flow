load("//devtools/python/blaze:pytype.bzl", "py_binary")

package(
    
    default_visibility = ["//dgf/src:__subpackages__"],
)

# Binary
# =========

py_binary(
    name = "node_classification_model",
    srcs = ["node_classification_model.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_binary(
    name = "node_classification_model_advanced",
    srcs = ["node_classification_model_advanced.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf",
        # flax:core dep,
        # jax dep,
        # jaxtyping dep,
        # matplotlib dep,
        # numpy dep,
        # optax dep,
    ],
)

py_binary(
    name = "count_nodes",
    srcs = ["count_nodes.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # apache_beam dep,
        "//dgf",
    ],
)

py_binary(
    name = "feature_statistics_on_hgraph",
    srcs = ["feature_statistics_on_hgraph.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # apache_beam dep,
        "//dgf",
    ],
)

py_binary(
    name = "feature_statistics_on_graph_samples",
    srcs = ["feature_statistics_on_graph_samples.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # apache_beam dep,
        "//dgf",
        # tensorflow_gnn/proto dep,
    ],
)

py_binary(
    name = "create_graph_samples_in_process",
    srcs = ["create_graph_samples_in_process.py"],
    deps = [
        "//file/colossus/public:cns",
        # absl:app dep,
        # absl/flags dep,
        "//dgf",
        # numpy dep,
        # tqdm dep,
    ],
)

py_binary(
    name = "create_graph_samples_semi_distributed_v2",
    srcs = ["create_graph_samples_semi_distributed_v2.py"],
    deps = [
        "//file/colossus/public:cns",
        # absl:app dep,
        # absl/flags dep,
        # apache_beam dep,
        "//dgf",
    ],
)

py_binary(
    name = "create_graph_samples_distributed",
    srcs = ["create_graph_samples_distributed.py"],
    deps = [
        "//file/colossus/public:cns",
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "//dgf",
    ],
)

py_binary(
    name = "node_classification_pyg",
    srcs = ["node_classification_pyg.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf",
        # matplotlib dep,
        # numpy dep,
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch_geometric:data",
        "//third_party/py/torch_geometric:nn",
        # tqdm dep,
    ],
)
