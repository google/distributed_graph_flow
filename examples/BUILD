load("//devtools/python/blaze:pytype.bzl", "py_binary")

package(
    
    default_visibility = ["//dgf/src:__subpackages__"],
)

# Binary
# =========

py_binary(
    name = "count_nodes",
    srcs = ["count_nodes.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # apache_beam dep,
        "//dgf",
        "//dgf:dgf_beam",
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
        "//dgf:dgf_beam",
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
        "//dgf:dgf_beam",
        # tensorflow_gnn/proto dep,
    ],
)

py_binary(
    name = "convert_hgraph",
    srcs = ["convert_hgraph.py"],
    deps = [
        "//file/colossus/public:cns",
        # absl:app dep,
        # absl/flags dep,
        # apache_beam dep,
        "//dgf",
        "//dgf:dgf_beam",
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
    name = "convert_hgraph_to_gf_graph",
    srcs = ["convert_hgraph_to_gf_graph.py"],
    deps = [
        "//file/colossus/public:cns",
        # absl:app dep,
        # absl/flags dep,
        # apache_beam dep,
        "//dgf",
        "//dgf:dgf_beam",
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
        "//dgf:dgf_beam",
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
    name = "validate_exported_graph",
    srcs = ["validate_exported_graph.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf",
    ],
)
