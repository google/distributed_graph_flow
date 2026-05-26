load("//devtools/python/blaze:pytype.bzl", "py_binary")

package(
    
    default_visibility = ["//dgf/src:__subpackages__"],
)

py_binary(
    name = "train_model_with_tf_gnn",
    srcs = ["train_model_with_tf_gnn.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "//dgf",
        "//dgf/src/learning/google/tfgnn:trainer_tfgnn",
    ],
)

py_binary(
    name = "create_graph_samples_semi_distributed_v1",
    srcs = ["create_graph_samples_semi_distributed_v1.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # apache_beam dep,
        "//dgf",
        "//dgf:dgf_beam",
    ],
)

py_binary(
    name = "create_in_memory_graph_reports",
    srcs = ["create_in_memory_graph_reports.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/flags:argparse_flags dep,
        "//dgf",
        # numpy dep,
    ],
)

py_binary(
    name = "link_model",
    srcs = ["link_model.py"],
    deps = [
        "//learning/brain/research/jax:gpu_support",
        # absl:app dep,
        # absl/logging dep,
        "//dgf",
    ],
)

py_binary(
    name = "ten_lines_arxiv",
    srcs = ["ten_lines_arxiv.py"],
    deps = [
        "//learning/brain/research/jax:gpu_support",
        # absl:app dep,
        # absl/logging dep,
        "//dgf",
    ],
)
