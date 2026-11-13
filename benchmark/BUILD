load("@rules_python//python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = ["//dgf/src:__subpackages__"],
)

# Libraries
# =========

py_library(
    name = "in_process_io",
    srcs = ["in_process_io.py"],
    type_checking = True,
    deps = [
        ":utils",
        "//dgf",
        "//dgf/src/io:spanner_graph",
        "//dgf/src/io:spanner_graph_in_memory",
        "//dgf/src/io/gcp:bigquery_graph",
    ],
)

py_library(
    name = "bq_parquet_io",
    srcs = ["bq_parquet_io.py"],
    type_checking = True,
    deps = [
        ":utils",
        "//dgf",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    type_checking = True,
    deps = [],
)

py_library(
    name = "in_process_sampling",
    srcs = ["in_process_sampling.py"],
    type_checking = True,
    deps = [
        ":utils",
        "//dgf",
        "//dgf/src/analyse:padding",
        "//dgf/src/transform:merge",
        "//dgf/src/util:log",
        # numpy dep,
    ],
)

py_library(
    name = "mpnn",
    srcs = ["mpnn.py"],
    type_checking = True,
    deps = [
        ":utils",
        "//dgf",
        "//dgf/src/sampling:config",
        "//dgf/src/util:log",
        # flax:core dep,
        # jax dep,
        # numpy dep,
    ],
)

# Binaries
# ========

py_binary(
    name = "in_process_io_main",
    srcs = ["in_process_io_main.py"],
    type_checking = True,
    deps = [
        ":in_process_io",
        ":utils",
        # "//learning/brain/research/jax:gpu_support",
        # absl:app dep,
        # absl/flags dep,
        "//dgf",
    ],
)

py_binary(
    name = "bq_parquet_io_main",
    srcs = ["bq_parquet_io_main.py"],
    type_checking = True,
    deps = [
        ":bq_parquet_io",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_library(
    name = "bq_sampling",
    srcs = ["bq_sampling.py"],
    type_checking = True,
    deps = [
        ":utils",
        "//dgf",
        "//dgf/src/io/gcp:bigquery_graph",
        "//dgf/src/io/gcp:bigquery_graph_metadata",
    ],
)

py_binary(
    name = "bq_sampling_main",
    srcs = ["bq_sampling_main.py"],
    type_checking = True,
    deps = [
        ":bq_sampling",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_binary(
    name = "in_process_sampling_main",
    srcs = ["in_process_sampling_main.py"],
    type_checking = True,
    deps = [
        ":in_process_sampling",
        #  "//learning/brain/research/jax:gpu_support",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_binary(
    name = "mpnn_main",
    srcs = ["mpnn_main.py"],
    type_checking = True,
    deps = [
        ":mpnn",
        # absl:app dep,
        # absl/flags dep,
    ],
)

# Tests
# =====

py_test(
    name = "in_process_io_test",
    srcs = ["in_process_io_test.py"],
    type_checking = True,
    deps = [
        ":in_process_io",
        ":utils",
        # absl/testing:absltest dep,
        "//dgf",
        "//dgf/src/util:gen_test_graph",
    ],
)

py_test(
    name = "in_process_sampling_test",
    srcs = ["in_process_sampling_test.py"],
    type_checking = True,
    deps = [
        ":in_process_sampling",
        # absl/testing:absltest dep,
        "//dgf/src/util:gen_test_graph",
    ],
)

py_test(
    name = "mpnn_test",
    srcs = ["mpnn_test.py"],
    type_checking = True,
    deps = [
        ":mpnn",
        # absl/testing:absltest dep,
        "//dgf",
        # jax dep,
    ],
)
