load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(default_applicable_licenses = ["//dgf:license"])

# Libraries
# =========

py_library(
    name = "in_process_io",
    srcs = ["in_process_io.py"],
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
    deps = [
        ":utils",
        "//dgf",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [],
)

py_library(
    name = "in_process_sampling",
    srcs = ["in_process_sampling.py"],
    deps = [
        ":utils",
        "//dgf",
        "//dgf/src/util:log",
    ],
)

# Binaries
# ========

py_binary(
    name = "in_process_io_main",
    srcs = ["in_process_io_main.py"],
    deps = [
        ":in_process_io",
        # "//learning/brain/research/jax:gpu_support",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_binary(
    name = "bq_parquet_io_main",
    srcs = ["bq_parquet_io_main.py"],
    deps = [
        ":bq_parquet_io",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_binary(
    name = "in_process_sampling_main",
    srcs = ["in_process_sampling_main.py"],
    deps = [
        ":in_process_sampling",
        #  "//learning/brain/research/jax:gpu_support",
        # absl:app dep,
        # absl/flags dep,
    ],
)

# Tests
# =====

py_test(
    name = "in_process_io_test",
    srcs = ["in_process_io_test.py"],
    deps = [
        ":in_process_io",
        # absl/testing:absltest dep,
        "//dgf",
        "//dgf/src/util:gen_test_graph",
    ],
)

py_test(
    name = "in_process_sampling_test",
    srcs = ["in_process_sampling_test.py"],
    deps = [
        ":in_process_sampling",
        # absl/testing:absltest dep,
        "//dgf/src/util:gen_test_graph",
    ],
)
