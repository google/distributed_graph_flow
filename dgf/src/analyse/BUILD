load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Libraries
# =========

py_library(
    name = "feature_statistics",
    srcs = ["feature_statistics.py"],
    deps = [
        ":reservoir_sampling",
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        # numpy dep,
    ],
)

py_library(
    name = "reservoir_sampling",
    srcs = ["reservoir_sampling.py"],
    deps = [
        # apache_beam dep,
        # numpy dep,
    ],
)

py_library(
    name = "in_memory_feature_statistics",
    srcs = ["in_memory_feature_statistics.py"],
    deps = [
        ":feature_statistics",
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
    ],
)

py_library(
    name = "padding",
    srcs = ["padding.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    deps = [
        "//dgf/src/sampling:config",
    ],
)

py_library(
    name = "in_process_feature_statistics",
    srcs = ["in_process_feature_statistics.py"],
    deps = [
        ":feature_statistics",
        # apache_beam dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
    ],
)

py_library(
    name = "histogram",
    srcs = ["histogram.py"],
    deps = [
        "//dgf/src/data:histogram",
        # numpy dep,
    ],
)

py_test(
    name = "histogram_test",
    srcs = ["histogram_test.py"],
    deps = [
        ":histogram",
        # absl/testing:absltest dep,
        "//dgf/src/data:histogram",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_library(
    name = "print_schema",
    srcs = ["print_schema.py"],
    deps = [
        "//dgf/src/data:schema",
        # tabulate dep,
    ],
)

py_library(
    name = "schema",
    srcs = ["schema.py"],
    deps = [
        "//dgf/src/data:schema",
        "//dgf/src/util:log",
    ],
)

# Test
# =========

py_test(
    name = "feature_statistics_test",
    srcs = ["feature_statistics_test.py"],
    deps = [
        ":feature_statistics",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/io:graph_in_beam",
        "//dgf/src/io:statistics",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "reservoir_sampling_test",
    srcs = ["reservoir_sampling_test.py"],
    deps = [
        ":reservoir_sampling",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        # numpy dep,
    ],
)

py_test(
    name = "in_memory_feature_statistics_test",
    srcs = ["in_memory_feature_statistics_test.py"],
    deps = [
        ":in_memory_feature_statistics",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/data:statistics",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "padding_test",
    srcs = ["padding_test.py"],
    deps = [
        ":padding",
        # absl/testing:absltest dep,
        "//dgf/src/data:padding",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "sampling_test",
    srcs = ["sampling_test.py"],
    deps = [
        ":sampling",
        # absl/testing:absltest dep,
        "//dgf/src/sampling:config",
        "//dgf/src/util:gen_test_graph",
    ],
)

py_test(
    name = "in_process_feature_statistics_test",
    srcs = ["in_process_feature_statistics_test.py"],
    deps = [
        ":in_process_feature_statistics",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/data:statistics",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "print_schema_test",
    srcs = ["print_schema_test.py"],
    deps = [
        ":print_schema",
        # absl/testing:absltest dep,
        "//dgf/src/util:gen_test_graph",
    ],
)

py_test(
    name = "schema_test",
    srcs = ["schema_test.py"],
    deps = [
        ":schema",
        # absl/testing:absltest dep,
        "//dgf/src/data:schema",
    ],
)
