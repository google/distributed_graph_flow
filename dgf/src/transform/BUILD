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
    name = "table_2_graph",
    srcs = ["table_2_graph.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        # numpy dep,
        # pandas dep,
    ],
)

py_library(
    name = "merge",
    srcs = ["merge.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:tf_in_memory_graph",
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "normalize",
    srcs = ["normalize.py"],
    deps = [
        # dataclasses_json dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/util:dataclass_registry",
        "//dgf/src/util:log",
        # jax dep,
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "extract",
    srcs = ["extract.py"],
    deps = [
        ":schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
    ],
)

py_library(
    name = "schema",
    srcs = ["schema.py"],
    deps = [
        "//dgf/src/data:schema",
    ],
)

py_library(
    name = "in_memory_graph_filter",
    srcs = ["in_memory_graph_filter.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        # numpy dep,
        # tqdm dep,
    ],
)

py_library(
    name = "nx",
    srcs = ["nx.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        # networkx dep,
        # numpy dep,
        # tqdm dep,
    ],
)

py_library(
    name = "homogenize",
    srcs = ["homogenize.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        # jax dep,
        # numpy dep,
    ],
)

py_library(
    name = "temporal",
    srcs = ["temporal.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        # numpy dep,
    ],
)

py_library(
    name = "timeseries",
    srcs = ["timeseries.py"],
    deps = [
        # dataclasses_json dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        "//dgf/src/util:temporal",
        # numpy dep,
    ],
)

# Tests
# =====

py_test(
    name = "table_2_graph_test",
    srcs = ["table_2_graph_test.py"],
    deps = [
        ":table_2_graph",
        # absl/testing:absltest dep,
        "//dgf/src/data:schema",
        # numpy dep,
        # pandas dep,
    ],
)

py_test(
    name = "merge_test",
    srcs = ["merge_test.py"],
    deps = [
        ":merge",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/io:tf",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "normalize_test",
    srcs = ["normalize_test.py"],
    deps = [
        ":normalize",
        # absl/testing:absltest dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/io:tf",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "extract_test",
    srcs = ["extract_test.py"],
    deps = [
        ":extract",
        # absl/testing:absltest dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "schema_test",
    srcs = ["schema_test.py"],
    deps = [
        ":schema",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "in_memory_graph_filter_test",
    srcs = ["in_memory_graph_filter_test.py"],
    deps = [
        ":in_memory_graph_filter",
        # absl/testing:absltest dep,
        "//dgf/src/data:in_memory_graph",
        # numpy dep,
    ],
)

py_test(
    name = "homogenize_test",
    srcs = ["homogenize_test.py"],
    deps = [
        ":homogenize",
        # absl/testing:absltest dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:jax",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # jax dep,
        # numpy dep,
    ],
)

py_test(
    name = "temporal_test",
    srcs = ["temporal_test.py"],
    deps = [
        ":temporal",
        # absl/testing:absltest dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "timeseries_test",
    srcs = ["timeseries_test.py"],
    deps = [
        ":timeseries",
        # absl/testing:absltest dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        # numpy dep,
    ],
)
