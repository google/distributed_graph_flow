load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
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
    name = "conform",
    srcs = ["conform.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        "//dgf/src/util:log",
        # numpy dep,
        # pandas dep,
    ],
)

py_library(
    name = "table_2_graph",
    srcs = ["table_2_graph.py"],
    type_checking = True,
    deps = [
        "//dgf/src/analyse:text_detection",
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
    type_checking = True,
    deps = [
        ":timeseries_padding",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/util:temporal",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
    ],
)

py_library(
    name = "normalize_base",
    srcs = ["normalize_base.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:schema",
        "//dgf/src/util:dataclass_registry",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
    ],
)

py_library(
    name = "normalize_text",
    srcs = ["normalize_text.py"],
    type_checking = True,
    deps = [
        ":normalize_base",
        # dataclasses_json dep,
        "//dgf/src/data:schema",
        "//dgf/src/util/weak_dep:weak_dep_keras_hub",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow_hub",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow_text",
        # numpy dep,
    ],
)

py_library(
    name = "normalize_text_test_util",
    testonly = True,
    srcs = ["normalize_text_test_util.py"],
    type_checking = True,
    deps = [
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
    ],
)

py_library(
    name = "normalize",
    srcs = ["normalize.py"],
    type_checking = True,
    deps = [
        ":normalize_base",
        ":normalize_text",
        # dataclasses_json dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/util:log",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # jax dep,
        # numpy dep,
    ],
)

py_library(
    name = "extract",
    srcs = ["extract.py"],
    type_checking = True,
    deps = [
        ":schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
    ],
)

py_library(
    name = "schema",
    srcs = ["schema.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:schema",
    ],
)

py_library(
    name = "in_memory_graph_filter",
    srcs = ["in_memory_graph_filter.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:in_memory_graph",
        # numpy dep,
        # tqdm dep,
    ],
)

py_library(
    name = "nx",
    srcs = ["nx.py"],
    type_checking = True,
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
    type_checking = True,
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
    type_checking = True,
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:temporal",
        # numpy dep,
    ],
)

py_library(
    name = "timeseries_padding",
    srcs = ["timeseries_padding.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        "//dgf/src/util:temporal",
        # numpy dep,
    ],
)

py_library(
    name = "combine_graph_snapshots",
    srcs = ["combine_graph_snapshots.py"],
    type_checking = True,
    deps = [
        "//dgf/src/analyse:schema",
        "//dgf/src/data:graph_snapshots",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        "//dgf/src/util:log",
        "//dgf/src/util:temporal",
        # numpy dep,
    ],
)

py_library(
    name = "canonicalize",
    srcs = ["canonicalize.py"],
    type_checking = True,
    deps = [
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        # numpy dep,
    ],
)

py_library(
    name = "perturb",
    srcs = ["perturb.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:temporal",
        # numpy dep,
    ],
)

# Tests
# =====

py_test(
    name = "table_2_graph_test",
    srcs = ["table_2_graph_test.py"],
    type_checking = True,
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
    type_checking = True,
    deps = [
        ":merge",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:temporal",
        "//dgf/src/util:test_util",
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "normalize_text_test",
    srcs = ["normalize_text_test.py"],
    type_checking = True,
    deps = [
        ":normalize",
        ":normalize_text",
        ":normalize_text_test_util",
        # absl/testing:absltest dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/io:tf",
        # keras_hub dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "normalize_test",
    srcs = ["normalize_test.py"],
    type_checking = True,
    deps = [
        ":normalize",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
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
    type_checking = True,
    deps = [
        ":extract",
        ":schema",
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
    type_checking = True,
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
    type_checking = True,
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
    type_checking = True,
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
    type_checking = True,
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
    name = "timeseries_padding_test",
    srcs = ["timeseries_padding_test.py"],
    type_checking = True,
    deps = [
        ":timeseries_padding",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "combine_graph_snapshots_test",
    srcs = ["combine_graph_snapshots_test.py"],
    type_checking = True,
    deps = [
        ":combine_graph_snapshots",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:graph_snapshots_metadata",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_constants",
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/io:graph_snapshots",
        "//dgf/src/io:graph_snapshots_metadata",
        "//dgf/src/io:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)

py_test(
    name = "canonicalize_test",
    srcs = ["canonicalize_test.py"],
    type_checking = True,
    deps = [
        ":canonicalize",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "perturb_test",
    srcs = ["perturb_test.py"],
    type_checking = True,
    deps = [
        ":perturb",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "conform_test",
    srcs = ["conform_test.py"],
    type_checking = True,
    deps = [
        ":conform",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
        # pandas dep,
    ],
)
