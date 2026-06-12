load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_library")

load("//dgf/src/util:compile.bzl", "all_proto_library")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Library
# =====

py_library(
    name = "distributed_graph",
    srcs = ["distributed_graph.py"],
    deps = [
        ":beam_coders",
        ":in_memory_graph",
        ":numpy_coder",
        ":schema",
        # apache_beam dep,
    ],
)

py_library(
    name = "schema",
    srcs = ["schema.py"],
    deps = [
        # dataclasses_json dep,
    ],
)

py_library(
    name = "numpy_coder",
    srcs = ["numpy_coder.py"],
    deps = [
        # apache_beam dep,
        # numpy dep,
    ],
)

py_library(
    name = "histogram",
    srcs = ["histogram.py"],
    deps = [
        # dataclasses_json dep,
    ],
)

py_library(
    name = "statistics",
    srcs = ["statistics.py"],
    deps = [
        ":histogram",
        # dataclasses_json dep,
        "//dgf/src/analyse:reservoir_sampling",
        "//dgf/src/util:util_py",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        # altair dep,
        # dataclasses_json dep,
        # numpy dep,
        # pandas dep,
    ],
)

py_library(
    name = "in_memory_graph",
    srcs = ["in_memory_graph.py"],
    deps = [
        # numpy dep,
    ],
)

# Small, generic c++ in-memory graph representation.
cc_library(
    name = "in_memory_graph_cc",
    hdrs = ["in_memory_graph.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/types:span",
    ],
)

py_library(
    name = "jax_in_memory_graph",
    srcs = ["jax_in_memory_graph.py"],
    deps = [
        # jax dep,
    ],
)

py_library(
    name = "tf_in_memory_graph",
    srcs = ["tf_in_memory_graph.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "padding",
    srcs = ["padding.py"],
    deps = [
        # numpy dep,
    ],
)

py_library(
    name = "evaluation_metrics",
    srcs = ["evaluation_metrics.py"],
    deps = [
        # dataclasses_json dep,
        "//dgf/src/learning:training_configuration",
    ],
)

py_library(
    name = "gf_metadata",
    srcs = ["gf_metadata.py"],
    deps = [
        # dataclasses_json dep,
    ],
)

py_library(
    name = "beam_coders",
    srcs = ["beam_coders.py"],
    deps = [
        ":in_memory_graph",
        ":schema",
        ":statistics",
        # apache_beam dep,
    ],
)

cc_library(
    name = "schema_cc",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/strings",
    ],
)

nanobind_library(
    name = "schema_nb",
    srcs = ["schema_nb.cc"],
    hdrs = ["schema_nb.h"],
    deps = [
        ":schema_cc",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "//dgf/src/util",
        "//dgf/src/util:nanobind_util",
    ],
)

# Reusable heavier nanobind library for in-memory graphs.
nanobind_library(
    name = "in_memory_graph_nb",
    srcs = ["in_memory_graph_nb.cc"],
    hdrs = ["in_memory_graph_nb.h"],
    deps = [
        ":in_memory_graph_cc",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "//dgf/src/util",
        "//dgf/src/util:nanobind_util",
    ],
)

# Nanobind
# =========

# Bindings for testing the in-memory graph nanobind library.
nanobind_extension(
    name = "in_memory_graph_nb_test_ext",
    srcs = ["in_memory_graph_nb_test_ext.cc"],
    deps = [
        ":in_memory_graph_nb",
        "@abseil-cpp//absl/status:statusor",
        "//dgf/src/util",
        "//dgf/src/util:status_caster",
    ],
)

nanobind_extension(
    name = "schema_ext",
    srcs = ["schema_ext.cc"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "//dgf/src/data:schema_cc",
        "//dgf/src/data:schema_nb",
        "//dgf/src/util",
        "//dgf/src/util:nanobind_util",
        "//dgf/src/util:status_caster",
    ],
)

# Proto
# =====

all_proto_library(
    name = "tensorflow_proto",
    srcs = ["tensorflow.proto"],
    compile_cc = True,
    deps = [],
)

# Test
# ====

py_test(
    name = "numpy_coder_test",
    srcs = ["numpy_coder_test.py"],
    deps = [
        ":numpy_coder",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "beam_coders_test",
    srcs = ["beam_coders_test.py"],
    deps = [
        ":statistics",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/analyse:reservoir_sampling",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "statistics_test",
    srcs = ["statistics_test.py"],
    deps = [
        ":histogram",
        ":statistics",
        # absl/testing:absltest dep,
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "evaluation_test",
    srcs = ["evaluation_test.py"],
    deps = [
        ":evaluation",
        # absl/testing:absltest dep,
        # dataclasses_json dep,
        # numpy dep,
    ],
)

py_test(
    name = "in_memory_graph_nb_test",
    srcs = ["in_memory_graph_nb_test.py"],
    deps = [
        ":in_memory_graph_nb_test_ext",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/util:gen_test_graph",
    ],
)

py_test(
    name = "schema_cc_test",
    srcs = ["schema_cc_test.py"],
    deps = [
        ":schema_ext",
        # absl/testing:absltest dep,
        "//dgf/src/util:gen_test_graph",
    ],
)

py_test(
    name = "histogram_test",
    srcs = ["histogram_test.py"],
    deps = [
        ":histogram",
        # absl/testing:absltest dep,
    ],
)
