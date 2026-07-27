load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_library")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Libraries
# =========

py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        # dataclasses_json dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:schema",
        "//dgf/src/util:temporal",
    ],
)

py_library(
    name = "temporal",
    srcs = ["temporal.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/util:temporal",
        # numpy dep,
    ],
)

py_library(
    name = "in_memory_sampler",
    srcs = ["in_memory_sampler.py"],
    deps = [
        ":_in_memory_sampler_ext",
        ":config",
        ":temporal",
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",  # buildcleaner: keep
        "//dgf/src/data:schema",
        "//dgf/src/util:temporal",
        # numpy dep,
    ],
)

cc_library(
    name = "in_memory_sampler_cc",
    srcs = ["in_memory_sampler.cc"],
    hdrs = ["in_memory_sampler.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "//dgf/src/util",
    ],
)

py_library(
    name = "beam_semi_distributed_sampler",
    srcs = ["beam_semi_distributed_sampler.py"],
    deps = [
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
    ],
)

py_library(
    name = "beam_semi_distributed_sampler_v1",
    srcs = ["beam_semi_distributed_sampler_v1.py"],
    deps = [
        ":config",
        ":in_memory_sampler",
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        # numpy dep,
    ],
)

py_library(
    name = "beam_semi_distributed_sampler_v2",
    srcs = ["beam_semi_distributed_sampler_v2.py"],
    deps = [
        ":config",
        ":in_memory_sampler",
        # apache_beam dep,
        # apache_beam/utils:low_level dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:feature_format",
        "//dgf/src/io:graph_in_beam",
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/io:schema",
        "//dgf/src/transform:schema",
        # numpy dep,
    ],
)

nanobind_library(
    name = "in_memory_sampler_nb",
    srcs = ["in_memory_sampler_nb.cc"],
    hdrs = ["in_memory_sampler_nb.h"],
    deps = [
        ":in_memory_sampler_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "//dgf/src/data:schema_cc",
        "//dgf/src/util",
        "//dgf/src/util:nanobind_util",
    ],
)

# Nanobind
# =========
nanobind_extension(
    name = "_in_memory_sampler_ext",
    srcs = ["in_memory_sampler_ext.cc"],
    # TODO(gbm): Enable those copts externally.
    # copts = [
    #     "-fexceptions",
    #     "-fno-strict-aliasing",
    # ],
    deps = [
        ":in_memory_sampler_cc",
        ":in_memory_sampler_nb",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "//dgf/src/data:schema_cc",
        "//dgf/src/data:schema_nb",
        "//dgf/src/util",
        "//dgf/src/util:concurrency",
        "//dgf/src/util:nanobind_util",
        "//dgf/src/util:status_caster",
    ],
)

# Test
# =========

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    deps = [
        ":config",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
    ],
)

py_test(
    name = "in_memory_sampler_test",
    srcs = ["in_memory_sampler_test.py"],
    deps = [
        ":_in_memory_sampler_ext",
        ":config",
        ":in_memory_sampler",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/transform:temporal",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
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
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:temporal",
        # numpy dep,
    ],
)

cc_test(
    name = "in_memory_sampler_cc_test",
    srcs = ["in_memory_sampler_test.cc"],
    deps = [
        ":in_memory_sampler_cc",
        "@googletest//:gtest_main",
        "@abseil-cpp//absl/status",
        "//dgf/src/util:test_util_cc",
    ],
)

py_test(
    name = "beam_semi_distributed_sampler_test",
    srcs = ["beam_semi_distributed_sampler_test.py"],
    deps = [
        ":beam_semi_distributed_sampler",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/io:hgraph_in_beam",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "beam_semi_distributed_sampler_v1_test",
    srcs = ["beam_semi_distributed_sampler_v1_test.py"],
    deps = [
        ":beam_semi_distributed_sampler",
        ":beam_semi_distributed_sampler_v1",
        ":config",
        ":in_memory_sampler",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/io:hgraph_in_beam",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "beam_semi_distributed_sampler_v2_test",
    srcs = ["beam_semi_distributed_sampler_v2_test.py"],
    deps = [
        ":beam_semi_distributed_sampler",
        ":beam_semi_distributed_sampler_v2",
        ":config",
        ":in_memory_sampler",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_in_beam",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)
