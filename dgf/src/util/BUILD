load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_library")

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Libraries
# =========

py_library(
    name = "gen_test_graph",
    srcs = ["gen_test_graph.py"],
    deps = [
        ":filesystem",
        ":proto",
        ":shard",
        ":weak_dep",
        # bagz dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:gf_metadata",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/io:graph_in_beam",
        "//dgf/src/io:schema",
        "//dgf/src/io/gcp:common",
        "//dgf/src/io/gcp:spanner_graph_metadata",
        # fastavro dep,
        # numpy dep,
        # pyarrow:pyarrow_modules dep,
        # tensorflow dep,
        # tensorflow_gnn/proto dep,  # buildcleaner: keep
    ],
)

py_library(
    name = "proto",
    srcs = ["proto.py"],
    deps = [
        ":filesystem",
    ],
)

py_library(
    name = "filesystem",
    srcs = ["filesystem.py"],
    deps = [
        # absl/logging dep,
        # etils/epath dep,
        # fsspec dep,
    ],
)

py_library(
    name = "shard",
    srcs = ["shard.py"],
    deps = [
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        # absl/flags dep,
        # absl/testing:absltest dep,
        "//dgf/src/data:distributed_graph",
        # immutabledict dep,
        # jax dep,
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "json",
    srcs = ["json.py"],
    deps = [
        ":filesystem",
    ],
)

cc_library(
    name = "status_caster",
    hdrs = ["status_caster.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "tf_feature_util",
    srcs = ["tf_feature_util.cc"],
    hdrs = ["tf_feature_util.h"],
    deps = [
        "@abseil-cpp//absl/strings",
        "@protobuf//:protobuf" , #:protobuf_lite",
        "//dgf/src/data:tensorflow_cc_proto",
    ],
)

nanobind_library(
    name = "nanobind_util",
    srcs = ["nanobind_util.cc"],
    hdrs = ["nanobind_util.h"],
    deps = [
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_library(
    name = "concurrency",
    srcs = ["concurrency.cc"],
    hdrs = ["concurrency.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
    ],
)

py_library(
    name = "util_py",
    srcs = ["util.py"],
    deps = [
        ":log",
        # numpy dep,
    ],
)

py_library(
    name = "log",
    srcs = ["log.py"],
    deps = [
        # absl/logging dep,
    ],
)

py_library(
    name = "temporal",
    srcs = ["temporal.py"],
    deps = [
        "//dgf/src/data:schema",
        # numpy dep,
    ],
)

cc_library(
    name = "test_util_cc",
    testonly = True,
    hdrs = ["test_util.h"],
    deps = ["@googletest//:gtest"],
)

py_library(
    name = "weak_dep",
    srcs = ["weak_dep.py"],
    deps = [
    ],
)

# Nanobind
# =========

nanobind_extension(
    name = "nanobind_util_test_ext",
    srcs = ["nanobind_util_test_ext.cc"],
    deps = [
        ":nanobind_util",
        ":status_caster",
        ":util",
        "@abseil-cpp//absl/status",
    ],
)

nanobind_extension(
    name = "util_ext",
    srcs = ["util_ext.cc"],
    deps = [
        ":nanobind_util",
        ":status_caster",
        ":util",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

# Bzl Libraries
# =============

bzl_library(
    name = "compile_bzl",
    srcs = ["compile.bzl"],
)

py_library(
    name = "dataclass_registry",
    srcs = ["dataclass_registry.py"],
    deps = [
        # dataclasses_json dep,
    ],
)

# Test
# =========

py_test(
    name = "gen_test_graph_test",
    srcs = ["gen_test_graph_test.py"],
    deps = [
        ":gen_test_graph",
        ":test_util",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)

py_test(
    name = "shard_test",
    srcs = ["shard_test.py"],
    deps = [
        ":shard",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
    ],
)

py_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    data = ["//test_data"],
    deps = [
        ":test_util",
        # absl/testing:absltest dep,
        # numpy dep,
    ],
)

py_test(
    name = "json_test",
    srcs = ["json_test.py"],
    deps = [
        ":json",
        # absl/testing:absltest dep,
    ],
)

py_test(
    name = "filesystem_integration_test",
    srcs = ["filesystem_integration_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":filesystem",
        # absl/flags dep,
        # absl/testing:absltest dep,
    ],
)

cc_test(
    name = "concurrency_test",
    srcs = ["concurrency_test.cc"],
    deps = [
        ":concurrency",
        ":test_util_cc",
        "@googletest//:gtest_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
    ],
)

py_test(
    name = "nanobind_util_test",
    srcs = ["nanobind_util_test.py"],
    deps = [
        ":nanobind_util_test_ext",
        ":test_util",
        # absl/testing:absltest dep,
        # numpy dep,
    ],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "@googletest//:gtest_main",
    ],
)

py_test(
    name = "filesystem_test",
    srcs = ["filesystem_test.py"],
    deps = [
        ":filesystem",
        # absl/testing:absltest dep,
    ],
)

py_test(
    name = "util_py_test",
    srcs = ["util_test.py"],
    main = "util_test.py",
    deps = [
        ":util_py",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
    ],
)

py_test(
    name = "dataclass_registry_test",
    srcs = ["dataclass_registry_test.py"],
    deps = [
        ":dataclass_registry",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # dataclasses_json dep,
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "util_ext_test",
    srcs = ["util_ext_test.py"],
    deps = [
        ":util_ext",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
    ],
)

py_test(
    name = "log_test",
    srcs = ["log_test.py"],
    deps = [
        ":log",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:flagsaver dep,
    ],
)

py_test(
    name = "weak_dep_test",
    srcs = ["weak_dep_test.py"],
    deps = [
        ":weak_dep",
        # absl/testing:absltest dep,
    ],
)

py_test(
    name = "temporal_test",
    srcs = ["temporal_test.py"],
    deps = [
        ":temporal",
        # absl/testing:absltest dep,
        "//dgf/src/data:schema",
        # numpy dep,
    ],
)
