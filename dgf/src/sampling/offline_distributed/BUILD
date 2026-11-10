load("@rules_python//python:py_binary.bzl", "py_binary")
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
    name = "common",
    srcs = ["common.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:schema",
        "//dgf/src/io:schema",
        "//dgf/src/sampling:config",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
    ],
)

py_library(
    name = "offline_distributed_gcp",
    srcs = ["offline_distributed_gcp.py"],
    type_checking = True,
    deps = [
        ":common",
        # dataclasses_json dep,
        "//dgf/src/data:schema",
        "//dgf/src/sampling:config",
        "//dgf/src/util:log",
        "//dgf/src/util:util_py",
        "//dgf/src/util/weak_dep:weak_dep_aiplatform",
        # tqdm dep,
    ],
)

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    type_checking = True,
    deps = [":offline_distributed_gcp"],
)

# Test
# =========

py_test(
    name = "offline_distributed_gcp_test",
    srcs = ["offline_distributed_gcp_test.py"],
    type_checking = True,
    deps = [
        ":common",
        ":offline_distributed_gcp",
        # absl/testing:absltest dep,
        "//dgf/src/sampling:config",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:log",
        "//dgf/src/util/weak_dep:weak_dep_aiplatform",
        # google/cloud/aiplatform dep,
    ],
)

# Binaries
# ========

py_binary(
    name = "offline_distributed_gcp_integration_test",
    srcs = ["offline_distributed_gcp_integration_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    type_checking = True,
    deps = [
        ":offline_distributed_gcp",
        # absl/flags dep,
        # absl/testing:absltest dep,
        "//dgf/src/io:schema",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/sampling:config",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        # google/cloud/aiplatform dep,
    ],
)
