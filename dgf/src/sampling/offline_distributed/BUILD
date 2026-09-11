load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Libraries
# =========

py_library(
    name = "offline_distributed_gcp",
    srcs = ["offline_distributed_gcp.py"],
    deps = [
        "//dgf/src/data:schema",
        "//dgf/src/io:schema",
        "//dgf/src/sampling:config",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//third_party/py/google/cloud/aiplatform",
        # tqdm dep,
    ],
)

# Test
# =========

py_test(
    name = "offline_distributed_gcp_test",
    srcs = ["offline_distributed_gcp_test.py"],
    deps = [
        ":offline_distributed_gcp",
        # absl/testing:absltest dep,
        "//dgf/src/sampling:config",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:log",
        "//third_party/py/google/cloud/aiplatform",
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
    deps = [
        ":offline_distributed_gcp",
        # absl/flags dep,
        # absl/testing:absltest dep,
        "//dgf/src/io:schema",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/sampling:config",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
    ],
)
