load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "vertex",
    srcs = ["vertex.py"],
    type_checking = True,
    deps = [
        "//dgf/src/learning/ten_lines:common",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util/weak_dep:weak_dep_aiplatform",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # jax dep,
        # yaml dep,
    ],
)

py_test(
    name = "vertex_test",
    srcs = ["vertex_test.py"],
    shard_count = 5,
    type_checking = True,
    deps = [
        ":vertex",
        # absl/testing:absltest dep,
        "//dgf/src/learning/ten_lines:common",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # google/cloud/aiplatform dep,
        # yaml dep,
    ],
)

py_test(
    name = "vertex_integration_test",
    timeout = "eternal",
    srcs = ["vertex_integration_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    type_checking = True,
    deps = [
        ":vertex",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/learning/ten_lines:node_prediction",
        "//dgf/src/util:gen_test_graph",
        # google/cloud/aiplatform dep,  # Weak dependency.
    ],
)

py_test(
    name = "vertex_integration_2_test",
    timeout = "eternal",
    srcs = ["vertex_integration_2_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    type_checking = True,
    deps = [
        ":vertex",
        # absl/testing:absltest dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:cache",
        "//dgf/src/io:tf",
        "//dgf/src/io/gcp:spanner_graph",
        "//dgf/src/learning/ten_lines:common",
        "//dgf/src/learning/ten_lines:node_prediction",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling/gcp:spanner_graph_sampler",
        # google/cloud/aiplatform dep,  # Weak dependency.
        # numpy dep,
        # tensorflow dep,
    ],
)
