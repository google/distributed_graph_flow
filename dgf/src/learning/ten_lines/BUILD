load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")

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
    deps = [
        ":dataset",
        # dataclasses_json dep,
        "//dgf/src/data:schema",
        "//dgf/src/learning/jax:common",
        "//dgf/src/learning/jax/layers:hetero_gnn",
        "//dgf/src/learning/jax/layers:hetero_graph_attention_network",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:util_py",
        # jax dep,
    ],
)

py_library(
    name = "node_prediction",
    srcs = [
        "node_prediction_core_model.py",
        "node_prediction_model.py",
        "node_prediction_train.py",
    ],
    deps = [
        ":common",
        ":dataset",
        ":evaluation",
        ":node_prediction_dataset",
        ":report",
        # dataclasses_json dep,
        "//dgf/src/analyse:print_schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/io:jax",
        "//dgf/src/io:tf",
        "//dgf/src/learning/jax:common",
        "//dgf/src/learning/jax:flax_train",
        "//dgf/src/learning/jax/layers:classification",
        "//dgf/src/learning/jax/layers:hetero_gnn",
        "//dgf/src/learning/jax/layers:preprocess",
        "//dgf/src/learning/jax/layers:registry",
        "//dgf/src/learning/jax/layers:regression",
        "//dgf/src/learning/jax/layers:standard",
        "//dgf/src/plot:network",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/transform:merge",
        "//dgf/src/transform:normalize",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util:util_py",
        # flax:core dep,
        # jax dep,
        # jax/experimental/jax2tf dep,
        # jaxtyping dep,
        # numpy dep,
        # optax dep,
        # orbax/checkpoint dep,
        # tensorflow dep,
        # tqdm dep,
    ],
)

nanobind_extension(
    name = "evaluation_ext",
    srcs = ["evaluation_ext.cc"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "//dgf/src/util:nanobind_util",
        "//dgf/src/util:status_caster",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        ":evaluation_ext",
        "//dgf/src/data:evaluation",
        # jax dep,
        # numpy dep,
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/transform:merge",
        "//dgf/src/util:util_py",
        # numpy dep,
    ],
)

py_library(
    name = "report",
    srcs = ["report.py"],
    deps = [
        ":common",
        # altair dep,
        "//dgf/src/analyse:padding",
        "//dgf/src/analyse:print_schema",
        "//dgf/src/analyse:sampling",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/sampling:config",
        "//dgf/src/util:util_py",
        # pandas dep,
    ],
)

py_library(
    name = "link_prediction_train",
    srcs = ["link_prediction_train.py"],
    deps = [
        ":common",
        ":evaluation",
        ":link_prediction_core_model",
        ":link_prediction_dataset",
        ":link_prediction_model",
        "//dgf/src/analyse:print_schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/generate:edge_neighbor_generator",
        "//dgf/src/io:jax",
        "//dgf/src/learning/jax:flax_train",
        "//dgf/src/learning/jax/layers:hetero_gnn",
        "//dgf/src/learning/jax/layers:preprocess",
        "//dgf/src/learning/jax/layers:residual_mlp",
        "//dgf/src/learning/jax/layers:standard",
        "//dgf/src/plot:network",
        "//dgf/src/sampling:config",
        "//dgf/src/transform:merge",
        "//dgf/src/transform:normalize",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util:util_py",
        # jax dep,
        # jaxtyping dep,
        # numpy dep,
        # optax dep,
        # tqdm dep,
    ],
)

py_library(
    name = "link_prediction_core_model",
    srcs = ["link_prediction_core_model.py"],
    deps = [
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/learning/jax:common",
        "//dgf/src/learning/jax/layers:hetero_gnn",
        "//dgf/src/learning/jax/layers:preprocess",
        "//dgf/src/learning/jax/layers:registry",
        "//dgf/src/learning/jax/layers:standard",
        "//dgf/src/util:log",
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "link_prediction_model",
    srcs = ["link_prediction_model.py"],
    deps = [
        ":common",
        ":evaluation",
        ":link_prediction_core_model",
        ":report",
        # dataclasses_json dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/generate:edge_neighbor_generator",
        "//dgf/src/io:jax",
        "//dgf/src/io:tf",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/transform:merge",
        "//dgf/src/transform:normalize",
        "//dgf/src/util:util_ext",
        "//dgf/src/util:util_py",
        # jax dep,
        # jax/experimental/jax2tf dep,
        # jaxtyping dep,
        # numpy dep,
        # orbax/checkpoint dep,
        # tensorflow dep,
        # tqdm dep,
    ],
)

py_library(
    name = "link_prediction_dataset",
    srcs = ["link_prediction_dataset.py"],
    deps = [
        ":node_prediction_dataset",
        "//dgf/src/analyse:in_process_feature_statistics",
        "//dgf/src/analyse:padding",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/generate:edge_neighbor_generator",
        "//dgf/src/io:jax",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/transform:merge",
        "//dgf/src/transform:normalize",
        "//dgf/src/util:log",
        "//dgf/src/util:util_py",
        # jax dep,
        # numpy dep,
    ],
)

py_library(
    name = "node_prediction_dataset",
    srcs = ["node_prediction_dataset.py"],
    deps = [
        ":common",
        ":dataset",
        "//dgf/src/analyse:in_process_feature_statistics",
        "//dgf/src/analyse:padding",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/io:jax",
        "//dgf/src/learning/jax:common",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/transform:normalize",
        "//dgf/src/util:log",
        "//dgf/src/util:util_py",
        # jax dep,
        # numpy dep,
    ],
)

# Test
# =========

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        ":common",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        # numpy dep,
    ],
)

py_test(
    name = "node_prediction_test",
    timeout = "long",
    srcs = ["node_prediction_test.py"],
    data = ["//test_data"],
    shard_count = 4,
    tags = [
        #"requires-gpu-nvidia",
    ],
    deps = [
        ":common",
        ":node_prediction",
        # "//learning/brain/research/jax:gpu_support",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/generate:graphs",
        "//dgf/src/io:tf",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/learning/jax/layers:standard",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # jax dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "evaluation_test",
    srcs = ["evaluation_test.py"],
    deps = [
        ":evaluation",
        # absl/testing:absltest dep,
        # dataclasses_json dep,
        # jax dep,
        # numpy dep,
        # sklearn dep,
    ],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    deps = [
        ":dataset",
        # absl/testing:absltest dep,
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/sampling:config",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
    ],
)

py_test(
    name = "report_test",
    srcs = ["report_test.py"],
    deps = [
        ":common",
        ":report",
        # absl/testing:absltest dep,
        "//dgf/src/data:padding",
        "//dgf/src/sampling:config",
    ],
)

py_test(
    name = "link_prediction_test",
    timeout = "long",
    srcs = ["link_prediction_test.py"],
    data = ["//test_data"],
    shard_count = 4,
    deps = [
        ":common",
        ":link_prediction_model",
        ":link_prediction_train",
        #"//learning/brain/research/jax:gpu_support",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:tf",
        "//dgf/src/learning/jax/layers:standard",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:test_util",
        # jax dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "link_prediction_dataset_test",
    srcs = ["link_prediction_dataset_test.py"],
    deps = [
        ":link_prediction_dataset",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/generate:edge_neighbor_generator",
        "//dgf/src/io:jax",
        "//dgf/src/plot:network",
        "//dgf/src/sampling:config",
        "//dgf/src/transform:normalize",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)

py_test(
    name = "node_prediction_dataset_test",
    srcs = ["node_prediction_dataset_test.py"],
    deps = [
        ":node_prediction_dataset",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
        "//dgf/src/io:jax",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/sampling:config",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)
