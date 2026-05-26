load("@rules_python//python:py_library.bzl", "py_library")

package(
    
    default_visibility = ["//visibility:public"],
)

# Library
# =====

py_library(
    name = "api",
    srcs = ["__init__.py"],
    deps = [
        ":analyse",
        ":convert",
        ":data",
        ":exception",
        ":filesystem",
        ":generate",
        ":io",
        ":learning",
        ":plot",
        ":print",
        ":sampling",
        ":train",
        ":transform",
        ":validate",
        "//dgf/src/api/beam",
        "//dgf/src/api/jax",
    ],
)

py_library(
    name = "io",
    srcs = ["io.py"],
    deps = [
        "//dgf/src/io:cache",
        "//dgf/src/io:dataset_loader",
        "//dgf/src/io:graph_in_memory",
        "//dgf/src/io:hgraph_in_memory",
        "//dgf/src/io:schema",
        "//dgf/src/io:spanner",
        "//dgf/src/io:statistics",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/io/gcp:bigquery_graph",
        "//dgf/src/io/gcp:spanner_graph",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:proto",
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        "//dgf/src/data:tf_in_memory_graph",
    ],
)

py_library(
    name = "validate",
    srcs = ["validate.py"],
    deps = [
        "//dgf/src/validate:in_memory_graph",
    ],
)

py_library(
    name = "convert",
    srcs = ["convert.py"],
    deps = [
        "//dgf/src/io:hgraph_in_memory",
        "//dgf/src/io:jax",
        "//dgf/src/io:networkx",
        "//dgf/src/io:spanner",
        "//dgf/src/io:sparse_deferred",
        "//dgf/src/io:tf",
        "//dgf/src/io:tf_graph_sample",
    ],
)

py_library(
    name = "transform",
    srcs = ["transform.py"],
    deps = [
        "//dgf/src/learning/ten_lines:node_prediction_dataset",
        "//dgf/src/transform:extract",
        "//dgf/src/transform:homogenize",
        "//dgf/src/transform:in_memory_graph_filter",
        "//dgf/src/transform:merge",
        "//dgf/src/transform:normalize",
        "//dgf/src/transform:nx",
        "//dgf/src/transform:schema",
        "//dgf/src/transform:table_2_graph",
        "//dgf/src/transform:temporal",
        "//dgf/src/util:util_py",
    ],
)

py_library(
    name = "analyse",
    srcs = ["analyse.py"],
    deps = [
        "//dgf/src/analyse:in_process_feature_statistics",
        "//dgf/src/analyse:padding",
        "//dgf/src/analyse:print_schema",
        "//dgf/src/analyse/reports:data_model",
        "//dgf/src/analyse/reports:reporter",
        "//dgf/src/analyse/topology:global_graph_topology",
        "//dgf/src/analyse/topology:node_degree",
    ],
)

py_library(
    name = "plot",
    srcs = ["plot.py"],
    deps = [
        "//dgf/src/plot:network",
        "//dgf/src/plot:nx",
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    deps = [
        "//dgf/src/sampling:beam_semi_distributed_sampler",
        "//dgf/src/sampling:beam_semi_distributed_sampler_v1",
        "//dgf/src/sampling:beam_semi_distributed_sampler_v2",
        "//dgf/src/sampling:config",
        "//dgf/src/sampling:in_memory_sampler",
        "//dgf/src/sampling/gcp:spanner_graph_sampler",
    ],
)

py_library(
    name = "exception",
    srcs = ["exception.py"],
    deps = [
        "//dgf/src/transform:merge",
    ],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        "//dgf/src/learning:feature",
    ],
)

py_library(
    name = "generate",
    srcs = ["generate.py"],
    deps = [
        "//dgf/src/generate:edge_neighbor_generator",
        "//dgf/src/generate:graphs",
    ],
)

py_library(
    name = "filesystem",
    srcs = ["filesystem.py"],
    deps = ["//dgf/src/util:filesystem"],
)

py_library(
    name = "learning",
    srcs = ["learning.py"],
    deps = [
        "//dgf/src/learning/ten_lines:common",
        "//dgf/src/learning/ten_lines:link_prediction_train",
        "//dgf/src/learning/ten_lines:node_prediction",
    ],
)

py_library(
    name = "print",
    srcs = ["print.py"],
    deps = [
        "//dgf/src/analyse:padding",
        "//dgf/src/analyse:print_schema",
        "//dgf/src/analyse:sampling",
    ],
)
