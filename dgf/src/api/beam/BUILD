load("@rules_python//python:py_library.bzl", "py_library")

package(
    
    default_visibility = ["//visibility:public"],
)

# Library
# =====

py_library(
    name = "beam",
    srcs = ["__init__.py"],
    deps = [
        ":analyse",
        ":data",
        ":io",
        ":sampling",
        ":transform",
        "//dgf/src/beam:runners",
        "//dgf/src/data:beam_coders",
    ],
)

py_library(
    name = "io",
    srcs = ["io.py"],
    deps = [
        "//dgf/src/io:graph_in_beam",
        "//dgf/src/io:hgraph_in_beam",
        "//dgf/src/io:spanner",
        "//dgf/src/io:statistics",
        "//dgf/src/io:tf_graph_sample",
        "//dgf/src/io/gcp:bigquery_graph_beam",
        "//dgf/src/io/gcp:spanner_graph_beam",
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        "//dgf/src/data:distributed_graph",
    ],
)

py_library(
    name = "transform",
    srcs = ["transform.py"],
    deps = [
        "//dgf/src/beam:reverse_edges",
    ],
)

py_library(
    name = "analyse",
    srcs = ["analyse.py"],
    deps = [
        "//dgf/src/analyse:feature_statistics",
        "//dgf/src/analyse:in_memory_feature_statistics",
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    deps = [
        "//dgf/src/sampling:beam_semi_distributed_sampler",
        "//dgf/src/sampling:beam_semi_distributed_sampler_v1",
        "//dgf/src/sampling:beam_semi_distributed_sampler_v2",
    ],
)
