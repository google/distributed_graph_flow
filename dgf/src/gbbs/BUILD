load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

cc_library(
    name = "gbbs_graph_handle",
    srcs = ["gbbs_graph_handle.cc"],
    hdrs = ["gbbs_graph_handle.h"],
    deps = [
        "//third_party/gbbs/gbbs:graph",
        "//third_party/gbbs/gbbs:macros",
        "//third_party/gbbs/gbbs:vertex",
        "//third_party/parlay/include/parlay:parallel",
        "//third_party/parlay/include/parlay:sequence",
    ],
)

cc_library(
    name = "connected_components",
    srcs = ["connected_components.cc"],
    hdrs = ["connected_components.h"],
    deps = [
        ":gbbs_graph_handle",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/time",
        "//third_party/gbbs/benchmarks/Connectivity:common",
        "//third_party/gbbs/benchmarks/Connectivity/BFSCC:Connectivity",
        "//third_party/gbbs/benchmarks/Connectivity/LabelPropagation:Connectivity",
        "//third_party/gbbs/benchmarks/Connectivity/ShiloachVishkin:Connectivity",
        "//third_party/gbbs/benchmarks/Connectivity/SimpleUnionAsync:Connectivity",
        "//third_party/gbbs/benchmarks/Connectivity/WorkEfficientSDB14:Connectivity",
        "//third_party/gbbs/gbbs:bridge",
        "//third_party/parlay/include/parlay:monoid",
        "//third_party/parlay/include/parlay:parallel",
        "//third_party/parlay/include/parlay:primitives",
    ],
)

# Nanobind Extension
# =================
nanobind_extension(
    name = "_gbbs_ext",
    srcs = ["gbbs_ext.cc"],
    deps = [
        ":connected_components",
        ":gbbs_graph_handle",
        "//third_party/gbbs/gbbs:macros",
        "//third_party/parlay/include/parlay:parallel",
        "//third_party/parlay/include/parlay:scheduler",
        "//dgf/src/util:status_caster",
    ],
)

# Python API Layer
# ================
py_library(
    name = "loader",
    srcs = ["loader.py"],
    deps = [
        ":_gbbs_ext",
        # absl/logging dep,
        "//dgf/src/api:io",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_constants",
        "//dgf/src/io:schema",
        # numpy dep,
    ],
)

py_library(
    name = "connected_components_py",
    srcs = ["connected_components.py"],
    deps = [
        ":_gbbs_ext",
        # tqdm dep,
    ],
)

py_library(
    name = "gbbs",
    srcs = ["__init__.py"],
)

py_test(
    name = "loader_test",
    srcs = ["loader_test.py"],
    deps = [
        ":loader",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io:graph_in_memory",
        # numpy dep,
    ],
)

py_test(
    name = "connected_components_test",
    srcs = ["connected_components_test.py"],
    deps = [
        ":connected_components_py",
        ":loader",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
    ],
)
