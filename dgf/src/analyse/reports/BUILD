load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

py_library(
    name = "data_model",
    srcs = ["data_model.py"],
    deps = [
        "//dgf/src/analyse/topology:global_graph_topology",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:statistics",
        # networkx dep,
    ],
)

py_library(
    name = "reporter",
    srcs = ["reporter.py"],
    deps = [
        ":data_model",
        # absl/logging dep,
        "//dgf/src/analyse/reports/renderers:html_renderer",
        "//dgf/src/analyse/reports/renderers:pdf_renderer",
    ],
)

py_library(
    name = "visual_utils",
    srcs = ["visual_utils.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        # networkx dep,
        # numpy dep,
    ],
)

py_test(
    name = "visual_inspection_test",
    srcs = ["visual_inspection_test.py"],
    deps = [
        ":data_model",
        ":reporter",
        ":visual_utils",
        # absl/testing:absltest dep,
        "//dgf/src/analyse/topology:global_graph_topology",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        # networkx dep,
        # numpy dep,
    ],
)

py_test(
    name = "reporter_test",
    srcs = ["reporter_test.py"],
    deps = [
        ":data_model",
        ":reporter",
        # absl/testing:absltest dep,
        "//dgf/src/analyse/topology:global_graph_topology",
    ],
)
