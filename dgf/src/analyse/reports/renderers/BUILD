load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = ["//dgf/src:__subpackages__"],
)

py_library(
    name = "html_renderer",
    srcs = ["html_renderer.py"],
    data = ["//dgf/src/analyse/reports/templates:report_templates"],
    deps = [
        # absl/logging dep,
        "//dgf/src/analyse/reports:data_model",
        "//dgf/src/analyse/reports:visual_utils",
        # jinja2 dep,
    ],
)

py_library(
    name = "pdf_renderer",
    srcs = ["pdf_renderer.py"],
    deps = [
        "//dgf/src/analyse/reports:data_model",
        # reportlab dep,
    ],
)

py_test(
    name = "html_renderer_test",
    srcs = ["html_renderer_test.py"],
    data = ["//dgf/src/analyse/reports/templates:report_templates"],
    deps = [
        ":html_renderer",
        # absl/testing:absltest dep,
        "//dgf/src/analyse/reports:data_model",
        "//dgf/src/analyse/topology:global_graph_topology",
        # networkx dep,
    ],
)
