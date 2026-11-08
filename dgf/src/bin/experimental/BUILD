load("@rules_python//python:py_binary.bzl", "py_binary")

package(default_applicable_licenses = ["//dgf:license"])

# Binary
# =========

py_binary(
    name = "gen_temporal_alert_graph_main",
    srcs = ["gen_temporal_alert_graph_main.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "//dgf/src/generate:gen_temporal_alert_graph",
    ],
)
