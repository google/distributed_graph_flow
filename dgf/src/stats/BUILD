load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

py_library(
    name = "cluster_sketch",
    srcs = ["cluster_sketch.py"],
    deps = [
        "//dgf/src/stats/independent:metrics",
        "//dgf/src/stats/independent:sketch",
        # numpy dep,
    ],
)

py_library(
    name = "cluster_bound",
    srcs = ["cluster_bound.py"],
    deps = [
        ":cluster_sketch",
        ":refutation",
        "//dgf/src/stats/independent:interval",
        "//dgf/src/stats/independent:metrics",
        "//dgf/src/stats/independent:sketch",
        # numpy dep,
    ],
)

py_test(
    name = "cluster_bound_test",
    srcs = ["cluster_bound_test.py"],
    deps = [
        ":cluster_bound",
        ":cluster_sketch",
        ":refutation",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/stats/independent:interval",
        "//dgf/src/stats/independent:metrics",
        "//dgf/src/stats/independent:sketch",
        # numpy dep,
    ],
)

py_test(
    name = "cluster_sketch_test",
    srcs = ["cluster_sketch_test.py"],
    deps = [
        ":cluster_bound",
        ":cluster_sketch",
        # absl/testing:absltest dep,
        "//dgf/src/stats/independent:metrics",
        "//dgf/src/stats/independent:sketch",
        # numpy dep,
    ],
)

py_library(
    name = "partitioners",
    srcs = ["partitioners.py"],
    deps = [
        # numpy dep,
    ],
)

py_test(
    name = "partitioners_test",
    size = "medium",
    srcs = ["partitioners_test.py"],
    shard_count = 4,
    deps = [
        ":partitioners",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
    ],
)

py_library(
    name = "refutation",
    srcs = ["refutation.py"],
    deps = [
        # numpy dep,
        # scipy dep,
    ],
)

py_test(
    name = "refutation_test",
    size = "medium",
    srcs = ["refutation_test.py"],
    deps = [
        ":graph",
        ":partitioners",
        ":refutation",
        # absl/testing:absltest dep,
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "graph",
    srcs = ["graph.py"],
    deps = [
        ":cluster_bound",
        ":cluster_sketch",
        ":partitioners",
        ":refutation",
        "//dgf/src/stats/independent:interval",
        "//dgf/src/stats/independent:metrics",
        # numpy dep,
        # scipy dep,
    ],
)

py_test(
    name = "graph_test",
    srcs = ["graph_test.py"],
    deps = [
        ":cluster_bound",
        ":cluster_sketch",
        ":graph",
        ":refutation",
        # absl/testing:absltest dep,
        "//dgf/src/stats/independent:interval",
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "temporal",
    srcs = ["temporal.py"],
    deps = [
        ":cluster_bound",
        ":cluster_sketch",
        ":refutation",
        "//dgf/src/stats/independent:interval",
        "//dgf/src/stats/independent:metrics",
        # numpy dep,
    ],
)

py_test(
    name = "temporal_test",
    srcs = ["temporal_test.py"],
    deps = [
        ":cluster_sketch",
        ":refutation",
        ":temporal",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/stats/independent:interval",
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "declare",
    srcs = ["declare.py"],
    deps = [
        # numpy dep,
    ],
)

py_test(
    name = "declare_test",
    srcs = ["declare_test.py"],
    deps = [
        ":cluster_bound",
        ":cluster_sketch",
        ":declare",
        ":graph",
        ":temporal",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/stats/independent:interval",
        # numpy dep,
    ],
)

py_library(
    name = "spacetime",
    srcs = ["spacetime.py"],
    deps = [
        ":cluster_bound",
        ":cluster_sketch",
        ":graph",
        ":partitioners",
        ":refutation",
        ":temporal",
        "//dgf/src/stats/independent:interval",
        "//dgf/src/stats/independent:metrics",
        # numpy dep,
    ],
)

py_test(
    name = "spacetime_test",
    srcs = ["spacetime_test.py"],
    deps = [
        ":declare",
        ":graph",
        ":refutation",
        ":spacetime",
        ":temporal",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/stats/independent:interval",
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "range_envelope",
    srcs = ["range_envelope.py"],
    deps = [
        ":graph",
        ":spacetime",
        # numpy dep,
    ],
)

py_test(
    name = "range_envelope_test",
    srcs = ["range_envelope_test.py"],
    deps = [
        ":range_envelope",
        ":spacetime",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/stats/independent:interval",
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "stats",
    srcs = ["__init__.py"],
)
