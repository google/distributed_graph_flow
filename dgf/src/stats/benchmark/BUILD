load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

py_library(
    name = "distributions",
    testonly = 1,
    srcs = ["distributions.py"],
    deps = [
        # numpy dep,
        # scipy dep,
    ],
)

py_test(
    name = "distributions_test",
    size = "large",
    srcs = ["distributions_test.py"],
    shard_count = 4,
    deps = [
        ":distributions",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
    ],
)

py_library(
    name = "generators",
    testonly = 1,
    srcs = ["generators.py"],
    deps = [
        ":distributions",
        # numpy dep,
    ],
)

py_test(
    name = "generators_test",
    size = "large",
    srcs = ["generators_test.py"],
    shard_count = 4,
    deps = [
        ":generators",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
    ],
)

py_library(
    name = "baselines",
    srcs = ["baselines.py"],
    deps = [
        # numpy dep,
        # scipy dep,
    ],
)

py_test(
    name = "baselines_test",
    srcs = ["baselines_test.py"],
    deps = [
        ":baselines",
        # absl/testing:absltest dep,
        # numpy dep,
    ],
)

py_library(
    name = "real_graph_topology",
    testonly = 1,
    srcs = ["real_graph_topology.py"],
    deps = [
        "//dgf",
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "range_generators",
    testonly = 1,
    srcs = ["range_generators.py"],
    deps = [
        # numpy dep,
    ],
)

py_test(
    name = "range_generators_test",
    srcs = ["range_generators_test.py"],
    deps = [
        ":range_generators",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
    ],
)

py_library(
    name = "coverage_dependent_truth",
    testonly = 1,
    srcs = ["coverage_dependent_truth.py"],
    deps = [
        # numpy dep,
        # scipy dep,
    ],
)

py_test(
    name = "coverage_dependent_truth_test",
    srcs = ["coverage_dependent_truth_test.py"],
    deps = [
        ":coverage_dependent_truth",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/stats:cluster_sketch",
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "coverage_dependent_lib",
    testonly = 1,
    srcs = ["coverage_dependent.py"],
    deps = [
        ":coverage_dependent_truth",
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/stats:cluster_bound",
        "//dgf/src/stats:cluster_sketch",
        "//dgf/src/stats:graph",
        "//dgf/src/stats:partitioners",
        "//dgf/src/stats:refutation",
        "//dgf/src/stats:temporal",
        "//dgf/src/stats/independent:interval",
        # numpy dep,
        # scipy dep,
    ],
)

py_binary(
    name = "coverage_dependent",
    testonly = 1,
    srcs = ["coverage_dependent.py"],
    deps = [
        ":coverage_dependent_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "coverage_dependent_test",
    size = "large",
    srcs = ["coverage_dependent_test.py"],
    deps = [
        ":coverage_dependent_lib",
        # absl/testing:absltest dep,
    ],
)

py_library(
    name = "coverage_independent_lib",
    testonly = 1,
    srcs = ["coverage_independent.py"],
    deps = [
        ":generators",
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/stats/independent:bound",
        "//dgf/src/stats/independent:interval",
        "//dgf/src/stats/independent:jax_accumulator",
        "//dgf/src/stats/independent:metrics",
        "//dgf/src/stats/independent:sketch",
        # jax dep,
        # numpy dep,
        # scipy dep,
    ],
)

py_binary(
    name = "coverage_independent",
    testonly = 1,
    srcs = ["coverage_independent.py"],
    deps = [
        ":coverage_independent_lib",
        # absl:app dep,
        # absl/flags dep,
        # jax dep,
    ],
)

py_test(
    name = "coverage_independent_test",
    size = "medium",
    srcs = ["coverage_independent_test.py"],
    deps = [
        ":coverage_independent_lib",
        # absl/testing:absltest dep,
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "textbook_baselines_lib",
    testonly = 1,
    srcs = ["textbook_baselines.py"],
    deps = [
        ":coverage_dependent_lib",
        ":coverage_dependent_truth",
        ":real_graph_topology",
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/stats:graph",
        "//dgf/src/stats:partitioners",
        "//dgf/src/stats:refutation",
        "//dgf/src/stats:temporal",
        # numpy dep,
        # scipy dep,
    ],
)

py_binary(
    name = "textbook_baselines",
    testonly = 1,
    srcs = ["textbook_baselines.py"],
    deps = [
        ":textbook_baselines_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "textbook_baselines_test",
    srcs = ["textbook_baselines_test.py"],
    deps = [
        ":textbook_baselines_lib",
        # absl/testing:absltest dep,
    ],
)

py_library(
    name = "validation_r2_buckets_lib",
    testonly = 1,
    srcs = ["validation_r2_buckets.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/stats:cluster_bound",
        "//dgf/src/stats:cluster_sketch",
        "//dgf/src/stats:graph",
        "//dgf/src/stats:partitioners",
        "//dgf/src/stats:refutation",
        "//dgf/src/stats:temporal",
        # numpy dep,
        # scipy dep,
    ],
)

py_binary(
    name = "validation_r2_buckets",
    testonly = 1,
    srcs = ["validation_r2_buckets.py"],
    deps = [
        ":validation_r2_buckets_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "validation_r2_buckets_test",
    srcs = ["validation_r2_buckets_test.py"],
    deps = [
        ":validation_r2_buckets_lib",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/stats:graph",
        "//dgf/src/stats:partitioners",
        # numpy dep,
    ],
)

py_library(
    name = "validation_spacetime_lib",
    testonly = 1,
    srcs = ["validation_spacetime.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/stats:graph",
        "//dgf/src/stats:partitioners",
        "//dgf/src/stats:refutation",
        "//dgf/src/stats:spacetime",
        "//dgf/src/stats:temporal",
        "//dgf/src/stats/independent:interval",
        # numpy dep,
        # scipy dep,
    ],
)

py_binary(
    name = "validation_spacetime",
    testonly = 1,
    srcs = ["validation_spacetime.py"],
    deps = [
        ":validation_spacetime_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "validation_spacetime_test",
    srcs = ["validation_spacetime_test.py"],
    deps = [
        ":validation_spacetime_lib",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/stats/independent:interval",
        # numpy dep,
    ],
)

py_library(
    name = "range_coverage_lib",
    testonly = 1,
    srcs = ["range_coverage.py"],
    deps = [
        ":range_generators",
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/stats:cluster_bound",
        "//dgf/src/stats:cluster_sketch",
        "//dgf/src/stats:range_envelope",
        "//dgf/src/stats:spacetime",
        "//dgf/src/stats/independent:interval",
        # numpy dep,
        # scipy dep,
    ],
)

py_binary(
    name = "range_coverage",
    testonly = 1,
    srcs = ["range_coverage.py"],
    deps = [
        ":range_coverage_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "range_coverage_test",
    srcs = ["range_coverage_test.py"],
    deps = [
        ":range_coverage_lib",
        ":range_generators",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
    ],
)

py_library(
    name = "textbook_rmse_coverage_lib",
    testonly = 1,
    srcs = ["textbook_rmse_coverage.py"],
    deps = [
        ":baselines",
        ":distributions",
        ":range_generators",
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "//dgf/src/stats:cluster_bound",
        "//dgf/src/stats:cluster_sketch",
        "//dgf/src/stats:range_envelope",
        "//dgf/src/stats:spacetime",
        "//dgf/src/stats/independent:interval",
        # numpy dep,
    ],
)

py_binary(
    name = "textbook_rmse_coverage",
    testonly = 1,
    srcs = ["textbook_rmse_coverage.py"],
    deps = [
        ":textbook_rmse_coverage_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "textbook_rmse_coverage_test",
    size = "medium",
    srcs = ["textbook_rmse_coverage_test.py"],
    deps = [
        ":textbook_rmse_coverage_lib",
        # absl/testing:absltest dep,
        # numpy dep,
    ],
)

py_library(
    name = "edge_guard_sweep_lib",
    testonly = 1,
    srcs = ["edge_guard_sweep.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/stats:graph",
        "//dgf/src/stats:partitioners",
        "//dgf/src/stats:refutation",
        # numpy dep,
    ],
)

py_binary(
    name = "edge_guard_sweep",
    testonly = 1,
    srcs = ["edge_guard_sweep.py"],
    deps = [
        ":edge_guard_sweep_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "edge_guard_sweep_test",
    srcs = ["edge_guard_sweep_test.py"],
    deps = [
        ":edge_guard_sweep_lib",
        # absl/testing:absltest dep,
    ],
)

py_library(
    name = "partitioner_robustness_lib",
    testonly = 1,
    srcs = ["partitioner_robustness.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/stats:partitioners",
        # numpy dep,
    ],
)

py_binary(
    name = "partitioner_robustness",
    testonly = 1,
    srcs = ["partitioner_robustness.py"],
    deps = [
        ":partitioner_robustness_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "partitioner_robustness_test",
    srcs = ["partitioner_robustness_test.py"],
    deps = [
        ":partitioner_robustness_lib",
        # absl/testing:absltest dep,
    ],
)

py_library(
    name = "hypergeom_cp_lib",
    testonly = 1,
    srcs = ["hypergeom_cp.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "//dgf/src/stats/independent:interval",
        # numpy dep,
        # scipy dep,
    ],
)

py_binary(
    name = "hypergeom_cp",
    testonly = 1,
    srcs = ["hypergeom_cp.py"],
    deps = [
        ":hypergeom_cp_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "hypergeom_cp_test",
    srcs = ["hypergeom_cp_test.py"],
    deps = [
        ":hypergeom_cp_lib",
        # absl/testing:absltest dep,
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "theory_sims_lib",
    testonly = 1,
    srcs = ["theory_sims.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # numpy dep,
    ],
)

py_binary(
    name = "theory_sims",
    testonly = 1,
    srcs = ["theory_sims.py"],
    deps = [
        ":theory_sims_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "theory_sims_test",
    srcs = ["theory_sims_test.py"],
    deps = [
        ":theory_sims_lib",
        # absl/testing:absltest dep,
    ],
)

py_library(
    name = "cluster_sweep_lib",
    testonly = 1,
    srcs = ["cluster_sweep.py"],
    deps = [
        ":baselines",
        ":generators",
        # absl:app dep,
        # absl/flags dep,
        "//dgf/src/stats:cluster_bound",
        "//dgf/src/stats/independent:interval",
        # numpy dep,
        # scipy dep,
    ],
)

py_binary(
    name = "cluster_sweep",
    testonly = 1,
    srcs = ["cluster_sweep.py"],
    deps = [
        ":cluster_sweep_lib",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_test(
    name = "cluster_sweep_test",
    srcs = ["cluster_sweep_test.py"],
    deps = [
        ":cluster_sweep_lib",
        # absl/testing:absltest dep,
    ],
)
