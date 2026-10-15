load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

py_library(
    name = "bound",
    srcs = ["bound.py"],
    deps = [
        ":metrics",
        # numpy dep,
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        # numpy dep,
    ],
)

py_library(
    name = "sketch",
    srcs = ["sketch.py"],
    deps = [
        ":metrics",
        # numpy dep,
    ],
)

py_library(
    name = "interval",
    srcs = ["interval.py"],
    deps = [
        ":bound",
        ":metrics",
        ":sketch",
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "independent",
    srcs = ["__init__.py"],
    deps = [
        ":bound",
        ":interval",
        ":jax_accumulator",
        ":metrics",
        ":sketch",
    ],
)

py_test(
    name = "interval_test",
    srcs = ["interval_test.py"],
    deps = [
        ":bound",
        ":interval",
        ":metrics",
        ":sketch",
        # absl/testing:absltest dep,
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "jax_accumulator",
    srcs = ["jax_accumulator.py"],
    deps = [
        ":metrics",
        ":sketch",
        # jax dep,
        # numpy dep,
    ],
)

py_test(
    name = "jax_accumulator_test",
    srcs = ["jax_accumulator_test.py"],
    deps = [
        ":interval",
        ":jax_accumulator",
        ":sketch",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
    ],
)
