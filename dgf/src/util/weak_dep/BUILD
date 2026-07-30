load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//dgf/src:__subpackages__",
    ],
)

py_library(
    name = "weak_dep_base",
    srcs = ["base.py"],
)

py_library(
    name = "weak_dep_tensorflow",
    srcs = ["weak_dep_tensorflow.py"],
    deps = [
        ":weak_dep_base",
        # tensorflow dep,  # build_cleaner: keep
    ],
)

py_library(
    name = "weak_dep_apache_beam",
    srcs = ["weak_dep_apache_beam.py"],
    deps = [
        ":weak_dep_base",
        # apache_beam dep,  # build_cleaner: keep
    ],
)

py_library(
    name = "weak_dep_tensorflow_gnn",
    srcs = ["weak_dep_tensorflow_gnn.py"],
    deps = [
        ":weak_dep_base",
        # tensorflow_gnn dep,  # build_cleaner: keep
    ],
)

py_library(
    name = "weak_dep_ogb",
    srcs = ["weak_dep_ogb.py"],
    deps = [":weak_dep_base"],
)

py_library(
    name = "weak_dep",
    srcs = ["__init__.py"],
)

py_test(
    name = "weak_dep_test",
    srcs = ["weak_dep_test.py"],
    deps = [
        ":weak_dep_base",
        # absl/testing:absltest dep,
    ],
)
