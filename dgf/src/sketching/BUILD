load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

py_library(
    name = "minhash",
    srcs = ["minhash.py"],
    type_checking = True,
    deps = [
        # apache_beam dep,
        # datasketch dep,
        "//dgf/src/util/weak_dep:weak_dep_apache_beam",
    ],
)

py_test(
    name = "minhash_test",
    srcs = ["minhash_test.py"],
    type_checking = True,
    deps = [
        ":minhash",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/util:test_util",
    ],
)
