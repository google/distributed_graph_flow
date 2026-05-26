load("@rules_python//python:py_library.bzl", "py_library")
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
    deps = [
        # apache_beam dep,
        # datasketch dep,
    ],
)

py_test(
    name = "minhash_test",
    srcs = ["minhash_test.py"],
    deps = [
        ":minhash",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/util:test_util",
    ],
)
