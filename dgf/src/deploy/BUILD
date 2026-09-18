load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "vertex",
    srcs = ["vertex.py"],
    deps = [
        "//dgf/src/learning/ten_lines:common",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        "//third_party/py/google/cloud/aiplatform",
        # yaml dep,
    ],
)

py_test(
    name = "vertex_test",
    srcs = ["vertex_test.py"],
    deps = [
        ":vertex",
        # absl/testing:absltest dep,
        "//dgf/src/learning/ten_lines:common",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # yaml dep,
    ],
)
