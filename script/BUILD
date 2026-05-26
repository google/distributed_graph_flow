load("//devtools/python/blaze:pytype.bzl", "py_binary")

package(
    
    default_visibility = ["//visibility:public"],
)

py_binary(
    name = "gen_api",
    srcs = ["gen_api.py"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        "//dgf",
    ],
)
