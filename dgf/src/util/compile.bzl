"""Utilities for the compilation of code."""

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

def all_proto_library(
        name = None,
        deps = [],
        srcs = [],
        compile_cc = False,
        compile_py = False,
        visibility = None,
        has_services = False,
        exports = None):
    """Create the set of proto, cc proto and py proto targets.

    Usage example:
        all_proto_library(name="toy_proto",srcs=[...])

        cc_library_ydf(deps=[":toy_cc_proto"], ...)
        py_library(deps=[":toy_py_proto"], ...)

    Args:
        name: The name of the proto_library rule. Must end with "_proto".
        deps: Dependencies for the proto_library.
        srcs: Source `.proto` files.
        compile_cc: If True, a `cc_proto_library` and `cc_grpc_library` (if has_services is True) will be created.
        compile_py: If True, a `py_proto_library` will be created.
        visibility: Visibility for the generated targets.
        has_services: Whether the proto defines services. Required for `cc_grpc_library`.
        exports: Targets to export from the `proto_library`.
    """

    suffix = "_proto"
    if not name.endswith(suffix):
        fail("Rule name should ends with _proto")
    base_name = name[0:-len(suffix)]

    proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
        visibility = visibility,
    )

    if has_services:
        cc_grpc_library(
            name = base_name + "_grpc_proto",
            srcs = [":" + name],
            deps = [base_name + "_cc_proto"],
            visibility = visibility,
            grpc_only = True,
        )

    if compile_cc:
        cc_proto_library(
            name = base_name + "_cc_proto",
            deps = [":" + name],
            visibility = visibility,
        )

    if compile_py:
        py_proto_library(
            name = base_name + "_py_proto",
            deps = [":" + name],
            visibility = visibility,
        )
