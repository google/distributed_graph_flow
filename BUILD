package_group(
    name = "users",
    packages = ["//..."],
)

package_group(
    name = "internal_users",
    packages = ["//..."],
)

load("@rules_python//python:py_runtime.bzl", "py_runtime")
load("@rules_python//python:py_runtime_pair.bzl", "py_runtime_pair")

py_runtime(
    name = "system_python3",
    interpreter_path = "/tmp/venv/bin/python3",
    python_version = "PY3",
)

py_runtime_pair(
    name = "system_py_runtime_pair",
    py3_runtime = ":system_python3",
)

toolchain(
    name = "system_python_toolchain",
    toolchain = ":system_py_runtime_pair",
    toolchain_type = "@rules_python//python:toolchain_type",
)
