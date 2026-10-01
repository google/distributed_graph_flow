"""NumPy C headers of the local Python installation.

Only used by the open-source Bazel build (see MODULE.bazel). In Google3, the
headers are provided by "//third_party/py/numpy:headers".
"""

_BUILD_FILE = """load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "headers",
    hdrs = glob(["include/numpy/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
"""

def _numpy_headers_repository_impl(repository_ctx):
    python = repository_ctx.which("python3")
    if not python:
        fail("Cannot find \"python3\" to locate the NumPy headers.")
    result = repository_ctx.execute(
        [python, "-c", "import numpy; print(numpy.get_include())"],
    )
    if result.return_code != 0:
        fail("Cannot locate the NumPy headers. Make sure NumPy is installed " +
             "for \"{}\": {}".format(python, result.stderr))
    repository_ctx.symlink(result.stdout.strip(), "include")
    repository_ctx.file("BUILD.bazel", _BUILD_FILE)

_numpy_headers_repository = repository_rule(
    implementation = _numpy_headers_repository_impl,
    environ = ["PATH", "VIRTUAL_ENV"],
    local = True,
)

def _numpy_headers_extension_impl(_module_ctx):
    _numpy_headers_repository(name = "numpy_headers")

numpy_headers = module_extension(implementation = _numpy_headers_extension_impl)
