#!/bin/bash
# This script runs INSIDE the docker container (Linux) or on the host (macOS) to
# build and test the pip package.
# This script IS exported by copybara.
set -vex

if [ -z "$1" ]; then
  echo "Usage: $0 <pyversion>"
  exit 1
fi

PYVERSION="$1"
PYVERSIONNODOT=${PYVERSION//./}
PYBIN="python${PYVERSION}"

if [ ! -f /tmp/venv/bin/activate ]; then
  echo "ERROR: Virtual environment not found at /tmp/venv. Run test.sh first. Exiting."
  exit 1
else
  VENV_PYVER=$(/tmp/venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
  if [ "$VENV_PYVER" != "$PYVERSION" ]; then
    echo "ERROR: Venv version mismatch (found $VENV_PYVER, expected $PYVERSION). Exiting."
    exit 1
  fi
fi
source /tmp/venv/bin/activate

# Name and version overrides, set by build_pip_package.sh for the nightly.
PACKAGE_NAME="${DGF_PACKAGE_NAME:-dgf}"
# Wheel file names use "_" instead of "-" (e.g. dgf_nightly-0.1.1.dev20261008-...).
WHEEL_NAME="${PACKAGE_NAME//-/_}"

# Repository root: "/work" in the Docker container (Linux), the exported
# directory on macOS (set by build_pip_package.sh).
WORK_DIR="${WORK_DIR:-/work}"
OS="$(uname -s)"
# The macOS wheels support macOS >= 12 (see .bazelrc) on arm64.
MACOS_PLATFORM="macosx_12_0_arm64"

# The final (repaired) wheel.
if [ "${OS}" = "Darwin" ]; then
  export MACOSX_DEPLOYMENT_TARGET=12.0
  WHEEL_FILE="dist/${WHEEL_NAME}-*-cp${PYVERSIONNODOT}-cp${PYVERSIONNODOT}-${MACOS_PLATFORM}.whl"
else
  WHEEL_FILE="dist/${WHEEL_NAME}-*-cp${PYVERSIONNODOT}-cp${PYVERSIONNODOT}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
fi

bazel build -c opt //dgf --@rules_python//python/config_settings:python_version=${PYVERSION}
cd "${WORK_DIR}/bazel-bin/dgf"
# Portable equivalent of "cp --parents" (GNU only).
find . -name '*.so' -type f | tar -cf - -T - | (cd "${WORK_DIR}/package/dgf" && tar -xf -)
if [ -n "${DGF_PACKAGE_VERSION:-}" ]; then
  # Patched in the packaged copy only: dgf.__version__ and setup.py (which
  # reads it) report the nightly version. "-i.bak" works with GNU and BSD sed.
  sed -i.bak "s/^__version__ = \".*\"$/__version__ = \"${DGF_PACKAGE_VERSION}\"/" "${WORK_DIR}/package/dgf/__init__.py"
  rm "${WORK_DIR}/package/dgf/__init__.py.bak"
  grep -q "^__version__ = \"${DGF_PACKAGE_VERSION}\"$" "${WORK_DIR}/package/dgf/__init__.py"
fi
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple/"
${PYBIN} -m pip install setuptools
cd "${WORK_DIR}/package"
if [ "${OS}" = "Darwin" ]; then
  # Explicit platform tag: the Python interpreter can be "universal2" while the
  # extensions are only compiled for arm64.
  ${PYBIN} setup.py bdist_wheel --dist-dir "${WORK_DIR}/dist" --plat-name "${MACOS_PLATFORM}"
else
  ${PYBIN} setup.py bdist_wheel --dist-dir "${WORK_DIR}/dist"
fi
cd "${WORK_DIR}"
chmod -R a+rw package
if [ "${OS}" = "Darwin" ]; then
  ${PYBIN} -m pip install delocate
  # Checks the architecture and minimum macOS version of the extensions, and
  # bundles the non-system shared libraries (if any). The repaired wheel has
  # the same name and replaces the original one.
  rm -rf dist/delocated
  delocate-wheel --require-archs arm64 -w dist/delocated -v ${WHEEL_FILE}
  mv -f dist/delocated/*.whl dist/
  rm -rf dist/delocated
else
  ${PYBIN} -m pip install auditwheel
  ${PYBIN} -m auditwheel repair --plat manylinux_2_28_x86_64 -w dist dist/${WHEEL_NAME}-*-cp${PYVERSIONNODOT}-cp${PYVERSIONNODOT}-linux_x86_64.whl
fi
chmod -R a+rw dist
# "dgf" and "dgf-nightly" both provide the "dgf" import package. Uninstalling a
# package that is not installed is a no-op for pip.
${PYBIN} -m pip uninstall -y dgf dgf-nightly
${PYBIN} -m pip install ${WHEEL_FILE}
# All the packages (requirements.txt and requirements-dev.txt) are installed.
${PYBIN} script/toy_minimal.py
${PYBIN} script/toy_full.py

echo "Setting up virtual environment at /tmp/venv_serving for testing serving..."
deactivate || true

if [ ! -f /tmp/venv_serving/bin/activate ]; then
  ${PYBIN} -m venv /tmp/venv_serving
else
  VENV_SERVING_PYVER=$(/tmp/venv_serving/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
  if [ "$VENV_SERVING_PYVER" != "$PYVERSION" ]; then
    echo "ERROR: venv_serving version mismatch (found $VENV_SERVING_PYVER, expected $PYVERSION). Exiting."
    exit 1
  fi
fi

source /tmp/venv_serving/bin/activate
${PYBIN} -m pip uninstall -y dgf dgf-nightly
${PYBIN} -m pip install -r requirements.txt
${PYBIN} -m pip install ${WHEEL_FILE}
# Only the packages of requirements.txt are installed. Make sure the weak
# dependencies are not installed (e.g. pulled transitively), otherwise
# toy_minimal.py would not test anything.
for MODULE in tensorflow tensorflow_gnn tf_keras apache_beam ogb clu; do
  if (cd /tmp && ${PYBIN} -c "import ${MODULE}" 2>/dev/null); then
    echo "ERROR: ${MODULE} is installed in the minimal venv."
    exit 1
  fi
done
${PYBIN} script/toy_minimal.py

# The installed package reports the expected version. Run outside of the work
# directory: with it as cwd, "import dgf" would import the source tree instead
# of the installed wheel.
if [ -n "${DGF_PACKAGE_VERSION:-}" ]; then
  INSTALLED_VERSION=$(cd /tmp && ${PYBIN} -c "import dgf; print(dgf.__version__)")
  if [ "${INSTALLED_VERSION}" != "${DGF_PACKAGE_VERSION}" ]; then
    echo "ERROR: dgf.__version__ is ${INSTALLED_VERSION}, expected ${DGF_PACKAGE_VERSION}."
    exit 1
  fi
fi
