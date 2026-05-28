#!/bin/bash
# This script runs INSIDE the docker container to build and test the pip package.
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

bazel build -c opt //dgf --@rules_python//python/config_settings:python_version=${PYVERSION}
cd /work/bazel-bin/dgf
find . -name '*.so' -type f -exec cp --parents {} /work/package/dgf \;
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple/"
${PYBIN} -m pip install setuptools auditwheel
cd /work/package
${PYBIN} setup.py bdist_wheel --dist-dir /work/dist
cd /work
chmod -R a+rw package
${PYBIN} -m auditwheel repair --plat manylinux_2_28_x86_64 -w dist dist/dgf-*-cp${PYVERSIONNODOT}-cp${PYVERSIONNODOT}-linux_x86_64.whl
chmod -R a+rw dist
${PYBIN} -m pip uninstall dgf -y
${PYBIN} -m pip install dist/dgf-*-cp${PYVERSIONNODOT}-cp${PYVERSIONNODOT}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# export TF_USE_LEGACY_KERAS=1 # TF_USE_LEGACY_KERAS=1 is not needed for this toy example.
${PYBIN} script/toy.py
