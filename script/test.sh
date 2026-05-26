#!/bin/bash
# This script runs INSIDE the docker container to run unit tests.
# This script IS exported by copybara.
set -vex

if [ -z "$1" ] ; then
  echo "Usage: $0 <pyversion>"
  exit 1
fi

PYVERSION="$1"
PYBIN="python${PYVERSION}"

if [ ! -f /tmp/venv/bin/activate ]; then
  echo "Creating virtual environment at /tmp/venv..."
  ${PYBIN} -m venv /tmp/venv
else
  VENV_PYVER=$(/tmp/venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
  if [ "$VENV_PYVER" != "$PYVERSION" ]; then
    echo "ERROR: Venv version mismatch (found $VENV_PYVER, expected $PYVERSION). Exiting."
    exit 1
  fi
fi
source /tmp/venv/bin/activate


# Install requirements
${PYBIN} -m pip install -r requirements.txt

# Check that absl is installed
${PYBIN} -c "import absl" || { echo "Error: absl is not installed in the virtual environment."; exit 1; }

export TF_USE_LEGACY_KERAS=1

# Run all tests via Bazel
echo "Running all tests via Bazel..."
bazel test \
    --test_output=errors \
    --spawn_strategy=standalone \
    --test_strategy=standalone \
    --test_env=TF_USE_LEGACY_KERAS \
    --test_env=PYTHONPATH \
    --test_env=PATH \
    //dgf/... || exit 1
