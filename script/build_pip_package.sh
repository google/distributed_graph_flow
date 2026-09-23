#!/bin/bash
# This script runs in the exported directory to prepare the build and run Docker.
# This script IS exported by copybara.

# Warning: Make sure to update the DGF version before creating a new pip:
# - third_party/py/dgf/setup.py

set -vex

which patchelf >/dev/null 2>&1 || sudo apt-get install -y patchelf

# Create the missing __init__.py files
find dgf -type d -exec sh -c 'touch "$1/__init__.py"' _ {} \;

docker build -t dgf-builder .

rm -fr package
mkdir -p package/dgf

# Copy python files
(cd dgf && find . -name '*.py' -type f -exec cp --parents {} ../package/dgf/ \; )

cp LICENSE setup.py requirements.txt README.md package/

# We will pass the python versions to build.sh
PYTHON_VERSIONS=( 3.11 3.12 3.13 )

INTERACTIVE_FLAG="-i"
[ -t 0 ] && INTERACTIVE_FLAG="-it"

DOCKER_OPTS=(
  --rm
  "$INTERACTIVE_FLAG"
  -e PYTHONDONTWRITEBYTECODE=1
  -v dgf_bazel_cache:/root/.cache
  -v "$(pwd):/work"
  -w /work
)

chmod +x script/build.sh
chmod +x script/test.sh

for PYVERSION in ${PYTHON_VERSIONS[*]} ; do
  # Version-specific venv cache volume
  VENV_VOLUME="dgf_venv_cache_${PYVERSION//./}"

  # Run tests
  docker run "${DOCKER_OPTS[@]}" \
      -v "${VENV_VOLUME}:/tmp/venv" \
      --entrypoint /work/script/test.sh \
      dgf-builder "$PYVERSION"

  # Build package
  docker run "${DOCKER_OPTS[@]}" \
      -v "${VENV_VOLUME}:/tmp/venv" \
      --entrypoint /work/script/build.sh \
      dgf-builder "$PYVERSION"
done
