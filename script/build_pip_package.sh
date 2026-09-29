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

if [ -n "$1" ]; then
  PYTHON_VERSIONS=( "$1" )
else
  PYTHON_VERSIONS=( 3.11 3.12 3.13 )
fi

INTERACTIVE_FLAG=""
[ -t 0 ] && INTERACTIVE_FLAG="-it" || INTERACTIVE_FLAG="-i"

BAZEL_CACHE="${BAZEL_CACHE:-dgf_bazel_cache}"

DOCKER_OPTS=(
  --rm
  "$INTERACTIVE_FLAG"
  -e PYTHONDONTWRITEBYTECODE=1
  # Lets script/test.sh report failures as GitHub Actions annotations.
  -e GITHUB_ACTIONS
  # Optional override of the number of tests run in parallel (see test.sh).
  -e DGF_TEST_JOBS
  -v "${BAZEL_CACHE}:/root/.cache"
  -v "$(pwd):/work"
  -w /work
)

chmod +x script/build.sh
chmod +x script/test.sh

# shellcheck source=script/github_annotations.sh
source script/github_annotations.sh

# Runs a build stage. On GitHub Actions, a failure is reported as an error
# annotation with the end of the stage output.
run_stage() {
  local stage="$1"
  shift
  local stage_log
  stage_log=$(mktemp)
  if ! "$@" 2>&1 | tee "${stage_log}"; then
    set +x
    emit_github_error "${stage} failed (Python ${PYVERSION})" \
      "$(tail -n 50 "${stage_log}")"
    rm -f "${stage_log}"
    exit 1
  fi
  rm -f "${stage_log}"
}

set -o pipefail
for PYVERSION in ${PYTHON_VERSIONS[*]} ; do
  # Version-specific venv cache volume
  VENV_VOLUME="dgf_venv_cache_${PYVERSION//./}"

  # Run tests
  run_stage "Tests" docker run "${DOCKER_OPTS[@]}" \
      -v "${VENV_VOLUME}:/tmp/venv" \
      --entrypoint /work/script/test.sh \
      dgf-builder "$PYVERSION"

  # Build package
  run_stage "Package build" docker run "${DOCKER_OPTS[@]}" \
      -v "${VENV_VOLUME}:/tmp/venv" \
      --entrypoint /work/script/build.sh \
      dgf-builder "$PYVERSION"
done
