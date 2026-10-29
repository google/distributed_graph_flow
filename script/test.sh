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
${PYBIN} -m pip install -r requirements.txt -r requirements-dev.txt
if [ "$(uname -s)" = "Darwin" ]; then
  # Installed with --no-deps to bypass its obsolete `tensorflow-macos`
  # dependency marker (see requirements-dev.txt); its actual runtime
  # dependencies (tensorflow, apache-beam, networkx, pyarrow, ml-collections,
  # google-vizier) are installed below.
  ${PYBIN} -m pip install --no-deps tensorflow-gnn
  ${PYBIN} -m pip install ml-collections google-vizier
fi

# Check that absl is installed
${PYBIN} -c "import absl" || { echo "Error: absl is not installed in the virtual environment."; exit 1; }

export TF_USE_LEGACY_KERAS=1

# shellcheck source=script/github_annotations.sh
source "$(dirname "$0")/github_annotations.sh"

# Prints the relevant part of a test log: From the first Python test failure,
# or the end of the log if there is none.
extract_test_failure() {
  local log="$1"
  if grep -qE '^(ERROR|FAIL): ' "${log}"; then
    sed -En '/^(ERROR|FAIL): /,$p' "${log}" | head -n 80
  else
    tail -n 60 "${log}"
  fi
}

# When running on GitHub Actions, reports the failing tests as error
# annotations.
report_failures_to_github() {
  local bazel_log="$1"
  if [ "${GITHUB_ACTIONS:-}" != "true" ]; then
    return 0
  fi
  set +v +x
  # Remove the color codes (see --color=yes in .bazelrc). The escape character
  # is given with $'...' since BSD sed (macOS) does not support "\x1b".
  local clean_log="${bazel_log}.clean"
  sed -E $'s/\x1b\\[[0-9;]*[A-Za-z]//g' "${bazel_log}" > "${clean_log}"
  local failed_targets
  failed_targets=$( (grep -E '^//[^ ]+ +(FAILED|TIMEOUT|NO STATUS|INCOMPLETE)' \
    "${clean_log}" || true) | awk '{print $1}' | sort -u)
  if [ -z "${failed_targets}" ]; then
    # E.g. build error, or Bazel crash.
    emit_github_error "bazel test failed (Python ${PYVERSION})" \
      "$(tail -n 50 "${clean_log}")"
    return 0
  fi
  local target pkg name log status emitted=0
  for target in ${failed_targets}; do
    status=$( (grep -E "^${target} +" "${clean_log}" || true) | head -n 1 \
      | sed -E 's/^[^ ]+ +//')
    pkg="${target#//}"
    pkg="${pkg%%:*}"
    name="${target##*:}"
    for log in "bazel-testlogs/${pkg}/${name}/test.log" \
      bazel-testlogs/"${pkg}/${name}"/shard_*/test.log; do
      if [ ! -f "${log}" ]; then
        continue
      fi
      # Skip the shards that passed.
      if tail -n 5 "${log}" | grep -qE '^OK( |$)'; then
        continue
      fi
      emit_github_error "${target} ${status} (Python ${PYVERSION})" \
        "$(extract_test_failure "${log}")"
      emitted=1
    done
  done
  # If any target failed without a test.log (e.g. FAILED TO BUILD or
  # NO STATUS), also emit the ERROR: lines from the Bazel log.
  if grep -qE '^ERROR: ' "${clean_log}"; then
    emit_github_error "bazel errors (Python ${PYVERSION})" \
      "$(grep -B 1 -A 6 -E '^ERROR: ' "${clean_log}" | tail -n 80)"
    emitted=1
  fi
  if [ "${emitted}" -eq 0 ]; then
    emit_github_error "bazel test failed (Python ${PYVERSION})" \
      "$(grep -E '^(ERROR: |//[^ ]+ +(FAILED|TIMEOUT|NO STATUS|INCOMPLETE))' "${clean_log}" | tail -n 60)"
  fi
}

# Number of tests to run in parallel. Each test can use several GB of memory
# (TensorFlow + JAX), so running one test per core gets tests OOM-killed on
# machines with little memory per core (e.g. GitHub Actions runners: 4 cores,
# 16GB). Can be overridden with DGF_TEST_JOBS.
if [ -z "${DGF_TEST_JOBS:-}" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    MEM_KB=$(( $(sysctl -n hw.memsize) / 1024 ))
    NUM_CPUS=$(sysctl -n hw.ncpu)
  else
    MEM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    NUM_CPUS=$(nproc)
  fi
  # Honor the memory limit of the container, if any (cgroup v2).
  if [ -r /sys/fs/cgroup/memory.max ] \
      && [ "$(cat /sys/fs/cgroup/memory.max)" != "max" ]; then
    CGROUP_MEM_KB=$(( $(cat /sys/fs/cgroup/memory.max) / 1024 ))
    if [ "${CGROUP_MEM_KB}" -lt "${MEM_KB}" ]; then
      MEM_KB="${CGROUP_MEM_KB}"
    fi
  fi
  DGF_TEST_JOBS=$(( MEM_KB / (6 * 1024 * 1024) ))
  if [ "${DGF_TEST_JOBS}" -lt 1 ]; then
    DGF_TEST_JOBS=1
  fi
  if [ "${DGF_TEST_JOBS}" -gt "${NUM_CPUS}" ]; then
    DGF_TEST_JOBS="${NUM_CPUS}"
  fi
fi
echo "Running ${DGF_TEST_JOBS} tests in parallel"

# Fingerprint of the installed Python packages. Bazel does not track the
# packages installed in the venv, so without it, cached test results would
# survive a dependency change (e.g. a new jax release).
# Computed with Python since "sha256sum" is not available on macOS.
VENV_FINGERPRINT=$(${PYBIN} -m pip freeze | ${PYBIN} -c "import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())")

# Run all tests via Bazel
echo "Running all tests via Bazel..."
BAZEL_TEST_LOG=/tmp/bazel_test_output.log
set -o pipefail
if ! bazel test \
    --test_output=errors \
    --local_test_jobs="${DGF_TEST_JOBS}" \
    --test_timeout=120,900,1800,7200 \
    --keep_going \
    --spawn_strategy=standalone \
    --test_strategy=standalone \
    --test_env=TF_USE_LEGACY_KERAS \
    --test_env=PYTHONPATH \
    --test_env=PATH \
    --test_env=DGF_VENV_FINGERPRINT="${VENV_FINGERPRINT}" \
    --@rules_python//python/config_settings:python_version="${PYVERSION}" \
    //dgf/... 2>&1 | tee "${BAZEL_TEST_LOG}"; then
  report_failures_to_github "${BAZEL_TEST_LOG}"
  exit 1
fi
