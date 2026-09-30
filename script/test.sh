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
    sed -n '/^\(ERROR\|FAIL\): /,$p' "${log}" | head -n 80
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
  set +x
  # Remove the color codes (see --color=yes in .bazelrc).
  local clean_log="${bazel_log}.clean"
  sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g' "${bazel_log}" > "${clean_log}"
  local failed_targets
  failed_targets=$( (grep -E '^//[^ ]+ +(FAILED|TIMEOUT|NO STATUS|INCOMPLETE)' \
    "${clean_log}" || true) | awk '{print $1}' | sort -u)
  if [ -z "${failed_targets}" ]; then
    # E.g. build error, or Bazel crash.
    emit_github_error "bazel test failed (Python ${PYVERSION})" \
      "$(tail -n 50 "${clean_log}")"
    return 0
  fi
  local target pkg name log status
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
    done
  done
}

# Number of tests to run in parallel. Each test can use several GB of memory
# (TensorFlow + JAX), so running one test per core gets tests OOM-killed on
# machines with little memory per core (e.g. GitHub Actions runners: 4 cores,
# 16GB). Can be overridden with DGF_TEST_JOBS.
if [ -z "${DGF_TEST_JOBS:-}" ]; then
  MEM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
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
  if [ "${DGF_TEST_JOBS}" -gt "$(nproc)" ]; then
    DGF_TEST_JOBS="$(nproc)"
  fi
fi
echo "Running ${DGF_TEST_JOBS} tests in parallel"

# Fingerprint of the installed Python packages. Bazel does not track the
# packages installed in the venv, so without it, cached test results would
# survive a dependency change (e.g. a new jax release).
VENV_FINGERPRINT=$(${PYBIN} -m pip freeze | sha256sum | cut -d' ' -f1)

# Run all tests via Bazel
echo "Running all tests via Bazel..."
BAZEL_TEST_LOG=/tmp/bazel_test_output.log
set -o pipefail
if ! bazel test \
    --test_output=errors \
    --local_test_jobs="${DGF_TEST_JOBS}" \
    --test_timeout=120,900,1800,7200 \
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
