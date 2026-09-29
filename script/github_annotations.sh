#!/bin/bash
# Helpers to report errors as GitHub Actions annotations. Unlike the step logs,
# annotations can be read without being a repository administrator, and they
# are forwarded to Critique.
# This script IS exported by copybara.

# Prints a GitHub Actions error annotation when running on GitHub Actions.
emit_github_error() {
  local title="$1"
  local message="$2"
  if [ "${GITHUB_ACTIONS:-}" != "true" ]; then
    return 0
  fi
  # Remove the color codes (see --color=yes in .bazelrc).
  message=$(printf '%s' "${message}" | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g')
  message="${message//'%'/%25}"
  message="${message//$'\r'/}"
  message="${message//$'\n'/%0A}"
  title="${title//'%'/%25}"
  title="${title//,/%2C}"
  title="${title//:/%3A}"
  echo "::error title=${title}::${message}"
}
