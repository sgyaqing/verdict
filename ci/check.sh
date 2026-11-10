#!/usr/bin/env bash
# The CI pipeline as runnable commands: format, then lint; called by .github/workflows/ci.yml.
set -euo pipefail

cargo fmt --check
cargo clippy --all-targets -- -D warnings
