#!/usr/bin/env bash
# The CI pipeline as runnable commands: format, then lint.
# .github/workflows/ci.yml calls this.
set -euo pipefail

cargo fmt --check
cargo clippy --all-targets -- -D warnings
