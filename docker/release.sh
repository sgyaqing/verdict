#!/usr/bin/env bash
# Verdict release-image build: compile the binary, stack it as one thin layer
# on the prebuilt base.
#
# Usage: docker/release.sh VERSION FLAVOR [REVISION]
#   VERSION   e.g. 0.2.0
#   FLAVOR    cpu | cuda
#   REVISION  git sha baked into the image label (optional)
#
# Env:
#   APT_MIRROR    debian mirror for the builder stage (default: stock sources)
#   CARGO_MIRROR  sparse cargo index for the builder stage
#   BASE_TAG      override the base image
#                 (default: sgyaqing/verdict:base-0.2-<flavor>)
#   ACR_REGISTRY  when set, the image is also tagged
#                 <ACR_REGISTRY>/sgyaqing/verdict:<ver>-<flavor>
#
# With warm caches a code-only change rebuilds in < 2 min and pushes one
# ~15 MB layer; a build that starts re-downloading torch means BASE_TAG has
# drifted from the built base -- fix the tag instead of waiting.
set -euo pipefail

VERSION=${1:?usage: release.sh VERSION FLAVOR [REVISION]}
FLAVOR=${2:?usage: release.sh VERSION FLAVOR [REVISION]}
REVISION=${3:-}
BASE_TAG=${BASE_TAG:-sgyaqing/verdict:base-0.2-${FLAVOR}}

case "$FLAVOR" in
  cpu)  BUILD_FILE=Dockerfile.build      THIN_FILE=Dockerfile ;;
  cuda) BUILD_FILE=Dockerfile.build.cuda THIN_FILE=Dockerfile.cuda ;;
  *) echo "unknown flavor: $FLAVOR (cpu|cuda)" >&2; exit 2 ;;
esac

step() { printf '\n== %s ==\n' "$*"; }

step "binary"
rm -rf out && mkdir out
docker buildx build -f "$BUILD_FILE" --target verdict-bin \
    --platform linux/amd64 \
    --build-arg APT_MIRROR="${APT_MIRROR:-}" \
    --build-arg CARGO_MIRROR="${CARGO_MIRROR:-}" \
    --output type=local,dest=out .
test -s out/verdict || { echo "binary missing after build" >&2; exit 1; }
ls -lh out/verdict

step "thin image"
docker buildx build -f "$THIN_FILE" \
    --build-context bin=out --platform linux/amd64 \
    --build-arg BASE="$BASE_TAG" \
    --build-arg VERSION="$VERSION" --build-arg REVISION="$REVISION" \
    --provenance=false --sbom=false \
    -t "sgyaqing/verdict:${VERSION}-${FLAVOR}" --load .

if [ -n "${ACR_REGISTRY:-}" ]; then
  docker tag "sgyaqing/verdict:${VERSION}-${FLAVOR}" \
             "${ACR_REGISTRY}/sgyaqing/verdict:${VERSION}-${FLAVOR}"
fi

step "layer report"
docker history "sgyaqing/verdict:${VERSION}-${FLAVOR}" \
    --format '{{.Size}}\t{{.CreatedBy}}' --no-trunc | cut -c1-120 | head -6
