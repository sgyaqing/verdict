# Verdict serving image (CPU): base image plus one layer, the binary.
# Build (docker/release.sh chains both steps; full source build: Dockerfile.build):
#   docker buildx build -f Dockerfile.build --target verdict-bin \
#       --output type=local,dest=bin/ .
#   docker buildx build -f Dockerfile --build-context bin=bin \
#       --platform linux/amd64 --build-arg VERSION=<ver> --build-arg REVISION=<sha> \
#       -t <ns>/verdict:<ver>-cpu --load .
# Run:  docker run -p 8080:8080 -v $PWD/<model-dir>:/model:ro verdict:<ver>-cpu

# Move BASE together with the dependency pins (rebuilt by Dockerfile.base-cpu).
ARG BASE=sgyaqing/verdict:base-0.2-cpu
FROM ${BASE}

# No maintainer/author labels.
ARG VERSION=0.0.0
ARG REVISION=
LABEL org.opencontainers.image.title="Verdict" \
      org.opencontainers.image.description="A high-performance serving engine for System One decision models" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="https://github.com/sgyaqing/verdict" \
      org.opencontainers.image.url="https://github.com/sgyaqing/verdict" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=bin /verdict /usr/local/bin/verdict
