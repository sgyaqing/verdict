# Verdict serving image (CPU): one binary + onnxruntime on debian-slim.
# The model is NOT baked in -- mount it at /model.
#
# Build:  docker buildx build --platform linux/amd64 -t <ns>/verdict:0.1.0-cpu .
# Run:    docker run -p 8080:8080 -v $PWD/<model-dir>:/model:ro verdict:0.1.0-cpu
# PIP_INDEX_URL=<mirror> redirects the wheel download below.

# ---- onnxruntime --------------------------------------------------------
# The .so comes from the official wheel (arch-specific) and the version is
# pinned: the engine loads the library by soname.
FROM python:3.12-slim AS ort
ARG ORT_VERSION=1.30.0
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends unzip; \
    rm -rf /var/lib/apt/lists/*; \
    pip download --no-deps --only-binary=:all: -i "${PIP_INDEX_URL}" \
        -d /tmp/ort "onnxruntime==${ORT_VERSION}"; \
    mkdir -p /ort; \
    unzip -j -o /tmp/ort/onnxruntime-*.whl 'onnxruntime/capi/libonnxruntime.so*' -d /ort; \
    ls -l /ort

# ---- builder ------------------------------------------------------------
FROM rust:1.90-slim AS builder
WORKDIR /build

# CARGO_MIRROR=<sparse index> routes cargo through a mirror on slow routes.
ARG CARGO_MIRROR=""
RUN if [ -n "$CARGO_MIRROR" ]; then \
      printf '[source.crates-io]\nreplace-with = "mirror"\n\n[source.mirror]\nregistry = "%s"\n' \
        "$CARGO_MIRROR" > "${CARGO_HOME:-/usr/local/cargo}/config.toml"; \
    fi

# g++ (cc-rs builds C/C++); its own RUN above the COPYs so src-only changes
# reuse the cached layer.
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime loads dynamically at runtime (ort load-dynamic): the build
# itself needs no onnxruntime headers. Only the manifest + src are needed:
# `cargo build --release` compiles the lib and bins, never the tests.
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# Cache mounts survive across builds, so src-only changes recompile only
# this crate. Mount contents are NOT in the layer snapshot: the binary must
# be copied out in the SAME RUN -- a later COPY of target/ fails "not found".
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release && cp target/release/verdict ./verdict

# ---- runtime ------------------------------------------------------------
FROM debian:trixie-slim

# No maintainer/author fields: published artifacts carry no personal
# identifiers. VERSION/REVISION come from the build script.
ARG VERSION=0.0.0
ARG REVISION=
LABEL org.opencontainers.image.title="Verdict" \
      org.opencontainers.image.description="A high-performance serving engine for System One decision models" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="https://github.com/sgyaqing/verdict" \
      org.opencontainers.image.url="https://github.com/sgyaqing/verdict" \
      org.opencontainers.image.licenses="Apache-2.0"

# Must match the builder's glibc generation (rust:1.90-slim = trixie); a
# bookworm runtime fails with GLIBC_2.38-not-found.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/verdict /usr/local/bin/verdict
# ldconfig puts the .so on the loader path ort's fallback dlopens; the
# engine's exe-dir lookup does NOT see this copy.
COPY --from=ort /ort/libonnxruntime.so.1.30.0 /usr/local/lib/libonnxruntime.so.1.30.0
RUN ln -s libonnxruntime.so.1.30.0 /usr/local/lib/libonnxruntime.so \
    && ldconfig

VOLUME /model
EXPOSE 8080
ENTRYPOINT ["verdict"]
CMD ["serve", "/model", "--host", "0.0.0.0", "--port", "8080"]
