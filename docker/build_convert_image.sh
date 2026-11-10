#!/usr/bin/env bash
# Build the verdict-convert image pair on the GPU box (domestic mirrors).
# Run AFTER any measurement window: the pip install spikes CPU for a minute.
set -euo pipefail
cd ~/verdict-0.4pk

docker build -f Dockerfile.base-convert -t sgyaqing/verdict:base-0.4-convert . \
  2>&1 | tee /tmp/base-convert-build.log | tail -3

cp target/release/verdict ./verdict
trap 'rm -f ./verdict' EXIT
docker build -f Dockerfile.convert -t sgyaqing/verdict:0.4.0-convert . \
  2>&1 | tee /tmp/convert-build.log | tail -3

docker run --rm sgyaqing/verdict:0.4.0-convert --version
docker run --rm --entrypoint python3.12 sgyaqing/verdict:0.4.0-convert -c \
  "import torch, transformers, onnx, onnxscript, onnxruntime, von, laya; \
   print('env', torch.__version__, transformers.__version__, onnx.__version__)"

set +e
out=$(docker run --rm sgyaqing/verdict:0.4.0-convert convert /nonexistent /tmp/out 2>&1)
rc=$?
set -e
if [ "$rc" -ne 2 ] || ! printf '%s' "$out" | grep -q "error: /nonexistent:"; then
  printf 'unexpected convert error path (exit %s):\n%s\n' "$rc" "$out"
  exit 1
fi
echo "CONVERT_IMAGE_DONE_X41"
