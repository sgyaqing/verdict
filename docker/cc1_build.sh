#!/bin/bash
# Build causal-conv1d 1.7.0 from the PyPI sdist inside the image build.
#
# PyPI ships no wheel for this package, and transformers' Qwen3.5 path falls
# back to a reference conv without it (about 10% slower forward time with the
# kernel enabled). Everything here mirrors the recipe: the pip CUDA-13 nvcc
# toolchain, the
# libcudart.so linker-name symlink, and the sm_89 patch (upstream's hardcoded
# arch list skips Ada and embeds no PTX, so an unpatched wheel would not load
# on a 4090).
#
# Runs in a python:3.12-slim-trixie stage with torch==2.14.1+cu130 already
# installed (the build links against the runtime stage's exact torch).
set -euo pipefail

CC1_VERSION=1.7.0

# nvcc needs a host C++ compiler; the torch stage it builds on is a slim
# python image with none.
apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

# `nvidia` is a namespace package: __file__ is None, so resolve the toolkit
# location through __path__.
CU=$(python -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))")
export CUDA_HOME="$CU"

pip install --no-cache-dir -i https://pypi.nvidia.com \
    nvidia-cuda-nvcc==13.4.92 nvidia-cuda-cccl==13.3.4.3.1
# The nvcc wheel's dependency resolver downgrades this to 13.0.x; the runtime
# headers must match nvcc 13.4 or CCCL's compatibility check aborts the build.
pip install --no-cache-dir --force-reinstall --no-deps nvidia-cuda-runtime==13.4.92

# The pip runtime wheel ships libcudart.so.13 but not the linker name.
[ -e "$CU/lib/libcudart.so" ] || ln -s libcudart.so.13 "$CU/lib/libcudart.so"

"$CU/bin/nvcc" --version | tail -1

# pip download of an sdist builds its metadata (which needs torch on the
# index); fetch the tarball straight from the PyPI JSON API and pin by sha256.
read -r URL SHA < <(python - <<'PY'
import json, urllib.request
d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/causal-conv1d/1.7.0/json", timeout=60))
f = [f for f in d["urls"] if f["packagetype"] == "sdist"][0]
print(f["url"], f["digests"]["sha256"])
PY
)
python -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], '/tmp/cc1.tar.gz')" "$URL"
echo "$SHA  /tmp/cc1.tar.gz" | sha256sum -c -

mkdir -p /cc1src
tar xzf /tmp/cc1.tar.gz -C /cc1src --strip-components=1
cd /cc1src

python - <<'PY'
src = open("setup.py").read()
needle = '        cc_flag.append("-gencode")\n        cc_flag.append("arch=compute_87,code=sm_87")'
patch = needle + '\n        cc_flag.append("-gencode")\n        cc_flag.append("arch=compute_89,code=sm_89")'
assert needle in src and "compute_89" not in src, "setup.py layout changed upstream -- re-derive the patch"
open("setup.py", "w").write(src.replace(needle, patch))
print("setup.py patched: sm_89 added")
PY

mkdir -p /wheels
pip wheel --no-deps --no-build-isolation -w /wheels .
ls -l /wheels/causal_conv1d-*.whl
