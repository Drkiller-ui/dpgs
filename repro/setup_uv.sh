#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Linux ]] || { echo 'Run on the Linux GPU server.' >&2; exit 1; }
for cmd in uv git nvcc gcc g++; do
    command -v "$cmd" >/dev/null || { echo "Missing system tool: $cmd" >&2; exit 1; }
done
nvcc --version | grep -q 'release 12.4' || {
    echo 'Use the CUDA 12.4 toolkit (nvcc), matching torch cu124.' >&2; exit 1;
}
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export MAX_JOBS="${MAX_JOBS:-4}"
uv python install 3.10.20
[[ -d .venv ]] || uv venv --python 3.10.20 .venv
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
# Only torch/torchvision use the cu124 index; other dependencies use PyPI.
[[ -f uv.lock ]] || uv lock
uv sync --frozen --extra visualization
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0)); print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)'

# Only environment-reporting import changed; no geometry/training code changes.
python repro/patch_metric3d.py
uv pip install --no-build-isolation --no-deps \
    ./submodules/diff-plane-rasterization ./submodules/simple-knn
# PyTorch3D 0.7.8 has no v0.7.8 tag: pin the official version-bump commit.
uv pip install --no-build-isolation --no-deps \
    'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
(
    cd mast3r/asmk/cython
    cythonize ./*.pyx
)
uv pip install --no-build-isolation --no-deps ./mast3r/asmk
(
    cd mast3r/dust3r/croco/models/curope
    python setup.py build_ext --inplace
)
python repro/check_env.py
mkdir -p repro/records
uv pip freeze > repro/records/installed.txt
if [[ -e .git ]]; then
    git rev-parse HEAD > repro/records/upstream-commit.txt
else
    cp repro/upstream_commit.txt repro/records/upstream-commit.txt
fi
nvidia-smi > repro/records/nvidia-smi.txt
nvcc --version > repro/records/nvcc.txt
echo 'Environment installed and import checks passed. Next: weights + DTU data.'
