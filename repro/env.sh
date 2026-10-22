#!/usr/bin/env bash
# Source from the project root before build/train/eval. Explicit cu124 selection.
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:/root/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=8
export UV_CACHE_DIR=/root/autodl-tmp/dpgs-cache/uv
export UV_HTTP_TIMEOUT=120
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export LOKY_MAX_CPU_COUNT=8
if [[ -d .venv/bin ]]; then
    export VIRTUAL_ENV="$PWD/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
fi
