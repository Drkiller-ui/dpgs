#!/usr/bin/env bash
# Source from DP-GS root. Keep all baseline packages and CUDA modules isolated.
source repro/env.sh
export VIRTUAL_ENV="$PWD/baselines/MAtCha/.venv"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:/root/.local/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=8
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export PYTHONUNBUFFERED=1
