#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source repro/matcha_env.sh
uv pip install --python "$VIRTUAL_ENV/bin/python" rich fvcore 'cmake<4' 'matplotlib==3.9.4' --no-build-isolation
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-build-isolation --no-deps ./baselines/MAtCha/2d-gaussian-splatting/submodules/diff-surfel-rasterization ./baselines/MAtCha/2d-gaussian-splatting/submodules/simple-knn
cd baselines/MAtCha/mast3r/dust3r/croco/models/curope
python setup.py build_ext --inplace
cd ../../../../..
export CMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
cmake -S 2d-gaussian-splatting/submodules/tetra-triangulation -B 2d-gaussian-splatting/submodules/tetra-triangulation -DPYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build 2d-gaussian-splatting/submodules/tetra-triangulation -j 8
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --no-build-isolation -e ./2d-gaussian-splatting/submodules/tetra-triangulation
python -c 'import torch, diff_surfel_rasterization, pytorch3d, simple_knn; from tetranerf.utils.extension import cpp;print("MAtCha native imports OK",torch.__version__)'
cd ../..
uv pip freeze --python "$VIRTUAL_ENV/bin/python" > repro/records/matcha_environment.txt
