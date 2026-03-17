#!/bin/bash
set -e
CONDA_ENV=/Users/sunwenrong/opt/anaconda3/envs/stl_pi_planner
DIR="$(cd "$(dirname "$0")" && pwd)"
echo ">>> 开始重新编译..."
CC=/usr/bin/clang CXX=/usr/bin/clang++ \
LDFLAGS="-L${CONDA_ENV}/lib -Wl,-rpath,${CONDA_ENV}/lib" \
OpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${CONDA_ENV}/include" \
OpenMP_CXX_LIB_NAMES="omp" \
OpenMP_omp_LIBRARY="${CONDA_ENV}/lib/libomp.dylib" \
CMAKE_PREFIX_PATH="${CONDA_ENV}" \
${CONDA_ENV}/bin/pip install -v "${DIR}"
echo ">>> 编译完成！"
