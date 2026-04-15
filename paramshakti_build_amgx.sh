#!/usr/bin/env bash

set -euo pipefail

# ParamShakti AMGX build helper.
# CUDA_ARCH defaults to a conservative Volta-era value and can be overridden.

AMGX_REPO_URL="${AMGX_REPO_URL:-https://github.com/NVIDIA/AMGX.git}"
AMGX_SRC_DIR="${AMGX_SRC_DIR:-$HOME/software/src/amgx}"
AMGX_BUILD_DIR="${AMGX_BUILD_DIR:-$AMGX_SRC_DIR/build}"
AMGX_INSTALL_DIR="${AMGX_INSTALL_DIR:-$HOME/software/amgx}"
CUDA_ARCH="${CUDA_ARCH:-70}"
BUILD_JOBS="${BUILD_JOBS:-8}"
AMGX_WITH_MPI="${AMGX_WITH_MPI:-1}"

if command -v module >/dev/null 2>&1; then
  module load gcc >/dev/null 2>&1 || true
  module load cuda >/dev/null 2>&1 || true
  module load openmpi >/dev/null 2>&1 || true
  module load cmake >/dev/null 2>&1 || true
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd git
need_cmd cmake
need_cmd nvcc
need_cmd gcc
need_cmd g++

if [[ "$AMGX_WITH_MPI" != "0" ]]; then
  need_cmd mpicc
  need_cmd mpicxx
fi

mkdir -p "$(dirname "$AMGX_SRC_DIR")" "$(dirname "$AMGX_INSTALL_DIR")"

if [[ ! -d "$AMGX_SRC_DIR/.git" ]]; then
  git clone --recursive "$AMGX_REPO_URL" "$AMGX_SRC_DIR"
else
  git -C "$AMGX_SRC_DIR" pull --ff-only
  git -C "$AMGX_SRC_DIR" submodule update --init --recursive
fi

mkdir -p "$AMGX_BUILD_DIR"
cd "$AMGX_BUILD_DIR"

cmake_args=(
  ..
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=gcc
  -DCMAKE_CXX_COMPILER=g++
  -DCMAKE_INSTALL_PREFIX="$AMGX_INSTALL_DIR"
  -DCUDA_ARCH="$CUDA_ARCH"
)

if [[ -n "${CUDA_HOME:-}" ]]; then
  cmake_args+=(-DCUDAToolkit_ROOT="$CUDA_HOME")
fi

if [[ "$AMGX_WITH_MPI" == "0" ]]; then
  cmake_args+=(-DCMAKE_NO_MPI=TRUE)
fi

echo "Configuring AMGX in $AMGX_BUILD_DIR"
cmake "${cmake_args[@]}"

echo "Building AMGX with $BUILD_JOBS parallel jobs"
cmake --build . -j "$BUILD_JOBS"
cmake --install .

echo
echo "AMGX installed to: $AMGX_INSTALL_DIR"
echo "Libraries: $AMGX_INSTALL_DIR/lib"
echo "Examples:  $AMGX_BUILD_DIR/examples"
