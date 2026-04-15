#!/bin/bash
#SBATCH -J amgx-test
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o /scratch/%u/amgx-test-%j.out

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$submit_dir"

module load gcc || true
module load cuda || true
module load openmpi || true
module load cmake || true

AMGX_SRC_DIR="${AMGX_SRC_DIR:-$HOME/software/src/amgx}"
AMGX_BUILD_DIR="${AMGX_BUILD_DIR:-$AMGX_SRC_DIR/build}"

export LD_LIBRARY_PATH="$HOME/software/amgx/lib:${LD_LIBRARY_PATH:-}"

"$AMGX_BUILD_DIR/examples/amgx_capi" \
  -m "$AMGX_SRC_DIR/examples/matrix.mtx" \
  -c "$AMGX_SRC_DIR/src/configs/FGMRES_AGGREGATION.json"
