#!/bin/bash
#SBATCH -J amgx-build
#SBATCH -p medium
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/%u/amgx-build-%j.out

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$submit_dir"

# Replace these with the exact module names shown by:
#   module avail gcc
#   module avail cuda
#   module avail openmpi
#   module avail cmake
module load gcc || true
module load cuda || true
module load openmpi || true
module load cmake || true

export BUILD_JOBS="${SLURM_CPUS_PER_TASK}"
export CUDA_ARCH=70
export AMGX_WITH_MPI=1

bash "$submit_dir/paramshakti_build_amgx.sh"
