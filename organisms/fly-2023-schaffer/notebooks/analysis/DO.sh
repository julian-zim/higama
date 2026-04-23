#!/usr/bin/env bash
#SBATCH --job-name=ANALYSIS_W
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=21-00:00:00
#SBATCH --mail-user=a12144285@unet.univie.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/lisc/home/user/zimmermann/Files/Workspace/hobe/organisms/worm-2025-fieseler/out/slurm/ANALYSIS.%A.out
#SBATCH --error=/lisc/home/user/zimmermann/Files/Workspace/hobe/organisms/worm-2025-fieseler/out/slurm/ANALYSIS.%A.err

module load Python/3.12.3-GCCcore-13.3.0
source "${HOME}/.venv/hgm/bin/activate"

python --version
echo "${VIRTUAL_ENV}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export XLA_FLAGS="--xla_gpu_enable_command_buffer="

bash scores_ns.sh
bash scores_ss.sh
bash manifolds_ns.sh
bash manifolds_ss.sh
bash traces_ns.sh
bash traces_ss.sh
