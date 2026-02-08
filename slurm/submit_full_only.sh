#!/bin/bash
#SBATCH --job-name=minimal_full_only
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1,vram:40G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/minimal_full_only_%j.out
#SBATCH --error=logs/minimal_full_only_%j.err

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load gcc/14.2.0
module load conda/miniforge3/24.11.3-0
module load cuda/12.8

CONDA_ENV_NAME=${CONDA_ENV_NAME:-snp2pheno}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

REPEATS=${REPEATS:-1}
SEED_START=${SEED_START:-42}
EPOCHS=${EPOCHS:-50}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-experiments/minimal_full_only}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "${BASE_OUTPUT_DIR}"

echo "Job ID:          ${SLURM_JOB_ID:-interactive}"
echo "Node:            ${SLURMD_NODENAME:-unknown}"
echo "Working dir:     $(pwd)"
echo "Conda env:       ${CONDA_ENV_NAME}"
echo "Repeats:         ${REPEATS}"
echo "Seed start:      ${SEED_START}"
echo "Epochs:          ${EPOCHS}"
echo "Base output dir: ${BASE_OUTPUT_DIR}"
echo "Extra args:      ${EXTRA_ARGS}"

echo "GPU inventory:"
nvidia-smi || true

SUCCESS=0
FAILED=0

for r in $(seq 0 $((REPEATS - 1))); do
    SEED=$((SEED_START + r))
    OUT_DIR="${BASE_OUTPUT_DIR}/seed_${SEED}"
    mkdir -p "${OUT_DIR}"

    CMD=(python run.py --task-mode full --seed "${SEED}" --epochs "${EPOCHS}" --output-dir "${OUT_DIR}")
    if [[ -n "${EXTRA_ARGS}" ]]; then
        read -r -a EXTRA_ARR <<< "${EXTRA_ARGS}"
        CMD+=("${EXTRA_ARR[@]}")
    fi

    echo ""
    echo "Run $((r + 1))/${REPEATS} | seed=${SEED}"
    printf 'Command: %q ' "${CMD[@]}"
    echo ""

    export PYTHONHASHSEED=${SEED}
    export PHEMART_BATCH_ID=${SLURM_JOB_ID:-manual}
    export PHEMART_BATCH_NAME=minimal_full_only
    export PHEMART_RUN_ID=full_seed_${SEED}

    if srun --unbuffered "${CMD[@]}"; then
        SUCCESS=$((SUCCESS + 1))
        echo "Run seed=${SEED} succeeded"
    else
        EXIT_CODE=$?
        FAILED=$((FAILED + 1))
        echo "Run seed=${SEED} failed (exit=${EXIT_CODE})"
    fi
done

echo ""
echo "Finished at: $(date)"
echo "Summary: success=${SUCCESS}, failed=${FAILED}, total=${REPEATS}"

if [ ${FAILED} -gt 0 ]; then
    exit 1
fi

exit 0
