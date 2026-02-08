#!/bin/bash
#
# Minimal PheMART2 多实验提交脚本（每个实验重复3次）
#
# 用法：
#   sbatch submit_minimal_multitask_suite_3x.sh
#
# 可选环境变量：
#   REPEATS=3
#   SEED_START=42
#   EPOCHS=50
#   BASE_OUTPUT_DIR=experiments/minimal_multitask_suite
#   EXPERIMENT_MATRIX="exp_a::--aux-update-hgt 1|exp_b::--use-inductive-graph-train 1"
#
# EXPERIMENT_MATRIX 格式：
#   name::args|name::args
# 示例：
#   EXPERIMENT_MATRIX="baseline::|inductive::--use-inductive-graph-train 1" sbatch submit_minimal_multitask_suite_3x.sh

#SBATCH --job-name=minimal_mt_suite
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1,vram:40G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/minimal_mt_suite_%j.out
#SBATCH --error=logs/minimal_mt_suite_%j.err

set -u -o pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# ============================================================================
# 环境配置
# ============================================================================

module load gcc/14.2.0
module load conda/miniforge3/24.11.3-0
module load cuda/12.8

CONDA_ENV_NAME="snp2pheno"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment: ${CONDA_ENV_NAME}"
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# ============================================================================
# 参数
# ============================================================================

REPEATS=${REPEATS:-3}
SEED_START=${SEED_START:-42}
EPOCHS=${EPOCHS:-50}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-experiments/minimal_multitask_suite}

# 默认基础任务组合（可用 EXPERIMENT_MATRIX 覆盖）
DEFAULT_EXPERIMENT_NAMES=(
  "main_only"
  "main_domain"
  "main_domain_mvp"
  "full"
)
DEFAULT_EXPERIMENT_ARGS=(
  "--task-mode main_only"
  "--task-mode main_domain"
  "--task-mode main_domain_mvp"
  "--task-mode full"
)

EXPERIMENT_NAMES=()
EXPERIMENT_ARGS=()

if [[ -n "${EXPERIMENT_MATRIX:-}" ]]; then
    IFS='|' read -r -a PAIRS <<< "${EXPERIMENT_MATRIX}"
    for pair in "${PAIRS[@]}"; do
        name="${pair%%::*}"
        args="${pair#*::}"
        if [[ -z "${name}" || "${name}" == "${args}" ]]; then
            echo "❌ Invalid EXPERIMENT_MATRIX item: ${pair}"
            echo "   Expected format: name::args"
            exit 1
        fi
        EXPERIMENT_NAMES+=("${name}")
        EXPERIMENT_ARGS+=("${args}")
    done
else
    EXPERIMENT_NAMES=("${DEFAULT_EXPERIMENT_NAMES[@]}")
    EXPERIMENT_ARGS=("${DEFAULT_EXPERIMENT_ARGS[@]}")
fi

NUM_EXPS=${#EXPERIMENT_NAMES[@]}
if [ ${NUM_EXPS} -eq 0 ]; then
    echo "❌ No experiments configured"
    exit 1
fi

mkdir -p "${BASE_OUTPUT_DIR}"

# ============================================================================
# 信息
# ============================================================================

echo "🚀 Minimal Multi-task Experiment Suite"
echo "======================================"
echo "Job ID:            ${SLURM_JOB_ID:-interactive}"
echo "Node:              ${SLURMD_NODENAME:-unknown}"
echo "Working dir:       $(pwd)"
echo "Repeats:           ${REPEATS}"
echo "Seed start:        ${SEED_START}"
echo "Epochs:            ${EPOCHS}"
echo "Base output dir:   ${BASE_OUTPUT_DIR}"
echo "Num experiments:   ${NUM_EXPS}"
echo "Start time:        $(date)"
echo ""
echo "🖥️ GPU inventory:"
nvidia-smi || true

echo ""
echo "📋 Experiment plan:"
for i in $(seq 0 $((NUM_EXPS - 1))); do
    echo "  - ${EXPERIMENT_NAMES[$i]} :: ${EXPERIMENT_ARGS[$i]}"
done

# ============================================================================
# 执行
# ============================================================================

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_RUNS=()
TOTAL_RUNS=$((NUM_EXPS * REPEATS))
RUN_IDX=0

for i in $(seq 0 $((NUM_EXPS - 1))); do
    EXP_NAME="${EXPERIMENT_NAMES[$i]}"
    EXP_ARGS_STR="${EXPERIMENT_ARGS[$i]}"

    for r in $(seq 0 $((REPEATS - 1))); do
        RUN_IDX=$((RUN_IDX + 1))
        CURRENT_SEED=$((SEED_START + r))
        OUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}/seed_${CURRENT_SEED}"

        mkdir -p "${OUT_DIR}"

        CMD=(python run.py --seed "${CURRENT_SEED}" --epochs "${EPOCHS}" --output-dir "${OUT_DIR}")
        if [[ -n "${EXP_ARGS_STR}" ]]; then
            read -r -a EXTRA_ARGS <<< "${EXP_ARGS_STR}"
            CMD+=("${EXTRA_ARGS[@]}")
        fi

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Run ${RUN_IDX}/${TOTAL_RUNS} | exp=${EXP_NAME} | repeat=$((r+1))/${REPEATS} | seed=${CURRENT_SEED}"
        echo "Output: ${OUT_DIR}"
        printf 'Command: %q ' "${CMD[@]}"
        echo ""

        export PYTHONHASHSEED=${CURRENT_SEED}
        export PHEMART_BATCH_ID=${SLURM_JOB_ID:-manual}
        export PHEMART_BATCH_NAME="minimal_mt_suite"
        export PHEMART_RUN_ID="${EXP_NAME}_seed_${CURRENT_SEED}"

        if srun --unbuffered "${CMD[@]}"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "✅ Success"
        else
            EXIT_CODE=$?
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAILED_RUNS+=("${EXP_NAME}:seed_${CURRENT_SEED}:exit_${EXIT_CODE}")
            echo "❌ Failed (exit=${EXIT_CODE})"
        fi
    done
done

# ============================================================================
# 总结
# ============================================================================

echo ""
echo "🏁 Finished at: $(date)"
echo "Summary: success=${SUCCESS_COUNT}, failed=${FAIL_COUNT}, total=${TOTAL_RUNS}"

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "Failed runs:"
    for x in "${FAILED_RUNS[@]}"; do
        echo "  - ${x}"
    done
    exit 1
fi

exit 0
