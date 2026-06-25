#!/bin/bash -l
#SBATCH --job-name=saratrx_pretrain
#SBATCH --output=saratrx_pretrain.o%j
#SBATCH --error=saratrx_pretrain.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=48:00:00
#SBATCH --account=project_462001182

set -euo pipefail

# ---------------------------------------------------------------------------
# Train SARATR-X (HiViT MAE) on SAR HDF5 tiles — 1 node, 8 MI250X GCDs.
#
# Standalone-clone form: uses isr-automatic-target-recognition's uv venv
# (extras `rocm`, `mtl-yolo`) + atr-base.sif, training script lives at
# ${SARATRX}/pre-training/train_h5_lumi.py.
#
# Usage:
#   sbatch /users/eacar/projects/SARATR-X-fm11/pre-training/train_saratrx_lumi.sh
#
# Resume: auto-resumes from checkpoint-latest.pth if present in OUTPUT_DIR.
# ---------------------------------------------------------------------------

# -- paths ----------------------------------------------------------------
ISR_REPO="${ISR_REPO:-${HOME}/projects/isr-automatic-target-recognition}"
SARATRX="${SARATRX:-/users/eacar/projects/SARATR-X-fm11}"

H5_TRAIN="/scratch/project_462001182/snow_owl/data/datasets/air_land_maritime_best_20260511_003_resampled_05/train.h5"
H5_TEST="/scratch/project_462001182/snow_owl/data/datasets/air_land_maritime_test_20260513_001_resampled_05/test.h5"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/project_462001182/foundation_model_dev/users/eacar/experiments/saratrx_pretrain}"

INIT_CKPT="${INIT_CKPT:-${SARATRX}/pre-training/mae_hivit_base_1600ep.pth}"

SIF="${SIF:-/scratch/project_462001182/snow_owl/containers/singularity/atr-base.sif}"

# -- hyperparameters ------------------------------------------------------
# For 768x768 resolution training (FM-3 experiment):
#   BATCH_SIZE=6, ACCUM_ITER=10 → effective = 6 * 10 * 8 = 480
#   BLR="3e-5" (lower LR for continued pretraining)
#   INPUT_SIZE=768
# For 224x224 (original):
#   BATCH_SIZE=64, ACCUM_ITER=1 → effective = 64 * 1 * 8 = 512
#   BLR="1.5e-4"
#   INPUT_SIZE=224
INPUT_SIZE="${INPUT_SIZE:-768}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-6}"
ACCUM_ITER="${ACCUM_ITER:-10}"
BLR="${BLR:-3e-5}"
MASK_RATIO=0.75
SAVE_INTERVAL=25
VAL_INTERVAL=25
WANDB_PROJECT="saratrx-pretrain"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# -- ROCm / MIOpen tuning ------------------------------------------------
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"

# -- NCCL hardening (LUMI MI250X can hang on collective init/broadcast) ---
# Default watchdog is 10 min — too tight when ~24 GCDs spin up simultaneously.
# init_process_group(timeout=30min) is also set in util/misc.py.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22

# -- uv cache on node-local NVMe -----------------------------------------
export UV_CACHE_DIR="/tmp/${USER}-uv-cache-${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:128"

# -- WandB ----------------------------------------------------------------
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.wandb_key ]]; then
    export WANDB_API_KEY=$(<~/.wandb_key tr -d '[:space:]')
fi

# -- Singularity binds ----------------------------------------------------
BIND="/scratch/project_462001182"
BIND="${BIND},${HOME}/projects:${HOME}/projects"
BIND="${BIND},/etc/ssl/certs:/etc/ssl/certs:ro"
[[ -f /etc/ssl/openssl.cnf ]] && BIND="${BIND},/etc/ssl/openssl.cnf:/etc/ssl/openssl.cnf:ro"
[[ -d /etc/pki ]]             && BIND="${BIND},/etc/pki:/etc/pki:ro"
BIND="${BIND},/etc/resolv.conf:/etc/resolv.conf:ro"
BIND="${BIND},/etc/hosts:/etc/hosts:ro"
BIND="${BIND},/etc/nsswitch.conf:/etc/nsswitch.conf:ro"
# Some containers ship without /opt/amdgpu/share/libdrm/amdgpu.ids; bind from host if available.
[[ -f /usr/share/libdrm/amdgpu.ids ]] && \
    BIND="${BIND},/usr/share/libdrm/amdgpu.ids:/opt/amdgpu/share/libdrm/amdgpu.ids:ro"
export SINGULARITY_BIND="${BIND}"
export SINGULARITYENV_SLURM_JOB_ID="${SLURM_JOB_ID}"

# -- select_gpu wrapper (inline) -----------------------------------------
# Each srun task sees exactly one GCD via ROCR_VISIBLE_DEVICES
SELECT_GPU_SCRIPT=$(mktemp /tmp/select_gpu_XXXX.sh)
cat > "${SELECT_GPU_SCRIPT}" <<'GPUEOF'
#!/bin/bash
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID}"
exec "$@"
GPUEOF
chmod +x "${SELECT_GPU_SCRIPT}"

# -- init ckpt sanity (skip if missing; train_h5_lumi handles --init_ckpt absence) --
INIT_CKPT_ARG=()
if [[ -f "${INIT_CKPT}" ]]; then
    INIT_CKPT_ARG=(--init_ckpt "${INIT_CKPT}")
else
    echo "WARNING: INIT_CKPT not found at ${INIT_CKPT} — falling back to random init / --resume auto."
fi

# -- launch ---------------------------------------------------------------
echo "=== SARATR-X pre-training ==="
echo "  Job ID     : ${SLURM_JOB_ID}"
echo "  Nodes      : ${SLURM_NODELIST}"
echo "  GPUs       : 8"
echo "  ISR repo   : ${ISR_REPO}"
echo "  SARATRX    : ${SARATRX}"
echo "  H5 train   : ${H5_TRAIN}"
echo "  H5 test    : ${H5_TEST}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Init ckpt  : ${INIT_CKPT} (present=$( [[ -f ${INIT_CKPT} ]] && echo yes || echo no ))"
echo "  Input size : ${INPUT_SIZE}"
echo "  Epochs     : ${EPOCHS}"
echo "  Batch/GPU  : ${BATCH_SIZE}"
echo "  Accum iter : ${ACCUM_ITER}"
echo "  Base LR    : ${BLR}"
echo "  Container  : ${SIF}"
echo ""

# Phase 1: sync venv once (single process) so srun tasks don't race
echo "==> Syncing venv (single process)..."
singularity exec --rocm "${SIF}" \
    bash -c "cd ${ISR_REPO} && uv sync --extra rocm --extra mtl-yolo && uv pip install timm==0.5.4 && uv pip install --force-reinstall matplotlib"
echo "==> Venv ready."

# Unset empty WANDB_RUN_ID so wandb SDK doesn't read it from env
[[ -z "${WANDB_RUN_ID}" ]] && unset WANDB_RUN_ID

# Phase 2: training (8 tasks, --frozen prevents venv modifications)
srun --kill-on-bad-exit=1 \
     --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 \
     "${SELECT_GPU_SCRIPT}" \
     singularity exec --rocm "${SIF}" \
     bash -c "
         cd ${ISR_REPO} && \
         PYTHONPATH=${SARATRX}/pre-training:\${PYTHONPATH:-} \
         uv run --frozen --extra rocm --extra mtl-yolo \
         python ${SARATRX}/pre-training/train_h5_lumi.py \
             --h5_train  ${H5_TRAIN} \
             --h5_test   ${H5_TEST} \
             ${INIT_CKPT_ARG[@]+--init_ckpt ${INIT_CKPT}} \
             --output_dir ${OUTPUT_DIR} \
             --resume auto \
             --input_size ${INPUT_SIZE} \
             --epochs ${EPOCHS} \
             --batch_size ${BATCH_SIZE} \
             --accum_iter ${ACCUM_ITER} \
             --blr ${BLR} \
             --mask_ratio ${MASK_RATIO} \
             --save_interval ${SAVE_INTERVAL} \
             --val_interval ${VAL_INTERVAL} \
             --wandb_project ${WANDB_PROJECT} \
             ${WANDB_RUN_ID:+--wandb_run_id ${WANDB_RUN_ID}} \
             ${WANDB_RUN_NAME:+--wandb_run_name ${WANDB_RUN_NAME}} \
             --num_workers 4 \
             --pin_mem \
             ${EXTRA_ARGS}
     "

rm -f "${SELECT_GPU_SCRIPT}"
echo "=== Done ==="
