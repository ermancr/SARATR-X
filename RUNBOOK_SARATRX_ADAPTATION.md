# SARATR-X Adaptation Experiments — LUMI Runbook

Run these experiments **in the order shown**. Phase 0 establishes the pure-YOLO baseline. Phase 1 (no re-pretraining) and Phase 2 (improved encoder) can be submitted in parallel. Phase 3 combines the best encoder with Phase 1 adaptation.

**Linear Parent Issue:** [FM-24 MAE (SARATR-X)](https://linear.app/iceye/issue/FM-24/mae-saratr-x) (under the *Modelling* project, team `foundationModelDevelopment`)

**W&B Project:** `snow_owl`

**Shared training stack across all YOLO experiments (Phase 0, 1, 3):** All use byte-identical training configuration — same model (`v9-m-obb`), same hyperparameters (40 epochs, 768x768, batch_size=32, accum=2, SGD lr=0.01, cosine schedule), same loss (`BCELoss=1.2, BoxLoss=7.0, DFLoss=2.0, AngleLoss=1.0, aux=0.25, matcher topk=6`), same train/val/test data (`air_land_maritime_best_20260511_003_resampled_05/train.h5` and `air_land_maritime_test_20260513_001_resampled_05/test.h5`), and same TTA. The **only** thing that differs between experiments is the `saratrx:` block in the config.

---

## Prerequisites

```bash
# Verify repos are up to date
cd ~/projects/isr-automatic-target-recognition && git fetch --all
cd ~/projects/SARATR-X && git fetch --all

# Verify YOLO training data (used by Phase 0, 1, and 3 — same train/val/test for all)
ls /scratch/project_462001182/snow_owl/data/datasets/air_land_maritime_best_20260511_003_resampled_05/train.h5
ls /scratch/project_462001182/snow_owl/data/datasets/air_land_maritime_test_20260513_001_resampled_05/test.h5

# Verify YOLO pretrained init weights (used by Phase 0, 1, and 3)
ls /scratch/project_462001182/snow_owl/data/models/yolo/pretrained_weights/v9-m.pt

# Verify SaRaTrX existing pretrained checkpoint (used by Phase 1)
ls /scratch/project_462001182/snow_owl/experiments/saratrx_pretrain/checkpoint-800.pth

# Verify SaRaTrX pretraining data (used by Phase 2 — same as YOLO data)
ls /scratch/project_462001182/snow_owl/data/datasets/air_land_maritime_best_20260511_003_resampled_05/train.h5
ls /scratch/project_462001182/snow_owl/data/datasets/air_land_maritime_test_20260513_001_resampled_05/test.h5

# Verify containers
ls /scratch/project_462001182/snow_owl/containers/uv_wrappers/mtl-yolo-rocm-macar-dev/bin/mtl-yolo
ls /scratch/project_462001182/snow_owl/containers/singularity/atr-base.sif
```

---

## Phase 0 — Pure YOLO Baseline (No SaRaTrX)

This is the **primary baseline** for measuring the contribution of SaRaTrX features. All Phase 1+ SaRaTrX experiments use the exact same model, hyperparameters, and dataset; only the `saratrx:` block in the config differs. Any mAP delta vs. this baseline therefore comes purely from SaRaTrX features and adaptation modules.

> **Status: Already completed.** Existing W&B run: [iceye/snow_owl/yqibzis6](https://wandb.ai/iceye/snow_owl/runs/yqibzis6) (run name `v9-m-obb_alm-saratrx-baseline_2026-05-17_08-41-52_j18680221`, SLURM job `j18680221`). Use this run as the reference baseline for all Phase 1 / Phase 3 comparisons. Do **not** re-submit unless you intentionally want a fresh baseline.

**Branch:** `fm-7-c3-feature-projection-heads-se-attention` (any branch with the saratrx code works — the config drives behavior; with `saratrx:` block omitted, the SaratrxYOLO wrapper is bypassed and a plain YOLO is built).
**Config:** `configs/mtl_yolo/lumi/train_alm_saratrx_onlyYolo.yaml`
**W&B experiment name:** `alm-saratrx-baseline`
**Expected duration:** ~24h (40 epochs)

```bash
# Only re-run if you want a fresh baseline (the existing run yqibzis6 already
# covers this configuration).
cd ~/projects/isr-automatic-target-recognition
git checkout fm-7-c3-feature-projection-heads-se-attention

sbatch infra/lumi/train/train_lumi_wrapper_saratrx.sh \
    configs/mtl_yolo/lumi/train_alm_saratrx_onlyYolo.yaml
```

**What's enabled:** Plain YOLOv9-m-obb with no HiViT injection. The `saratrx:` block is omitted from the config, so `SaratrxYOLO` is not instantiated.

**Monitor:** W&B experiment `alm-saratrx-baseline` in project `snow_owl`. Reference run: [yqibzis6](https://wandb.ai/iceye/snow_owl/runs/yqibzis6).

---

## Phase 1 — YOLO + SaRaTrX Adaptive Features (No Re-pretraining)

These experiments add SaRaTrX HiViT feature injection on top of the Phase 0 baseline. They share the same training config as Phase 0 (`train_alm_saratrx_injected.yaml` is byte-identical to `train_alm_saratrx_onlyYolo.yaml` except for the `saratrx:` block). All use the existing 224px `checkpoint-800.pth` SaRaTrX checkpoint. Submit in parallel with Phase 0.

### Experiment 1C: Frozen SaRaTrX Features — raw injection

**Branch:** `fm-1-train-mtl-yolo-with-sar-atrx-representation` (ISR repo)
**Expected duration:** ~24h (40 epochs)

```bash
cd ~/projects/isr-automatic-target-recognition
git checkout fm-1-train-mtl-yolo-with-sar-atrx-representation

sbatch infra/lumi/train/train_lumi_wrapper_saratrx.sh \
    configs/mtl_yolo/lumi/train_alm_saratrx_1c_frozen.yaml
```

**What's enabled:** Raw frozen HiViT features injected via widened AConv layers. No projection, no LoRA. Measures the contribution of HiViT features alone (vs. Phase 0).

**Monitor:** W&B experiment `alm-saratrx-1c-frozen`

### Experiment 1B [C3]: + Feature Projection + SE Attention (FM-7 ablation)

**Branch:** `fm-7-c3-feature-projection-heads-se-attention` (ISR repo)
**Linear:** FM-7

```bash
cd ~/projects/isr-automatic-target-recognition
git checkout fm-7-c3-feature-projection-heads-se-attention

sbatch infra/lumi/train/train_lumi_wrapper_saratrx.sh \
    configs/mtl_yolo/lumi/train_alm_saratrx_1b_projection_se.yaml
```

**What's enabled:** Projection heads (Conv1x1+BN+SiLU+SE) only — no LoRA. Measures channel-adaptation contribution.

**Monitor:** W&B experiment `alm-saratrx-1b-projection-se`

### Experiment 1A [C2+C3]: + Feature Projection + SE + LoRA (FM-6 + FM-7, full Phase 1 stack)

**Branch:** `fm-6-c2-lora-adapters-on-hivit-attention` (ISR repo)
**Linear:** FM-6, FM-7
**Expected duration:** ~24h (40 epochs)

```bash
cd ~/projects/isr-automatic-target-recognition
git checkout fm-6-c2-lora-adapters-on-hivit-attention

sbatch infra/lumi/train/train_lumi_wrapper_saratrx.sh \
    configs/mtl_yolo/lumi/train_alm_saratrx_1a_projection_se_lora.yaml
```

**What's enabled:** Projection heads (Conv1x1+BN+SiLU+SE) + LoRA rank-4 on last 4 HiViT attention blocks.

**Monitor:** W&B experiment `alm-saratrx-1a-projection-se-lora`

---

## Phase 2 — Improved SaRaTrX Pretraining

These improve the encoder itself. Submit after Phase 1 starts (they run independently). Each experiment isolates one pretraining improvement for proper ablation.

All experiments use `train_saratrx_lumi.sh` as the base launcher with env-var overrides for `OUTPUT_DIR` and experiment-specific flags.

### Experiment 2A [B]: 768px Resolution Only (FM-3)

**Branch:** `fm-3-b-resolution-matched-pretraining-768x768` (SARATR-X repo)
**Linear:** [FM-3](https://linear.app/iceye/issue/FM-3)
**Expected duration:** 3-4 days (200 epochs at 768px)

```bash
cd ~/projects/SARATR-X && git checkout fm-3-b-resolution-matched-pretraining-768x768 && \
OUTPUT_DIR=/scratch/project_462001182/snow_owl/experiments/saratrx_pretrain_2a_768_baseline \
sbatch pre-training/train_saratrx_lumi.sh
```

**What's enabled:** 768px resolution pretraining from checkpoint-800.pth. No OAM, no MSL. Measures resolution impact alone.
**Monitor:** W&B project `saratrx-pretrain`

### Experiment 2B [B+D3]: 768px + Object-Aware Masking (FM-3 + FM-10)

**Branch:** `fm-10-d3-object-aware-masking` (SARATR-X repo)
**Linear:** [FM-3](https://linear.app/iceye/issue/FM-3), [FM-10](https://linear.app/iceye/issue/FM-10)
**Expected duration:** 3-4 days (200 epochs at 768px)

```bash
cd ~/projects/SARATR-X && git checkout fm-10-d3-object-aware-masking && \
OUTPUT_DIR=/scratch/project_462001182/snow_owl/experiments/saratrx_pretrain_2b_768_oam \
sbatch pre-training/train_saratrx_lumi.sh
```

**What's enabled:** 768px + saliency-biased masking (objects masked more aggressively). Measures OAM contribution on top of 768px.
**Monitor:** W&B project `saratrx-pretrain`

### Experiment 2C [B+D4]: 768px + Multi-Scale Loss (FM-3 + FM-11)

**Branch:** `fm-11-d4-multi-scale-decoder-loss` (SARATR-X repo)
**Linear:** [FM-3](https://linear.app/iceye/issue/FM-3), [FM-11](https://linear.app/iceye/issue/FM-11)
**Expected duration:** 3-4 days (200 epochs at 768px)

```bash
cd ~/projects/SARATR-X && git checkout fm-11-d4-multi-scale-decoder-loss && \
OUTPUT_DIR=/scratch/project_462001182/snow_owl/experiments/saratrx_pretrain_2c_768_msl \
sbatch pre-training/train_saratrx_lumi.sh
```

**What's enabled:** 768px + per-scale decoder heads (each HiViT stage independently supervised). Measures MSL contribution on top of 768px.
**Monitor:** W&B project `saratrx-pretrain`

---

## Phase 3 — YOLO Training with Improved Encoder

After Phase 2 completes, re-run Phase 1 experiments but pointing to the new checkpoint.

### Experiment 3A [best-2x+C2+C3]: Best Encoder + Projection + LoRA

Pick the best checkpoint from Phase 2 experiments (2A, 2B, or 2C), then update the checkpoint path in the config.

```bash
cd ~/projects/isr-automatic-target-recognition
git checkout fm-6-c2-lora-adapters-on-hivit-attention

# Edit the checkpoint path to point to the best Phase 2 result:
# configs/mtl_yolo/lumi/train_alm_saratrx_3a_768full_proj_lora.yaml
#   saratrx.checkpoint: /scratch/.../saratrx_pretrain_2X_768_.../checkpoint-200.pth

sbatch infra/lumi/train/train_lumi_wrapper_saratrx.sh \
    configs/mtl_yolo/lumi/train_alm_saratrx_3a_768full_proj_lora.yaml
```

---

## Results Comparison Matrix

All YOLO experiments (Phase 0, 1, 3) use **identical** training data, model, and hyperparameters; they differ only in the `saratrx:` block of the config. Compare against Phase 0 to measure the contribution of SaRaTrX features.

| Experiment | Code | SaRaTrX Encoder | Projection | LoRA | Pretraining Modifier | Expected Contribution |
|---|---|---|---|---|---|---|
| **0 (primary baseline)** [done: [yqibzis6](https://wandb.ai/iceye/snow_owl/runs/yqibzis6)] | — | — (no injection) | — | — | — | YOLO-only baseline mAP |
| 1C | — | Frozen 224px | No | No | 224px MAE | +raw foundation features |
| 1B | C3 | Frozen 224px | SE+Proj | No | 224px MAE | +channel adaptation |
| 1A | C2+C3 | Frozen 224px | SE+Proj | Yes | 224px MAE | +attention adaptation |
| 2A | B | (encoder pretraining only) | — | — | 768px only | Resolution ablation |
| 2B | B+D3 | (encoder pretraining only) | — | — | 768px+OAM | +object-aware masking |
| 2C | B+D4 | (encoder pretraining only) | — | — | 768px+MSL | +multi-scale loss |
| 3A | best-2x+C2+C3 | Frozen 768px (best) | SE+Proj | Yes | best Phase 2 | Best expected mAP |

**Reading the matrix:** rows show YOLO-side experiments (0, 1*, 3A) and pretraining-side experiments (2*). Phase 2 outputs feed into Phase 3 via the `saratrx.checkpoint` path.

---

## Monitoring & Success Criteria

### W&B Dashboards

- **YOLO experiments:** W&B project `snow_owl`, filter by experiment name prefix `alm-saratrx-*`
- **Pretraining:** W&B project `saratrx-pretrain`

### Key Metrics to Compare

| Metric | Where | What to Look For |
|---|---|---|
| `val_global/h_f1` | YOLO training | Main metric — hierarchical F1 |
| `val_global/mAP_50` | YOLO training | Detection quality |
| `train/loss` | Pretraining | Should converge (not diverge) |
| Per-class AP | YOLO final test | Improvement on hard classes |

### Success Thresholds

- Phase 0: Establishes the absolute mAP/F1 baseline (pure YOLO). **Completed**: [yqibzis6](https://wandb.ai/iceye/snow_owl/runs/yqibzis6).
- Phase 1: Any improvement over Phase 0 demonstrates SaRaTrX feature contribution. Compare 1C → 1B → 1A to isolate the effect of projection and LoRA.
- Phase 2: Pretraining loss should decrease and stabilize at 768x768.
- Phase 3: Target 2-5% mAP improvement over Phase 0.

---

## Troubleshooting

### OOM at 768px pretraining

Reduce `BATCH_SIZE` from 6 to 4 and increase `ACCUM_ITER` from 10 to 15:

```bash
BATCH_SIZE=4 ACCUM_ITER=15 sbatch pre-training/train_saratrx_lumi.sh
```

### LoRA destabilizes training

Reduce LoRA rank or increase warmup. Edit config:

```yaml
saratrx:
  lora_rank: 2          # was 4
  lora_alpha: 0.5       # was 1.0
```

### Pretraining loss diverges at 768px

Lower the base LR further:

```bash
BLR=1e-5 sbatch pre-training/train_saratrx_lumi.sh
```

### W&B offline (network issues on LUMI)

The wrapper scripts auto-detect and sync offline runs post-training. If sync fails:

```bash
wandb sync /scratch/project_462001182/snow_owl/logs/runs/<run_dir>/wandb/wandb/offline-run-*
```

---

## Cleanup After Experiments

```bash
# Keep only best checkpoints, remove optimizer states to save space
cd /scratch/project_462001182/snow_owl/experiments/saratrx_pretrain_768_full
rm checkpoint-latest.pth  # 1.2GB (has optimizer state)
# Keep checkpoint-{25,50,...,200}.pth for the best performing epoch
```
