#!/usr/bin/env python3
"""Train SARATR-X (HiViT MAE) on SAR HDF5 tiles on LUMI.

Reads linear-intensity tiles from train.h5, applies dB-percentile
normalization (per-image bounds from parquet sidecar, or per-tile fallback),
random-crops 224x224 patches, and trains the masked autoencoder with
multi-scale SAR gradient feature targets.

Usage (single GPU):
    python train_h5_lumi.py --h5_train /path/to/train.h5 --output_dir ./out

Usage (LUMI 8-GPU via srun):
    See train_saratrx_lumi.sh
"""

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# ---------------------------------------------------------------------------
# Compat: torch._six was removed in PyTorch 2.0+
# ---------------------------------------------------------------------------
if not hasattr(torch, "_six"):
    import types

    _mod = types.ModuleType("torch._six")
    _mod.inf = float("inf")
    torch._six = _mod
    sys.modules["torch._six"] = _mod

# SARATR-X imports (PYTHONPATH must include pre-training/)
import models  # noqa: E402
import util.lr_sched as lr_sched  # noqa: E402
import util.misc as misc  # noqa: E402
from util.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa: E402

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_db(tile: np.ndarray, lo: float, hi: float,
                 eps: float = 1e-7) -> np.ndarray:
    """Linear intensity -> dB -> scale with pre-computed bounds -> [0, 1]."""
    clamped = np.maximum(tile, eps)
    db = 10.0 * np.log10(clamped)
    return np.clip((db - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


def normalize_db_percentile(tile: np.ndarray, eps: float = 1e-7,
                            p_low: float = 1.0,
                            p_high: float = 99.0) -> np.ndarray:
    """Fallback: compute dB percentile bounds from the tile itself."""
    I = np.maximum(tile.squeeze().astype(np.float32), eps)
    I_dB = 10.0 * np.log10(I)
    lo, hi = np.percentile(I_dB, [p_low, p_high])
    return np.clip((I_dB - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class H5TileDataset(torch.utils.data.Dataset):
    """HDF5 SAR tile dataset with dB-percentile normalisation.

    Expects an HDF5 file with at least ``images`` of shape ``(N, H, W, C)``
    containing **linear intensity** float32 values.  An optional parquet
    sidecar provides pre-computed per-image dB ``lo``/``hi`` bounds; without
    it the percentiles are computed per-tile on the fly.
    """

    def __init__(self, h5_path: str, parquet_path: str | None = None,
                 crop_size: int = 224, flip: bool = True):
        self.h5_path = str(h5_path)
        self.crop_size = crop_size
        self.flip = flip
        self._h5 = None

        with h5py.File(self.h5_path, "r") as f:
            self.length = f["images"].shape[0]
            self.tile_h = f["images"].shape[1]
            self.tile_w = f["images"].shape[2]

        self.norm_bounds: dict[int, tuple[float, float]] | None = None
        if parquet_path and _HAS_PANDAS and Path(parquet_path).exists():
            self._load_parquet_bounds(parquet_path)

    # -- parquet helpers --------------------------------------------------

    def _load_parquet_bounds(self, path: str) -> None:
        df = pd.read_parquet(path)
        bounds: dict[int, tuple[float, float]] = {}
        for _, row in df.iterrows():
            idx = row.get("h5_index", None)
            if idx is None:
                continue
            norm = row.get("norm", None)
            if norm is None or not isinstance(norm, dict):
                continue
            lo_raw = norm.get("lo")
            hi_raw = norm.get("hi")
            if lo_raw is None or hi_raw is None:
                continue
            lo = lo_raw[0] if isinstance(lo_raw, (list, np.ndarray)) else float(lo_raw)
            hi = hi_raw[0] if isinstance(hi_raw, (list, np.ndarray)) else float(hi_raw)
            bounds[int(idx)] = (float(lo), float(hi))
        # #region agent log
        import json as _json
        _dbg = {"sessionId": "2709b8", "hypothesisId": "C", "location": "train_h5_lumi.py:_load_parquet_bounds", "message": "parquet load complete", "timestamp": int(time.time() * 1000), "data": {"path": path, "total_rows": len(df), "bounds_found": len(bounds)}}
        with open("/users/eacar/projects/.cursor/debug-2709b8.log", "a") as _f: _f.write(_json.dumps(_dbg) + "\n")
        # #endregion
        if bounds:
            self.norm_bounds = bounds
            print(f"  Loaded {len(bounds)} norm-bound entries from parquet")

    # -- h5 handle (lazy, per-worker) -------------------------------------

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # -- dataset API ------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        tile = self.h5["images"][idx]           # (H, W, C) float32
        tile = tile.squeeze().astype(np.float32)  # (H, W)

        if self.norm_bounds and idx in self.norm_bounds:
            lo, hi = self.norm_bounds[idx]
            tile = normalize_db(tile, lo, hi)
        else:
            tile = normalize_db_percentile(tile)

        h, w = tile.shape[:2]
        cs = self.crop_size
        if h >= cs and w >= cs:
            y0 = np.random.randint(0, h - cs + 1)
            x0 = np.random.randint(0, w - cs + 1)
            tile = tile[y0 : y0 + cs, x0 : x0 + cs]
        else:
            pad_h = max(0, cs - h)
            pad_w = max(0, cs - w)
            tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode="reflect")
            tile = tile[:cs, :cs]

        if self.flip and np.random.random() > 0.5:
            tile = np.flip(tile, axis=1).copy()

        return torch.from_numpy(tile).unsqueeze(0).float(), 0

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Validation visualizer
# ---------------------------------------------------------------------------

class ValidationVisualizer:
    """Periodic visualisation of encoder / decoder behaviour on test tiles."""

    def __init__(self, h5_path: str, parquet_path: str | None = None,
                 num_tiles: int = 10, crop_size: int = 224):
        self.h5_path = str(h5_path)
        self.crop_size = crop_size
        self.tile_indices: list[int] = []
        self.norm_bounds: dict[int, tuple[float, float]] = {}

        with h5py.File(self.h5_path, "r") as f:
            has_labels = f["has_labels"][:]
        labeled = np.where(has_labels == 1)[0]
        self.tile_indices = labeled[:num_tiles].tolist()

        if parquet_path and _HAS_PANDAS and Path(parquet_path).exists():
            df = pd.read_parquet(parquet_path)
            for _, row in df.iterrows():
                idx = row.get("h5_index")
                norm = row.get("norm")
                if idx is None or norm is None or not isinstance(norm, dict):
                    continue
                lo_raw, hi_raw = norm.get("lo"), norm.get("hi")
                if lo_raw is None or hi_raw is None:
                    continue
                lo = lo_raw[0] if isinstance(lo_raw, (list, np.ndarray)) else float(lo_raw)
                hi = hi_raw[0] if isinstance(hi_raw, (list, np.ndarray)) else float(hi_raw)
                self.norm_bounds[int(idx)] = (float(lo), float(hi))

    @torch.no_grad()
    def generate(self, model, epoch: int, output_dir: str, device) -> dict:
        if not _HAS_MPL:
            return {}

        model_raw = model.module if hasattr(model, "module") else model
        model.eval()
        viz_dir = Path(output_dir) / "viz" / f"epoch_{epoch}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        wb_imgs: dict = {}

        with h5py.File(self.h5_path, "r") as f:
            for tidx in self.tile_indices:
                tile = f["images"][tidx].squeeze().astype(np.float32)

                if tidx in self.norm_bounds:
                    lo, hi = self.norm_bounds[tidx]
                    tile_n = normalize_db(tile, lo, hi)
                else:
                    tile_n = normalize_db_percentile(tile)

                patch = _center_crop(tile_n, self.crop_size)
                x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                x3 = torch.cat([x, x, x], dim=1)

                # full reconstruction (mask_ratio=0)
                lat_f, _, idr_f = model_raw.forward_encoder(x3, mask_ratio=0.0)
                _, pred_f = model_raw.forward_decoder(lat_f, idr_f)

                # masked reconstruction (mask_ratio=0.75)
                lat_m, mask_m, idr_m = model_raw.forward_encoder(x3, mask_ratio=0.75)
                _, pred_m = model_raw.forward_decoder(lat_m, idr_m)

                # take first 256 dims = first GF scale, unpatchify to (1,1,H,W)
                recon_f = model_raw.unpatchify(pred_f[:, :, :256]).squeeze().cpu().numpy()
                recon_m = model_raw.unpatchify(pred_m[:, :, :256]).squeeze().cpu().numpy()

                grid = self.crop_size // 16
                mask_2d = mask_m.squeeze().cpu().numpy().reshape(grid, grid)
                mask_up = np.repeat(np.repeat(mask_2d, 16, axis=0), 16, axis=1)
                masked_in = patch.copy()
                masked_in[mask_up > 0.5] = 0.5

                _norm01 = lambda a: (a - a.min()) / (a.max() - a.min() + 1e-8)

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(patch, cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("Input")
                axes[1].imshow(masked_in, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Masked (75 %)")
                axes[2].imshow(_norm01(recon_f), cmap="gray")
                axes[2].set_title("GF recon (full)")
                axes[3].imshow(_norm01(recon_m), cmap="gray")
                axes[3].set_title("GF recon (masked)")
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.suptitle(f"Tile {tidx} — epoch {epoch}", fontweight="bold")
                plt.tight_layout()

                out_path = viz_dir / f"tile_{tidx}.png"
                fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
                plt.close(fig)

                if _HAS_WANDB:
                    wb_imgs[f"val/tile_{tidx}"] = wandb.Image(str(out_path))

        model.train()
        return wb_imgs


def _center_crop(img: np.ndarray, cs: int) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = max(0, (h - cs) // 2)
    x0 = max(0, (w - cs) // 2)
    return img[y0 : y0 + cs, x0 : x0 + cs]


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(model, data_loader, optimizer, device, epoch,
                    loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for step, (samples, _) in enumerate(
            metric_logger.log_every(data_loader, 20, header)):
        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(args, epoch, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    payload = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
    }
    misc.save_on_master(payload, output_dir / "checkpoint-latest.pth")
    if epoch % args.save_interval == 0 or epoch == args.epochs:
        misc.save_on_master(payload, output_dir / f"checkpoint-{epoch}.pth")


def find_latest_checkpoint(output_dir: str) -> str | None:
    p = Path(output_dir) / "checkpoint-latest.pth"
    return str(p) if p.exists() else None


# ---------------------------------------------------------------------------
# SLURM → DDP env-var bridge
# ---------------------------------------------------------------------------

def _setup_slurm_ddp_env() -> None:
    """Populate MASTER_ADDR / MASTER_PORT / WORLD_SIZE from SLURM vars."""
    if "SLURM_PROCID" not in os.environ:
        return
    if "MASTER_ADDR" not in os.environ:
        node_list = os.environ.get("SLURM_NODELIST", "localhost")
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", node_list],
                capture_output=True, text=True, check=True)
            os.environ["MASTER_ADDR"] = result.stdout.strip().split("\n")[0]
        except Exception:
            os.environ["MASTER_ADDR"] = "localhost"
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE",
                          os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
    # LOCAL_RANK=0 because select_gpu sets ROCR_VISIBLE_DEVICES=$SLURM_LOCALID,
    # making each process see only its own GPU as device 0.
    os.environ.setdefault("LOCAL_RANK", "0")

    # #region agent log
    import json as _json
    _dbg = {"sessionId": "2709b8", "hypothesisId": "A,B", "location": "train_h5_lumi.py:_setup_slurm_ddp_env", "message": "DDP env vars after setup", "timestamp": int(time.time() * 1000), "data": {"RANK": os.environ.get("RANK"), "LOCAL_RANK": os.environ.get("LOCAL_RANK"), "WORLD_SIZE": os.environ.get("WORLD_SIZE"), "MASTER_ADDR": os.environ.get("MASTER_ADDR"), "MASTER_PORT": os.environ.get("MASTER_PORT"), "SLURM_PROCID": os.environ.get("SLURM_PROCID"), "SLURM_LOCALID": os.environ.get("SLURM_LOCALID"), "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"), "cuda_device_count": torch.cuda.device_count()}}
    with open("/users/eacar/projects/.cursor/debug-2709b8.log", "a") as _f: _f.write(_json.dumps(_dbg) + "\n")
    # #endregion


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args_parser():
    p = argparse.ArgumentParser("SARATR-X H5 pre-training", add_help=False)

    # data
    p.add_argument("--h5_train", type=str, required=True)
    p.add_argument("--h5_test", type=str, default="")
    p.add_argument("--parquet_train", type=str, default="")
    p.add_argument("--parquet_test", type=str, default="")

    # model
    p.add_argument("--model", default="mae_hivit_base_dec512d6b", type=str)
    p.add_argument("--input_size", default=224, type=int)
    p.add_argument("--mask_ratio", default=0.75, type=float)
    p.add_argument("--norm_pix_loss", action="store_true")
    p.set_defaults(norm_pix_loss=False)

    # checkpointing
    p.add_argument("--init_ckpt", type=str, default="",
                   help="ImageNet-pretrained HiViT weights (strict=False)")
    p.add_argument("--resume", type=str, default="",
                   help='Checkpoint path or "auto" (find latest in output_dir)')
    p.add_argument("--output_dir", default="./output_dir", type=str)
    p.add_argument("--save_interval", default=50, type=int)
    p.add_argument("--val_interval", default=50, type=int)

    # optimiser
    p.add_argument("--batch_size", default=64, type=int,
                   help="Per-GPU batch size")
    p.add_argument("--epochs", default=800, type=int)
    p.add_argument("--accum_iter", default=1, type=int)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--blr", type=float, default=1.5e-4,
                   help="Base LR: actual_lr = blr * eff_batch / 256")
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=5)

    # logging
    p.add_argument("--wandb_project", type=str, default="saratrx-pretrain")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_run_id", type=str, default="",
                   help="W&B run ID to resume (appends to existing run)")
    p.add_argument("--log_dir", default=None, type=str)

    # system
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--pin_mem", action="store_true")
    p.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    p.set_defaults(pin_mem=True)
    p.add_argument("--start_epoch", default=0, type=int)

    # distributed
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_on_itp", action="store_true")
    p.add_argument("--dist_url", default="env://", type=str)

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # -- resolve parquet defaults -----------------------------------------
    if not args.parquet_train:
        pq = Path(args.h5_train).with_suffix(".parquet")
        if pq.exists():
            args.parquet_train = str(pq)
    if not args.parquet_test and args.h5_test:
        pq = Path(args.h5_test).with_suffix(".parquet")
        if pq.exists():
            args.parquet_test = str(pq)

    # -- DDP env vars from SLURM ------------------------------------------
    _setup_slurm_ddp_env()
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    if args.log_dir is None:
        args.log_dir = args.output_dir

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # -- dataset ----------------------------------------------------------
    dataset_train = H5TileDataset(
        args.h5_train,
        parquet_path=args.parquet_train or None,
        crop_size=args.input_size,
        flip=True,
    )
    print(f"Training dataset: {len(dataset_train)} tiles")

    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=misc.get_world_size(),
            rank=global_rank, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem,
        drop_last=True)

    # -- validation -------------------------------------------------------
    val_viz = None
    if args.h5_test and misc.is_main_process():
        val_viz = ValidationVisualizer(
            args.h5_test,
            parquet_path=args.parquet_test or None,
            num_tiles=10, crop_size=args.input_size)
        print(f"Validation tiles: {val_viz.tile_indices}")

    # -- model ------------------------------------------------------------
    model = models.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    # optional init from ImageNet-pretrained checkpoint
    resolved_resume = args.resume
    if resolved_resume == "auto":
        resolved_resume = find_latest_checkpoint(args.output_dir) or ""

    if args.init_ckpt and not resolved_resume:
        ckpt_path = Path(args.init_ckpt)
        if ckpt_path.exists():
            print(f"Loading init checkpoint: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = ckpt["model"]
            msg = model.load_state_dict(ckpt, strict=False)
            print(f"  Init load result: {msg}")
        else:
            print(f"WARNING: init_ckpt not found at {ckpt_path}, "
                  "training from scratch")

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"effective batch size: {eff_batch_size}")

    # timm compat: add_weight_decay was renamed in newer versions
    try:
        from timm.optim.optim_factory import add_weight_decay
        param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    except ImportError:
        from timm.optim import create_optimizer_v2  # noqa: F401
        param_groups = [
            {"params": [p for n, p in model_without_ddp.named_parameters()
                        if p.requires_grad and "bias" not in n
                        and "norm" not in n],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model_without_ddp.named_parameters()
                        if p.requires_grad and ("bias" in n or "norm" in n)],
             "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # -- resume -----------------------------------------------------------
    if resolved_resume:
        args.resume = resolved_resume
        print(f"Resuming from: {args.resume}")
        misc.load_model(args=args, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler)

    # -- wandb ------------------------------------------------------------
    wb_run = None
    if _HAS_WANDB and misc.is_main_process():
        run_name = (args.wandb_run_name
                    or f"saratrx_{args.model}_ep{args.epochs}")
        wb_kwargs = dict(project=args.wandb_project, name=run_name,
                         config=vars(args))
        if args.wandb_run_id:
            wb_kwargs.update(id=args.wandb_run_id, resume="must")
        else:
            wb_kwargs["resume"] = "allow"
        wb_run = wandb.init(**wb_kwargs)

    # -- validation on resume (verify viz works before burning GPU hours) --
    if resolved_resume and val_viz is not None and args.start_epoch > 0:
        print(f"Running validation before resuming training (epoch {args.start_epoch})...")
        wb_imgs = val_viz.generate(model, args.start_epoch, args.output_dir, device)
        if wb_run and wb_imgs:
            wb_run.log(wb_imgs, step=args.start_epoch)
        print("Validation OK — continuing training.")

    # -- train loop -------------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Start training for {args.epochs} epochs "
          f"(from epoch {args.start_epoch})")
    t0 = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        ep1 = epoch + 1  # 1-based for logging / checkpoints
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        stats = train_one_epoch(model, data_loader_train, optimizer,
                                device, epoch, loss_scaler, args=args)

        save_checkpoint(args, ep1, model_without_ddp, optimizer, loss_scaler)

        if wb_run is not None:
            wb_run.log({"train/loss": stats["loss"],
                        "train/lr": stats["lr"],
                        "epoch": ep1})

        if (val_viz is not None
                and (ep1 % args.val_interval == 0 or ep1 == args.epochs)):
            wb_imgs = val_viz.generate(model, ep1, args.output_dir, device)
            if wb_run and wb_imgs:
                wb_run.log(wb_imgs, step=ep1)

        log_entry = {**{f"train_{k}": v for k, v in stats.items()},
                     "epoch": ep1}
        if misc.is_main_process():
            with open(Path(args.output_dir) / "log.txt", "a") as fh:
                fh.write(json.dumps(log_entry) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"Training time {elapsed}")

    if wb_run:
        wb_run.finish()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
