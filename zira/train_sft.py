"""
Supervised Fine-Tuning (SFT / chat alignment) script for Zira.

Identical TPU infrastructure to train_pretrain.py but uses SFTDataset
and supports loading a pre-trained checkpoint before fine-tuning.

Usage:
    python -m zira.train_sft \\
        --model micro \\
        --pretrain_ckpt checkpoints/ckpt_step0050000.pt \\
        --dataset tatsu-lab/alpaca
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from .config import get_config, ZiraConfig
from .model import ZiraModel
from .tokenizer import ZiraTokenizer
from .dataset import SFTDataset
from .utils import (
    CosineSchedulerWithWarmup,
    ThroughputTracker,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    maybe_mount_drive,
    log_step,
)

try:
    import torch_xla.core.xla_model as xm              # type: ignore
    import torch_xla.distributed.parallel_loader as pl  # type: ignore
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def get_device():
    if HAS_XLA:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_sft(config: ZiraConfig, args: argparse.Namespace) -> None:
    device = get_device()
    is_tpu = HAS_XLA
    is_master = (not is_tpu) or xm.is_master_ordinal()

    if is_master:
        print(f"[train_sft] device={device}  TPU={is_tpu}")

    # ------------------------------------------------------------------ model
    model = ZiraModel(config).to(device)
    if is_tpu:
        model = model.to(torch.bfloat16)

    # ----------------------------------------- load pretrained weights
    if args.pretrain_ckpt and os.path.isfile(args.pretrain_ckpt):
        load_checkpoint(
            args.pretrain_ckpt, model, map_location=str(device)
        )
        if is_master:
            print(f"[train_sft] Loaded pretrained weights from {args.pretrain_ckpt}")
    elif is_master:
        print("[train_sft] No pretrain checkpoint supplied – training from scratch.")

    if is_master:
        print(f"[train_sft] Parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------ tokenizer
    tokenizer = ZiraTokenizer(hf_name=args.tokenizer)

    # -------------------------------------------------------------- dataset
    dataset = SFTDataset(
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        dataset_name=args.dataset,
        split="train",
        cache_dir=config.dataset_cache_dir,
        max_samples=args.max_samples,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size() if is_tpu else 1,
        rank=xm.get_ordinal() if is_tpu else 0,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=not is_tpu,
    )

    # ------------------------------------------------------------ optimizer
    # Lower LR for SFT to avoid catastrophic forgetting
    sft_lr = config.lr * 0.1
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sft_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    sft_steps = args.sft_steps or (config.max_steps // 10)
    scheduler = CosineSchedulerWithWarmup(
        optimizer,
        warmup_steps=max(100, config.warmup_steps // 10),
        max_steps=sft_steps,
        min_lr=sft_lr * 0.1,
    )
    tracker = ThroughputTracker()

    # --------------------------------------------------- checkpoint dir
    ckpt_dir = maybe_mount_drive() if args.use_drive else config.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------------------------- resume SFT ckpt
    start_step = 0
    if args.resume:
        ckpt_path = args.resume if os.path.isfile(args.resume) else find_latest_checkpoint(ckpt_dir)
        if ckpt_path:
            start_step, _ = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, map_location=str(device)
            )

    # ------------------------------------------ wrap with ParallelLoader
    if is_tpu:
        para_loader = pl.MpDeviceLoader(loader, device)
        data_iter = iter(para_loader)
    else:
        data_iter = iter(loader)

    # ---------------------------------------------------------------- loop
    model.train()
    step = start_step
    accum_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(
        total=sft_steps,
        initial=start_step,
        desc="SFT",
        unit="step",
        disable=not is_master,
    )

    while step < sft_steps:
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            sampler.set_epoch(step // max(1, len(loader)))
            if is_tpu:
                data_iter = iter(para_loader)
            else:
                data_iter = iter(loader)
            input_ids, labels = next(data_iter)

        if not is_tpu:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

        if is_tpu:
            loss, _ = model(input_ids, labels)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss, _ = model(input_ids, labels)

        loss = loss / config.grad_accum_steps
        loss.backward()
        accum_loss += loss.item()

        tracker.update(input_ids.numel())

        if (step + 1) % config.grad_accum_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            if is_tpu:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            if is_master:
                pbar.set_postfix(
                    loss=f"{accum_loss * config.grad_accum_steps:.4f}",
                    lr=f"{scheduler.current_lr:.2e}",
                    tok_s=f"{tracker.tokens_per_sec:,.0f}",
                    gnorm=f"{float(grad_norm):.3f}",
                )
                if step % args.log_every == 0:
                    log_step(
                        step=step,
                        loss=accum_loss * config.grad_accum_steps,
                        lr=scheduler.current_lr,
                        tokens_per_sec=tracker.tokens_per_sec,
                        grad_norm=float(grad_norm),
                    )
            accum_loss = 0.0

        if is_master and (step + 1) % config.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step + 1, loss.item(), config, ckpt_dir)

        pbar.update(1)
        step += 1

    pbar.close()
    if is_master:
        save_checkpoint(model, optimizer, scheduler, step, accum_loss, config, ckpt_dir)
        print("[train_sft] SFT complete.")


# ---------------------------------------------------------------------------
# XLA multiprocessing entry point
# ---------------------------------------------------------------------------

def _mp_fn(rank: int, config: ZiraConfig, args: argparse.Namespace) -> None:
    train_sft(config, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zira SFT training")
    parser.add_argument("--model",         default="micro",
                        choices=["micro", "small", "compact", "base", "pro"])
    parser.add_argument("--dataset",       default="tatsu-lab/alpaca",
                        choices=["tatsu-lab/alpaca", "HuggingFaceH4/ultrachat_200k"])
    parser.add_argument("--tokenizer",     default="huggyllama/llama-7b")
    parser.add_argument("--pretrain_ckpt", default="",
                        help="Path to a pre-training checkpoint to initialise from")
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--sft_steps",     type=int, default=None,
                        help="Override total SFT steps (default: max_steps // 10)")
    parser.add_argument("--log_every",     type=int, default=50)
    parser.add_argument("--resume",        default="",
                        help="Path to SFT checkpoint to resume from")
    parser.add_argument("--use_drive",     action="store_true")
    parser.add_argument("--max_samples",   type=int, default=None)
    parser.add_argument("--num_workers",   type=int, default=2)
    parser.add_argument("--tpu_spawn",     action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config(args.model)

    if args.tpu_spawn and HAS_XLA:
        import torch_xla.distributed.xla_multiprocessing as xmp  # type: ignore
        xmp.spawn(_mp_fn, args=(cfg, args))
    else:
        train_sft(cfg, args)
