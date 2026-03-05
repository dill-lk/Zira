"""
Utility helpers for Zira training.

Includes:
  * Cosine LR scheduler with linear warmup
  * Checkpoint save / load
  * Throughput (tokens/sec) tracker
  * Google Drive mount helper (Colab)
"""

import os
import time
import math
from typing import Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

class CosineSchedulerWithWarmup:
    """
    Linear warmup → cosine decay to min_lr.

    Usage:
        scheduler = CosineSchedulerWithWarmup(optimizer, ...)
        for step in training_loop:
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-5,
    ):
        self.optimizer   = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps   = max_steps
        self.min_lr      = min_lr
        self._step       = 0
        # Cache base LRs from the initial param groups
        self._base_lrs   = [pg["lr"] for pg in optimizer.param_groups]

    def step(self) -> None:
        self._step += 1
        lrs = self._get_lrs()
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr

    def _get_lrs(self):
        step = self._step
        results = []
        for base_lr in self._base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * (step + 1) / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            results.append(lr)
        return results

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @property
    def current_step(self) -> int:
        return self._step


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    config,
    checkpoint_dir: str,
) -> str:
    """Save training state to disk. Returns path of saved file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_step{step:07d}.pt")
    state = {
        "step":          step,
        "loss":          loss,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler_step": scheduler.current_step,
        "config":        config.__dict__,
    }
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
):
    """
    Load a checkpoint.

    Returns:
        step (int), loss (float)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler_step" in state:
        scheduler._step = state["scheduler_step"]
    step = state.get("step", 0)
    loss = state.get("loss", float("nan"))
    print(f"[Checkpoint] Loaded from {path}  (step={step}, loss={loss:.4f})")
    return step, loss


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path of the latest checkpoint in a directory, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_step") and f.endswith(".pt")]
    )
    if not files:
        return None
    return os.path.join(checkpoint_dir, files[-1])


# ---------------------------------------------------------------------------
# Throughput tracker
# ---------------------------------------------------------------------------

class ThroughputTracker:
    """Track tokens/sec throughput over a sliding window."""

    def __init__(self, window: int = 100):
        self._window = window
        self._times: list = []
        self._tokens: list = []

    def update(self, n_tokens: int) -> None:
        self._times.append(time.time())
        self._tokens.append(n_tokens)
        if len(self._times) > self._window:
            self._times.pop(0)
            self._tokens.pop(0)

    @property
    def tokens_per_sec(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return sum(self._tokens[1:]) / elapsed


# ---------------------------------------------------------------------------
# Google Drive helper (Colab)
# ---------------------------------------------------------------------------

def maybe_mount_drive(base_dir: str = "/content/drive/MyDrive/zira") -> str:
    """
    Attempt to mount Google Drive (no-op outside Colab).
    Returns the resolved output directory.
    """
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        os.makedirs(base_dir, exist_ok=True)
        print(f"[Drive] Checkpoints will be saved to {base_dir}")
        return base_dir
    except ImportError:
        print("[Drive] Not running in Colab – using local checkpoint directory.")
        return "checkpoints"


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def log_step(
    step: int,
    loss: float,
    lr: float,
    tokens_per_sec: float,
    grad_norm: Optional[float] = None,
) -> None:
    gnorm_str = f"  grad_norm={grad_norm:.3f}" if grad_norm is not None else ""
    tqdm.write(
        f"[step {step:7d}]  loss={loss:.4f}  lr={lr:.2e}"
        f"  tok/s={tokens_per_sec:,.0f}{gnorm_str}"
    )
