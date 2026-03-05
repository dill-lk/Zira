"""
Text generation for Zira models.

Supports:
  * Greedy decoding
  * Temperature sampling
  * Top-k sampling
  * Top-p (nucleus) sampling

Usage:
    python -m zira.generate \\
        --model micro \\
        --checkpoint checkpoints/ckpt_step0050000.pt \\
        --prompt "Once upon a time" \\
        --max_new_tokens 200 \\
        --temperature 0.8
"""

import argparse
import os

import torch
import torch.nn.functional as F

from .config import get_config
from .model import ZiraModel
from .tokenizer import ZiraTokenizer
from .utils import load_checkpoint

try:
    import torch_xla.core.xla_model as xm  # type: ignore
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out all logits except the top-k."""
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, top_k)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out logits outside the nucleus (cumulative probability ≤ top_p)."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Shift right so the token that first crosses top_p is retained
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    # Scatter back to original order
    logits.scatter_(-1, sorted_idx, sorted_logits)
    return logits


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: ZiraModel,
    tokenizer: ZiraTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device=None,
) -> str:
    """
    Auto-regressive text generation.

    Args:
        model:          ZiraModel in eval mode.
        tokenizer:      ZiraTokenizer.
        prompt:         Text prompt string.
        max_new_tokens: Maximum tokens to generate.
        temperature:    Softmax temperature (1.0 = no change, <1 = sharper).
        top_k:          Keep only top-k logits (0 = disabled).
        top_p:          Nucleus filtering threshold (1.0 = disabled).
        device:         torch.device; inferred from model if None.

    Returns:
        Generated text string (prompt + continuation).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    seq_len = model.config.seq_len

    ids = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Truncate to model's context window
        context = input_ids[:, -seq_len:]

        logits = model(context)                 # (1, ctx, vocab)
        next_logits = logits[:, -1, :]          # (1, vocab)

        if temperature != 1.0:
            next_logits = next_logits / max(temperature, 1e-8)

        next_logits = top_k_filter(next_logits, top_k)
        next_logits = top_p_filter(next_logits, top_p)

        if temperature == 0.0:
            # Greedy
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_id:
            break

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zira text generation")
    parser.add_argument("--model",          default="micro",
                        choices=["micro", "small", "compact", "base", "pro"])
    parser.add_argument("--checkpoint",     required=True,
                        help="Path to a trained checkpoint")
    parser.add_argument("--tokenizer",      default="huggyllama/llama-7b")
    parser.add_argument("--prompt",         default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top_k",          type=int, default=50)
    parser.add_argument("--top_p",          type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config(args.model)

    if HAS_XLA:
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = ZiraModel(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    tokenizer = ZiraTokenizer(hf_name=args.tokenizer)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    print("\n" + "=" * 60)
    print(output)
    print("=" * 60)
