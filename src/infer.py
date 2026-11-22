"""Interactive autoregressive inference for one tutorial model run.

Run from the repository root:

    python -m src.infer models/shakespeare_demo

This script intentionally recomputes the current context on every generated
token.  That is slower than production inference with a KV cache, but it makes
the Part IV.14 loop obvious: predict, choose one token, append it, repeat.
Weights remain fixed during generation; only the temporary context grows, up
to the configured window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from .model import GPT, ModelConfig, model_summary


def color_label(label: str, color_code: str) -> str:
    """Color a terminal prompt label; this is display plumbing, not model logic."""

    if not sys.stdout.isatty() or "NO_COLOR" in os.environ:
        return label
    return f"\033[{color_code}m{label}\033[0m"


def parse_args() -> argparse.Namespace:
    """Define generation controls for sampling, token limit, and device."""

    parser = argparse.ArgumentParser(description="Chat with a tutorial Transformer checkpoint.")
    parser.add_argument("model_dir", help="A folder created by src.train under models/.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Use auto to select CUDA when available.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    """Select where inference tensors run and validate explicit CUDA requests."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def synchronize_if_needed(device: torch.device) -> None:
    """Wait for queued GPU work so Part IV.14 timing metrics are meaningful."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_with_metrics(
    model: GPT,
    prompt_ids: list[int],
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[list[int], dict[str, float | int]]:
    """Generate one answer and measure the user-visible timing quantities.

    ``first_token_ms`` is measured from the beginning of the model call until
    the first token has been selected.  It includes prompt processing, often
    called prefill.  ``output_tokens_per_second`` describes the generated
    decode loop.  This simple implementation does not cache K/V activations,
    so it is purposefully a teaching baseline rather than a serving engine.
    This illustrates Part IV.14's autoregressive loop and finite context while
    leaving KV-cache mechanics for separate study.
    """

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    initial_prompt_length = len(prompt_ids)
    end_id = tokenizer.token_to_id("<|endoftext|>")
    generated = 0
    first_token_ms = None

    synchronize_if_needed(device)
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        # Keep only the last context window.  This illustrates the finite
        # inference context from Part IV.4; old tokens outside the window are
        # no longer directly visible to attention.
        context = token_ids[:, -model.config.block_size :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :] / temperature

        # Top-k is a small, easy-to-see decoding choice.  The model creates a
        # distribution; this inference procedure decides how to select a token.
        if top_k > 0 and top_k < next_logits.size(-1):
            values, _ = torch.topk(next_logits, top_k)
            cutoff = values[:, [-1]]
            next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
        probabilities = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        token_ids = torch.cat((token_ids, next_token), dim=1)
        generated += 1

        synchronize_if_needed(device)
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - started) * 1000.0
        if end_id is not None and int(next_token.item()) == end_id:
            break

    synchronize_if_needed(device)
    total_seconds = time.perf_counter() - started
    output_ids = token_ids[0, initial_prompt_length:].tolist()
    metrics = {
        "input_tokens": initial_prompt_length,
        "output_tokens": generated,
        "first_token_ms": float(first_token_ms or 0.0),
        "total_seconds": total_seconds,
        "output_tokens_per_second": generated / total_seconds if total_seconds > 0 else 0.0,
    }
    return output_ids, metrics


def main() -> None:
    """Rebuild a saved model and run interactive fixed-parameter generation.

    The checkpoint supplies learned weights, tokenizer, and architecture;
    prompt text is encoded and passed to the Part IV.14 generation loop.
    """

    args = parse_args()
    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    weights_path = model_dir / "model.pt"
    for path in (config_path, tokenizer_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")

    device = choose_device(args.device)
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config = ModelConfig(
        vocab_size=int(config_data["vocab_size"]),
        block_size=int(config_data["block_size"]),
        n_embd=int(config_data["n_embd"]),
        n_head=int(config_data["n_head"]),
        n_layer=int(config_data["n_layer"]),
        dropout=float(config_data["dropout"]),
        ffn_multiplier=int(config_data.get("ffn_multiplier", 4)),
    )

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = GPT(config).to(device)
    # weights_only avoids unpickling arbitrary Python objects and makes the
    # checkpoint boundary explicit: this file contains tensors only.
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    print(model_summary(model))
    print(f"Device: {device}")
    print(f"Checkpoint: {model_dir}")
    print("Type 'exit' to quit.")
    print()

    while True:
        prompt = input(f"{color_label('You >', '36')} ")
        if prompt.strip().lower() == "exit":
            break
        if not prompt.strip():
            print("Please enter a non-empty prompt.\n")
            continue

        prompt_ids = tokenizer.encode(prompt).ids
        output_ids, metrics = generate_with_metrics(
            model,
            prompt_ids,
            tokenizer,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        )
        answer = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"\n{color_label('Model >', '35')}")
        print(answer)
        print("\nMetrics")
        print(f"  total input tokens       : {int(metrics['input_tokens'])}")
        print(f"  total output tokens      : {int(metrics['output_tokens'])}")
        print(f"  first token time         : {metrics['first_token_ms']:.1f} ms")
        print(f"  output throughput        : {metrics['output_tokens_per_second']:.2f} tokens/second")
        print(f"  total time to full answer: {metrics['total_seconds']:.3f} seconds")
        print()


if __name__ == "__main__":
    main()

