"""Train the readable decoder-only Transformer tutorial.

Run from the repository root:

    python -m src.train --data data/data_shakespeare.txt --name shakespeare_demo

This script connects the data helpers to Part IV.13: sample shifted windows,
run a causal full-window forward pass, compute cross-entropy, backpropagate,
and update parameters. The optimizer and automatic differentiation are
intentionally delegated to PyTorch. They are important for making the model
learn, but the educational
focus of this repository is the architecture and the self-supervised target,
not the implementation of AdamW or reverse-mode autodiff.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch

from .data import (
    build_tokenizer,
    discover_text_files,
    encode_files,
    get_batch,
    split_tokens,
    write_json,
    write_run_readme,
)
from .model import GPT, ModelConfig, model_summary


def parse_args() -> argparse.Namespace:
    """Define dataset, architecture, optimization, and device controls.

    Options expose Part IV quantities such as context, width, heads, and block
    depth, along with controls for the surrounding training procedure.
    """

    parser = argparse.ArgumentParser(description="Train the Part IV decoder-only Transformer.")
    parser.add_argument(
        "--data",
        nargs="+",
        default=["data"],
        help="One or more .txt files or directories containing .txt files (default: data).",
    )
    parser.add_argument("--name", default=None, help="Folder name under models/ for this run.")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Use auto to select CUDA when available.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    """Select CPU or CUDA for tensor computation; this is runtime plumbing."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    """Seed sampling and initialization to make demonstrations repeatable.

    Reproducibility helps compare Part IV training runs, though hardware and
    backend details can still cause small differences.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confirm_overwrite(output_dir: Path) -> bool:
    """Protect an existing checkpoint before starting a fresh training run."""

    while True:
        try:
            answer = input(f"Output folder already exists: {output_dir}\nOverwrite it? [y/N]: ").strip().lower()
        except EOFError:
            print("No confirmation received. Exiting without changing the existing run.")
            return False

        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            print("Keeping the existing run. Exiting.")
            return False
        print("Please answer yes or no.")


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int,
) -> tuple[float, float]:
    """Estimate Part IV.13 next-token loss on train and held-out windows.

    Evaluation disables dropout and does not update parameters; it measures
    the current model on newly sampled windows from each split.
    """

    model.eval()
    values: dict[str, list[float]] = {"train": [], "validation": []}
    for split_name, tokens in (("train", train_tokens), ("validation", validation_tokens)):
        for _ in range(eval_iters):
            x, y = get_batch(tokens, batch_size, block_size, device)
            _, loss = model(x, y)
            assert loss is not None
            values[split_name].append(float(loss.item()))
    model.train()
    return sum(values["train"]) / len(values["train"]), sum(values["validation"]) / len(values["validation"])


def main() -> None:
    """Orchestrate tokenization, self-supervised updates, evaluation, and saving.

    The core loop is Part IV.13: text provides shifted targets, the decoder
    computes position-wise logits, cross-entropy measures surprise, and
    backpropagation updates learned parameters.
    """

    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    files = discover_text_files(args.data)

    run_name = args.name or datetime.now().strftime("tutorial_%Y%m%d_%H%M%S")
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name or "/" in run_name or "\\" in run_name:
        raise ValueError("--name must be a single folder name, not a path.")

    models_dir = Path("models")
    if models_dir.is_symlink():
        raise RuntimeError(f"Refusing to manage run folders through a symbolic link: {models_dir}")
    output_dir = models_dir / run_name
    if output_dir.is_symlink():
        raise RuntimeError(f"Refusing to overwrite a symbolic link: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a folder: {output_dir}")
        if not confirm_overwrite(output_dir):
            return
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"Dataset files ({len(files)}):")
    for file_path in files:
        print(f"  - {file_path}")
    print(f"Device: {device}")
    print("Training tokenizer...")
    tokenizer = build_tokenizer(files, output_dir / "tokenizer.json", args.vocab_size)
    token_ids, token_counts = encode_files(tokenizer, files)
    train_tokens, validation_tokens = split_tokens(token_ids)

    if len(train_tokens) <= args.block_size or len(validation_tokens) <= args.block_size:
        raise ValueError(
            "The dataset is too small for the requested block size. Add more text or reduce --block-size."
        )

    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Save the architecture before training as well as after it.  The saved
    # configuration is what lets inference rebuild exactly this model instead
    # of relying on today's Python defaults.
    write_json(
        output_dir / "config.json",
        {
            **config.to_dict(),
            "dataset_sources": [str(path) for path in files],
            "train_tokens": len(train_tokens),
            "validation_tokens": len(validation_tokens),
            "seed": args.seed,
        },
    )
    write_run_readme(output_dir, files, token_counts, Path("data").resolve())

    print(model_summary(model))
    print(f"Encoded tokens     : {len(token_ids):,}")
    print(f"Train / validation  : {len(train_tokens):,} / {len(validation_tokens):,}")
    print(f"Training steps      : {args.max_iters:,}")
    print()

    started = time.perf_counter()
    last_loss = None
    for step in range(args.max_iters):
        # One batch is B windows of T input tokens plus B windows of T shifted
        # target tokens.  The full Transformer predicts B*T positions at once.
        inputs, targets = get_batch(train_tokens, args.batch_size, args.block_size, device)
        _, loss = model(inputs, targets)
        assert loss is not None

        # PyTorch performs the mechanics that Part IV.13.11 describes:
        # backpropagate the scalar loss through logits, all blocks, and the
        # embeddings, then update all learned parameters.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            train_loss, validation_loss = estimate_loss(
                model,
                train_tokens,
                validation_tokens,
                args.batch_size,
                args.block_size,
                device,
                args.eval_iters,
            )
            elapsed = time.perf_counter() - started
            print(
                f"step {step:>5}/{args.max_iters} | "
                f"train loss {train_loss:.4f} | validation loss {validation_loss:.4f} | "
                f"elapsed {elapsed / 60:.1f} min"
            )

    torch.save(model.state_dict(), output_dir / "model.pt")
    write_json(
        output_dir / "training_summary.json",
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - started, 2),
            "last_batch_loss": last_loss,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )

    # The end-of-training report is intentionally repeated in inference.  A
    # learner can compare the fixed architecture used by both programs.
    print()
    print("Training complete.")
    print(model_summary(model))
    print(f"Saved model folder : {output_dir}")
    print(f"  weights          : {output_dir / 'model.pt'}")
    print(f"  tokenizer        : {output_dir / 'tokenizer.json'}")
    print(f"  config           : {output_dir / 'config.json'}")
    print(f"  dataset note     : {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
