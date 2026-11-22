"""Dataset and tokenizer helpers for the Part IV Transformer tutorial.

This module intentionally keeps data preparation visible.  A decoder-only
language model does not receive a table of question/answer labels here.  It
receives a stream of text, turns that text into token IDs, and learns from a
one-token shift:

    input:  [t_1, t_2, t_3, ...]
    target: [t_2, t_3, t_4, ...]

That is the self-supervised objective described in Part IV.13.  The tokenizer
is the one piece of preprocessing that would be distracting to implement from
scratch in this tutorial, so the small ``tokenizers`` package is used for it.
The Transformer itself is implemented in ``src/model.py`` rather than hidden
behind a ready-made Transformer class. These helpers connect raw text to
Part IV.5's token IDs and Part IV.13's shifted training windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


# This token is inserted between source documents.  It is a useful boundary
# because otherwise the last character of one file and the first character of
# the next file would look like one continuous sentence to the model.
END_OF_DOCUMENT = "<|endoftext|>"


def discover_text_files(sources: Sequence[str]) -> list[Path]:
    """Return sorted ``.txt`` files from files and/or directories.

    A directory is deliberately supported because it makes dataset
    experimentation easy: put another text file in ``data/`` and run the
    same training command again.  Sorting keeps the training stream
    reproducible when the random seed is fixed. This is input plumbing around
    Part IV's model computation, not part of the Transformer itself.
    """

    discovered: set[Path] = set()
    for source in sources:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        if path.is_file():
            if path.suffix.lower() == ".txt":
                discovered.add(path.resolve())
            continue

        for text_file in path.rglob("*.txt"):
            if text_file.is_file():
                discovered.add(text_file.resolve())

    files = sorted(discovered, key=lambda item: str(item).lower())
    if not files:
        raise ValueError("No .txt files were found in the requested dataset paths.")
    return files


def build_tokenizer(files: Sequence[Path], output_path: Path, vocab_size: int) -> Tokenizer:
    """Train and save a small byte-level BPE tokenizer.

    The tokenizer changes text into discrete IDs.  It is not the Transformer:
    it only defines the vocabulary and the reversible text/ID conversion.
    Byte-level BPE is convenient for a tutorial because it can represent any
    UTF-8 text without maintaining a separate unknown-character vocabulary.
    This is the text-to-token-ID boundary in Part IV.5; it does not determine
    how the Transformer learns relationships between token IDs.
    """

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # [UNK] is required by the BPE model.  END_OF_DOCUMENT is a genuine
        # vocabulary item so we can append its ID between files ourselves.
        special_tokens=["[UNK]", END_OF_DOCUMENT],
        show_progress=True,
    )
    tokenizer.train([str(path) for path in files], trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def encode_files(tokenizer: Tokenizer, files: Sequence[Path]) -> tuple[list[int], dict[str, int]]:
    """Encode every file and join them with an explicit document-boundary ID.

    The returned token stream is what the training windows are sampled from.
    The second return value is metadata used in the per-run README, so a
    learner can see exactly which source files influenced a model without
    copying the potentially large files into ``models/<run>/``.
    """

    separator_id = tokenizer.token_to_id(END_OF_DOCUMENT)
    if separator_id is None:
        raise RuntimeError(f"Tokenizer did not contain special token {END_OF_DOCUMENT!r}.")

    token_ids: list[int] = []
    token_counts: dict[str, int] = {}
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        encoded = tokenizer.encode(text).ids
        token_ids.extend(encoded)
        token_ids.append(separator_id)
        token_counts[str(file_path)] = len(encoded)
    return token_ids, token_counts


def split_tokens(token_ids: Sequence[int], validation_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Make a simple contiguous train/validation split.

    Keeping the split contiguous is easy to explain: the first part of the
    token stream is used for updates and the final part is held out for a
    rough generalization check.  More advanced projects may split by document
    or shuffle windows, but that would hide the basic idea this repository is
    meant to teach. The held-out stream supports the rough generalization
    check alongside the training objective in Part IV.13.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    boundary = int(len(token_ids) * (1.0 - validation_fraction))
    train = torch.tensor(token_ids[:boundary], dtype=torch.long)
    validation = torch.tensor(token_ids[boundary:], dtype=torch.long)
    return train, validation


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``B`` overlapping training windows and their shifted targets.

    ``x`` and ``y`` both have shape ``[B, T]``.  The target is one position to
    the right of the input.  Notice that no human annotation is involved: the
    next token in the same text supplies the label, which is precisely why
    this is self-supervised learning. This is the training-window and
    teacher-forcing setup from Part IV.13.3 and IV.13.10.
    """

    if len(tokens) <= block_size:
        raise ValueError(
            f"The split has only {len(tokens)} tokens, but block_size={block_size} "
            "needs at least block_size + 1 tokens."
        )

    starts = torch.randint(0, len(tokens) - block_size, (batch_size,))
    x = torch.stack([tokens[start : start + block_size] for start in starts])
    y = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


def write_run_readme(
    output_dir: Path,
    files: Sequence[Path],
    token_counts: dict[str, int],
    source_root: Path,
) -> None:
    """Write a small provenance note instead of copying the full dataset.

    A checkpoint is meaningful only together with the tokenizer and the
    training corpus.  The note links back to the original files using paths
    relative to the model directory.  This keeps each run self-describing
    without duplicating tens of megabytes of text. This records data
    provenance for a run; it is not a Transformer operation from Part IV.
    """

    lines = [
        "# This model run",
        "",
        "This folder contains the weights and tokenizer for one tutorial training run.",
        "The source text was intentionally not copied here; the links below point",
        "back to the dataset files in the repository.",
        "",
        "## Dataset files",
        "",
    ]
    for file_path in files:
        try:
            relative_to_model = Path("..") / ".." / file_path.relative_to(source_root)
        except ValueError:
            # If a learner trains from a file outside the repository, retain a
            # readable absolute path in the note instead of inventing a bad link.
            relative_to_model = file_path
        count = token_counts.get(str(file_path), "unknown")
        lines.append(f"- `{relative_to_model.as_posix()}` — {count} encoded tokens")

    lines.extend(
        [
            "",
            "The model must be loaded with the matching `model.pt`, `tokenizer.json`,",
            "and `config.json` from this same folder.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    """Write readable run metadata beside a checkpoint.

    This stores architecture and training metadata, not the learned tensor
    parameters themselves.
    """

    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
