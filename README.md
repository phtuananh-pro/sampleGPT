# SampleGPT: a Part IV Transformer code tutorial

## Purpose

This repository is a runnable code companion to Part IV, “The Transformer Revolution,” of the [Frontier Models guide](https://frontiermodels.guide). Its purpose is to make the guide's decoder-only Transformer mechanics concrete and inspectable in Python.

The program reads one or more plain-text (`.txt`) files, converts the text to token IDs, and trains a decoder-only Transformer to predict each next token from the preceding context. The next token already present in the text supplies the target, so training needs no separately labeled examples. After training, the model can generate a continuation in patterns learned from that text.

The implementation makes the main Part IV path visible: embeddings and positions; Q/K/V projections; scaled causal self-attention; multi-head recombination; position-wise FFNs; residual connections and normalization; vocabulary logits; shifted-token cross-entropy; parameter updates; and autoregressive generation. Each Python class and function has explanatory documentation that connects its role to this path or identifies it as supporting code.

This is a small teaching model, not a production LLM. It uses learned absolute positional embeddings, dense GELU FFNs, LayerNorm, and recomputes the context for each generated token. It does not implement every subject covered by Part IV: MoE routing, auxiliary multi-token prediction, KV caching, alternative sequence-memory designs, or production-scale training and serving. The Transformer operations are written explicitly in `src/model.py`; PyTorch supplies tensor operations, automatic differentiation, and optimization.

## Install

Python 3.11 or later is recommended. From the repository root, create a virtual environment and install the dependencies.

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:
```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
The requirements install NumPy and PyTorch. On Windows/Linux they require the CUDA 13.2 PyTorch build, which also runs on the CPU; macOS uses the standard PyTorch build. You do not need to install the CUDA Toolkit separately. A working NVIDIA driver is required to use an NVIDIA GPU.


## Repository layout

```text
sampleGpt/
├── data/
│   ├── data_shakespeare.txt     # included text dataset
├── models/                      # one folder per training run
├── src/
│   ├── data.py                  # text files, tokenizer, windows, metadata
│   ├── model.py                 # decoder-only Transformer
│   ├── train.py                 # self-supervised training
│   └── infer.py                 # interactive generation and metrics
├── requirements.txt
└── README.md
```

## Train

The included Shakespeare text is the default demonstration dataset. From the repository root:

macOS/Linux:
```bash
python -m src.train --data data/data_shakespeare.txt --name shakespeare_demo --device auto
```

Windows PowerShell:
```pwsh
python -m src.train --data .\data\data_shakespeare.txt --name shakespeare_demo --device auto
```

`--data` accepts one or more `.txt` files or directories containing `.txt` files. Directories are searched recursively. To train on a larger text collection, provide its file path or directory path; each run gets its own model folder.

Select the compute device with `--device`:

| Value | Behavior |
|---|---|
| `auto` | Use CUDA when available; otherwise use CPU. This is the default. |
| `cuda` | Require an available NVIDIA CUDA device. |
| `cpu` | Force CPU execution. |

Example forcing the NVIDIA GPU:

```pwsh
python -m src.train --data .\data\data_shakespeare.txt --name shakespeare_gpu --device cuda
```

Default training settings:

```text
vocabulary         2,048 BPE tokens
context window     128 tokens
model width        128
attention heads    4
Transformer blocks 4
FFN width          512
batch size         32 windows
training steps     3,000
```

Common options include `--max-iters`, `--block-size`, `--batch-size`, `--n-layer`, `--n-head`, and `--n-embd`. Run `python -m src.train --help` to see all options.

At the end of training, the console reports the model configuration and elapsed time. The run also records its final loss and parameter count.

## Training-time estimates

These broad estimates are for the default 4-block model and 3,000 steps on the included Shakespeare dataset. Actual time depends on the processor, GPU, PyTorch build, storage, and system load.

| Run | CPU-only computer | Computer with NVIDIA GPU |
|---|---:|---:|
| `data_shakespeare.txt` (~1.1 MB), 3,000 steps | about 20–60 minutes | about 1–5 minutes |

Training time is driven mainly by the number of steps and model settings. A larger text file takes longer to tokenize and needs more memory, but does not by itself make each training step proportionally slower. If a run is too slow, reduce `--max-iters`, `--batch-size`, `--block-size`, `--n-layer`, or `--n-embd`.

## Model output folders

Each run creates a new folder under `models/`:

```text
models/<run-name>/
├── model.pt              # learned tensor parameters
├── tokenizer.json        # tokenizer used by this run
├── config.json           # architecture and dataset metadata
├── training_summary.json # duration, final loss, parameter count
└── README.md             # source dataset files and run details
```

The source dataset is not copied into the model folder. Keep `model.pt`, `tokenizer.json`, and `config.json` together when moving a run.

## Run inference

Start an interactive session with the trained Shakespeare model:

macOS/Linux:
```bash
python -m src.infer models/shakespeare_demo
```

Windows PowerShell:
```pwsh
python -m src.infer .\models\shakespeare_demo
```

Enter a text prefix that resembles the training data, for example:

```text
ROMEO:
```
```text
HAMLET:
```
```text
To be, or not to be:
```

Type `exit` to close the session. The model generates a continuation based on patterns in its training text; it is not expected to answer factual questions.

Inference also accepts `--device auto`, `--device cuda`, or `--device cpu`, plus `--max-new-tokens`, `--temperature`, and `--top-k`. It reports prompt and output token counts and generation timing. This tutorial recomputes the context for each generated token and does not use a KV cache, so it is intentionally slower than production inference.


## What this tutorial demonstrates

| Part IV idea | Where to see it |
|---|---|
| Text becomes token IDs | `src/data.py` |
| Training context windows | `src/data.py:get_batch` |
| Self-supervised shifted targets | `src/data.py:get_batch` and `src/model.py:GPT.forward` |
| Token and positional embeddings | `src/model.py:GPT` |
| Q/K/V projections and scaled dot products | `src/model.py:CausalSelfAttention` |
| Causal masking before softmax | `src/model.py:CausalSelfAttention.forward` |
| Multiple attention heads and output projection | `src/model.py:CausalSelfAttention.forward` |
| Feed-forward network | `src/model.py:FeedForward` |
| Residual connections and normalization | `src/model.py:TransformerBlock` |
| Stacked Transformer blocks | `src/model.py:GPT.forward` |
| Logits and cross-entropy loss | `src/model.py:GPT.forward` |
| Backpropagation and parameter updates | `src/train.py` |
| Autoregressive generation | `src/infer.py:generate_with_metrics` |

Python source files are under `src/` and include comments to support line-by-line study.


## Training arguments

Pass these options to `python -m src.train`. Defaults are shown for arguments that have one.

| Argument | Default | Meaning |
|---|---:|---|
| `--data` | `data` | One or more `.txt` files or directories containing `.txt` files. Directories are searched recursively. |
| `--name` | `tutorial_YYYYMMDD_HHMMSS` | Name of the output folder under `models/`. If it already exists, training asks whether to replace it; answering yes deletes that folder and creates a fresh run. |
| `--vocab-size` | `2048` | Target maximum size of the byte-level BPE tokenizer vocabulary. |
| `--block-size` | `128` | Context length: maximum number of preceding tokens the model can use to predict the next token. |
| `--batch-size` | `32` | Number of training windows processed together in each optimizer step. Larger batches use more memory. |
| `--max-iters` | `3000` | Number of training optimizer steps. |
| `--eval-interval` | `100` | Run loss evaluation after this many training steps. |
| `--eval-iters` | `20` | Number of batches from each split (training and validation) used to estimate loss at each evaluation. |
| `--learning-rate` | `0.0003` | AdamW optimizer step size. |
| `--n-embd` | `128` | Width of each token representation, also called the model dimension. |
| `--n-head` | `4` | Number of attention heads in each Transformer block. `n-embd` must be divisible by this value. |
| `--n-layer` | `4` | Number of Transformer blocks stacked in the model. |
| `--dropout` | `0.1` | Dropout probability used during training to reduce overfitting. |
| `--seed` | `1337` | Seed for Python and PyTorch random number generators to make runs more repeatable. |
| `--device` | `auto` | `auto` uses CUDA if available, otherwise CPU; `cuda` requires CUDA; `cpu` forces CPU. |

Example with several training options:

```pwsh
python -m src.train `
  --data .\data\data_shakespeare.txt `
  --name shakespeare_experiment `
  --device cuda `
  --max-iters 5000 `
  --block-size 128 `
  --n-layer 4 `
  --n-head 4 `
  --n-embd 128
```

## Acknowledgements

The repository's teaching style is inspired by Andrew Karpathy's educational GPT implementations, especially [minGPT](https://github.com/karpathy/minGPT), [nanoGPT](https://github.com/karpathy/nanoGPT), and the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) material. This implementation is organized around the terminology and chapter sequence of Part IV of the [Frontier Models guide](https://frontiermodels.guide).
