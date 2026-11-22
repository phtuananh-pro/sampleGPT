"""A small decoder-only Transformer written for reading, not production.

The code follows the notation and flow of Part IV of the Frontier Models
guide.  The most important design choice is that the implementation spells out
the operations instead of calling ``torch.nn.Transformer`` or a framework GPT:

    token IDs [B,T]
      -> token + positional embeddings [B,T,d_model]
      -> repeated pre-norm decoder blocks [B,T,d_model]
      -> vocabulary logits [B,T,vocab_size]

PyTorch still supplies tensors, automatic differentiation, familiar linear
layers, and LayerNorm.  Those are useful tools for the tutorial.  The actual
Q/K/V projections, score matrix, causal mask, softmax, value mixing,
concatenation, residual paths, and FFN are visible below.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    """Store the choices that define the decoder architecture.

    These fields correspond to Part IV's vocabulary, context length, model
    width, head count, block depth, and position-wise FFN width. Keeping them
    together lets training and inference rebuild the same model.
    """

    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    ffn_multiplier: int = 4

    def to_dict(self) -> dict[str, int | float]:
        """Serialize architecture choices for rebuilding a checkpoint.

        Part IV distinguishes learned parameters from architecture settings;
        this writes the latter so inference can reconstruct the model first.
        """

        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit left-to-right mask.

    1. project the current representations into Q, K, and V;
    2. split the model width into several attention heads;
    3. compute scaled QK^T scores;
    4. add the causal mask *before* softmax;
    5. use the normalized weights to mix V;
    6. concatenate heads and project back to ``d_model``.

    The output shape is still ``[B,T,d_model]``.  The content changes because
    tokens communicate, but shape preservation is what lets blocks stack.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Create learned projections and fixed causal visibility for each head.

        The projections implement Part IV.7's Q/K/V transforms; the
        lower-triangular buffer implements Part IV.8's no-future-reading rule.
        """
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head for this tutorial model.")

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout

        # These Linear layers contain learned parameters.  q, k, and v below
        # are temporary activations computed from the current token states.
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # A lower-triangular matrix represents visibility: row i may read
        # columns 0..i, never a future column.  It is a fixed buffer, not a
        # learned parameter.  It is broadcast over batch and head dimensions.
        causal = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route information between visible token positions."""

        batch, time, model_width = x.shape
        if time > self.causal_mask.size(0):
            raise ValueError("The input sequence is longer than the configured context window.")

        # A projection changes each token vector independently.  At this point
        # no token-to-token communication has happened yet.
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # [B,T,C] -> [B,H,T,d_head].  Each head receives a smaller learned
        # subspace, which allows several routing patterns under one model-wide
        # width budget (Part IV.9).
        q = q.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_size).transpose(1, 2)

        # QK^T creates [B,H,T,T]: every query position scores every source
        # position.  Scaling by sqrt(d_head) keeps dot products in a useful
        # range before softmax.
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        # Future positions are still present in this full-window tensor.  The
        # mask only changes their scores to -infinity, so softmax turns their
        # weights into zero.  This is what prevents target leakage during
        # parallel training (Part IV.8.5 and IV.13.9).
        visible = self.causal_mask[:time, :time]
        scores = scores.masked_fill(~visible, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        # A @ V is the actual information-routing operation: each destination
        # token receives a weighted mixture of source Value vectors.
        routed = weights @ v

        # Reassemble [B,H,T,d_head] into [B,T,C], then let a learned output
        # projection mix information from the different heads.
        routed = routed.transpose(1, 2).contiguous().view(batch, time, model_width)
        return self.output_dropout(self.output(routed))


class FeedForward(nn.Module):
    """The dense, position-wise nonlinear transformation sublayer.

    Attention mixes information *between positions*.  The FFN below does not
    mix positions: the same learned expansion/activation/projection is applied
    independently to every token vector.  It expands ``d_model`` to ``d_ff``
    and projects back, preserving the block interface (Part IV.10).
    """

    def __init__(self, config: ModelConfig) -> None:
        """Create the shared position-wise feature transformation.

        This dense FFN follows Part IV.10: expand each token's features, apply
        a nonlinearity, then project back to the residual-stream width.
        """
        super().__init__()
        hidden = config.ffn_multiplier * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            # GELU is a smooth, easy-to-read nonlinear activation.  The exact
            # activation is less important here than seeing why a nonlinearity
            # is needed between two linear maps.
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the feature vector at each token position."""

        return self.net(x)


class TransformerBlock(nn.Module):
    """One pre-norm decoder block: attention update plus FFN update.

    This is Part IV.12's repeated depth unit. Each block owns separate
    attention, FFN, and normalization parameters while preserving tensor shape.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Assemble normalization, causal attention, and dense FFN sublayers."""
        super().__init__()
        self.norm_for_attention = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm_for_ffn = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two residual updates from Part IV.12.5."""

        # Pre-norm form:
        #   Y = X + MHA(Norm(X))
        # The addition is a structural residual path, not an extra learned
        # layer.  It preserves information and gives gradients an identity-like
        # route through the depth stack.
        x = x + self.attention(self.norm_for_attention(x))

        # The second pre-norm update is:
        #   Z = Y + FFN(Norm(Y))
        # The FFN returns to d_model so this addition is shape-compatible.
        x = x + self.ffn(self.norm_for_ffn(x))
        return x


class GPT(nn.Module):
    """A compact decoder-only language model for next-token prediction.

    Its forward path follows Part IV.12: embeddings, stacked pre-norm blocks,
    final normalization, and vocabulary logits. Optional targets activate the
    self-supervised objective from Part IV.13.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build embeddings, the block stack, final norm, and vocabulary head.

        This implementation chooses learned absolute positions and dense GELU
        FFNs to keep core Part IV mechanics visible; it does not implement all
        architecture variants discussed in the guide.
        """
        super().__init__()
        self.config = config

        # Token IDs are discrete.  The embedding lookup is the first operation
        # that turns them into continuous vectors: [B,T] -> [B,T,d_model].
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Learned absolute positions are intentionally used here because they
        # are simple to connect to Part IV.5: select the first T rows and add
        # them to token embeddings.  More advanced systems may inject position
        # through RoPE or attention biases instead.
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Each block has a separate set of learned parameters.  The architecture
        # repeats, but the weights are not shared across depth (Part IV.2.5).
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)

        # Each final token representation becomes one logit per vocabulary item.
        # Logits are not probabilities until softmax is applied by the loss or
        # by the inference decoder.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map token IDs to logits and optionally compute next-token loss.

        Positions are evaluated in parallel under causal masking. The targets
        come from the one-token shift in ``src.data.get_batch``: Part IV.13's
        teacher-forced training pass.
        """

        batch, time = token_ids.shape
        if time > self.config.block_size:
            raise ValueError("Input sequence is longer than the model context window.")

        positions = torch.arange(time, device=token_ids.device)
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        # Addition combines token identity and position without changing the
        # external representation shape: [B,T,C] + [T,C] -> [B,T,C].
        x = token_vectors + position_vectors

        # This is a sequence in *depth*, not a recurrent sequence in time.
        # Every block sees the output of the block before it.
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten the B*T positions because cross_entropy expects one row
            # of vocabulary logits and one target ID per example.  The model
            # predicts all positions in parallel, but causal masking ensures
            # position t could only use tokens <= t.
            loss = F.cross_entropy(
                logits.reshape(batch * time, self.config.vocab_size),
                targets.reshape(batch * time),
            )
        return logits, loss


def parameter_count(model: nn.Module) -> int:
    """Count learned scalar parameters, excluding masks and fixed buffers.

    This follows Part IV's distinction between persistent learned parameters
    and temporary activations or fixed architectural structures.
    """

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_summary(model: GPT) -> str:
    """Describe the dimensions that shape the Part IV computation.

    Reports depth, residual width, attention heads, FFN width, context length,
    and vocabulary size for training and inference.
    """

    config = model.config
    total = parameter_count(model)
    return "\n".join(
        [
            "Model information",
            f"  total parameters : {total:,}",
            f"  Transformer blocks: {config.n_layer}",
            f"  model width      : {config.n_embd}",
            f"  attention heads  : {config.n_head}",
            f"  head width       : {config.n_embd // config.n_head}",
            f"  FFN width        : {config.ffn_multiplier * config.n_embd}",
            f"  context length   : {config.block_size} tokens",
            f"  vocabulary size  : {config.vocab_size} tokens",
        ]
    )

