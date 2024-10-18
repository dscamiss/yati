"""Implementation of blocks which are shared between the encoder/decoder sides."""

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch import Tensor


class AddAndNorm(nn.Module):
    """Add-and-norm block."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute add-and-norm output.

        Args:
            x: Input tensor of size (b, n, d_model).

        """
        return self.layer_norm(x + y)


class MultiHeadAttention(nn.Module):
    """Multi-head attention block.

    Args:
        h: Number of heads.
        d_model: Embedding dimension.
        d_k: Number of rows in "Q" and "K" matrices.
        d_v: Number of rows in "V" matrices.
        causal_mask: True <==> Apply causal mask.  Defaults to False.
        max_seq_len: Maximum input sequence length.  Only used when
            `causal_mask` is True.  Defaults to -1.
    Note:
        Following AIAYN, head i has parameter matrices W_i^Q, W_i^K, W_i^V.
        In this implementation, these parameter matrices are not maintained
        separately for each head.  Instead, they are maintained collectively
        as sub-matrices of larger matrices.

        For example, instead of separately maintaining parameter matrices

            W_1^Q, W_2^Q, ..., W_h^Q,

        we maintain the larger matrix

            W^Q = [W_1^Q W_2^Q ... W_h^Q].

    Note:
        It is not necessary for `d_model` to be divisible by `h`, which is
        enforced in some implementations.
    """

    def __init__(
        self,
        h: int,
        d_model: int,
        d_k: int,
        d_v: int,
        apply_causal_mask: bool = False,
        max_seq_len: int = -1,
    ) -> None:
        super().__init__()
        self.h = h
        self.w_q = nn.Linear(d_model, h * d_k, bias=False)
        self.w_k = nn.Linear(d_model, h * d_k, bias=False)
        self.w_v = nn.Linear(d_model, h * d_v, bias=False)
        self.w_o = nn.Linear(h * d_v, d_model, bias=False)
        self.scaling_factor = math.sqrt(d_k)
        self.apply_causal_mask = apply_causal_mask

        if self.apply_causal_mask:
            # Sanity check: `max_seq_len` must be positive to apply causal mask
            assert max_seq_len > 0, "max_seq_len > 0 is required to apply causal mask"

            # Compute causal mask
            causal_mask = torch.ones(max_seq_len, max_seq_len)
            causal_mask = (
                torch.tril(causal_mask).unsqueeze(0).unsqueeze(0)
            )  # (1, 1, max_seq_len, max_seq_len)

            # Ensure `causal_mask` is saved as part of the module state
            self.register_buffer("causal_mask", causal_mask)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute multi-head attention output.

            y(Q, K, V) = [head_1(Q, K, V) ... head_h(Q, K, V)] W^O,

            where:

            - head_i(Q, K, V) = Attention(Q (W_i^Q)^t, K (W_i^K)^t, V (W_i^V)^t)
            - Attention(Q, K, V) = Softmax(Q K^t / sqrt(d_k)) V

        Args:
        ----
            q: Tensor of shape (b, n, d_model)
            k: Tensor of shape (b, n, d_model)
            v: Tensor of shape (b, n, d_model)

        """
        q = self.w_q(q)  # (b, n, h * d_k)
        k = self.w_k(k)  # (b, n, h * d_k)
        v = self.w_v(v)  # (b, n, h * d_v)

        q = rearrange(q, "b n (h k) -> b h n k", h=self.h)  # (b, h, n, d_k)
        k = rearrange(k, "b n (h k) -> b h k n", h=self.h)  # (b, h, d_k, n)
        v = rearrange(v, "b n (h v) -> b h n v", h=self.h)  # (b, h, n, d_v)

        x = (
            einsum(q, k, "b h i j, b h j k -> b h i k") / self.scaling_factor
        )  # (b, h, n, n)

        if self.apply_causal_mask:
            n = x.shape[-1]
            x = x.masked_fill(self.causal_mask[:, :, :n, :n] == 0, float("-inf"))
        x = torch.softmax(x, dim=-1)  # (b, h, n, n)

        x = einsum(x, v, "b h i j, b h j k -> b h i k")  # (b, h, n, d_v)
        x = rearrange(x, "b h n v -> b n (h v)", h=self.h)  # (b, n, h * d_v)

        return self.w_o(x)  # (b, n, d_model)


class Embedding(nn.Module):
    """Embedding block.

    Args:
        d_model (:obj:`int`): Embedding dimension.
        num_embeddings (:obj:`int`): Vocabulary size for embedding.

    Attributes:
        embedding (:obj:`int`): Instance of `torch.nn.Embedding`.
        scaling_factor (:obj:`float`): Scaling factor.

    """

    def __init__(self, d_model: int, num_embeddings: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, d_model)
        self.scaling_factor = math.sqrt(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Compute (scaled) embedding.

        Args:
            x (`torch.Tensor`): Input tensor of size (b, n)

        """
        return self.scaling_factor * self.embedding(x)


class FeedForward(nn.Module):
    """Feed-forward block.

    Args:
        d_input: Input/output dimension.
        d_ff: Hidden layer dimension.

    Note:
        This does not apply dropout after the first affine transformation.  This is because
        in AIAYN, the authors write "We apply dropout [33] to the output of each sub-layer,
        before it is added to the sub-layer input and normalized."  Here, "sub-layer" has
        a precise meaning: In an encoder block, the sub-layers are (1) multi-head attention
        and (2) feed-forward.  In a decoder block, the sub-layers are (1) masked multi-head
        attention, (2) multi-head attention, and (3) feed-forward.
    """

    def __init__(self, d_input: int, d_ff: int) -> None:
        super().__init__()
        self.affine_1 = nn.Linear(d_input, d_ff, bias=True)
        self.affine_2 = nn.Linear(d_ff, d_input, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Compute feed-forward output.

        Args:
            x: Input tensor of size (b, n, d_input).
        """
        x = self.affine_1(x)
        x = torch.relu(x)
        x = self.affine_2(x)

        return x


class LayerNormalization(nn.Module):
    """Layer normalization block.

    Args:
        d_model: Embedding dimension.

    Attributes:
        layer_norm: Instance of `torch.nn.LayerNorm`.

    Note:
        This uses `nn.LayerNorm` defaults for the layer normalization
        behavior, since these details are not specified in AIAYN.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        """Compute layer-normalized version of input tensor.

        Args:
        ----
            x (`torch.Tensor`): Input tensor of shape (b, n, d_model)

        """
        return self.layer_norm(x)


class PositionalEncoding(nn.Module):
    """Positional encoding block.

    Definition from AIAYN:
        PE(pos, 2i)     = sin(pos / 10000^{2i / d_model})
        PE(pos, 2i + 1) = cos(pos / 10000^{2i / d_model})

    For numerical stability, we exponentiate logarithms:
        log(1 / 10000^{2i / d_model}) = -(log(10000) / d_model) 2i

    """

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(-1)  # (n, 1)
        idx = torch.arange(0, d_model, 2, dtype=torch.float)
        log_den = -(math.log(10000.0) / d_model) * idx
        den = torch.exp(log_den).unsqueeze(0)  # (1, d_model)

        pe = torch.zeros(1, max_seq_len, d_model)
        pe[:, :, 0::2] = torch.sin(pos * den)
        pe[:, :, 1::2] = torch.cos(pos * den)

        # Ensure `pe` is saved as part of the module state.
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input tensor.

        Args:
        ----
            x (`torch.Tensor`): Input tensor of shape (b, n, d_model)

        """
        n = x.shape[1]
        return x + self.pe[:, :n, :].requires_grad_(False)
