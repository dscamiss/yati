"""Implementation of multi-head attention layer."""

import math
from typing import Tuple

import torch
from einops import einsum, rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer.

    Args:
        d_input (int): Input dimension.
        h (int): Number of heads.
        d_k (int): Number of rows in {Q, K}-matrices.
        d_v (int): Number of rows in V-matrix.
        apply_causal_mask (bool): Apply causal mask (default = False).
        max_seq_len (int): Maximum input sequence length.  This argument is
            only used when apply_causal_mask is True (default = -1).

    Raises:
        ValueError: If apply_causal_mask is True and max_seq_len <= 0.

    Note:
        Following AIAYN, head i has parameter matrices W_i^Q, W_i^K, W_i^V.
        In this implementation, these parameter matrices are not maintained
        separately for each head, but are maintained as sub-matrices of
        larger parameter matrices.

        For example, instead of separately maintaining

            W_1^Q, W_2^Q, ..., W_h^Q

        of size (d_k, d_input), we maintain the block matrix

            W^Q = [
                W_1^Q
                W_2^Q
                ...
                W_h^Q
            ]

        of size (h * d_k, d_input).
    """

    def __init__(
        self,
        d_input: int,
        h: int,
        d_k: int,
        d_v: int,
        apply_causal_mask: bool = False,
        max_seq_len: int = -1,
    ) -> None:
        super().__init__()
        self._h = h
        self._scaling_factor = math.sqrt(d_k)
        self._apply_causal_mask = apply_causal_mask

        self._w_q = nn.Linear(d_input, h * d_k, bias=False)
        self._w_k = nn.Linear(d_input, h * d_k, bias=False)
        self._w_v = nn.Linear(d_input, h * d_v, bias=False)
        self._w_o = nn.Linear(h * d_v, d_input, bias=False)

        if self._apply_causal_mask:
            # Sanity check: max_seq_len must be positive to apply causal mask
            if max_seq_len <= 0:
                raise ValueError("max_seq_len > 0 is required to apply causal mask")

            # Compute causal mask
            mask = torch.ones(max_seq_len, max_seq_len)
            mask = (
                torch.tril(mask).unsqueeze(0).unsqueeze(0)
            )  # (1, 1, max_seq_len, max_seq_len)

            # Ensure mask is saved as part of the module state
            self.register_buffer("_causal_mask", mask)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        q: Float[Tensor, "b n d_input"],
        k: Float[Tensor, "b n d_input"],
        v: Float[Tensor, "b n d_input"],
    ) -> Float[Tensor, "b n d_input"]:
        """Compute multi-head attention output.

        Args:
            q (Tensor): Input tensor for "query".
            k (Tensor): Input tensor for "key".
            v (Tensor): Input tensor for "value".
        """
        q = self._w_q(q)  # (b, n, h * d_k)
        k = self._w_k(k)  # (b, n, h * d_k)
        v = self._w_v(v)  # (b, n, h * d_v)

        q = rearrange(q, "b n (h d_k) -> b h n d_k", h=self._h)  # (b, h, n, d_k)
        k = rearrange(k, "b n (h d_k) -> b h d_k n", h=self._h)  # (b, h, d_k, n)
        v = rearrange(v, "b n (h d_v) -> b h n d_v", h=self._h)  # (b, h, n, d_v)

        x = (
            einsum(q, k, "b h n_1 j, b h j n_2 -> b h n_1 n_2") / self._scaling_factor
        )  # (b, h, n, n)

        if self._apply_causal_mask:
            n = x.shape[-1]
            x = x.masked_fill(self.causal_mask[:, :, :n, :n] == 0, float("-inf"))

        x = torch.softmax(x, dim=-1)  # (b, h, n, n)

        x = einsum(x, v, "b h n j, b h j d_v -> b h n d_v")  # (b, h, n, d_v)
        x = rearrange(x, "b h n d_v -> b n (h d_v)", h=self._h)  # (b, n, h * d_v)

        return self._w_o(x)  # (b, n, d_input)

    @property
    def apply_causal_mask(self) -> bool:
        """Getter for apply_causal_mask."""
        return self._apply_causal_mask

    @property
    def causal_mask(self) -> Tensor:
        """Getter for causal_mask."""
        return self._causal_mask

    @property
    def h(self) -> int:
        """Getter for h."""
        return self._h

    def get_weights(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Getter for weight matrices in Linear layers."""
        return (self._w_q.weight, self._w_k.weight, self._w_v.weight, self._w_o.weight)
