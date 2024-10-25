"""Implementation of positional encoding layer."""

import math

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


class PositionalEncoding(nn.Module):
    """Positional encoding layer.

    Args:
        d_input: Input dimension.  Must be divisible by 2.
        max_seq_len: Maximum input sequence length.

    Raises:
        ValueError: If d_input is not divisible by 2.

    Note:
        From AIAYN, the positional encoding is defined by:

            PE(pos, 2i)     = sin(pos / 10000^(2i / d_input))
            PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_input))
                                               ^^ Note 2i used in both cases

        Here, pos is the input sequence index and i is the component index.
    """

    def __init__(self, d_input: int, max_seq_len: int) -> None:  # noqa: DCO010
        super().__init__()
        # Sanity check: d_input must be divisible by 2
        if d_input % 2 != 0:
            raise ValueError("d_input must be divisible by 2")

        self.max_seq_len = max_seq_len

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(-1)  # (max_seq_len, 1)
        idx = torch.arange(0, d_input, 2, dtype=torch.float)  # (d_input / 2)

        # Compute 10000^(2i / d_input) in log-space for numerical stability
        log_den = (idx / d_input) * math.log(10000.0)  # (d_input / 2)
        den = torch.exp(-1.0 * log_den).unsqueeze(0)  # (1, d_input / 2)

        encoding = torch.zeros(1, max_seq_len, d_input)
        encoding[:, :, 0::2] = torch.sin(pos * den)
        encoding[:, :, 1::2] = torch.cos(pos * den)

        # Ensure encoding is saved as part of the module state
        self.register_buffer("encoding", encoding)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b n d_input"]) -> Float[Tensor, "b n d_input"]:
        """
        Compute positional encoding output.

        Args:
            x: Input tensor.

        Raises:
            ValueError: If input x has an invalid shape (i.e., its shape is not
                compatible with the pre-computed positional encoding).

        Returns:
            Tensor: Final output of positional encoding layer.
        """
        if x.shape[1] > self.encoding.shape[1] or x.shape[2] != self.encoding.shape[2]:
            raise ValueError(f"x has invalid shape {x.shape}")
        return x + self.encoding[:, : x.shape[1], :].requires_grad_(False)
