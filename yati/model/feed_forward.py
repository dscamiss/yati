"""Implementation of feed-forward layer."""

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


class FeedForward(nn.Module):
    """Feed-forward layer.

    Args:
        d_input: Input/output dimension.
        d_ff: Hidden layer dimension.

    Note:
        This does not apply dropout after the first affine transformation.
        This is because in AIAYN, the authors write "We apply dropout [33] to
        the output of each sub-layer, before it is added to the sub-layer input
        and normalized."  In the paper, "sub-layer" has the following meaning:
        In an encoder layer, the sub-layers are (1) multi-head attention and
        (2) feed-forward.  In a decoder layer, the sub-layers are (1) masked
        multi-head attention, (2) multi-head attention, and (3) feed-forward.
    """

    def __init__(self, d_input: int, d_ff: int) -> None:  # noqa: DCO010
        super().__init__()
        self.affine_1 = nn.Linear(d_input, d_ff, bias=True)
        self.affine_2 = nn.Linear(d_ff, d_input, bias=True)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b n d_input"]) -> Float[Tensor, "b n d_input"]:
        """Compute feed-forward output.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Final output of feed-forward layer.
        """
        x = self.affine_1(x)  # (b, n, d_ff)
        x = torch.relu(x)  # (b, n, d_ff)
        x = self.affine_2(x)  # (b, n, d_input)

        return x
