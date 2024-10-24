"""Implementation of add-and-norm layer."""

from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


class AddAndNorm(nn.Module):
    """Add-and-norm layer.

    Args:
        d_input (int): Input dimension.
    """

    def __init__(self, d_input: int) -> None:  # noqa: DCO010
        super().__init__()
        self._layer_norm = nn.LayerNorm(d_input)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Tensor, "b n d_input"], y: Float[Tensor, "b n d_input"]
    ) -> Float[Tensor, "b n d_input"]:
        """Compute add-and-norm output.

        Args:
            x (Tensor): Input tensor for first summand.
            y (Tensor): Input tensor for second summand.

        Returns:
            Tensor: Final output of add-and-norm layer.
        """
        return self._layer_norm(x + y)
