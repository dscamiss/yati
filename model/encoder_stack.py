"""Implementation of encoder stack."""

from dataclasses import astuple, dataclass
from typing import Any, Iterator

from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from .add_and_norm import AddAndNorm
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


@dataclass
class EncoderParams:
    """Dataclass for encoder layer parameters.

    Args:
        d_input (int): Input dimension.
        h (int): Number of heads; used in sub-layer 1.
        d_k (int): Number of rows in {Q, K}-matrices; used in sub-layer 1.
        d_v (int): Number of rows in V-matrix; used in sub-layer 1.
        d_ff (int): Hidden layer dimension; used in sub-layer 2.
        p_dropout (float): Dropout probability.
    """

    d_input: int
    h: int
    d_k: int
    d_v: int
    d_ff: int
    p_dropout: float

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self))


class Encoder(nn.Module):
    """Encoder layer.

    Args:
        encoder_params (EncoderParams): Encoder layer parameters.

    Note:
        Regarding the placement of the dropout layers, we follow the remark in AIAYN
        which states: "We apply dropout [33] to the output of each sub-layer, before
        it is added to the sub-layer input and normalized."

    Note:
        Following AIAYN, we do not apply causal masking in sub-layer 1.
    """

    def __init__(self, encoder_params: EncoderParams) -> None:
        super().__init__()
        d_input, h, d_k, d_v, d_ff, p_dropout = encoder_params

        self._multi_head_attention = MultiHeadAttention(d_input, h, d_k, d_v)
        self._multi_head_attention_dropout = nn.Dropout(p_dropout)
        self._multi_head_attention_add_and_norm = AddAndNorm(d_input)
        self._feed_forward = FeedForward(d_input, d_ff)
        self._feed_forward_dropout = nn.Dropout(p_dropout)
        self._feed_forward_add_and_norm = AddAndNorm(d_input)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b n d_input"]) -> Float[Tensor, "b n d_input"]:
        """Compute encoder output.

        Args:
            x (Tensor): Input tensor.
        """
        # Compute sub-layer 1 output
        y = self._multi_head_attention(x, x, x)  # (b, n, d_input)
        y = self._multi_head_attention_dropout(y)  # (b, n, d_input)
        y = self._multi_head_attention_add_and_norm(x, y)  # (b, n, d_input)

        # Compute sub-layer 2 output
        x = self._feed_forward(y)  # (b, n, d_input)
        x = self._feed_forward_dropout(x)  # (b, n, d_input)
        x = self._feed_forward_add_and_norm(y, x)  # (b, n, d_input)

        return x


class EncoderStack(nn.Module):
    """Encoder stack.

    Args:
        num_layers (int): Number of encoder layers.
        encoder_params (EncoderParams): Encoder parameters.
    """

    def __init__(self, num_layers: int, encoder_params: EncoderParams) -> None:
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [Encoder(encoder_params) for _ in range(num_layers)]
        )

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b n d_input"]) -> Float[Tensor, "b n d_input"]:
        """Compute encoder stack output.

        Args:
            x (Tensor): Input tensor.
        """
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)  # (b, n, d_input)

        return x
