"""Implementation of encoder stack."""

from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from .add_and_norm import AddAndNorm
from .encoder_decoder_params import EncoderDecoderParams
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


class Encoder(nn.Module):
    """Encoder layer.

    Args:
        params (EncoderDecoderParams): Encoder layer parameters.

    Note:
        Regarding the placement of the dropout layers, we follow the remark in AIAYN
        which states: "We apply dropout [33] to the output of each sub-layer, before
        it is added to the sub-layer input and normalized."

    Note:
        Following AIAYN, causal masking is not applied in sub-layer 1.
    """

    def __init__(self, params: EncoderDecoderParams) -> None:
        super().__init__()
        d_input, h, d_k, d_v, d_ff, p_dropout = params

        # Sub-layer 1 objects
        self._multi_head_attention = MultiHeadAttention(d_input, h, d_k, d_v)
        self._multi_head_attention_dropout = nn.Dropout(p_dropout)
        self._multi_head_attention_add_and_norm = AddAndNorm(d_input)

        # Sub-layer 2 objects
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
        params (EncoderDecoderParams): Encoder parameters.
    """

    def __init__(self, num_layers: int, params: EncoderDecoderParams) -> None:
        super().__init__()
        self._layers = nn.ModuleList([Encoder(params) for _ in range(num_layers)])

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b n d_input"]) -> Float[Tensor, "b n d_input"]:
        """Compute encoder stack output.

        Args:
            x (Tensor): Input tensor.
        """
        for layer in self._layers:
            x = layer(x)  # (b, n, d_input)

        return x
