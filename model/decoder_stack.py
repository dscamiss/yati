"""Implementation of decoder stack objects."""

from dataclasses import astuple, dataclass
from typing import Any, Iterator

from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from .add_and_norm import AddAndNorm
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


@dataclass
class DecoderParams:
    """Dataclass for decoder layer parameters.

    Args:
        d_input (int): Input dimension.
        h (int): Number of heads; used in sub-layers 1 and 2.
        d_k (int): Number of rows in {Q, K}-matrices; used in sub-layers 1 and 2.
        d_v (int): Number of rows in V-matrix; used in sub-layers 1 and 2.
        d_ff (int): Hidden layer dimension; used in sub-layer 3.
        p_dropout (float): Dropout probability.
        max_seq_len (int): Maximum input sequence length.
    """

    d_input: int
    h: int
    d_k: int
    d_v: int
    d_ff: int
    p_dropout: float
    max_seq_len: int

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self))


class Decoder(nn.Module):
    """Decoder layer.

    Args:
        decoder_params (DecoderParams): Decoder layer parameters.

    Note:
        Regarding the placement of the dropout layers, we follow the remark in AIAYN
        which states: "We apply dropout [33] to the output of each sub-layer, before
        it is added to the sub-layer input and normalized."

    Note:
        Following AIAYN, causal masking is applied in the multi-head attention block
        used in sub-layer 1, and causal masking is not applied in the multi-head
        attention block used in sub-layer 2.
    """

    def __init__(self, decoder_params: DecoderParams) -> None:
        super().__init__()
        d_input, h, d_k, d_v, d_ff, p_dropout, max_seq_len = decoder_params

        # Sub-layer 1 objects
        self._multi_head_attention_1 = MultiHeadAttention(d_input, h, d_k, d_v, True, max_seq_len)
        self._multi_head_attention_dropout_1 = nn.Dropout(p_dropout)
        self._multi_head_attention_add_and_norm_1 = AddAndNorm(d_input)

        # Sub-layer 2 objects
        self._multi_head_attention_2 = MultiHeadAttention(d_input, h, d_k, d_v)
        self._multi_head_attention_dropout_2 = nn.Dropout(p_dropout)
        self._multi_head_attention_add_and_norm_2 = AddAndNorm(d_input)

        # Sub-layer 3 objects
        self._feed_forward = FeedForward(d_input, d_ff)
        self._feed_forward_dropout = nn.Dropout(p_dropout)
        self._feed_forward_add_and_norm = AddAndNorm(d_input)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, "b n d_input"],
        x_cross: Float[Tensor, "b n d_input"],
    ) -> Float[Tensor, "b n d_input"]:
        """Compute decoder output.

        Args:
            x (Tensor): Input tensor.
            x_cross (Tensor): Cross-attention input tensor.

        Note:
            In the encoder/decoder Transformer architecture, x_cross is the final output
            of the encoder stack.
        """
        # Compute sub-layer 1 output
        y = self._multi_head_attention_1(x, x, x)  # (b, n, d_input)
        y = self._multi_head_attention_dropout_1(y)  # (b, n, d_input)
        y = self._multi_head_attention_add_and_norm_1(x, y)  # (b, n, d_input)

        # Compute sub-layer 2 output
        x = self._multi_head_attention_2(x_cross, x_cross, y)  # (b, n, d_input)
        x = self._multi_head_attention_dropout_2(x)  # (b, n, d_input)
        x = self._multi_head_attention_add_and_norm_2(y, x)  # (b, n, d_input)

        # Compute sub-layer 3 output
        y = self._feed_forward(x)  # (b, n, d_input)
        y = self._feed_forward_dropout(x)  # (b, n, d_input)
        y = self._feed_forward_add_and_norm(x, y)  # (b, n, d_input)

        return y


class DecoderStack(nn.Module):
    """Decoder stack.

    Args:
        num_layers (int): Number of decoder layers.
        decoder_params (DecoderParams): Decoder layer parameters.

    Note:
        Regarding the use of the encoder stack output, we refer to the remark in AIAYN which
        states "the decoder inserts a third sub-layer, which performs multi-head attention
        over the output of the encoder stack."  In other words, the encoder stack output is
        used as the cross-attention input in each decoder block.
    """

    def __init__(self, num_layers: int, decoder_params: DecoderParams) -> None:
        super().__init__()

        self.layers = nn.ModuleList([Decoder(decoder_params) for _ in range(num_layers)])

    def forward(
        self,
        x: Float[Tensor, "b n d_input"],
        x_cross: Float[Tensor, "b n d_input"],
    ) -> Float[Tensor, "b n d_input"]:
        """Compute decoder stack output.

        Args:
            x (Tensor): Input tensor.
            x_cross (Tensor): Cross-attention input tensor.

        Note:
            In the encoder/decoder Transformer architecture, x_cross is the final output
            of the encoder stack.
        """
        for layer in self.layers:
            x = layer(x, x_cross)  # (b, n, d_input)
        return x
