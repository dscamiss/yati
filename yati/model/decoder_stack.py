"""Implementation of decoder stack objects."""

from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from yati.model.add_and_norm import AddAndNorm
from yati.model.feed_forward import FeedForward
from yati.model.multi_head_attention import MultiHeadAttention
from yati.params.encoder_decoder_params import EncoderDecoderParams


class Decoder(nn.Module):
    """Decoder layer.

    Args:
        params (EncoderDecoderParams): Decoder layer parameters.
        max_seq_len (int): Maximum input sequence length.

    Note:
        Regarding the placement of the dropout layers, we follow the remark in AIAYN
        which states: "We apply dropout [33] to the output of each sub-layer, before
        it is added to the sub-layer input and normalized."

    Note:
        Following AIAYN, causal masking is applied in sub-layer 1, and is not applied
        in sub-layer 2.
    """

    def __init__(self, params: EncoderDecoderParams, max_seq_len: int) -> None:
        super().__init__()
        d_input, h, d_k, d_v, d_ff, p_dropout = params

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
        params (EncoderDecoderParams): Decoder parameters.
        max_seq_len (int): Maximum input sequence length.

    Note:
        Regarding the use of the encoder stack output, we refer to the remark in AIAYN which
        states "the decoder inserts a third sub-layer, which performs multi-head attention
        over the output of the encoder stack."  In other words, the encoder stack output is
        used as the cross-attention input in each decoder block.
    """

    def __init__(self, num_layers: int, params: EncoderDecoderParams, max_seq_len: int) -> None:
        super().__init__()
        self._layers = nn.ModuleList([Decoder(params, max_seq_len) for _ in range(num_layers)])

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[Tensor, "b n d_input"],
        x_cross: Float[Tensor, "b n d_input"],
    ) -> Float[Tensor, "b n d_input"]:
        """Compute decoder stack output.

        Args:
            x (Tensor): Input tensor.
            x_cross (Tensor): Cross-attention input tensor.
        """
        for layer in self._layers:
            x = layer(x, x_cross)  # (b, n, d_input)

        return x
