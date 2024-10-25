"""Implementation of reduced decoder stack objects."""

from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from yati.model.add_and_norm import AddAndNorm
from yati.model.feed_forward import FeedForward
from yati.model.multi_head_attention import MultiHeadAttention
from yati.params.encoder_decoder_params import EncoderDecoderParams


class ReducedDecoder(nn.Module):
    """Reduced decoder layer.

    Args:
        params: Decoder layer parameters.
        apply_causal_mask: Apply causal mask.
        max_seq_len: Maximum input sequence length.

    Note:
        This is a "reduced" decoder layer in the sense that it has no sub-layer
        2 (and therefore no cross-attention input).  In all other respects, it
        is identical to the "full" decoder layer.
    """

    def __init__(  # noqa: DCO010
        self, params: EncoderDecoderParams, apply_causal_mask: bool, max_seq_len: int
    ) -> None:
        super().__init__()
        d_input, h, d_k, d_v, d_ff, p_dropout = params

        # Sub-layer 1 objects
        self._multi_head_attention_1 = MultiHeadAttention(
            d_input, h, d_k, d_v, apply_causal_mask, max_seq_len
        )
        self._multi_head_attention_dropout_1 = nn.Dropout(p_dropout)
        self._multi_head_attention_add_and_norm_1 = AddAndNorm(d_input)

        # Sub-layer 2 objects
        self._feed_forward = FeedForward(d_input, d_ff)
        self._feed_forward_dropout = nn.Dropout(p_dropout)
        self._feed_forward_add_and_norm = AddAndNorm(d_input)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b n d_input"]) -> Float[Tensor, "b n d_input"]:
        """Compute reduced decoder output.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Final output of reduced decoder layer.
        """
        # Compute sub-layer 1 output
        y = self._multi_head_attention_1(x, x, x)  # (b, n, d_input)
        y = self._multi_head_attention_dropout_1(y)  # (b, n, d_input)
        y = self._multi_head_attention_add_and_norm_1(x, y)  # (b, n, d_input)

        # Compute sub-layer 2 output
        x = self._feed_forward(y)  # (b, n, d_input)
        x = self._feed_forward_dropout(x)  # (b, n, d_input)
        x = self._feed_forward_add_and_norm(x, y)  # (b, n, d_input)

        return x


class ReducedDecoderStack(nn.Module):
    """Reduced decoder stack.

    Args:
        num_layers: Number of decoder layers.
        params: Decoder parameters.
        apply_causal_mask: Apply causal mask in each decoder layer.
        max_seq_len: Maximum input sequence length.
    """

    def __init__(  # noqa: DCO010
        self,
        num_layers: int,
        params: EncoderDecoderParams,
        apply_causal_mask: bool,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList(
            [ReducedDecoder(params, apply_causal_mask, max_seq_len) for _ in range(num_layers)]
        )

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b n d_input"]) -> Float[Tensor, "b n d_input"]:
        """Compute reduced decoder stack output.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Final output of reduced decoder stack.
        """
        for layer in self._layers:
            x = layer(x)  # (b, n, d_input)

        return x
