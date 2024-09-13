"""Decoder portion of yati: Yet another transformer implementation."""

import torch.nn as nn

from shared import (
    AddAndNorm,
    EncoderDecoderBlockParams,
    FeedForward,
    MultiHeadAttention,
)
from torch import Tensor


class DecoderBlock(nn.Module):
    """Decoder block.

    Args:
        p: Instance of `EncoderDecoderBlockParams`.

    Attributes:
        multi_head_attention_1: Instance of `MultiHeadAttention` used in sub-layer 1.
        multi_head_attention_dropout_1: Instance of `torch.nn.Dropout` used in sub-layer 1.
        multi_head_add_and_norm_1: Instance of `AddAndNorm` used in sub-layer 1.
        multi_head_attention_2: Instance of `MultiHeadAttention` used in sub-layer 2.
        multi_head_attention_dropout_2: Instance of `torch.nn.Dropout` used in sub-layer 2.
        multi_head_add_and_norm_2: Instance of `AddAndNorm` used in sub-layer 2.
        feed_forward: Instance of `FeedForward` used in sub-layer 3.
        feed_forward_dropout: Instance of `torch.nn.Dropout` used in sub-layer 3.
        feed_forward_add_and_norm: Instance of `AddAndNorm` used in sub-layer 3.

    Note:
        Regarding the placement of the dropout layers, we follow the remark in AIAYN
        which states: "We apply dropout [33] to the output of each sub-layer,
        before it is added to the sub-layer input and normalized."

    Note:
        Following AIAYN, causal masking is applied in the multi-head attention block used
        in sub-layer 1, and no masking is applied in the multi-head attention block used in
        sub-layer 2.
    """

    def __init__(self, p: EncoderDecoderBlockParams) -> None:
        super().__init__()
        self.multi_head_attention_1 = MultiHeadAttention(p.h, p.d_model, p.d_k, p.d_v)
        self.multi_head_attention_dropout_1 = nn.Dropout(p.dropout_prob)
        self.multi_head_add_and_norm_1 = AddAndNorm(p.d_model)
        self.multi_head_attention_2 = MultiHeadAttention(p.h, p.d_model, p.d_k, p.d_v)
        self.multi_head_attention_dropout_2 = nn.Dropout(p.dropout_prob)
        self.multi_head_add_and_norm_2 = AddAndNorm(p.d_model)
        self.feed_forward = FeedForward(p.d_model, p.d_ff)
        self.feed_forward_dropout = nn.Dropout(p.dropout_prob)
        self.feed_forward_add_and_norm = AddAndNorm(p.d_model)

    def forward(self, x: Tensor, y_enc: Tensor) -> Tensor:
        """Compute decoder block output.

        Args:
            x: Input tensor of size (b, n, d_model).
            y_enc: Encoder stack output tensor of size (b, n, d_model).
        """
        # Compute sub-layer 1 output
        y = self.multi_head_attention_1(x, x, x, causal_mask=True)
        y = self.multi_head_attention_dropout_1(y)
        y = self.multi_head_add_and_norm_1(x, y)

        # Compute sub-layer 2 output
        x = self.multi_head_attention_2(y_enc, y_enc, y)
        x = self.multi_head_attention_dropout_2(x)
        x = self.multi_head_add_and_norm_2(y, x)

        # Compute sub-layer 3 output
        y = self.feed_forward(x)
        y = self.feed_forward_dropout(x)
        y = self.feed_forward_add_and_norm(x, y)

        return y


class DecoderStack(nn.Module):
    """Decoder stack.

    Args:
        p: Instance of `EncoderDecoderBlockParams`.
        num_decoder_blocks: Number of decoder blocks in the stack.

    Note:
        Regarding the use of the encoder stack output, we refer to the remark in AIAYN which
        states "the decoder inserts a third sub-layer, which performs multi-head attention
        over the output of the encoder stack."   Our interpretation of this remark is that the
        output of the _entire_ encoder stack is used as the first two arguments of the multi-head
        attention in sub-layer 2 of each decoder block.  (An alternative interpretation is that
        the output of the i-th encoder block is used as the first two arguments of the multi-head
        attention in sub-layer 2 of the i-th decoder block, provided that the number of encoder
        blocks is equal to the number of decoder blocks.)
    """

    def __init__(self, p: EncoderDecoderBlockParams, num_decoder_blocks: int) -> None:
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(p) for _ in range(num_decoder_blocks)]
        )

    def forward(self, x: Tensor, y_enc: Tensor) -> Tensor:
        """Compute decoder stack output.

        Args:
            x: Input tensor of size (b, n, d_model).
            y_enc: Encoder stack output tensor of size (b, n, d_model).
        """
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, y_enc)
        return x
