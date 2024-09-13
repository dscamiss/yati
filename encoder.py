"""Encoder portion of yati: Yet another transformer implementation."""

import torch.nn as nn

from shared import (
    AddAndNorm,
    EncoderDecoderBlockParams,
    FeedForward,
    MultiHeadAttention,
)
from torch import Tensor


class EncoderBlock(nn.Module):
    """Encoder block.

    Args:
        p: Instance of `EncoderDecoderBlockParams`.

    Attributes:
        multi_head_attention: Instance of `MultiHeadAttention` used in sub-layer 1.
        multi_head_attention_dropout: Instance of `torch.nn.Dropout` used in sub-layer 1.
        multi_head_add_and_norm: Instance of `AddAndNorm` used in sub-layer 1.
        feed_forward: Instance of `FeedForward` used in sub-layer 2.
        feed_forward_dropout: Instance of `torch.nn.Dropout` used in sub-layer 2.
        feed_forward_add_and_norm: Instance of `AddAndNorm` used in sub-layer 2.

    Note:
        Regarding the placement of the dropout layers, we follow the remark in AIAYN
        which states: "We apply dropout [33] to the output of each sub-layer,
        before it is added to the sub-layer input and normalized."

    Note:
        Following AIAYN, no masking is applied in the multi-head attention block used
        in sub-layer 1.
    """

    def __init__(self, p: EncoderDecoderBlockParams) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(p.h, p.d_model, p.d_k, p.d_v)
        self.multi_head_attention_dropout = nn.Dropout(p.dropout_prob)
        self.multi_head_add_and_norm = AddAndNorm(p.d_model)
        self.feed_forward = FeedForward(p.d_model, p.d_ff)
        self.feed_forward_dropout = nn.Dropout(p.dropout_prob)
        self.feed_forward_add_and_norm = AddAndNorm(p.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Compute encoder block output.

        Input:
            x: Input tensor of size (b, n, d_model)
        """
        # Compute sub-layer 1 output
        y = self.multi_head_attention(x, x, x)
        y = self.multi_head_attention_dropout(y)
        y = self.multi_head_add_and_norm(x, y)

        # Compute sub-layer 2 output
        x = self.feed_forward(y)
        x = self.feed_forward_dropout(x)
        x = self.feed_forward_add_and_norm(y, x)

        return x


class EncoderStack(nn.Module):
    """Encoder stack.

    Args:
        p: Instance of `EncoderDecoderBlockParams`.
        num_encoder_blocks: Number of encoder blocks in the stack.
    """

    def __init__(self, p: EncoderDecoderBlockParams, num_encoder_blocks: int) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(p) for _ in range(num_encoder_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute encoder stack output.

        Input:
            x: Input tensor of size (b, n, d_model)
        """
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x
